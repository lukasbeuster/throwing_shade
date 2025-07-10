import os
import re
import glob
import importlib
import rasterio
from rasterio.transform import rowcol
import datetime as dt
from osgeo import gdal, osr
from osgeo.gdalconst import *
import shade_setup as shade
import numpy as np
import geopandas as gpd
from datetime import datetime, date, time, timedelta
import pandas as pd
from shapely.geometry import box
import concurrent.futures
from pathlib import Path
import json

importlib.reload(shade)

# Set exception handling
gdal.UseExceptions()

def run_shade_processing(config, osmid, year, year_data):
    """
    Main driver function for the shade simulation pipeline. It processes input geospatial data,
    runs building and/or tree shade simulations, extracts and aggregates shade metrics, and
    exports the final dataset as a GeoJSON file.

    Parameters:
        osmid (str): OSM ID used to locate raster directories and output structure.
        year_data (Dataframe): Dataset for the year to simulate shade for

    Returns:
        GeoDataFrame: Final processed dataset with averaged shade metrics.
    """
    binned = int(config['simulation']['bin_size']) > 0

    dataset_gdf, tile_grouped_days, original_dataset = process_dataset(year_data, year, osmid, config)

    if tile_grouped_days is None:
        # Create an empty GeoDataFrame with a specified CRS
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        print(f"User Warning: The dataset result for year {year} is empty. Make sure dataset geometry overlaps with processed rasters in step 4")
        return empty_gdf

    # run_shade_simulations(tile_grouped_days, dataset_gdf, osmid, year, config)

    dataset_with_shade = extract_and_merge_shade_values(dataset_gdf, osmid, binned, config)

    dataset_with_shade = dataset_with_shade.set_crs("EPSG:4326", allow_override=True)

    dataset_final = aggregate_results(dataset_with_shade, original_dataset, config)

    dataset_final = dataset_final.set_crs("EPSG:4326", allow_override=True)

    return dataset_final

# MAIN HELPERS

def run_shade_simulations(tile_grouped_days, dataset_gdf, osmid, year, config):
    """
    Submits shade simulation jobs (building/tree) for each tile and binned date
    based on seasonal classification.

    Parameters:
        tile_grouped_days (dict): Mapping of tile IDs to {binned_date: [timestamps]}.
        dataset_gdf (GeoDataFrame): Dataset containing binned and seasonal labels.
        osmid (str): OSM ID used for locating raster files.

    Returns:
        None
    """
    summer_params = config['seasons']['summer']
    winter_params = config['seasons']['winter']

    with concurrent.futures.ProcessPoolExecutor(max_workers=config['max_workers']) as executor:
        for tile_id, dates in tile_grouped_days.items():
            tile_dataset = dataset_gdf[dataset_gdf['tile_number'] == tile_id]
            for sim_date, timestamps in dates.items():
                subset = tile_dataset[tile_dataset['binned_date'] == sim_date]
                if not subset.empty:
                    season = subset["season"].values[0]
                    params = summer_params if season == 1 else winter_params
                    # SHADE
                    executor.submit(main_shade, osmid, tile_id, timestamps, sim_date, params, config)
                elif sim_date == datetime.fromisoformat(config['year_configs'][year]['solstice_day']).date():
                    executor.submit(main_shade, osmid, tile_id, [None, []], sim_date, summer_params, config)
                else:
                    raise ValueError(f"No data available for the day: {sim_date}")

def extract_and_merge_shade_values(dataset_gdf, osmid, binned, config):
    """
    Extracts shade values for each timestamp-tile subset using parallel execution,
    based on specified simulation parameters.

    Parameters:
        dataset_gdf (GeoDataFrame): Input dataset with assigned tiles and timestamps.
        osmid (str): OSM ID used to locate raster files.
        parameters (dict): Dict with keys:
            - 'building_shade_step', 'combined_shade_step',
              'bldg_shadow_fraction', 'combined_shadow_fraction',
              'hours_before' (list of float/int)

    Returns:
        DataFrame: Concatenated results of all processed subsets with shade metrics.
    """
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=config['max_workers']) as executor:
        futures = []
        for tile_no in dataset_gdf["tile_number"].unique():
            tile_data = dataset_gdf[dataset_gdf["tile_number"] == tile_no]
            for timestamp in tile_data["rounded_timestamp"].unique():
                subset = tile_data[tile_data["rounded_timestamp"] == timestamp]
                future = executor.submit(get_dataset_shaderesult, subset, osmid, binned, config)
                futures.append(future)

        for future in futures:
            results.append(future.result())

    return pd.concat(results, axis=0)

def aggregate_results(dataset_with_shade, original_dataset, config):
    """
    Aggregates extracted shade metrics by the given unique ID column and merges
    the results back into the original dataset.

    Parameters:
        dataset_with_shade (DataFrame): Dataset with shade values at the point level.
        original_dataset (GeoDataFrame): Original dataset before processing.
        unique_ID_column (str): Column used to group for averaging (e.g., trajectory ID).
        parameters (dict): Dict with shade extraction flags and hours_before list.

    Returns:
        GeoDataFrame: Final dataset with averaged shade metrics per unique ID.
    """
    shade_columns = []
    critical_columns = []  # Only columns we require to be non-NaN

    if config['extra_outputs']['building_shade_step']:
        for bf in config['simulation']['buffers']:
            shade_columns.append(f"building_shade_buffer{bf}")
            critical_columns.append(f"building_shade_buffer{bf}")  # Required
            shade_columns += [f"bldg_{h}_before_shadow_fraction_buffer{bf}" for h in config['extra_outputs']['hours_before']]

    if config['extra_outputs']['bldg_shadow_fraction']:
        for bf in config['simulation']['buffers']:
            shade_columns.append(f"bldg_shadow_fraction_buffer{bf}")
            critical_columns.append(f"bldg_shadow_fraction_buffer{bf}")  # Required

    if config['extra_outputs']['tree_shade_step']:
        for bf in config['simulation']['buffers']:
            shade_columns.append(f"combined_shade_buffer{bf}")
            critical_columns.append(f"combined_shade_buffer{bf}")  # Required
            shade_columns += [f"combined_{h}_before_shadow_fraction_buffer{bf}" for h in config['extra_outputs']['hours_before']]

    if config['extra_outputs']['tree_shadow_fraction']:
        for bf in config['simulation']['buffers']:
            shade_columns.append(f"combined_shadow_fraction_buffer{bf}")
            critical_columns.append(f"combined_shadow_fraction_buffer{bf}")  # Required

    # Drop only if required columns are NaN — not the before_fraction columns
    dataset_cleaned = dataset_with_shade.dropna(subset=critical_columns)

    # Aggregate using all shade-related columns
    dataset_aggregated = dataset_cleaned.groupby(config['columns']['unique_id'], as_index=False)[shade_columns].mean()

    merged = original_dataset.merge(dataset_aggregated, on=config['columns']['unique_id'], how="inner")
    return gpd.GeoDataFrame(merged, geometry='geometry')

def main_shade(osmid, tile_id, timestamps, sim_date, inputs, config):
    """
    Coordinates and initiates shade simulation for a specific tile by locating building and canopy DSM rasters
    and calling the `shade_processing` function with appropriate parameters.

    Parameters:
        osmid (str): OpenStreetMap identifier used to locate the dataset directory.
        tile_id (str): Tile ID used to match raster filenames.
        timestamps (dict): Dictionary where keys are datetime objects, and values are lists of additional timestamps
                           to simulate. Used to define which time intervals to compute shade for.
                           Format: {final_timestamp: [intermediate_timestamp1, intermediate_timestamp2, ...]}.
        date_c (datetime.date or datetime.datetime): The base date of the simulation.
        shade_interval (int, optional): Time interval in minutes for shade simulation (default is 30).
        inputs (dict, optional): Dictionary of simulation parameters. Common keys include:
                                 - 'utc': UTC offset (int)
                                 - 'dst': Daylight Saving Time offset (int)
                                 - 'trs': Transmissivity value (e.g., 10 for 10%)
                                 Defaults to {'utc': 1, 'dst': 0, 'trs': 10}.
        start_time (datetime.datetime, optional): Optional simulation start time. If None, it defaults to 00:00 of `date_c`.
        combined (bool, optional): If True, run vegetation shade simulation (tree/canopy DSM).
        building (bool, optional): If True, run building shade simulation.

    Returns:
        None

    Raises:
        Exception: If the matched canopy DSM file cannot be identified based on the building DSM filename.
    """
    # Directory containing the raster files
    processing_dir = str(Path(config["output_dir"]) / f"step4_raster_processing/{osmid}")

    start_time = config['simulation']['start_time']

    # Handle start_time input flexibly
    if start_time == 'None':
        start_time = datetime.combine(sim_date, time(0, 0))
    elif isinstance(start_time, str):
        # Assume format like "11:00"
        start_time_obj = datetime.strptime(start_time, "%H:%M").time()
        start_time = datetime.combine(sim_date, start_time_obj)
    elif isinstance(start_time, time):
        start_time = datetime.combine(sim_date, start_time)
    elif isinstance(start_time, datetime):
        # Use it directly
        pass
    else:
        raise ValueError("Invalid format for start_time")

    building_file = [
        bldg_path for bldg_path in glob.glob(os.path.join(processing_dir, '*building_dsm.tif')) if (f"{tile_id}_" in bldg_path)
    ]

    canopy_file = [
        chm_path for chm_path in glob.glob(os.path.join(processing_dir, '*canopy_dsm.tif')) if (f"{tile_id}_" in chm_path)
    ]

    identifier = extract_identifier(building_file[0])

    if identifier+"_" in canopy_file[0]:
        matched_chm_path = canopy_file[0]
        shade_processing(building_file[0], matched_chm_path, osmid, sim_date, timestamps,
                         start_time, inputs, config)
    else:
        raise Exception("Wasn't able to match chm_path to building path in shade processing")

def shade_processing(bldg_path, matched_chm_path, osmid, date, timestamps, start_time, inputs, config):
    """
    Executes shade simulation for a given tile using building and canopy DSM raster files,
    selectively computing intervals that do not already have processed shade outputs.

    Parameters:
        bldg_path (str): Path to the building DSM raster file.
        matched_chm_path (str): Path to the matched canopy DSM raster file.
        osmid (str): OpenStreetMap ID used to organize output directories.
        date (datetime.date): Date for which the shade simulation is being run.
        shade_interval (int): Interval length in minutes for which shade is simulated.
        timestamps (tuple): A tuple containing:
            - final_stamp (datetime): The final timestamp to simulate.
            - intervals (list): List of intermediate datetime intervals.
        start_time (datetime.datetime): Starting time for the shade simulation window.
        inputs (dict): Dictionary of simulation parameters, including:
            - 'utc', 'dst', 'trs'.
        combined (bool): If True, run tree shade simulation.
        building (bool): If True, run building shade simulation.

    Returns:
        None
    """

    def run_building_shade(inputs, bldg_path, matched_chm_path, tile_no, date, shade_interval,
                           final_stamp, start_time, building_intervals_needed, building_directory):
        shade.shadecalculation_setup(
            filepath_dsm=bldg_path,
            filepath_veg=matched_chm_path,
            tile_no=tile_no,
            date=date,
            intervalTime=shade_interval,
            final_stamp=final_stamp,
            start_time=start_time,
            shade_fractions=building_intervals_needed,
            onetime=0,
            filepath_save=str(building_directory),
            UTC=inputs['utc'],
            dst=inputs['dst'],
            useveg=0,
            trunkheight=25,
            transmissivity=inputs['tree_transmissivity']
        )

    def run_tree_shade(inputs, bldg_path, matched_chm_path, tile_no, date, shade_interval,
                       final_stamp, start_time, tree_intervals_needed, tree_directory):
        shade.shadecalculation_setup(
            filepath_dsm=bldg_path,
            filepath_veg=matched_chm_path,
            tile_no=tile_no,
            date=date,
            intervalTime=shade_interval,
            final_stamp=final_stamp,
            start_time=start_time,
            shade_fractions=tree_intervals_needed,
            onetime=0,
            filepath_save=str(tree_directory),
            UTC=inputs['utc'],
            dst=inputs['dst'],
            useveg=1,
            trunkheight=25,
            transmissivity=inputs['tree_transmissivity']
        )

    final_stamp, intervals = timestamps[0], timestamps[1]

    if final_stamp is not None:
        date = final_stamp
        all_intervals = intervals + [final_stamp]
    else:
        date = datetime.combine(date, datetime.min.time()).replace(hour=23, minute=59, second=59)
        all_intervals = intervals

    if not all_intervals:
        all_intervals = False

    bldg_path = bldg_path.replace("\\", "/")
    matched_chm_path = matched_chm_path.replace("\\", "/")
    identifier = extract_identifier(bldg_path)

    # Check if the file exists
    if os.path.isfile(matched_chm_path):
        print(f"The file {matched_chm_path} exists.")
    else:
        print(f"The file {matched_chm_path} does not exist.")

    folder_no = identifier.split('_')[-1]
    folder_no = folder_no
    tile_no = identifier

    building_directory = Path(config['output_dir']) / f"step5_shade_results/{osmid}/building_shade/{folder_no}"
    tree_directory = Path(config['output_dir']) / f"step5_shade_results/{osmid}/combined_shade/{folder_no}"

    print(building_directory)

    shade_interval = config['simulation']['shade_interval_minutes']

    if config['simulation']['building_shade']:
        building_shadow_files_exist = directory_check(
            building_directory, shadow_check=True, shade_intervals=all_intervals, date=date
        )
        building_intervals_needed = (
            filter_intervals(all_intervals, building_shadow_files_exist) if all_intervals else False
        )

        print("Processing building shade...")
        if not building_shadow_files_exist or (
            isinstance(building_shadow_files_exist, list) and not all(building_shadow_files_exist)
        ):
            run_building_shade(
                inputs, str(bldg_path), str(matched_chm_path), tile_no, date,
                shade_interval, final_stamp, start_time,
                building_intervals_needed, building_directory
            )

    if config['simulation']['combined_shade']:
        tree_shadow_files_exist = directory_check(
            tree_directory, shadow_check=True, shade_intervals=all_intervals, date=date
        )
        tree_intervals_needed = (
            filter_intervals(all_intervals, tree_shadow_files_exist) if all_intervals else False
        )

        print("Processing combined shade...")
        if not tree_shadow_files_exist or (
            isinstance(tree_shadow_files_exist, list) and not all(tree_shadow_files_exist)
        ):
            run_tree_shade(
                inputs, str(bldg_path), str(matched_chm_path), tile_no, date,
                shade_interval, final_stamp, start_time,
                tree_intervals_needed, tree_directory
            )

def save_run_parameters(output_path, osmid, summer_params, winter_params, year_configs,
                        sh_int, building_sh, combined_sh, parameters, bin_size, buffer, start_time):
    """
    Saves the simulation run parameters as a JSON-formatted .txt file for reproducibility.
    """
    script_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_data = {
        "script_run_start": script_start_time,
        "osmid": osmid,
        "summer_params": summer_params,
        "winter_params": winter_params,
        "year_configs": {
            year: {
                "solstice_day": str(cfg["solstice_day"]),
                "dst_start": str(cfg["dst_start"]),
                "dst_end": str(cfg["dst_end"]),
            }
            for year, cfg in year_configs.items()
        },
        "shade_interval_minutes": sh_int,
        "building_shade": building_sh,
        "combined_shade": combined_sh,
        "shade_parameters": parameters,
        "bin_size_days": bin_size,
        "buffer_meters": buffer,
        "shade_simulation_start_time": str(start_time) if start_time else None
    }

    # Construct file name
    base_dir = os.path.dirname(output_path)
    file_name = f"parameters_file_{script_start_time}.txt"
    file_path = os.path.join(base_dir, file_name)

    with open(file_path, 'w') as f:
        json.dump(param_data, f, indent=4)

    print(f"📝 Parameters saved to {file_path}")

# SHADE DATA JOIN

def get_dataset_shaderesult(dataset, osmid, binned, config):
    """
    Extracts and appends shade-related raster values for each point in a dataset based on
    a specified timestamp and tile location.

    The function retrieves shadow values from building and/or tree shade rasters, including
    instantaneous shade and time-averaged shadow fraction over prior hours. It supports both
    direct and binned timestamp usage, and integrates all extracted values back into the original dataset.

    Parameters:
        dataset (GeoDataFrame): GeoDataFrame containing point geometries and associated metadata,
                                including 'tile_number' and 'rounded_timestamp' columns.
        osmid (str): Unique identifier for the tile, used to construct file paths.
        building_shade_step (bool): If True, extracts instantaneous building shade values.
        combined_shade_step (bool): If True, extracts instantaneous combined shade values.
        bldg_shadow_fraction (bool): If True, extracts building shadow fraction raster values.
        combined_shadow_fraction (bool): If True, extracts combined shadow fraction raster values.
        hours_before (list of int/float or None): Optional list of hour values for computing average
                                                  shadow fractions prior to the timestamp.
        buffer (float): Buffer distance (in meters) around each point for raster value extraction.
        binned (bool): If True, use the 'binned_date' column to construct timestamps instead of the default.

    Returns:
        GeoDataFrame: A copy of the original dataset with additional columns for each requested
                      shade-related raster value.

    Raises:
        Exception: If the corresponding building mask file cannot be found for the specified tile.
        AssertionError: If elements in `hours_before` are not numeric types.

    Notes:
        - Raster file paths are constructed using the provided `osmid`, tile ID, and timestamp.
        - Each shadow type (instantaneous or averaged) is extracted only if explicitly requested.
        - Values are extracted using a raster mask that excludes building-covered areas.
    """
    tile_id = dataset["tile_number"].unique()[0]
    rounded_timestamp = dataset["rounded_timestamp"].unique()[0]
    tile_number = tile_id.split("_")[-1]

    # Resolve timestamp
    if binned:
        binned_date = dataset["binned_date"].unique()[0]
        if isinstance(binned_date, date) and not isinstance(binned_date, datetime):
            binned_date = datetime.combine(binned_date, time())
        rounded_ts = binned_date.replace(
            hour=rounded_timestamp.hour,
            minute=rounded_timestamp.minute,
            second=rounded_timestamp.second,
        )
    else:
        rounded_ts = rounded_timestamp

    start_time = config['simulation']['start_time']

    # Resolve start_time
    if start_time != 'None':
        if isinstance(start_time, str):
            start_time = datetime.combine(rounded_ts.date(), datetime.strptime(start_time, "%H:%M").time())
        elif isinstance(start_time, time):
            start_time = datetime.combine(rounded_ts.date(), start_time)

        if rounded_ts < start_time:
            # Build columns and return NaNs early
            shade_columns = []

            for bf in config['simulation']['buffers']:
                if config['extra_outputs']['building_shade_step']:
                    shade_columns.append(f"building_shade_buffer{bf}")
                if config['extra_outputs']['tree_shade_step']:
                    shade_columns.append(f"combined_shade_buffer{bf}")
                if config['extra_outputs']['bldg_shadow_fraction']:
                    shade_columns.append(f"bldg_shadow_fraction_buffer{bf}")
                if config['extra_outputs']['tree_shadow_fraction']:
                    shade_columns.append(f"combined_shadow_fraction_buffer{bf}")
                if config['extra_outputs']['hours_before']:
                    for hr in config['extra_outputs']['hours_before']:
                        assert isinstance(hr, (int, float)), "hours_before must be int/float"
                        if config['extra_outputs']['tree_shade_step']:
                            shade_columns.append(f"combined_{hr}_before_shadow_fraction_buffer{bf}")
                        if config['extra_outputs']['building_shade_step']:
                            shade_columns.append(f"bldg_{hr}_before_shadow_fraction_buffer{bf}")

            return pd.concat([dataset, pd.DataFrame({col: np.nan for col in shade_columns}, index=dataset.index)], axis=1)

    # Start raster extraction
    base_path = Path(config['output_dir']) / f"step5_shade_results/{osmid}"
    ts_str = rounded_ts.strftime('%Y%m%d_%H%M')

    paths = {
        "building_shade": f"{base_path}/building_shade/{tile_number}/{osmid}_{tile_id}_Shadow_{ts_str}_LST.tif",
        "tree_shade": f"{base_path}/combined_shade/{tile_number}/{osmid}_{tile_id}_Shadow_{ts_str}_LST.tif",
        "bldg_shadow_fraction": f"{base_path}/building_shade/{tile_number}/{osmid}_{tile_id}_shadow_fraction_on_{ts_str}.tif",
        "tree_shadow_fraction": f"{base_path}/combined_shade/{tile_number}/{osmid}_{tile_id}_shadow_fraction_on_{ts_str}.tif",
    }

    building_mask_file = [
        b for b in glob.glob(os.path.join(str(Path(config['output_dir']) / f"step2_solar_data/{osmid}"), '*mask.tif')) if f"{tile_id}_" in b
    ]
    if not building_mask_file:
        raise Exception("Couldn't find building mask file to extract shade values")
    building_mask_path = building_mask_file[0]

    result_df = pd.DataFrame(index=dataset.index)
    crop_pixels = config.get('raster_crop_pixels', 50)

    if config['extra_outputs']['building_shade_step']:
        for bf in config['simulation']['buffers']:
            result_df[f"building_shade_buffer{bf}"] = extract_values_from_raster(paths["building_shade"], building_mask_path, dataset, bf, crop_pixels)

    if config['extra_outputs']['tree_shade_step']:
        for bf in config['simulation']['buffers']:
            result_df[f"combined_shade_buffer{bf}"] = extract_values_from_raster(paths["tree_shade"], building_mask_path, dataset, bf, crop_pixels)

    if config['extra_outputs']['bldg_shadow_fraction']:
        for bf in config['simulation']['buffers']:
            result_df[f"bldg_shadow_fraction_buffer{bf}"] = extract_values_from_raster(paths["bldg_shadow_fraction"], building_mask_path, dataset, bf, crop_pixels)

    if config['extra_outputs']['tree_shadow_fraction']:
        for bf in config['simulation']['buffers']:
            result_df[f"combined_shadow_fraction_buffer{bf}"] = extract_values_from_raster(paths["tree_shadow_fraction"], building_mask_path, dataset, bf, crop_pixels)

    if config['extra_outputs']['hours_before']:
        for hr in config['extra_outputs']['hours_before']:
            assert isinstance(hr, (int, float)), "hours_before must be int/float"
            if config['extra_outputs']['tree_shade_step']:
                for bf in config['simulation']['buffers']:
                    col = f"combined_{hr}_before_shadow_fraction_buffer{bf}"
                    result_df[col] = hours_before_shadow_fr(dataset, base_path, building_mask_path, "combined_shade", rounded_ts, tile_number, osmid, hr, bf, crop_pixels)
            if config['extra_outputs']['building_shade_step']:
                for bf in config['simulation']['buffers']:
                    col = f"bldg_{hr}_before_shadow_fraction_buffer{bf}"
                    result_df[col] = hours_before_shadow_fr(dataset, base_path, building_mask_path, "building_shade", rounded_ts, tile_number, osmid, hr, bf, crop_pixels)

    return pd.concat([dataset, result_df], axis=1)

def extract_values_from_raster(raster_path, building_mask_path, dataset, buffer, crop_pixels):
    """
    Extracts shade values, correctly accounting for a cropped shade raster
    by using the building mask as a reference grid and applying a pixel offset.
    """
    if not os.path.exists(raster_path):
        print(f"Warning: Raster file {raster_path} not found.")
        return np.full(len(dataset), np.nan)

    with rasterio.open(raster_path) as src, rasterio.open(building_mask_path) as bsrc:
        raster_data = src.read(1, masked=False)
        building_mask = bsrc.read(1, masked=False)

        raster_nodata = src.nodata if src.nodata is not None else np.nan

        # We only need the transform from the larger, reference raster (building mask)
        building_transform = bsrc.transform

        # Reproject points to match the reference raster CRS
        dataset = dataset.to_crs(bsrc.crs)
        dataset = dataset.reset_index(drop=True)

        values = np.full(len(dataset), np.nan)

        # Buffer in pixels is calculated from the reference raster's resolution
        res_x, _ = bsrc.res
        buffer_pixels = int(buffer / res_x) if buffer > 0 else 0

        for idx, row in dataset.iterrows():
            x, y = row.geometry.x, row.geometry.y

            try:
                # --- Step 1: Get the pixel location ONLY from the un-cropped building mask ---
                building_row, building_col = rowcol(building_transform, x, y)
            except Exception as e:
                print(f"⚠️ Error converting coordinates: {e}")
                continue

            # Check bounds for the reference building mask
            if not (0 <= building_row < building_mask.shape[0] and 0 <= building_col < building_mask.shape[1]):
                print(f"❌ Point is outside the building mask boundary")
                continue

            # --- Step 2: Calculate the corresponding location in the cropped raster ---
            #    This is the key step: we apply the offset.
            raster_row = building_row - crop_pixels
            raster_col = building_col - crop_pixels

            # --- Step 3: Check if this new, calculated location is valid in the cropped raster ---
            if not (0 <= raster_row < raster_data.shape[0] and 0 <= raster_col < raster_data.shape[1]):
                # This means the point was in the margin that got cropped out.
                print(f"❌ Point falls within the cropped margin, no shade data available.")
                continue

            if buffer == 0:
                if building_mask[building_row, building_col] == 1:
                    values[idx] = np.nan
                else:
                    val = raster_data[raster_row, raster_col]
                    values[idx] = np.nan if val == raster_nodata else val
            else:
                # Buffer window for the building mask (using its original indices)
                b_row_start = max(building_row - buffer_pixels, 0)
                b_row_end = min(building_row + buffer_pixels + 1, building_mask.shape[0])
                b_col_start = max(building_col - buffer_pixels, 0)
                b_col_end = min(building_col + buffer_pixels + 1, building_mask.shape[1])
                building_window = building_mask[b_row_start:b_row_end, b_col_start:b_col_end]

                # Buffer window for the shade raster (using its OFFSET indices)
                r_row_start = max(raster_row - buffer_pixels, 0)
                r_row_end = min(raster_row + buffer_pixels + 1, raster_data.shape[0])
                r_col_start = max(raster_col - buffer_pixels, 0)
                r_col_end = min(raster_col + buffer_pixels + 1, raster_data.shape[1])
                raster_window = raster_data[r_row_start:r_row_end, r_col_start:r_col_end]

                # The windows are now geographically aligned but might have slightly different
                # shapes if a buffer goes over an edge. We trim to the smallest intersection.
                min_rows = min(raster_window.shape[0], building_window.shape[0])
                min_cols = min(raster_window.shape[1], building_window.shape[1])

                raster_window_synced = raster_window[:min_rows, :min_cols]
                building_window_synced = building_window[:min_rows, :min_cols]

                filtered = np.where(
                    (building_window_synced == 1) | (raster_window_synced == raster_nodata),
                    np.nan,
                    raster_window_synced
                )

                valid_vals = filtered[~np.isnan(filtered)]
                values[idx] = np.nanmean(valid_vals) if valid_vals.size > 0 else np.nan

    return values

def hours_before_shadow_fr(dataset, base_path, building_mask_path, shade_type, rounded_timestamp, tile_number, osmid, hours_before, buffer, crop_pixels):
    """
    Computes the average shadow fraction for each point in the dataset by aggregating shadow data
    from raster files over a specified number of hours prior to a given timestamp.

    This function ensures temporal robustness by handling missing or misaligned raster files and
    adjusting the start time if necessary based on the earliest available data.

    Parameters:
        dataset (GeoDataFrame): GeoDataFrame containing the point geometries for which shadow fractions will be computed.
        base_path (str): Root directory where the shadow raster files are stored.
        building_mask_path (str): File path to a raster mask used to exclude building-covered areas when extracting values.
        shade_type (str): Type of shade raster to use (e.g., "combined_shade" or "building_shade").
        rounded_timestamp (datetime.datetime): The main timestamp of interest for computing shadow coverage.
        tile_number (str): Tile ID to locate corresponding raster files.
        osmid (str): Unique identifier for the tile (used in constructing raster filenames).
        hours_before (float): Number of hours prior to `rounded_timestamp` over which shadow data should be averaged.
        buffer (float): Buffer radius (in meters) to apply when extracting raster values around each point.

    Returns:
        np.ndarray: A NumPy array of average shadow fractions for all points in the dataset.
                    If no valid rasters exist within the specified range, the array is filled with NaNs.

    Raises:
        Exception: If no shadow files are found for the given date or within the desired time range.

    Notes:
        - If `start_hour` is earlier than the first available shade raster, it is adjusted forward.
        - If no files exist in the range, the function returns an array of NaNs.
        - Shadow values from all matching rasters are averaged per point to compute the final fraction.
    """
    # Compute the starting timestamp based on hours_before
    start_hour = rounded_timestamp - timedelta(hours=hours_before)  # Ensure `hours_before` supports floats

    # Get the earliest available shadow file timestamp for the given day
    first_shade_time = get_earliest_timestamp(f"{base_path}/{shade_type}/{tile_number}", rounded_timestamp)

    if first_shade_time is None:
        print("There are no shade files in the directory for this date. Assigning NaNs.")
        return np.full(len(dataset), np.nan)

    if start_hour < first_shade_time:
        print("Start_hour is earlier than the first available shade file — assigning NaNs.")
        return np.full(len(dataset), np.nan)

    # If the exact `start_hour` shadow file doesn't exist, find the closest valid one
    shadow_file_path = f"{base_path}/{shade_type}/{tile_number}/{osmid}_p_{tile_number}_Shadow_{start_hour.strftime('%Y%m%d_%H%M')}_LST.tif"

    if not os.path.exists(shadow_file_path):
        print(f"This shade file for start hour doesn't exist: {shadow_file_path}")
        start_hour_file = get_closest_shade_file(base_path, shade_type, tile_number, osmid, start_hour)
        start_hour = extract_datetime_from_path(start_hour_file)
        if start_hour >= rounded_timestamp:
            # there are no shade files available
            return np.full(len(dataset), np.nan)

    # Retrieve all shadow files within the time range [start_hour, rounded_timestamp]
    shade_files_for_shadow_frac = get_shade_files_in_range(base_path, shade_type, tile_number, osmid, start_hour, rounded_timestamp)

    if not shade_files_for_shadow_frac:
        raise Exception("Didn't find shade files between start time and timestamp")

    # Compute the shadow fraction by averaging the extracted values from all retrieved shade rasters
    shadow_values = np.zeros(len(dataset))

    for shade_raster in shade_files_for_shadow_frac:
        raster_values = extract_values_from_raster(shade_raster, building_mask_path, dataset, buffer, crop_pixels)
        shadow_values += np.nan_to_num(raster_values)  # Ensure NaN values don't affect summation

    # Compute the final shadow fraction (average)
    shadow_fractions = shadow_values / len(shade_files_for_shadow_frac)

    return shadow_fractions

def get_shade_files_in_range(base_path, shade_type, tile_number, osmid, start_hour, rounded_timestamp):
    """
    Get all shade files in a directory within the range of start_hour and rounded_timestamp (inclusive).

    Parameters:
        base_path (str): The base directory where shade files are stored.
        shade_type (str): The type of shade to get the files
        tile_number (str): The tile number for shade calculations.
        osmid (str): The unique ID for the dataset.
        start_hour (datetime): The lower bound timestamp (inclusive).
        rounded_timestamp (datetime): The upper bound timestamp (inclusive).

    Returns:
        list: List of full file paths that fall within the specified time range.
    """
    directory = f"{base_path}/{shade_type}/{tile_number}/"

    # Ensure directory exists
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return []

    # Regex pattern to extract timestamp from filenames
    pattern = re.compile(rf"{osmid}_p_{tile_number}_Shadow_(\d{{8}}_\d{{4}})_LST\.tif")

    # List all files in directory
    all_files = os.listdir(directory)

    # Filter and extract timestamps
    valid_files = []
    for filename in all_files:
        if filename.endswith(".tif") and not filename.endswith(".tif.ovr"):  # Ensure only `.tif` files, exclude `.tif.ovr`
            match = pattern.search(filename)
            if match:
                file_timestamp_str = match.group(1)  # Extract timestamp string
                file_timestamp = datetime.strptime(file_timestamp_str, "%Y%m%d_%H%M")  # Convert to datetime

                # Check if the timestamp is within the range (inclusive)
                if start_hour <= file_timestamp <= rounded_timestamp:
                    valid_files.append(os.path.join(directory, filename))

    return sorted(valid_files)  # Return sorted list of file paths

def get_closest_shade_file(base_path, shade_type, tile_number, osmid, start_hour):
    """
    Get the closest existing shade file to `start_hour`.
    If two timestamps are equidistant, choose the later one.

    Parameters:
        base_path (str): The base directory where shade files are stored.
        tile_number (str): The tile number for shade calculations.
        osmid (str): The unique ID for the dataset.
        start_hour (datetime): The target timestamp.

    Returns:
        str: Full file path of the closest shade file, or None if no files exist.
    """
    directory = f"{base_path}/{shade_type}/{tile_number}/"

    # Ensure directory exists
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return None

    # Regex pattern to extract timestamp from filenames
    pattern = re.compile(rf"{osmid}_p_{tile_number}_Shadow_(\d{{8}}_\d{{4}})_LST\.tif")

    # List all files in directory
    all_files = os.listdir(directory)

    # Extract timestamps from filenames
    timestamps = []
    file_map = {}  # Dictionary to map timestamps to filenames
    for filename in all_files:
        if filename.endswith(".tif") and not filename.endswith(".tif.ovr"):
            match = pattern.search(filename)
            if match:
                file_timestamp_str = match.group(1)  # Extract timestamp string
                file_timestamp = datetime.strptime(file_timestamp_str, "%Y%m%d_%H%M")  # Convert to datetime
                timestamps.append(file_timestamp)
                file_map[file_timestamp] = os.path.join(directory, filename)

    # If no valid timestamps were found
    if not timestamps:
        print(f"No valid shade files found in {directory}")
        return None

    # Sort timestamps
    timestamps.sort()

    # Find the closest timestamp
    closest_timestamp = min(
        timestamps,
        key=lambda t: (abs((t - start_hour).total_seconds()), -t.timestamp())  # Prioritize later timestamps
    )

    return file_map[closest_timestamp]

def get_earliest_timestamp(directory, date_obj):
    """
    Finds the earliest timestamp from raster filenames in a directory
    that match the given date.

    Parameters:
    - directory (str): Path to the directory containing the raster files.
    - date_obj (datetime): The reference date.

    Returns:
    - datetime: The earliest timestamp for the given date, or None if no match is found.
    """
    date_str = date_obj.strftime("%Y%m%d")  # Convert date to string format YYYYMMDD
    pattern = re.compile(r".*_(\d{8})_(\d{4})_LST\.tif")  # Regex to match date & time in filename

    timestamps = []

    if not os.path.exists(directory):
        print(f"Shade directory {directory} doesn't exist. Skipping extraction.")
        return None

    for filename in os.listdir(directory):
        if filename.endswith(".tif") and not filename.endswith(".tif.ovr"):  # Ensure only `.tif` files, exclude `.tif.ovr`
            match = pattern.match(filename)
            if match:
                file_date, file_time = match.groups()
                if file_date == date_str:  # Check if the date matches
                    timestamp = datetime.strptime(f"{file_date} {file_time}", "%Y%m%d %H%M")
                    timestamps.append(timestamp)

    return min(timestamps) if timestamps else None

def extract_datetime_from_path(file_path):
    """
    Extracts the datetime object from the given file path.

    Parameters:
    - file_path (str): The full file path of the raster.

    Returns:
    - datetime: Extracted datetime object.
    """
    # Extract filename
    filename = os.path.basename(file_path)

    # Regex pattern to find the date and time in the filename
    match = re.search(r"_Shadow_(\d{8})_(\d{4})_LST\.tif", filename)

    if match:
        date_part = match.group(1)  # '20230823'
        time_part = match.group(2)  # '1200'

        # Convert to datetime object
        return datetime.strptime(date_part + time_part, "%Y%m%d%H%M")

    # Return None if no match is found
    return None

# DATASET PROCESSING

def process_dataset(dataset, year, osmid, config):
    """
    Processes a spatiotemporal dataset by assigning each point to a raster tile and binning timestamps
    based on proximity to a reference solstice day.

    This function:
    - Converts a tabular dataset into a GeoDataFrame if needed.
    - Loads raster footprints from DSM tiles and assigns each point to a tile via spatial join.
    - Rounds timestamps to a specified interval.
    - Computes each point's temporal distance from a given solstice.
    - Bins points into groups based on temporal proximity to the solstice using `bin_data`.
    - Assigns a seasonal label (summer or winter) based on daylight saving time bounds.

    Parameters:
        dataset (DataFrame): Input dataset containing at least longitude, latitude, and timestamp columns.

    Returns:
        tuple:
            - modified_dataset (GeoDataFrame): Dataset with spatial tile IDs, rounded timestamps, binned dates, and seasonal labels.
            - tile_grouped_days (dict): Mapping of each tile to its binned dates and associated timestamps, as produced by `bin_data2`.

    Raises:
        ValueError: If raster files are not found or tile assignment fails due to CRS mismatch or missing geometries.

    Notes:
        - The `tile_number` column is added to associate each point with a raster tile.
        - Timestamp binning and seasonal assignment support downstream shading simulations.
    """
    raster_dir = Path(config["output_dir"]) / f"step4_raster_processing/{osmid}"
    raster_files = list(raster_dir.glob('*building_dsm.tif'))

    # Extract tile footprints from raster files
    raster_tiles = []
    raster_crs = None  # Store the raster CRS

    for raster_path in raster_files:
        match = re.search(r"p_\d+", str(raster_path))  # Extract tile number like 'p_0'
        if match:
            tile_number = match.group(0)
            with rasterio.open(raster_path) as src:
                raster_crs = src.crs  # Ensure all rasters use the same CRS
                transform = src.transform
                width = src.width
                height = src.height

                # Get actual polygon footprint instead of just bounds
                tile_polygon = box(*rasterio.transform.array_bounds(height, width, transform))
                raster_tiles.append({"tile_number": tile_number, "geometry": tile_polygon})

    # Convert raster tile footprints to a GeoDataFrame
    tiles_gdf = gpd.GeoDataFrame(raster_tiles, crs=raster_crs)

    dataset_copy = dataset.copy()

    # Convert DataFrame to GeoDataFrame
    dataset_copy["geometry"] = gpd.points_from_xy(dataset_copy[config['columns']['longitude']], dataset_copy[config['columns']['latitude']])
    df_gdf = gpd.GeoDataFrame(dataset_copy, geometry="geometry", crs="EPSG:4326")

    # Reproject points to match raster CRS
    df_gdf = df_gdf.to_crs(raster_crs)

    # Spatial join to assign each point to the correct tile
    df_gdf = gpd.sjoin(df_gdf, tiles_gdf, how="left", predicate="intersects")

    # Drop unnecessary columns from spatial join
    df_gdf.drop(columns=["index_right"], inplace=True, errors="ignore")

    df_gdf = df_gdf.dropna(subset=["tile_number"])

    timestamp_column = config['columns']['timestamp']

    # Convert timestamp column to datetime
    df_gdf[timestamp_column] = pd.to_datetime(df_gdf[timestamp_column])

    # Apply correct rounding to nearest interval
    df_gdf["rounded_timestamp"] = df_gdf[timestamp_column].apply(lambda x: get_interval_stamp(x, config['simulation']['shade_interval_minutes']))

    solstice_day = datetime.fromisoformat(config['year_configs'][year]['solstice_day'])
    df_gdf["diff_solstice_day"] = df_gdf["rounded_timestamp"].dt.date - solstice_day.date()

    tile_grouped_days, modified_dataset = bin_data(df_gdf, config, solstice_day)

    if tile_grouped_days is None:
        return (None, None, None)

    modified_dataset["season"] = modified_dataset["binned_date"].apply(
    lambda date: assign_summer_winter(date, datetime.fromisoformat(config['year_configs'][year]['dst_start']).date(), datetime.fromisoformat(config['year_configs'][year]['dst_end']).date())
    )

    modified_dataset = modified_dataset.reset_index()
    modified_dataset = modified_dataset.drop(['index', 'diff_solstice_day', 'abs_diff_solstice_day'], axis=1)

    return modified_dataset, tile_grouped_days, dataset_copy

def bin_data(dataset_gdf, config, solstice_day):
    """
    Bins geospatial-temporal data by grouping days around a reference solstice day and assigning
    each data point to the closest binned date. This is useful for aggregating or simplifying
    time-series analyses around seasonal anchors like solstices.

    Parameters:
        dataset_gdf (GeoDataFrame): Input dataset containing at least the following columns:
            - 'tile_number': Spatial tile ID.
            - 'diff_solstice_day': Time difference from the solstice day (as timedelta).
            - 'rounded_timestamp': Timestamp(s) associated with each observation.
        solstice_day (datetime.datetime): The reference solstice date for binning.
        simulate_solstice (bool): If True, includes the solstice day as a bin even if no observations fall in that window.
        grouping_cutoff (int, optional): Number of days to use for bin grouping radius. Default is 7 (resulting in 14-day bins).

    Returns:
        tuple:
            - grouped_days (dict): Nested dictionary structured as
              {tile_number: {binned_date: [final_timestamp, [intermediate_timestamps]]}},
              storing the timestamp groupings for each bin.
            - final_modified_dataset (GeoDataFrame): Modified copy of the input dataset
              with an additional column `binned_date` assigning each row to its temporal bin.

    Notes:
        - Binning starts with the solstice window, then proceeds outward in ±`grouping_cutoff` increments.
        - Each tile is processed independently for performance and locality of data.
        - Bins are formed based on `abs(diff_solstice_day)` and do not account for direction unless post-processed.
    """
    def sort_unique_list(l):
        return sorted(set(l))  # More efficient than list(set(l))

    def add_date_timestamp(grouped_days, last_calc_date, filtered_rows, subset_to_add):
        '''
        Add the day to calculate to grouped_days with with last and intermediate timestamps
        based on filtered rows
        '''
        bin_added_subset = subset_to_add.copy()
        bin_added_subset['binned_date'] = [last_calc_date]*bin_added_subset.shape[0]

        if len(filtered_rows) == 0:
            grouped_days[tile][last_calc_date] = [None, []]

        else:
            all_timestamps = filtered_rows['rounded_timestamp'].explode().tolist()

            matched_all_timestamps = sort_unique_list([match_date(ts, last_calc_date) for ts in all_timestamps])
            last_timestamp = matched_all_timestamps[-1]
            intermediate_timestamps = matched_all_timestamps[:-1]

            grouped_days[tile][last_calc_date] = [last_timestamp, intermediate_timestamps]

        return grouped_days, bin_added_subset

    grouping_cutoff=config['simulation']['bin_size']

    # Convert grouping_cutoff to timedelta (ensuring correct format)
    bin_size = pd.to_timedelta(grouping_cutoff * 2, unit='D')
    grouping_cutoff = pd.to_timedelta(grouping_cutoff, unit='D')
    first_bin_size = grouping_cutoff

    dataset_gdf['abs_diff_solstice_day'] = dataset_gdf["diff_solstice_day"].abs()

    # **Precompute unique days**
    unique_days = dataset_gdf.groupby(['tile_number', 'abs_diff_solstice_day'])['rounded_timestamp']\
                             .apply(sort_unique_list).reset_index()

    # Store results
    grouped_days = {}
    results = []

    # **Pre-split dataset by tile for faster lookups**
    dataset_by_tile = {tile: df for tile, df in dataset_gdf.groupby('tile_number')}

    # Iterate over each tile group
    for tile, group in unique_days.groupby('tile_number'):
        grouped_days[tile] = {}  # Initialize storage for this tile

        start_diff = pd.to_timedelta(0, unit='D')
        last_calc_date = solstice_day.date()
        max_diff = pd.to_timedelta(pd.Timedelta(max(group['abs_diff_solstice_day'].values)).days, unit="D")

        # **Filter dataset for this tile once (faster lookups)**
        tile_dataset = dataset_by_tile[tile]

        # **Step 1: Always Add Solstice**
        mask = (tile_dataset['abs_diff_solstice_day'] >= start_diff) & \
            (tile_dataset['abs_diff_solstice_day'] <= start_diff + first_bin_size)
        filtered_rows = tile_dataset[mask]

        if len(filtered_rows) == 0 and not config['simulation']['simulate_solstice']:
            end_diff = start_diff + grouping_cutoff
        else:
            grouped_days, filtered_rows_added = add_date_timestamp(grouped_days, last_calc_date, filtered_rows, filtered_rows)
            end_diff = start_diff + grouping_cutoff
            results.append(filtered_rows_added)

        # **Step 2: Bin Remaining Data**
        while end_diff <= max_diff:
            # Get the next minimum `abs_diff_solstice_day`
            next_values = group.loc[group['abs_diff_solstice_day'] > end_diff, 'abs_diff_solstice_day'].values
            if next_values.size > 0:
                start_diff = pd.to_timedelta(next_values.min(), unit="D")
            else:
                break  # No more bins to process

            start_date = solstice_day + start_diff

            # **Check if last bin overflows max_diff**
            if start_diff + bin_size > max_diff:
                last_calc_date = (start_date + (max_diff - start_diff) / 2).date()
                end_diff = max_diff
            else:
                end_diff = start_diff + bin_size
                last_calc_date = (solstice_day + (start_diff + grouping_cutoff)).date()

            # **Filter dataset for this bin (optimized)**
            mask = (tile_dataset['abs_diff_solstice_day'] >= start_diff) & \
                   (tile_dataset['abs_diff_solstice_day'] <= end_diff)
            filtered_rows = tile_dataset[mask]

            grouped_days, filtered_rows_added = add_date_timestamp(grouped_days, last_calc_date, filtered_rows, filtered_rows)
            results.append(filtered_rows_added)

    if not results:
        return (None, None)

    # **Step 3: Merge all binned results efficiently**
    final_modified_dataset = pd.concat(results, ignore_index=True)

    return grouped_days, final_modified_dataset

def assign_summer_winter(p_date, dst_start, dst_end):
    """
    Determine if a date falls within summer (daylight savings) or winter time.

    Parameters:
        date (datetime.datetime): The date to evaluate.
        dst_start (datetime.datetime): The start of daylight savings (UTC).
        dst_end (datetime.datetime): The end of daylight savings (UTC).

    Returns:
        int: 1 if the date is during summer (DST), 0 if during winter.
    """
    if dst_start<= p_date < dst_end:
        return 1  # Summer time
    else:
        return 0  # Winter time

def get_interval_stamp(timestamp, interval=30):
    """
    Rounds a given timestamp to the nearest time interval boundary (in minutes) since midnight.

    This function is useful for aligning timestamps to consistent time bins (e.g., 30-minute intervals)
    when processing or aggregating time-based data.

    Parameters:
        timestamp (datetime.datetime): The input timestamp to round.
        interval (int, optional): The interval size in minutes. Defaults to 30.

    Returns:
        datetime.datetime: A new timestamp rounded to the nearest interval boundary.
    """
    minutes_since_midnight = timestamp.hour * 60 + timestamp.minute
    rounded_minutes = round(minutes_since_midnight / interval) * interval
    r_hour = rounded_minutes // 60
    r_minute = rounded_minutes % 60

    return timestamp.replace(hour=r_hour, minute=r_minute, second=0, microsecond=0)

def match_date(ts, target_date):
    """
    Replaces the date component of a timestamp with a target date, preserving the original time.

    This is useful for aligning times (e.g., from a binned or reference timestamp) to a specific day.

    Parameters:
        ts (datetime.datetime): The original timestamp whose time component will be preserved.
        target_date (datetime.date or datetime.datetime): The date to assign to the new timestamp.
                                                          If a `date` is provided, it will be converted to `datetime`.

    Returns:
        datetime.datetime: A new timestamp with `target_date` as the date and `ts`'s hour, minute, and second as the time.
    """
    if isinstance(target_date, date) and not isinstance(target_date, datetime):  # Use 'date' and 'datetime' from datetime module
        target_date = datetime.combine(target_date, time())  # Convert date-only to full datetime

    return target_date.replace(hour=ts.hour, minute=ts.minute, second=ts.second)

# DIRECTORY

def check_files_exist(file_paths):
    """
    Check if all files in the list exist.

    Parameters:
    file_paths (list of str): List of file paths to check.

    Returns:
    bool: True if all files exist, False otherwise.
    """
    return all(os.path.exists(file_path) for file_path in file_paths)

def extract_identifier(path):
    """
    Extracts an identifier from a file path string by isolating the portion
    before a year-based timestamp pattern (e.g., "_2023_") in the filename.

    Parameters:
        path (str): Full file path or filename string.

    Returns:
        str: Extracted identifier, typically the prefix before a "_20xx_" pattern
             (e.g., "tileXYZ" from "tileXYZ_2023_0801_LST.tif").

    """
    # Extract the last segment of the path
    last_segment = path.split('/')[-1]

    # Use regular expression to match the pattern before _20xx_
    match = re.match(r'(.*)_20\d{2}_', last_segment)

    if match:
        identifier = match.group(1)
    else:
        identifier = last_segment.split('_20')[0]

    return identifier

def directory_check(directory, shadow_check=True, shade_intervals=False, date=dt.datetime.now()):
    """
    Checks if a directory exists and optionally verifies the presence of shadow fraction files.

    If the directory does not exist, it is created. If `shadow_check` is enabled, the function
    searches for files containing 'shadow_fraction_on_' followed by the given date. If
    `shade_intervals` is provided as a list of datetime objects, it returns a list of booleans
    indicating whether a file exists for each interval.

    Parameters:
    ----------
    directory : str
        The path to the directory to check or create.
    shadow_check : bool, optional
        Whether to check for shadow fraction files (default is True).
    shade_intervals : list of datetime, optional
        A list of datetime objects representing specific intervals to check for shadow fraction files.
    date : datetime, optional
        The reference date for file checking (default is the current date).

    Returns:
    -------
    bool or list of bool
        - If `shade_intervals` is not provided, returns True if at least one shadow fraction file is found,
          otherwise returns False.
        - If `shade_intervals` is provided, returns a list of booleans where each element corresponds to whether
          a shadow fraction file exists for a specific interval.

    Appends False if it doesn't exist, True if it exists
    returns list if shade_intervals exist, True or False if it doesn't
    """

    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    else:
        print(f"Directory {directory} already exists.")

    # Convert date to string format
    timestr = date.strftime("%Y%m%d")

    if shadow_check:
        # Check for files containing 'shadow_fraction_on_' with the given date
        shadow_files = [f for f in os.listdir(directory) if f'shadow_fraction_on_{timestr}' in f]

        if shadow_files:
            if shade_intervals:
                # Ensure shade_intervals is a list of datetime objects
                if not isinstance(shade_intervals, list) or not all(isinstance(ts, dt.datetime) for ts in shade_intervals):
                    raise ValueError("shade_intervals must be a list of datetime objects.")

                shade_int_check = []
                for interval in shade_intervals:
                    int_time = interval.strftime("%Y%m%d_%H%M")
                    shadow_files_interval = [f for f in shadow_files if f'shadow_fraction_on_{int_time}' in f]
                    if shadow_files_interval:
                        print(f"File containing 'shadow_fraction_on_{int_time}' found")
                        shade_int_check.append(True)
                    else:
                        print(f"No files containing 'shadow_fraction_on_{int_time}' found.")
                        shade_int_check.append(False)
                return shade_int_check
            else:
                print(f"Files containing 'shadow_fraction_on_{timestr}' found: {shadow_files}")
                return True  # Required files found
        else:
            print(f"No files containing 'shadow_fraction_on_{timestr}' found.")
            return False  # Required files not found

def filter_intervals(intervals, shadow_files_exist):
    """
    Filters a list of time intervals to determine which intervals still require processing
    based on the existence of building and combined shadow files.

    Parameters:
        intervals (list): A list of time interval identifiers (e.g., datetime strings or objects).
        shadow_files_exist (list or bool): If a list, it must match the length of `intervals`
            and contain booleans indicating the presence of corresponding shadow files.
            If False, all intervals are considered needed. If True, no intervals are needed.

    Returns:
        tuple:
            - intervals_needed (list or bool): List of intervals that require shadow
              processing, or False if none are needed.

    Raises:
        AssertionError: If `building_shadow_files_exist` or `tree_shadow_files_exist` are lists but
        do not match the length of `intervals`.
    """
    # filter to only calculate intervals that don't have a file
    if isinstance(shadow_files_exist, list):
        assert len(intervals) == len(shadow_files_exist), "Directory check for the intervals is broken"
        intervals_needed = [intervals[i] for i, check in enumerate(shadow_files_exist) if not check]
        if len(intervals_needed) < 1:
            intervals_needed = False
    elif not shadow_files_exist:
        intervals_needed = intervals # need to simulate all
    else:
        intervals_needed = False # don't need to simulate any

    return intervals_needed
