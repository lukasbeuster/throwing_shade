import argparse
import os
import re
import glob
import importlib
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.transform import Affine
from rasterio.transform import rowcol
from rasterio.mask import mask
import datetime as dt
from osgeo import gdal, osr
from osgeo.gdalconst import *
import shade_setup as shade
import numpy as np
import geopandas as gpd
from datetime import datetime, date, time, timedelta
import pandas as pd
from shapely.geometry import box
from shapely.geometry import mapping

import matplotlib.pyplot as plt

from scipy.ndimage import median_filter
from scipy.ndimage import uniform_filter
from scipy.ndimage import gaussian_filter
from scipy.ndimage import minimum_filter

import startinpy

import concurrent.futures

importlib.reload(shade)

# Set exception handling
gdal.UseExceptions()

# MAIN

def main(dataset_path, osmid, unique_ID_column, raster_dir, solstice_day,
         longitude_column, latitude_column, timestamp_column_name, dst_start, dst_end,
         output_path, summer_params, winter_params,
         combined_sh=False, building_sh=False, interval=30,
         geometry=False, crs="EPSG:4326", simulate_solstice=False, bin_size=0,
         parameters=None):
    """
    Main driver function for the shade simulation pipeline. It processes input geospatial data,
    runs building and/or tree shade simulations, extracts and aggregates shade metrics, and
    exports the final dataset as a GeoJSON file.

    Parameters:
        dataset_path (str): Path to the input GeoJSON or shapefile dataset.
        osmid (str): OSM ID used to locate raster directories and output structure.
        unique_ID_column (str): Column name used to group data during final aggregation.
        raster_dir (str): Directory path containing processed DSM raster files.
        solstice_day (datetime): Reference date used for temporal binning.
        longitude_column (str): Name of the longitude column in the dataset.
        latitude_column (str): Name of the latitude column in the dataset.
        timestamp_column_name (str): Name of the timestamp column in the dataset.
        dst_start (datetime.date): Start date of daylight saving time.
        dst_end (datetime.date): End date of daylight saving time.
        output_path (str): Path to save the final processed GeoJSON file.
        summer_params (dict): Parameter dictionary used during summer season shade simulation.
        winter_params (dict): Parameter dictionary used during winter season shade simulation.
        combined_sh (bool): Whether to run tree shade simulations.
        building_sh (bool): Whether to run building shade simulations.
        interval (int): Time interval (in minutes) to round timestamps. Default is 30.
        geometry (bool): If True, input dataset already contains geometry.
        crs (str): Coordinate reference system of the input dataset. Default is EPSG:4326.
        simulate_solstice (bool): Whether to always simulate the solstice date even if no data exists for it.
        bin_size (int): Binning radius in days around the solstice.
        parameters (dict): Dictionary with keys:
            - 'building_shade_step', 'tree_shade_step', 'bldg_shadow_fraction',
              'tree_shadow_fraction', and 'hours_before' (list of float/int).

    Returns:
        GeoDataFrame: Final processed dataset with averaged shade metrics.
    """
    if parameters is None:
        parameters = {
            'building_shade_step': False,
            'tree_shade_step': False,
            'bldg_shadow_fraction': False,
            'tree_shadow_fraction': False,
            'hours_before': []
        }

    try:
        main_raster(osmid, raster_dir)
    except Exception as e:
        print(f"{e}: Failing in raster creation")

    # dataset_gdf, tile_grouped_days, original_dataset = load_and_preprocess_dataset(
    #     dataset_path, osmid, solstice_day, longitude_column, latitude_column, timestamp_column_name,
    #     dst_start, dst_end, interval, geometry, crs, simulate_solstice, bin_size
    # )

    # run_shade_simulations(tile_grouped_days, dataset_gdf, osmid, solstice_day, summer_params, winter_params, combined_sh, building_sh, interval)

    # dataset_with_shade = extract_and_merge_shade_values(dataset_gdf, osmid, parameters)

    # dataset_final = aggregate_results(dataset_with_shade, original_dataset, unique_ID_column, parameters)

    # dataset_final.to_file(output_path, driver="GeoJSON")
    # return dataset_final

# MAIN HELPERS

def load_and_preprocess_dataset(dataset_path, osmid, solstice_day,
                                lon_col, lat_col, ts_col, dst_start, dst_end,
                                interval, geometry, crs, simulate_solstice, bin_size):
    """
    Loads the raw spatial dataset, assigns DSM tiles and rounded timestamps,
    and bins data temporally based on proximity to a solstice date.

    Parameters:
        dataset_path (str): Path to the input dataset (GeoJSON or shapefile).
        osmid (str): OSM ID used for locating the DSM raster directory.
        solstice_day (datetime): Date used as the center for time-based binning.
        lon_col (str): Name of the longitude column.
        lat_col (str): Name of the latitude column.
        ts_col (str): Name of the timestamp column.
        dst_start (datetime.date): Start of daylight saving time.
        dst_end (datetime.date): End of daylight saving time.
        interval (int): Minutes to round timestamps. Default is 30.
        geometry (bool): Whether the dataset already includes geometry.
        crs (str): CRS to use if creating geometry.
        simulate_solstice (bool): Force include a bin for the solstice.
        bin_size (int): Number of days before/after the solstice for bin grouping.

    Returns:
        tuple:
            - GeoDataFrame: Preprocessed dataset with tile and time bin assignment.
            - dict: Mapping of tiles to binned timestamps for shade simulation.
            - GeoDataFrame: Original unmodified dataset for merging at the end.
    """
    dataset = gpd.read_file(dataset_path)
    dataset_copy = dataset.copy()
    dataset_gdf, tile_grouped_days = process_dataset(
        dataset_copy, solstice_day,
        f"../data/clean_data/solar/{osmid}/rdy_for_processing",
        lon_col, lat_col, ts_col, dst_start, dst_end,
        interval=interval, geometry=geometry, crs=crs,
        simulate_solstice=simulate_solstice, bin_size=bin_size
    )
    return dataset_gdf, tile_grouped_days, dataset_copy

def run_shade_simulations(tile_grouped_days, dataset_gdf, osmid, solstice_day,
                          summer_params, winter_params, combined, building, interval):
    """
    Submits shade simulation jobs (building/tree) for each tile and binned date
    based on seasonal classification.

    Parameters:
        tile_grouped_days (dict): Mapping of tile IDs to {binned_date: [timestamps]}.
        dataset_gdf (GeoDataFrame): Dataset containing binned and seasonal labels.
        osmid (str): OSM ID used for locating raster files.
        solstice_day (datetime): Solstice date for special simulation handling.
        summer_params (dict): Simulation inputs to use during summer season.
        winter_params (dict): Simulation inputs to use during winter season.
        combined (bool): If True, run tree shade simulation.
        building (bool): If True, run building shade simulation.
        interval (int): Time interval for simulation in minutes.

    Returns:
        None
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for tile_id, dates in tile_grouped_days.items():
            tile_dataset = dataset_gdf[dataset_gdf['tile_number'] == tile_id]
            for date_t, timestamps in dates.items():
                subset = tile_dataset[tile_dataset['binned_date'] == date_t]
                if not subset.empty:
                    season = subset["season"].values[0]
                    params = summer_params if season == 1 else winter_params
                    executor.submit(main_shade, osmid, tile_id, timestamps, date_t, interval, params, combined=combined, building=building)
                elif date_t == solstice_day.date():
                    executor.submit(main_shade, osmid, tile_id, [None, []], date_t, interval, summer_params, combined=combined, building=building)
                else:
                    raise ValueError(f"No data available for the day: {date_t}")

def extract_and_merge_shade_values(dataset_gdf, osmid, parameters):
    """
    Extracts shade values for each timestamp-tile subset using parallel execution,
    based on specified simulation parameters.

    Parameters:
        dataset_gdf (GeoDataFrame): Input dataset with assigned tiles and timestamps.
        osmid (str): OSM ID used to locate raster files.
        parameters (dict): Dict with keys:
            - 'building_shade_step', 'tree_shade_step',
              'bldg_shadow_fraction', 'tree_shadow_fraction',
              'hours_before' (list of float/int)

    Returns:
        DataFrame: Concatenated results of all processed subsets with shade metrics.
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for tile_no in dataset_gdf["tile_number"].unique():
            tile_data = dataset_gdf[dataset_gdf["tile_number"] == tile_no]
            for timestamp in tile_data["rounded_timestamp"].unique():
                subset = tile_data[tile_data["rounded_timestamp"] == timestamp]
                future = executor.submit(process_subset, subset, osmid,
                                         building_shade_step=parameters['building_shade_step'],
                                         tree_shade_step=parameters['tree_shade_step'],
                                         bldg_shadow_fraction=parameters['bldg_shadow_fraction'],
                                         tree_shadow_fraction=parameters['tree_shadow_fraction'],
                                         hours_before=parameters['hours_before'])
                futures.append(future)

        for future in futures:
            results.append(future.result())

    return pd.concat(results, axis=0)

def aggregate_results(dataset_with_shade, original_dataset, unique_ID_column, parameters):
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

    if parameters['building_shade_step']:
        shade_columns.append("building_shade")
        shade_columns += [f"bldg_{h}_before_shadow_fraction" for h in parameters['hours_before']]

    if parameters['bldg_shadow_fraction']:
        shade_columns.append("bldg_shadow_fraction")

    if parameters['tree_shade_step']:
        shade_columns.append("tree_shade")
        shade_columns += [f"tree_{h}_before_shadow_fraction" for h in parameters['hours_before']]

    if parameters['tree_shadow_fraction']:
        shade_columns.append("tree_shadow_fraction")

    dataset_cleaned = dataset_with_shade.dropna(subset=shade_columns)
    dataset_aggregated = dataset_cleaned.groupby(unique_ID_column, as_index=False)[shade_columns].mean()

    merged = original_dataset.merge(dataset_aggregated, on=unique_ID_column, how="inner")
    return gpd.GeoDataFrame(merged, geometry='geometry')

# SHADE

def main_shade(osmid, tile_id, timestamps, date_c, shade_interval=30, inputs={'utc':1, 'dst':0, 'trs':10}, start_time=None,
               combined=False, building=False):
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
    print('in main shade')
    # Directory containing the raster files
    processing_dir = f"../data/clean_data/solar/{osmid}/rdy_for_processing/"
    # processing_dir = f'../data/clean_data/solar/{osmid}/rdy_for_processing/'

    if start_time is None:
        if isinstance(date_c, datetime):
            start_time = date_c.replace(hour=0, minute=0, second=0)
        else:
            # Assume it's a date, convert to datetime
            start_time = datetime.combine(date_c, datetime.min.time())

    building_file = [
        bldg_path for bldg_path in glob.glob(os.path.join(processing_dir, '*building_dsm.tif')) if (f"{tile_id}_" in bldg_path)
    ]

    canopy_file = [
        chm_path for chm_path in glob.glob(os.path.join(processing_dir, '*canopy_dsm.tif')) if (f"{tile_id}_" in chm_path)
    ]

    identifier = extract_identifier(building_file[0])

    if identifier+"_" in canopy_file[0]:
        matched_chm_path = canopy_file[0]
        # tile = "_".join(identifier.split('_')[-2:])
        shade_processing(building_file[0], matched_chm_path, osmid, date_c, shade_interval, timestamps, start_time, inputs, combined, building)
    else:
        raise Exception("Wasn't able to match chm_path to building path in shade processing")

def shade_processing(bldg_path, matched_chm_path, osmid, date, shade_interval, timestamps, start_time, inputs, combined, building):
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
        inputs (dict): Dictionary of simulation parameters, typically including:
            - 'utc' (int): UTC offset.
            - 'dst' (int): Daylight Saving Time offset.
            - 'trs' (float): Transmissivity value (e.g., 10 for 10%).
        combined (bool): If True, run tree shade simulation.
        building (bool): If True, run building shade simulation.

    Returns:
        None

    Notes:
        - Shade simulation is skipped for intervals with existing output files.
        - Output files are saved to '../results/output/{osmid}/building_shade/' and
          '../results/output/{osmid}/tree_shade/' respectively.
        - `directory_check` is used to determine which files already exist.
        - `filter_intervals` is used to avoid redundant computation.
    """
    def run_building_shade(inputs):
        shade_bldg = shade.shadecalculation_setup(
                    filepath_dsm=bldg_path,
                    filepath_veg=matched_chm_path,
                    tile_no=tile_no,
                    date=date,
                    intervalTime=shade_interval,
                    final_stamp=final_stamp,
                    start_time=start_time,
                    shade_fractions=building_intervals_needed,
                    onetime=0,
                    filepath_save=building_directory,
                    UTC=inputs['utc'],
                    dst=inputs['dst'],
                    useveg=0,
                    trunkheight=25,
                    # CHANGED TRANSMISSIVITY from 15 to 10 percent
                    transmissivity=inputs['trs']
                )

    def run_tree_shade(inputs):
        shade_veg = shade.shadecalculation_setup(
            filepath_dsm=bldg_path,
            filepath_veg=matched_chm_path,
            tile_no=tile_no,
            date=date,
            intervalTime=shade_interval,
            final_stamp=final_stamp,
            start_time=start_time,
            shade_fractions=tree_intervals_needed,
            onetime=0,
            filepath_save=tree_directory,
            UTC=inputs['utc'],
            dst=inputs['dst'],
            useveg=1,
            trunkheight=25,
            transmissivity=inputs['trs']
        )


    final_stamp, intervals = timestamps[0], timestamps[1]

    if final_stamp is not None:
        date = final_stamp
    else:
        date = datetime.combine(date, datetime.min.time()).replace(hour=23, minute=59, second=59)

    if not intervals:
        intervals = False

    bldg_path = bldg_path.replace("\\", "/")
    matched_chm_path = matched_chm_path.replace("\\", "/")
    identifier = extract_identifier(bldg_path)

    # Check if the file exists
    if os.path.isfile(matched_chm_path):
        print(f"The file {matched_chm_path} exists.")
    else:
        print(f"The file {matched_chm_path} does not exist.")

    # Create directories
    folder_no = identifier.split('_')[-1]
    folder_no = '/' + folder_no
    tile_no = identifier
    # tile_no = '/' + identifier

    building_directory = f"results/output/{osmid}/building_shade/{folder_no}/"
    tree_directory = f"results/output/{osmid}/tree_shade/{folder_no}/"

    # if shade_intervals is not empty, the return here is a list of booleans whether the file exists for each interval
    # if it is empty it is a single boolean for the final timestamp basically
    # TODO: Maybe should I add the final_stamp to the interval check somehow
    building_shadow_files_exist = directory_check(building_directory, shadow_check=True, shade_intervals=intervals, date=date)
    tree_shadow_files_exist = directory_check(tree_directory, shadow_check=True, shade_intervals=intervals, date=date)

    if intervals:
        # filter to only calculate intervals that don't have a file
        building_intervals_needed, tree_intervals_needed = filter_intervals(intervals, building_shadow_files_exist, tree_shadow_files_exist)
    else:
        building_intervals_needed = False
        tree_intervals_needed = False

    if building:
        print("Processing building shade...")
        if not building_shadow_files_exist:
            run_building_shade(inputs)
        elif isinstance(building_shadow_files_exist, list):
            if not all(building_shadow_files_exist):
                run_building_shade(inputs)

    if combined:
        print("Processing tree shade...")
        if not tree_shadow_files_exist:
            run_tree_shade(inputs)
        elif isinstance(tree_shadow_files_exist, list):
            if not all(tree_shadow_files_exist):
                run_tree_shade(inputs)

# SHADE DATA JOIN

def process_subset(timestamp_data, osmid, building_shade_step=False, tree_shade_step=False,
                   bldg_shadow_fraction=False, tree_shadow_fraction=False, hours_before=[], buffer=0,
                   binned=False):
    """
    Wrapper function for processing a subset of spatial-temporal data, intended for use in parallelized workflows.

    Parameters:
        timestamp_data (GeoDataFrame): Subset of the dataset to be processed, including point geometries,
                                       timestamps, and tile metadata.
        osmid (str): Unique identifier used to construct raster file paths.
        building_shade_step (bool): If True, extracts building shade values.
        tree_shade_step (bool): If True, extracts tree shade values.
        bldg_shadow_fraction (bool): If True, extracts building shadow fraction rasters.
        tree_shadow_fraction (bool): If True, extracts tree shadow fraction rasters.
        hours_before (list of int/float): List of hour values to compute time-averaged shadow fractions
                                          before each timestamp.
        buffer (float): Buffer distance (in meters) to apply around each point for raster extraction.
        binned (bool): If True, uses 'binned_date' instead of direct timestamps for raster matching.

    Returns:
        GeoDataFrame: The input `timestamp_data` with additional columns for each requested shade metric.
    """
    return get_dataset_shaderesult(
        timestamp_data, osmid,
        building_shade_step=building_shade_step,
        tree_shade_step=tree_shade_step,
        bldg_shadow_fraction=bldg_shadow_fraction,
        tree_shadow_fraction=tree_shadow_fraction,
        hours_before=hours_before,
        buffer=buffer,
        binned=binned
    )

def get_dataset_shaderesult(dataset, osmid, building_shade_step, tree_shade_step,
                            bldg_shadow_fraction, tree_shadow_fraction, hours_before,
                            buffer, binned):
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
        tree_shade_step (bool): If True, extracts instantaneous tree shade values.
        bldg_shadow_fraction (bool): If True, extracts building shadow fraction raster values.
        tree_shadow_fraction (bool): If True, extracts tree shadow fraction raster values.
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

    if binned:
        binned_date = dataset["binned_date"].unique()[0]
        if isinstance(binned_date, date) and not isinstance(binned_date, datetime):  # Use 'date' and 'datetime' from datetime module
            binned_date = datetime.combine(binned_date, time())  # Convert date-only to full datetime
            binned_rounded_ts = binned_date.replace(hour=rounded_timestamp.hour, minute=rounded_timestamp.minute, second=rounded_timestamp.second)
        else:
            binned_rounded_ts = binned_date.replace(hour=rounded_timestamp.hour, minute=rounded_timestamp.minute, second=rounded_timestamp.second)

        rounded_ts = binned_rounded_ts

    else:
        rounded_ts = rounded_timestamp

    # Define paths for shade rasters
    base_path = f"../code/results/output/{osmid}"

    building_shade_path = f"{base_path}/building_shade/{tile_number}/{osmid}_{tile_id}_Shadow_{rounded_ts.strftime('%Y%m%d_%H%M')}_LST.tif"
    tree_shade_path = f"{base_path}/tree_shade/{tile_number}/{osmid}_{tile_id}_Shadow_{rounded_ts.strftime('%Y%m%d_%H%M')}_LST.tif"
    bldg_shadow_fraction_path = f"{base_path}/building_shade/{tile_number}/{osmid}_{tile_id}_shadow_fraction_on_{rounded_ts.strftime('%Y%m%d_%H%M')}.tif"
    tree_shadow_fraction_path = f"{base_path}/tree_shade/{tile_number}/{osmid}_{tile_id}_shadow_fraction_on_{rounded_ts.strftime('%Y%m%d_%H%M')}.tif"

    # Initialize empty result DataFrame
    result_df = pd.DataFrame(index=dataset.index)

    # TODO: get building mask path and submit to extract_values_from_raster
    building_mask_file = [
        bldg_path for bldg_path in glob.glob(os.path.join(f"../data/clean_data/solar/{osmid}", '*mask.tif')) if f"{tile_id}_" in bldg_path
    ]

    if not building_mask_file:
        raise Exception("Couldn't find building mask file to extract shade values")
    else:
        building_mask_path = building_mask_file[0]

    # Extract values if the respective shade calculations exist
    if building_shade_step:
        result_df["building_shade"] = extract_values_from_raster(building_shade_path, building_mask_path, dataset, buffer)

    if tree_shade_step:
        result_df["tree_shade"] = extract_values_from_raster(tree_shade_path, building_mask_path, dataset, buffer)

    if bldg_shadow_fraction:
        result_df["bldg_shadow_fraction"] = extract_values_from_raster(bldg_shadow_fraction_path, building_mask_path, dataset, buffer)

    if tree_shadow_fraction:
        result_df["tree_shadow_fraction"] = extract_values_from_raster(tree_shadow_fraction_path, building_mask_path, dataset, buffer)

    if hours_before:
        for hr_before in hours_before:
            # Ensure `hours_before` is a valid number
            assert isinstance(hr_before, (int, float)), "hours_before must be an int or float"

            if tree_shade_step:
                result_df[f"tree_{hr_before}_before_shadow_fraction"] = hours_before_shadow_fr(
                    dataset, base_path, building_mask_path, "tree_shade", rounded_ts, tile_number, osmid, hr_before, buffer
                )

            if building_shade_step:
                result_df[f"bldg_{hr_before}_before_shadow_fraction"] = hours_before_shadow_fr(
                    dataset, base_path, building_mask_path, "building_shade", rounded_ts, tile_number, osmid, hr_before, buffer
                )

    # Merge results back into dataset
    dataset_final = pd.concat([dataset, result_df], axis=1)

    return dataset_final

def extract_values_from_raster(raster_path, building_mask_path, dataset, buffer=0):
    """
    Extracts shade (or similar raster) values at each point location in a dataset, optionally
    averaging over a surrounding buffer, and excluding areas covered by buildings.

    Parameters:
    ----------
    raster_path : str
        Path to the main raster file containing shade (or other) values.

    building_mask_path : str
        Path to a building mask raster file. Building pixels are assumed to have value 1.

    dataset : GeoDataFrame
        GeoDataFrame containing point geometries at which to extract values.

    buffer : float, optional (default = 0)
        Buffer radius in meters. If greater than 0, the average raster value
        is computed over a square window of surrounding pixels.

    Returns:
    -------
    np.ndarray
        Array of flipped raster values (`1 - value`) for each point.
        Returns NaN for:
        - Invalid raster coordinates,
        - Nodata raster values,
        - Points on buildings (for `buffer=0`), or
        - Buffers fully covered by buildings/nodata.
    """
    if not os.path.exists(raster_path):
        print(f"Warning: Raster file {raster_path} not found.")
        return np.full(len(dataset), np.nan)

    with rasterio.open(raster_path) as src, rasterio.open(building_mask_path) as bsrc:
        raster_data = src.read(1, masked=False)
        building_mask = bsrc.read(1, masked=False)

        raster_nodata = src.nodata if src.nodata is not None else np.nan
        building_nodata = bsrc.nodata if bsrc.nodata is not None else np.nan

        raster_transform = src.transform
        building_transform = bsrc.transform

        # Reproject points to match raster CRS
        dataset = dataset.to_crs(src.crs)
        dataset = dataset.reset_index(drop=True)

        values = np.full(len(dataset), np.nan)

        res_x, _ = src.res
        buffer_pixels = int(buffer / res_x) if buffer > 0 else 0

        for idx, row in dataset.iterrows():
            x, y = row.geometry.x, row.geometry.y
            record_id = row['RECORD']

            try:
                raster_row, raster_col = rowcol(raster_transform, x, y)
                building_row, building_col = rowcol(building_transform, x, y)
            except Exception as e:
                print(f"⚠️ Error converting coordinates for RECORD {record_id}: {e}")
                continue

            # Check bounds before accessing raster/building arrays
            if not (0 <= raster_row < raster_data.shape[0] and 0 <= raster_col < raster_data.shape[1]):
                print(f"❌ Raster index out of bounds for RECORD {record_id}")
                continue
            if not (0 <= building_row < building_mask.shape[0] and 0 <= building_col < building_mask.shape[1]):
                print(f"❌ Building mask index out of bounds for RECORD {record_id}")
                continue

            if buffer == 0:
                bm_value = building_mask[building_row, building_col]
                if bm_value == 1:
                    print(f"🚫 Point on building — RECORD: {record_id}")
                    values[idx] = np.nan
                else:
                    val = raster_data[raster_row, raster_col]
                    values[idx] = np.nan if val == raster_nodata else val
            else:
                # Raster window bounds
                row_start = max(raster_row - buffer_pixels, 0)
                row_end = min(raster_row + buffer_pixels + 1, raster_data.shape[0])
                col_start = max(raster_col - buffer_pixels, 0)
                col_end = min(raster_col + buffer_pixels + 1, raster_data.shape[1])
                raster_window = raster_data[row_start:row_end, col_start:col_end]

                # Building mask window bounds
                row_start_b = max(building_row - buffer_pixels, 0)
                row_end_b = min(building_row + buffer_pixels + 1, building_mask.shape[0])
                col_start_b = max(building_col - buffer_pixels, 0)
                col_end_b = min(building_col + buffer_pixels + 1, building_mask.shape[1])
                building_window = building_mask[row_start_b:row_end_b, col_start_b:col_end_b]

                # Match shapes by trimming to smallest size
                min_rows = min(raster_window.shape[0], building_window.shape[0])
                min_cols = min(raster_window.shape[1], building_window.shape[1])
                raster_window = raster_window[:min_rows, :min_cols]
                building_window = building_window[:min_rows, :min_cols]

                filtered = np.where(
                    (building_window == 1) | (raster_window == raster_nodata),
                    np.nan,
                    raster_window
                )
                valid_vals = filtered[~np.isnan(filtered)]
                if valid_vals.size == 0:
                    print(f"⚠️ Empty window after masking — RECORD: {record_id}")
                    values[idx] = np.nan
                else:
                    values[idx] = np.nanmean(valid_vals)


    values_flipped = np.where(np.isnan(values), np.nan, 1 - values)
    return values_flipped

def hours_before_shadow_fr(dataset, base_path, building_mask_path, shade_type, rounded_timestamp, tile_number, osmid, hours_before, buffer):
    """
    Computes the average shadow fraction for each point in the dataset by aggregating shadow data
    from raster files over a specified number of hours prior to a given timestamp.

    This function ensures temporal robustness by handling missing or misaligned raster files and
    adjusting the start time if necessary based on the earliest available data.

    Parameters:
        dataset (GeoDataFrame): GeoDataFrame containing the point geometries for which shadow fractions will be computed.
        base_path (str): Root directory where the shadow raster files are stored.
        building_mask_path (str): File path to a raster mask used to exclude building-covered areas when extracting values.
        shade_type (str): Type of shade raster to use (e.g., "tree_shade" or "building_shade").
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
        raise Exception("There are no shade files in the directory for this date")

    # Handle case where start_hour is before the first available shadow timestamp
    if start_hour <= first_shade_time:
        print("Start_hour is earlier or the same as first_shade_time, adjusting to first available time.")
        start_hour = first_shade_time

        # Construct the path to the shadow fraction raster file for `rounded_timestamp`
        timestamp_shadow_fraction_raster = f"{base_path}/{shade_type}/{tile_number}/{osmid}_p_{tile_number}_shadow_fraction_on_{rounded_timestamp.strftime('%Y%m%d_%H%M')}.tif"

        # If the shadow fraction raster file exists, extract values for all points
        return extract_values_from_raster(timestamp_shadow_fraction_raster, building_mask_path, dataset, buffer)  # Direct return if available

    # If the exact `start_hour` shadow file doesn't exist, find the closest valid one
    shadow_file_path = f"{base_path}/{shade_type}/{tile_number}/{osmid}_p_{tile_number}_Shadow_{start_hour.strftime('%Y%m%d_%H%M')}_LST.tif"

    if not os.path.exists(shadow_file_path):
        print(f"This shade file for start hour doesn't exist: {shadow_file_path}")
        start_hour_file = get_closest_shade_file(base_path, shade_type, tile_number, osmid, start_hour)
        start_hour = extract_datetime_from_path(start_hour_file)
        print(f"This is the new start hour: {start_hour}")
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
        raster_values = extract_values_from_raster(shade_raster, building_mask_path, dataset, buffer)
        shadow_values += np.nan_to_num(raster_values)  # Ensure NaN values don't affect summation

    # Compute the final shadow fraction (average)
    shadow_fractions = shadow_values / len(shade_files_for_shadow_frac)

    return shadow_fractions

def get_shade_files_in_range(base_path, shade_type, tile_number, osmid, start_hour, rounded_timestamp):
    """
    Get all tree shade files in a directory within the range of start_hour and rounded_timestamp (inclusive).

    Parameters:
        base_path (str): The base directory where tree shade files are stored.
        tile_number (str): The tile number for shade calculations.
        osmid (str): The unique ID for the dataset.
        start_hour (datetime): The lower bound timestamp (inclusive).
        rounded_timestamp (datetime): The upper bound timestamp (inclusive).

    Returns:
        list: List of full file paths that fall within the specified time range.
    """
    # Directory containing the tree shade files
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
        base_path (str): The base directory where tree shade files are stored.
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

def process_dataset(dataset, solstice_day, processed_raster_dir, longitude_column, latitude_column, timestamp_column,
                    dst_start, dst_end, interval=30, geometry=False, crs="EPSG:4326", simulate_solstice=False, bin_size=0):
    """
    Processes a spatiotemporal dataset by assigning each point to a raster tile and binning timestamps
    based on proximity to a reference solstice day.

    This function:
    - Converts a tabular dataset into a GeoDataFrame if needed.
    - Loads raster footprints from DSM tiles and assigns each point to a tile via spatial join.
    - Rounds timestamps to a specified interval.
    - Computes each point's temporal distance from a given solstice.
    - Bins points into groups based on temporal proximity to the solstice using `bin_data2`.
    - Assigns a seasonal label (summer or winter) based on daylight saving time bounds.

    Parameters:
        dataset (DataFrame): Input dataset containing at least longitude, latitude, and timestamp columns.
        solstice_day (datetime.datetime): Reference solstice date used for temporal binning.
        processed_raster_dir (str): Directory containing building DSM raster files with tile footprints.
        longitude_column (str): Name of the column containing longitude values.
        latitude_column (str): Name of the column containing latitude values.
        timestamp_column (str): Name of the column containing timestamp values.
        dst_start (datetime.date): Start date of daylight saving time.
        dst_end (datetime.date): End date of daylight saving time.
        interval (int, optional): Time interval in minutes to which timestamps will be rounded. Defaults to 30.
        geometry (bool, optional): If False, geometries are created from lat/lon columns. If True, assumes `geometry` column exists.
        crs (str, optional): Coordinate reference system for point geometries if creating them. Defaults to "EPSG:4326".
        simulate_solstice (bool, optional): If True, ensures a bin is added for the solstice even if no data points fall within its window.
        bin_size (int, optional): Number of days to use as the temporal grouping cutoff from the solstice. Defaults to 0 (no binning).

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
    if not geometry:
        dataset["ID"] = range(len(dataset))

        # Convert DataFrame to GeoDataFrame
        dataset["geometry"] = gpd.points_from_xy(dataset[longitude_column], dataset[latitude_column])
        df_gdf = gpd.GeoDataFrame(dataset, geometry="geometry", crs=crs)

    else:
        df_gdf = gpd.GeoDataFrame(dataset, geometry="geometry", crs="EPSG:32631")

    raster_files = glob.glob(os.path.join(processed_raster_dir, '*building_dsm.tif'))

    # Extract tile footprints from raster files
    raster_tiles = []
    raster_crs = None  # Store the raster CRS

    for raster_path in raster_files:
        match = re.search(r"p_\d+", raster_path)  # Extract tile number like 'p_0'
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

    # Reproject points to match raster CRS
    df_gdf = df_gdf.to_crs(raster_crs)

    # Spatial join to assign each point to the correct tile
    df_gdf = gpd.sjoin(df_gdf, tiles_gdf, how="left", predicate="intersects")

    # Drop unnecessary columns from spatial join
    df_gdf.drop(columns=["index_right"], inplace=True, errors="ignore")

    df_gdf = df_gdf.dropna(subset=["tile_number"])

    # Convert timestamp column to datetime
    df_gdf[timestamp_column] = pd.to_datetime(df_gdf[timestamp_column])

    # Apply correct rounding to nearest interval
    df_gdf["rounded_timestamp"] = df_gdf[timestamp_column].apply(lambda x: get_interval_stamp(x, interval))

    df_gdf["diff_solstice_day"] = df_gdf["rounded_timestamp"].dt.date - solstice_day.date()

    tile_grouped_days, modified_dataset = bin_data2(df_gdf, solstice_day, simulate_solstice, grouping_cutoff=bin_size)

    # modified_dataset = add_dataset_grouped_days(df_gdf, solstice_day, tile_grouped_days)

    modified_dataset["season"] = modified_dataset["binned_date"].apply(
    lambda date: assign_summer_winter(date, dst_start, dst_end)
    )

    modified_dataset = modified_dataset.reset_index()
    modified_dataset = modified_dataset.drop(['index', 'diff_solstice_day', 'abs_diff_solstice_day'], axis=1)

    return modified_dataset, tile_grouped_days

def bin_data2(dataset_gdf, solstice_day, simulate_solstice, grouping_cutoff=7):
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

        if len(filtered_rows) == 0 and not simulate_solstice:
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
    if dst_start.date() <= p_date < dst_end.date():
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

# RASTERS

def main_raster(osmid, raster_dir):
    """
    Processes all DSM raster files in a specified directory using multithreaded execution.

    Parameters:
        osmid (str): Unique identifier used in the processing logic (typically associated with the dataset or tile).
        raster_dir (str): Path to the directory containing DSM raster files.

    Returns:
        None

    Notes:
        - Uses `ThreadPoolExecutor` with up to 16 threads for concurrent processing.
        - Catches and prints any exceptions raised during individual raster processing tasks.
    """
    # Get a list of all raster files in the directory so we can load them incrementally
    raster_files = glob.glob(os.path.join(raster_dir, '*dsm.tif'))

    print(f"Processing {len(raster_files)} raster files.")

    # Use a ProcessPoolExecutor to process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(process_raster, file_path, osmid) for file_path in raster_files]
        # Optionally, wait for all tasks to complete and handle exceptions
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred: {e}")

def process_raster(path, osmid):
    try:
        print(f"Starting processing for {path} with OSMID: {osmid}")

        # Process each DSM file
        fixed_path = path.replace("\\", "/")
        last_slash_index = fixed_path.rfind("/")
        file_name = fixed_path[last_slash_index + 1:]

        # Define new file paths based on the osmid
        file_name_building = f"../data/clean_data/solar/{osmid}/rdy_for_processing/{file_name[:-7]}building_dsm.tif"
        file_name_trees = f"../data/clean_data/solar/{osmid}/rdy_for_processing/{file_name[:-7]}canopy_dsm.tif"
        # file_name_building = f'../data/clean_data/solar/{osmid}/rdy_for_processing/{file_name[:-7]}building_dsm.tif'
        # file_name_trees = f'../data/clean_data/solar/{osmid}/rdy_for_processing/{file_name[:-7]}canopy_dsm.tif'

        # List of file paths to check
        file_paths = [file_name_building, file_name_trees]

        # Check if the files already exist
        if check_files_exist(file_paths):
            print("Files already exist. Skipping creation.")
        else:
            # Read DSM
            with rasterio.open(path) as src:
                dsm_data = src.read(1)
                dsm_meta = src.meta.copy()
                dsm_crs = src.crs
                dsm_bounds = src.bounds
                dsm_transform = src.transform
                dsm_shape = dsm_data.shape

                # Extract further metadata
                width = src.width
                height = src.height
                nodata_value = src.nodata
                dtype = src.dtypes[0]

                # Calculate resolution
                resolution_x = dsm_transform[0]
                resolution_y = -dsm_transform[4]  # Typically negative in the geotransform

                # Calculate extent
                xmin = dsm_transform[2]
                ymax = dsm_transform[5]
                xmax = xmin + (width * resolution_x)
                ymin = ymax + (height * dsm_transform[4])  # Typically negative

            # # Create a bounding box polygon from the raster bounds
            dsm_bbox = box(dsm_bounds.left, dsm_bounds.bottom, dsm_bounds.right, dsm_bounds.top)
            dsm_bbox_gdf = gpd.GeoDataFrame({'geometry': [dsm_bbox]}, crs=dsm_crs)

            print("Making CHM mask")

            # New CHM mask path, identified by the OSMID and filename in the new folder
            chm_mask_folder = f"../data/clean_data/canopy_masks/{osmid}/"
            # chm_mask_folder = f'../data/clean_data/canopy_masks/{osmid}/'
            chm_mask_file = f'{chm_mask_folder}{file_name[:-7]}rgb_segmented.tif'

            if os.path.exists(chm_mask_file):
                # Load the CHM mask (assuming it’s already in the same resolution and extent)
                with rasterio.open(chm_mask_file) as chm_src:
                    chm_mask = chm_src.read(1)  # Assuming it's a single-band mask

                # Apply the CHM mask to the DSM data
                canopy_dsm = np.where(chm_mask, dsm_data, np.nan)  # Use NaN for masked-out areas

            else:
                print(f"CHM mask not found: {chm_mask_file}. Skipping mask application.")

            # Read building_mask
            mask_path = path.replace("dsm", "mask")

            with rasterio.open(mask_path) as src:
                bldg_mask = src.read(1)
                bldg_mask_meta = src.meta.copy()
                bldg_transform = src.transform
                bldg_crs = src.crs
                bldg_dtype = src.dtypes[0]

            # Load corresponding AHN subtiles
            # buildings_path = f'../data/clean_data/solar/{osmid}/{osmid}_buildings.gpkg'
            buildings_path = f"../data/clean_data/solar/{osmid}/{osmid}_buildings.gpkg"
            buildings = gpd.read_file(buildings_path, mask=dsm_bbox_gdf)

            # Check if buildings GeoDataFrame is empty
            if buildings.empty:
                print("No buildings found in the mask area.")
                osm_bldg_mask = np.zeros(dsm_data.shape, dtype='uint8')  # Create an empty mask
            else:
                if buildings.crs != dsm_crs:
                    buildings = buildings.to_crs(dsm_crs)

                # Buffer to combat artefacts.
                buildings.geometry = buildings.buffer(1.5)

                # Rasterize building polygons (same size as dsm so it works with UMEP)
                print("Rasterizing building polygons")
                try:
                    osm_bldg_mask = rasterize(
                        ((mapping(geom), 1) for geom in buildings.geometry),
                        out_shape=dsm_data.shape,
                        transform=dsm_meta['transform'],
                        fill=0,
                        dtype='uint8'
                    )
                except Exception as e:
                    print(f"Error during rasterization: {e}")
                    osm_bldg_mask = np.zeros(dsm_data.shape, dtype='uint8')  # Create an empty mask in case of failure

            combined_building_mask = np.logical_or(bldg_mask, osm_bldg_mask).astype(np.uint8)
            combined_bldg_tree_mask = np.logical_or(chm_mask, combined_building_mask).astype(np.uint8)

            dtm_raw = np.where(combined_bldg_tree_mask == 0, dsm_data, np.nan)

            ### Filter the raw data
            ## Apply minimum filter
            filtered_data = apply_minimum_filter(dtm_raw, np.nan, size=50)
            # filtered_data = apply_minimum_filter(filtered_data, np.nan, size=50)
            filtered_data = apply_minimum_filter(filtered_data, np.nan, size=30)
            filtered_data = apply_minimum_filter(filtered_data, np.nan, size=10)

            ### Interpolate:

            t = dsm_transform
            pts = []
            coords = []
            for i in range(filtered_data.shape[0]):
                for j in range(filtered_data.shape[1]):
                    x = t[2] + (j * t[0]) + (t[0] / 2)
                    y = t[5] + (i * t[4]) + (t[4] / 2)
                    z = filtered_data[i][j]
                    # Add all point coordinates. Laplace interpolation keeps existing values.
                    coords.append([x,y])
                    if not np.isnan(z):
                        pts.append([x, y, z])
                        # print('data found')
            dt = startinpy.DT()
            dt.insert(pts, insertionstrategy="BBox")

            interpolated = dt.interpolate({"method": "Laplace"}, coords)

            # Calculate the number of rows and columns
            ncols = int((xmax - xmin) / resolution_x)
            nrows = int((ymax - ymin) / resolution_y)

            # Create an empty raster array
            raster_array = np.full((nrows, ncols), np.nan, dtype=np.float32)

            # Ensure the points are in the correct structure (startinpy returns a flattened 1D array containing only the interpolated values for some reason)

            # Combine the coordinates and values into a 2D array with shape (n, 3)
            points = np.array([(x, y, val) for (x, y), val in zip(coords, interpolated)])
            points

            # Check if the array length is a multiple of 3
            if points.size % 3 != 0:
                raise ValueError(f"Array size {points.size} is not a multiple of 3, cannot reshape.")
            # Reshape the points array if it's flattened
            if points.ndim == 1:
                points = points.reshape(-1, 3)

            # Map the points to the raster grid
            for point in points:
                if len(point) != 3:
                    raise ValueError(f"Expected point to have 3 elements (x, y, value), but got {len(point)} elements.")
                x, y, value = point
                # Skip points with NaN values
                if np.isnan(value):
                    continue

                col = int((x - xmin) / resolution_x)
                row = int((ymax - y) / resolution_y)

                # Ensure the indices are within bounds
                if 0 <= col < ncols and 0 <= row < nrows:
                    raster_array[row, col] = value

            # Define the transform (mapping from pixel coordinates to spatial coordinates)
            transform = from_origin(xmin, ymax, resolution_x, resolution_y)

            # print(transform)

            # Define the metadata for the new raster
            meta = {
                'driver': 'GTiff',
                'dtype': dtype,
                'nodata': nodata_value,
                'width': width,
                'height': height,
                'count': 1,
                'crs': dsm_crs,
                'transform': transform
            }

            post_interpol_filter = apply_minimum_filter(raster_array, np.nan, size=40)
            post_interpol_filter = apply_minimum_filter(post_interpol_filter, np.nan, size=20)

            dsm_buildings = np.where(combined_building_mask == 0, post_interpol_filter, dsm_data)

            # Save building dsm and canopy dsm
            print("Saving DSM and Canopy DSM")

            # Find the index of the last '/' character
            path = path.replace("\\", "/")
            last_slash_index = path.rfind('/')
            # Extract the part after the last '/' (excluding '/')
            file_name = path[last_slash_index + 1:]
            file_name_building = f"../data/clean_data/solar/{osmid}/rdy_for_processing/" + file_name[:-7] + "building_dsm.tif"
            file_name_trees = f"../data/clean_data/solar/{osmid}/rdy_for_processing/" + file_name[:-7] + "canopy_dsm.tif"
            # file_name_building = f'../data/clean_data/solar/{osmid}/rdy_for_processing/' + file_name[:-7] + "building_dsm.tif"
            # file_name_trees = f'../data/clean_data/solar/{osmid}/rdy_for_processing/' + file_name[:-7] + "canopy_dsm.tif"

            # processing_directory = f'../data/clean_data/solar/{osmid}/rdy_for_processing/'
            processing_directory = f"../data/clean_data/solar/{osmid}/rdy_for_processing/"

            directory_check(directory=processing_directory, shadow_check=False)


            # Replace nan values with 0 for canopy raster:
            canopy_dsm = np.nan_to_num(canopy_dsm, nan=0)

            n = 50

            crop_and_save_raster(canopy_dsm, dsm_transform, dsm_meta, nodata_value, n,file_name_trees)
            crop_and_save_raster(dsm_buildings, dsm_transform, dsm_meta, nodata_value, n,file_name_building)

    except Exception as e:
        print(f"Error processing {path} with OSMID: {osmid}: {e}")

# Function to apply a median filter to a raster dataset
def apply_median_filter(data, nodata_value, size=3, nodata=True):

    if nodata:
        # Create a mask for nodata values
        mask = (data == nodata_value)

        # Apply the median filter only to valid data
        filtered_data = data.copy()
        filtered_data[~mask] = median_filter(data[~mask], size=size)
    else:
        filtered_data = data.copy()
        filtered_data = median_filter(data, size=size)

    return filtered_data


# Function to apply a mean filter to a raster dataset
def apply_mean_filter(data, nodata_value, size=3):
    # Create a mask for nodata values
    mask = (data == nodata_value)

    # Apply the mean filter only to valid data
    filtered_data = data.copy()
    filtered_data[~mask] = uniform_filter(data[~mask], size=size)

    return filtered_data

# Function to apply a Gaussian filter to a raster dataset
def apply_gaussian_filter(data, nodata_value, sigma=1):
    # Create a mask for nodata values
    mask = (data == nodata_value)

    # Apply the Gaussian filter only to valid data
    filtered_data = data.copy()
    filtered_data[~mask] = gaussian_filter(data[~mask], sigma=sigma)

    return filtered_data

# Function to apply a minimum filter to a raster dataset
def apply_minimum_filter(data, nodata_value, size=3, nodata=True):

    if nodata:
        # Create a mask for nodata values
        mask = (data == nodata_value)

        # Apply the Gaussian filter only to valid data
        filtered_data = data.copy()
        filtered_data[~mask] = minimum_filter(data[~mask], size=size)
    else:
        filtered_data = data.copy()
        filtered_data = minimum_filter(data, size=size)


    return filtered_data

def crop_and_save_raster(raster, transform, meta, nodata, n, out_path):
    # TODO: MAYBE JUST REPLACE THE NAN WITH MIN INSTEAD OF CROPPING?
    # Calculate new top-left corner coordinates
    new_x = transform.c + n * transform.a
    new_y = transform.f + n * transform.e

    # Calculate new transformation matrix
    new_transform = Affine(transform.a, transform.b, new_x,
                        transform.d, transform.e, new_y)

    # Crop the data by removing n pixels from each edge
    cropped_data = raster[n:-n, n:-n]

    # Find the minimum value of the non-NaN elements
    min_value = np.nanmin(cropped_data)

    # Fill NaN values with the minimum value
    cropped_data = np.where(np.isnan(cropped_data), min_value, cropped_data)

    # Update the metadata
    meta.update({
        'height': cropped_data.shape[0],
        'width': cropped_data.shape[1],
        'transform': new_transform
    })

    # Save the cropped raster data
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(cropped_data, 1)
        if nodata is not None:
            dst.nodata = nodata

def reproject_raster_to_dsm(src_raster, src_transform, src_crs, dst_crs, dst_transform, dst_shape):
    """
    Reprojects a source raster (e.g., a Canopy Height Model) to match the coordinate
    reference system and grid layout of a target raster (e.g., a Digital Surface Model).

    Parameters:
        src_raster (ndarray): 2D array of the source raster data to reproject.
        src_transform (Affine): Affine transform of the source raster.
        src_crs (CRS or str): Coordinate reference system of the source raster.
        dst_crs (CRS or str): Target coordinate reference system (e.g., DSM CRS).
        dst_transform (Affine): Affine transform of the destination raster grid.
        dst_shape (tuple): Shape (height, width) of the destination raster.

    Returns:
        ndarray: The reprojected raster array aligned to the destination CRS and grid.
    """
    dst_raster = np.empty(dst_shape, dtype=src_raster.dtype)
    reproject(
        source=src_raster,
        destination=dst_raster,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest
    )
    return dst_raster

def update_mask_within_extent(raster_path, combined_mask, dsm_bounds, dsm_transform, dsm_crs, dsm_shape):
    """
    Updates a combined mask by integrating data from a secondary raster (e.g., CHM)
    that overlaps partially or fully with a DSM tile, accounting for CRS differences.

    Parameters:
        raster_path (str): File path to the CHM (or other secondary) raster.
        combined_mask (ndarray): 2D boolean array representing the current cumulative mask.
        dsm_bounds (BoundingBox): Bounding box of the DSM tile.
        dsm_transform (Affine): Affine transform of the DSM raster.
        dsm_crs (CRS or str): Coordinate reference system of the DSM raster.
        dsm_shape (tuple): Shape (height, width) of the DSM raster.

    Returns:
        None: The function modifies `combined_mask` in-place by OR-ing the mask derived
              from the reprojected CHM within the overlapping region.

    Notes:
        - Assumes the mask is created from non-zero values in the secondary raster.
        - Skips update if the calculated overlap window falls outside the mask bounds.
    """
    with rasterio.open(raster_path) as src:
        # Reproject the entire additional raster to the DSM's CRS
        src_data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs

        reprojected_data = reproject_raster_to_dsm(
            src_data, src_transform, src_crs, dsm_crs, dsm_transform, dsm_shape
        )

        # Compute the window of the reprojected raster that overlaps with the DSM extent
        window = from_bounds(
            left=dsm_bounds.left, bottom=dsm_bounds.bottom,
            right=dsm_bounds.right, top=dsm_bounds.top,
            transform=dsm_transform
        )

        # Convert window indices to integers
        row_off = int(window.row_off)
        col_off = int(window.col_off)
        height = int(window.height)
        width = int(window.width)

        # Extract the overlapping window from the reprojected raster
        window_data = reprojected_data[
            row_off:row_off + height,
            col_off:col_off + width
        ]

        # Create a mask from the reprojected raster within the window
        additional_mask = window_data > 0  # Example condition to create a mask from the additional raster

        # Update the combined mask using the window's index
        row_start, col_start = row_off, col_off
        row_end, col_end = row_start + additional_mask.shape[0], col_start + additional_mask.shape[1]

        # Ensure the indices are within the bounds of the combined mask
        if row_start < 0 or col_start < 0 or row_end > combined_mask.shape[0] or col_end > combined_mask.shape[1]:
            print(f"Skipping {raster_path} due to out of bounds indices")
            return

        combined_mask[row_start:row_end, col_start:col_end] |= additional_mask

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
                        print(f"File containing 'shadow_fraction_on_{int_time}' found: {shadow_files_interval}")
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

def filter_intervals(intervals, building_shadow_files_exist, tree_shadow_files_exist):
    """
    Filters a list of time intervals to determine which intervals still require processing
    based on the existence of building and tree shadow files.

    Parameters:
        intervals (list): A list of time interval identifiers (e.g., datetime strings or objects).
        building_shadow_files_exist (list or bool): If a list, it must match the length of `intervals`
            and contain booleans indicating the presence of corresponding building shadow files.
            If False, all intervals are considered needed. If True, no intervals are needed.
        tree_shadow_files_exist (list or bool): Same as `building_shadow_files_exist`, but for tree shadow files.

    Returns:
        tuple:
            - building_intervals_needed (list or bool): List of intervals that require building shadow
              processing, or False if none are needed.
            - tree_intervals_needed (list or bool): List of intervals that require tree shadow
              processing, or False if none are needed.

    Raises:
        AssertionError: If `building_shadow_files_exist` or `tree_shadow_files_exist` are lists but
        do not match the length of `intervals`.
    """
    # filter to only calculate intervals that don't have a file
    if isinstance(building_shadow_files_exist, list):
        assert len(intervals) == len(building_shadow_files_exist), "Directory check for the intervals is broken"
        building_intervals_needed = [intervals[i] for i, check in enumerate(building_shadow_files_exist) if not check]
        if len(building_intervals_needed) < 1:
            building_intervals_needed = False
    elif not building_shadow_files_exist:
        building_intervals_needed = intervals
    else:
        building_intervals_needed = False

    if isinstance(tree_shadow_files_exist, list):
        assert len(intervals) == len(tree_shadow_files_exist), "Directory check for the intervals is broken"
        tree_intervals_needed = [intervals[i] for i, check in enumerate(tree_shadow_files_exist) if not check]
        if len(tree_intervals_needed) < 1:
            tree_intervals_needed = False
    elif not tree_shadow_files_exist:
        tree_intervals_needed = intervals
    else:
        tree_intervals_needed = False

    return building_intervals_needed, tree_intervals_needed
