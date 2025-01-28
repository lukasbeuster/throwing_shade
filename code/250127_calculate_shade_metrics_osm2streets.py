import os
import re
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterstats import zonal_stats
import numpy as np
import argparse
from shapely.geometry import box
import tqdm
import datetime as dt
import pandas as pd
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

def main(osmid, dates):
    """
    Main function to process shade metrics for a given OSMID and multiple dates.

    Parameters:
        osmid (int): OSMID of the area to process.
        dates (list): List of dates to process in datetime format.
    """
    print(f"Working on OSMID: {osmid}; Dates: {dates}")

    # Load polygons from osm2streets lane data
    polygons = load_sidewalk_data(osmid)

    # Ensure spatial index exists for polygons
    if not polygons.sindex:
        polygons.sindex = polygons.sindex

    for date in dates:
        timestr = date.strftime("%Y%m%d")
        print(f"Processing date: {timestr}")

        # Process building and tree files separately
        polygons = process_shade_metrics(osmid, timestr, polygons)

    # Save the updated polygons after processing all dates
    output_path = f"../results/output/{osmid}/{osmid}_sidewalks_with_stats_multiple_dates.gpkg"
    polygons.to_file(output_path, driver="GPKG")
    print(f"Results saved to {output_path}")

def load_sidewalk_data(osmid):
    """
    Load and filter osm2streets lane data for sidewalks.

    Parameters:
        osmid (int): OSMID of the area to process.

    Returns:
        GeoDataFrame: GeoDataFrame containing sidewalk polygons.
    """
    input_dir = f"../data/raw_data/osm2streets/{osmid}/processed/"
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Lane data folder does not exist: {input_dir}")

    files = [f for f in os.listdir(input_dir) if f.endswith('lanes.geojson')]
    if not files:
        raise FileNotFoundError(f"No lane data found in: {input_dir}")

    filepath = os.path.join(input_dir, files[0])
    lanes = gpd.read_file(filepath, engine="pyogrio")  # Specify pyogrio as the engine
    # lanes = gpd.read_file(filepath)

    # Filter for sidewalks and footpaths
    valid_types = ['Sidewalk', 'Footway']
    sidewalks = lanes[lanes['type'].isin(valid_types)]

    print(f"Loaded {len(sidewalks)} sidewalk features from {filepath}")
    return sidewalks

def process_shade_metrics(osmid, timestr, polygons):
    """
    Process shade metrics for a given date and update the polygon dataset.

    Parameters:
        osmid (int): OSMID of the area to process.
        timestr (str): Date in YYYYMMDD format.
        timestr_extension (str): Date with extension (e.g., YYYYMMDD.tif).
        polygons (GeoDataFrame): GeoDataFrame containing sidewalk polygons.

    Returns:
        GeoDataFrame: Updated polygons with shade metrics.
    """
    # Reset the index to ensure unique labels
    polygons = polygons.reset_index(drop=True)
    
    for bldg_tree in ['building', 'tree']:
        root_directory = f"../results/output/{osmid}/{bldg_tree}_shade/"
        print(f"Processing {bldg_tree} shade for {timestr}")

        # Find raster files
        raster_files = find_raster_files(root_directory, f"{timestr}.tif")
        if not raster_files:
            print(f"[DEBUG] No daily raster files found for {bldg_tree} on {timestr}. Skipping.")
            continue

        # Open the first raster file and extract the CRS
        try:
            with rasterio.open(raster_files[0]) as src:
                raster_crs = src.crs
                print(f"[DEBUG] Raster CRS for {raster_files[0]}: {raster_crs}")
        except Exception as e:
            print(f"[ERROR] Failed to open raster file {raster_files[0]}. Error: {e}")
            raise

        # Ensure polygons have a valid CRS
        if polygons.crs is None:
            raise ValueError("[ERROR] Polygons do not have a CRS set. Please ensure the input data has a defined CRS.")

        # Debug: Show polygon CRS before transformation
        print(f"[DEBUG] Polygons CRS before transformation: {polygons.crs}")

        # Transform polygons to raster CRS if necessary
        if polygons.crs != raster_crs:
            print(f"[DEBUG] Transforming polygons from {polygons.crs} to match raster CRS.")
            polygons = polygons.to_crs(raster_crs)
        else:
            print(f"[DEBUG] Polygons CRS already matches raster CRS.")

        # Process daily statistics
        polygons = process_daily_stats(polygons, root_directory, timestr, bldg_tree)

        # # Process hourly statistics
        # polygons = process_hourly_statistics(polygons, root_directory, bldg_tree, timestr)

        # Process hourly statistics in parallel
        polygons = process_hourly_statistics_parallel(polygons, root_directory, bldg_tree, timestr)


    return polygons

def process_daily_stats(polygons, root_directory, timestr, bldg_tree):
    """
    Calculate daily statistics for each tile separately and update polygons.
    """
    raster_files = find_raster_files(root_directory, f"{timestr}.tif")
    if not raster_files:
        print(f"No daily raster files found for {bldg_tree} on {timestr}")
        return polygons
    

    # Ensure spatial index exists for polygons
    if not polygons.sindex:
        polygons.sindex = polygons.sindex

    for tile in raster_files:
        print(f"[DEBUG] Processing daily tile: {tile}")
        with rasterio.open(tile) as src:
            raster_data = src.read(1)
            affine = src.transform

            # Clip polygons to the tile's bounds
            tile_bounds = box(*src.bounds)

            # Use spatial index to filter polygons
            possible_matches_index = list(polygons.sindex.intersection(tile_bounds.bounds))
            possible_matches = polygons.iloc[possible_matches_index]

            # Clip polygons to the tile's bounds
            clipped_polygons = gpd.clip(possible_matches, tile_bounds)

            # Skip processing if no polygons intersect the tile
            if clipped_polygons.empty:
                print(f"[DEBUG] No polygons intersect tile {tile}. Skipping.")
                continue

            # Ensure only valid geometries are passed to zonal_stats
            clipped_polygons = clipped_polygons[clipped_polygons.geometry.is_valid]

            # Compute zonal stats
            stats = zonal_stats(
                clipped_polygons.geometry, raster_data,
                affine=affine, stats=["mean", "std", "min", "max"],
                nodata=src.nodata
            )

            # Update polygons with daily stats
            for stat_type in ["mean", "std", "min", "max"]:
                column_name = f"{timestr}_{bldg_tree}_{stat_type}"
                if column_name not in polygons.columns:
                    polygons[column_name] = np.nan

                update_values = pd.Series(
                    [s[stat_type] for s in stats],
                    index=clipped_polygons.index
                )
                polygons[column_name] = polygons[column_name].combine_first(update_values)

    return polygons


def process_hourly_statistics(polygons, root_directory, bldg_tree, timestr):
    """
    Process hourly statistics for a given date, tile by tile, and update the polygon dataset.
    """
    available_times = find_available_times(root_directory, timestr)
    print(f"Available times for {bldg_tree} on {timestr}: {available_times}")

    # Ensure spatial index exists for polygons
    if not polygons.sindex:
        polygons.sindex = polygons.sindex

    # Initialize columns for all available times
    for time in available_times:
        shade_column = f"{timestr}_{bldg_tree}_shade_percent_at_{time}"
        if shade_column not in polygons.columns:
            polygons[shade_column] = np.nan

    for time in available_times:
        timestamp = f"{timestr}_{time}_LST.tif"
        print(f"Processing hourly stats at {time} for {bldg_tree}")

        hour_files = find_raster_files(root_directory, timestamp)
        if not hour_files:
            print(f"No hourly files found for {time} on {timestr}")
            continue

        for tile in hour_files:
            print(f"[DEBUG] Processing hourly tile: {tile}")
            with rasterio.open(tile) as src:
                raster_data = src.read(1)
                affine = src.transform

                # Clip polygons to the tile's bounds
                tile_bounds = box(*src.bounds)

                # Use spatial index to filter polygons
                possible_matches_index = list(polygons.sindex.intersection(tile_bounds.bounds))
                possible_matches = polygons.iloc[possible_matches_index]

                # Clip polygons to the tile's bounds
                clipped_polygons = gpd.clip(possible_matches, tile_bounds)

                # Skip processing if no polygons intersect the tile
                if clipped_polygons.empty:
                    print(f"[DEBUG] No polygons intersect tile {tile}. Skipping.")
                    continue

                # Ensure only valid geometries are passed to zonal_stats
                clipped_polygons = clipped_polygons[clipped_polygons.geometry.is_valid]

                # Replace nodata values with NaN for processing
                raster_data = np.where(raster_data == src.nodata, np.nan, raster_data)

                # Invert the raster for shaded pixels
                inverted_raster = np.where(
                    (raster_data < 1) & (raster_data >= 0), 1,  # Shaded pixels
                    np.where(raster_data == 1, 0, np.nan)       # Exposed pixels and NaN
                )

                # Compute zonal stats
                stats = zonal_stats(
                    clipped_polygons.geometry, inverted_raster,
                    affine=affine, stats=["sum", "count"], nodata=np.nan
                )

                # Update polygons with hourly stats
                shade_column = f"{timestr}_{bldg_tree}_shade_percent_at_{time}"
                update_values = pd.Series(
                    [
                        (s["sum"] / s["count"]) * 100 if s["count"] > 0 else np.nan
                        for s in stats
                    ],
                    index=clipped_polygons.index
                )
                polygons[shade_column] = polygons[shade_column].combine_first(update_values)

    return polygons

def process_time_step_wrapper(args):
    """
    Wrapper function for process_time_step to support parallel processing.

    Parameters:
        args (tuple): Tuple of arguments for process_time_step.

    Returns:
        Any: Result of process_time_step.
    """
    return process_time_step(*args)


def process_time_step(time, polygons, root_directory, bldg_tree, timestr):
    """
    Process a single time step for hourly statistics.

    Parameters:
        time (str): Time of the hourly raster (e.g., '0600').
        polygons (GeoDataFrame): GeoDataFrame containing sidewalk polygons.
        root_directory (str): Directory containing raster files.
        bldg_tree (str): Type of shade ('building' or 'tree').
        timestr (str): Date in YYYYMMDD format.

    Returns:
        GeoDataFrame: Updated polygons for the time step.
    """
    timestamp = f"{timestr}_{time}_LST.tif"
    hour_files = find_raster_files(root_directory, timestamp)
    if not hour_files:
        print(f"No hourly files found for {time} on {timestr}")
        return None

    shade_column = f"{timestr}_{bldg_tree}_shade_percent_at_{time}"
    results = []

    for tile in hour_files:
        print(f"[DEBUG] Processing hourly tile: {tile}")
        with rasterio.open(tile) as src:
            raster_data = src.read(1)
            affine = src.transform

            # Clip polygons to the tile's bounds
            tile_bounds = box(*src.bounds)
            clipped_polygons = gpd.clip(polygons, tile_bounds)

            if clipped_polygons.empty:
                print(f"[DEBUG] No polygons intersect tile {tile}. Skipping.")
                continue

            clipped_polygons = clipped_polygons[clipped_polygons.geometry.is_valid]
            raster_data = np.where(raster_data == src.nodata, np.nan, raster_data)
            inverted_raster = np.where(
                (raster_data < 1) & (raster_data >= 0), 1,
                np.where(raster_data == 1, 0, np.nan)
            )

            stats = zonal_stats(
                clipped_polygons.geometry, inverted_raster,
                affine=affine, stats=["sum", "count"], nodata=np.nan
            )

            update_values = pd.Series(
                [
                    (s["sum"] / s["count"]) * 100 if s["count"] > 0 else np.nan
                    for s in stats
                ],
                index=clipped_polygons.index
            )
            results.append(update_values)

    if results:
        combined_results = pd.concat(results, axis=0)
        return shade_column, combined_results
    return None


def process_hourly_statistics_parallel(polygons, root_directory, bldg_tree, timestr):
    """
    Parallelized processing of hourly statistics.

    Parameters:
        polygons (GeoDataFrame): GeoDataFrame containing sidewalk polygons.
        root_directory (str): Directory containing raster files for the day.
        bldg_tree (str): Type of shade ('building' or 'tree').
        timestr (str): Date in YYYYMMDD format.

    Returns:
        GeoDataFrame: Updated polygons with hourly statistics.
    """
    available_times = find_available_times(root_directory, timestr)
    print(f"Available times for {bldg_tree} on {timestr}: {available_times}")

    for time in available_times:
        shade_column = f"{timestr}_{bldg_tree}_shade_percent_at_{time}"
        if shade_column not in polygons.columns:
            polygons[shade_column] = np.nan

    # Prepare arguments for parallel processing
    args = [(time, polygons, root_directory, bldg_tree, timestr) for time in available_times]

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_time_step_wrapper, args))

    for result in results:
        if result:
            shade_column, combined_results = result

            # Ensure indices in combined_results are unique
            if not combined_results.index.is_unique:
                print(f"[DEBUG] Duplicate indices in combined_results for {shade_column}. Resolving duplicates...")
                combined_results = combined_results.groupby(level=0).mean()  # Aggregate duplicates

            # Update polygons only for rows present in combined_results
            polygons.loc[combined_results.index, shade_column] = combined_results

    return polygons

# def process_hourly_statistics_parallel(polygons, root_directory, bldg_tree, timestr):
#     """
#     Parallelized processing of hourly statistics.

#     Parameters:
#     polygons (GeoDataFrame): GeoDataFrame containing sidewalk polygons.
#     root_directory (str): Directory containing raster files for the day.
#     bldg_tree (str): Type of shade ('building' or 'tree').
#     timestr (str): Date in YYYYMMDD format.

#     Returns:
#         GeoDataFrame: Updated polygons with hourly statistics.
#     """
#     available_times = find_available_times(root_directory, timestr)
#     print(f"Available times for {bldg_tree} on {timestr}: {available_times}")

#     for time in available_times:
#         shade_column = f"{timestr}_{bldg_tree}_shade_percent_at_{time}"
#         if shade_column not in polygons.columns:
#             polygons[shade_column] = np.nan

#     # Prepare arguments for parallel processing
#     args = [(time, polygons, root_directory, bldg_tree, timestr) for time in available_times]

#     with ProcessPoolExecutor() as executor:
#         results = list(executor.map(process_time_step_wrapper, args))

#     for result in results:
#         if result:
#             shade_column, combined_results = result

#             # Ensure unique index in combined_results
#             combined_results = combined_results.reset_index(drop=True)

#             polygons[shade_column] = polygons[shade_column].combine_first(combined_results)

#     return polygons


# def process_daily_stats(polygons, root_directory, timestr, bldg_tree):
#     """
#     Calculate daily statistics for each tile separately and update polygons.

#     Parameters:
#         polygons (GeoDataFrame): GeoDataFrame containing sidewalk polygons.
#         root_directory (str): Directory containing daily raster tiles.
#         timestr (str): Date in YYYYMMDD format.
#         bldg_tree (str): Type of shade ('building' or 'tree').

#     Returns:
#         GeoDataFrame: Updated polygons with daily statistics.
#     """
#     raster_files = find_raster_files(root_directory, f"{timestr}.tif")
#     if not raster_files:
#         print(f"No daily raster files found for {bldg_tree} on {timestr}")
#         return polygons

#     for tile in raster_files:
#         print(f"[DEBUG] Processing daily tile: {tile}")
#         with rasterio.open(tile) as src:
#             raster_data = src.read(1)
#             affine = src.transform

#             # Clip polygons to the tile's bounds
#             tile_bounds = box(*src.bounds)
#             clipped_polygons = gpd.clip(polygons, tile_bounds)

#             # Skip processing if no polygons intersect the tile
#             if clipped_polygons.empty:
#                 print("[DEBUG] No polygons intersect this tile. Skipping.")
#                 continue

#             # Ensure only valid geometries are passed to zonal_stats
#             clipped_polygons = clipped_polygons[clipped_polygons.geometry.is_valid]

#             # Compute zonal stats
#             stats = zonal_stats(
#                 clipped_polygons.geometry, raster_data,
#                 affine=affine, stats=["mean", "std", "min", "max"],
#                 nodata=src.nodata
#             )

#             # Update polygons with daily stats
#             for stat_type in ["mean", "std", "min", "max"]:
#                 column_name = f"{timestr}_{bldg_tree}_{stat_type}"
#                 if column_name not in polygons.columns:
#                     polygons[column_name] = np.nan

#                 # Update only the relevant rows
#                 update_values = pd.Series(
#                     [s[stat_type] for s in stats],
#                     index=clipped_polygons.index
#                 )
#                 polygons[column_name] = polygons[column_name].combine_first(update_values)

#     return polygons


# def process_hourly_statistics(polygons, root_directory, bldg_tree, timestr):
#     """
#     Process hourly statistics for a given date, tile by tile, and update the polygon dataset.

#     Parameters:
#         polygons (GeoDataFrame): GeoDataFrame containing sidewalk polygons.
#         root_directory (str): Directory containing raster files for the day.
#         bldg_tree (str): Type of shade ('building' or 'tree').
#         timestr (str): Date in YYYYMMDD format.

#     Returns:
#         GeoDataFrame: Updated polygons with hourly statistics.
#     """
#     available_times = find_available_times(root_directory, timestr)
#     print(f"Available times for {bldg_tree} on {timestr}: {available_times}")

#     # Initialize columns for all available times
#     for time in available_times:
#         shade_column = f"{timestr}_{bldg_tree}_shade_percent_at_{time}"
#         if shade_column not in polygons.columns:
#             polygons[shade_column] = np.nan

#     for time in available_times:
#         timestamp = f"{timestr}_{time}_LST.tif"
#         print(f"Processing hourly stats at {time} for {bldg_tree}")

#         hour_files = find_raster_files(root_directory, timestamp)
#         if not hour_files:
#             print(f"No hourly files found for {time} on {timestr}")
#             continue

#         for tile in hour_files:
#             print(f"[DEBUG] Processing hourly tile: {tile}")
#             with rasterio.open(tile) as src:
#                 raster_data = src.read(1)
#                 affine = src.transform

#                 # Clip polygons to the tile's bounds
#                 tile_bounds = box(*src.bounds)
#                 clipped_polygons = gpd.clip(polygons, tile_bounds)

#                 # Skip processing if no polygons intersect the tile
#                 if clipped_polygons.empty:
#                     print("[DEBUG] No polygons intersect this tile. Skipping.")
#                     continue

#                 # Ensure only valid geometries are passed to zonal_stats
#                 clipped_polygons = clipped_polygons[clipped_polygons.geometry.is_valid]

#                 # Replace nodata values with NaN for processing
#                 raster_data = np.where(raster_data == src.nodata, np.nan, raster_data)

#                 # Invert the raster for shaded pixels
#                 inverted_raster = np.where(
#                     (raster_data < 1) & (raster_data >= 0), 1,  # Shaded pixels
#                     np.where(raster_data == 1, 0, np.nan)       # Exposed pixels and NaN
#                 )

#                 # Compute zonal stats
#                 stats = zonal_stats(
#                     clipped_polygons.geometry, inverted_raster,
#                     affine=affine, stats=["sum", "count"], nodata=np.nan
#                 )

#                 # Update polygons with hourly stats
#                 shade_column = f"{timestr}_{bldg_tree}_shade_percent_at_{time}"
#                 update_values = pd.Series(
#                     [
#                         (s["sum"] / s["count"]) * 100 if s["count"] > 0 else np.nan
#                         for s in stats
#                     ],
#                     index=clipped_polygons.index
#                 )
#                 polygons[shade_column] = polygons[shade_column].combine_first(update_values)

#     return polygons

def find_raster_files(root_dir, file_pattern):
    """
    Find raster files matching a specific pattern in a directory.

    Parameters:
        root_dir (str): Directory to search.
        file_pattern (str): File pattern to match.

    Returns:
        list: List of matching file paths.
    """
    return [os.path.join(root, f) for root, _, files in os.walk(root_dir) for f in files if file_pattern in f]

def find_available_times(root_dir, timestr):
    """
    Extract available times for hourly rasters from filenames.

    Parameters:
        root_dir (str): Directory containing raster files.
        timestr (str): Date in YYYYMMDD format.

    Returns:
        list: Sorted list of available times as strings (e.g., ['0600', '1200']).
    """
    time_pattern = re.compile(rf".*{timestr}_(\d{{4}})_LST\.tif")
    available_times = set()

    for root, _, files in os.walk(root_dir):
        for file in files:
            match = time_pattern.match(file)
            if match:
                available_times.add(match.group(1))

    return sorted(available_times)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a OSM area for shade metrics.")
    parser.add_argument('osmid', type=int, help='OSMID to be processed')
    parser.add_argument('dates', type=str, nargs='+', help='Dates in YYYY-MM-DD format (multiple dates allowed)')
    args = parser.parse_args()

    osmid = args.osmid
    dates_input = args.dates
    dates = [dt.datetime.strptime(date, "%Y-%m-%d") for date in dates_input]

    main(osmid, dates)