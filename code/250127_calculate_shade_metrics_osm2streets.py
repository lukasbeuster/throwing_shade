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

def main(osmid, dates):
    """
    Main function to process shade metrics for a given OSMID and multiple dates.

    Parameters:
        osmid (int): OSMID of the area to process.
        dates (list): List of dates to process in datetime format.
    """
    print(f'Working on OSMID: {osmid}; Dates: {dates}')

    # Load polygons from osm2streets lane data
    polygons = load_sidewalk_data(osmid)

    for date in dates:
        timestr = date.strftime("%Y%m%d")
        print(f'Processing date: {timestr}')

        # Process building and tree files separately
        for bldg_tree in ['building', 'tree']:
            root_directory = f"../results/output/{osmid}/{bldg_tree}_shade/"

            # Process daily stats for individual tiles
            polygons = process_daily_stats(polygons, root_directory, timestr, bldg_tree)

            # Process hourly stats for individual tiles
            polygons = process_hourly_stats(polygons, root_directory, timestr, bldg_tree)

    # Save the updated polygons after processing all dates
    output_path = f'../results/output/{osmid}/{osmid}_sidewalks_with_stats_multiple_dates.gpkg'
    polygons.to_file(output_path, driver='GPKG')
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
    lanes = gpd.read_file(filepath)

    # Filter for sidewalks and footpaths
    valid_types = ['Sidewalk', 'Footway']
    sidewalks = lanes[lanes['type'].isin(valid_types)]

    print(f"Loaded {len(sidewalks)} sidewalk features from {filepath}")
    return sidewalks

def process_daily_stats(polygons, root_directory, timestr, bldg_tree):
    """
    Calculate daily statistics for each tile separately and update polygons.

    Parameters:
        polygons (GeoDataFrame): GeoDataFrame containing sidewalk polygons.
        root_directory (str): Directory containing daily raster tiles.
        timestr (str): Date in YYYYMMDD format.
        bldg_tree (str): Type of shade ('building' or 'tree').

    Returns:
        GeoDataFrame: Updated polygons with daily statistics.
    """
    raster_files = find_raster_files(root_directory, f"{timestr}.tif")
    if not raster_files:
        print(f"No daily raster files found for {bldg_tree} on {timestr}")
        return polygons

    for tile in raster_files:
        print(f"[DEBUG] Processing daily tile: {tile}")

        with rasterio.open(tile) as src:
            raster_data = src.read(1)
            affine = src.transform

            # Replace nodata values with NaN for consistency
            nodata = src.nodata
            if nodata is not None:
                raster_data = np.where(raster_data == nodata, np.nan, raster_data)

            # Compute zonal stats for the tile
            stats = zonal_stats(
                polygons,
                raster_data,
                affine=affine,
                stats=['mean', 'std', 'min', 'max'],
                nodata=np.nan,
            )

            # Update polygons with calculated statistics
            for stat_type in ['mean', 'std', 'min', 'max']:
                column_name = f"{timestr}_{bldg_tree}_{stat_type}"

                # Initialize the column with NaN if it doesn't exist
                if column_name not in polygons:
                    polygons[column_name] = np.nan

                # Create a temporary Series for the current stats
                temp_series = pd.Series(
                    [s[stat_type] if s[stat_type] is not None else np.nan for s in stats],
                    index=polygons.index
                )

                # Use combine_first to preserve existing values
                polygons[column_name] = polygons[column_name].combine_first(temp_series)

    return polygons

def process_hourly_stats(polygons, root_directory, bldg_tree, timestr):
    """
    Process hourly statistics for a given date, tile by tile, and update the polygon dataset.

    Parameters:
        polygons (GeoDataFrame): GeoDataFrame containing sidewalk polygons.
        root_directory (str): Directory containing raster files for the day.
        bldg_tree (str): Type of shade ('building' or 'tree').
        timestr (str): Date in YYYYMMDD format.

    Returns:
        GeoDataFrame: Updated polygons with hourly statistics.
    """
    # Get all available hourly times for the current directory and date
    available_times = find_available_times(root_directory, timestr)
    print(f"Available times for {bldg_tree} on {timestr}: {available_times}")

    # Initialize columns for all available times (if not already initialized)
    for time in available_times:
        shade_column = f"{timestr}_{bldg_tree}_shade_percent_at_{time}"
        if shade_column not in polygons.columns:
            polygons[shade_column] = np.nan  # Initialize with NaN

    # Process each tile for each available time
    for time in available_times[11:12]:
        timestamp = f"{timestr}_{time}_LST.tif"
        print(f"Processing hourly stats at {time} for {bldg_tree}")

        # Find raster files matching this timestamp
        hour_files = find_raster_files(root_directory, timestamp)
        if not hour_files:
            print(f"No hourly files found for {time} on {timestr}")
            continue

        # Process each tile individually
        for hour_file in hour_files:
            print(f"[DEBUG] Processing file: {hour_file}")

            # Open the tile and retrieve raster metadata
            with rasterio.open(hour_file) as src:
                raster_data = src.read(1)
                raster_meta = src.meta
                nodata = src.nodata

                # Replace nodata values with NaN for processing
                raster_data = np.where(raster_data == nodata, np.nan, raster_data)

                # Invert the raster for shaded pixels (exposed = 0, shaded = 1)
                inverted_raster = np.where(
                    (raster_data < 1) & (raster_data >= 0), 1,  # Shaded pixels
                    np.where(raster_data == 1, 0, np.nan)       # Exposed pixels and NaN
                )

                # Compute zonal stats (sum and count) for the current tile
                stats = zonal_stats(
                    polygons, inverted_raster,
                    affine=src.transform,
                    stats=["sum", "count"],
                    nodata=np.nan,
                )
                # Update polygons with calculated percentages
                shade_column = f"{timestr}_{bldg_tree}_shade_percent_at_{time}"

                # Create a temporary series for the current shade_column
                temp_series = pd.Series(
                    [(stat["sum"] / stat["count"]) * 100 if stat["count"] > 0 else np.nan for stat in stats],
                    index=polygons.index,
                )

                # Use combine_first to only update NaN values in the shade column
                polygons[shade_column] = temp_series.combine_first(polygons[shade_column])

    return polygons

# def process_hourly_stats(polygons, root_directory, timestr, bldg_tree):
#     """
#     Calculate hourly statistics for each tile separately and update polygons.

#     Parameters:
#         polygons (GeoDataFrame): GeoDataFrame containing sidewalk polygons.
#         root_directory (str): Directory containing hourly raster tiles.
#         timestr (str): Date in YYYYMMDD format.
#         bldg_tree (str): Type of shade ('building' or 'tree').

#     Returns:
#         GeoDataFrame: Updated polygons with hourly statistics.
#     """
#     available_times = find_available_times(root_directory, timestr)
#     print(f"Available times for {bldg_tree} on {timestr}: {available_times}")

#     for time in available_times:
#         timestamp = f"{timestr}_{time}_LST.tif"
#         raster_files = find_raster_files(root_directory, timestamp)

#         if not raster_files:
#             print(f"No hourly raster files found for {bldg_tree} at {time} on {timestr}")
#             continue

#         for tile in raster_files:
#             with rasterio.open(tile) as src:
#                 raster_data = src.read(1)
#                 raster_meta = src.meta
#                 affine = src.transform

#                 # Compute zonal stats for the tile and update polygons
#                 stats = zonal_stats(polygons, raster_data, affine=affine, stats=['sum', 'count'], nodata=src.nodata)

#                 # Calculate percentage shade coverage
#                 shade_percentages = []
#                 for stat in stats:
#                     if stat['count'] > 0:
#                         shade_percentage = (stat['sum'] / stat['count']) * 100
#                     else:
#                         shade_percentage = np.nan
#                     shade_percentages.append(shade_percentage)

#                 column_name = f"{timestr}_{bldg_tree}_shade_percent_at_{time}"
#                 if column_name not in polygons:
#                     polygons[column_name] = np.nan
#                 polygons[column_name].update(shade_percentages)

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