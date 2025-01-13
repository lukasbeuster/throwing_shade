import os
import re
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterstats import zonal_stats
import numpy as np
import argparse
from shapely.geometry import mapping, box
import tqdm
import datetime as dt


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
        timestr_extension = timestr + '.tif'
        
        polygons = process_shade_metrics(osmid, timestr, timestr_extension, polygons)

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
    
    # sidewalks = lanes[lanes['type'] == 'Sidewalk']
    print(f"Loaded {len(sidewalks)} sidewalk features from {filepath}")
    
    return sidewalks


def process_shade_metrics(osmid, timestr, timestr_extension, polygons):
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
    for bldg_tree in ['building', 'tree']:
        root_directory = f"../results/output/{osmid}/{bldg_tree}_shade/"
        raster_files = find_raster_files(root_directory, timestr_extension)

        print(f'Found {len(raster_files)} {bldg_tree} shade files to process...')
        
        if not raster_files:
            print(f"No raster files found for {bldg_tree} on {timestr}")
            continue
        
        mosaic, out_trans, out_meta, out_bounds, out_nodata = merge_rasters_with_mask(raster_files)

        # Ensure polygons are in the same CRS as the raster
        raster_crs = out_meta['crs']
        if polygons.crs != raster_crs:
            polygons = polygons.to_crs(raster_crs)

        stats = compute_zonal_stats(polygons, mosaic[0], affine=out_trans, nodata_value=out_nodata)
        print('Stats computed, adding to polygons')

        for stat_type in ['mean', 'std', 'min', 'max']:
            polygons[f'{timestr}_{bldg_tree}_{stat_type}'] = [s[stat_type] for s in stats]
        
        # Get the list of available times dynamically for hourly statistics
        available_times = find_available_times(root_directory, timestr)

        # Process hourly statistics
        for time in available_times:
            timestamp = timestr + '_' + time + '_LST.tif'

            print(f'Processing hourly stats at {time} for {bldg_tree}')
            hour_files = find_raster_files(root_directory, timestamp)

            if len(hour_files) > 0:
                mosaic, out_trans, out_meta, out_bounds, out_nodata = merge_rasters_with_mask(hour_files)
                polygons = calculate_percentage_covered(mosaic, out_meta, out_bounds, out_nodata, polygons, time, bldg_tree, timestr)

    return polygons


def find_raster_files(root_dir, file_extension):
    """
    Find raster files with a specific extension in a directory.

    Parameters:
        root_dir (str): Directory to search.
        file_extension (str): File extension to match.

    Returns:
        list: List of file paths matching the extension.
    """
    raster_files = []
    for root, _, files in os.walk(root_dir):
        raster_files.extend(os.path.join(root, file) for file in files if file.endswith(file_extension))
    return raster_files

def find_available_times(root_dir, timestr):
    """
    This function scans the directory and extracts the available times from filenames.
    Expected file format: {timestr}_{time}_LST.tif, where time is in HHMM format.
    """
    time_pattern = re.compile(rf".*{timestr}_(\d{{4}})_LST\.tif")
    available_times = set()

    # Walk through the directory and extract times from filenames
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # print(f"Processing file: {file}")  # Debugging line to see the filenames
            match = time_pattern.match(file)
            if match:
                # print(f"Added {file} to the available times")
                available_times.add(match.group(1))
                # print(f"Availabletimes for {timestr}: {available_times}")
    
    # If no times are found, print a message
    if not available_times:
        print(f"No matching times found for {timestr}.")

    # Sort and return the list of available times
    return sorted(available_times)

def merge_rasters_with_mask(raster_files):
    """
    Merge raster files into a mosaic.

    Parameters:
        raster_files (list): List of raster file paths.

    Returns:
        tuple: Mosaic array, affine transform, metadata, bounding box, nodata value.
    """
    src_files_to_mosaic = [rasterio.open(raster) for raster in raster_files]
    mosaic, out_trans = merge(src_files_to_mosaic)

    nodata = src_files_to_mosaic[0].nodata
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "count": mosaic.shape[0],
        "nodata": nodata,
    })

    for src in src_files_to_mosaic:
        src.close()

    return mosaic, out_trans, out_meta, src_files_to_mosaic[0].bounds, nodata


def compute_zonal_stats(polygons, raster_data, affine, nodata_value=None):
    """
    Compute zonal statistics for polygons using raster data.

    Parameters:
        polygons (GeoDataFrame): Polygons to analyze.
        raster_data (ndarray): Raster data array.
        affine (Affine): Affine transform of the raster.
        nodata_value (int, optional): NoData value in the raster.

    Returns:
        list: List of dictionaries containing zonal statistics.
    """
    return zonal_stats(polygons, raster_data, affine=affine, stats=['mean', 'std', 'min', 'max'], nodata=nodata_value)

def calculate_percentage_covered(raster_data, raster_meta, raster_bounds, nodata, polygon_file, time, building_tree, timestr):
    gdf = polygon_file.copy()
    coverage_dict = {}
    raster_data[raster_data < 1.0] = 0
    inverted_raster_data = np.where(raster_data == 0, 1, 0)

    if inverted_raster_data.ndim == 4:
        inverted_raster_data = inverted_raster_data.squeeze()

    if inverted_raster_data.ndim == 2:
        inverted_raster_data = inverted_raster_data[np.newaxis, ...]

    raster_meta.update({
        "count": inverted_raster_data.shape[0],
        "dtype": inverted_raster_data.dtype.name
    })

    with MemoryFile() as memfile:
        with memfile.open(**raster_meta) as mem:
            mem.write(inverted_raster_data)
            for idx, row in tqdm.tqdm(gdf.iterrows(), total=gdf.shape[0]):
                polygon = row['geometry']
                if polygon.intersects(box(*raster_bounds)):
                    out_image, out_transform = mask(mem, [mapping(polygon)], crop=True)
                    masked_data = out_image[0].flatten()
                    masked_data = masked_data[masked_data != nodata]
                    total_pixels = len(masked_data)
                    percentage_ones = np.sum(masked_data == 1) / total_pixels * 100 if total_pixels > 0 else 0
                    coverage_dict[idx] = percentage_ones

    for idx, percentage in coverage_dict.items():
        gdf.at[idx, f'{timestr}_{building_tree}_shade_percent_at_{time}'] = percentage

    return gdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a OSM area for shade metrics.")
    parser.add_argument('osmid', type=int, help='OSMID to be processed')
    parser.add_argument('dates', type=str, nargs='+', help='Dates in YYYY-MM-DD format (multiple dates allowed)')
    args = parser.parse_args()

    osmid = args.osmid
    dates_input = args.dates
    dates = [dt.datetime.strptime(date, "%Y-%m-%d") for date in dates_input]

    main(osmid, dates)