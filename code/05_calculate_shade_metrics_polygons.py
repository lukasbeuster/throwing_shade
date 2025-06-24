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
    print(f'Working on OSMID: {osmid}; Dates: {dates}')
    
    # Load polygon dataset
    polygons = gpd.read_file('../data/raw_data/singapore/Footpath_Jul2023/Footpath.shp')

    for date in dates:
        timestr = date.strftime("%Y%m%d")
        print(timestr)
        timestr_extension = timestr + '.tif'
        
        print(f'Processing date: {timestr}')
        polygons = process_shade_metrics(osmid, timestr, timestr_extension, polygons)

    # Save the updated polygons after processing all dates
    polygons.to_file(f'../results/output/{osmid}/{osmid}_verhardingen_with_stats_multiple_dates.gpkg', driver='GPKG')

def process_shade_metrics(osmid, timestr, timestr_extension, poly):
    for bldg_tree in ['building', 'tree']:
        root_directory = f"../results/output/{osmid}/{bldg_tree}_shade/"
        raster_files = find_raster_files(root_directory, timestr_extension)

        print(f'Found {len(raster_files)} {bldg_tree} - shade files to process, trying merge next...')
        
        mosaic, out_trans, out_meta, out_bounds, out_nodata = merge_rasters_with_mask(raster_files)

        # Ensure polygons are in the same CRS as the raster
        raster_crs = out_meta['crs']
        if poly.crs != raster_crs:
            poly = poly.to_crs(raster_crs)

        print('Merge successful; Server is sweating a little. Stats are next')
        
        nodata_value = out_meta.get('nodata', None)
        
        stats = compute_zonal_stats(poly, mosaic[0], affine=out_trans, nodata_value=nodata_value)
        
        print('Stats computed, adding to polygons')

        # Add statistics to polygon dataset for each date
        for stat_type in ['mean', 'std', 'min', 'max']:
            poly[f'{timestr}_{bldg_tree}_{stat_type}'] = [s[stat_type] for s in stats]

        # Get the list of available times dynamically for hourly statistics
        available_times = find_available_times(root_directory, timestr)

        # Process hourly statistics (if needed, you can keep or remove this)
        for time in available_times:
            timestamp = timestr + '_' + time + '_LST.tif'

            print(f'Processing hourly stats at {time} for {bldg_tree}')
            hour_files = find_raster_files(root_directory, timestamp)

            if len(hour_files) > 0:
                mosaic, out_trans, out_meta, out_bounds, out_nodata = merge_rasters_with_mask(hour_files)
                poly = calculate_percentage_covered(mosaic, out_meta, out_bounds, out_nodata, poly, time, bldg_tree, timestr)

        print(f'Done processing for {timestr}')
        
    return poly

def find_raster_files(root_dir, file_extension='xx'):
    raster_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(file_extension):
                raster_files.append(os.path.join(root, file))
    return raster_files

def merge_rasters_with_mask(raster_files):
    src_files_to_mosaic = [rasterio.open(raster) for raster in raster_files]
    mosaic, out_trans = merge(src_files_to_mosaic)
    
    print(f"Mosaic shape after merge: {mosaic.shape}")
    
    if mosaic.ndim == 4:
        mosaic = mosaic.mean(axis=0)
        print(f"Mosaic shape after averaging overlap: {mosaic.shape}")

    mosaic_bounds = rasterio.coords.BoundingBox(
        left=out_trans.c, bottom=out_trans.f + out_trans.e * mosaic.shape[1], 
        right=out_trans.c + out_trans.a * mosaic.shape[2], top=out_trans.f)

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
    
    return mosaic, out_trans, out_meta, mosaic_bounds, nodata

def compute_zonal_stats(polygons, raster_data, affine, nodata_value=None):
    stats = zonal_stats(polygons, raster_data, affine=affine, stats=['mean', 'std', 'min', 'max'], nodata=nodata_value)
    return stats


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
    parser = argparse.ArgumentParser(description="Process a OSM area.")
    parser.add_argument('number', type=int, help='OSMID to be processed')
    parser.add_argument('dates', type=str, nargs='+', help='Dates to be processed in YYYY-MM-DD format (multiple dates allowed)')
    args = parser.parse_args()

    osmid = args.number
    dates_input = args.dates
    dates = [dt.datetime.strptime(date, "%Y-%m-%d") for date in dates_input]

    main(osmid, dates)