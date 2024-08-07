import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterstats import zonal_stats
import numpy as np
import os
import argparse

import datetime as dt

### THIS CODE CAN BE USED FOR PRETTY MUCH ANY ANALYSIS AT ANY SCALE. COULD ALSO DO THE NEIGHBOURHOOD ANALYSIS HERE (just replace verhardingen with neighbourhoods)

def main(osmid, date):
    print(f'working on OSMID:{osmid}; Date: {date}')
    timestr = date.strftime("%Y%m%d")
    timestr_extension = timestr + '.tif'


    # Load polygon dataset
    polygons = gpd.read_file('../data/raw_data/AMS/Q1_20230126_ingekort.gpkg')

    polygons = process_shade_metrics(osmid, timestr, timestr_extension, polygons)

    # Save or further process the updated polygons if needed
    polygons.to_file(f'../results/output/{osmid}/verhardingen_with_stats_{timestr}.gpkg', driver='GPKG')

def process_shade_metrics(osmid,timestr,timestr_extension,poly):

    for bldg_tree in ['building', 'tree']:

        root_directory = f"../results/output/{osmid}/{bldg_tree}_shade/"
        raster_files = find_raster_files(root_directory, timestr_extension)

        print(f'Found {len(raster_files)} {bldg_tree} - shade files to process, trying merge next...')
        
        mosaic, out_trans, out_meta = merge_rasters_with_mask(raster_files)

        # Ensure polygons are in the same CRS as the raster
        raster_crs = out_meta['crs']
        if poly.crs != raster_crs:
            poly = poly.to_crs(raster_crs)

        # Update metadata for the virtual raster
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })

        print('Merge successful; Server is sweating a little. Stats are next')

        nodata_value = out_meta.get('nodata', None)

        # Assuming 'mosaic' is your merged raster data
        stats = compute_zonal_stats(poly, mosaic[0], affine=out_trans, nodata_value=nodata_value)

        print('Stats computed, adding to polygons')

        # Add statistics to polygon dataset
        for stat_type in ['mean', 'std', 'min', 'max']:
            poly[f'{timestr}_{bldg_tree}_{stat_type}'] = [s[stat_type] for s in stats]

        print('Done!')
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
    
    # Merge data
    mosaic, out_trans = merge(src_files_to_mosaic)
    
    # Update the metadata with the merged data properties
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "count": mosaic.shape[0]  # Update to match the number of layers in the mosaic
    })
    
    for src in src_files_to_mosaic:
        src.close()
    
    return mosaic, out_trans, out_meta

def compute_zonal_stats(polygons, raster_data, affine, nodata_value=None):
    stats = zonal_stats(polygons, raster_data, affine=affine, stats=['mean', 'std', 'min', 'max'], nodata=nodata_value)
    return stats


if __name__ == "__main__":

    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process a OSM area.")
    
    # Add the argument
    parser.add_argument('number', type=int, help='OSMID to be processed')

    # Add the date argument
    parser.add_argument('date', type=str, help='Date to be processed in YYYY-MM-DD format')

    # Parse the arguments
    args = parser.parse_args()

    # Access the number argument
    osmid = args.number

    date_input = args.date

    # Convert the date string to a datetime object
    date = dt.datetime.strptime(date_input, "%Y-%m-%d")

    main(osmid,date)
