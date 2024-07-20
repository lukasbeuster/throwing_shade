import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterstats import zonal_stats
import numpy as np
import os
import argparse

import datetime as dt

### THIS CODE CAN BE USED FOR PRETTY MUCH ANY ANALYSIS AT ANY SCALE. COULD ALSO DO THE NEIGHBOURHOOD ANALYSIS HERE (just replace verhardingen with neighbourhoods)

def main(osmid, date, bldg_tree):
    print(f'working on OSMID:{osmid}; Date: {date}')
    timestr = date.strftime("%Y%m%d")
    timestr_extension = timestr + '.tif'


    # Load polygon dataset
    polygons = gpd.read_file('../data/raw_data/AMS/Q1_20230126_ingekort.gpkg')

    root_directory = f"../results/output/{osmid}/{bldg_tree}_shade/"
    raster_files = find_raster_files(root_directory, timestr_extension)

    print(f'Found {len(raster_files)} to process, trying merge next...')
    
    mosaic, out_trans, out_meta = merge_rasters_with_mask(raster_files)

    # Ensure polygons are in the same CRS as the raster
    raster_crs = out_meta['crs']
    if polygons.crs != raster_crs:
        polygons = polygons.to_crs(raster_crs)

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
    stats = compute_zonal_stats(polygons, mosaic[0], affine=out_trans, nodata_value=nodata_value)

    # Add statistics to polygon dataset
    for stat_type in ['mean', 'std', 'min', 'max']:
        polygons[f'{timestr}_{bldg_tree}_{stat_type}'] = [s[stat_type] for s in stats]


    polygons.to_file('../results/output/testing_poly.gpkg', driver='GPKG')

    print('Done!')



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

# def merge_rasters_with_mask(raster_files):
#     src_files_to_mosaic = []
#     for raster in raster_files:
#         with rasterio.open(raster) as src:
#             data = src.read(1)
#             mask = data != src.nodata
#             src_files_to_mosaic.append({'data': data, 'mask': mask, 'meta': src.meta})
    
#     # Merge data and masks
#     merged_data, out_trans = merge([{'data': src['data'], 'transform': src['meta']['transform']} for src in src_files_to_mosaic])
#     merged_mask, _ = merge([{'data': src['mask'], 'transform': src['meta']['transform']} for src in src_files_to_mosaic])
    
#     # Apply mask to merged data
#     merged_data = np.ma.masked_array(merged_data, mask=~merged_mask)
    
#     return merged_data, out_trans, src_files_to_mosaic[0]['meta']

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

    # Add the buildings_or_trees argument
    parser.add_argument('bldgtree', type=str, help='Are we processing building or building and tree shade?')

    # Parse the arguments
    args = parser.parse_args()

    # Access the number argument
    osmid = args.number

    date_input = args.date

    bldg_tree = args.bldgtree

    # Convert the date string to a datetime object
    date = dt.datetime.strptime(date_input, "%Y-%m-%d")

    main(osmid,date, bldg_tree)
