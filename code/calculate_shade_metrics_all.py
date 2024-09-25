import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterstats import zonal_stats
import numpy as np
import os
import argparse
from shapely.geometry import mapping, box
import tqdm

import datetime as dt

### THIS CODE CAN BE USED FOR PRETTY MUCH ANY ANALYSIS AT ANY SCALE. COULD ALSO DO THE NEIGHBOURHOOD ANALYSIS HERE (just replace verhardingen with neighbourhoods, but don't forget to mask out the buildings.)


def main(osmid, date):
    print(f'working on OSMID:{osmid}; Date: {date}')
    timestr = date.strftime("%Y%m%d")
    timestr_extension = timestr + '.tif'

    # Load polygon dataset
    polygons = gpd.read_file('../data/raw_data/AMS/Q1_20230126_ingekort.gpkg')

    polygons = process_shade_metrics(osmid, timestr, timestr_extension, polygons)

    # Save or further process the updated polygons if needed
    polygons.to_file(f'../results/output/{osmid}/{osmid}_verhardingen_with_stats_{timestr}.gpkg', driver='GPKG')

def process_shade_metrics(osmid,timestr,timestr_extension,poly):

    for bldg_tree in ['building', 'tree']:

        root_directory = f"../results/output/{osmid}/{bldg_tree}_shade/"
        raster_files = find_raster_files(root_directory, timestr_extension)

        print(f'Found {len(raster_files)} {bldg_tree} - shade files to process, trying merge next...')
        
        mosaic, out_trans, out_meta, out_bounds, out_nodata = merge_rasters_with_mask(raster_files)

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
        
        # Hourly statistics
        for time in ['0900', '0930', '1000', '1030', '1100', '1130', '1200', '1230', '1300', '1330', '1400', '1430', '1500', '1530', '1600', '1630', '1700', '1730', '1800', '1830', '1900']:
            timestamp = timestr + '_' + time + '_LST.tif'

            print('Done with daily stats, working on hourly stats now:')
            hour_files = find_raster_files(root_directory, timestamp)

            print(f'Found {len(hour_files)} {bldg_tree} - shade files at {time} to process, trying merge next...')

            mosaic, out_trans, out_meta, out_bounds, out_nodata = merge_rasters_with_mask(hour_files)

            # Ensure polygons are in the same CRS as the raster
            raster_crs = out_meta['crs']
            if poly.crs != raster_crs:
                poly = poly.to_crs(raster_crs)

            print('Merged hourly files')
            # Update metadata for the virtual raster
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans
            })
            
            poly = calculate_percentage_covered(mosaic, out_meta, out_bounds, out_nodata, poly, time, bldg_tree)
            
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

    # Print the shape of the mosaic for debugging
    print(f"Mosaic shape after merge: {mosaic.shape}")

    # Handle potential extra dimension due to overlap
    if mosaic.ndim == 4:
        # Example: Take the mean across the overlapping layers
        mosaic = mosaic.mean(axis=0)
        print(f"Mosaic shape after averaging overlap: {mosaic.shape}")

    # Get the bounds from the mosaic
    mosaic_bounds = rasterio.coords.BoundingBox(
        left=out_trans.c, bottom=out_trans.f + out_trans.e * mosaic.shape[1], 
        right=out_trans.c + out_trans.a * mosaic.shape[2], top=out_trans.f)

    # Extract nodata value from the first raster
    nodata = src_files_to_mosaic[0].nodata

    # Update the metadata with the merged data properties
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "count": mosaic.shape[0],  # Update to match the number of layers in the mosaic,
        "nodata": nodata,  # Include nodata value in metadata
    })
    
    for src in src_files_to_mosaic:
        src.close()
    
    return mosaic, out_trans, out_meta, mosaic_bounds, nodata


def compute_zonal_stats(polygons, raster_data, affine, nodata_value=None):
    stats = zonal_stats(polygons, raster_data, affine=affine, stats=['mean', 'std', 'min', 'max'], nodata=nodata_value)
    return stats

########################
# Working on this 

def calculate_percentage_covered(raster_data, raster_meta, raster_bounds, nodata, polygon_file, time, building_tree):
    # Read the shapefile
    gdf = polygon_file.copy()
    
    # Initialize a dictionary to accumulate results
    coverage_dict = {}
    #coverage_dict = {idx: [] for idx in gdf.index}
    
    # Set all values < 1.0 to 0
    raster_data[raster_data < 1.0] = 0

    # Invert the binary values: 0 becomes 1 and 1 becomes 0
    inverted_raster_data = np.where(raster_data == 0, 1, 0)
    # Print the shape of the inverted raster data for debugging
    print(f"Inverted raster data shape before squeeze: {inverted_raster_data.shape}")

    # Print the shape of the inverted raster data for debugging
    print(f"Inverted raster data shape after squeeze: {inverted_raster_data.shape}")
    
    # Ensure the shape is compatible
    if inverted_raster_data.ndim == 4:
        # Remove the extra dimension if present
        inverted_raster_data = inverted_raster_data.squeeze()

    # Convert the inverted raster data to 3D array if it's a single layer
    if inverted_raster_data.ndim == 2:
        inverted_raster_data = inverted_raster_data[np.newaxis, ...]

    
    # Update metadata if necessary
    raster_meta.update({
        "count": inverted_raster_data.shape[0],  # Update to match the number of layers in the data
        "dtype": inverted_raster_data.dtype.name  # Ensure dtype matches the numpy array
    })

    # Print metadata for debugging
    print(f"Raster metadata: {raster_meta}")

    # Create an in-memory file for the modified raster data
    with MemoryFile() as memfile:
        with memfile.open(**raster_meta) as mem:
            mem.write(inverted_raster_data)

            # Loop over each polygon and calculate the percentage of coverage
            for idx, row in tqdm.tqdm(gdf.iterrows(), total=gdf.shape[0]):
                polygon = row['geometry']
                
                # Check if the polygon is within the raster bounds
                if polygon.intersects(box(*raster_bounds)):
                    # Mask the raster with the current polygon using the in-memory file
                    out_image, out_transform = mask(mem, [mapping(polygon)], crop=True)
                    # Flatten the array and remove masked values
                    masked_data = out_image[0].flatten()
                    masked_data = masked_data[masked_data != nodata]
                    # Calculate the percentage of 1s
                    total_pixels = len(masked_data)
                    if total_pixels > 0:
                        percentage_ones = np.sum(masked_data == 1) / total_pixels * 100
                    else:
                        percentage_ones = 0
                    # coverage_dict[idx].append(percentage_ones)
                    coverage_dict[idx] = percentage_ones
    # Combine the results from overlapping rasters
    for idx, percentage in coverage_dict.items():  # Changed to iterate over single values
        gdf.at[idx, f'{building_tree}_shade_percent_at_{time}'] = percentage  # Changed to assign single value
    # # Combine the results from overlapping rasters
    # for idx in coverage_dict:
    #     gdf.at[idx, f'shade_percent_at_{time}'] = coverage_dict[idx]
    return gdf

#############

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
