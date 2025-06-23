import os
import re
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import mapping
import pandas as pd
from rasterstats import zonal_stats
import numpy as np
import argparse
import datetime as dt

def load_sidewalk_data(osmid):
    """
    Load and filter osm2streets lane data for sidewalks, fixing invalid geometries if needed.
    """
    input_dir = f"../data/raw_data/osm2streets/{osmid}/processed/"
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Lane data folder does not exist: {input_dir}")

    files = [f for f in os.listdir(input_dir) if f.endswith('lanes.geojson')]
    if not files:
        raise FileNotFoundError(f"No lane data found in: {input_dir}")

    filepath = os.path.join(input_dir, files[0])
    lanes = gpd.read_file(filepath, engine="pyogrio")  # Specify pyogrio as the engine

    # Filter for sidewalks and footpaths
    valid_types = ['Sidewalk', 'Footway']
    sidewalks = lanes[lanes['type'].isin(valid_types)]

    print(f"Loaded {len(sidewalks)} sidewalk features from {filepath}")

    # Validate and fix geometries
    invalid_count = sidewalks[~sidewalks.geometry.is_valid].shape[0]
    if invalid_count > 0:
        print(f"[WARNING] Found {invalid_count} invalid geometries. Attempting to fix...")
        sidewalks["geometry"] = sidewalks.geometry.buffer(0)
        fixed_count = sidewalks[~sidewalks.geometry.is_valid].shape[0]
        if fixed_count > 0:
            print(f"[ERROR] {fixed_count} geometries could not be fixed. Please inspect the data.")
        else:
            print(f"[INFO] All invalid geometries were successfully fixed.")

    return sidewalks

def main(osmid, dates, neighborhood_path, group_col):
    print(f'Processing OSMID {osmid} for dates: {dates}')
    
    sidewalks = load_sidewalk_data(osmid)
    
    neighborhoods_orig = gpd.read_file(neighborhood_path)

    for date in dates:
        timestr = date.strftime("%Y%m%d")
        print(f"\nDate: {timestr}")
        
        raster_root = f"../results/output/{osmid}/tree_shade/"
        available_times = find_available_times(raster_root, timestr)

        if not available_times:
            print(f"No rasters found for {timestr}")
            continue

        # Prepare one cumulative neighborhoods dataframe
        neighborhoods_all = neighborhoods_orig.copy()
        neighborhoods_all = neighborhoods_all[~neighborhoods_all.geometry.is_empty & neighborhoods_all.geometry.notnull()]
        
        for time in available_times:
            timestamp = f"{timestr}_{time}_LST.tif"
            hour_files = find_raster_files(raster_root, timestamp)
            if not hour_files:
                continue

            mosaic, out_trans, out_meta, out_bounds, out_nodata = merge_rasters_with_mask(hour_files)
            raster_crs = out_meta['crs']
            sidewalks = sidewalks.to_crs(raster_crs)
            neighborhoods_all = neighborhoods_all.to_crs(raster_crs)

            neighborhoods_all = process_boston_zonal_stats(neighborhoods_all, sidewalks, mosaic[0], out_trans, timestr, time, group_col)

        # Export after all timesteps processed
        output_path = f"../results/output/{osmid}_neighborhood_sun_{timestr}.csv"
        neighborhoods_all.to_csv(output_path, index=False)
        print(f"Saved combined output to {output_path}")

def process_boston_zonal_stats(neighborhoods, sidewalks, mosaic_array, transform, timestr, time, group_col):
    # Intersect sidewalks and neighborhoods
    sidewalks_neigh = gpd.overlay(neighborhoods, sidewalks, how='intersection')
    sidewalks_neigh = sidewalks_neigh[~sidewalks_neigh.geometry.is_empty & sidewalks_neigh.geometry.notnull()]
    
    # Compute sunlit zonal stats for sidewalk-neighborhood intersections
    sidewalk_stats = zonal_stats(sidewalks_neigh, mosaic_array, affine=transform, stats=['mean', 'sum'], nodata=None)
    sidewalks_neigh.loc[:, f'sidewalk_mean_sun_{time}'] = [s['mean'] for s in sidewalk_stats]
    sidewalks_neigh.loc[:, f'sidewalk_sum_sun_{time}'] = [s['sum'] for s in sidewalk_stats]

    # --- New: compute shade metrics ---
    # Average shade = 1 - average sunlit
    sidewalks_neigh.loc[:, f'sidewalk_avg_shade_{time}'] = 1 - sidewalks_neigh[f'sidewalk_mean_sun_{time}']

    # Area in m² (requires projected CRS)
    sidewalks_neigh["area_m2"] = sidewalks_neigh.geometry.area

    # Cumulative shade = average shade × area
    sidewalks_neigh.loc[:, f'sidewalk_cum_shade_{time}'] = (
        sidewalks_neigh[f'sidewalk_avg_shade_{time}'] * sidewalks_neigh["area_m2"]
    )

    # Group by neighborhood (group_col) to get mean and total shade values
    sidewalk_grouped = sidewalks_neigh.groupby(group_col)[
        [f'sidewalk_avg_shade_{time}', f'sidewalk_cum_shade_{time}']
    ].agg({
        f'sidewalk_avg_shade_{time}': 'mean',
        f'sidewalk_cum_shade_{time}': 'sum'
    }).reset_index()

    # Also compute zonal stats over full neighborhoods (optional)
    neighborhood_stats = zonal_stats(neighborhoods, mosaic_array, affine=transform, stats=['mean', 'sum'], nodata=None)
    neighborhoods.loc[:, f'total_mean_sun_{time}'] = [s['mean'] for s in neighborhood_stats]
    neighborhoods.loc[:, f'total_sum_sun_{time}'] = [s['sum'] for s in neighborhood_stats]

    # Merge shade metrics into the neighborhoods GeoDataFrame
    neighborhoods = neighborhoods.merge(sidewalk_grouped, on=group_col, how="left")

    return neighborhoods

def find_raster_files(root_dir, file_extension='xx'):
    raster_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(file_extension):
                raster_files.append(os.path.join(root, file))
    return raster_files

def merge_rasters_with_mask(raster_files):
    src_files = [rasterio.open(f) for f in raster_files]
    mosaic, out_trans = merge(src_files)
    
    if mosaic.ndim == 4:
        mosaic = mosaic.mean(axis=0)

    nodata = src_files[0].nodata
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "count": 1,
        "nodata": nodata
    })

    for src in src_files:
        src.close()
        
    bounds = rasterio.coords.BoundingBox(
        left=out_trans.c, bottom=out_trans.f + out_trans.e * mosaic.shape[1], 
        right=out_trans.c + out_trans.a * mosaic.shape[2], top=out_trans.f)

    return mosaic, out_trans, out_meta, bounds, nodata

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a OSM area and export zonal stats.")
    parser.add_argument('number', type=int, help='OSMID to be processed')
    parser.add_argument('dates', type=str, nargs='+', help='Dates in YYYY-MM-DD format')
    parser.add_argument('--neighborhood_path', type=str, required=True, help='Path to the neighborhood GeoJSON or GPKG file')
    parser.add_argument('--group_col', type=str, required=True, help='Column name used to group by neighborhood')
    args = parser.parse_args()

    osmid = args.number
    dates = [dt.datetime.strptime(d, "%Y-%m-%d") for d in args.dates]
    
    main(osmid, dates, args.neighborhood_path, args.group_col)
