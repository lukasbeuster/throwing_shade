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

def main(osmid, dates):
    print(f'Processing OSMID {osmid} for dates: {dates}')
    
    sidewalks_path = "../data/raw_data/boston/sidewalk_inventory.gpkg"
    neighborhood_path = "../data/raw_data/boston/2020-census-tracts-in-boston.json"
    
    sidewalks_orig = gpd.read_file(sidewalks_path)
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
            sidewalks = sidewalks_orig.to_crs(raster_crs)
            neighborhoods_all = neighborhoods_all.to_crs(raster_crs)

            neighborhoods_all = process_boston_zonal_stats(neighborhoods_all, sidewalks, mosaic[0], out_trans, timestr, time)

        # Export after all timesteps processed
        output_path = f"../results/output/boston_neighborhood_sun_{timestr}.csv"
        neighborhoods_all.to_csv(output_path, index=False)
        print(f"Saved combined output to {output_path}")

# def process_boston_zonal_stats(neighborhoods, sidewalks, mosaic_array, transform, timestr, time):
#     # Intersect sidewalks and neighborhoods
#     sidewalks_neigh = gpd.overlay(neighborhoods, sidewalks, how='intersection')

#     # Remove any empty or missing geometries
#     sidewalks_neigh = sidewalks_neigh[~sidewalks_neigh.geometry.is_empty & sidewalks_neigh.geometry.notnull()]
#     neighborhoods = neighborhoods[~neighborhoods.geometry.is_empty & neighborhoods.geometry.notnull()]

#     # Zonal stats over sidewalk-neighborhood intersections
#     sidewalk_stats = zonal_stats(sidewalks_neigh, mosaic_array, affine=transform, stats=['mean', 'sum'])
#     sidewalks_neigh[[f'sidewalk_mean_{time}', f'sidewalk_sum_{time}']] = pd.DataFrame(sidewalk_stats)

#     # Zonal stats over full neighborhoods
#     neighborhood_stats = zonal_stats(neighborhoods, mosaic_array, affine=transform, stats=['mean', 'sum'])
#     neighborhoods[[f'total_mean_{time}', f'total_sum_{time}']] = pd.DataFrame(neighborhood_stats)
#     # neighborhoods.loc[:, f'total_mean_{time}'] = [stat['mean'] for stat in neighborhood_stats]
#     # neighborhoods.loc[:, f'total_sum_{time}'] = [stat['sum'] for stat in neighborhood_stats]

#     # Aggregate sidewalk stats per neighborhood
#     sidewalk_grouped = sidewalks_neigh.groupby("geoid20")[[f'sidewalk_mean_{time}', f'sidewalk_sum_{time}']].mean().reset_index()
#     final = neighborhoods.merge(sidewalk_grouped, on="geoid20", how="left")

#     # Export
#     output_path = f"../results/output/boston_neighborhood_sun_{timestr}_{time}.csv"
#     final.to_csv(output_path, index=False)
#     print(f"Saved output to {output_path}")

## The last version I ran for Xinyue to get the results.
def process_boston_zonal_stats(neighborhoods, sidewalks, mosaic_array, transform, timestr, time):
    sidewalks_neigh = gpd.overlay(neighborhoods, sidewalks, how='intersection')
    sidewalks_neigh = sidewalks_neigh[~sidewalks_neigh.geometry.is_empty & sidewalks_neigh.geometry.notnull()]
    
    sidewalk_stats = zonal_stats(sidewalks_neigh, mosaic_array, affine=transform, stats=['mean', 'sum'])
    sidewalks_neigh.loc[:, f'sidewalk_mean_{time}'] = [s['mean'] for s in sidewalk_stats]
    sidewalks_neigh.loc[:, f'sidewalk_sum_{time}'] = [s['sum'] for s in sidewalk_stats]

    sidewalk_grouped = sidewalks_neigh.groupby("geoid20")[
        [f'sidewalk_mean_{time}', f'sidewalk_sum_{time}']
    ].mean().reset_index()

    neighborhood_stats = zonal_stats(neighborhoods, mosaic_array, affine=transform, stats=['mean', 'sum'])
    neighborhoods.loc[:, f'total_mean_{time}'] = [s['mean'] for s in neighborhood_stats]
    neighborhoods.loc[:, f'total_sum_{time}'] = [s['sum'] for s in neighborhood_stats]

    neighborhoods = neighborhoods.merge(sidewalk_grouped, on="geoid20", how="left")

    return neighborhoods

# def process_boston_zonal_stats(neighborhoods, sidewalks, mosaic_array, transform, timestr, time):
#     # Intersect sidewalks and neighborhoods
#     sidewalks_neigh = gpd.overlay(neighborhoods, sidewalks, how='intersection')
#     sidewalks_neigh = sidewalks_neigh[~sidewalks_neigh.geometry.is_empty & sidewalks_neigh.geometry.notnull()]
    
#     # Compute sunlit zonal stats for sidewalk-neighborhood intersections
#     sidewalk_stats = zonal_stats(sidewalks_neigh, mosaic_array, affine=transform, stats=['mean', 'sum'], nodata=None)
#     sidewalks_neigh.loc[:, f'sidewalk_mean_sun_{time}'] = [s['mean'] for s in sidewalk_stats]
#     sidewalks_neigh.loc[:, f'sidewalk_sum_sun_{time}'] = [s['sum'] for s in sidewalk_stats]

#     # --- New: compute shade metrics ---
#     # Average shade = 1 - average sunlit
#     sidewalks_neigh.loc[:, f'sidewalk_avg_shade_{time}'] = 1 - sidewalks_neigh[f'sidewalk_mean_sun_{time}']

#     # Area in m² (requires projected CRS)
#     sidewalks_neigh["area_m2"] = sidewalks_neigh.geometry.area

#     # Cumulative shade = average shade × area
#     sidewalks_neigh.loc[:, f'sidewalk_cum_shade_{time}'] = (
#         sidewalks_neigh[f'sidewalk_avg_shade_{time}'] * sidewalks_neigh["area_m2"]
#     )

#     # Group by neighborhood (geoid20) to get mean and total shade values
#     sidewalk_grouped = sidewalks_neigh.groupby("geoid20")[
#         [f'sidewalk_avg_shade_{time}', f'sidewalk_cum_shade_{time}']
#     ].agg({
#         f'sidewalk_avg_shade_{time}': 'mean',
#         f'sidewalk_cum_shade_{time}': 'sum'
#     }).reset_index()

#     # Also compute zonal stats over full neighborhoods (optional)
#     neighborhood_stats = zonal_stats(neighborhoods, mosaic_array, affine=transform, stats=['mean', 'sum'], nodata=None)
#     neighborhoods.loc[:, f'total_mean_sun_{time}'] = [s['mean'] for s in neighborhood_stats]
#     neighborhoods.loc[:, f'total_sum_sun_{time}'] = [s['sum'] for s in neighborhood_stats]

#     # Merge shade metrics into the neighborhoods GeoDataFrame
#     neighborhoods = neighborhoods.merge(sidewalk_grouped, on="geoid20", how="left")

#     return neighborhoods

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
    args = parser.parse_args()

    osmid = args.number
    dates = [dt.datetime.strptime(d, "%Y-%m-%d") for d in args.dates]
    
    main(osmid, dates)

