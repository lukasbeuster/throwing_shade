import os
import re
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.io import MemoryFile
# from rasterio.mask import mask
from rasterio.features import geometry_mask
from rasterio.features import rasterize
from rasterstats import zonal_stats
import numpy as np
import argparse
from shapely.geometry import mapping, box
import numpy.ma as ma  # For masked arrays
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


def process_shade_metrics(osmid, timestr, timestr_extension, poly):
    """
    Process shade metrics for a given date and update the polygon dataset.

    Parameters:
        osmid (int): OSMID of the area to process.
        timestr (str): Date in YYYYMMDD format.
        timestr_extension (str): Date with extension (e.g., YYYYMMDD.tif).
        poly (GeoDataFrame): GeoDataFrame containing sidewalk polygons.

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
        if poly.crs != raster_crs:
            poly = poly.to_crs(raster_crs)
            # print('Changed polygon crs to raster crs')
        stats = compute_zonal_stats(poly, mosaic[0], affine=out_trans, nodata_value=out_nodata)
        print('Stats computed, adding to polygons')

        # Add daily statistics
        for stat_type in ['mean', 'std', 'min', 'max']:
            poly[f'{timestr}_{bldg_tree}_{stat_type}'] = [s[stat_type] for s in stats]

        debug_log_polygons(poly, 'after daily')

        # Process hourly statistics in a dedicated function
        poly = process_hourly_statistics(poly, root_directory, bldg_tree, timestr)

        debug_log_polygons(poly, 'after hourly')

    return poly

def debug_log_polygons(polygons, step):
    print(f"[DEBUG] {step}:")
    print(f"  Number of polygons: {len(polygons)}")
    print(f"  CRS: {polygons.crs}")
    print(f"  Columns: {polygons.columns}")
    print(polygons.head(3))


def process_hourly_statistics(polygons_hours, root_directory, bldg_tree, timestr):
    """
    Process hourly statistics for a given date and update the polygon dataset.

    Parameters:
        polygons (GeoDataFrame): GeoDataFrame containing sidewalk polygons.
        root_directory (str): Directory containing raster files.
        bldg_tree (str): Type of shade ('building' or 'tree').
        timestr (str): Date in YYYYMMDD format.

    Returns:
        GeoDataFrame: Updated polygons with hourly statistics.
    """
    available_times = find_available_times(root_directory, timestr)
    print(f"Available times for {bldg_tree} on {timestr}: {available_times}")

    for time in available_times:
        timestamp = f"{timestr}_{time}_LST.tif"
        print(f"Processing hourly stats at {time} for {bldg_tree}")

        hour_files = find_raster_files(root_directory, timestamp)
        # print(f"[DEBUG] Raster files found for {time}: {hour_files}")
        if not hour_files:
            print(f"No hourly files found for {time} on {timestr}")
            continue

        mosaic, out_trans, out_meta, out_bounds, out_nodata = merge_rasters_with_mask(hour_files)
        polygons_hours = calculate_percentage_covered_with_zonal_stats(
            mosaic,
            out_meta,
            out_nodata,
            polygons_hours,
            time,
            bldg_tree,
            timestr,
        )

    return polygons_hours


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

    # print(f"[DEBUG] Found raster files for {file_extension}: {raster_files}")

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
    Merge raster files into a mosaic with proper nodata handling.

    Parameters:
        raster_files (list): List of raster file paths.

    Returns:
        tuple: Mosaic array, affine transform, metadata, bounding box, nodata value.
    """
    # Open all raster files
    src_files_to_mosaic = [rasterio.open(raster) for raster in raster_files]
    dtype = src_files_to_mosaic[0].dtypes[0]  # Get the data type of the first raster
    nodata = src_files_to_mosaic[0].nodata  # Get the original nodata value from the raster
    print(nodata)

    # Assign a valid nodata value if not already defined
    if nodata is None:
        if dtype in ['int32', 'int64']:
            nodata = -9999  # Use -9999 for integer rasters
        elif dtype in ['float32', 'float64']:
            nodata = np.nan  # Use NaN for floating-point rasters
        else:
            raise ValueError(f"Unsupported raster data type: {dtype}")

    # Merge raster files and set nodata
    mosaic, out_trans = merge(src_files_to_mosaic, nodata=nodata)
    mosaic = np.where(mosaic == nodata, nodata, mosaic)  # Ensure nodata consistency

    # Calculate full bounds from the mosaic transform
    mosaic_height, mosaic_width = mosaic.shape[1], mosaic.shape[2]
    full_bounds = (
        out_trans.c,
        out_trans.f + mosaic_height * out_trans.e,
        out_trans.c + mosaic_width * out_trans.a,
        out_trans.f,
    )

    # Update metadata
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "count": mosaic.shape[0],
        "dtype": dtype,
        "nodata": nodata,  # Set consistent nodata in metadata
    })

    # Close all open raster files
    for src in src_files_to_mosaic:
        src.close()

    return mosaic, out_trans, out_meta, full_bounds, nodata


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


def calculate_percentage_covered_with_zonal_stats(
    raster_data, raster_meta, nodata, polygon_file, time, building_tree, timestr
):
    """
    Calculate the percentage of each polygon covered by shade using zonal_stats.

    Parameters:
        raster_data (ndarray): Raster data array (values between 0 and 1, with nodata as -9999).
        raster_meta (dict): Metadata of the raster.
        nodata (float): NoData value of the raster (-9999).
        polygon_file (GeoDataFrame): GeoDataFrame containing sidewalk polygons.
        time (str): Time of day (e.g., '0830').
        building_tree (str): Type of shade ('building' or 'tree').
        timestr (str): Date in YYYYMMDD format.

    Returns:
        GeoDataFrame: Updated GeoDataFrame with percentage shade coverage added.
    """
    # Ensure the CRS of the polygons matches the raster
    if polygon_file.crs != raster_meta["crs"]:
        polygon_file = polygon_file.to_crs(raster_meta["crs"])

    # Replace nodata values with NaN and invert the raster
    raster_data = np.where(raster_data == nodata, np.nan, raster_data)
    inverted_raster = np.where(
        (raster_data < 1) & (raster_data >= 0), 1,  # Shaded pixels
        np.where(raster_data == 1, 0, np.nan)       # Exposed pixels and NaN
    )

    # Convert the inverted raster to a format compatible with zonal_stats
    raster_affine = raster_meta["transform"]

    # Calculate zonal stats for sum and count
    stats = zonal_stats(
        polygon_file, inverted_raster[0],  # Use the first band of the raster
        affine=raster_affine,
        stats=["sum", "count"],
        nodata=np.nan,
    )

    for idx, stat in enumerate(stats[:5]):  # Print stats for the first 5 polygons
        print(f"[DEBUG] Polygon {idx}: sum={stat['sum']}, count={stat['count']}")

    # Extract shade percentage for each polygon
    shade_percentages = []
    for stat in stats:
        shaded_sum = stat["sum"]  # Sum of shaded pixel values (number of shaded pixels)
        pixel_count = stat["count"]  # Total number of valid pixels

        # Calculate percentage of shade
        if pixel_count > 0:
            shade_percentage = (shaded_sum / pixel_count) * 100
        else:
            shade_percentage = np.nan  # No valid pixels
        shade_percentages.append(shade_percentage)

    nan_count = sum(1 for p in shade_percentages if np.isnan(p))
    print(f"[DEBUG] Polygons with NaN shade percentage: {nan_count}")

    # Add the shade percentages as a new column to the GeoDataFrame
    shade_column = f"{timestr}_{building_tree}_shade_percent_at_{time}"
    polygon_file[shade_column] = shade_percentages

    return polygon_file

# def calculate_percentage_covered(raster_data, raster_meta, raster_bounds, nodata, polygon_file, time, building_tree, timestr):
#     if polygon_file.crs != raster_meta["crs"]:
#         print("[DEBUG] CRS mismatch detected. Transforming polygon CRS.")
#         polygon_file = polygon_file.to_crs(raster_meta["crs"])

#     gdf = polygon_file.copy().reset_index(drop=True)

#     # Replace nodata values with NaN
#     raster_data = np.where(raster_data == nodata, np.nan, raster_data)

#     # Invert raster
#     inverted_raster = np.full_like(raster_data, np.nan)
#     inverted_raster[(raster_data >= 0) & (raster_data < 1)] = 1
#     inverted_raster[raster_data == 1] = 0

#     print(f"[DEBUG] Inverted raster: min={np.nanmin(inverted_raster)}, max={np.nanmax(inverted_raster)}")

#     coverage_dict = {}
#     transform = raster_meta["transform"]
#     pixel_width = transform.a
#     pixel_height = -transform.e

#     for idx, row in tqdm.tqdm(gdf.iterrows(), total=len(gdf)):
#         polygon = row["geometry"]

#         # Skip polygons that do not intersect raster bounds
#         if not polygon.intersects(box(*raster_bounds)):
#             print(f"[DEBUG] Polygon {idx} does not intersect raster bounds.")
#             coverage_dict[idx] = 0
#             continue

#         # Calculate raster indices for polygon bounding box
#         minx, miny, maxx, maxy = polygon.bounds
#         row_start = max(0, int((transform.f - maxy) / pixel_height))
#         row_stop = min(raster_data.shape[1], int((transform.f - miny) / pixel_height))
#         col_start = max(0, int((minx - transform.c) / pixel_width))
#         col_stop = min(raster_data.shape[2], int((maxx - transform.c) / pixel_width))

#         print(f"[DEBUG] Polygon bounds: {polygon.bounds}")
#         print(f"[DEBUG] Calculated raster indices: row_start={row_start}, row_stop={row_stop}, col_start={col_start}, col_stop={col_stop}")

#         # Validate raster indices
#         if row_start >= row_stop or col_start >= col_stop:
#             print(f"[DEBUG] Invalid raster indices for polygon {idx}. Skipping.")
#             coverage_dict[idx] = 0
#             continue

#         # Clip the raster to the polygon's bounding box
#         clipped_raster = inverted_raster[0, row_start:row_stop, col_start:col_stop]

#         # Create a binary mask for the polygon
#         polygon_mask = geometry_mask(
#             [polygon],
#             transform=transform,
#             invert=True,
#             out_shape=clipped_raster.shape,
#         )

#         # Apply mask to raster
#         valid_pixels = clipped_raster[polygon_mask]
#         valid_pixels = valid_pixels[~np.isnan(valid_pixels)]

#         print(f"[DEBUG] Masked valid pixels: {len(valid_pixels)}")

#         # Skip polygons entirely within nodata regions
#         if len(valid_pixels) == 0:
#             coverage_dict[idx] = 0
#             continue

#         # Calculate the percentage of shaded pixels
#         shaded_pixels = np.sum(valid_pixels == 1)
#         coverage_dict[idx] = (shaded_pixels / len(valid_pixels)) * 100

#     # Add results to GeoDataFrame
#     shade_column = f"{timestr}_{building_tree}_shade_percent_at_{time}"
#     gdf[shade_column] = gdf.index.map(coverage_dict)

#     return gdf

# def calculate_percentage_covered(raster_data, raster_meta, raster_bounds, nodata, polygon_file, time, building_tree, timestr):
#     """
#     Calculate the percentage of each polygon covered by shade, excluding nodata regions.

#     Parameters:
#         raster_data (ndarray): Raster data array (values between 0 and 1, with nodata as -9999).
#         raster_meta (dict): Metadata of the raster.
#         raster_bounds (tuple): Bounds of the raster (minx, miny, maxx, maxy).
#         nodata (float): NoData value of the raster (-9999).
#         polygon_file (GeoDataFrame): GeoDataFrame containing sidewalk polygons.
#         time (str): Time of day (e.g., '0830').
#         building_tree (str): Type of shade ('building' or 'tree').
#         timestr (str): Date in YYYYMMDD format.

#     Returns:
#         GeoDataFrame: Updated GeoDataFrame with percentage shade coverage added.
#     """
#     # Ensure the CRS of the polygons matches the raster
#     if polygon_file.crs != raster_meta["crs"]:
#         polygon_file = polygon_file.to_crs(raster_meta["crs"])

#     gdf = polygon_file.copy()
#     gdf = gdf.reset_index(drop=True)

#     # # Replace nodata values with NaN
#     # raster_data = np.where(raster_data == nodata, np.nan, raster_data)

#     # # Invert the raster:
#     # # - Fully exposed pixels (value == 1) become 0
#     # # - All shaded pixels (value < 1) become 1
#     # # - NoData values remain as NaN
#     # inverted_raster = np.where(
#     #     (raster_data < 1) & (raster_data >= 0),  # Shaded pixels
#     #     1,  # Set shaded pixels to 1
#     #     np.where(raster_data == 1, 0, np.nan),  # Exposed pixels to 0, nodata to NaN
#     # )
#     # Step 1: Replace nodata values with NaN
#     raster_data = np.where(raster_data == nodata, np.nan, raster_data)

#     # Step 2: Initialize inverted raster with NaN (default for invalid values)
#     inverted_raster = np.full_like(raster_data, np.nan)

#     # Step 3: Set shaded pixels (< 1) to 1
#     inverted_raster[(raster_data < 1) & (raster_data >= 0)] = 1

#     # Step 4: Set exposed pixels (== 1) to 0
#     inverted_raster[raster_data == 1] = 0

#     print(f"[DEBUG] Original raster min: {np.nanmin(raster_data)}, max: {np.nanmax(raster_data)}")
#     print(f"[DEBUG] Inverted raster min: {np.nanmin(inverted_raster)}, max: {np.nanmax(inverted_raster)}")
#     print(f"[DEBUG] Nodata replaced with NaN: {np.sum(np.isnan(raster_data))} pixels")

#     # Create a dictionary to store shade percentages for each polygon
#     coverage_dict = {}

#     # Get raster transform and resolution
#     transform = raster_meta["transform"]
#     pixel_width = transform.a
#     pixel_height = -transform.e  # Negative because y decreases in the raster grid

#     for idx, row in tqdm.tqdm(gdf.iterrows(), total=len(gdf)):
#         polygon = row["geometry"]

#         # Skip polygons that do not intersect raster bounds
#         if not polygon.intersects(box(*raster_bounds)):
#             coverage_dict[idx] = 0  # No shade coverage
#             continue

#         # Get the bounding box of the polygon in pixel coordinates
#         minx, miny, maxx, maxy = polygon.bounds
#         row_start = max(0, int((miny - transform.f) / transform.e))
#         row_stop = min(raster_data.shape[1], int((maxy - transform.f) / transform.e))
#         col_start = max(0, int((minx - transform.c) / transform.a))
#         col_stop = min(raster_data.shape[2], int((maxx - transform.c) / transform.a))

#         # Clip the raster to the polygon's bounding box
#         clipped_raster = inverted_raster[0, row_start:row_stop, col_start:col_stop]

#         # Create a mask for pixels inside the polygon
#         mask = np.zeros(clipped_raster.shape, dtype=bool)
#         for row_idx, y in enumerate(
#             np.arange(
#                 transform.f + row_start * pixel_height,
#                 transform.f + row_stop * pixel_height,
#                 pixel_height,
#             )
#         ):
#             for col_idx, x in enumerate(
#                 np.arange(
#                     transform.c + col_start * pixel_width,
#                     transform.c + col_stop * pixel_width,
#                     pixel_width,
#                 )
#             ):
#                 point = Point(x, y)
#                 mask[row_idx, col_idx] = polygon.contains(point)

#         # Apply the mask and calculate valid shaded pixels
#         valid_pixels = clipped_raster[mask]
#         valid_pixels = valid_pixels[~np.isnan(valid_pixels)]

#         # Skip polygons entirely within nodata regions
#         if len(valid_pixels) == 0:
#             coverage_dict[idx] = 0
#             continue

#         # Calculate the percentage of shaded pixels (value == 1 after inversion)
#         total_pixels = len(valid_pixels)
#         shaded_pixels = np.sum(valid_pixels == 1)  # Count pixels with value 1
#         shade_percentage = (shaded_pixels / total_pixels) * 100

#         coverage_dict[idx] = shade_percentage

#     # Update the GeoDataFrame with the calculated percentages
#     shade_column = f"{timestr}_{building_tree}_shade_percent_at_{time}"
#     gdf[shade_column] = gdf.index.map(coverage_dict)

#     return gdf

# def calculate_percentage_covered(raster_data, raster_meta, raster_bounds, nodata, polygon_file, time, building_tree, timestr):
#     """
#     Calculate the percentage of each polygon covered by shade, excluding nodata regions.

#     Parameters:
#         raster_data (ndarray): Raster data array (values between 0 and 1, with nodata as -9999).
#         raster_meta (dict): Metadata of the raster.
#         raster_bounds (tuple): Bounds of the raster (minx, miny, maxx, maxy).
#         nodata (float): NoData value of the raster (-9999).
#         polygon_file (GeoDataFrame): GeoDataFrame containing sidewalk polygons.
#         time (str): Time of day (e.g., '0830').
#         building_tree (str): Type of shade ('building' or 'tree').
#         timestr (str): Date in YYYYMMDD format.

#     Returns:
#         GeoDataFrame: Updated GeoDataFrame with percentage shade coverage added.
#     """
#     # Ensure the CRS of the polygons matches the raster
#     if polygon_file.crs != raster_meta["crs"]:
#         polygon_file = polygon_file.to_crs(raster_meta["crs"])

#     gdf = polygon_file.copy()
#     gdf = gdf.reset_index(drop=True)

#         # Exclude nodata values from inversion and handle valid values
#     inverted_raster = np.where(
#         (raster_data >= 0) & (raster_data <= 1),  # Only valid pixels
#         1 - raster_data,  # Invert valid pixels
#         np.nan,  # Set nodata and invalid values to NaN
#     )

#     # Ensure nodata is explicitly marked as NaN
#     inverted_raster = np.where(raster_data == nodata, np.nan, inverted_raster)

#     # # Handle dimensions to ensure the shape is (1, height, width)
#     # if inverted_raster.ndim == 3:  # Already correct shape
#     #     pass
#     # elif inverted_raster.ndim == 4:  # Remove extra dimension
#     #     inverted_raster = inverted_raster.squeeze(axis=0)
#     # elif inverted_raster.ndim == 2:  # Add band dimension if missing
#     #     inverted_raster = inverted_raster[np.newaxis, ...]
#     # else:
#     #     raise ValueError(f"Unexpected raster shape: {inverted_raster.shape}")

#     # # Validate final shape
#     # if inverted_raster.ndim != 3 or inverted_raster.shape[0] != 1:
#     #     raise ValueError(f"Final raster shape is invalid: {inverted_raster.shape}")


#     # print(f"[DEBUG] Original raster min: {np.nanmin(raster_data)}, max: {np.nanmax(raster_data)}")
#     # print(f"[DEBUG] Inverted raster min: {np.nanmin(inverted_raster)}, max: {np.nanmax(inverted_raster)}")
#     # print(f"[DEBUG] Number of NaN pixels in inverted raster: {np.sum(np.isnan(inverted_raster))}")
#     # print(f"[DEBUG] Original raster shape: {raster_data.shape}")
#     # print(f"[DEBUG] Inverted raster shape (after expand_dims): {inverted_raster.shape}")
#     # print(f"[DEBUG] Raster meta count: {raster_meta['count']}")

#     # # Update raster metadata to reflect the shape of inverted_raster
#     # raster_meta.update({
#     #     "count": inverted_raster.shape[0],  # Number of bands (should be 1)
#     #     "dtype": "float32",                # Ensure dtype matches the raster data
#     #     "nodata": np.nan,                  # Set nodata value to NaN
#     # })



#     # # Validate raster metadata and data shape
#     # print(f"[DEBUG] raster_meta['count']: {raster_meta['count']}")
#     # print(f"[DEBUG] raster_meta['dtype']: {raster_meta['dtype']}")
#     # print(f"[DEBUG] inverted_raster shape: {inverted_raster.shape}")
#     # print(f"[DEBUG] inverted_raster dtype: {inverted_raster.dtype}")

#     # Ensure dtype consistency
#     inverted_raster = inverted_raster.astype("float32")

#     # Explicitly enforce the correct shape for inverted_raster
#     inverted_raster = inverted_raster.squeeze()  # Remove redundant dimensions
#     if len(inverted_raster.shape) == 2:  # Add band dimension if missing
#         inverted_raster = inverted_raster[np.newaxis, ...]

#     # Validate the shape
#     if len(inverted_raster.shape) != 3 or inverted_raster.shape[0] != 1:
#         raise ValueError(f"Unexpected raster shape before write: {inverted_raster.shape}")

#     # Ensure dtype consistency
#     inverted_raster = inverted_raster.astype("float32")

#     # Update raster metadata
#     raster_meta.update({
#         "count": inverted_raster.shape[0],  # Number of bands (should be 1)
#         "dtype": "float32",                # Ensure dtype matches the raster data
#         "nodata": np.nan,                  # Set nodata value to NaN
#     })

#     # Debug outputs
#     print(f"[DEBUG] Final inverted raster shape before write: {inverted_raster.shape}")
#     print(f"[DEBUG] Writing raster with metadata: {raster_meta}")

#     # Create a dictionary to store shade percentages for each polygon
#     coverage_dict = {}

#     # Write the raster to a MemoryFile
#     with MemoryFile() as memfile:
#         with memfile.open(**raster_meta) as mem:
#             mem.write(inverted_raster, indexes=1)  # Write to the first band

#             # Iterate over each polygon in the GeoDataFrame
#             for idx, row in tqdm.tqdm(gdf.iterrows(), total=len(gdf)):
#                 polygon = row['geometry']

#                 # Skip polygons that do not intersect raster bounds
#                 if not polygon.intersects(box(*raster_bounds)):
#                     coverage_dict[idx] = 0  # No shade coverage
#                     continue

#                 # Mask the raster using the polygon
#                 out_image, _ = mask(mem, [mapping(polygon)], crop=True)
#                 masked_data = out_image[0].flatten()

#                 # Remove nodata pixels
#                 valid_pixels = masked_data[~np.isnan(masked_data)]

#                 # Skip polygons entirely within nodata regions
#                 if len(valid_pixels) == 0:
#                     coverage_dict[idx] = 0
#                     continue

#                 # Calculate the percentage of shaded pixels (value == 1 after inversion)
#                 total_pixels = len(valid_pixels)
#                 shaded_pixels = np.sum(valid_pixels == 1)  # Count pixels with value 1
#                 shade_percentage = (shaded_pixels / total_pixels) * 100

#                 coverage_dict[idx] = shade_percentage

#     # Update the GeoDataFrame with the calculated percentages
#     shade_column = f'{timestr}_{building_tree}_shade_percent_at_{time}'
#     gdf[shade_column] = gdf.index.map(coverage_dict)

#     return gdf

def validate_polygons(gdf, raster_bounds):
    """
    Checks how many polygons have valid geometries and intersect with raster bounds.

    Parameters:
        gdf (GeoDataFrame): GeoDataFrame containing polygons to validate.
        raster_bounds (tuple): Bounds of the raster (minx, miny, maxx, maxy).

    Returns:
        dict: A summary containing counts of valid, invalid, and intersecting polygons.
    """
    # Check validity of geometries
    valid_geometries = gdf.geometry.is_valid
    num_valid = valid_geometries.sum()
    num_invalid = (~valid_geometries).sum()

    # Check intersections with raster bounds
    raster_box = box(*raster_bounds)
    intersecting_polygons = gdf.geometry.apply(lambda geom: geom.is_valid and geom.intersects(raster_box))
    num_intersecting = intersecting_polygons.sum()

    # # Debugging output
    # print(f"[DEBUG] Total polygons: {len(gdf)}")
    # print(f"[DEBUG] Valid polygons: {num_valid}")
    # print(f"[DEBUG] Invalid polygons: {num_invalid}")
    # print(f"[DEBUG] Polygons intersecting raster bounds: {num_intersecting}")

    return {
        "total_polygons": len(gdf),
        "valid_polygons": num_valid,
        "invalid_polygons": num_invalid,
        "intersecting_polygons": num_intersecting
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a OSM area for shade metrics.")
    parser.add_argument('osmid', type=int, help='OSMID to be processed')
    parser.add_argument('dates', type=str, nargs='+', help='Dates in YYYY-MM-DD format (multiple dates allowed)')
    args = parser.parse_args()

    osmid = args.osmid
    dates_input = args.dates
    dates = [dt.datetime.strptime(date, "%Y-%m-%d") for date in dates_input]

    main(osmid, dates)