import argparse
import os
import re
import glob
import importlib
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.transform import Affine
from rasterio.transform import rowcol
import datetime as dt
from osgeo import gdal, osr
from osgeo.gdalconst import *
import shade_setup as shade
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from shapely.geometry import mapping

import matplotlib.pyplot as plt

from scipy.ndimage import median_filter
from scipy.ndimage import uniform_filter
from scipy.ndimage import gaussian_filter
from scipy.ndimage import minimum_filter

import startinpy

import concurrent.futures

importlib.reload(shade)

# Set exception handling
gdal.UseExceptions()

def main(dataset, osmid, longitude_column, latitude_column, timestamp_column_name, unique_ID_column, interval=30):
    """
    Processes a dataset for solar shading analysis using raster data and timestamps.

    This function:
    1. Calls `main_raster()` to process raster files.
    2. Converts the dataset into a GeoDataFrame and processes it for spatial analysis.
    3. Iterates over unique seasons and dates in the dataset.
    4. Extracts relevant data points for each day and runs shading analysis.
    5. Computes shade results for each point in the dataset.

    Args:
        dataset (pd.DataFrame): The input dataset containing location and timestamp data.
        osmid (str): Unique identifier for the study area.
        longitude_column (str): Name of the column containing longitude values.
        latitude_column (str): Name of the column containing latitude values.
        timestamp_column_name (str): Name of the column containing timestamp values.
        unique_ID_column (str): Name of the column containing unique point identifiers.
        interval (int, optional): Time interval (in minutes) for rounding timestamps. Defaults to 30.

    Returns:
        gpd.GeoDataFrame: The processed dataset with added shading results.

    Raises:
        ValueError: If no data is available for a given day in the dataset.
    """
    # process the rasters
    main_raster(osmid, f"../data/clean_data/solar/{osmid}")

    dataset_gdf = process_dataset(dataset,
                                  "../data/clean_data/solar/e6a08fb8/rdy_for_processing",
                                  longitude_column, latitude_column, timestamp_column_name, interval=30, geometry=False, crs="EPSG:4326"
                                  )

    # which seasons in dataset
    unique_seasons = dataset_gdf["season"].unique()

    for season in unique_seasons:
        to_run = dataset_gdf[dataset_gdf["season"] == season]
        # Extract the date (without time)
        unique_days = to_run[timestamp_column_name].dt.date.unique()
        unique_days_sorted = sorted(unique_days)
        for day in unique_days_sorted:
            # get the data points for that dat
            day_data = to_run[to_run[timestamp_column_name].dt.date == day]

            # check if data in that day
            if not day_data.empty:
                # get the input data for each timestamp
                tile_stamps = get_tile_data(day_data) # dictionary of tile number as key and (final_stamp, [shade_fr_1, shade_fr_2]) as value
                print(f"There are {len(list(tile_stamps.keys()))} tiles for this day")

                # run the analysis
                if season == 1:
                    main_shade(osmid, tile_stamps, day, sh_int, summer_params)
                else:
                    main_shade(osmid, tile_stamps, day, sh_int, winter_params)
            else:
                raise ValueError(f"No data available for the day: {day}")

    # Apply the function to each row in the GeoDataFrame
    dataset_gdf["shade_results"] = dataset_gdf.apply(
        lambda row: get_point_shaderesult(
            point=row.geometry,  # Geometry of the point
            index=row[unique_ID_column],
            rounded_timestamp=row["rounded_timestamp"],  # Timestamp column
            tile_id=row["tile_number"],  # Tile ID column
            building_shade_step=True,
            tree_shade_step=True,
            bldg_shadow_fraction=True,
            tree_shadow_fraction=True,
            hours_before=2
        ), axis=1
    )

    dataset_gdf[["building_shade", "tree_shade", "bldg_shadow_fraction", "tree_shadow_fraction", "bldg_hrs_before_shadow_fraction", "tree_hrs_before_shadow_fraction"]] = \
        pd.DataFrame(dataset_gdf["shade_results"].tolist(), index=dataset_gdf.index)

    return dataset_gdf

def main_raster(osmid, raster_dir):
    """
    Processes multiple raster files in parallel using a ThreadPoolExecutor.

    This function scans a directory for raster files (with filenames ending in `dsm.tif`),
    and processes each file concurrently using the `process_raster` function.

    Args:
        osmid (str): The unique identifier for the area of interest.
        raster_dir (str): The directory containing raster files.

    Returns:
        None: The function prints processing progress and handles any exceptions.

    Raises:
        Exception: Catches and prints any errors encountered during processing.
    """
    # Get a list of all raster files in the directory so we can load them incrementally
    raster_files = glob.glob(os.path.join(raster_dir, '*dsm.tif'))

    print(f"Processing {len(raster_files)} raster files.")

    # Use a ProcessPoolExecutor to process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(process_raster, file_path, osmid) for file_path in raster_files]
        # Optionally, wait for all tasks to complete and handle exceptions
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred: {e}")

def main_shade(osmid, timestamps, date, shade_interval=30, inputs={'utc':1, 'dst':0, 'trs':10}):
    """
    Processes shading analysis for multiple raster tiles.

    This function finds and processes raster files related to buildings and canopy cover
    based on given timestamps. It pairs corresponding building and canopy files and
    processes them in parallel using `shade_processing`.

    Args:
        osmid (str): The unique identifier for the area of interest.
        timestamps (dict): A dictionary where keys are tile identifiers (e.g., 'p_1')
                           and values are timestamp information.
        date (str): The date for shading calculations.
        shade_interval (int, optional): Time interval (in minutes) for shading analysis. Defaults to 30.
        inputs (dict, optional): Dictionary containing additional parameters:
            - `utc` (int): UTC offset.
            - `dst` (int): Daylight saving time adjustment.
            - `trs` (int): Some threshold value (context-dependent). Defaults to {'utc':1, 'dst':0, 'trs':10}.

    Returns:
        None: The function prints processing progress.

    Raises:
        Exception: If file paths or expected raster files are missing.
    """
    tile_numbers = timestamps.keys()  # e.g., ['p_1', 'p_5', 'p_15']

    # Directory containing the raster files
    processing_dir = f"../data/clean_data/solar/{osmid}/rdy_for_processing/"

    filtered_building_files = [
        bldg_path for bldg_path in glob.glob(os.path.join(processing_dir, '*building_dsm.tif')) if any(f"{tile}_" in bldg_path for tile in tile_numbers)
    ]

    filtered_canopy_files = [
        chm_path for chm_path in glob.glob(os.path.join(processing_dir, '*canopy_dsm.tif')) if any(f"{tile}_" in chm_path for tile in tile_numbers)
    ]

    # TODO: based on the previous filtered raster files get the building and canopy files
    # check based on len(raster files) == len(building files) == len(canopy files)

    print(f'Found {len(filtered_building_files)} building files')

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for bldg_path in filtered_building_files:
            identifier = extract_identifier(bldg_path)
            print("identifier: ", identifier)

            matched_chm_path = None
            for chm_path in filtered_canopy_files:
                if identifier+"_" in chm_path:
                    ## TODO: check if the file path and slashes might cause issues
                    matched_chm_path = chm_path
                    break
            if matched_chm_path:
                tile = "_".join(identifier.split('_')[-2:])
                executor.submit(shade_processing, bldg_path, matched_chm_path, osmid, date, shade_interval, timestamps[tile], inputs)

def assign_summer_winter(date):
    """
    Give a binary result whether it is winter or summer months

    Parameters:
        date (datetime.datetime): The date for which to assign the values.
    Returns:
        binary: 1 if summer months, 0 if winter months
    """
    if 4 <= date.month <= 10:
        return 1
    else:
        return 0

def get_interval_stamp(timestamp, interval=30):
    """
    Given a datetime object representing the timestamp for a day,
    return a new datetime object for the nearest interval boundary (in minutes)
    to that timestamp. The calculation assumes the day starts at 00:00.
    """
    minutes_since_midnight = timestamp.hour * 60 + timestamp.minute
    rounded_minutes = round(minutes_since_midnight / interval) * interval
    r_hour = rounded_minutes // 60
    r_minute = rounded_minutes % 60

    return timestamp.replace(hour=r_hour, minute=r_minute, second=0, microsecond=0)

def process_dataset(dataset, processed_raster_dir, longitude_column, latitude_column, timestamp_column, interval=30, geometry=False, crs="EPSG:4326"):
    """
    Processes a dataset of points and assigns each point to a DSM tile based on raster footprints.

    Parameters:
    - dataset: A DataFrame with longitude, latitude, and timestamp columns.
    - longitude_column: Name of the column containing longitude values.
    - latitude_column: Name of the column containing latitude values.
    - interval (optional): Time interval in minutes for rounding timestamps.
    - geometry (optional): Whether the dataset already has a geometry column.

    Returns:
    - GeoDataFrame with an added 'tile_number' column indicating the DSM tile each point falls in.
    """

    if not geometry:
        # Convert DataFrame to GeoDataFrame
        dataset["geometry"] = gpd.points_from_xy(dataset[longitude_column], dataset[latitude_column])
        df_gdf = gpd.GeoDataFrame(dataset, geometry="geometry", crs=crs)

        raster_files = glob.glob(os.path.join(processed_raster_dir, '*building_dsm.tif'))

        # Extract tile footprints from raster files
        raster_tiles = []
        raster_crs = None  # Store the raster CRS

        for raster_path in raster_files:
            match = re.search(r"p_\d+", raster_path)  # Extract tile number like 'p_0'
            if match:
                tile_number = match.group(0)
                with rasterio.open(raster_path) as src:
                    raster_crs = src.crs  # Ensure all rasters use the same CRS
                    transform = src.transform
                    width = src.width
                    height = src.height

                    # Get actual polygon footprint instead of just bounds
                    tile_polygon = box(*rasterio.transform.array_bounds(height, width, transform))
                    raster_tiles.append({"tile_number": tile_number, "geometry": tile_polygon})

        # Convert raster tile footprints to a GeoDataFrame
        tiles_gdf = gpd.GeoDataFrame(raster_tiles, crs=raster_crs)

        # Reproject points to match raster CRS
        df_gdf = df_gdf.to_crs(raster_crs)

        # Spatial join to assign each point to the correct tile
        df_gdf = gpd.sjoin(df_gdf, tiles_gdf, how="left", predicate="intersects")

        # Drop unnecessary columns from spatial join
        df_gdf.drop(columns=["index_right"], inplace=True, errors="ignore")

        df_gdf = df_gdf.dropna(subset=["tile_number"])

        # Convert timestamp column to datetime
        df_gdf[timestamp_column] = pd.to_datetime(df_gdf[timestamp_column])

        # assign binary based on month
        df_gdf["season"] = df_gdf[timestamp_column].apply(assign_summer_winter)

        # Apply correct rounding to nearest interval
        df_gdf["rounded_timestamp"] = df_gdf[timestamp_column].apply(lambda x: get_interval_stamp(x, interval))

        return df_gdf
    else:
        # TODO: write a case where there is a geometry column already
        pass

def get_tile_data(processed_df):
    """
    Given a processed dataset with tile numbers and rounded timestamps,
    return a dictionary with each tile's final timestamp and a list of interval timestamps.

    Parameters:
    - processed_df (GeoDataFrame): Processed dataframe with 'tile_number' and 'rounded_timestamp' columns.

    Returns:
    - dict: {tile_number: (final_timestamp, [interval_timestamp1, interval_timestamp2, ...])}
    """
    tile_data = {}

    # Group by tile number
    grouped = processed_df.groupby("tile_number")["rounded_timestamp"]

    print(grouped)

    for tile, timestamps in grouped:
        unique_timestamps = sorted(timestamps.unique())  # Get unique timestamps sorted

        if unique_timestamps:
            final_timestamp = unique_timestamps[-1]  # Last timestamp is the final one
            interval_timestamps = unique_timestamps[:-1]  # All but the final timestamp

            tile_data[tile] = (final_timestamp, interval_timestamps)

    return tile_data

def process_raster(path, osmid):
    try:
        print(f"Starting processing for {path} with OSMID: {osmid}")

        # Process each DSM file
        fixed_path = path.replace("\\", "/")
        last_slash_index = fixed_path.rfind("/")
        file_name = fixed_path[last_slash_index + 1:]
        print("File name:", file_name)

        # Define new file paths based on the osmid
        file_name_building = f"C:/Users/Dila Ozberkman/Desktop/AMS Research/Urban Shade/throwing_shade/data/clean_data/solar/{osmid}/rdy_for_processing/{file_name[:-7]}building_dsm.tif"
        file_name_trees = f"C:/Users/Dila Ozberkman/Desktop/AMS Research/Urban Shade/throwing_shade/data/clean_data/solar/{osmid}/rdy_for_processing/{file_name[:-7]}canopy_dsm.tif"
        # file_name_building = f'../data/clean_data/solar/{osmid}/rdy_for_processing/{file_name[:-7]}building_dsm.tif'
        # file_name_trees = f'../data/clean_data/solar/{osmid}/rdy_for_processing/{file_name[:-7]}canopy_dsm.tif'

        # List of file paths to check
        file_paths = [file_name_building, file_name_trees]

        # Check if the files already exist
        if check_files_exist(file_paths):
            print("Files already exist. Skipping creation.")
        else:
            # Read DSM
            with rasterio.open(path) as src:
                dsm_data = src.read(1)
                dsm_meta = src.meta.copy()
                dsm_crs = src.crs
                dsm_bounds = src.bounds
                dsm_transform = src.transform
                dsm_shape = dsm_data.shape

                # Extract further metadata
                width = src.width
                height = src.height
                nodata_value = src.nodata
                dtype = src.dtypes[0]

                # Calculate resolution
                resolution_x = dsm_transform[0]
                resolution_y = -dsm_transform[4]  # Typically negative in the geotransform

                # Calculate extent
                xmin = dsm_transform[2]
                ymax = dsm_transform[5]
                xmax = xmin + (width * resolution_x)
                ymin = ymax + (height * dsm_transform[4])  # Typically negative

            # # Create a bounding box polygon from the raster bounds
            dsm_bbox = box(dsm_bounds.left, dsm_bounds.bottom, dsm_bounds.right, dsm_bounds.top)
            dsm_bbox_gdf = gpd.GeoDataFrame({'geometry': [dsm_bbox]}, crs=dsm_crs)

            print("Making CHM mask")

            # New CHM mask path, identified by the OSMID and filename in the new folder
            chm_mask_folder = f"C:/Users/Dila Ozberkman/Desktop/AMS Research/Urban Shade/throwing_shade/data/clean_data/canopy_masks/{osmid}/"
            # chm_mask_folder = f'../data/clean_data/canopy_masks/{osmid}/'
            chm_mask_file = f'{chm_mask_folder}{file_name[:-7]}rgb_segmented.tif'

            if os.path.exists(chm_mask_file):
                print(f"CHM mask found: {chm_mask_file}")

                # Load the CHM mask (assuming it’s already in the same resolution and extent)
                with rasterio.open(chm_mask_file) as chm_src:
                    chm_mask = chm_src.read(1)  # Assuming it's a single-band mask

                # Apply the CHM mask to the DSM data
                canopy_dsm = np.where(chm_mask, dsm_data, np.nan)  # Use NaN for masked-out areas

            else:
                print(f"CHM mask not found: {chm_mask_file}. Skipping mask application.")

            # Read building_mask
            mask_path = path.replace("dsm", "mask")

            with rasterio.open(mask_path) as src:
                bldg_mask = src.read(1)
                bldg_mask_meta = src.meta.copy()
                bldg_transform = src.transform
                bldg_crs = src.crs
                bldg_dtype = src.dtypes[0]

            # Load corresponding AHN subtiles
            # buildings_path = f'../data/clean_data/solar/{osmid}/{osmid}_buildings.gpkg'
            buildings_path = f"C:/Users/Dila Ozberkman/Desktop/AMS Research/Urban Shade/throwing_shade/data/clean_data/solar/{osmid}/{osmid}_buildings.gpkg"
            buildings = gpd.read_file(buildings_path, mask=dsm_bbox_gdf)

            # Check if buildings GeoDataFrame is empty
            if buildings.empty:
                print("No buildings found in the mask area.")
                osm_bldg_mask = np.zeros(dsm_data.shape, dtype='uint8')  # Create an empty mask
            else:
                if buildings.crs != dsm_crs:
                    buildings = buildings.to_crs(dsm_crs)

                # Buffer to combat artefacts.
                buildings.geometry = buildings.buffer(1.5)

                # Rasterize building polygons (same size as dsm so it works with UMEP)
                print("Rasterizing building polygons")
                try:
                    osm_bldg_mask = rasterize(
                        ((mapping(geom), 1) for geom in buildings.geometry),
                        out_shape=dsm_data.shape,
                        transform=dsm_meta['transform'],
                        fill=0,
                        dtype='uint8'
                    )
                except Exception as e:
                    print(f"Error during rasterization: {e}")
                    osm_bldg_mask = np.zeros(dsm_data.shape, dtype='uint8')  # Create an empty mask in case of failure

            combined_building_mask = np.logical_or(bldg_mask, osm_bldg_mask).astype(np.uint8)
            combined_bldg_tree_mask = np.logical_or(chm_mask, combined_building_mask).astype(np.uint8)

            dtm_raw = np.where(combined_bldg_tree_mask == 0, dsm_data, np.nan)

            print("Filtering data")

            ### Filter the raw data
            ## Apply minimum filter
            filtered_data = apply_minimum_filter(dtm_raw, np.nan, size=50)
            # filtered_data = apply_minimum_filter(filtered_data, np.nan, size=50)
            filtered_data = apply_minimum_filter(filtered_data, np.nan, size=30)
            filtered_data = apply_minimum_filter(filtered_data, np.nan, size=10)

            ### Interpolate:

            print("Doing Laplace interpolation")

            t = dsm_transform
            pts = []
            coords = []
            for i in range(filtered_data.shape[0]):
                for j in range(filtered_data.shape[1]):
                    x = t[2] + (j * t[0]) + (t[0] / 2)
                    y = t[5] + (i * t[4]) + (t[4] / 2)
                    z = filtered_data[i][j]
                    # Add all point coordinates. Laplace interpolation keeps existing values.
                    coords.append([x,y])
                    if not np.isnan(z):
                        pts.append([x, y, z])
                        # print('data found')
            dt = startinpy.DT()
            dt.insert(pts, insertionstrategy="BBox")

            interpolated = dt.interpolate({"method": "Laplace"}, coords)

            if interpolated.shape != dsm_data.shape:
                print("Interpolation is fucking it up")
                print(f"Interpolated shape: {interpolated.shape}")
                print(f"Initial dsm shape: {dsm_data.shape}")

            # Calculate the number of rows and columns
            ncols = int((xmax - xmin) / resolution_x)
            nrows = int((ymax - ymin) / resolution_y)

            # Create an empty raster array
            raster_array = np.full((nrows, ncols), np.nan, dtype=np.float32)

            # Ensure the points are in the correct structure (startinpy returns a flattened 1D array containing only the interpolated values for some reason)

            # Combine the coordinates and values into a 2D array with shape (n, 3)
            points = np.array([(x, y, val) for (x, y), val in zip(coords, interpolated)])
            points

            # Check if the array length is a multiple of 3
            if points.size % 3 != 0:
                raise ValueError(f"Array size {points.size} is not a multiple of 3, cannot reshape.")
            # Reshape the points array if it's flattened
            if points.ndim == 1:
                points = points.reshape(-1, 3)

            # In this example, points should be a 2D array with shape (n, 3)
            print(f"Points shape: {points.shape}")

            # Map the points to the raster grid
            for point in points:
                if len(point) != 3:
                    raise ValueError(f"Expected point to have 3 elements (x, y, value), but got {len(point)} elements.")
                x, y, value = point
                # Skip points with NaN values
                if np.isnan(value):
                    continue

                col = int((x - xmin) / resolution_x)
                row = int((ymax - y) / resolution_y)

                # Ensure the indices are within bounds
                if 0 <= col < ncols and 0 <= row < nrows:
                    raster_array[row, col] = value

            # Define the transform (mapping from pixel coordinates to spatial coordinates)
            transform = from_origin(xmin, ymax, resolution_x, resolution_y)

            # print(transform)

            # Define the metadata for the new raster
            meta = {
                'driver': 'GTiff',
                'dtype': dtype,
                'nodata': nodata_value,
                'width': width,
                'height': height,
                'count': 1,
                'crs': dsm_crs,
                'transform': transform
            }

            post_interpol_filter = apply_minimum_filter(raster_array, np.nan, size=40)
            post_interpol_filter = apply_minimum_filter(post_interpol_filter, np.nan, size=20)

            dsm_buildings = np.where(combined_building_mask == 0, post_interpol_filter, dsm_data)

            # Save building dsm and canopy dsm
            print("Saving DSM and Canopy DSM")

            # Find the index of the last '/' character
            path = path.replace("\\", "/")
            last_slash_index = path.rfind('/')
            # Extract the part after the last '/' (excluding '/')
            file_name = path[last_slash_index + 1:]
            file_name_building = f"C:/Users/Dila Ozberkman/Desktop/AMS Research/Urban Shade/throwing_shade/data/clean_data/solar/{osmid}/rdy_for_processing/" + file_name[:-7] + "building_dsm.tif"
            file_name_trees = f"C:/Users/Dila Ozberkman/Desktop/AMS Research/Urban Shade/throwing_shade/data/clean_data/solar/{osmid}/rdy_for_processing/" + file_name[:-7] + "canopy_dsm.tif"
            # file_name_building = f'../data/clean_data/solar/{osmid}/rdy_for_processing/' + file_name[:-7] + "building_dsm.tif"
            # file_name_trees = f'../data/clean_data/solar/{osmid}/rdy_for_processing/' + file_name[:-7] + "canopy_dsm.tif"

            # processing_directory = f'../data/clean_data/solar/{osmid}/rdy_for_processing/'
            processing_directory = f"C:/Users/Dila Ozberkman/Desktop/AMS Research/Urban Shade/throwing_shade/data/clean_data/solar/{osmid}/rdy_for_processing/"

            directory_check(directory=processing_directory, shadow_check=False)


            # Replace nan values with 0 for canopy raster:
            canopy_dsm = np.nan_to_num(canopy_dsm, nan=0)

            n = 50

            crop_and_save_raster(canopy_dsm, dsm_transform, dsm_meta, nodata_value, n,file_name_trees)
            crop_and_save_raster(dsm_buildings, dsm_transform, dsm_meta, nodata_value, n,file_name_building)

            print(f"Original transform: {dsm_transform}")
            print(f"New transform: {transform}")

    except Exception as e:
        print(f"Error processing {path} with OSMID: {osmid}: {e}")

# Function to apply a median filter to a raster dataset
def apply_median_filter(data, nodata_value, size=3, nodata=True):

    if nodata:
        # Create a mask for nodata values
        mask = (data == nodata_value)

        # Apply the median filter only to valid data
        filtered_data = data.copy()
        filtered_data[~mask] = median_filter(data[~mask], size=size)
        print('Filtering: Ignoring Nodata')
    else:
        filtered_data = data.copy()
        filtered_data = median_filter(data, size=size)
        print('Filtering; Handling nodata')

    return filtered_data


# Function to apply a mean filter to a raster dataset
def apply_mean_filter(data, nodata_value, size=3):
    # Create a mask for nodata values
    mask = (data == nodata_value)

    # Apply the mean filter only to valid data
    filtered_data = data.copy()
    filtered_data[~mask] = uniform_filter(data[~mask], size=size)

    return filtered_data

# Function to apply a Gaussian filter to a raster dataset
def apply_gaussian_filter(data, nodata_value, sigma=1):
    # Create a mask for nodata values
    mask = (data == nodata_value)

    # Apply the Gaussian filter only to valid data
    filtered_data = data.copy()
    filtered_data[~mask] = gaussian_filter(data[~mask], sigma=sigma)

    return filtered_data

# Function to apply a minimum filter to a raster dataset
def apply_minimum_filter(data, nodata_value, size=3, nodata=True):

    if nodata:
        # Create a mask for nodata values
        mask = (data == nodata_value)

        # Apply the Gaussian filter only to valid data
        filtered_data = data.copy()
        filtered_data[~mask] = minimum_filter(data[~mask], size=size)
        print('Filtering: Ignoring Nodata')
    else:
        filtered_data = data.copy()
        filtered_data = minimum_filter(data, size=size)
        print('Filtering; Handling nodata')


    return filtered_data

def crop_and_save_raster(raster, transform, meta, nodata, n, out_path):
    # TODO: MAYBE JUST REPLACE THE NAN WITH MIN INSTEAD OF CROPPING?
    print(f"Before cropping: {raster.shape}")
    # Calculate new top-left corner coordinates
    new_x = transform.c + n * transform.a
    new_y = transform.f + n * transform.e

    # Calculate new transformation matrix
    new_transform = Affine(transform.a, transform.b, new_x,
                        transform.d, transform.e, new_y)

    # Crop the data by removing n pixels from each edge
    cropped_data = raster[n:-n, n:-n]

    # Find the minimum value of the non-NaN elements
    min_value = np.nanmin(cropped_data)

    # Fill NaN values with the minimum value
    cropped_data = np.where(np.isnan(cropped_data), min_value, cropped_data)

    print(f"After cropping: {cropped_data.shape}")

    # Update the metadata
    meta.update({
        'height': cropped_data.shape[0],
        'width': cropped_data.shape[1],
        'transform': new_transform
    })

    # Save the cropped raster data
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(cropped_data, 1)
        if nodata is not None:
            dst.nodata = nodata

    print(f"Cropped raster saved to {out_path}")

def check_files_exist(file_paths):
    """
    Check if all files in the list exist.

    Parameters:
    file_paths (list of str): List of file paths to check.

    Returns:
    bool: True if all files exist, False otherwise.
    """
    return all(os.path.exists(file_path) for file_path in file_paths)

# Function to extract the identifier from the file path
def extract_identifier(path):
    # Extract the last segment of the path
    last_segment = path.split('/')[-1]

    # Use regular expression to match the pattern before _20xx_
    match = re.match(r'(.*)_20\d{2}_', last_segment)

    if match:
        identifier = match.group(1)
    else:
        identifier = last_segment.split('_20')[0]

    return identifier

import os
import datetime as dt

def directory_check(directory, shadow_check=True, shade_intervals=False, date=dt.datetime.now()):
    """
    Checks if a directory exists and optionally verifies the presence of shadow fraction files.

    If the directory does not exist, it is created. If `shadow_check` is enabled, the function
    searches for files containing 'shadow_fraction_on_' followed by the given date. If
    `shade_intervals` is provided as a list of datetime objects, it returns a list of booleans
    indicating whether a file exists for each interval.

    Parameters:
    ----------
    directory : str
        The path to the directory to check or create.
    shadow_check : bool, optional
        Whether to check for shadow fraction files (default is True).
    shade_intervals : list of datetime, optional
        A list of datetime objects representing specific intervals to check for shadow fraction files.
    date : datetime, optional
        The reference date for file checking (default is the current date).

    Returns:
    -------
    bool or list of bool
        - If `shade_intervals` is not provided, returns True if at least one shadow fraction file is found,
          otherwise returns False.
        - If `shade_intervals` is provided, returns a list of booleans where each element corresponds to whether
          a shadow fraction file exists for a specific interval.

    Raises:
    ------
    ValueError:
        If `shade_intervals` is provided but is not a list of datetime objects.
    """

    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    else:
        print(f"Directory {directory} already exists.")

    # Convert date to string format
    timestr = date.strftime("%Y%m%d")

    if shadow_check:
        # Check for files containing 'shadow_fraction_on_' with the given date
        shadow_files = [f for f in os.listdir(directory) if f'shadow_fraction_on_{timestr}' in f]

        if shadow_files:
            if shade_intervals:
                # Ensure shade_intervals is a list of datetime objects
                if not isinstance(shade_intervals, list) or not all(isinstance(ts, dt.datetime) for ts in shade_intervals):
                    raise ValueError("shade_intervals must be a list of datetime objects.")

                shade_int_check = []
                for interval in shade_intervals:
                    int_time = interval.strftime("%Y%m%d_%H%M")
                    shadow_files_interval = [f for f in shadow_files if f'shadow_fraction_on_{int_time}' in f]
                    if shadow_files_interval:
                        print(f"File containing 'shadow_fraction_on_{int_time}' found: {shadow_files_interval}")
                        shade_int_check.append(True)
                    else:
                        print(f"No files containing 'shadow_fraction_on_{int_time}' found.")
                        shade_int_check.append(False)
                return shade_int_check
            else:
                print(f"Files containing 'shadow_fraction_on_{timestr}' found: {shadow_files}")
                return True  # Required files found
        else:
            print(f"No files containing 'shadow_fraction_on_{timestr}' found.")
            return False  # Required files not found

def filter_intervals(intervals, building_shadow_files_exist, tree_shadow_files_exist):
    # filter to only calculate intervals that don't have a file
    if isinstance(building_shadow_files_exist, list):
        assert len(intervals) == len(building_shadow_files_exist), "Directory check for the intervals is broken"
        building_intervals_needed = [intervals[i] for i, check in enumerate(building_shadow_files_exist) if not check]
        if len(building_intervals_needed) < 1:
            building_intervals_needed = False
    elif not building_shadow_files_exist:
        building_intervals_needed = intervals
    else:
        building_intervals_needed = False

    if isinstance(tree_shadow_files_exist, list):
        assert len(intervals) == len(tree_shadow_files_exist), "Directory check for the intervals is broken"
        tree_intervals_needed = [intervals[i] for i, check in enumerate(tree_shadow_files_exist) if not check]
        if len(tree_intervals_needed) < 1:
            tree_intervals_needed = False
    elif not tree_shadow_files_exist:
        tree_intervals_needed = intervals
    else:
        tree_intervals_needed = False

    return building_intervals_needed, tree_intervals_needed

def shade_processing(bldg_path, matched_chm_path, osmid, date, shade_interval, timestamps, inputs):
    final_stamp, intervals = timestamps

    if not intervals:
        intervals = False

    bldg_path = bldg_path.replace("\\", "/")
    matched_chm_path = matched_chm_path.replace("\\", "/")
    identifier = extract_identifier(bldg_path)

    print(f"This is the building path I am looking at: {bldg_path}, This is the matched canopy path I am looking at: {matched_chm_path}")

    # Check if the file exists
    if os.path.isfile(matched_chm_path):
        print(f"The file {matched_chm_path} exists.")
    else:
        print(f"The file {matched_chm_path} does not exist.")

    # Create directories
    folder_no = identifier.split('_')[-1]
    folder_no = '/' + folder_no
    tile_no = identifier
    # tile_no = '/' + identifier

    print("Tile no or identifier:", tile_no)

    building_directory = f"C:/Users/Dila Ozberkman/Desktop/AMS Research/Urban Shade/throwing_shade/code/results/output/{osmid}/building_shade{folder_no}/"
    tree_directory = f"C:/Users/Dila Ozberkman/Desktop/AMS Research/Urban Shade/throwing_shade/code/results/output/{osmid}/tree_shade{folder_no}/"
    # building_directory = f'../results/output/{osmid}/building_shade{folder_no}/'
    # tree_directory = f'../results/output/{osmid}/tree_shade{folder_no}/'

    # if shade_intervals is not empty, the return here is a list of booleans whether the file exists for each interval
    # if it is empty it is a single boolean for the final timestamp basically
    # TODO: Maybe should I add the final_stamp to the interval check somehow
    building_shadow_files_exist = directory_check(building_directory, shadow_check=True, shade_intervals=intervals, date=date)
    tree_shadow_files_exist = directory_check(tree_directory, shadow_check=True, shade_intervals=intervals, date=date)

    print("These are the intervals I need to calculate for: ", intervals)
    print("These are the building shade directory check for these intervals: ", building_shadow_files_exist)

    if intervals:
        # filter to only calculate intervals that don't have a file
        building_intervals_needed, tree_intervals_needed = filter_intervals(intervals, building_shadow_files_exist, tree_shadow_files_exist)
    else:
        building_intervals_needed = False
        tree_intervals_needed = False


    # if not building_shadow_files_exist:
    #     shade_bldg = shade.shadecalculation_setup(
    #         filepath_dsm=bldg_path,
    #         filepath_veg=matched_chm_path,
    #         tile_no=tile_no,
    #         date=date,
    #         intervalTime=shade_interval,
    #         final_stamp=final_stamp,
    #         shade_fractions=building_intervals_needed,
    #         onetime=0,
    #         filepath_save=building_directory,
    #         UTC=inputs['utc'],
    #         dst=inputs['dst'],
    #         useveg=0,
    #         trunkheight=25,
    #         # CHANGED TRANSMISSIVITY from 15 to 10 percent
    #         transmissivity=inputs['trs']
        # )

    print("Processing tree shade...")

    if not tree_shadow_files_exist:
        shade_veg = shade.shadecalculation_setup(
            filepath_dsm=bldg_path,
            filepath_veg=matched_chm_path,
            tile_no=tile_no,
            date=date,
            intervalTime=shade_interval,
            final_stamp=final_stamp,
            shade_fractions=tree_intervals_needed,
            onetime=0,
            filepath_save=tree_directory,
            UTC=inputs['utc'],
            dst=inputs['dst'],
            useveg=1,
            trunkheight=25,
            transmissivity=inputs['trs']
        )


def reproject_raster_to_dsm(src_raster, src_transform, src_crs, dst_crs, dst_transform, dst_shape):
    """
    Reproject a raster to the DSM's CRS.

    The CHM might be in a different CRS to the DSM.

    This function reprojects the CHM to the DSM crs.
    """
    dst_raster = np.empty(dst_shape, dtype=src_raster.dtype)
    reproject(
        source=src_raster,
        destination=dst_raster,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest
    )
    return dst_raster

def update_mask_within_extent(raster_path, combined_mask, dsm_bounds, dsm_transform, dsm_crs, dsm_shape):
    """
    This function reflects the fact that each DSM tile only overlaps the CHM partially, or overlaps multiple CHM tiles.

    We already have a list of CHM tiles that overlap.

    In this function we open the tile(s), reproject, and compute the window in which the CHM overlaps with the DSM

    Then we use the data in the window to create a raster mask for the original DSM"""
    with rasterio.open(raster_path) as src:
        # Reproject the entire additional raster to the DSM's CRS
        src_data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs

        reprojected_data = reproject_raster_to_dsm(
            src_data, src_transform, src_crs, dsm_crs, dsm_transform, dsm_shape
        )

        # Compute the window of the reprojected raster that overlaps with the DSM extent
        window = from_bounds(
            left=dsm_bounds.left, bottom=dsm_bounds.bottom,
            right=dsm_bounds.right, top=dsm_bounds.top,
            transform=dsm_transform
        )

        # Convert window indices to integers
        row_off = int(window.row_off)
        col_off = int(window.col_off)
        height = int(window.height)
        width = int(window.width)

        # Extract the overlapping window from the reprojected raster
        window_data = reprojected_data[
            row_off:row_off + height,
            col_off:col_off + width
        ]

        # Create a mask from the reprojected raster within the window
        additional_mask = window_data > 0  # Example condition to create a mask from the additional raster

        # Update the combined mask using the window's index
        row_start, col_start = row_off, col_off
        row_end, col_end = row_start + additional_mask.shape[0], col_start + additional_mask.shape[1]

        # Ensure the indices are within the bounds of the combined mask
        if row_start < 0 or col_start < 0 or row_end > combined_mask.shape[0] or col_end > combined_mask.shape[1]:
            print(f"Skipping {raster_path} due to out of bounds indices")
            return

        combined_mask[row_start:row_end, col_start:col_end] |= additional_mask

def get_earliest_timestamp(directory, date_obj):
    """
    Finds the earliest timestamp from raster filenames in a directory
    that match the given date.

    Parameters:
    - directory (str): Path to the directory containing the raster files.
    - date_obj (datetime): The reference date.

    Returns:
    - datetime: The earliest timestamp for the given date, or None if no match is found.
    """
    date_str = date_obj.strftime("%Y%m%d")  # Convert date to string format YYYYMMDD
    pattern = re.compile(r".*_(\d{8})_(\d{4})_LST\.tif")  # Regex to match date & time in filename

    timestamps = []

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            file_date, file_time = match.groups()
            if file_date == date_str:  # Check if the date matches
                timestamp = datetime.strptime(f"{file_date} {file_time}", "%Y%m%d %H%M")
                timestamps.append(timestamp)

    return min(timestamps) if timestamps else None

import os
import re
from datetime import datetime

def get_shade_files_in_range(base_path, tile_number, osmid, start_hour, rounded_timestamp):
    """
    Get all tree shade files in a directory within the range of start_hour and rounded_timestamp (inclusive).

    Parameters:
        base_path (str): The base directory where tree shade files are stored.
        tile_number (str): The tile number for shade calculations.
        osmid (str): The unique ID for the dataset.
        start_hour (datetime): The lower bound timestamp (inclusive).
        rounded_timestamp (datetime): The upper bound timestamp (inclusive).

    Returns:
        list: List of full file paths that fall within the specified time range.
    """
    # Directory containing the tree shade files
    directory = f"{base_path}/tree_shade/{tile_number}/"

    # Ensure directory exists
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return []

    # Regex pattern to extract timestamp from filenames
    pattern = re.compile(rf"{osmid}_p_{tile_number}_Shadow_(\d{{8}}_\d{{4}})_LST\.tif")

    # List all files in directory
    all_files = os.listdir(directory)

    # Filter and extract timestamps
    valid_files = []
    for filename in all_files:
        match = pattern.search(filename)
        if match:
            file_timestamp_str = match.group(1)  # Extract timestamp string
            file_timestamp = datetime.strptime(file_timestamp_str, "%Y%m%d_%H%M")  # Convert to datetime

            # Check if the timestamp is within the range (inclusive)
            if start_hour <= file_timestamp <= rounded_timestamp:
                valid_files.append(os.path.join(directory, filename))

    return sorted(valid_files)  # Return sorted list of file paths


def get_closest_shade_file(base_path, tile_number, osmid, start_hour):
    """
    Get the closest existing shade file to `start_hour`.
    If two timestamps are equidistant, choose the later one.

    Parameters:
        base_path (str): The base directory where tree shade files are stored.
        tile_number (str): The tile number for shade calculations.
        osmid (str): The unique ID for the dataset.
        start_hour (datetime): The target timestamp.

    Returns:
        str: Full file path of the closest shade file, or None if no files exist.
    """
    directory = f"{base_path}/tree_shade/{tile_number}/"

    # Ensure directory exists
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return None

    # Regex pattern to extract timestamp from filenames
    pattern = re.compile(rf"{osmid}_p_{tile_number}_Shadow_(\d{{8}}_\d{{4}})_LST\.tif")

    # List all files in directory
    all_files = os.listdir(directory)

    # Extract timestamps from filenames
    timestamps = []
    file_map = {}  # Dictionary to map timestamps to filenames
    for filename in all_files:
        match = pattern.search(filename)
        if match:
            file_timestamp_str = match.group(1)  # Extract timestamp string
            file_timestamp = datetime.strptime(file_timestamp_str, "%Y%m%d_%H%M")  # Convert to datetime
            timestamps.append(file_timestamp)
            file_map[file_timestamp] = os.path.join(directory, filename)

    # If no valid timestamps were found
    if not timestamps:
        print(f"No valid shade files found in {directory}")
        return None

    # Sort timestamps
    timestamps.sort()

    # Find the closest timestamp
    closest_timestamp = min(
        timestamps,
        key=lambda t: (abs((t - start_hour).total_seconds()), -t.timestamp())  # Prioritize later timestamps
    )

    return file_map[closest_timestamp]

from datetime import timedelta
import os

def hours_before_shadow_fr(point, index, base_path, shade_type, tile_number, rounded_timestamp, osmid, hours_before):
    """
    Computes the average shadow fraction for a given point over a specified time range before the rounded timestamp.

    Parameters:
    - base_path (str): The root directory where the shadow files are stored.
    - shade_type (str): Type of shade to consider (e.g., "tree_shade" or "building_shade").
    - tile_number (str or int): The tile identifier for the dataset.
    - rounded_timestamp (datetime.datetime): The timestamp for which the shadow fraction is being computed.
    - osmid (str): The unique identifier for the dataset.
    - hours_before (int or float): The number of hours before `rounded_timestamp` to consider for shadow fraction calculation.

    Returns:
    - float: The computed shadow fraction for the given time period.

    Raises:
    - Exception: If no shade files are found for the given date.
    - Exception: If no shade files are found between `start_hour` and `rounded_timestamp`.
    """
    # Get the earliest available shadow file timestamp for the given day
    first_shade_time = get_earliest_timestamp(f"{base_path}/{shade_type}/{tile_number}", rounded_timestamp)

    if first_shade_time is None:
        raise Exception("There are no shade files in the directory for this date")

    # Compute the starting timestamp based on hours_before
    start_hour = rounded_timestamp - timedelta(hours=hours_before)  # Ensure `hours_before` supports floats

    # Handle case where start_hour is before the first available shadow timestamp
    if start_hour <= first_shade_time:
        print("Start_hour is earlier or the same as first_shade_time, adjusting to first available time.")
        start_hour = first_shade_time

        # Construct the path to the shadow fraction raster file for `rounded_timestamp`
        timestamp_shadow_fraction_raster = f"{base_path}/{shade_type}/{tile_number}/{osmid}_{tile_id}_shadow_fraction_on_{rounded_timestamp.strftime('%Y%m%d_%H%M')}.tif"

        # If the shadow fraction raster file exists, return its value for the given point
        if os.path.exists(timestamp_shadow_fraction_raster):
            return extract_value_from_raster(timestamp_shadow_fraction_raster, point, index)  # Direct return if available

    # If the exact `start_hour` shadow file doesn't exist, find the closest valid one
    shadow_file_path = f"{base_path}/{shade_type}/{tile_number}/{osmid}_p_{tile_number}_Shadow_{start_hour.strftime('%Y%m%d_%H%M')}_LST.tif"
    if not os.path.exists(shadow_file_path):
        start_hour = get_closest_shade_file(base_path, tile_number, osmid, start_hour)

    # Retrieve all shadow files within the time range [start_hour, rounded_timestamp]
    shade_files_for_shadow_frac = get_shade_files_in_range(base_path, tile_number, osmid, start_hour, rounded_timestamp)

    if not shade_files_for_shadow_frac:
        raise Exception("Didn't find shade files between start time and timestamp")

    # Compute the shadow fraction by averaging the extracted values from all retrieved shade rasters
    shtot = 0
    for shade_raster in shade_files_for_shadow_frac:
        point_shade_value = extract_value_from_raster(shade_raster, point, index)
        if point_shade_value is not None:
            shtot += point_shade_value

    # Return the computed shadow fraction
    return shtot / len(shade_files_for_shadow_frac)

def extract_value_from_raster(raster_path, point, index):
    """Extract the raster value at the given point location."""
    if not os.path.exists(raster_path):
        print(f"Warning: Raster file {raster_path} not found.")
        return None

    with rasterio.open(raster_path) as src:
        row, col = rowcol(src.transform, point.x, point.y)
        # Ensure row, col are within valid range
        if row < 0 or row >= src.height or col < 0 or col >= src.width:
            with rasterio.open(raster_path) as src:
                # Compute new dimensions
                new_height = int(src.height * 2)
                new_width = int(src.width * 2)

                # Resample the raster data
                data = src.read(
                    out_shape=(src.count, new_height, new_width),
                    resampling=Resampling.bilinear
                )[0]  # Assuming single-band raster, extract first band

                # Compute new transform
                new_transform = src.transform * src.transform.scale(
                    (src.width / new_width),
                    (src.height / new_height)
                )

                # Get the row/col index of the point in the resampled raster
                row, col = rowcol(new_transform, point.x, point.y)

            # Ensure row, col are within valid bounds
            if row < 0 or row >= new_height or col < 0 or col >= new_width:
                print(f"Point index {index} is out of bounds for resampled raster {raster_path}.")
                return None  # Or np.nan

            return data[row, col]

        return src.read(1)[row, col]

def get_point_shaderesult(point, index, rounded_timestamp, tile_id, building_shade_step=False, tree_shade_step=False,
                          bldg_shadow_fraction = False, tree_shadow_fraction=False, hours_before=False):
    """
    Retrieve shade values for a given point based on tile ID and rounded timestamp.

    Parameters:
    - point (shapely.geometry.Point): The point geometry.
    - rounded_timestamp (datetime.datetime): The rounded timestamp to match shade calculations.
    - tile_id (str): The tile identifier (e.g., 'p_1', 'p_2').
    - building_shade (bool): Whether to include building shade data.
    - tree_shade (bool): Whether to include tree shade data.
    - shadow_fraction (bool): Whether to include shadow fraction data.

    Returns:
    - tuple: (building_shade_value, tree_shade_value, shadow_fraction_value)
    """
    tile_number = tile_id.split("_")[-1]
    # Define paths for building, tree, and shadow fraction rasters
    base_path = f"C:/Users/Dila Ozberkman/Desktop/AMS Research/Urban Shade/throwing_shade/code/results/output/{osmid}"

    building_shade_path = f"{base_path}/building_shade/{tile_number}/{osmid}_{tile_id}_Shadow_{rounded_timestamp.strftime('%Y%m%d_%H%M')}_LST.tif"
    tree_shade_path = f"{base_path}/tree_shade/{tile_number}/{osmid}_{tile_id}_Shadow_{rounded_timestamp.strftime('%Y%m%d_%H%M')}_LST.tif"
    bldg_shadow_fraction_path = f"{base_path}/building_shade/{tile_number}/{osmid}_{tile_id}_shadow_fraction_on_{rounded_timestamp.strftime('%Y%m%d_%H%M')}.tif"
    tree_shadow_fraction_path = f"{base_path}/tree_shade/{tile_number}/{osmid}_{tile_id}_shadow_fraction_on_{rounded_timestamp.strftime('%Y%m%d_%H%M')}.tif"

    # Initialize result variables
    building_shade_value, tree_shade_value, bldg_shadow_fraction_value, tree_shadow_fraction_value, tree_hrs_before_shadow_fraction, bldg_hrs_before_shadow_fraction = None, None, None, None, None, None

    # Extract values if the respective shade calculations exist
    # TODO: exceptions if any path not existing
    if building_shade_step:
        building_shade_value = extract_value_from_raster(building_shade_path, point, index)

    if tree_shade_step:
        tree_shade_value = extract_value_from_raster(tree_shade_path, point, index)

    if bldg_shadow_fraction:
        bldg_shadow_fraction_value = extract_value_from_raster(bldg_shadow_fraction_path, point, index)

    if tree_shadow_fraction:
        tree_shadow_fraction_value = extract_value_from_raster(tree_shadow_fraction_path, point, index)

    # idk if this is the right statement
    if hours_before:
        # TODO: add an assertation for hours_before
        if tree_shade_step:
            tree_hrs_before_shadow_fraction = hours_before_shadow_fr(point, index, base_path, "tree_shade", tile_number, rounded_timestamp, osmid, hours_before)

        if building_shade_step:
            bldg_hrs_before_shadow_fraction = hours_before_shadow_fr(point, index, base_path, "building_shade", tile_number, rounded_timestamp, osmid, hours_before)

    return (building_shade_value, tree_shade_value, bldg_shadow_fraction_value, tree_shadow_fraction_value, bldg_hrs_before_shadow_fraction, tree_hrs_before_shadow_fraction)
