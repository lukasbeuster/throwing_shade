import os
import glob
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine, from_origin
from osgeo.gdalconst import *
import numpy as np
import geopandas as gpd
from shapely.geometry import box, mapping
from pathlib import Path
from scipy.ndimage import minimum_filter
import startinpy
import concurrent.futures

def raster_processing_main(config, osmid):
    # Get a list of all raster files in the directory so we can load them incrementally
    raster_dir = Path(config["output_dir"]) / f"step2_solar_data/{osmid}"
    raster_files = glob.glob(os.path.join(str(raster_dir), '*dsm.tif'))

    print(f"Processing {len(raster_files)} raster files.")

    max_workers = config['max_workers']
    # Use a ProcessPoolExecutor to process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(process_raster, config, file_path, osmid) for file_path in raster_files]
        # Optionally, wait for all tasks to complete and handle exceptions
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred: {e}")

def process_raster(config, path, osmid):
    try:
        print(f"Starting processing for {path} with OSMID: {osmid}")

        # Process each DSM file
        fixed_path = path.replace("\\", "/")
        last_slash_index = fixed_path.rfind("/")
        file_name = fixed_path[last_slash_index + 1:]

        # Define new file paths based on the osmid
        file_name_building = Path(config["output_dir"]) / f"step4_raster_processing/{osmid}/{file_name[:-7]}building_dsm.tif"
        file_name_trees = Path(config["output_dir"]) / f"step4_raster_processing/{osmid}/{file_name[:-7]}canopy_dsm.tif"

        # List of file paths to check
        file_paths = [str(file_name_building), str(file_name_trees)]

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
            chm_mask_file = Path(config["output_dir"]) / f"step4_solar_data/{osmid}/{file_name[:-7]}rgb_segmented.tif"

            if os.path.exists(chm_mask_file):
                print(f"CHM mask found: {chm_mask_file}")

                # Load the CHM mask (assuming it’s already in the same resolution and extent)
                with rasterio.open(str(chm_mask_file)) as chm_src:
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

            # Load OSM building footprints
            buildings_path = Path(config['output_dir']) / f'step1_solar_coverage/{osmid}_buildings.gpkg'
            buildings = gpd.read_file(str(buildings_path), mask=dsm_bbox_gdf)

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

            ### Filter the raw data
            ## Apply minimum filter
            filtered_data = apply_minimum_filter(dtm_raw, np.nan, size=50)
            filtered_data = apply_minimum_filter(filtered_data, np.nan, size=30)
            filtered_data = apply_minimum_filter(filtered_data, np.nan, size=10)

            ### Interpolate:

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
            file_name_building = Path(config['output']) / f"step4_raster_processing/{osmid}/{file_name[:-7]}building_dsm.tif"
            file_name_trees = Path(config['output']) / f"step4_raster_processing/{osmid}/{file_name[:-7]}canopy_dsm.tif"

            # processing_directory = f'../data/clean_data/solar/{osmid}/rdy_for_processing/'
            processing_directory = Path(config['output']) / f"step4_raster_processing/{osmid}"
            processing_directory.mkdir(parents=True, exist_ok=True)

            # Replace nan values with 0 for canopy raster:
            canopy_dsm = np.nan_to_num(canopy_dsm, nan=0)

            n = 50

            crop_and_save_raster(canopy_dsm, dsm_transform, dsm_meta, nodata_value, n, file_name_trees)
            crop_and_save_raster(dsm_buildings, dsm_transform, dsm_meta, nodata_value, n, file_name_building)

    except Exception as e:
        print(f"Error processing {path} with OSMID: {osmid}: {e}")

def check_files_exist(file_paths):
    """
    Check if all files in the list exist.
    """
    return all(os.path.exists(file_path) for file_path in file_paths)

def apply_minimum_filter(data, nodata_value, size=3, nodata=True):
    if nodata:
        # Create a mask for nodata values
        mask = (data == nodata_value)

        # Apply the Gaussian filter only to valid data
        filtered_data = data.copy()
        filtered_data[~mask] = minimum_filter(data[~mask], size=size)
    else:
        filtered_data = data.copy()
        filtered_data = minimum_filter(data, size=size)

    return filtered_data

def crop_and_save_raster(raster, transform, meta, nodata, n, out_path):
    # TODO: MAYBE JUST REPLACE THE NAN WITH MIN INSTEAD OF CROPPING?
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
