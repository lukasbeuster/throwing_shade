import os
import glob
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine, from_origin
import numpy as np
import geopandas as gpd
from shapely.geometry import box, mapping
from pathlib import Path
from scipy.ndimage import minimum_filter
import startinpy
import concurrent.futures

# --- Main Entry Point for the Module ---

def raster_processing_main(config, osmid):
    """
    Finds all raw DSMs and orchestrates their processing in parallel.
    """
    raster_dir = Path(config["output_dir"]) / f"step2_solar_data/{osmid}"
    raster_files = list(raster_dir.glob('*dsm.tif'))

    print(f"Found {len(raster_files)} raster files to process.")
    max_workers = config.get('max_workers', 2) # Use .get for a safe default

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_raster, config, file_path, osmid) for file_path in raster_files]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"A raster processing task failed: {e}")

# --- High-Level Workflow for a Single Raster ---

def process_raster(config, path, osmid):
    """
    Orchestrates the full processing pipeline for a single DSM tile.
    """
    try:
        output_dir_raster = Path(config["output_dir"]) / f"step4_raster_processing/{osmid}"
        output_dir_raster.mkdir(parents=True, exist_ok=True)

        tile_stem = Path(path).stem.replace('_dsm', '')
        output_paths = {
            "building_dsm": output_dir_raster / f"{tile_stem}_building_dsm.tif",
            "canopy_dsm": output_dir_raster / f"{tile_stem}_canopy_dsm.tif"
        }

        if all(p.exists() for p in output_paths.values()):
            print(f"Skipping {Path(path).name}, output files already exist.")
            return

        print(f"Processing {Path(path).name}...")

        # 1. Read source DSM and its metadata
        with rasterio.open(path) as src:
            dsm_data = src.read(1)
            dsm_meta = src.meta.copy()
            dsm_bounds = src.bounds
            dsm_crs = src.crs

        # 2. Prepare all necessary masks (buildings, trees)
        combined_bldg_tree_mask, combined_building_mask, canopy_dsm = _prepare_masks(
            config, osmid, dsm_data, dsm_crs, dsm_bounds, tile_stem, dsm_meta
        )

        # 3. Create Digital Terrain Model (DTM) by interpolating ground points
        dtm_raw = np.where(combined_bldg_tree_mask == 0, dsm_data, np.nan)
        interpolated_dtm = _interpolate_dtm(dtm_raw, dsm_meta)

        # 4. Create the final analysis-ready DSMs
        dsm_buildings = np.where(combined_building_mask == 0, interpolated_dtm, dsm_data)

        # 5. Crop and save the final output rasters
        _save_output_rasters(
            config=config,
            canopy_dsm=canopy_dsm,
            dsm_buildings=dsm_buildings,
            dsm_meta=dsm_meta,
            output_paths=output_paths
        )
        print(f"Successfully processed and saved outputs for {Path(path).name}.")

    except Exception as e:
        print(f"Error processing {path}: {e}")
        # Optionally re-raise the exception if you want the main executor to catch it
        # raise e

# --- Internal Helper Functions ---

def _prepare_masks(config, osmid, dsm_data, dsm_crs, dsm_bounds, tile_stem, dsm_meta):
    """
    Loads and combines building and canopy masks to create analysis masks.
    """
    # Load Canopy Height Model (CHM) mask from segmentation step
    chm_mask_template = config['paths']['chm_mask_template']
    chm_mask_file = Path(config["output_dir"]) / chm_mask_template.format(osmid=osmid, tile_name=tile_stem)

    if chm_mask_file.exists():
        with rasterio.open(chm_mask_file) as chm_src:
            chm_mask = chm_src.read(1).astype(bool)
        canopy_dsm = np.where(chm_mask, dsm_data, np.nan)
    else:
        print(f"Warning: CHM mask not found at {chm_mask_file}. Canopy DSM will be empty.")
        chm_mask = np.zeros_like(dsm_data, dtype=bool)
        canopy_dsm = np.full_like(dsm_data, np.nan, dtype=float)

    # Load building mask from original solar data
    mask_path_template = config['paths']['solar_data_input'] + '/{tile_stem}_mask.tif'
    mask_path = Path(config['output_dir']) / mask_path_template.format(osmid=osmid, tile_stem=tile_stem)
    with rasterio.open(mask_path) as src:
        bldg_mask = src.read(1).astype(bool)

    # Load and rasterize OSM building footprints
    dsm_bbox_gdf = gpd.GeoDataFrame({'geometry': [box(*dsm_bounds)]}, crs=dsm_crs)
    buildings_path = Path(config['output_dir']) / config['paths']['buildings_gpkg'].format(osmid=osmid)
    buildings = gpd.read_file(str(buildings_path), mask=dsm_bbox_gdf)

    if not buildings.empty:
        osm_bldg_mask = rasterize(
            ((mapping(geom.buffer(1.5)), 1) for geom in buildings.geometry),
            out_shape=dsm_data.shape,
            transform=dsm_meta['transform'],
            fill=0,
            dtype='uint8'
        ).astype(bool)
    else:
        osm_bldg_mask = np.zeros_like(dsm_data, dtype=bool)

    # Combine all masks
    combined_building_mask = np.logical_or(bldg_mask, osm_bldg_mask)
    combined_bldg_tree_mask = np.logical_or(chm_mask, combined_building_mask)

    return combined_bldg_tree_mask, combined_building_mask, canopy_dsm

def _interpolate_dtm(dtm_raw, dsm_meta):
    """
    Takes a raw DTM with holes and interpolates the gaps using startinpy.
    """
    # Apply minimum filters to expand ground points
    filtered_data = apply_minimum_filter(dtm_raw, np.nan, size=50)
    filtered_data = apply_minimum_filter(filtered_data, np.nan, size=30)
    filtered_data = apply_minimum_filter(filtered_data, np.nan, size=10)

    # Prepare points for interpolation
    t = dsm_meta['transform']
    rows, cols = np.where(~np.isnan(filtered_data))
    x_coords, y_coords = t * (cols, rows)
    z_coords = filtered_data[rows, cols]

    pts = np.vstack([x_coords, y_coords, z_coords]).T

    # Triangulate and interpolate
    dt = startinpy.DT()
    dt.insert(pts)

    # Define grid to interpolate onto
    grid_x, grid_y = np.meshgrid(
        np.arange(dsm_meta['width']) * t.a + t.c,
        np.arange(dsm_meta['height']) * t.e + t.f
    )
    grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    interpolated_values = dt.interpolate_laplace(grid_coords)
    interpolated_dtm = interpolated_values.reshape(dsm_meta['height'], dsm_meta['width'])

    # Apply post-interpolation filter
    interpolated_dtm = apply_minimum_filter(interpolated_dtm, np.nan, size=40, nodata=False)
    interpolated_dtm = apply_minimum_filter(interpolated_dtm, np.nan, size=20, nodata=False)

    return interpolated_dtm

def _save_output_rasters(config, canopy_dsm, dsm_buildings, dsm_meta, output_paths):
    """
    Crops and saves the final canopy and building DSM rasters.
    """
    # Replace NaN in canopy with 0 before saving
    canopy_dsm_filled = np.nan_to_num(canopy_dsm, nan=0)

    crop_pixels = config.get('raster_crop_pixels', 50) # Make crop size configurable

    # Save Canopy DSM
    crop_and_save_raster(
        raster=canopy_dsm_filled,
        meta=dsm_meta,
        n=crop_pixels,
        out_path=output_paths['canopy_dsm']
    )

    # Save Building DSM
    crop_and_save_raster(
        raster=dsm_buildings,
        meta=dsm_meta,
        n=crop_pixels,
        out_path=output_paths['building_dsm']
    )

def apply_minimum_filter(data, nodata_value, size=3, nodata=True):
    # (Your existing function - no changes needed)
    if nodata:
        mask = np.isnan(data) if np.isnan(nodata_value) else (data == nodata_value)
        filtered_data = data.copy()
        valid_data = data[~mask]
        if valid_data.size > 0:
            filtered_data[~mask] = minimum_filter(valid_data, size=size)
    else:
        filtered_data = minimum_filter(data, size=size)
    return filtered_data

def crop_and_save_raster(raster, meta, n, out_path):
    # (Your existing function, simplified slightly)
    # Calculate new transformation matrix
    t = meta['transform']
    new_transform = t * Affine.translation(n, n)

    # Crop the data by removing n pixels from each edge
    cropped_data = raster[n:-n, n:-n]

    # Fill remaining NaN values with the local minimum
    if np.isnan(cropped_data).any():
        min_value = np.nanmin(cropped_data)
        cropped_data = np.nan_to_num(cropped_data, nan=min_value)

    # Update the metadata
    meta.update({
        'height': cropped_data.shape[0],
        'width': cropped_data.shape[1],
        'transform': new_transform
    })

    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(cropped_data.astype(meta['dtype']), 1)
