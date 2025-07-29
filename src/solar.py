import os
import hashlib
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box
from shapely.ops import unary_union
import osmnx as ox
from dotenv import load_dotenv

import src.solar_api_downloader as sapi

## MAIN

def check_coverage(config):
    """
    Calculates required solar tiles without downloading.
    Saves a preview map and returns information to the user.
    """
    print("Checking solar tile coverage...")
    dataset_path = config['dataset_path']
    lon_col = config['columns']['longitude']
    lat_col = config['columns']['latitude']

    # Determine CRS settings
    input_crs = config.get('input_crs', 'EPSG:4326')
    output_crs = config.get('output_crs', 'EPSG:32632')

    # Load data into GeoDataFrame regardless of format
    suffix = Path(dataset_path).suffix.lower()
    if suffix == '.csv':
        df = pd.read_csv(dataset_path)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs=input_crs
        )
    elif suffix in ('.pkl', '.pickle'):
        df = pd.read_pickle(dataset_path)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs=input_crs
        )
    elif suffix == ('.parquet'):
        df = pd.read_parquet(dataset_path)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs=input_crs
        )
    else:
        # GeoJSON, GPKG, shapefiles, etc.
        gdf = gpd.read_file(dataset_path)

    # Reproject to output CRS
    points_dataset = gdf.to_crs(output_crs)

    # Now run your existing grid logic:
    valid_grid_gdf_filtered = make_valid_grid(points_dataset, config)

    # calculate the number of tiles to downlaod
    tile_count = len(valid_grid_gdf_filtered)

    # --- SAVE INTERMEDIATE RESULTS ---
    output_dir = Path(config['output_dir']) / 'step1_solar_coverage/coverage_previews'
    output_dir.mkdir(parents=True, exist_ok=True)

    preview_path = output_dir / f'coverage_preview_{tile_count}_tiles.geojson'
    valid_grid_gdf_filtered.to_file(preview_path, driver='GeoJSON')

    # --- Mocked output for demonstration ---
    print(f"Coverage check complete. Found {tile_count} tiles.")

    return tile_count, preview_path

def download_data(config, preview_path):
    """
    Downloads the data for the tiles identified in the check_coverage step.
    """
    grid_gdf = gpd.read_file(preview_path)

    print(f"Downloading solar data for {grid_gdf.shape[0]} tiles...")

    # define request area
    request_area = define_request_area(grid_gdf)

    # generate osmid
    osmid = generate_osm_id(request_area.geometry.iloc[0].bounds)
    print(f"Generated OSM ID for dataset: {osmid}")

    # generate centroids
    centroid_gdf = generate_centroids(grid_gdf, osmid)

    # Save request points
    points_save_path = Path(config['output_dir']) / f'step1_solar_coverage/{osmid}_query_points.gpkg'
    save_points(centroid_gdf, points_save_path)

    buildings_save_path = Path(config['output_dir']) / f'step1_solar_coverage/{osmid}_buildings.gpkg'
    all_buildings = download_building_footprints(request_area, buildings_save_path)

    valid_request_points = get_valid_request_points(config, centroid_gdf, request_area)

    print("Solar API downloading...")

    api_response = download_google_api_data(valid_request_points, osmid, config)

    print("Download finished.")
    # This function should return the 'osmid' or a similar identifier for the next step.
    return osmid

## HELPERS

def generate_osm_id(bounds):
    """
    Generate a unique identifier for a bounding box using a hash function.
    """
    minx, miny, maxx, maxy = bounds
    bbox_str = f"{minx}_{miny}_{maxx}_{maxy}"
    return hashlib.md5(bbox_str.encode()).hexdigest()[:8]  # Generate an 8-character hash

def save_points(points_gdf, output_path):
    """Save points GeoDataFrame to a GeoPackage."""
    points_gdf.to_file(output_path, driver='GPKG')
    print(f"Request points saved to: {output_path}")

def make_valid_grid(points_dataset, config):
    """Makes grid based on dataset extent"""
    # generate grid
    hull, grid_gdf = adaptive_grid_from_convex_hull(points_dataset, buffer_distance=50, cell_size=950)

    # filter to only cells containing points
    joined = gpd.sjoin(grid_gdf, points_dataset, how="left", predicate="intersects")
    valid_joined = joined[joined["index_right"].notna()]

    # Use the indices from valid_joined to select the corresponding rows from grid_gdf
    valid_grid_gdf = grid_gdf.loc[valid_joined.index.unique()]

    # Count the number of points per grid cell
    point_counts = valid_joined.groupby(valid_joined.index).size().reset_index(name="point_count")

    # Merge the count data into the valid grid GeoDataFrame
    valid_grid_gdf = valid_grid_gdf.merge(point_counts, left_index=True, right_on="index", how="left")

    # filter by minimum point count
    valid_grid_gdf_filtered = valid_grid_gdf[valid_grid_gdf["point_count"] >= config['solar_api']['min_points_per_tile']]

    valid_grid_gdf_filtered = valid_grid_gdf_filtered.to_crs("EPSG:4326")

    return valid_grid_gdf_filtered

def define_request_area(grid_gdf):
    # define request area based on grid
    request_area = gpd.GeoDataFrame(geometry=[grid_gdf.geometry.unary_union], crs=grid_gdf.crs)
    # Reproject back to geographic CRS if needed
    if not request_area.crs.is_geographic:
        request_area = request_area.to_crs(4326)
    # Validating the input geometry
    if not request_area.is_valid.all():
        request_area = request_area.buffer(0)  # Fix invalid geometries
    if request_area.geometry.is_empty.any():
        print("The geometry for the region is empty or invalid.")
    return request_area

# In src/solar.py

def generate_centroids(grid_gdf, osmid):
    """
    Generates a SolarAPI suitable GeoDataframe for request
    points (grid centroids), ensuring accurate calculation by using a
    projected CRS.
    """
    # Save the original geographic CRS to convert back to later
    original_crs = grid_gdf.crs

    # Re-project to a projected CRS suitable for distance calculations (e.g., Web Mercator)
    projected_gdf = grid_gdf.to_crs("EPSG:3857")

    # Calculate the centroid on the ACCURATE projected data
    projected_gdf["centroid"] = projected_gdf.geometry.centroid

    # Create the new GeoDataFrame using the projected data
    centroid_gdf = gpd.GeoDataFrame(projected_gdf.drop(columns="geometry"),
                                    geometry="centroid",
                                    crs="EPSG:3857")

    # The rest of your function remains the same, but now it operates on the new centroid_gdf
    centroid_gdf = centroid_gdf.rename(columns={"centroid": "geometry"})
    centroid_gdf.set_geometry("geometry", inplace=True)

    centroid_gdf = centroid_gdf.reset_index(drop=True)
    centroid_gdf['osm_id'] = osmid
    centroid_gdf['id'] = 'p_'+ centroid_gdf.index.astype(str)

    # Project back to the original geographic CRS (WGS84) for the Solar API
    centroid_gdf = centroid_gdf.to_crs(original_crs)

    return centroid_gdf

def get_valid_request_points(config, centroid_gdf, request_area):
    """Checks that the request points are valid based on SolarAPI coverage"""
    # Load SolarAPIMediumArea and SolarAPIHighArea
    solar_coverage_medium = gpd.read_file(config['dependencies']['solar_coverage_medium'])
    solar_coverage_high = gpd.read_file(config['dependencies']['solar_coverage_high'])

    # Reproject both to match gdf's CRS
    solar_coverage_medium = solar_coverage_medium.to_crs(request_area.crs)
    solar_coverage_high = solar_coverage_high.to_crs(request_area.crs)

    # Generate the medium and high boundaries
    medium_boundary = solar_coverage_medium.geometry.union_all()
    high_boundary = solar_coverage_high.geometry.union_all()

    # Check if centroids are within the medium or high boundary
    inside_medium = centroid_gdf.geometry.within(medium_boundary)
    inside_high = centroid_gdf.geometry.within(high_boundary)

    # Combine results: A point is valid if it's in either boundary
    valid_points = inside_medium | inside_high

    # Check if all points are valid
    all_points_valid = valid_points.all()
    if all_points_valid:
        print(f"All points within SolarAPI coverage: {all_points_valid}")
    else:
        # This is not an error, just a filter. Let's change the message.
        print(f"Filtering to {valid_points.sum()} points within SolarAPI coverage.")

    valid_centroids_gdf = centroid_gdf[valid_points]

    return valid_centroids_gdf

def download_building_footprints(gdf, save_path):
    """
    Download building footprints (polygons only) for the geometries in the GeoDataFrame.
    """
    try:
        if save_path.exists():
            print(f"Buildings file already exists at: {save_path}")
            return gpd.read_file(save_path)

        all_building_footprints = []
        tags = {"building": True}

        for polygon in gdf.geometry:
            if polygon.is_valid and not polygon.is_empty:
                try:
                    # Query OSM buildings
                    buildings = ox.features_from_polygon(polygon, tags)

                    if not buildings.empty:
                        # --- THE FIX: Filter for only polygon geometries ---
                        footprints = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()

                        if not footprints.empty:
                            all_building_footprints.append(footprints)
                        else:
                            print("No building footprints (polygons) found for a given polygon, though other building types might exist.")

                except Exception as e:
                    print(f"Could not retrieve features for a polygon: {e}")

        if not all_building_footprints:
             print("No building footprints were found for the entire area.")
             return gpd.GeoDataFrame()

        # Concatenate all found footprints into a single GeoDataFrame
        final_gdf = gpd.GeoDataFrame(pd.concat(all_building_footprints, ignore_index=True))

        # Reset CRS if lost during concat
        if final_gdf.crs is None and all_building_footprints:
             final_gdf.set_crs(all_building_footprints[0].crs, inplace=True)

        # Clean up columns for saving
        # Convert any list-type columns to strings to avoid saving errors
        for col in final_gdf.columns:
            if final_gdf[col].dtype == 'object':
                is_list = final_gdf[col].fillna(0).apply(lambda x: isinstance(x, list))
                if is_list.any():
                    final_gdf[col] = final_gdf[col].astype(str)

        # Keep only essential columns that exist
        columns_to_keep = ['geometry', 'name', 'building']
        existing_cols_to_keep = [col for col in columns_to_keep if col in final_gdf.columns]
        final_gdf = final_gdf[existing_cols_to_keep]

        # Save the GeoPackage
        save_path.parent.mkdir(parents=True, exist_ok=True)
        final_gdf.to_file(save_path, driver="GPKG")
        print(f"Success: Downloaded and saved {len(final_gdf)} building footprints to {save_path}")

        return final_gdf

    except Exception as e:
        print(f"An error occurred in download_building_footprints: {e}")
        return gpd.GeoDataFrame()

def download_google_api_data(points_gdf, osm_id, config):
    """
    Request data from the Google API using the given parameters.
    """
    try:
        print("Getting Solar API Data")
        # Load API key from environment
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("Google API key not found. Ensure it is set in the environment or .env file.")

        os.environ["GOOGLE_API_KEY"] = google_api_key

        # Prepare request parameters
        save_dir = Path(config['output_dir']) / 'step2_solar_data' / osm_id
        save_dir.mkdir(parents=True, exist_ok=True)

        sample_point = points_gdf.sample(1)  # Random sample point
        radiusMeters = 500
        view = "IMAGERY_AND_ANNUAL_FLUX_LAYERS"
        requiredQuality = "HIGH"
        pixelSizeMeters = 0.5

        # Assuming `sapi` is already imported and configured
        req = sapi.request_data(
            points_gdf,
            radiusMeters,
            view,
            requiredQuality,
            pixelSizeMeters,
            save_dir,
            osm_id=osm_id,
        )

        print(f"Google API data saved to: {save_dir}")
        return req

    except Exception as e:
        print(f"Error during Google API data request: {e}")
        return None

def adaptive_grid_from_convex_hull(points_dataset, cell_size=950, buffer_distance=50):
    """
    Generate a grid of square cells that adaptively cover a buffered convex hull around a set of points.

    Parameters:
        points_dataset (GeoDataFrame): A GeoDataFrame of point geometries in a projected CRS.
        cell_size (float): The width and height of each square cell in meters. Default is 950.
        buffer_distance (float): Distance (in meters) to expand the convex hull. Default is 50.

    Returns:
        tuple:
            - buffered_hull (shapely.geometry.Polygon): The convex hull of the points expanded by the buffer distance.
            - grid_gdf (GeoDataFrame): A GeoDataFrame of square grid cells that intersect the buffered hull.
    """
    convex_hull = points_dataset.unary_union.convex_hull

    # Buffer the convex hulls by the desired amount (50 m)
    buffered_hull = convex_hull.buffer(buffer_distance)

    # Get the bounding box of the buffered hull (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = buffered_hull.bounds

    # Calculate the number of columns and rows needed, based on the fixed cell_size.
    n_cols = math.ceil((maxx - minx) / cell_size)
    n_rows = math.ceil((maxy - miny) / cell_size)

    cells = []

    # Special case: only one cell needed in either direction
    if n_cols == 1 and n_rows == 1:
        centroid = buffered_hull.centroid
        center_x, center_y = centroid.x, centroid.y

        cell = box(
            center_x - cell_size / 2, center_y - cell_size / 2,
            center_x + cell_size / 2, center_y + cell_size / 2
        )

        if cell.intersects(buffered_hull):
            cells.append(cell)
    else:
        for i in range(n_cols):
            for j in range(n_rows):
                # Create a cell that is exactly cell_size x cell_size.
                cell = box(minx + i * cell_size, miny + j * cell_size,
                           minx + (i + 1) * cell_size, miny + (j + 1) * cell_size)

                # Only include the cell if it intersects the buffered hull.
                if cell.intersects(buffered_hull):
                    cells.append(cell)

    # Create a GeoDataFrame for these cells.
    grid_gdf = gpd.GeoDataFrame(geometry=cells, crs=points_dataset.crs)

    return buffered_hull, grid_gdf
