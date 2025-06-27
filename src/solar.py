import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box
from shapely.ops import unary_union
import osmnx as ox
import src.solar_api_utils as sapi
from dotenv import load_dotenv
import hashlib
import math
from . import utils

def check_coverage_logic(config):
    """
    Calculates required solar tiles without downloading.
    Saves a preview map and returns information to the user.
    """
    print("Checking solar tile coverage...")
    dataset = gpd.read_file(config['dataset_path'])

    # --- ADAPT YOUR EXISTING LOGIC HERE ---
    # Use the logic from solarAPI_main to determine which tiles are needed based on
    # config['solar_api']['min_points_per_tile'] and the dataset points.
    # Instead of downloading, just create a GeoDataFrame of the required tile boundaries.
    longitude_column = config['columns']['longitude']
    latitude_column =config['columns']['latitude']

    geometry = [Point(xy) for xy in zip(dataset[longitude_column], dataset[latitude_column])]
    points_dataset = gpd.GeoDataFrame(dataset, geometry=geometry, crs="EPSG:4326")

    # Assuming gdf is your GeoDataFrame in EPSG:4326
    points_dataset = points_dataset.to_crs("EPSG:32632")

    hull, grid_gdf = adaptive_grid_from_convex_hull(points_dataset, buffer_distance=50, cell_size=950)

    # Perform the spatial join: every row in grid_gdf is preserved, but cells with no intersecting points
    joined = gpd.sjoin(grid_gdf, points_dataset, how="left", predicate="intersects")

    # Filter out rows where "index_right" is NaN (i.e., no point was found)
    valid_joined = joined[joined["index_right"].notna()]

    # Use the indices from valid_joined to select the corresponding rows from grid_gdf
    valid_grid_gdf = grid_gdf.loc[valid_joined.index.unique()]

    # For example:
    # tile_count = len(required_tiles_gdf)

    # --- SAVE INTERMEDIATE RESULTS ---
    output_dir = Path(config['output_dir']) / 'step1_solar_coverage'
    output_dir.mkdir(parents=True, exist_ok=True)

    preview_path = output_dir / 'coverage_preview.geojson'
    # required_tiles_gdf.to_file(preview_path, driver='GeoJSON')

    # Also save the list of tile IDs needed for the next step
    # tile_ids = required_tiles_gdf['tile_id_column'].tolist()
    # with open(output_dir / 'tile_list.json', 'w') as f:
    #     json.dump(tile_ids, f)

    # --- Mocked output for demonstration ---
    tile_count = 50 # Replace with your actual calculated count
    print(f"Calculation complete. Found {tile_count} tiles.")

    return tile_count, preview_path

def solarAPI_main(dataset, latitude_column, longitude_column, solar_coverage_medium, solar_coverage_high, geometry=False):
    """
    Processes geographic data and interacts with SolarAPI to obtain solar potential information
    for a given set of points.

    This function performs the following tasks:
    1. Converts latitude and longitude columns into a GeoDataFrame with geometries.
    2. Transforms the coordinate reference system (CRS) to UTM (EPSG:32632).
    3. Generates an adaptive grid from the convex hull of the dataset.
    4. Performs a spatial join to filter relevant grid cells containing data points.
    5. Computes centroids of valid grid cells for API requests.
    6. Generates an OSM ID for the bounding box of the study area.
    7. Saves centroid points to a file.
    8. Downloads building footprints from OpenStreetMap.
    9. Loads and processes solar coverage areas.
    10. Checks if centroids fall within the solar coverage zones.
    11. Calls Google API to retrieve additional data.

    Args:
        dataset (pd.DataFrame): The input dataset containing latitude and longitude columns.
        latitude_column (str): Name of the column containing latitude values.
        longitude_column (str): Name of the column containing longitude values.
        solar_coverage_medium (str): File path to the medium solar coverage GeoDataFrame.
        solar_coverage_high (str): File path to the high solar coverage GeoDataFrame.

    Returns:
        None: The function primarily performs data processing, saves outputs,
        and interacts with external APIs.
    """
    if geometry:
        points_dataset = gpd.GeoDataFrame(dataset, geometry=geometry, crs="EPSG:4326")
    else:
        geometry = [Point(xy) for xy in zip(dataset[longitude_column], dataset[latitude_column])]
        points_dataset = gpd.GeoDataFrame(dataset, geometry=geometry, crs="EPSG:4326")

    # Assuming gdf is your GeoDataFrame in EPSG:4326
    points_dataset = points_dataset.to_crs("EPSG:32632")

    hull, grid_gdf = adaptive_grid_from_convex_hull(points_dataset, buffer_distance=50, cell_size=950)

    # Perform the spatial join: every row in grid_gdf is preserved, but cells with no intersecting points
    joined = gpd.sjoin(grid_gdf, points_dataset, how="left", predicate="intersects")

    # Filter out rows where "index_right" is NaN (i.e., no point was found)
    valid_joined = joined[joined["index_right"].notna()]

    # Use the indices from valid_joined to select the corresponding rows from grid_gdf
    valid_grid_gdf = grid_gdf.loc[valid_joined.index.unique()]

    print(f"Requesting {valid_grid_gdf.shape[0]} cells from SolarAPI")

    # Calculate the centroid for each grid cell
    valid_grid_gdf["centroid"] = valid_grid_gdf.geometry.centroid

    # Create a new GeoDataFrame for the centroids
    centroid_gdf = gpd.GeoDataFrame(valid_grid_gdf.drop(columns="geometry"),
                                    geometry="centroid",
                                    crs=valid_grid_gdf.crs)

    # Optionally, rename the geometry column to something standard (like 'geometry')
    centroid_gdf = centroid_gdf.rename(columns={"centroid": "geometry"})
    centroid_gdf.set_geometry("geometry", inplace=True)

    osm_id = generate_osm_id(hull.bounds)
    print(f"Generated OSM ID for dataset: {osm_id}")

    centroid_gdf = centroid_gdf.reset_index()
    centroid_gdf = centroid_gdf.drop(['index'], axis=1)
    centroid_gdf['osm_id'] = osm_id
    centroid_gdf['id'] = 'p_'+ centroid_gdf.index.astype(str)

    centroid_gdf = centroid_gdf.to_crs(4326)

    save_points(centroid_gdf, osm_id)

    gdf = gpd.GeoDataFrame(geometry=[hull], crs=centroid_gdf.crs)

    # Ensure CRS is defined
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame does not have a CRS defined.")

    # Define save path for building footprints
    save_path = f'../data/clean_data/solar/{osm_id}/{osm_id}_buildings.gpkg'

    # Reproject back to geographic CRS if needed
    if not gdf.crs.is_geographic:
        gdf = gdf.to_crs(4326)

    # Validating the input geometry
    if not gdf.is_valid.all():
        gdf = gdf.buffer(0)  # Fix invalid geometries
    if gdf.geometry.is_empty.any():
        print("The geometry for the region is empty or invalid.")

    all_buildings = download_building_footprints(gdf, save_path)

    # Load SolarAPIMediumArea and SolarAPIHighArea
    solar_coverage_medium = gpd.read_file(solar_coverage_medium)
    solar_coverage_high = gpd.read_file(solar_coverage_high)

    # Reproject both to match gdf's CRS
    solar_coverage_medium = solar_coverage_medium.to_crs(gdf.crs)
    solar_coverage_high = solar_coverage_high.to_crs(gdf.crs)

    # Generate the medium and high boundaries
    medium_boundary = solar_coverage_medium.geometry.union_all()
    print("Medium Union")
    high_boundary = solar_coverage_high.geometry.union_all()
    print("High Union")

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
        raise Exception("Not all points are within SolarAPI coverage")

    print("Solar API downloading...")
    # Call Google API for additional data
    api_response = download_google_api_data(centroid_gdf, osm_id)
    print("Solar API download completed")

    return osm_id

def generate_osm_id(bounds):
    """
    Generate a unique identifier for a bounding box using a hash function.
    """
    minx, miny, maxx, maxy = bounds
    bbox_str = f"{minx}_{miny}_{maxx}_{maxy}"
    return hashlib.md5(bbox_str.encode()).hexdigest()[:8]  # Generate an 8-character hash

def save_points(points_gdf, osm_id):
    """Save points GeoDataFrame to a GeoPackage."""
    save_dir = f'../data/clean_data/solar/{osm_id}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'{osm_id}_query_points.gpkg')
    points_gdf.to_file(save_path, driver='GPKG')
    print(f"Points saved to: {save_path}")

def download_building_footprints(gdf, save_path):
    """
    Download building footprints (polygons only) for the geometries in the GeoDataFrame.
    """
    try:
        if os.path.exists(save_path):
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
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        final_gdf.to_file(save_path, driver="GPKG")
        print(f"Success: Downloaded and saved {len(final_gdf)} building footprints to {save_path}")

        return final_gdf

    except Exception as e:
        print(f"An error occurred in download_building_footprints: {e}")
        return gpd.GeoDataFrame()

def download_google_api_data(points_gdf, osm_id):
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
        save_dir = f'../data/clean_data/solar/{osm_id}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

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

def convert_lists_to_strings(column):
    if column.dtype == 'object' and column.apply(lambda x: isinstance(x, list)).any():
        return column.apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else x)
    else:
        return column
