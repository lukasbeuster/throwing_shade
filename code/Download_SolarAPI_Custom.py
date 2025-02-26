import os
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import unary_union
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
import osmnx as ox
import matplotlib.pyplot as plt
import solar_api_utils as sapi
from dotenv import load_dotenv
import hashlib
from IPython.display import display
import math

def solarAPI_main(dataset, latitude_column, longitude_column, solar_coverage_medium, solar_coverage_high):
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
    print(f"Generated OSM ID for custom bounding box: {osm_id}")

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

    all_buildings = download_building_footprints(gdf, osm_id, save_path)
    all_buildings_filtered = all_buildings[all_buildings.geom_type != "Point"]

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
    print(f"All points within SolarAPI coverage: {all_points_valid}")

    # Call Google API for additional data
    api_response = download_google_api_data(centroid_gdf, osm_id)
    print("Solar API download completed")

    return osm_id

def create_bounding_box_gdf(minx, miny, maxx, maxy, crs="EPSG:4326"):
    """Create a GeoDataFrame with a bounding box polygon."""
    bbox_polygon = box(minx, miny, maxx, maxy)  # Create a rectangle
    bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox_polygon]}, crs=crs)
    print("Created bounding box")
    return bbox_gdf

def generate_osm_id(bounds):
    """
    Generate a unique identifier for a bounding box using a hash function.
    """
    minx, miny, maxx, maxy = bounds
    bbox_str = f"{minx}_{miny}_{maxx}_{maxy}"
    return hashlib.md5(bbox_str.encode()).hexdigest()[:8]  # Generate an 8-character hash

def lat_lon_to_utm_epsg(min_x, min_y, max_x, max_y):
    """Convert latitude and longitude coordinates to the corresponding UTM projection CRS."""
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=min_x,
            south_lat_degree=min_y,
            east_lon_degree=max_x,
            north_lat_degree=max_y,
        ),
    )
    return CRS.from_epsg(utm_crs_list[0].code)

def generate_points_within_polygon(polygon, spacing, min_distance_boundary=None):
    """
    Generate points within a polygon with a specified spacing. Optionally filter points
    that are too close to a boundary defined by another geometry.
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    x_coords = np.arange(min_x, max_x, spacing)
    y_coords = np.arange(min_y, max_y, spacing)

    points = [
        Point(x, y)
        for x in x_coords
        for y in y_coords
        if polygon.contains(Point(x, y))
    ]

    if min_distance_boundary is not None:
        boundary_buffer = min_distance_boundary.buffer(-spacing)
        points = [point for point in points if boundary_buffer.contains(point)]

    return points

def create_points_geodataframe(gdf, spacing, boundary=None):
    """
    Create a GeoDataFrame of points generated within geometries of an input GeoDataFrame.
    Optionally exclude points too close to a boundary defined by another GeoDataFrame.
    """
    all_points, point_ids, osm_ids = [], [], []
    point_id_counter = 1

    for _, row in gdf.iterrows():
        geom = row.geometry
        osm_id = row.osm_id

        if geom.geom_type == 'Polygon':
            points = generate_points_within_polygon(geom, spacing, boundary)
        elif geom.geom_type == 'MultiPolygon':
            points = [
                pt for poly in geom.geoms
                for pt in generate_points_within_polygon(poly, spacing, boundary)
            ]
        else:
            continue

        all_points.extend(points)
        point_ids.extend([f"p_{point_id_counter + i}" for i in range(len(points))])
        osm_ids.extend([osm_id] * len(points))
        point_id_counter += len(points)

    print("Creating points within geometry")

    return gpd.GeoDataFrame({'geometry': all_points, 'id': point_ids, 'osm_id': osm_ids}, crs=gdf.crs)

def save_points(points_gdf, osm_id):
    """Save points GeoDataFrame to a GeoPackage."""
    save_dir = f'../data/clean_data/solar/{osm_id}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'{osm_id}_query_points.gpkg')
    points_gdf.to_file(save_path, driver='GPKG')
    print(f"Points saved to: {save_path}")

def convert_lists_to_strings(column):
    """
    Convert list elements in a column to comma-separated strings for saving in GeoPackage.
    """
    if column.dtype == 'object' and column.apply(lambda x: isinstance(x, list)).any():
        return column.apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else x)
    else:
        return column


def download_building_footprints(gdf, osm_id, save_path):
    """
    Download building footprints for the geometries in the GeoDataFrame if not already saved.
    """
    try:
        # Check if file already exists
        if os.path.exists(save_path):
            print(f"Buildings already downloaded and saved at: {save_path}")
            return gpd.read_file(save_path)

        all_buildings = gpd.GeoDataFrame()  # Initialize an empty GeoDataFrame
        tags = {"building": True}

        for polygon in gdf.geometry:
            if polygon.is_valid and not polygon.is_empty:
                try:
                    # Query OSM buildings
                    buildings = ox.features_from_polygon(polygon, tags)
                    if buildings.empty:
                        print("No buildings found for the given polygon.")
                        continue

                    # Convert lists to strings for saving
                    buildings = buildings.apply(convert_lists_to_strings, axis=0)
                    all_buildings = gpd.GeoDataFrame(pd.concat([all_buildings, buildings], ignore_index=True))
                except Exception as e:
                    print(f"Error querying buildings for polygon: {e}")

        # Remove duplicate columns
        duplicate_columns = all_buildings.columns[all_buildings.columns.duplicated()]
        if not duplicate_columns.empty:
            print(f"Duplicate columns found: {duplicate_columns}")
            all_buildings = all_buildings.rename(columns=lambda x: f"{x}_dup" if x in duplicate_columns else x)

        # Keep only essential columns
        columns_to_keep = ['geometry', 'name', 'building']
        all_buildings = all_buildings[columns_to_keep]

        # Save the GeoPackage if any buildings were found
        if not all_buildings.empty:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            all_buildings.to_file(save_path, driver="GPKG")
            print(f"Success: Downloaded and saved {all_buildings.shape[0]} buildings.")
        else:
            print("No buildings found for the specified region.")

        return all_buildings

    except Exception as e:
        print(f"Error in download_building_footprints: {e}")

def dissolve_to_singlepolygon(geometries, crs="EPSG:32632"):
    """
    Dissolve a collection of geometries into a GeoDataFrame containing individual polygon features.

    Parameters:
        geometries (GeoSeries or iterable of shapely geometries):
            The input geometries to be dissolved.
        crs (str, optional):
            The coordinate reference system for the output GeoDataFrame.
            Default is "EPSG:32632".
    Returns:
        GeoDataFrame:
            A GeoDataFrame where the 'geometry' column contains one or more Polygon features
            resulting from the dissolution of the input geometries, with the specified CRS.
    """
    dissolved = unary_union(geometries)
    # If dissolved is a MultiPolygon, split it into separate polygons:
    if dissolved.geom_type == "MultiPolygon":
        final_geo = list(dissolved.geoms)
    else:
        final_geo = [dissolved]
    return gpd.GeoDataFrame(geometry=final_geo, crs=crs)

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
    Create a grid of non-overlapping cells that minimally covers the points

    Instead of using the axis-aligned bounding box, this function computes
    the convex hull of the points, buffers it by buffer_distance, and then generates a grid
    covering the hull's bounding box. Finally, only cells that intersect the buffered hull are kept.

    Parameters:
        cluster_gdf (GeoDataFrame): GeoDataFrame containing the points of a cluster (in a projected CRS).
        cell_size (float): Desired side length of each grid cell (in meters), e.g., 950.
        buffer_distance (float): Distance to buffer the convex hull (in meters), e.g., 50.

    Returns:
        grid_gdf (GeoDataFrame): GeoDataFrame of grid cells (non-overlapping) that cover the buffered convex hull.
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
