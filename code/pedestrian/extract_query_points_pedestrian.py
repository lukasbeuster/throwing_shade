import gpxpy
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box
from shapely.ops import unary_union
import osmnx as ox
import src.solar_api_downloader as sapi
from dotenv import load_dotenv
import hashlib
from IPython.display import display
import math
from tqdm import tqdm

def solarapi_request(dataset, latitude_column, longitude_column, solar_coverage_medium, solar_coverage_high, geometry=False):
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
        points_dataset = dataset.copy()
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

    all_buildings = download_building_footprints(gdf, osm_id, save_path)

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

# --- [main execution script] ---
def main():
    base_folder = '/mnt/SCL-NAS/breeze/data_190416'
    # Folder paths
    city_file_paths = {
        'boston': {
            'matched': f'{base_folder}/gpx-matched-bos-252570',
            'shortest': f'{base_folder}/gpx-shortest-bos-252570'
        },
        'sf': {
            'matched': f'{base_folder}/gpx-matched-sf-280623',
            'shortest': f'{base_folder}/gpx-shortest-sf-280623'
        }
    }

    # Storage for final results
    city_gdfs = {}

    for city in ['boston', 'sf']:
        matched_gpx_folder = city_file_paths[city]['matched']

        # List all GPX files in the folder
        gpx_files = [os.path.join(matched_gpx_folder, f) for f in os.listdir(matched_gpx_folder) if f.endswith('.gpx')]

        all_track_points = []

        print(f"Processing {len(gpx_files)} GPX files for {city}...")

        for gpx_path in tqdm(gpx_files):
            try:
                # Load the GPX file
                with open(gpx_path, 'r') as gpx_file:
                    gpx = gpxpy.parse(gpx_file)

                # Extract filename ID
                filename = os.path.basename(gpx_path)
                file_id = filename.split('-')[0]

                # Extract track points
                i = 0
                for track in gpx.tracks:
                    for segment in track.segments:
                        for point in segment.points:
                            all_track_points.append({
                                'path_id': file_id,
                                'point_id': i,
                                'latitude': point.latitude,
                                'longitude': point.longitude,
                                'time': point.time
                            })
                            i += 1

            except Exception as e:
                print(f"⚠️ Failed to process {gpx_path}: {e}")

        # Create a GeoDataFrame
        df = pd.DataFrame(all_track_points)
        if not df.empty:
            df['geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
            gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        else:
            gdf = gpd.GeoDataFrame(columns=['path_id', 'point_id', 'latitude', 'longitude', 'time', 'geometry'], crs='EPSG:4326')

        # Save for each city
        city_gdfs[city] = gdf

    print("✅ Finished creating GeoDataframe for each city.")

    # Now call your main solar API processing
    for city, dataset in city_gdfs.items():
        osmid = solarapi_request(
            dataset,
            'latitude', 'longitude',
            'C:/Users/Dila Ozberkman/Desktop/AMS Research/Urban Shade/Data/solar-api-coverage/SolarAPIMediumArea.shp',
            'C:/Users/Dila Ozberkman/Desktop/AMS Research/Urban Shade/Data/solar-api-coverage/SolarAPIHighArea.shp',
            geometry=True
        )
        print(f"Download for {city} with osmid {osmid} is done!")

# --- [Python main entry point] ---
if __name__ == "__main__":
    main()
