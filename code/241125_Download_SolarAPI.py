import os
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
import osmnx as ox
import matplotlib.pyplot as plt
import solar_api_utils as sapi
from dotenv import load_dotenv

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
            return

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

    except Exception as e:
        print(f"Error in download_building_footprints: {e}")

# def download_building_footprints(gdf, osm_id, save_path):
#     """
#     Download building footprints for the geometries in the GeoDataFrame if not already saved.
#     """
#     # Check if the file already exists
#     if os.path.exists(save_path):
#         print(f"Buildings already downloaded and saved at: {save_path}")
#         return

#     all_buildings = gpd.GeoDataFrame()  # Initialize an empty GeoDataFrame to hold all building footprints
#     tags = {"building": True}

#     print(len(gdf))

#     # Iterate over each polygon in the GeoDataFrame
#     for polygon in gdf.geometry:
#         if polygon.is_valid and isinstance(polygon, (Polygon, MultiPolygon)):
#             try:
#                 # Download building footprints for the current polygon
#                 buildings = ox.features_from_polygon(polygon, tags)
#                 print(buildings)
#                 # Convert lists in the GeoDataFrame to strings for saving
#                 buildings = buildings.apply(convert_lists_to_strings, axis=0)
#                 all_buildings = gpd.GeoDataFrame(pd.concat([all_buildings, buildings], ignore_index=True))
#             except Exception as e:
#                 print(f"Error processing polygon: {e}")

#     # Save the combined building footprints if any buildings were found
#     if not all_buildings.empty:
#         if not os.path.exists(os.path.dirname(save_path)):
#             os.makedirs(os.path.dirname(save_path))
#         all_buildings.to_file(save_path, driver='GPKG')
#         print(f"Success: Downloaded and saved {all_buildings.shape[0]} buildings.")
#     else:
#         print("No buildings found for the specified area.")

def main(place, spacing):
    try:
        # Geocode the place and get OSM data
        gdf = ox.geocoder.geocode_to_gdf(place)
        if gdf.empty:
            print(f"Error: No data found for place '{place}'.")
            sys.exit(1)

        osm_id = gdf.osm_id.loc[0]

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

        # Download building footprints
        download_building_footprints(gdf, osm_id, save_path)

        # Reproject to UTM if in geographic CRS
        if gdf.crs.is_geographic:
            bounds = gdf.bounds
            min_x, max_x = bounds['minx'].min(), bounds['maxx'].max()
            min_y, max_y = bounds['miny'].min(), bounds['maxy'].max()
            utm_crs = lat_lon_to_utm_epsg(min_x, min_y, max_x, max_y)
            gdf = gdf.to_crs(utm_crs)

        # Load SolarAPIMediumArea and SolarAPIHighArea
        solar_coverage_medium = gpd.read_file('../data/clean_data/solar/solar-api-coverage-032024/SolarAPIMediumArea.shp')
        solar_coverage_high = gpd.read_file('../data/clean_data/solar/solar-api-coverage-032024/SolarAPIHighArea.shp')

        # Reproject both to match gdf's CRS
        solar_coverage_medium = solar_coverage_medium.to_crs(gdf.crs)
        solar_coverage_high = solar_coverage_high.to_crs(gdf.crs)

        # Generate the medium and high boundaries
        medium_boundary = solar_coverage_medium.geometry.unary_union
        print("Medium Union")
        high_boundary = solar_coverage_high.geometry.unary_union
        print("High Union")
        # Attempt to generate points using the medium boundary first
        points_gdf = create_points_geodataframe(gdf, spacing, boundary=medium_boundary)

        # If no points were generated, fallback to the high boundary
        if points_gdf.empty:
            print("No points generated within the medium area boundary. Switching to high area boundary.")
            points_gdf = create_points_geodataframe(gdf, spacing, boundary=high_boundary)

        if points_gdf.empty:
            print("No points could be generated in both medium and high area boundaries.")
            sys.exit(1)

        # Save the points GeoDataFrame
        save_points(points_gdf, osm_id)

        if not points_gdf.crs.is_geographic:
            points_gdf = points_gdf.to_crs(4326)

         # Call Google API for additional data
        api_response = download_google_api_data(points_gdf, osm_id)


        # # Plot results
        # points_gdf.plot()
        # gdf.boundary.plot(ax=plt.gca(), color='red')
        # plt.show()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def download_google_api_data(points_gdf, osm_id):
    """
    Request data from the Google API using the given parameters.
    """
    try:
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

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <place> <spacing>")
        sys.exit(1)

    place = sys.argv[1]
    try:
        spacing = float(sys.argv[2])
    except ValueError:
        print("Error: Spacing must be a numeric value.")
        sys.exit(1)

    main(place, spacing)