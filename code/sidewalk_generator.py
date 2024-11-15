import os
import json
import requests
from time import sleep
from geopy.geocoders import Nominatim
import osm2streets_python
import geopandas as gpd
import pandas as pd
import argparse


def initialize_geolocator(user_agent="osm2streets_python/0.1.0"):
    """
    Initialize the geolocator for querying location data.
    
    Args:
        user_agent (str): User agent string for the Nominatim geolocator.

    Returns:
        geopy.geocoders.Nominatim: Initialized geolocator instance.
    """
    return Nominatim(user_agent=user_agent)


def get_location_info(geolocator, location_name):
    """
    Retrieve location information, including bounding box and OSMID, for a given location name.

    Args:
        geolocator (geopy.geocoders.Nominatim): Geolocator instance.
        location_name (str): Name of the location to query.

    Returns:
        tuple: OSMID (int) and bounding box (tuple of floats).

    Raises:
        ValueError: If the location or OSMID cannot be found.
    """
    location = geolocator.geocode(location_name)
    if location:
        print(f"Location found: {location.address}")
        osmid = location.raw.get("osm_id", None)
        if not osmid:
            raise ValueError("OSMID not found for location.")
        bbox = (
            float(location.raw['boundingbox'][0]),
            float(location.raw['boundingbox'][1]),
            float(location.raw['boundingbox'][2]),
            float(location.raw['boundingbox'][3]),
        )
        return osmid, bbox
    else:
        raise ValueError(f"Location not found: {location_name}")


def download_tiles(osmid, bbox, tile_size, output_dir, overpass_url="https://lz4.overpass-api.de/api/interpreter"):
    """
    Download OSM tiles within the specified bounding box, divided into smaller tiles.

    Args:
        osmid (int): OpenStreetMap ID for the location.
        bbox (tuple): Bounding box (min_lat, max_lat, min_lon, max_lon).
        tile_size (float): Size of each tile in degrees.
        output_dir (str): Directory to save the downloaded tiles.
        overpass_url (str): URL of the Overpass API.

        Alternative overpass_urls: 
        https://z.overpass-api.de/api/interpreter
        http://overpass-api.de/api/interpreter

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    min_lat, max_lat, min_lon, max_lon = bbox
    lat_steps = int((max_lat - min_lat) / tile_size) + 1
    lon_steps = int((max_lon - min_lon) / tile_size) + 1
    print(f"Creating {lat_steps * lon_steps} tiles...")

    if len(os.listdir(output_dir)) != (lat_steps * lon_steps):
        tile_count = 0
        for i in range(lat_steps):
            for j in range(lon_steps):
                tile_min_lat = min_lat + i * tile_size
                tile_max_lat = min(tile_min_lat + tile_size, max_lat)
                tile_min_lon = min_lon + j * tile_size
                tile_max_lon = min(tile_min_lon + tile_size, max_lon)

                bbox = (tile_min_lat, tile_min_lon, tile_max_lat, tile_max_lon)
                ways_and_nodes_query = f"""
                [out:xml];
                (
                way["highway"]["area"!~"yes"]["highway"!~"abandoned|construction|no|planned|platform|proposed|raceway|razed"]
                ({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]});
                >;);
                out;
                """

                sleep(1)  # Add delay between requests
                headers = {"User-Agent": "osm2streets_python/0.1.0"}
                response = requests.post(overpass_url, data=ways_and_nodes_query, headers=headers)
                if response.status_code == 200:
                    print(f"Tile {tile_count + 1}/{lat_steps * lon_steps} fetched successfully!")
                    tile_file = os.path.join(output_dir, f"{osmid}_tile_{tile_count}.osm")
                    with open(tile_file, "w") as f:
                        f.write(response.text)
                    tile_count += 1
                else:
                    print(f"Error fetching tile {tile_count + 1}: {response.status_code}")
                sleep(1)  # Add delay between requests
        print(f"Downloaded {tile_count} tiles to {output_dir}.")
    else:
        print(f"Files for {osmid} are already tiled and downloaded. Skipping download for now")


def process_tiles(tile_dir, input_options, output_dir):
    """
    Process downloaded OSM tiles to generate combined GeoDataFrames.

    Args:
        tile_dir (str): Directory containing OSM tiles.
        input_options (dict): Input options for osm2streets.
        output_dir (str): Directory to save the processed GeoDataFrames.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    combined_gdf = gpd.GeoDataFrame()
    combined_gdf_lanes = gpd.GeoDataFrame()
    combined_gdf_intersections = gpd.GeoDataFrame()

    for tile_file in os.listdir(tile_dir):
        if tile_file.endswith(".osm"):
            tile_path = os.path.join(tile_dir, tile_file)
            print(f"Processing tile: {tile_file}")
            with open(tile_path, "rb") as file:
                osm_input = file.read()

            try:
                network = osm2streets_python.PyStreetNetwork(osm_input, "", json.dumps(input_options))
                gdf = gpd.GeoDataFrame.from_features(json.loads(network.to_geojson_plain())["features"], crs=4326)
                gdf_lanes = gpd.GeoDataFrame.from_features(json.loads(network.to_lane_polygons_geojson())["features"], crs=4326)
                gdf_intersections = gpd.GeoDataFrame.from_features(json.loads(network.to_intersection_markings_geojson())["features"], crs=4326)

                combined_gdf = gpd.GeoDataFrame(pd.concat([combined_gdf, gdf], ignore_index=True))
                combined_gdf_lanes = gpd.GeoDataFrame(pd.concat([combined_gdf_lanes, gdf_lanes], ignore_index=True))
                combined_gdf_intersections = gpd.GeoDataFrame(pd.concat([combined_gdf_intersections, gdf_intersections], ignore_index=True))
            except Exception as e:
                print(f"Error processing tile {tile_file}: {e}")

    combined_gdf.to_file(os.path.join(output_dir, "combined_network.geojson"), driver="GeoJSON")
    combined_gdf_lanes.to_file(os.path.join(output_dir, "combined_lanes.geojson"), driver="GeoJSON")
    combined_gdf_intersections.to_file(os.path.join(output_dir, "combined_intersections.geojson"), driver="GeoJSON")
    print(f"Processed data saved to {output_dir}.")


def main(location_name, tile_size, driving_side):
    """
    Main function to orchestrate tile downloading and processing for a given location.

    Args:
        location_name (str): Name of the location to process.
        tile_size (float): Size of each tile in degrees.
        driving_side (str): Driving side ('Right' or 'Left').

    Returns:
        None
    """
    os.environ["RUST_LOG"] = "off"
    geolocator = initialize_geolocator()

    try:
        osmid, bbox = get_location_info(geolocator, location_name)
        base_dir = f"../data/raw_data/osm2streets/{osmid}"
        tile_dir = os.path.join(base_dir, "tiles")
        processed_dir = os.path.join(base_dir, "processed")

        # Step 1: Download Tiles
        download_tiles(osmid, bbox, tile_size, tile_dir)

        # Step 2: Process Tiles
        input_options = {
            "debug_each_step": False,
            "dual_carriageway_experiment": False,
            "sidepath_zipping_experiment": False,
            "inferred_sidewalks": True,
            "inferred_kerbs": True,
            "date_time": None,
            "override_driving_side": driving_side
        }
        process_tiles(tile_dir, input_options, processed_dir)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process OSM data for a given location.")
    parser.add_argument("location", type=str, help="Name of the location to process (e.g., 'West, Amsterdam').")
    parser.add_argument("--tile_size", type=float, default=0.01, help="Tile size in degrees (default: 0.01).")
    parser.add_argument("--driving_side", type=str, default="Right", choices=["Right", "Left"],
                        help="Driving side ('Right' or 'Left', default: 'Right').")
    args = parser.parse_args()
    main(args.location, args.tile_size, args.driving_side)