import os
import argparse
import solar_api_utils as sapi
import datetime as dt

def main(date):
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process a OSM area.")
    
    # # Add the argument
    # parser.add_argument('number', type=int, help='OSMID to be processed')

    parser.add_argument('place', type=str, help='Placename to run through the geocoding service')

    # Parse the arguments
    args = parser.parse_args()

    # # Access the number argument
    # osmid = args.number

    # Access the place argument
    place = args.place

    # Get administrative boundaries of the place in question and download the buildings if necessary
    gdf, osm_id = sapi.get_admin_area(place)


    points_gdf = sapi.get_query_points(gdf, spacing=900,contains=False, solar_coverage=True, solar_coverage_path='../data/clean_data/solar/solar-api-coverage-032024/SolarAPIHighArea.shp', osm_id=osm_id)

    points_path = f'../data/clean_data/solar/{osm_id}/{osm_id}_query_points.gpkg'


    points_gdf.to_file(points_path, driver='GPKG')


    save_dir = '../data/clean_data/solar/{OSMID}'
    sample_point = points_gdf.sample(1)
    radiusMeters = 500
    view = "FULL_LAYERS" #instead of "FULL_LAYERS"
    requiredQuality = "HIGH" # instead of "HIGH"
    pixelSizeMeters  = 0.5 # instead of 0.25
    req = sapi.request_data(sample_point, radiusMeters, view, requiredQuality, pixelSizeMeters, save_dir, osm_id=osm_id)


current_date = dt.datetime.now()

if __name__ == "__main__":
    main(current_date)
