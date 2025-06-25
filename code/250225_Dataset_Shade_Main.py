import geopandas as gpd
import pandas as pd
from datetime import datetime
from Download_SolarAPI_Custom import solarAPI_main
from tree_segmentation import tree_segment_main
from process_shade import main

# === 1. Required Input Files ===
# Path to the input dataset (must be a GeoJSON or shapefile with latitude & longitude columns)
dataset_path = "xx"

# Column names in your dataset
latitude_column = "latitude"
longitude_column = "longitude"
timestamp_column_name = "TIME"  # Format must be compatible with `pd.to_datetime()`
unique_ID_column = "ID"   # A column that uniquely identifies each row or trace

# Solar API coverage shapefiles (already downloaded locally)
solar_coverage_medium = "C:/Users/xx/solar-api-coverage/SolarAPIMediumArea.shp"
solar_coverage_high = "C:/Users/xx/solar-api-coverage/SolarAPIHighArea.shp"

# Path to SAM tree segmentation model checkpoint
sam_checkpoint = "C:/Users/xx/throwing_shade/data/clean_data/solar/sam/sam_vit_h_4b8939.pth"

# Path to save final output
output_path = "C:/Users/xx/throwing_shade/output/final_with_shade.geojson"

# === 2. Shade Simulation Parameters ===

# Tree transmissivity & UTC offset for summer (DST) and winter (non-DST)
# DON'T CHANGE DST HERE: only change the UTC for your dataset
summer_params = {'utc': 1, 'dst': 0, 'trs': 10}
winter_params = {'utc': 0, 'dst': 0, 'trs': 45}

# input the date daylight savings start and ends for years, in dictionary example below
year_configs = {
    2021: {
        "solstice_day": datetime(2021, 6, 21),
        "dst_start": datetime(2021, 3, 28).date(),
        "dst_end": datetime(2021, 10, 31).date()
        },
    2022: {
        "solstice_day": datetime(2022, 6, 21),
        "dst_start": datetime(2022, 3, 27).date(),
        "dst_end": datetime(2022, 10, 30).date()
    }}

# Shade interval in minutes
sh_int = 30

# Simulation type
building_sh = False
combined_sh = True

# Additional outputs to include
parameters = {
    'building_shade_step': False,
    'tree_shade_step': False,
    'bldg_shadow_fraction': False,
    'tree_shadow_fraction': False,
    'hours_before': []
}

# Input buffer values to calculate
buffer = [0, 1, 2, 4]

# If you want to bin data based on dates, enter an integer bin size > 1
bin_size = 0

# If you want to simulate solstice even if not in bin, put True for simulate_solstice
simulate_solstice = True

# If you don't want to simulate from sunrise to timestamp, can set a
# start_time in the form '11:00', if not can put None
start_time = '11:00'

# input save path for the final dataset with shade values
output_path = f"results/output/osmid/xx.geojson"

# === 3. Run Pipeline ===

dataset = gpd.read_file(dataset_path)

# 1. Download Solar API data (requires solar shapefiles and lat/lon)
osmid = solarAPI_main(
    dataset,
    latitude_column,
    longitude_column,
    solar_coverage_medium,
    solar_coverage_high,
)

# 2. Run tree segmentation using SAM model
tree_segment_main(osmid, sam_checkpoint)

# 3. Shade simulation
raster_dir = f"../data/clean_data/solar/{osmid}"

dataset_final = main(
    dataset,
    osmid,
    unique_ID_column,
    raster_dir,
    year_configs,
    longitude_column,
    latitude_column,
    timestamp_column_name,
    output_path,
    summer_params,
    winter_params,
    combined_sh=combined_sh,
    building_sh=building_sh,
    interval=sh_int,
    start_time=start_time,
    crs="EPSG:4326",
    simulate_solstice=simulate_solstice,
    bin_size=bin_size,
    parameters=parameters,
    buffer=buffer,
    save=True)
