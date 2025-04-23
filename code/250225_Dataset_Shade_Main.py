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

# DST start and end dates (adjust per year)
dst_start = datetime(2022, 3, 27)
dst_end = datetime(2022, 10, 30)

# Solstice day for temporal binning
solstice_day = pd.to_datetime("2022-06-21")

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

# === 3. Run Pipeline ===

dataset = gpd.read_file(dataset_path)

# 1. Download Solar API data (requires solar shapefiles and lat/lon)
osmid = solarAPI_main(
    dataset,
    latitude_column,
    longitude_column,
    solar_coverage_medium,
    solar_coverage_high,
    geometry=False
)

# 2. Run tree segmentation using SAM model
tree_segment_main(osmid, sam_checkpoint)

# 3. Shade simulation
raster_dir = f"../data/clean_data/solar/{osmid}"

dataset_final = main(
    dataset_path,
    osmid,
    unique_ID_column,
    raster_dir,
    solstice_day,
    longitude_column,
    latitude_column,
    timestamp_column_name,
    dst_start,
    dst_end,
    output_path,
    summer_params,
    winter_params,
    combined_sh=combined_sh,
    building_sh=building_sh,
    interval=sh_int,
    geometry=False,
    crs="EPSG:4326",
    simulate_solstice=False,
    bin_size=0,
    parameters=parameters
)
