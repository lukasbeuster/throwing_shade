import geopandas as gpd
import download_solarapi_custom
import tree_segmentation.py

dataset = gpd.read_file("C:/Users/Dila Ozberkman/Desktop/AMS Research/Urban Shade/Data/gertjandatafiets.csv")

latitude_column = "latitude"
longitude_column = "longitude"
timestamp_column_name = "TIMESTAMP"
unique_ID_column = "RECORD"

solar_coverage_medium = "C:/Users/Dila Ozberkman/Desktop/AMS Research/Urban Shade/Data/solar-api-coverage/SolarAPIMediumArea.shp"
solar_coverage_high = "C:/Users/Dila Ozberkman/Desktop/AMS Research/Urban Shade/Data/solar-api-coverage/SolarAPIHighArea.shp"
sam_checkpoint = "C:/Users/Dila Ozberkman/Desktop/AMS Research/Urban Shade/throwing_shade/data/clean_data/solar/sam/sam_vit_h_4b8939.pth"
base_path = "C:/Users/Dila Ozberkman/Desktop/AMS Research/Urban Shade/throwing_shade/"

osmid = solarAPI_main(dataset, latitude_column, longitude_column, solar_coverage_medium, solar_coverage_high)
tree_segment_main(osmid, sam_checkpoint, base_path)

# input shade parameters
summer_params = {'utc':2, 'dst':0, 'trs':10}
winter_params = {'utc':1, 'dst':0, 'trs':45}

# input desired shade calculation interval
sh_int = 30

# directory for dsm data
raster_dir = f'../data/clean_data/solar/{osmid}'
