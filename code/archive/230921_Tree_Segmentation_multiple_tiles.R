#To-do:
#  Using gadm administrative areas in NL, identify tiles necessary from geotiles
#  Then sequentially download the tiles into memory, load, filter them and process them
#  Save the results, clear memory, continue until you run out of tiles.

require(sf)
require(dplyr)
require(curl)
require(lidR)
require(rlas)
require(rgdal)
require(tictoc)
require(raster)
require(stringr)
# function:

# Define your custom function
download_and_process_tiles <- function(tile_list) {
  # Define the base URL template
  base_url <- "https://geotiles.nl/AHN4_T/tile_name.LAZ"
  save_directory <- "results/tile_name.gpkg"
  # Define the file path where you want to save the CHM
  chm_file_directory <- "results/chm_file.tif"
  
  classification_filter <- 1
  elevation_filter_min <- 2
  elevation_filter_max <- 40
  
  # Loop through the list of tile names

  for (tile_name in tile_list) {
    tryCatch({
      # Construct the complete URL for the current tile
      url <- str_replace(base_url, "tile_name", tile_name)
      
      # Check if the file already exists in the results folder
      save_path <- str_replace(save_directory, "tile_name", tile_name)
      chm_file_path <- str_replace(chm_file_directory, "chm_file", tile_name)
      
      if (file.exists(save_path)) {
        cat("Skipping", tile_name, "as the file already exists.\n")
        next  # Skip to the next iteration of the loop
      }
      cat("Downloading", tile_name,"\n")
      # Download the file into memory
      response <- tryCatch({
        curl::curl_fetch_memory(url)
      }, error = function(e) {
        return(NULL)
      })
      
      # Check if the download was successful
      if (!is.null(response) && !is.null(response$content)) {
        # Save the downloaded content to a temporary file
        temp_file <- tempfile(fileext = ".laz")
        writeBin(response$content, temp_file)
        # Read and process the downloaded file
        las <- readLAS(temp_file,filter="-keep_class 1 2 9")
        
        las <- normalize_height(las, tin())
        
        # try performance difference with this:
        las <- filter_poi(las, !(Classification == classification_filter & Z <= elevation_filter_min) & !(Classification == classification_filter & Z >= elevation_filter_max))
        
        las_denoised <- filter_noise(las, sensitivity = 1.2)
        # Generate canopy height model
        chm <- rasterize_canopy(las_denoised, res = 0.25, algorithm = p2r(subcircle = 0.2))
        #chm <- grid_canopy(las_denoised, 0.25, pitfree(c(0,5,10,20,30,40), c(0,1), subcircle = 0.2))
        # Fill empty raster cells with 0 (ground)
        chm[is.na(chm)] <- 0
        
        # Smoothing of the raster, make 2 passes
        w <- matrix(1, 3, 3)
        chm <- terra::focal(chm, w, fun = mean, na.rm = TRUE)
        chm <- terra::focal(chm, w, fun = mean, na.rm = TRUE)
        
        # Segment individual trees
        algo <- watershed(chm, th = 3, tol = 2.5, ext = 1)
        las_watershed  <- segment_trees(las_denoised, algo)
        
        # remove points that are not assigned to a tree
        trees <- filter_poi(las_watershed, !is.na(treeID))
        
        # Use the writeRaster function to save the CHM
        terra::writeRaster(chm, chm_file_path, overwrite=TRUE)
        
        # question is, do I save the pointcloud with trees as well?
        # View the results
        hulls  <- delineate_crowns(trees, type = "concave", concavity = 2, func = .stdmetrics)
        
        writeOGR(hulls, save_path, layer = "hulls", driver = "GPKG")
       
        unlink(temp_file)
      } else {
      # Handle the case when the file couldn't be downloaded
        cat("Failed to download:", tile_name, "\n")
      }
    }, error = function(e) {
      # Handle errors by printing a message and continuing to the next iteration
      cat("Error processing", tile_name, ":", conditionMessage(e), "\n")
    })
  }
}

# Create a filter to remove points above 95th percentile of height
filter_noise = function(las, sensitivity)
{
  p95 <- grid_metrics(las, ~quantile(Z, probs = 0.95), 10)
  las <- merge_spatial(las, p95, "p95")
  las <- filter_poi(las, Z < p95*sensitivity)
  las$p95 <- NULL
  return(las)
}

# Specify the URL of the GeoPackage file
admin_url <- "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_NLD.gpkg"

# Download and load the GeoPackage
admin_areas <- st_read(admin_url, layer = 'ADM_ADM_2')

# Assuming you have an attribute column named 'polygon_id'
target_area <- 'Amsterdam'  # Change this to the desired polygon ID

selected_area <- admin_areas %>% 
  filter(NAME_2 == target_area)

# PART 2: 
# Specify the URL of the zipped shapefile
shapefile_zip_url <- "https://static.fwrite.org/2023/01/AHN_subunits_GeoTiles.zip"

# Define a temporary directory to store the downloaded file
temp_dir <- tempdir()

# Download the zipped shapefile
download.file(shapefile_zip_url, destfile = file.path(temp_dir, "AHN_subunits_GeoTiles.zip"))

# Unzip the shapefile
unzip(file.path(temp_dir, "AHN_subunits_GeoTiles.zip"), exdir = temp_dir)

# List the shapefile files in the unzipped directory
geotiles_files <- list.files(temp_dir, pattern = ".shp$", full.names = TRUE)

# Load the shapefile
geotiles <- st_read(geotiles_files)

# Assuming you want to reproject geotiles to match the CRS of selected_area
#geotiles <- st_transform(geotiles, crs = st_crs(selected_area))

selected_area <- st_transform(selected_area, crs = st_crs(geotiles))


# Identify which tiles are in the area

contained_polygons <- st_intersection(geotiles, selected_area)

# Extract identifiers or any other attributes you need
tile_list <- contained_polygons$GT_AHNSUB  # Replace 'identifier_column' with the actual column name

# Select random tiles for testing
selected_tiles <- sample(tile_list, 1)

sorted_list <- sort(tile_list)

# Download and process tiles:
download_and_process_tiles(sorted_list)


