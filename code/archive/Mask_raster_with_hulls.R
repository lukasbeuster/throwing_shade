# Load required packages
library(rgdal)
library(raster)
library(sf)


# Load raster and vector data
raster_file <- "../data/clean_data/solar/2230_175_0_5/2022_06_02_monthlyFlux.tif"
vector_file <- "../../lidR/hulls_tolerance_2_5.gpkg"

# Define the CRS for the raster
raster_crs <- CRS("+proj=utm +zone=31 +datum=WGS84 +units=m +no_defs")

# Create a raster layer with the specified CRS
raster_data <- raster(raster_file, crs = raster_crs)

vector_data <- st_read(vector_file)


# Reproject the vector data to match the CRS of the raster
vector_data <- st_transform(vector_data, crs = raster_data@crs)

# Create a mask for the polygon overlap
mask <- rasterize(vector_data, raster_data, field = 1)

plot(mask)

# Set pixel values where polygons overlap to a specific value
specific_value <- your_specific_value
raster_data[mask == 1] <- specific_value

# Save the modified raster
writeRaster(raster_data, "output_raster.tif", format = "GTiff", overwrite = TRUE)