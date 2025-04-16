from deepforest import main
import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from segment_anything import SamPredictor, sam_model_registry
import torch
import matplotlib.pyplot as plt
import os

def tree_segment_main(osmid, sam_checkpoint, base_path):
    # Define the folder path and osmid
    folder_path = f"../data/clean_data/solar/{osmid}"

    # List files in the folder and count those containing 'dsm' in their name
    dsm_files = [f for f in os.listdir(folder_path) if 'dsm' in f.lower()]

    no_of_tiles = len(dsm_files)

    # Load the DeepForest model
    model = main.deepforest()
    model.use_release()

    for tile_no in range(no_of_tiles):
        # get the rgb raster for the tile number
        rgb_file = [f for f in os.listdir(folder_path) if ('rgb' in f.lower()) & (f"p_{tile_no}_" in f.lower())]

        # Load the image and predict bounding boxes using DeepForest
        raster_path = f"../data/clean_data/solar/{osmid}/" + rgb_file[0]

        image = cv2.imread(raster_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # predicted_mask = model.predict_tile(raster_path, return_plot=False, patch_size=300, patch_overlap=0.25)
        predicted_mask = model.predict_tile(raster_path, return_plot=False, patch_size=800, patch_overlap=0.25)

        # Load the SAM model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_type = "vit_h"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)

        # Set the image for segmentation
        predictor.set_image(image)

        # Create an empty mask with the same dimensions as the original image
        final_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Iterate through the detected objects and use SAM to segment them
        for i, prediction in predicted_mask.iterrows():
            x1, y1, x2, y2 = int(prediction['xmin']), int(prediction['ymin']), int(prediction['xmax']), int(prediction['ymax'])

            # Create a box prompt for SAM
            box = np.array([x1, y1, x2, y2])

            # Predict the mask using SAM
            masks, _, _ = predictor.predict(box=box)

            # Combine the SAM mask into the final mask
            for mask in masks:
                final_mask[mask] = 255  # Assuming binary segmentation, 255 for foreground

        # Define output path
        output_path = f"../data/clean_data/canopy_masks/{osmid}/" + rgb_file[0][:-4] + "_segmented.tif"

        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Open source raster to get metadata
        with rasterio.open(raster_path) as src:
            transform = src.transform
            crs = src.crs

        # Save the final mask as a GeoTIFF
        with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=final_mask.shape[0],
                width=final_mask.shape[1],
                count=1,
                dtype=final_mask.dtype,
                crs=crs,
                transform=transform,
        ) as dst:
            dst.write(final_mask, 1)

base_path = "C:/Users/Dila Ozberkman/Desktop/AMS Research/Urban Shade/throwing_shade"
sam_checkpoint = "../data/clean_data/solar/sam/sam_vit_h_4b8939.pth"
tree_segment_main('fff2ea05', sam_checkpoint, base_path)
