from deepforest import main
import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin
from segment_anything import SamPredictor, sam_model_registry
import torch
from pathlib import Path

def run_segmentation(config, osmid):
    # Define the folder path and osmid
    folder_path = Path(config['output_dir']) / f"step2_solar_data/{osmid}"

    # Load the DeepForest model
    model = main.deepforest()
    model.use_release()

    # Load SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=config['dependencies']['sam_checkpoint'])
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # List files in the folder and count those containing 'dsm' in their name
    rgb_files = sorted(list(folder_path.glob("*_rgb.tif")))
    no_of_tiles = len(rgb_files)

    for tile_no in range(no_of_tiles):
        # get the rgb raster for the tile number
        rgb_file = [p for p in rgb_files if f"p_{tile_no}_" in p.lower()]

        # Load the image and predict bounding boxes using DeepForest
        raster_path = folder_path / rgb_file[0]

        image = cv2.imread(str(raster_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # predicted_mask = model.predict_tile(raster_path, return_plot=False, patch_size=300, patch_overlap=0.25)
        predicted_mask = model.predict_tile(raster_path, return_plot=False, patch_size=800, patch_overlap=0.25)

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
        output_path = Path(config['output_dir']) / f"step3_segmented_trees/{osmid}/{rgb_file[0][:-4]}_segmented.tif"

        # Ensure the directory exists before saving
        output_path.parent.mkdir(parents=True, exist_ok=True)

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
