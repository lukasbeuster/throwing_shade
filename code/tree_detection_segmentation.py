import os
import glob
import argparse
from deepforest import main
import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin
from segment_anything import SamPredictor, sam_model_registry
import torch
import matplotlib.pyplot as plt

# Limit threads to 32
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
torch.set_num_threads(16)
cv2.setNumThreads(16)

def process_raster_files(osmid, raster_dir, output_dir, sam_checkpoint):
    # Load the DeepForest model
    model = main.deepforest()
    model.use_release()
    # model.config["workers"] = 8  # Adjust this based on your system

    # Get a list of all raster files in the directory
    raster_files = glob.glob(os.path.join(raster_dir, '*rgb.tif'))
    print(f'Found {len(raster_files)} tiles')

    # Load the SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    for raster_path in raster_files:
            print(f"Processing {raster_path}")
            try:
                # Load the image for detection and segmentation
                image = cv2.imread(raster_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Predict bounding boxes using DeepForest
                predicted_mask = model.predict_tile(raster_path, return_plot=False, patch_size=300, patch_overlap=0.25)

                # Set the image for SAM segmentation
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

                # Save the final mask as a georeferenced TIFF
                with rasterio.open(raster_path) as src:
                    transform = src.transform
                    crs = src.crs

                # Define the output path for saving the segmented mask
                output_path = os.path.join(output_dir, f"{os.path.basename(raster_path).replace('.tif', '_segmented.tif')}")

                # Save the mask as a GeoTIFF
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

                print(f"Segmented mask saved to {output_path}")

            except Exception as e:
                print(f"Error processing {raster_path}: {e}")

    # for raster_path in raster_files:
    #     print(f"Processing {raster_path}")

    #     # Load the image for detection and segmentation
    #     image = cv2.imread(raster_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #     # Predict bounding boxes using DeepForest
    #     predicted_mask = model.predict_tile(raster_path, return_plot=False, patch_size=300, patch_overlap=0.25)

    #     # Set the image for SAM segmentation
    #     predictor.set_image(image)

    #     # Create an empty mask with the same dimensions as the original image
    #     final_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    #     # Iterate through the detected objects and use SAM to segment them
    #     for i, prediction in predicted_mask.iterrows():
    #         x1, y1, x2, y2 = int(prediction['xmin']), int(prediction['ymin']), int(prediction['xmax']), int(prediction['ymax'])

    #         # Create a box prompt for SAM
    #         box = np.array([x1, y1, x2, y2])

    #         # Predict the mask using SAM
    #         masks, _, _ = predictor.predict(box=box)

    #         # Combine the SAM mask into the final mask
    #         for mask in masks:
    #             final_mask[mask] = 255  # Assuming binary segmentation, 255 for foreground

    #     # Save the final mask as a georeferenced TIFF
    #     with rasterio.open(raster_path) as src:
    #         transform = src.transform
    #         crs = src.crs

    #     # Define the output path for saving the segmented mask
    #     output_path = os.path.join(output_dir, f"{os.path.basename(raster_path).replace('.tif', '_segmented.tif')}")

    #     # Save the mask as a GeoTIFF
    #     with rasterio.open(
    #             output_path,
    #             'w',
    #             driver='GTiff',
    #             height=final_mask.shape[0],
    #             width=final_mask.shape[1],
    #             count=1,
    #             dtype=final_mask.dtype,
    #             crs=crs,
    #             transform=transform,
    #     ) as dst:
    #         dst.write(final_mask, 1)

    #     print(f"Segmented mask saved to {output_path}")


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process a OSM area.")
    
    # Add the argument
    parser.add_argument('osmid', type=int, help='OSMID to be processed')
    # parser.add_argument('--sam_checkpoint', type=str, required=True, help='Path to the SAM model checkpoint')

    # Parse the arguments
    args = parser.parse_args()

    osmid = args.osmid
    sam_checkpoint = "../data/clean_data/sam/sam_vit_h_4b8939.pth"

    print(f'working on OSMID: {osmid}')

    # Define directories
    raster_dir = f'../data/clean_data/solar/{osmid}'
    output_dir = f'../data/clean_data/canopy_masks/{osmid}'

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process all raster files and save the masks
    process_raster_files(osmid, raster_dir, output_dir, sam_checkpoint)