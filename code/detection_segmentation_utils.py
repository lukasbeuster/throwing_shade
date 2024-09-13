from deepforest import main
import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin
from segment_anything import SamPredictor, sam_model_registry
import torch
import matplotlib.pyplot as plt

# TODO: - persist workers, -take in file name to reflect in savepath, -separate loading of the model
def tree_detection_segmentation(raster_path, output_path, model):
    # Load the DeepForest model
    model = main.deepforest()
    model.use_release()

    # Load the image and predict bounding boxes using DeepForest
    raster_path = "../data/clean_data/solar/12011952/12011952_p_1_2022_06_02_rgb.tif"
    image = cv2.imread(raster_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predicted_mask = model.predict_tile(raster_path, return_plot=False, patch_size=300, patch_overlap=0.25)

    # Load the SAM model
    sam_checkpoint = "../data/clean_data/sam/sam_vit_h_4b8939.pth"
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

    # Visualize the final overlay
    overlay_image = image.copy()
    overlay_image[final_mask > 0] = [0, 255, 0]  # Color the masked area

    plt.imshow(overlay_image)
    plt.show()

    # # Save the final mask as a georeferenced TIFF
    # with rasterio.open(raster_path) as src:
    #     transform = src.transform
    #     crs = src.crs

    # # Save the mask as a GeoTIFF
    # output_path = "segmented_output.tif"
    # with rasterio.open(
    #         output_path,
    #         'w',
    #         driver='GTiff',
    #         height=final_mask.shape[0],
    #         width=final_mask.shape[1],
    #         count=1,
    #         dtype=final_mask.dtype,
    #         crs=crs,
    #         transform=transform,
    # ) as dst:
    #     dst.write(final_mask, 1)

    # print(f"Segmented mask saved to {output_path}")