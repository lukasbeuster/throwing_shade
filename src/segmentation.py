from deepforest import main
import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin
from segment_anything import SamPredictor, sam_model_registry
import torch
from pathlib import Path

from deepforest import main
import cv2
import numpy as np
import rasterio
from segment_anything import SamPredictor, sam_model_registry
import torch
from pathlib import Path

def run_segmentation(config, osmid):
    """
    Finds all downloaded RGB tiles and runs tree detection (DeepForest)
    and segmentation (SAM) on each one.
    """
    # Define the input and output paths from the config
    input_dir = Path(config['output_dir']) / f"step2_solar_data/{osmid}"
    output_dir = Path(config['output_dir']) / f"step3_segmentation/{osmid}"
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    # Load the DeepForest model
    model = main.deepforest()
    model.use_release()

    # Load the SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=config['dependencies']['sam_checkpoint'])
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Get a sorted list of all RGB files to process
    rgb_files = sorted(list(input_dir.glob("*_rgb.tif")))
    print(f"Found {len(rgb_files)} RGB tiles to segment.")

    for raster_path in rgb_files:
        print(f"-> Processing {raster_path.name}...")

        # Define the output path for the segmented mask
        output_path = output_dir / f"{raster_path.stem}_segmented.tif"

        # Skip if the output file already exists
        if output_path.exists():
            print(f"   Output file already exists. Skipping.")
            continue

        # Load the image using its full path
        image = cv2.imread(str(raster_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Predict bounding boxes using DeepForest
        predicted_boxes = model.predict_tile(str(raster_path), return_plot=False, patch_size=800, patch_overlap=0.25)

        # Check if any trees were detected
        if predicted_boxes is None or predicted_boxes.empty:
            print("   No trees detected by DeepForest. Skipping SAM.")
            continue

        # Set the image for SAM segmentation
        predictor.set_image(image)

        # Create an empty mask to store all segmented trees
        final_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Iterate through the detected bounding boxes and use SAM to segment them
        for i, prediction in predicted_boxes.iterrows():
            box = np.array([
                int(prediction['xmin']),
                int(prediction['ymin']),
                int(prediction['xmax']),
                int(prediction['ymax'])
            ])

            # Predict the mask using SAM
            masks, _, _ = predictor.predict(box=box, multimask_output=False)

            # Combine the SAM mask into the final mask
            # The first mask (masks[0]) is typically the best one.
            final_mask[masks[0]] = 255  # Set segmented pixels to 255

        # Get metadata from the source raster to save the new one
        with rasterio.open(raster_path) as src:
            transform = src.transform
            crs = src.crs

        # Save the final mask as a GeoTIFF
        with rasterio.open(
                output_path, 'w', driver='GTiff',
                height=final_mask.shape[0], width=final_mask.shape[1],
                count=1, dtype=final_mask.dtype, crs=crs, transform=transform,
        ) as dst:
            dst.write(final_mask, 1)
