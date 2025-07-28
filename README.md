# Throwing Shade

**Throwing Shade** is an ongoing research project focused on high-resolution shade simulation and analysis to support urban climate adaptation and heat stress mitigation.  
This repository contains the core code, workflows, and data structure supporting the analyses presented in our manuscript submitted to *npj Urban Sustainability*.  
👉 [Read the preprint for full details](https://doi.org/10.21203/rs.3.rs-6966874/v1).

You'll find the expanded pipeline referenced in _"Sun Blocked: Integrating Shade into Urban Climate Assessments"_, which adds the functionality to add shade information to your geo/time-referenced data [here](https://github.com/lukasbeuster/throwing_shade/tree/feature/cli-refactor).

---

## 📌 Study Overview

The project quantifies how **buildings** and **trees** contribute to urban shading — a critical aspect for mitigating pedestrian heat exposure.  
This is achieved using a **dual-stage simulation**:
1. **Grey shade:** Simulate shade using a digital surface model (DSM) with trees removed, capturing only buildings and topography.
2. **Combined shade:** Integrate a canopy height model (CHM) with the DSM to simulate the combined effect of buildings and trees.

This approach mirrors real-world conditions where building and tree shade overlap dynamically, providing a robust picture of available shade at different times of day.

The analysis focuses on **sidewalks**, as they are key areas for pedestrian thermal comfort. Shade statistics are summarized for each sidewalk polygon, capturing both daily averages and 30-minute interval variation throughout the day.

For full methodology, study area descriptions, and city-specific context, see the [preprint](#) *(TODO: add link)*.

---
📦 Installation

This project uses two separate Python environments:
	•	Tree Detection & Segmentation (DeepForest & SAM)
	•	Shade Simulation & Analysis ([UMEP](https://umep-docs.readthedocs.io/en/latest/)-based shadow casting, [startinpy](https://startinpy.readthedocs.io/latest/) for terrain processing & spatial stats)

1️⃣ Tree Detection Environment

# Create and activate

```bash
python3 -m venv segment_trees
source segment_trees/bin/activate
```

# Install tree detection and segmentation dependencies

For detailed instructions on setting up tree detection, please refer to the [DeepForest documentation](https://deepforest.readthedocs.io/en/latest/) and [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything). These provide guidance on downloading pre-trained models and installing required dependencies.

```bash
pip install -r requirements_trees.txt
```

2️⃣ Shade Simulation Environment

# Deactivate tree env if needed
```bash
deactivate
```

# Create and activate
```bash
python3 -m venv throwing_shade
source throwing_shade/bin/activate
```

# Install shade simulation dependencies

```bash
pip install -r requirements_shade.txt
```

✅ Switching Environments
	•	For tree detection scripts: activate segment_trees
	•	For shade simulation & analysis: activate throwing_shade


## ⚙️ Core Workflows — Quick How-To

Below is a brief summary of the main steps and scripts.


### ☀️ 1. SolarAPI Data Download

Uses geocoding and polygon tiling to generate request points inside your study area's bounds.

Automates UTM conversion, checks SolarAPI coverage, and safely saves query points and building footprints.

Calls the Google Solar API to download high-resolution DSMs, RGB imagery, building masks, and optional annual flux layers.

👉 Typical usage:
```bash
python code/01_download_SolarAPI.py "<Place Name>" <spacing in meters>
```
Example:
```bash
python code/01_download_SolarAPI.py "West, Amsterdam" 950
```
This sets up SolarAPI tiles with overlap, downloads necessary building footprints, and saves query points for shade simulation.

Note: Requires a valid Google API key set in your ```.env``` file.

Relevant script:code/01_download_SolarAPI.py

---

### 🌳 2. Tree Detection & Segmentation

Runs DeepForest to detect trees in aerial imagery.

Uses Segment Anything (SAM) to refine each detected tree into a precise canopy mask.

Saves each canopy mask as a georeferenced raster aligned with the DSM.

```bash
python code/02_tree_detection_segmentation.py <OSMID>
```
Example:
```bash
python code/02_tree_detection_segmentation.py 123456
```
Note: Requires the SAM checkpoint file in ```data/clean_data/sam/```.

Script: code/02_tree_detection_segmentation.py

--- 

### 🏗️ 3. DSM Preprocessing & Shadow Casting

Prepares separate DSMs for buildings and tree canopies after detection/segmentation.

Applies masking, interpolation, and smoothing to create high-quality building DSMs and canopy DSMs.

Runs computationally heavy shadow simulations for multiple key days using a standalone UMEP-based method.

Processes tiles in parallel (CPU-intensive).

```bash
python code/03_process_area_gilfoyle_parallel_multiple_days.py <OSMID>
```
Example:
```bash
python code/03_process_area_gilfoyle_parallel_multiple_days.py 15419236
```
This script saves ready-to-use shadow rasters per timestep and daily averages for both "buildings only" and "buildings + trees" scenarios.

NOTE: UTC for your study area has to be set manually inside the script. 

---

### 🚶 4. OSM2streets Sidewalk Extraction

If no high-quality sidewalk polygons are available, use [KerbSide](https://github.com/lukasbeuster/KerbSide), based on [osm2streets](https://github.com/lukasbeuster/osm2streets, to infer sidewalks and lanes for the area of interest. 

This requires a separate environment, follow the instructions [here](https://github.com/lukasbeuster/KerbSide). Once installed, sidewalks can be generated using

```bash
python code/04_sidewalk_generator.py "<Place Name>" --tile_size 0.01 --driving_side Right
```

Example:
```bash
python code/04_sidewalk_generator.py "West, Amsterdam" --tile_size 0.01 --driving_side Right
```
This will:

- Query the location with Nominatim.
- Download raw OSM tiles.
- Process each tile using OSM2streets to extract sidewalks, lanes, and intersections.
- Save the output as GeoJSON files.


### 📏 5. Shade Metrics on Sidewalks

Calculate shade statistics for each sidewalk polygon, either using OSM2streets output or an existing detailed sidewalk dataset:

- If using OSM2streets:
```bash
python code/05_calculate_shade_metrics_osm2streets.py <OSMID> <YYYY-MM-DD> <YYYY-MM-DD> ...
```
Example:
```bash
python code/05_calculate_shade_metrics_osm2streets.py 15419236 2024-06-21 2024-07-15
```
- If using an external sidewalk or public space dataset:
Use ```05_calculate_shade_metrics_polygons.py``` and edit the script to specify your input polygons.

This will compute both daily average and timestep-specific shade metrics for each polygon.

> NOTE: More details for each step are in the script comments and the preprint.

---

## 📁 Folder Structure

```plaintext
code/                 # Scripts and notebooks
data/
└── clean_data/
    ├── solar/{OSMID}/      # SolarAPI DSM, RGB, flux
    └── chm/{SUBTILE}.tif   # Canopy DSMs
results/
└── figures/
└── output/{OSMID}/
    ├── building_shade/
    └── tree_shade/{point_id}/
        ├── {OSMID}_{point_id}_Shadow_{DATE}_{TIME}_LST.tif
        └── {OSMID}_{point_id}_shadow_fraction_on_{DATE}.tif
```

## 📑 References & Related Work

This workflow builds on and integrates the following tools and research:

- [UMEP](https://umep-docs.readthedocs.io/en/latest/) (Lindberg et al., 2018) — an urban microclimate model used here for standalone shadow pattern simulation.
- [Google SolarAPI](https://developers.google.com/maps/documentation/solar/overview) — provides high-resolution DSMs, RGB imagery, and solar flux data.
- [DeepForest](https://deepforest.readthedocs.io/en/latest/) (Weinstein et al., 2020; 2022) — used for automated tree detection from aerial imagery.
- [Segment Anything Model (SAM)](https://segment-anything.com/) (Kirillov et al., 2023) — used for fine-grained tree canopy segmentation.

Please cite these resources appropriately if you use or extend this project.

---

## ✨ Contributing

Contributions and feedback are welcome. Please open an issue or submit a pull request.

---

## 📜 License

---
## 📎 More Information

📄 [Preprint link](https://doi.org/10.21203/rs.3.rs-6966874/v1) — Coming soon!


