# Throwing Shade

**Throwing Shade** is an ongoing research project focused on high-resolution shade simulation and analysis to support urban climate adaptation and heat stress mitigation.  
This repository contains the core code, workflows, and data structure supporting the analyses presented in our manuscript submitted to *Nature Communications*.  
👉 [Read the preprint for full details](#) *(add your link once available)*.

---

## 📌 Study Overview

The project quantifies how **buildings** and **trees** contribute to urban shading — a critical aspect for mitigating pedestrian heat exposure.  
This is achieved using a **dual-stage simulation**:
1. **Grey shade:** Simulate shade using a digital surface model (DSM) with trees removed, capturing only buildings and topography.
2. **Combined shade:** Integrate a canopy height model (CHM) with the DSM to simulate the combined effect of buildings and trees.

This approach mirrors real-world conditions where building and tree shade overlap dynamically, providing a robust picture of available shade at different times of day.

The analysis focuses on **sidewalks**, as they are key areas for pedestrian thermal comfort. Shade statistics are summarized for each sidewalk polygon, capturing both daily averages and 30-minute interval variation throughout the day.

For full methodology, study area descriptions, and city-specific context, see the [preprint](#) *(add link)*.

---

## 📦 Installation

Use the provided `requirements.txt` to install all dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Core Workflows

Below is a brief summary of the main steps and scripts.

⸻

☀️ 1. SolarAPI Data Download

Download raw data via Google SolarAPI and OpenStreetMap:

OSM:
	•	OSM buildings
	•	Query points

Solar API
	•	DSM and RGB imagery
	•	Building mask
	•	Annual flux (optional)

Relevant script:
code/01_download_solar_api.py

---

## 🗺️ 2. Preprocessing

Prepare DSMs for shade simulation:
	•	Derive a Building DSM and a Canopy DSM (CHM).
	•	Trees are detected with DeepForest and segmented using Segment Anything Model (SAM).
	•	Use startinpy for interpolation.

Relevant script:
code/process_area_gilfoyle_parallel_multiple_days.py

---

## 🌳 3. Shadow Simulation

Run shade simulation using a custom Python implementation of the UMEP Shadow Pattern tool:
	•	Two runs: one for buildings only, one for buildings + trees.
	•	Generates shade rasters per timestep and daily averages.
	•	Parallelised for multiple days.

Relevant script:
code/03_process_area_parallel_multiple_days.py

---

## 🚶 4. Sidewalk Extraction

Generate sidewalk polygons using osm2streets (https://github.com/lukasbeuster/osm2streets):

```bash
python3 sidewalk_generator.py "West, Amsterdam"
```
	•	Downloads raw OSM.
	•	Tiles & processes data to produce detailed lane and sidewalk GeoJSONs.

---

## 📂 5. Other Key Scripts & Notebooks
	•	shade_metrics_on_graph.ipynb — Assigns shade weights to network edges.
	•	calculate_shade_metrics_all.py — Computes shade metrics for hard surfaces.
	•	tree_detection_segmentation.ipynb — Tests tree detection outside NL using DeepForest + SAM.
	•	momepy_importance.ipynb — Explores using momepy for network analysis.
	•	240912_Download_SolarAPI.ipynb — Example SolarAPI workflow.

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

This workflow builds upon:
	•	UMEP (Lindberg et al., 2018)
	•	Google SolarAPI
	•	DeepForest (Weinstein et al., 2020; Weinstein et al., 2022)
	•	Segment Anything Model (SAM) (Kirillov et al., 2023)

Please cite these resources when using or extending this project.


---

## ✨ Contributing

Contributions and feedback are welcome. Please open an issue or submit a pull request.

---

## 📜 License

---
## 📎 More Information

📄 Preprint link — Coming soon!

## Usage



### Preprocessing

The UMEP plugin requires two inputs (if you want to include trees): 
- DSM 
- CHM

Requirement (due to data-sources): Process DSM into Building DSM and Canopy Height Model (CHM/Canopy DSM)


Dataflow: from data/clean_data/solar/{OSMID} to data/clean_data/solar/{OSMID}/rdy_for_processing

Relevant code: 
file: process_area_gilfoyle_parallel_multiple_days.py
function: process_raster

Steps:

- Use CHM generated via AHN as raster mask (data/clean_data/chm) -> Canopy DSM

(See 230921_Tree_Segmentation_multiple_tiles.R for AHN processing script (uses LidR package, thus written in R))

![alt text](DSM_to_BuildingDSM.png)

Fill in missing ground values.

- Mask out buildings and trees from DSM -> prepare for interpolation 

![alt text](DSM_to_Ground.png)

- Interpolate using startinpy, add buildings and save newly created Building DSM

![alt text](DSM_to_BuildingDSM.png)

### Shadowcasting

UMEP Shadow Pattern as standalone implementation.

NOTE: For my research I'm running the shade simulation twice. 1st run with buildings only, 2nd run with buildings and trees. One of the things I'm working on is the difference between building and tree shade, so I require both. 

Execution in parallel, for multiple days.

Results (see results folder): 
- shade raster per timestep
- shade raster for daily shading. 
for both buildings only (passing only the building DSM to the function) and buildings and trees (including both buildings and trees)

You'll have to untangle this. 


Relevant code:
file: process_area_gilfoyle_parallel_multiple_days.py
function: shade_processing

links to
- shade_setup.py
- shadowingfunctions.py
- sun_position.py


### Sidewalk download

needs osm2streets_python from this fork, or from the main project as soon as my pull request was accepted: https://github.com/lukasbeuster/osm2streets

to download lane polygons for any area in the world use the `sidewalk_generator.py`

```bash
python3 sidewalk_generator.py "West, Amsterdam"
```

This will:
- find the corresponding location
- create tiles and download raw osm xml files
- process these tiles sequentially using osm2streets, saving geojson files of lanes and intersections

### Other noteworthy code:

- shade_metrics_on_graph.ipynb: notebook with the code to calculate shade_weights per edge in a network graph. 

- calculate_shade_metrics_multiple_days.py: calculate shade metrics on polygons of public space (hard surfaces).

- tree_detection_segmentation.ipynb: exploration of tree detection + segmentation workflow using DeepForest + SAM. Only relevant for scaling up to contexts outside NL (where AHN is not available)

- momepy_importance.ipynb: Exploration of using multiple centrality assessment (from momepy) to identify the most important edges in a network

- 240912_Download_SolarAPI.ipynb: Example flow of SolarAPI download. You likely don't need to use this at all. 



IGNORE ARCHIVE AND ANALYSIS FOLDERS


