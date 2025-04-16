# Throwing Shade: Urban Shade Simulation Pipeline

**Throwing Shade** is an ongoing project for simulating and analyzing spatiotemporal urban shade from buildings and trees. It uses Solar API data, tree segmentation models, and DSM raster inputs to compute shade at the pedestrian level.

⚠️ **Contains ongoing work — _please keep forks private_**, thanks!

---

## Installation

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Usage

Before running any individual steps, note that the entire pipeline can be executed from a single script:

### 🧠 Main Pipeline File

**`250225_Dataset_Shade_Main.py`** is the entry point of the full pipeline. It runs the following steps:

#### 1. Download Solar Data
Downloads DSMs, RGB imagery, building masks, and more from the Google Solar API based on input GPS points.

#### 2. Run Tree Segmentation
Segments trees from the RGB imagery using DeepForest and Segment Anything (SAM).

#### 3. Simulate Shade
Computes tree/building shade at multiple timestamps and merges numeric shade metrics into the original dataset.

---

### Inputs Required

- A dataset (GeoJSON or shapefile) with:
  - Latitude and longitude columns
  - A timestamp column (readable by `pd.to_datetime()`)
  - A unique ID column (e.g., `trajectory_id`)
- Solar API coverage shapefiles:
  [Google Solar API Coverage Shapefiles](https://developers.google.com/maps/documentation/solar/coverage)
- SAM model checkpoint:
  [SAM ViT-H Checkpoint](https://github.com/facebookresearch/segment-anything#model-checkpoints)

---

### Important Parameters

- `combined_sh`, `building_sh`: control whether to simulate tree or building shade
- `summer_params`, `winter_params`: specify UTC offset and tree transmissivity per season
- `parameters`: enables extra outputs like shadow fractions and pre-timestamp shade history

---

### ⚠️ Caveats

- A `.env` file with your Solar API key is required:
  ```env
  GOOGLE_API_KEY=your-key-here
  ```
- Simulations are compute-intensive, especially with many timestamps or tiles
- Ensure all file paths are set correctly before execution

---

## Pipeline Steps

### 1) SolarAPI Download

Generates query tiles and fetches data from the Google Solar API based on point locations.

**Required columns:**
- `latitude`, `longitude`
- `timestamp` (datetime-compatible)
- unique ID

**Requirements:**
- `SolarAPIMediumArea.shp` and `SolarAPIHighArea.shp`
- `.env` file with a valid API key

**Returns:**
- OSM buildings
- RGB imagery
- DSMs
- Building mask
- Annual solar flux
- Query tiles for each point

**Relevant code:**
- Main: `Download_SolarAPI_Custom.py`
- Subs: `solar_api_utils.py`

---

### 2) Tree Segmentation

Uses a two-step process:
- **DeepForest** for bounding box detection of tree crowns
- **Segment Anything (SAM)** for pixel-accurate mask segmentation

**Requirements:**
- RGB rasters from SolarAPI
- [SAM model checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

**Returns:**
- Tree canopy masks per tile in `canopy_masks/{OSMID}/`

**Relevant code:**
- `tree_segmentation.py`

---

### 3) Shade Simulation

Simulates the impact of **building** and/or **tree** shade at all GPS points and timestamps. Runs in parallel across tiles and time.

#### 1. Process Raster (Preprocessing)
- Produces two DSM variants per tile:
  - Building-only DSM
  - Tree-only DSM (masked using segmentation)
- Interpolates terrain using Laplace smoothing and filters noise

#### 2. Simulate Shade
- Runs `shadecalculation_setup()` across multiple time intervals
- Uses canopy and building DSMs per tile
- Output: timestamped shade rasters (instantaneous + shadow fraction)

#### 3. Merge Back into Dataset
- Extracts raster values at GPS points (excluding areas under buildings)
- Computes inverse shade (exposure)
- Averages values across each unique ID
- Exports final GeoJSON

**Returns:**
- `results/output/{OSMID}/...`
  - Instantaneous shade rasters
  - Shadow fraction rasters
  - Final merged GeoJSON

**Relevant code:**
- Main: `process_shade.py`
- Subs: `shade_setup.py`, `shadowingfunctions.py`, `sun_position.py`

---

> 🚫 **IGNORE**: `archive/`, `analysis/` folders – for experiments only

---

## Folder Structure

```
code/
    (All scripts etc.)
data/
└── clean_data/
    ├── solar/
    │   └── {OSMID}/
    │       ├── raw downloads
    │       └── rdy_for_processing/
    │           ├── {OSMID}_{tile_id}_{DATE}_building_dsm.tif
    │           └── {OSMID}_{tile_id}_{DATE}_canopy_dsm.tif
    └── canopy_masks/
        └── {OSMID}/
            └── segmented RGB files
results/
└── output/
    └── {OSMID}/
        ├── building_shade/
        └── tree_shade/
            └── {tile_id}/
                ├── {OSMID}_{tile_id}_Shadow_{DATE}_{TIME}_LST.tif
                └── {OSMID}_{tile_id}_shadow_fraction_on_{DATE}_{TIME}.tif
```
