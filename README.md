# Throwing Shade: Urban Shade Simulation Pipeline

**Throwing Shade** is an ongoing project for simulating and analyzing spatiotemporal urban shade from buildings and trees. It uses Solar API data, tree segmentation models, and DSM raster inputs to compute shade. The pipeline is an extension fo this project to automate the process of enhancing any geolocated dataset with shade data based timestamp and location

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

- A dataset (csv/GeoJSON/shapefile) with:
  - Latitude and longitude columns
  - A timestamp column (readable by `pd.to_datetime()`)
  - A unique ID column (e.g., `trajectory_id`)
- Solar API coverage shapefiles:
  [Google Solar API Coverage Shapefiles](https://developers.google.com/maps/documentation/solar/coverage)
- SAM model checkpoint:
  [SAM ViT-H Checkpoint](https://github.com/facebookresearch/segment-anything#model-checkpoints)
- Solar API Key

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
- OSM building polygons for dataset area
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
- Tree canopy masks per tile in `data/clean_data/canopy_masks/{OSMID}/`

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
- Computes inverse raster values (exposure)
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

### ⚠️ Caveats

- **DON'T** change the 'dst' (daylight savings) parameter from 0, instead change the UTC input for daylight savings period
- Be very careful about what UTC your dataset is in, mistakes with this might cause simulations to be off by hours

## Folder Structure

```
code/
└── (All scripts etc.)
└── results/
    └── output/
        └── {OSMID}/
            ├── building_shade/
            └── combined_shade/
                └── {tile_id}/
                    ├── {OSMID}_{tile_id}_Shadow_{DATE}_{TIME}_LST.tif
                    └── {OSMID}_{tile_id}_shadow_fraction_on_{DATE}_{TIME}.tif
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
```


# Sun Blocked: A Spatiotemporal Urban Shade Simulation Pipeline

**Throwing Shade** is a Python-based pipeline for simulating and analyzing the impact of urban shade from buildings and trees on any geolocated dataset.

The tool enhances your existing timeseries data (e.g., GPS tracks, sensor readings) with high-resolution shade metrics. It uses data from the Google Solar API, tree masks generated by a segmentation model (SAM), and Digital Surface Models (DSMs) to calculate shade values for every point at every timestamp in your dataset. By default, the pipeline simulates shade per tile from sunrise to latest timestamp in the tile for each day.

This project is structured as a powerful Command-Line Interface (CLI) that guides the user through the necessary steps, from data acquisition to final analysis.

⚠️ **Contains ongoing work**

-----

## Features

  - **Step-by-Step CLI**: An intuitive command-line interface guides you through the entire workflow.
  - **Interactive Coverage Check**: Visualize the required Solar API tiles for your dataset and adjust parameters *before* downloading gigabytes of data.
  - **Automated Data Fetching**: Downloads all necessary building, terrain, and RGB data from the Google Solar API and OpenStreetMap.
  - **AI-Powered Tree Segmentation**: Uses the Segment Anything Model (SAM) from Meta AI to generate accurate tree canopy masks from aerial imagery.
  - **Advanced Shade Simulation**: Computes detailed shade rasters, including instantaneous shade and time-averaged shadow fractions.
  - **Parallel Processing**: Leverages multiple CPU cores to significantly speed up heavy processing tasks.

-----

## How It Works

The pipeline is broken down into five distinct, sequential steps, each run by a simple command. This modular approach allows you to inspect the output of each stage and re-run steps with different parameters without starting from scratch.

  1.  **`check`**: Analyzes your dataset and determines the geographic tiles needed for the analysis.
2\.  **`download`**: Fetches all raw data (DSMs, imagery, building footprints) for the selected tiles.
3\.  **`segment`**: Runs the AI model to identify trees and create canopy masks.
4\.  **`process-rasters`**: A heavy-lifting step that processes the raw DSMs into analysis-ready terrain and canopy models.
5\.  **`process-shade`**: The final step, which runs the shade simulation and merges the results back into your original dataset.

## 🚀 Getting Started

### 1\. Installation

First, clone the pipeline branch of the repository to your local machine:

```bash
git clone --branch feature/cli-refactor https://github.com/lukasbeuster/throwing_shade.git
cd throwing_shade
```

Next, it's highly recommended to create a dedicated Python environment to avoid conflicts.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

Install all the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### 2\. Configuration

Before running the pipeline, you must configure it for your project.

**A. Set Up Your API Key**

The pipeline requires a Google Solar API key.

Open the `.env` file and paste in your API key:
```
GOOGLE_API_KEY=YOUR_API_KEY_HERE
```

**B. Download Required Files**

Download the Solar API coverage shapefiles and Segment Anything Model.

1.  Download medium and high resolution coverage shapefiles [here](https://developers.google.com/maps/documentation/solar/coverage)
2.  Download SAM ViT-H checkpoint [here](https://github.com/facebookresearch/segment-anything#model-checkpoints)


**C. Configure Your Run**
1.  Copy the main configuration template:
    ```bash
    cp config.yaml.template config.yaml
    ```
2.  Open `config.yaml` and edit the paths and parameters. **At a minimum, you must set:**
      * `dataset_path`: The full path to your input data file.
      * `dependencies`: The full paths to your Solar API coverage shapefiles and the SAM model checkpoint.
      * `columns`: The column names matching the required columns.

    **Rest of the parameters:**
    * `simulation`:
      * `shade_interval_minutes` (int): The frequency of shade simulation in minutes. Default is 30 minutes.
      * `combined_shade`, `building_shade` (Boolean): Control whether to simulate combined and/or building shade. Combined shade is shade from both buildings and trees.
      * `start_time` (str): Can provide a start time in the 'HH:MM' format if you don't want to simulate from sunrise to timestamp.

-----

## 💻 Usage: Running the Pipeline

All commands are run from your terminal in the project's root directory.

### See All Commands

To see a full list of available commands and their options, run:

```bash
python pipeline.py --help
```

### Step 1: Check Coverage (Interactive)

This crucial first step analyzes your dataset and shows you which geographic tiles you need to download. It allows you to adjust the density of points required per tile, giving you control over the cost and scope of your analysis. Solar API quotas and pricing can be checked [here](https://mapsplatform.google.com/pricing/?_gl=1*14kp531*_ga*NzgzMjQ1MzIyLjE3NDQ3OTI2MzM.*_ga_NRWSTWS78N*czE3NTE1NDcxOTEkbzEkZzEkdDE3NTE1NDgwNjgkajQ1JGwwJGgw&utm_experiment=13102542) under Solar API Data Layers.

```bash
python pipeline.py check --min-points 10
```

  * **Action Required:** The script will save a `coverage_preview_...geojson` file in the `output/` directory. **You must open this file in a GIS viewer** (like the "GeoJSON" extension in VS Code, QGIS, ArcGIS) to inspect the tile layout.
  * The script will then pause and ask for your confirmation. If the number of tiles is too high, cancel the operation (`N`), and re-run the `check` command with a higher `--min-points` value.

Output: Coverage preview file in `{output_dir}/step1_solar_coverage`

### Step 2: Download Data

Once you are satisfied with the coverage, run the download command. This will fetch all the necessary data from Solar API (digital surface models, RGB images and building masks) and generate a unique `osmid` (a run ID) for your project. Take note of this ID for future.

```bash
python pipeline.py download
```

Output: 4 rasters files per tile in `{output_dir}/step2_solar_data/{osmid}`

### Step 3: Segment Trees

This step runs the tree segmentation model on the downloaded RGB imagery. The segmented tree masks are saved as rasters.

```bash
python pipeline.py segment
```
Output: Segmented raster per tile in `{output_dir}/step3_segmentation/{osmid}`

### Step 4: Process Rasters

This is a time-consuming step that prepares the raw DSMs into analysis-ready terrain and canopy models. You only need to run this once per dataset.

```bash
python pipeline.py process-rasters
```

Output: CHM and DTM raster per tile in `{output_dir}/step4_raster_processing/{osmid}`

### Step 5: Process Shade

This is the final step. It runs the shade simulation using the prepared data and merges the results back into your original dataset.

```bash
python pipeline.py process-shade
```

Your final, shade-enhanced dataset will be saved in `{output_dir}/step5_final_results/{osmid}/{file_name}.geojson`.

### Run All Steps Automatically

For automated workflows, you can execute the entire pipeline with a single command. This will skip the interactive confirmation step and the checkpoints.

```bash
python pipeline.py run-all --min-points 10
```

-----

## ⚙️ Configuration Details

All pipeline parameters are controlled in `config.yaml`.

| Parameter | Section | Description |
| :--- | :--- | :--- |
| `dataset_path` | `Input and Output Paths` | Path to your input csv, GeoJSON or Shapefile with longitude and latitude columns. |
| `output_dir` | `Input and Output Paths` | The base directory where all results will be saved. |
| `columns` | `Dataset Column Names`| Maps the required column names (`latitude`, `longitude`, etc.) to the names in your dataset. |
| `dependencies` | `Dependency Paths`| Paths to essential files like the SAM model checkpoint and Solar API coverage shapefiles. |
| `simulation` | `Shade Simulation Parameters`| Controls the core simulation settings like shade interval, buffer sizes, and whether to include building/tree shade. |
| `year_configs` | `Daylight Saving Time`| **Crucial for accuracy.** Define the start and end dates of DST for each year present in your data. |

## Folder Structure

```
code/
└── (All scripts etc.)
└── results/
    └── output/
        └── {OSMID}/
            ├── building_shade/
            └── combined_shade/
                └── {tile_id}/
                    ├── {OSMID}_{tile_id}_Shadow_{DATE}_{TIME}_LST.tif
                    └── {OSMID}_{tile_id}_shadow_fraction_on_{DATE}_{TIME}.tif
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
```

## License

----

## Cite
