import click
import json
import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
from datetime import datetime

# This makes sure Python can find your 'src' directory
sys.path.append(str(Path(__file__).parent / 'src'))

# --- Import the main "engine" functions from your src modules ---
from solar import check_coverage, download_data
from segmentation import run_segmentation
from raster import raster_processing_main
from processing import run_shade_processing

# --- Helper Functions for State Management ---

def load_config(config_path):
    """Loads the YAML configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_run_info(output_dir: Path):
    """Loads the run_info.json handoff file."""
    run_info_path = output_dir / 'run_info.json'
    if not run_info_path.exists():
        return {} # Return empty dict if it doesn't exist yet
    with open(run_info_path, 'r') as f:
        return json.load(f)

def save_run_info(output_dir: Path, data: dict, fresh_start: bool = False):
    """
    Saves data to the run_info.json handoff file.
    If fresh_start is True, it overwrites the file completely.
    """
    run_info_path = output_dir / 'run_info.json'

    # If it's a fresh start, begin with an empty dictionary.
    # Otherwise, load the existing data to update it.
    if fresh_start:
        existing_data = {}
    else:
        existing_data = load_run_info(output_dir)

    existing_data.update(data)
    with open(run_info_path, 'w') as f:
        json.dump(existing_data, f, indent=4)


# --- Flexible Dataset Loader ---
def load_dataset_flexibly(config):
    """Loads a dataset (CSV, Parquet, Pickle, GeoJSON, etc.) into a GeoDataFrame."""
    dataset_path = config['dataset_path']
    lon_col = config['columns']['longitude']
    lat_col = config['columns']['latitude']

    # Determine CRS settings
    input_crs = config.get('input_crs', 'EPSG:4326')

    # Load data into GeoDataFrame regardless of format
    suffix = Path(dataset_path).suffix.lower()
    if suffix == '.csv':
        df = pd.read_csv(dataset_path)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs=input_crs
        )
    elif suffix in ('.pkl', '.pickle'):
        df = pd.read_pickle(dataset_path)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs=input_crs
        )
    elif suffix == '.parquet':
        df = pd.read_parquet(dataset_path)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs=input_crs
        )
    else:
        # GeoJSON, GPKG, shapefiles, etc.
        gdf = gpd.read_file(dataset_path)

    return gdf

# --- The Main CLI Group ---

@click.group()
def cli():
    """
    A command-line tool for the 'Throwing Shade' simulation pipeline.
    Run steps in order: check -> download -> segment -> process-rasters -> process-shade
    Or use 'run-all' to execute the entire pipeline.
    """
    pass

# --- STEP 1: Interactive Coverage Check ---

@cli.command()
@click.option('--config', default='config.yaml', type=click.Path(exists=True), help='Path to the configuration file.')
@click.option('--min-points', default=1, type=int, help='Override min_points_per_tile from the config file.')
@click.option('--yes', is_flag=True, help='Skip interactive confirmation (for automation).')
def check(config, min_points, yes):
    """STEP 1: Check required Solar API tiles without downloading."""
    cfg = load_config(config)
    if min_points:
        cfg['solar_api']['min_points_per_tile'] = min_points

    click.echo("--- Running Step 1: Check Solar Coverage ---")
    tile_count, preview_path = check_coverage(cfg)

    # Save the path to the preview file for the next step
    output_dir = Path(cfg['output_dir'])

    # By setting fresh_start=True, we signal the beginning of a new run.
    save_run_info(output_dir, {'preview_path': str(preview_path)}, fresh_start=True)

    click.secho(f"\n✅ Found {tile_count} tiles to download.", fg='green')
    click.echo("A preview map has been saved to:")
    click.secho(f"   {preview_path}", fg='cyan')
    click.echo("\n=> ACTION: Open this GeoJSON file in a viewer (like vscode-geojson or QGIS) to inspect the coverage.")

    if not yes:
        if not click.confirm("\nIs this coverage acceptable to proceed?"):
            click.echo("❌ Operation cancelled. Please re-run 'check' with a different '--min-points' value.")
            sys.exit(0) # Exit gracefully

    click.echo("Confirmation received. You can now run the 'download' step.")

# --- STEP 2: Download Data ---

@cli.command()
@click.option('--config', default='config.yaml', type=click.Path(exists=True), help='Path to the configuration file.')
def download(config):
    """STEP 2: Download Solar API data and generate a run ID (osmid)."""
    cfg = load_config(config)
    output_dir = Path(cfg['output_dir'])
    run_info = load_run_info(output_dir)

    preview_path = run_info.get('preview_path')
    if not preview_path:
        click.secho("Error: 'preview_path' not found. Please run the 'check' step first.", fg='red')
        return

    click.echo("--- Running Step 2: Downloading Solar API Data ---")
    osmid = download_data(cfg, preview_path)

    # Save the generated osmid to our handoff file
    save_run_info(output_dir, {'osmid': osmid})

    click.secho(f"\n✅ Download complete. Run ID '{osmid}' saved to {output_dir / 'run_info.json'}.", fg='green')

# --- STEP 3: Tree Segmentation ---

@cli.command()
@click.option('--config', default='config.yaml', type=click.Path(exists=True), help='Path to the configuration file.')
def segment(config):
    """STEP 3: Run tree segmentation on the downloaded RGB tiles."""
    cfg = load_config(config)
    run_info = load_run_info(Path(cfg['output_dir']))
    osmid = run_info.get('osmid')
    if not osmid:
        click.secho("Error: 'osmid' not found. Please run the 'download' step first.", fg='red')
        return

    click.echo(f"--- Running Step 3: Segmenting Trees for Run ID: {osmid} ---")
    run_segmentation(cfg, osmid)
    click.secho("\n✅ Tree segmentation complete.", fg='green')

# --- STEP 4: Raster Processing ---

@cli.command()
@click.option('--config', default='config.yaml', type=click.Path(exists=True), help='Path to the configuration file.')
def process_rasters(config):
    """STEP 4: Process raw DSMs into analysis-ready DSMs."""
    cfg = load_config(config)
    run_info = load_run_info(Path(cfg['output_dir']))
    osmid = run_info.get('osmid')
    if not osmid:
        click.secho("Error: 'osmid' not found. Please run the 'download' step first.", fg='red')
        return

    click.echo(f"--- Running Step 4: Processing Raster Files for Run ID: {osmid} ---")
    raster_processing_main(cfg, osmid)
    click.secho("\n✅ Raster processing complete.", fg='green')

# --- STEP 5: Final Shade Processing ---

@cli.command()
@click.option('--config', default='config.yaml', type=click.Path(exists=True), help='Path to the configuration file.')
def process_shade(config):
    """STEP 5: Run the final shade analysis and generate results."""
    cfg = load_config(config)
    output_dir = Path(cfg['output_dir'])
    run_info = load_run_info(output_dir)
    osmid = run_info.get('osmid')
    if not osmid:
        click.secho("Error: 'osmid' not found. Please run the previous steps first.", fg='red')
        return

    click.echo(f"--- Running Step 5: Final Shade Processing for Run ID: {osmid} ---")

    # Load dataset flexibly based on file format
    dataset = load_dataset_flexibly(cfg)
    timestamp_col = cfg['columns']['timestamp']
    dataset[timestamp_col] = pd.to_datetime(dataset[timestamp_col], errors='coerce')
    
    n_invalid = dataset[timestamp_col].isna().sum()
    print(f"{n_invalid} rows failed to parse timestamps and became NaT")

    # Drop the bad ones before analysis
    dataset = dataset[timestamp_col].dropna()

    all_year_results = []

    for year_str, year_config in cfg['year_configs'].items():
        year = int(year_str)
        click.echo(f"-> Processing data for year {year}...")
        year_data = dataset[dataset[timestamp_col].dt.year == year].copy()

        if year_data.empty:
            click.secho(f"  No data found for year {year}, skipping.", fg='yellow')
            continue

        # Call the engine function for this year's data
        single_year_result = run_shade_processing(cfg, osmid, year, year_data)
        all_year_results.append(single_year_result)

    # Combine and save the final result
    if all_year_results:
        final_dataset = pd.concat(all_year_results, ignore_index=True)
        final_dataset = gpd.GeoDataFrame(final_dataset, geometry='geometry')

        # Add a placeholder for your final output path in config if needed
        final_output_dir = output_dir / f"step6_final_result/{osmid}"
        final_output_dir.mkdir(parents=True, exist_ok=True)
        final_output_path = final_output_dir / "shaded_dataset.geojson"

        final_dataset.to_file(final_output_path, driver="GeoJSON")
        click.secho(f"\n✅ Pipeline complete! Final output saved to: {final_output_path}", fg='green')
    else:
        click.secho("\n❌ No data was processed. No output file created.", fg='red')


# --- Convenience Command to Run All Steps ---

@cli.command()
@click.option('--config', default='config.yaml', type=click.Path(exists=True), help='Path to the configuration file.')
@click.option('--min-points', type=int, help='Set min_points_per_tile for the entire run.')
@click.pass_context
def run_all(ctx, config, min_points):
    """Runs the entire pipeline sequentially."""
    click.secho("--- Running Full Pipeline ---", bold=True, fg='magenta')

    # Use ctx.invoke to call other click commands from this one
    ctx.invoke(check, config=config, min_points=min_points, yes=True)
    ctx.invoke(download, config=config)
    ctx.invoke(segment, config=config)
    ctx.invoke(process_rasters, config=config)
    ctx.invoke(process_shade, config=config)

    click.secho("\n🎉 All pipeline steps completed successfully! 🎉", bold=True, fg='magenta')


if __name__ == '__main__':
    cli()
