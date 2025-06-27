import click

@cli.command()
@click.option('--config', default='config.yaml', help='Path to the configuration file.')
@click.option('--min-points', type=int, help='Override the minimum points per tile.')

def check_coverage(config, min_points):
    """STEP 1: Check solar tile coverage without downloading."""
    cfg = load_config(config)
    if min_points:
        cfg['solar_api']['min_points_per_tile'] = min_points
        click.secho(f"Overriding min_points_per_tile to {min_points}", fg='yellow')

    click.echo("--- Running Step 1: Check Solar Coverage ---")

    # This function now returns the tile count and the path to the saved file
    tile_count, preview_path = check_coverage_logic(cfg) # We assume this is your refactored logic

    # --- Part 1: Provide a clear summary ---
    click.secho(f"\nAnalysis complete. Found {tile_count} tiles to download.", fg='green')
    click.echo("A preview map has been saved to:")
    click.secho(f"  {preview_path}", fg='cyan') # Highlight the file path
    click.echo("\n=> Please open this file to visually inspect the tile coverage.")

    # --- Part 2: Wait for user confirmation ---
    if not click.confirm("\nDo you want to proceed with downloading the data for these tiles?"):
        click.echo("Operation cancelled by user.")
        # The 'abort=True' option could also be used to exit immediately
        return # Exit the function

    click.echo("Confirmation received. You can now run the 'download' command.")
