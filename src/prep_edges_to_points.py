import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path

def densify_lines_to_points(gdf_lines, spacing_m=10, edge_id_col="edge_uid"):
    # Work in metric CRS for distances
    gdf_m = gdf_lines.to_crs(3857)
    out_rows = []

    for _, row in gdf_m.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        length = geom.length
        if length == 0:
            continue

        edge_uid = row[edge_id_col]
        dists = np.arange(0, length + spacing_m, spacing_m)
        pts_m = [geom.interpolate(float(d)) for d in dists]
        pts = gpd.GeoSeries(pts_m, crs=3857).to_crs(4326)

        for i, p in enumerate(pts):
            out_rows.append({
                "edge_uid": edge_uid,
                "sample_idx": i,
                "geometry": p
            })

    gdf_pts = gpd.GeoDataFrame(out_rows, geometry="geometry", crs="EPSG:4326")
    gdf_pts["sample_id"] = gdf_pts["edge_uid"].astype(str) + "_" + gdf_pts["sample_idx"].astype(str)
    gdf_pts["latitude"]  = gdf_pts.geometry.y
    gdf_pts["longitude"] = gdf_pts.geometry.x
    return gdf_pts[["sample_id", "edge_uid", "sample_idx", "latitude", "longitude", "geometry"]]

def load_hours_csvs(hour_csv_paths):
    # Expect columns: edge_uid, hour, osm_reference_id
    frames = []
    for p in hour_csv_paths:
        df = pd.read_csv(p, dtype={"edge_uid": str, "osm_reference_id": str})
        # Parse YYYY-MM-DDTHH -> keep as **naive local** timestamps (matches LST filenames)
        df["time"] = pd.to_datetime(df["hour"], format="%Y-%m-%dT%H", errors="coerce")
        frames.append(df[["edge_uid", "time", "osm_reference_id"]])
    hours = pd.concat(frames, ignore_index=True).dropna(subset=["time"])
    # normalize types
    hours["edge_uid"] = hours["edge_uid"].astype(str)
    hours["osm_reference_id"] = hours["osm_reference_id"].astype(str)
    return hours.drop_duplicates()

def main():
    # ---- EDIT THESE INPUTS ----
    edges_path = "../data/clean_data/strava/data_to_share/all_edges_hourly_2024-04-01-2024-04-30_ped_boston_filtered/53f3ef7e32738022bd45a2ed224bcbcf8cb63f07dc3d5bca7e9221dfe7fe4451-1738340934480.shp"            # LineStrings with edgeUID
    back_bay_poly = "../data/raw_data/bos/back_bay.json"            # Optional AOI
    hour_csvs = [
        "../data/clean_data/strava/data_to_share/all_edges_hourly_2024-04-01-2024-04-30_ped_boston_filtered/53f3ef7e32738022bd45a2ed224bcbcf8cb63f07dc3d5bca7e9221dfe7fe4451-1738340934480.csv",
        # "PATH/TO/hours_june.csv",
        # "PATH/TO/hours_october.csv",
        # "PATH/TO/hours_december.csv",
    ]
    spacing_m = 10
    prepared_out = Path("../data/clean_data/strava/back_bay_points_hours.parquet")

    # 1) Load edges, clip to Back Bay
    edges = gpd.read_file(edges_path)
    # Check for edgeUID column and rename it to edge_uid for consistency with CSV files
    if "edgeUID" not in edges.columns:
        raise ValueError("edges file must contain an 'edgeUID' column")
    edges = edges.rename(columns={"edgeUID": "edge_uid"})
    
    poly = gpd.read_file(back_bay_poly)
    if poly.crs != edges.crs:
        poly = poly.to_crs(edges.crs)
    edges_aoi = gpd.clip(edges, poly.union_all())

    # 2) Load hours (per edge)
    hours = load_hours_csvs(hour_csvs)

    # 3) Keep only edges that appear in hours
    edges_aoi["edge_uid"] = edges_aoi["edge_uid"].astype(str)
    target_edges = edges_aoi.merge(hours[["edge_uid"]].drop_duplicates(), on="edge_uid", how="inner")

    # 4) Densify those edges to points
    pts = densify_lines_to_points(target_edges, spacing_m=spacing_m, edge_id_col="edge_uid")

    # 5) Join points to their **own** hours (per edge), carrying osm_reference_id
    #    (small, per-edge cartesian product)
    pts["edge_uid"] = pts["edge_uid"].astype(str)
    hours["edge_uid"] = hours["edge_uid"].astype(str)
    df = pts.merge(hours, on="edge_uid", how="inner")

    # 6) Save Parquet for the pipeline (columns match your config expectations)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    prepared_out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(prepared_out)
    print(f"Saved {len(gdf):,} rows to {prepared_out}")

if __name__ == "__main__":
    main()
