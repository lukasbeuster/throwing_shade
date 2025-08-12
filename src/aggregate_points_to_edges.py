import geopandas as gpd
import pandas as pd
from pathlib import Path
import sys
import numpy as np

# INPUTS
points_path = Path("../results/output/step6_final_result/cbdb17d4/shaded_dataset.geojson")
edges_path  = Path("../data/clean_data/strava/data_to_share/all_edges_hourly_2024-04-01-2024-04-30_ped_boston_filtered/53f3ef7e32738022bd45a2ed224bcbcf8cb63f07dc3d5bca7e9221dfe7fe4451-1738340934480.shp")  # must have edge_uid (or edgeUID)
stat = "mean"                                # or "median"

# 1) Load
if not points_path.exists():
    raise FileNotFoundError(f"Point output not found at {points_path}. Run process_shade first.")

pts = gpd.read_file(points_path)
edges = gpd.read_file(edges_path)

# Normalize id column name and dtype on edges
if "edge_uid" not in edges.columns and "edgeUID" in edges.columns:
    edges = edges.rename(columns={"edgeUID": "edge_uid"})
if "edge_uid" not in edges.columns:
    raise ValueError("edges file must contain an 'edge_uid' (or 'edgeUID') column")
edges["edge_uid"] = edges["edge_uid"].astype(str)

# Ensure required columns exist in points
if "edge_uid" not in pts.columns and "edgeUID" in pts.columns:
    pts = pts.rename(columns={"edgeUID": "edge_uid"})
if "edge_uid" not in pts.columns:
    raise ValueError("points dataset must include 'edge_uid' so we can aggregate to edges.\n"
                     "If it's missing, make sure your prep step preserved it.")
pts["edge_uid"] = pts["edge_uid"].astype(str)

if "time" not in pts.columns:
    raise ValueError("points dataset must include a 'time' column (hourly timestamps)")
# Parse time if needed
if not pd.api.types.is_datetime64_any_dtype(pts["time"]):
    pts["time"] = pd.to_datetime(pts["time"], errors="coerce")

# 2) Pick shade columns
SHADE_PREFIXES = (
    "building_shade_buffer", "combined_shade_buffer",
    "bldg_shadow_fraction_buffer", "combined_shadow_fraction_buffer",
    "combined_", "bldg_",
)
shade_cols = [c for c in pts.columns if any(c.startswith(p) for p in SHADE_PREFIXES)]
if not shade_cols:
    raise ValueError(
        "No shade columns found in point dataset. Expected columns starting with one of: "
        + ", ".join(SHADE_PREFIXES)
    )

# 3) Aggregate by edge_uid × time
if stat not in {"mean", "median"}:
    raise ValueError("stat must be 'mean' or 'median'")
agg = (
    pts.groupby(["edge_uid", "time"], as_index=False)[shade_cols]
       .agg(stat)
)

# 4) Attach geometries
out = agg.merge(edges[["edge_uid", "geometry"]], on="edge_uid", how="left")
missing_geom = out["geometry"].isna().sum()
if missing_geom:
    print(f"[warn] {missing_geom} edge_uid rows had no matching geometry in edges file.", file=sys.stderr)

gout = gpd.GeoDataFrame(out, geometry="geometry", crs=edges.crs)

# 5) Save
save_path = points_path.parent / f"edges_aggregated_{stat}.geojson"
save_path.parent.mkdir(parents=True, exist_ok=True)
gout.to_file(save_path, driver="GeoJSON")

# 6) Report
print(
    f"Saved: {save_path}\n"
    f"Edges×hours: {len(gout):,}\n"
    f"Shade columns aggregated: {len(shade_cols)}\n"
)