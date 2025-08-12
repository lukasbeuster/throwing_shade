# src/analyze_edges_shade.py
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ---- Inputs ----
RESULTS_DIR = Path("../results/output/step6_final_result/cbdb17d4")  # adjust run id if needed
AGG_FILE    = RESULTS_DIR / "edges_aggregated_mean.geojson"          # or ..._median.geojson
OUT_DIR     = RESULTS_DIR / "qc_plots"
N_SAMPLE_EDGES = 6   # how many random edges to plot time series for

# ---- Load ----
if not AGG_FILE.exists():
    raise FileNotFoundError(f"Missing {AGG_FILE}")
gdf = gpd.read_file(AGG_FILE)

# ---- Detect shade columns ----
SHADE_PREFIXES = (
    "building_shade_buffer", "combined_shade_buffer",
    "bldg_shadow_fraction_buffer", "combined_shadow_fraction_buffer",
    "combined_", "bldg_"
)
shade_cols = [c for c in gdf.columns if any(c.startswith(p) for p in SHADE_PREFIXES)]
if not shade_cols:
    raise ValueError("No shade columns found.")
print(f"Detected {len(shade_cols)} shade columns:\n  " + "\n  ".join(shade_cols))

# ---- Basic shape / coverage ----
print("\n=== Shape ===")
print(gdf.shape, "rows, cols")
print("Unique edges:", gdf["edge_uid"].nunique())
print("Unique timestamps:", gdf["time"].nunique())

# Ensure datetime
if not pd.api.types.is_datetime64_any_dtype(gdf["time"]):
    gdf["time"] = pd.to_datetime(gdf["time"], errors="coerce")

# ---- Overall stats for each shade column ----
print("\n=== Overall stats (each column) ===")
desc = gdf[shade_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
print(desc[["count","mean","std","min","5%","25%","50%","75%","95%","max"]])

# Share of NaNs
nan_share = gdf[shade_cols].isna().mean().sort_values(ascending=False)
print("\nNaN share per column:")
print(nan_share)

# ---- Hourly coverage & sanity (min/max per hour) ----
per_hour = (gdf
            .groupby("time")[shade_cols]
            .agg(["count","mean","min","max"]))
print("\n=== Per-hour summary (head) ===")
print(per_hour.head())

# ---- Save simple distributions ----
OUT_DIR.mkdir(parents=True, exist_ok=True)
for c in shade_cols:
    fig = plt.figure()
    gdf[c].dropna().hist(bins=40)
    plt.title(f"Histogram: {c}")
    plt.xlabel(c); plt.ylabel("Count")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"hist_{c}.png", dpi=150)
    plt.close(fig)

# ---- Time series for a few random edges (for first shade col) ----
first = shade_cols[0]
sample_edges = (gdf.dropna(subset=[first])
                  .drop_duplicates(subset=["edge_uid"])[ "edge_uid" ]
                  .sample(min(N_SAMPLE_EDGES, gdf["edge_uid"].nunique()), random_state=42)
                  .tolist())

fig = plt.figure(figsize=(10, 6))
for eid in sample_edges:
    s = (gdf.loc[gdf["edge_uid"]==eid, ["time", first]]
            .dropna()
            .sort_values("time"))
    plt.plot(s["time"], s[first], marker="o", linewidth=1, label=str(eid))
plt.title(f"Time series on {len(sample_edges)} edges — {first}")
plt.xlabel("Time"); plt.ylabel(first)
plt.legend(fontsize=7, ncol=2)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT_DIR / f"timeseries_{first}.png", dpi=150)
plt.close(fig)

# ---- Quick flags: values outside [0,1] for fraction-like columns ----
def looks_like_fraction(colname: str) -> bool:
    return "fraction" in colname or "combined_" in colname or "bldg_" in colname

suspects = {}
for c in shade_cols:
    if looks_like_fraction(c):
        s = gdf[c]
        bad = gdf.loc[(s.notna()) & ((s < 0) | (s > 1)), ["edge_uid","time",c]]
        if not bad.empty:
            suspects[c] = bad
            bad.to_csv(OUT_DIR / f"out_of_range_{c}.csv", index=False)

if suspects:
    print("\n[WARN] Found values outside [0,1] (saved CSVs in qc_plots/):")
    for c, dfbad in suspects.items():
        print(f"  {c}: {len(dfbad)} rows")
else:
    print("\nNo out-of-range [0,1] values detected in fraction-like columns.")

# ---- MAP SNAPSHOT & HOUR×EDGE HEATMAP ----
# Prefer a combined-shade column for mapping; fall back to the first shade column
combined_candidates = [
    c for c in shade_cols if c.startswith("combined_shade_buffer") or c.startswith("combined_")
]
if combined_candidates:
    # Prefer buffer0 if present, else the smallest buffer number
    def _bufnum(name: str) -> int:
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else 9999
    target_col = sorted(combined_candidates, key=_bufnum)[0]
else:
    target_col = first

# Choose the timestamp with the most non-NaN edge values in target_col (best coverage hour)
hour_counts = (
    gdf.groupby('time')[target_col]
       .apply(lambda s: s.notna().sum())
       .sort_values(ascending=False)
)
if hour_counts.empty or hour_counts.iloc[0] == 0:
    print("[warn] No non-NaN values found to select a map hour; skipping map snapshot and heatmap.")
else:
    # Helper to render a full-network map for a given timestamp
    def _render_map_for_time(ts, colname):
        # Build unique edges geometry (one geometry per edge_uid)
        edges_unique = (gdf[['edge_uid', 'geometry']]
                          .drop_duplicates(subset=['edge_uid'])
                          .copy())
        edges_unique = gpd.GeoDataFrame(edges_unique, geometry='geometry', crs=gdf.crs)

        # Values at this time (mean per edge_uid if duplicates)
        vals = (gdf.loc[gdf['time'] == ts, ['edge_uid', colname]]
                  .groupby('edge_uid', as_index=False)[colname].mean())

        edges_join = edges_unique.merge(vals, on='edge_uid', how='left')

        fig, ax = plt.subplots(figsize=(11, 9))
        # Base: all edges in light grey
        edges_unique.plot(ax=ax, linewidth=0.7, color='#dddddd')
        # Overlay: edges with data, colored by combined shade
        edges_with_vals = edges_join.dropna(subset=[colname])
        vmin, vmax = (0.0, 1.0) if looks_like_fraction(colname) else (None, None)
        edges_with_vals.plot(column=colname, ax=ax, linewidth=1.3, legend=True, vmin=vmin, vmax=vmax)
        ax.set_title(f"Edges — {pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M')} — {colname}")
        ax.set_axis_off()
        fig.tight_layout()
        mp = OUT_DIR / f"map_{colname}_{pd.to_datetime(ts).strftime('%Y%m%d_%H%M')}.png"
        fig.savefig(mp, dpi=180)
        plt.close(fig)
        print(f"Saved map snapshot: {mp}")

    # Select multiple times from the same day as `chosen_time`
    chosen_time = hour_counts.index[0]  # Select the hour with the most non-NaN values
    chosen_date = pd.to_datetime(chosen_time).date()
    same_day_times = sorted([t for t in hour_counts.index if pd.to_datetime(t).date() == chosen_date])

    # Keep only times that have non-NaN values in target_col
    same_day_times = [t for t in same_day_times
                      if gdf.loc[gdf['time'] == t, target_col].notna().any()]

    # Pick up to 4 evenly spaced times across that day
    K = 4
    if len(same_day_times) > K:
        idxs = np.linspace(0, len(same_day_times)-1, K).round().astype(int)
        selected_times = [same_day_times[i] for i in idxs]
    else:
        selected_times = same_day_times

    # Render maps for selected times
    for ts in selected_times:
        _render_map_for_time(ts, target_col)

    # 2) Hour × Edge heatmap (values of target_col)
    heat = gdf.pivot_table(index='edge_uid', columns='time', values=target_col, aggfunc='mean')
    # Order by time
    heat = heat.reindex(sorted(heat.columns), axis=1)

    fig = plt.figure(figsize=(12, 6))
    vmin, vmax = (0.0, 1.0) if looks_like_fraction(target_col) else (np.nanmin(heat.values), np.nanmax(heat.values))
    im = plt.imshow(heat.values, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im)
    cbar.set_label(target_col)
    plt.title(f"Hour × Edge heatmap — {target_col}")
    # X ticks: show up to ~12 labels evenly spaced
    n_cols = heat.shape[1]
    if n_cols > 0:
        step = max(1, n_cols // 12)
        xticks = np.arange(0, n_cols, step)
        xticklabels = [pd.to_datetime(t).strftime('%m-%d %H:%M') for t in heat.columns[::step]]
        plt.xticks(xticks, xticklabels, rotation=45, ha='right')
    # Y ticks: too many edges; show none (keeps figure readable)
    plt.yticks([])
    fig.tight_layout()
    hm_path = OUT_DIR / f"heatmap_{target_col}.png"
    fig.savefig(hm_path, dpi=180)
    plt.close(fig)

    print(f"Saved heatmap:      {hm_path}")

print(f"\nSaved plots in: {OUT_DIR}")