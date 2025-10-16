#!/usr/bin/env python3
"""
Auto-flag (and optionally collapse) cross-OSM-way duplicates using
spatial proximity + area-similarity. Robust to missing geo columns by
auto-detecting lat/lon or joining from a footprints CSV.

Usage examples
--------------
1) Columns already in the detections CSV (e.g., Latitude/Longitude):
   python auto_flag_crossway_duplicates_v2.py \
     --in_csv path/transformer_detections_stabilized.csv \
     --out_flagged_csv path/transformer_detections_stabilized_flagged.csv \
     --out_canonical_map path/osm_to_canonical_map.csv \
     --out_per_canonical path/per_canonical_site.csv \
     --lat_col Latitude --lon_col Longitude

2) No geo columns in detections CSV, but you have footprints with lat/lon:
   python auto_flag_crossway_duplicates_v2.py \
     --in_csv path/transformer_detections_stabilized.csv \
     --out_flagged_csv path/transformer_detections_stabilized_flagged.csv \
     --out_canonical_map path/osm_to_canonical_map.csv \
     --out_per_canonical path/per_canonical_site.csv \
     --footprints_csv path/all_substations_footprint_results.csv \
     --foot_lat_col Latitude --foot_lon_col Longitude

3) Inspect columns quickly (to know what to pass):
   python auto_flag_crossway_duplicates_v2.py --in_csv your.csv --print_columns

Notes
-----
- Distance is computed on per-OSM centroids (mean of available lat/lon rows
  or lat/lon from footprints), then union-find links OSM IDs within a given
  distance whose median transformer areas are similar.
- If no usable lat/lon can be found anywhere, the script will still run but
  will NOT link across ways (it will warn and set duplicate_crossway_flag=False).
"""

import argparse
import math
import sys
import pandas as pd
import numpy as np

# ----------------------- small helpers -----------------------

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

class UF:
    def __init__(self):
        self.p = {}
    def find(self, x):
        if x not in self.p:
            self.p[x] = x
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

# ----------------------- core logic -----------------------

def auto_detect_latlon_cols(df):
    """Return (lat_col, lon_col) if found, else (None, None)."""
    candidates = [
        ("centroid_y_geo", "centroid_x_geo"),  # your earlier pipeline
        ("Latitude", "Longitude"),
        ("latitude", "longitude"),
        ("lat", "lon"),
        ("centroid_lat", "centroid_lon"),
        ("y_geo", "x_geo"),
        ("y", "x"),
    ]
    cols = set(df.columns)
    for lat, lon in candidates:
        if lat in cols and lon in cols:
            return lat, lon
    return None, None

def pick_area_col(df, area_cols):
    for c in area_cols:
        if c in df.columns:
            return c
    return None

# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser(description="Flag/collapse cross-OSM-way duplicates (robust geocolumn handling)")
    ap.add_argument("--in_csv", required=True, help="stabilized detections CSV (row-level)")
    ap.add_argument("--out_flagged_csv", required=False, default=None, help="row-level CSV with flags added")
    ap.add_argument("--out_canonical_map", required=False, default=None, help="mapping CSV (osm_id → canonical_site_id)")
    ap.add_argument("--out_per_canonical", required=False, default=None, help="aggregated per canonical site CSV")
    ap.add_argument("--osm_col", default="osm_id")
    ap.add_argument("--area_cols", default="area_m2_shrunk,area_m2", help="comma-separated preference order")

    # Geo sources
    ap.add_argument("--lat_col", default=None, help="lat column in detections CSV (optional)")
    ap.add_argument("--lon_col", default=None, help="lon column in detections CSV (optional)")
    ap.add_argument("--footprints_csv", default=None, help="Optional CSV with columns [osm_id, Latitude, Longitude] (or custom via flags)")
    ap.add_argument("--foot_osm_col", default="osm_id")
    ap.add_argument("--foot_lat_col", default="Latitude")
    ap.add_argument("--foot_lon_col", default="Longitude")

    # Linking thresholds
    ap.add_argument("--min_rows_per_osm", type=int, default=2)
    ap.add_argument("--max_link_dist_m", type=float, default=150.0)
    ap.add_argument("--min_area_ratio", type=float, default=0.6)
    ap.add_argument("--max_area_ratio", type=float, default=1.6)

    ap.add_argument("--print_columns", action="store_true", help="print available columns and exit")

    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    if args.print_columns:
        print("Columns in", args.in_csv, ":")
        for c in df.columns:
            print(" -", c)
        return

    # check osm col
    if args.osm_col not in df.columns:
        sys.exit(f"Missing column '{args.osm_col}' in {args.in_csv}")

    # decide area column
    area_pref = [c.strip() for c in args.area_cols.split(",") if c.strip()]
    area_col = pick_area_col(df, area_pref)
    if area_col is None:
        sys.exit(f"No area column found. Tried: {area_pref}")

    # find lat/lon in detections CSV or auto-detect
    lat_col = args.lat_col
    lon_col = args.lon_col
    if lat_col is None or lon_col is None:
        lat_col_auto, lon_col_auto = auto_detect_latlon_cols(df)
        if lat_col is None:
            lat_col = lat_col_auto
        if lon_col is None:
            lon_col = lon_col_auto

    # prepare per-OSM summary (n, median area, centroid lat/lon)
    # Start with what we can compute directly from detections CSV
    per = df.groupby(args.osm_col).agg(
        n=("component_id", "count"),
        median_area=(area_col, "median")
    ).reset_index()

    # attach lat/lon from detections CSV if available
    have_csv_latlon = lat_col in df.columns and lon_col in df.columns
    if have_csv_latlon:
        geo_valid = df[lat_col].notna() & df[lon_col].notna()
        per_geo = (df[geo_valid]
                   .groupby(args.osm_col)
                   .agg(mean_lat=(lat_col, "mean"), mean_lon=(lon_col, "mean"))
                   .reset_index())
        per = per.merge(per_geo, on=args.osm_col, how="left")

    # if still missing mean_lat/mean_lon and footprints provided, join from footprints
    if ("mean_lat" not in per.columns or per["mean_lat"].isna().all() or per["mean_lon"].isna().all()) and args.footprints_csv:
        fps = pd.read_csv(args.footprints_csv)
        required = {args.foot_osm_col, args.foot_lat_col, args.foot_lon_col}
        if not required.issubset(set(fps.columns)):
            missing = list(required - set(fps.columns))
            sys.exit(f"Footprints CSV missing columns: {missing}")
        fps_small = fps[[args.foot_osm_col, args.foot_lat_col, args.foot_lon_col]].dropna()
        fps_small = fps_small.rename(columns={
            args.foot_osm_col: args.osm_col,
            args.foot_lat_col: "mean_lat",
            args.foot_lon_col: "mean_lon",
        })
        per = per.merge(fps_small, on=args.osm_col, how="left", suffixes=(None, "_fp"))
        # if both CSV and footprints exist, prefer CSV means where present
        if "mean_lat_x" in per.columns:
            per["mean_lat"] = per["mean_lat_x"].fillna(per["mean_lat_y"])
            per["mean_lon"] = per["mean_lon_x"].fillna(per["mean_lon_y"]) 
            per = per.drop(columns=[c for c in per.columns if c.endswith("_x") or c.endswith("_y")])

    # keep only OSM ids with enough rows
    per = per[per["n"] >= args.min_rows_per_osm].copy()

    # if no usable lat/lon anywhere, we cannot link by distance
    can_link = ("mean_lat" in per.columns and "mean_lon" in per.columns and per["mean_lat"].notna().any() and per["mean_lon"].notna().any())

    if not can_link:
        # produce no-link outputs but keep pipeline consistent
        print("WARNING: No usable lat/lon found in detections CSV or footprints;"
              " cannot link across OSM ways. Flags will be False.")
        # trivial canonical: each osm_id is its own component
        canonical = {osm: osm for osm in per[args.osm_col].tolist()}
    else:
        # Union-find linking using (distance <= max_link_dist_m) AND area ratio within bounds
        uf = UF()
        ids = per[args.osm_col].to_numpy()
        lat = per["mean_lat"].to_numpy()
        lon = per["mean_lon"].to_numpy()
        med = per["median_area"].to_numpy()

        # light-weight bucketing by ~0.001 deg (~100 m)
        bkey = np.floor(lat*1000).astype(int)*10**6 + np.floor(lon*1000).astype(int)
        buckets = {}
        for i, key in enumerate(bkey):
            buckets.setdefault(key, []).append(i)

        def neighbors(idx):
            ky = bkey[idx]
            y0 = ky // 10**6; x0 = ky % 10**6
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    k = (y0+dy)*10**6 + (x0+dx)
                    for j in buckets.get(k, []):
                        yield j

        for i in range(len(ids)):
            for j in neighbors(i):
                if j <= i: 
                    continue
                d = haversine_m(lat[i], lon[i], lat[j], lon[j])
                if d > args.max_link_dist_m:
                    continue
                r = (max(med[i], med[j]) / (min(med[i], med[j]) or 1e-9))
                if args.min_area_ratio <= r <= args.max_area_ratio:
                    uf.union(ids[i], ids[j])

        # build components
        root = {osm: uf.find(osm) for osm in ids}
        comp_members = {}
        for osm, r in root.items():
            comp_members.setdefault(r, []).append(osm)
        canonical_root = {r: min(members) for r, members in comp_members.items()}
        canonical = {osm: canonical_root[root[osm]] for osm in ids}

    # map to all rows
    df["canonical_site_id"] = df[args.osm_col].map(canonical)

    # compute group size per canonical
    grp_sizes = df.groupby("canonical_site_id")[args.osm_col].nunique().rename("canonical_group_size")
    df = df.join(grp_sizes, on="canonical_site_id")
    df["duplicate_crossway_flag"] = df["canonical_group_size"].fillna(1) > 1

    # write outputs
    out_flagged = args.out_flagged_csv or args.in_csv.replace(".csv", "_flagged.csv")
    df.to_csv(out_flagged, index=False)

    # canonical map
    if df["canonical_site_id"].notna().any():
        map_rows = df[[args.osm_col, "canonical_site_id"]].dropna().drop_duplicates()
    else:
        map_rows = pd.DataFrame(columns=[args.osm_col, "canonical_site_id"])
    out_map = args.out_canonical_map or out_flagged.replace("_flagged.csv", "_canonical_map.csv")
    map_rows.to_csv(out_map, index=False)

    # per-canonical aggregate (robust stats for downstream)
    grp = df.groupby(df["canonical_site_id"].fillna(df[args.osm_col]))
    def _cv(s: pd.Series):
        m = s.mean()
        return float(s.std(ddof=1) / m) if m and np.isfinite(m) else np.nan
    per_canon = grp.agg(
        n=("component_id", "count"),
        n_unique_osm=(args.osm_col, "nunique"),
        area_median=(area_col, "median"),
        area_p25=(area_col, lambda s: float(np.quantile(s.dropna(), 0.25)) if s.notna().any() else np.nan),
        area_p75=(area_col, lambda s: float(np.quantile(s.dropna(), 0.75)) if s.notna().any() else np.nan),
        area_cv=(area_col, _cv),
        duplicate_group_flag=("canonical_group_size", lambda s: bool((s.max() or 1) > 1)),
    ).reset_index(names="canonical_site_id")
    out_per = args.out_per_canonical or out_flagged.replace("_flagged.csv", "_per_canonical.csv")
    per_canon.to_csv(out_per, index=False)

    print("Wrote:")
    print(" -", out_flagged)
    print(" -", out_map)
    print(" -", out_per)

if __name__ == "__main__":
    main()
