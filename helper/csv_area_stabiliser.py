#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV-only area stabilization + dedup (no re-inference needed)

What it does
------------
• Reads your existing transformer_detections.csv.
• De-duplicates overlapping detections across tiles per site using a meters radius.
• Produces a stabilized area column via robust per-site shrinkage toward the site median.
• Adds context columns: normalized_area_shrunk, log_area_shrunk, duplicate_flag, cluster_id.
• Writes an improved CSV you can hand to the capacity-classification team now.

Usage
-----
python csv_area_stabilizer.py \
  --csv /path/to/transformer_detections.csv \
  --out /path/to/transformer_detections_stabilized.csv \
  --eps_merge_m 12

Notes
-----
• Works whether your geo coords are degrees (WGS84-like) or projected meters.
• Chooses site key in this order: assigned_osm_ref → osm_id → image_name.
• Keeps the single best candidate per spatial duplicate cluster (prefers higher region_peak_prob, then higher threshold, then area closest to the site median).
"""

from __future__ import annotations
import argparse
import math
import numpy as np
import pandas as pd

# ------------------------- helpers -------------------------

def choose_site_key(df: pd.DataFrame) -> str:
    for k in ["assigned_osm_ref", "osm_id", "image_name"]:
        if k in df.columns and df[k].notna().any():
            return k
    return "image_name"


def looks_like_degrees(x: pd.Series, y: pd.Series) -> bool:
    if x.isna().all() or y.isna().all():
        return False
    try:
        return (x.abs().max() <= 180) and (y.abs().max() <= 90)
    except Exception:
        return False


def meters_to_degrees(lat_deg: float, meters: float) -> float:
    # conservative: use the smaller meters-per-degree to avoid over-merging
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat_deg if lat_deg is not None else 0.0))
    m_per_deg = max(min(m_per_deg_lat, m_per_deg_lon), 1.0)
    return meters / m_per_deg


def pairwise_clusters(x: np.ndarray, y: np.ndarray, eps: float) -> list[list[int]]:
    """Simple O(N^2) single-link clustering with radius eps.
    Returns list of index lists (clusters)."""
    n = len(x)
    if n == 0:
        return []
    used = np.zeros(n, dtype=bool)
    clusters = []
    for i in range(n):
        if used[i]:
            continue
        grp = [i]
        used[i] = True
        # grow cluster by checking neighbors iteratively
        j = 0
        while j < len(grp):
            a = grp[j]
            dx = x - x[a]
            dy = y - y[a]
            d = np.hypot(dx, dy)
            hits = np.where((d <= eps) & (~used))[0]
            for h in hits:
                used[h] = True
                grp.append(h)
            j += 1
        clusters.append(grp)
    return clusters


def select_best(df_grp: pd.DataFrame, site_median: float) -> int:
    """Return index of best row to keep from df_grp (same cluster).
    Priority: region_peak_prob (desc), threshold (desc), |area - site_median| (asc)."""
    cols = df_grp.columns
    has_peak = "region_peak_prob" in cols
    has_thr = "threshold" in cols
    def score_row(r):
        peak = float(r.get("region_peak_prob", 0.0)) if has_peak else 0.0
        thr  = float(r.get("threshold", 0.0)) if has_thr else 0.0
        dist = abs(float(r.get("area_m2", np.nan)) - site_median) if pd.notna(r.get("area_m2", np.nan)) else 1e9
        return (peak, thr, -dist)
    scores = df_grp.apply(score_row, axis=1, result_type="expand")
    # lexicographic: max peak, then max thr, then min |dist|
    order = np.lexsort(( -scores[2].values, -scores[1].values, -scores[0].values ))
    return df_grp.index[ order[0] ]


def stabilize_site(df_site: pd.DataFrame, eps_merge_m: float) -> pd.DataFrame:
    out_rows = []
    xg = df_site.get("centroid_x_geo")
    yg = df_site.get("centroid_y_geo")

    # robust stats for this site
    areas = df_site["area_m2"].astype(float)
    valid = areas.notna() & (areas > 0)
    site_median = float(areas[valid].median()) if valid.any() else np.nan
    q1, q3 = (float(areas[valid].quantile(0.25)), float(areas[valid].quantile(0.75))) if valid.any() else (np.nan, np.nan)
    iqr = (q3 - q1) if (pd.notna(q1) and pd.notna(q3)) else np.nan

    # duplicate clustering
    keep_idx = set()
    dup_idx = set()
    cluster_id = np.full(len(df_site), -1, dtype=int)

    if xg is not None and yg is not None and (xg.notna().any() and yg.notna().any()):
        deg = looks_like_degrees(xg, yg)
        if deg:
            lat = float(yg.median()) if yg.notna().any() else 0.0
            eps = meters_to_degrees(lat, eps_merge_m)
        else:
            eps = eps_merge_m
        # indices within site
        idx = df_site.index.to_numpy()
        X = xg.to_numpy(dtype=float)
        Y = yg.to_numpy(dtype=float)
        clusters = pairwise_clusters(X, Y, eps)
        for cid, grp_local in enumerate(clusters):
            grp_idx = idx[grp_local]
            df_grp = df_site.loc[grp_idx]
            # pick best to keep
            best = select_best(df_grp, site_median if pd.notna(site_median) else 0.0)
            keep_idx.add(best)
            dup_idx.update(set(grp_idx) - {best})
            cluster_id[ [np.where(idx==g)[0][0] for g in grp_idx] ] = cid
    else:
        # no geo → no spatial clustering; keep all
        keep_idx.update(df_site.index.tolist())
        cluster_id[:] = np.arange(len(df_site))

    df_site = df_site.copy()
    df_site["duplicate_flag"] = df_site.index.isin(dup_idx)
    df_site["cluster_id"] = cluster_id

    # shrinkage toward site median (James–Stein-ish but simple & bounded)
    if pd.notna(site_median) and site_median > 0 and valid.any():
        # shrink weight proportional to dispersion (IQR/median) capped at 0.6
        rel_disp = (iqr / site_median) if (pd.notna(iqr) and site_median > 0) else 0.0
        alpha = float(np.clip(rel_disp / 0.6, 0.0, 0.6))
        a = df_site["area_m2"].astype(float)
        a_shrunk = (1 - alpha) * a + alpha * site_median
        # winsorize gently to [q1-1.5*IQR, q3+1.5*IQR]
        lo = q1 - 1.5 * iqr if (pd.notna(q1) and pd.notna(iqr)) else a.min()
        hi = q3 + 1.5 * iqr if (pd.notna(q3) and pd.notna(iqr)) else a.max()
        df_site["area_m2_shrunk"] = a_shrunk.clip(lower=lo, upper=hi)
    else:
        df_site["area_m2_shrunk"] = df_site["area_m2"].astype(float)

    # normalized + log variants for downstream teams
    med_for_norm = df_site["area_m2_shrunk"].median(skipna=True)
    df_site["normalized_area_shrunk"] = df_site["area_m2_shrunk"] / med_for_norm if med_for_norm and med_for_norm>0 else np.nan
    df_site["log_area_shrunk"] = np.log10(df_site["area_m2_shrunk"]) \
        .where(df_site["area_m2_shrunk"]>0, np.nan)

    return df_site


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Stabilize transformer areas from CSV (dedup + shrink)")
    ap.add_argument("--csv", required=True, help="Path to transformer_detections.csv")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--eps_merge_m", type=float, default=12.0, help="Duplicate merge radius in meters")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "area_m2" not in df.columns:
        raise SystemExit("CSV must contain area_m2 column")

    site_key = choose_site_key(df)
    df["__site_key__"] = df[site_key]

    # process per site
    pieces = []
    for site, g in df.groupby("__site_key__", dropna=False):
        pieces.append(stabilize_site(g, eps_merge_m=args.eps_merge_m))
    out = pd.concat(pieces, axis=0, ignore_index=True)

    # quick global metrics
    def robust_stats(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return {"n":0}
        return {
            "n": int(s.size),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std(ddof=1)),
            "cv": float(s.std(ddof=1)/s.mean()) if s.mean()!=0 else np.nan,
            "iqr": float(s.quantile(0.75)-s.quantile(0.25)),
        }

    base = robust_stats(out["area_m2"]) if "area_m2" in out.columns else {"n":0}
    stab = robust_stats(out["area_m2_shrunk"]) if "area_m2_shrunk" in out.columns else {"n":0}
    print("Global area stats → before:", base)
    print("Global area stats → after :", stab)

    # write
    out.to_csv(args.out, index=False)
    print(f"Wrote stabilized CSV → {args.out}")

if __name__ == "__main__":
    main()
