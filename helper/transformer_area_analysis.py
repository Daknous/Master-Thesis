#!/usr/bin/env python3
import argparse
import os
import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- Helpers --------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters between two (lat, lon) points (degrees)."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def choose_site_key(df):
    """Pick the most reliable site/group key available for normalization & dedup."""
    if 'assigned_osm_ref' in df.columns and df['assigned_osm_ref'].notna().any():
        return 'assigned_osm_ref'
    if 'osm_id' in df.columns and df['osm_id'].notna().any():
        return 'osm_id'
    return 'image_name'  # fallback

def add_context_columns(df):
    """Add normalized_area, log_area, site median columns. Purely derived; no classification."""
    if df.empty or 'area_m2' not in df.columns:
        return df
    site_key = choose_site_key(df)
    mask = df['area_m2'].notna() & (df['area_m2'] > 0)
    df.loc[mask, 'site_area_median_m2'] = df.loc[mask].groupby(site_key)['area_m2'].transform('median')
    df['normalized_area'] = np.where(mask, df['area_m2'] / df['site_area_median_m2'], np.nan)
    df['log_area'] = np.where(mask, np.log10(df['area_m2']), np.nan)
    df['normalization_site_key'] = site_key
    return df

def looks_degrees_row(row):
    """Try to detect if centroid coords are likely degrees (EPSG:4326)."""
    yg = row.get('centroid_y_geo', None)
    xg = row.get('centroid_x_geo', None)
    if yg is None or xg is None:
        return None
    try:
        yg = float(yg); xg = float(xg)
    except Exception:
        return None
    return (abs(yg) <= 90.0) and (abs(xg) <= 180.0)

def pairwise_dedup_indices(points, thr_m):
    """
    Simple connected components on pairwise <= thr_m.
    points: list of (x,y,'deg'|None) in meters-space only if 'deg', we compute haversine internally.
    We return indices to KEEP (one per group), favouring larger/cleaner areas if provided later.
    """
    n = len(points)
    if n == 0:
        return []

    # Build adjacency
    adj = [[] for _ in range(n)]
    for i in range(n):
        xi, yi, deg_i = points[i]
        for j in range(i+1, n):
            xj, yj, deg_j = points[j]
            d = None
            if deg_i and deg_j:
                # Interpreting as (lon,lat) pairs: points stored as (x, y) == (lon, lat)
                d = haversine_m(yi, xi, yj, xj)  # (lat1, lon1, lat2, lon2)
            else:
                # Assume planar units are meters (best effort); if not, skip dedup
                if (deg_i is None) and (deg_j is None):
                    # both projected → treat as meters
                    d = math.hypot(xi - xj, yi - yj)
                else:
                    # mixed CRS → can't compare safely
                    d = None

            if d is not None and d <= thr_m:
                adj[i].append(j)
                adj[j].append(i)

    # Connected components via DFS
    visited = [False]*n
    groups = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        groups.append(comp)

    # Keep the "best" index in each group (we'll decide later using a scoring function)
    # For now, just keep the first; caller can re-score.
    keep = [g[0] for g in groups]
    return keep, groups

def dedup_within_site(df_site, site_key, thr_m=10.0):
    """
    Deduplicate detections within a site based on centroid geo distance (meters).
    Returns a filtered df_site and mapping of old->keep index.
    """
    if df_site.empty:
        return df_site.copy(), []

    # Build points list; detect CRS per row
    pts = []
    for _, r in df_site.iterrows():
        xg = r.get('centroid_x_geo', None)
        yg = r.get('centroid_y_geo', None)
        if pd.isna(xg) or pd.isna(yg):
            pts.append((None, None, None))
            continue
        try:
            xg = float(xg); yg = float(yg)
        except Exception:
            pts.append((None, None, None)); continue
        is_deg = looks_degrees_row(r)
        pts.append((xg, yg, bool(is_deg) if is_deg is not None else None))

    # If no valid coords, return as-is
    if all(p[0] is None for p in pts):
        return df_site.copy(), list(range(len(df_site)))

    # Compute groups
    keep_idx_guess, groups = pairwise_dedup_indices(pts, thr_m=thr_m)

    # Score members within each group to choose the best detection:
    # prefer higher region_peak_prob > higher region_median_prob > higher threshold > area close to 100 m2 (arbitrary tie-break)
    scores = []
    for i, r in enumerate(df_site.itertuples(index=False)):
        peak = getattr(r, 'region_peak_prob', np.nan)
        med  = getattr(r, 'region_median_prob', np.nan)
        thr  = getattr(r, 'threshold', np.nan)
        area = getattr(r, 'area_m2', np.nan)
        score = (
            (peak if pd.notna(peak) else 0.0),
            (med  if pd.notna(med)  else 0.0),
            (thr  if pd.notna(thr)  else 0.0),
            -(abs((area if pd.notna(area) else 100.0) - 100.0))
        )
        scores.append((i, score))

    keep_final = set()
    for g in groups:
        if len(g) == 1:
            keep_final.add(g[0])
        else:
            # pick max by score
            best = max(g, key=lambda k: scores[k][1])
            keep_final.add(best)

    # Filter
    df_site = df_site.reset_index(drop=True)
    keep_mask = [i in keep_final for i in range(len(df_site))]
    return df_site.loc[keep_mask].copy(), sorted(list(keep_final))

def summarise(df, outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    summary = {}

    total = len(df)
    sites = df['__site_key__'].nunique() if '__site_key__' in df.columns else np.nan
    summary['rows'] = int(total)
    summary['sites'] = int(sites) if not pd.isna(sites) else None

    # Global area stats
    mask = df['area_m2'].notna() & (df['area_m2'] > 0)
    if mask.any():
        arr = df.loc[mask, 'area_m2'].values
        summary['area_m2_mean'] = float(np.mean(arr))
        summary['area_m2_median'] = float(np.median(arr))
        summary['area_m2_std'] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        q25, q75 = np.percentile(arr, [25, 75])
        summary['area_m2_iqr'] = float(q75 - q25)
        summary['area_m2_min'] = float(arr.min())
        summary['area_m2_max'] = float(arr.max())

    # Normalized stats (if present/added)
    if 'normalized_area' in df.columns:
        na = df['normalized_area'].dropna().values
        if na.size > 0:
            summary['normalized_area_median'] = float(np.median(na))
            summary['normalized_area_std'] = float(np.std(na, ddof=1)) if len(na) > 1 else 0.0
            q25, q75 = np.percentile(na, [25, 75])
            summary['normalized_area_iqr'] = float(q75 - q25)

    # Dispersion proxy: Coefficient of Variation (CV) across all detections
    if mask.any():
        arr = df.loc[mask, 'area_m2'].values
        mu = np.mean(arr)
        cv = (np.std(arr, ddof=1) / mu) if mu > 0 and len(arr) > 1 else np.nan
        summary['cv_area_m2'] = float(cv) if not pd.isna(cv) else None
    if 'normalized_area' in df.columns:
        na = df['normalized_area'].dropna().values
        mu = np.mean(na) if na.size > 0 else np.nan
        cv = (np.std(na, ddof=1) / mu) if na.size > 1 and not pd.isna(mu) and mu > 0 else np.nan
        summary['cv_normalized_area'] = float(cv) if not pd.isna(cv) else None

    # Save JSON summary
    with open(os.path.join(outdir, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-site stats table
    per_site = []
    site_key = df['__site_key__'].name if isinstance(df['__site_key__'], pd.Series) else '__site_key__'
    sites = df['__site_key__'].nunique()
    if site_key is not None:
        for s, g in df.groupby(site_key):
            m = g['area_m2'].dropna()
            if m.empty:
                continue
            row = {
                'site': s,
                'n': int(len(g)),
                'area_m2_median': float(m.median()),
                'area_m2_iqr': float(np.percentile(m, 75) - np.percentile(m, 25)),
                'area_m2_cv': float(m.std(ddof=1) / m.mean()) if len(m) > 1 and m.mean() > 0 else np.nan
            }
            if 'normalized_area' in g.columns:
                na = g['normalized_area'].dropna()
                if not na.empty:
                    row['normalized_area_median'] = float(np.median(na))
                    row['normalized_area_iqr'] = float(np.percentile(na, 75) - np.percentile(na, 25))
                    row['normalized_area_cv'] = float(na.std(ddof=1) / na.mean()) if len(na) > 1 and na.mean() > 0 else np.nan
            per_site.append(row)
        pd.DataFrame(per_site).to_csv(os.path.join(outdir, f"per_site_stats_{tag}.csv"), index=False)

    # Plots (1 chart per figure, matplotlib defaults only)
    if mask.any():
        plt.figure()
        plt.hist(df.loc[mask, 'area_m2'].values, bins=60)
        plt.title(f"Area (m²) - {tag}")
        plt.xlabel("area_m2")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"hist_area_m2_{tag}.png"))
        plt.close()

    if 'log_area' in df.columns and df['log_area'].notna().any():
        plt.figure()
        plt.hist(df['log_area'].dropna().values, bins=60)
        plt.title(f"log10(Area) - {tag}")
        plt.xlabel("log10(area_m2)")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"hist_log_area_{tag}.png"))
        plt.close()

    if 'normalized_area' in df.columns and df['normalized_area'].notna().any():
        plt.figure()
        plt.hist(df['normalized_area'].dropna().values, bins=60)
        plt.title(f"Normalized area (÷ site median) - {tag}")
        plt.xlabel("normalized_area")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"hist_normalized_area_{tag}.png"))
        plt.close()

    return summary

def load_and_prepare(path):
    df = pd.read_csv(path)
    # Standardize required columns if possible
    if 'area_m2' not in df.columns and 'area_sq_m' in df.columns:
        df = df.rename(columns={'area_sq_m': 'area_m2'})
    # Create site key and add derived cols
    site_key = choose_site_key(df)
    df['__site_key__'] = df[site_key].fillna('UNK')
    df = add_context_columns(df)
    return df

def optional_dedup(df, dedup_m=10.0):
    """Apply near-duplicate removal per site in-place (non-destructive copy returned)."""
    site_key = choose_site_key(df)
    parts = []
    for s, g in df.groupby(site_key):
        g2, keep_idx = dedup_within_site(g, site_key, thr_m=dedup_m)
        parts.append(g2)
    out = pd.concat(parts, ignore_index=True) if parts else df.copy()
    return out

# -------------------------- Main --------------------------

def main():
    ap = argparse.ArgumentParser(description="Analyse transformer areas CSV (before/after tweaks).")
    ap.add_argument("--baseline", required=True, help="Path to baseline CSV (current results).")
    ap.add_argument("--post", help="Path to post-tweak CSV (optional).")
    ap.add_argument("--outdir", default="./area_report", help="Output directory for figures and summaries.")
    ap.add_argument("--dedup_m", type=float, default=10.0, help="Deduplication distance threshold in meters (per site).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Baseline
    base_df = load_and_prepare(args.baseline)
    base_df_dedup = optional_dedup(base_df, dedup_m=args.dedup_m)
    base_sum = summarise(base_df_dedup, args.outdir, tag="baseline")

    # Write a cleaned baseline CSV with derived columns for downstream use
    base_out_csv = os.path.join(args.outdir, "baseline_cleaned.csv")
    base_df_dedup.to_csv(base_out_csv, index=False)

    combined = {"baseline": base_sum}

    # Post-tweak (optional)
    if args.post and os.path.exists(args.post):
        post_df = load_and_prepare(args.post)
        post_df_dedup = optional_dedup(post_df, dedup_m=args.dedup_m)
        post_sum = summarise(post_df_dedup, args.outdir, tag="post")
        post_out_csv = os.path.join(args.outdir, "post_cleaned.csv")
        post_df_dedup.to_csv(post_out_csv, index=False)
        combined["post"] = post_sum

        # Headline comparison metrics
        comp = {}
        def get(d, k): return d.get(k) if d else None
        comp["cv_area_reduction_abs"] = (get(base_sum, "cv_area_m2") - get(post_sum, "cv_area_m2")) if get(base_sum, "cv_area_m2") is not None and get(post_sum, "cv_area_m2") is not None else None
        comp["cv_norm_area_reduction_abs"] = (get(base_sum, "cv_normalized_area") - get(post_sum, "cv_normalized_area")) if get(base_sum, "cv_normalized_area") is not None and get(post_sum, "cv_normalized_area") is not None else None
        combined["comparison"] = comp

    # Save combined JSON
    with open(os.path.join(args.outdir, "combined_summary.json"), "w") as f:
        json.dump(combined, f, indent=2)

    print("Done. Outputs in:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
