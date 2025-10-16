#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare baseline vs. stabilized transformer areas (CSV → report)

Given two CSVs (your original detection export and the stabilized one from
`csv_area_stabilizer.py`), this script computes global and per‑site metrics,
plots histograms, and writes a single `combined_summary.json` describing the
before/after deltas. No re‑inference required.

Usage
-----
python csv_area_compare.py \
  --baseline /path/to/transformer_detections.csv \
  --stabilized /path/to/transformer_detections_stabilized.csv \
  --outdir /path/to/area_compare_report

Optional flags
--------------
  --site_key assigned_osm_ref|osm_id|image_name  # auto‑chosen if omitted
  --bins 80                                      # histogram bins (default 80)

Outputs in --outdir
-------------------
  per_site_stats_baseline.csv
  per_site_stats_stabilized.csv
  per_site_stats_delta.csv                   # sorted by CV improvement
  summary_baseline.json
  summary_stabilized.json
  combined_summary.json                      # baseline vs stabilized deltas
  hist_area_baseline.png / hist_area_stabilized.png
  hist_log_area_baseline.png / hist_log_area_stabilized.png
  hist_norm_area_baseline.png / hist_norm_area_stabilized.png
"""
from __future__ import annotations
import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------- helpers -----------------------------

def choose_site_key(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for k in ["assigned_osm_ref", "osm_id", "image_name"]:
        if k in df.columns and df[k].notna().any():
            return k
    return "image_name"


def as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def add_site_norm(df: pd.DataFrame, site_key: str, area_col: str, out_col: str) -> pd.DataFrame:
    df = df.copy()
    a = as_numeric(df.get(area_col, np.nan))
    df[area_col] = a
    mask = a.notna() & (a > 0)
    if mask.any():
        med = df.loc[mask].groupby(site_key)[area_col].transform("median")
        df[out_col] = np.where(mask, a / med, np.nan)
        df[out_col + "_iqr"] = df.loc[mask].groupby(site_key)[area_col].transform(lambda s: s.quantile(0.75) - s.quantile(0.25))
    else:
        df[out_col] = np.nan
        df[out_col + "_iqr"] = np.nan
    df["log_" + area_col] = np.where(a > 0, np.log10(a), np.nan)
    return df


def series_stats(x: pd.Series) -> Dict:
    x = as_numeric(x).dropna()
    if x.empty:
        return {"n": 0}
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    return {
        "n": int(x.size),
        "mean": mean,
        "median": float(x.median()),
        "std": std,
        "cv": float(std / mean) if mean != 0 else np.nan,
        "iqr": float(q3 - q1),
        "q1": float(q1),
        "q3": float(q3),
        "min": float(x.min()),
        "max": float(x.max()),
        "p95": float(x.quantile(0.95)),
    }


def per_site_stats(df: pd.DataFrame, site_key: str, area_col: str) -> pd.DataFrame:
    def _row(g: pd.DataFrame) -> Dict:
        a = as_numeric(g[area_col])
        a = a[a > 0]
        if a.empty:
            return {"n": 0, "median": np.nan, "iqr": np.nan, "cv": np.nan, "mean": np.nan}
        q1, q3 = a.quantile(0.25), a.quantile(0.75)
        mean = float(a.mean())
        std = float(a.std(ddof=1)) if a.size > 1 else 0.0
        return {
            "n": int(a.size),
            "median": float(a.median()),
            "iqr": float(q3 - q1),
            "cv": float(std / mean) if mean != 0 else np.nan,
            "mean": mean,
        }
    recs = []
    for site, g in df.groupby(site_key, dropna=False):
        r = _row(g)
        r[site_key] = site
        recs.append(r)
    out = pd.DataFrame(recs)
    cols = [site_key, "n", "median", "iqr", "cv", "mean"]
    return out[cols].sort_values("n", ascending=False)


def plot_hist(x: pd.Series, title: str, xlabel: str, out_path: str, bins: int = 80):
    x = as_numeric(x).dropna()
    if x.empty:
        return
    plt.figure(figsize=(8, 6))
    plt.hist(x, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


# ----------------------------- pipeline -----------------------------

def run(baseline_csv: str, stabilized_csv: str, outdir: str, site_key_hint: str | None, bins: int = 80):
    os.makedirs(outdir, exist_ok=True)

    # ---- load ----
    base = pd.read_csv(baseline_csv)
    stab = pd.read_csv(stabilized_csv)

    # ---- site key ----
    site_key_b = choose_site_key(base, preferred=site_key_hint)
    site_key_s = choose_site_key(stab, preferred=site_key_hint)
    # keep names consistent for downstream
    if site_key_b != site_key_s:
        # create a unified column name
        base["__site_key__"] = base[site_key_b]
        stab["__site_key__"] = stab[site_key_s]
        site_key = "__site_key__"
    else:
        site_key = site_key_b

    # ---- add per-site normalization columns ----
    base = add_site_norm(base, site_key, "area_m2", "normalized_area")
    area_stab_col = "area_m2_shrunk" if "area_m2_shrunk" in stab.columns else "area_m2"
    stab = add_site_norm(stab, site_key, area_stab_col, "normalized_area")

    # ---- global summaries ----
    base_sum = {
        "area": series_stats(base["area_m2"]) if "area_m2" in base.columns else {"n": 0},
        "log_area": series_stats(base.get("log_area", base.get("log_area_m2", base["log_area_m2"]) if "log_area_m2" in base.columns else base["log_area_area_m2"]) ) if False else series_stats(base["log_area_m2"]) if "log_area_m2" in base.columns else series_stats(base["log_area"]) if "log_area" in base.columns else series_stats(base["log_area_area_m2"]) if "log_area_area_m2" in base.columns else {},
        "normalized_area": series_stats(base["normalized_area"]) if "normalized_area" in base.columns else {"n": 0},
        "sites": int(base[site_key].nunique()),
    }
    stab_sum = {
        "area": series_stats(stab[area_stab_col]) if area_stab_col in stab.columns else {"n": 0},
        "log_area": series_stats(stab["log_" + area_stab_col]) if ("log_" + area_stab_col) in stab.columns else {"n": 0},
        "normalized_area": series_stats(stab["normalized_area"]) if "normalized_area" in stab.columns else {"n": 0},
        "sites": int(stab[site_key].nunique()),
    }

    # ---- per-site ----
    ps_base = per_site_stats(base, site_key, "area_m2")
    ps_stab = per_site_stats(stab, site_key, area_stab_col)

    # ---- deltas (align on site) ----
    delta = ps_base.merge(ps_stab, on=site_key, how="outer", suffixes=("_base", "_stab"))
    for m in ["median", "iqr", "cv", "mean", "n"]:
        if m + "_base" not in delta.columns:
            delta[m + "_base"] = np.nan
        if m + "_stab" not in delta.columns:
            delta[m + "_stab"] = np.nan
    delta["cv_delta"] = delta["cv_stab"] - delta["cv_base"]
    delta["cv_delta_pct"] = 100.0 * (delta["cv_stab"] - delta["cv_base"]) / delta["cv_base"]
    delta["iqr_delta"] = delta["iqr_stab"] - delta["iqr_base"]
    delta["median_delta"] = delta["median_stab"] - delta["median_base"]
    delta_sorted = delta.sort_values(["cv_delta", "iqr_delta"]).reset_index(drop=True)

    # ---- write per-site tables ----
    ps_base.to_csv(os.path.join(outdir, "per_site_stats_baseline.csv"), index=False)
    ps_stab.to_csv(os.path.join(outdir, "per_site_stats_stabilized.csv"), index=False)
    delta_sorted.to_csv(os.path.join(outdir, "per_site_stats_delta.csv"), index=False)

    # ---- write JSON summaries ----
    with open(os.path.join(outdir, "summary_baseline.json"), "w") as f:
        json.dump(base_sum, f, indent=2)
    with open(os.path.join(outdir, "summary_stabilized.json"), "w") as f:
        json.dump(stab_sum, f, indent=2)

    combined = {
        "site_key": site_key,
        "area_col_baseline": "area_m2",
        "area_col_stabilized": area_stab_col,
        "global": {
            "cv_area": {
                "baseline": base_sum["area"].get("cv"),
                "stabilized": stab_sum["area"].get("cv"),
                "delta": (stab_sum["area"].get("cv") - base_sum["area"].get("cv")) if base_sum["area"].get("cv") is not None else None,
                "delta_pct": (100.0 * (stab_sum["area"].get("cv") - base_sum["area"].get("cv")) / base_sum["area"].get("cv") ) if base_sum["area"].get("cv") not in (None, 0, np.nan) else None,
            },
            "iqr_area": {
                "baseline": base_sum["area"].get("iqr"),
                "stabilized": stab_sum["area"].get("iqr"),
                "delta": (stab_sum["area"].get("iqr") - base_sum["area"].get("iqr")) if base_sum["area"].get("iqr") is not None else None,
            },
            "std_norm_area": {
                "baseline": base_sum["normalized_area"].get("std"),
                "stabilized": stab_sum["normalized_area"].get("std"),
                "delta": (stab_sum["normalized_area"].get("std") - base_sum["normalized_area"].get("std")) if base_sum["normalized_area"].get("std") is not None else None,
            },
            "iqr_norm_area": {
                "baseline": base_sum["normalized_area"].get("iqr"),
                "stabilized": stab_sum["normalized_area"].get("iqr"),
                "delta": (stab_sum["normalized_area"].get("iqr") - base_sum["normalized_area"].get("iqr")) if base_sum["normalized_area"].get("iqr") is not None else None,
            },
            "sites": {
                "baseline": base_sum.get("sites"),
                "stabilized": stab_sum.get("sites"),
            }
        }
    }
    with open(os.path.join(outdir, "combined_summary.json"), "w") as f:
        json.dump(combined, f, indent=2)

    # ---- plots ----
    plot_hist(base["area_m2"], "Area (m²) - baseline", "area_m2", os.path.join(outdir, "hist_area_baseline.png"), bins=bins)
    plot_hist(stab[area_stab_col], "Area (m²) - stabilized", area_stab_col, os.path.join(outdir, "hist_area_stabilized.png"), bins=bins)

    plot_hist(base["log_area_m2"] if "log_area_m2" in base.columns else np.log10(as_numeric(base["area_m2"]).where(as_numeric(base["area_m2"])>0)),
              "log10(Area) - baseline", "log10(area_m2)", os.path.join(outdir, "hist_log_area_baseline.png"), bins=bins)
    plot_hist(stab["log_" + area_stab_col] if ("log_" + area_stab_col) in stab.columns else np.log10(as_numeric(stab[area_stab_col]).where(as_numeric(stab[area_stab_col])>0)),
              "log10(Area) - stabilized", "log10(area)", os.path.join(outdir, "hist_log_area_stabilized.png"), bins=bins)

    plot_hist(base["normalized_area"], "Normalized area (÷ site median) - baseline", "normalized_area", os.path.join(outdir, "hist_norm_area_baseline.png"), bins=bins)
    plot_hist(stab["normalized_area"], "Normalized area (÷ site median) - stabilized", "normalized_area", os.path.join(outdir, "hist_norm_area_stabilized.png"), bins=bins)

    print("Wrote per-site tables + summaries + plots to:", outdir)


# ----------------------------- CLI -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare baseline vs stabilized transformer areas (CSV)")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--stabilized", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--site_key", default=None)
    ap.add_argument("--bins", type=int, default=80)
    args = ap.parse_args()

    run(args.baseline, args.stabilized, args.outdir, args.site_key, bins=args.bins)
