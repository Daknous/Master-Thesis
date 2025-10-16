#!/usr/bin/env python3
import argparse, os, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_csv(path):
    df = pd.read_csv(path)
    # normalize common types
    if 'osm_id' in df.columns:
        df['osm_id'] = pd.to_numeric(df['osm_id'], errors='coerce')
    if 'area_m2' in df.columns:
        df['area_m2'] = pd.to_numeric(df['area_m2'], errors='coerce')
    return df

def basic_overview(df, name):
    out = {}
    out['name'] = name
    out['rows'] = len(df)
    out['unique_images'] = int(df['image_name'].nunique()) if 'image_name' in df.columns else None
    out['unique_sites']  = int(df['osm_id'].nunique()) if 'osm_id' in df.columns else None
    # georef status
    if 'world_file_found' in df.columns:
        out['world_file_found_share'] = float(df['world_file_found'].mean())
    else:
        out['world_file_found_share'] = None
    # area stats
    if 'area_m2' in df.columns:
        s = df['area_m2'].dropna()
        if len(s):
            out['area_m2_mean'] = float(s.mean())
            out['area_m2_median'] = float(s.median())
            out['area_m2_min'] = float(s.min())
            out['area_m2_max'] = float(s.max())
        else:
            out['area_m2_mean'] = out['area_m2_median'] = out['area_m2_min'] = out['area_m2_max'] = None
    return out

def hist_compare(df_base, df_new, outdir, col='area_m2', bins=60, prefix='area'):
    os.makedirs(outdir, exist_ok=True)
    b = df_base[col].dropna().values if col in df_base.columns else np.array([])
    n = df_new[col].dropna().values if col in df_new.columns else np.array([])
    if b.size == 0 and n.size == 0:
        return None
    # shared bins
    all_vals = np.concatenate([b, n]) if b.size and n.size else (b if b.size else n)
    if all_vals.size == 0:
        return None
    lo, hi = np.percentile(all_vals, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = all_vals.min(), all_vals.max()
    edges = np.linspace(lo, hi, bins+1)

    # linear
    plt.figure()
    plt.hist(b, bins=edges, alpha=0.6, label='baseline')
    plt.hist(n, bins=edges, alpha=0.6, label='new')
    plt.xlabel(col); plt.ylabel('count'); plt.title(f'{col} distribution (linear)')
    plt.legend()
    f1 = os.path.join(outdir, f'hist_{prefix}_linear.png')
    plt.savefig(f1, dpi=160, bbox_inches='tight'); plt.close()

    # log
    b_log = np.log10(b[b>0]) if b.size else np.array([])
    n_log = np.log10(n[n>0]) if n.size else np.array([])
    if b_log.size or n_log.size:
        all_log = np.concatenate([b_log, n_log]) if b_log.size and n_log.size else (b_log if b_log.size else n_log)
        lo, hi = np.percentile(all_log, [0.5, 99.5])
        edges = np.linspace(lo, hi, bins+1)
        plt.figure()
        if b_log.size: plt.hist(b_log, bins=edges, alpha=0.6, label='baseline')
        if n_log.size: plt.hist(n_log, bins=edges, alpha=0.6, label='new')
        plt.xlabel(f'log10({col})'); plt.ylabel('count'); plt.title(f'{col} distribution (log10)')
        plt.legend()
        f2 = os.path.join(outdir, f'hist_{prefix}_log.png')
        plt.savefig(f2, dpi=160, bbox_inches='tight'); plt.close()
    else:
        f2 = None

    return {'linear': f1, 'log': f2}

def per_site_stats(df, area_col='area_m2'):
    if 'osm_id' not in df.columns or area_col not in df.columns:
        return pd.DataFrame()
    g = df.groupby('osm_id', dropna=False)[area_col]
    out = g.agg(
        n=('count'),
        median=('median'),
        p25=(lambda s: np.nanpercentile(s,25)),
        p75=(lambda s: np.nanpercentile(s,75)),
        mean=('mean'),
        std=('std')
    ).reset_index()
    out.rename(columns={'n':'n_components'}, inplace=True)
    out['iqr'] = out['p75'] - out['p25']
    out['cv'] = out['std'] / (out['mean'] + 1e-9)
    return out

def correlate_site_medians(base_site, new_site, outdir):
    if base_site.empty or new_site.empty:
        return None
    merged = pd.merge(base_site[['osm_id','median']], new_site[['osm_id','median']],
                      on='osm_id', how='inner', suffixes=('_base','_new'))
    if merged.empty:
        return None
    r = float(np.corrcoef(merged['median_base'], merged['median_new'])[0,1])
    plt.figure()
    plt.scatter(merged['median_base'], merged['median_new'], s=8, alpha=0.5)
    plt.xlabel('site median area_m2 (baseline)')
    plt.ylabel('site median area_m2 (new)')
    plt.title(f'Per-site median correlation (r={r:.3f}, n={len(merged)})')
    lim = [min(merged[['median_base','median_new']].min()), max(merged[['median_base','median_new']].max())]
    plt.plot(lim, lim, 'k--', lw=1)
    f = os.path.join(outdir, 'scatter_site_medians.png')
    plt.savefig(f, dpi=160, bbox_inches='tight'); plt.close()
    return {'r': r, 'plot': f, 'n_sites': int(len(merged))}

def delta_tables(base_site, new_site, outdir, topn=40):
    out = {}
    if base_site.empty or new_site.empty:
        return out
    m = pd.merge(base_site[['osm_id','median','n_components']], new_site[['osm_id','median','n_components']],
                 on='osm_id', how='outer', suffixes=('_base','_new'))
    m['delta_median'] = m['median_new'] - m['median_base']
    m['abs_delta'] = m['delta_median'].abs()
    m['delta_rel'] = m['delta_median'] / (m['median_base'] + 1e-9)

    m.to_csv(os.path.join(outdir, 'per_site_median_delta.csv'), index=False)

    improved = m.sort_values('delta_median', ascending=False).head(topn)
    worsened = m.sort_values('delta_median', ascending=True).head(topn)
    volatile = m.sort_values('abs_delta', ascending=False).head(topn)

    improved.to_csv(os.path.join(outdir, f'top{topn}_improved_sites.csv'), index=False)
    worsened.to_csv(os.path.join(outdir, f'top{topn}_worsened_sites.csv'), index=False)
    volatile.to_csv(os.path.join(outdir, f'top{topn}_most_changed_sites.csv'), index=False)

    out['improved_path'] = os.path.join(outdir, f'top{topn}_improved_sites.csv')
    out['worsened_path'] = os.path.join(outdir, f'top{topn}_worsened_sites.csv')
    out['volatile_path'] = os.path.join(outdir, f'top{topn}_most_changed_sites.csv')
    return out

def feature_coverage(df_new):
    want = [
        'region_peak_prob','region_mean_prob','region_median_prob',
        'area_m2_tminus','area_m2_t','area_m2_tplus','area_slope','area_stability',
        'major_axis_m','minor_axis_m','axis_ratio','perimeter_m','compactness',
        'rectangularity','solidity','orientation_deg',
        'edge_cut_flag','edge_touch_ratio','edge_direction',
        'nn_dist_m','image_row_theta_deg','image_row_anisotropy','image_row_nn_gap_median',
        'pixel_area_simple_m2','pixel_area_det_m2','flag_scale_disagreement',
        'area_m2_subpx','world_is_degrees','world_has_rotation'
    ]
    rows = []
    for c in want:
        if c in df_new.columns:
            nonnull = df_new[c].notna().sum()
            rows.append({'column': c, 'present': True, 'nonnull_count': int(nonnull),
                         'nonnull_share': float(nonnull/len(df_new)) if len(df_new) else None})
        else:
            rows.append({'column': c, 'present': False, 'nonnull_count': 0, 'nonnull_share': 0.0})
    return pd.DataFrame(rows)

def determinant_disagreement(df_new, thr=0.10):
    if 'pixel_area_simple_m2' not in df_new.columns or 'pixel_area_det_m2' not in df_new.columns:
        return None
    s = pd.to_numeric(df_new['pixel_area_simple_m2'], errors='coerce')
    d = pd.to_numeric(df_new['pixel_area_det_m2'], errors='coerce')
    mask = s > 0
    rel = (d[mask] - s[mask]).abs() / s[mask]
    share = float((rel > thr).mean()) if rel.size else None
    top = pd.DataFrame({
        'image_name': df_new.loc[mask, 'image_name'],
        'osm_id': df_new.loc[mask, 'osm_id'],
        'rel_diff': rel
    }).sort_values('rel_diff', ascending=False).head(50)
    return {'share_over_thr': share, 'top_examples': top}

def footprint_section(df_new):
    out = {}
    if 'inside_footprint' in df_new.columns:
        out['inside_footprint_share'] = float(df_new['inside_footprint'].mean())
    if 'crossway_dup_flag' in df_new.columns:
        out['crossway_dup_share'] = float(df_new['crossway_dup_flag'].mean())
    if 'coverage_ratio' in df_new.columns:
        s = pd.to_numeric(df_new['coverage_ratio'], errors='coerce').dropna()
        if len(s):
            out['coverage_ratio_median'] = float(s.median())
            out['coverage_ratio_p90'] = float(np.nanpercentile(s, 90))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True, help='baseline detections CSV')
    ap.add_argument('--new', required=True, help='new detections CSV')
    ap.add_argument('--outdir', required=True, help='output directory for report & plots')
    ap.add_argument('--topn', type=int, default=40)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    base = read_csv(args.baseline)
    new  = read_csv(args.new)

    report = {}
    report['overview'] = {
        'baseline': basic_overview(base, 'baseline'),
        'new': basic_overview(new, 'new')
    }

    # histograms for area_m2
    report['hist_paths'] = hist_compare(base, new, args.outdir, col='area_m2', bins=60, prefix='area_m2')

    # per-site stats & correlation
    base_site = per_site_stats(base, 'area_m2')
    new_site  = per_site_stats(new, 'area_m2')
    base_site.to_csv(os.path.join(args.outdir, 'per_site_baseline.csv'), index=False)
    new_site.to_csv(os.path.join(args.outdir, 'per_site_new.csv'), index=False)

    report['site_correlation'] = correlate_site_medians(base_site, new_site, args.outdir)
    report['delta_tables'] = delta_tables(base_site, new_site, args.outdir, topn=args.topn)

    # feature coverage on new CSV
    cov = feature_coverage(new)
    cov.to_csv(os.path.join(args.outdir, 'new_feature_coverage.csv'), index=False)
    report['new_feature_coverage_nonnull_share_mean'] = float(cov['nonnull_share'].mean())

    # determinant vs simple pixel-area disagreement (if present)
    det = determinant_disagreement(new, thr=0.10)
    if det:
        report['determinant_disagreement_share_over_10pct'] = det['share_over_thr']
        det['top_examples'].to_csv(os.path.join(args.outdir, 'determinant_big_diffs.csv'), index=False)

    # footprint/duplicate extras (if present)
    foot = footprint_section(new)
    report['footprint_section'] = foot

    # write JSON report
    with open(os.path.join(args.outdir, 'compare_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    # lightweight console summary
    print("\n=== Overview ===")
    for k, v in report['overview'].items():
        print(k, v)
    print("\nSite median correlation:", report['site_correlation'])
    print("\nFeature coverage (mean non-null share across new features):",
          report['new_feature_coverage_nonnull_share_mean'])
    if 'determinant_disagreement_share_over_10pct' in report:
        print("\nShare with |det-simple|/simple > 10%:",
              report['determinant_disagreement_share_over_10pct'])
    if report.get('footprint_section'):
        print("\nFootprint section:", report['footprint_section'])
    print("\nArtifacts written to:", args.outdir)

if __name__ == '__main__':
    main()
