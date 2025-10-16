# ==== Production Batch Processing Script - Baseline-Preserving + Rich Features ====
import os
import glob
import json
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import logging
from skimage import filters, measure, feature, draw
from skimage.segmentation import watershed, relabel_sequential, find_boundaries
from skimage.morphology import remove_small_objects
from skimage.transform import resize as sk_resize
from skimage.measure import find_contours
import torch
import warnings

# Optional: shapely for tighter rectangularity; code falls back if unavailable
try:
    from shapely.geometry import Polygon
    from shapely import affinity as _aff
    _HAVE_SHAPELY = True
except Exception:
    _HAVE_SHAPELY = False

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ------------------------------ Segmentation ------------------------------

class TransformerSegmentationPipeline:
    """Baseline-preserving transformer segmentation pipeline (unchanged behavior)."""

    def __init__(self, constraints=None):
        self.constraints = constraints or {
            "min_transformer_m2": 50.0,
            "typical_transformer_m2": 150.0,
            "max_transformer_m2": 600.0,
            "min_separation_m": 5.0,
            "typical_separation_m": 10.0,
            "fragment_threshold_m2": 50.0,
            "proximity_threshold_m": 3.0,
            "max_merge_distance_m": 15.0
        }

    def _m2_to_px(self, m2, area_per_px_m2):
        if not np.isfinite(area_per_px_m2) or area_per_px_m2 <= 0:
            return 40  # Fallback default
        return max(1, int(round(m2 / area_per_px_m2)))

    def process(self, probs, area_per_px_m2):
        try:
            if not self._validate_inputs(probs, area_per_px_m2):
                return np.zeros_like(probs, dtype=np.int32), {"error": "Invalid inputs"}

            threshold = self._adaptive_threshold(probs)
            mask_lo = self._create_mask(probs, threshold, area_per_px_m2)
            seeds = self._generate_seeds(probs, mask_lo, area_per_px_m2)
            labeled = self._watershed_segmentation(probs, mask_lo, seeds)
            labeled = self._conservative_merge(labeled, probs, area_per_px_m2)
            labeled = self._aggressive_merge_nearby(labeled, area_per_px_m2)
            labeled = self._merge_fragments(labeled, area_per_px_m2)
            labeled = self._final_cleanup(labeled, area_per_px_m2)

            info = {
                "threshold": float(threshold),
                "n_seeds": int(seeds.max()),
                "n_final": int(labeled.max()),
                "success": True
            }
            return labeled, info

        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            return np.zeros_like(probs, dtype=np.int32), {"error": str(e), "success": False}

    def _validate_inputs(self, probs, area_per_px_m2):
        if probs is None or probs.size == 0:
            logger.error("Invalid probability map")
            return False
        if not np.isfinite(area_per_px_m2) or area_per_px_m2 <= 0:
            logger.warning(f"Invalid area_per_px_m2: {area_per_px_m2}, will use defaults")
        return True

    def _adaptive_threshold(self, probs):
        from skimage.filters import threshold_otsu
        p_finite = probs[np.isfinite(probs)]
        if p_finite.size < 100:
            return 0.15
        try:
            t = threshold_otsu(p_finite)
            return float(np.clip(t, 0.08, 0.35))
        except Exception:
            return 0.15

    def _create_mask(self, probs, threshold, area_per_px_m2):
        mask = (probs > threshold).astype(np.uint8)
        min_area_m2 = max(0.25 * self.constraints["min_transformer_m2"], 20.0)
        min_area_px = self._m2_to_px(min_area_m2, area_per_px_m2)
        return remove_small_objects(mask.astype(bool), min_size=min_area_px).astype(np.uint8)

    def _generate_seeds(self, probs, mask_lo, area_per_px_m2):
        if not mask_lo.any():
            return np.zeros_like(mask_lo, dtype=np.int32)
        p_masked = probs[mask_lo.astype(bool)]
        p_max = p_masked.max() if p_masked.size > 0 else 0
        if p_max > 0.8:
            sigma, min_dist_factor = 0.5, 0.8
        elif p_max < 0.5:
            sigma, min_dist_factor = 1.5, 1.5
        else:
            sigma, min_dist_factor = 1.0, 1.0
        p_smooth = filters.gaussian(probs, sigma=sigma, preserve_range=True)
        if np.isfinite(area_per_px_m2) and area_per_px_m2 > 0:
            typical_sep_m = self.constraints["typical_separation_m"] * min_dist_factor
            min_dist_px = int(np.clip(typical_sep_m / np.sqrt(area_per_px_m2), 5, 40))
        else:
            min_dist_px = 20
        coords = feature.peak_local_max(
            p_smooth,
            labels=mask_lo.astype(np.uint8),
            min_distance=min_dist_px,
            exclude_border=False,
            threshold_rel=0.3
        )
        markers = np.zeros_like(mask_lo, dtype=np.int32)
        for i, (r, c) in enumerate(coords, 1):
            markers[r, c] = i
        return markers

    def _watershed_segmentation(self, probs, mask_lo, seeds):
        if seeds.max() > 0:
            gradient = filters.sobel(filters.gaussian(probs, sigma=1.0))
            return watershed(gradient, markers=seeds, mask=mask_lo.astype(bool))
        else:
            return measure.label(mask_lo, connectivity=2)

    def _conservative_merge(self, labeled, probs, area_per_px_m2):
        if labeled.max() < 2:
            return labeled
        props = measure.regionprops(labeled, intensity_image=probs)
        px_per_m = 1.0 / np.sqrt(area_per_px_m2) if area_per_px_m2 > 0 else 30
        merge_distance_px = self.constraints["proximity_threshold_m"] * px_per_m
        edges = filters.sobel(probs)
        merged = labeled.copy()
        for i, p1 in enumerate(props):
            for p2 in props[i+1:]:
                dist = np.hypot(p1.centroid[0] - p2.centroid[0], p1.centroid[1] - p2.centroid[1])
                if dist < merge_distance_px:
                    rr, cc = draw.line(int(p1.centroid[0]), int(p1.centroid[1]),
                                       int(p2.centroid[0]), int(p2.centroid[1]))
                    valid = (rr >= 0) & (rr < edges.shape[0]) & (cc >= 0) & (cc < edges.shape[1])
                    rr, cc = rr[valid], cc[valid]
                    if rr.size == 0:
                        continue
                    edge_strength = edges[rr, cc].mean()
                    prob_continuity = probs[rr, cc].min() / max(p1.mean_intensity, p2.mean_intensity, 1e-6)
                    if edge_strength < 0.2 and prob_continuity > 0.7:
                        if p1.area >= p2.area:
                            merged[merged == p2.label] = p1.label
                        else:
                            merged[merged == p1.label] = p2.label
        merged, _, _ = relabel_sequential(merged)
        return merged

    def _aggressive_merge_nearby(self, labeled, area_per_px_m2):
        if labeled.max() < 2:
            return labeled
        px_per_m = 1.0 / np.sqrt(area_per_px_m2) if area_per_px_m2 > 0 else 30
        aggressive_merge_dist = 2.5 * px_per_m
        merged = labeled.copy()
        props = {p.label: p for p in measure.regionprops(merged)}
        groups, used = [], set()
        for l1, p1 in props.items():
            if l1 in used:
                continue
            group = {l1}; used.add(l1)
            for l2, p2 in props.items():
                if l2 in used or l2 == l1:
                    continue
                dist = np.hypot(p1.centroid[0] - p2.centroid[0], p1.centroid[1] - p2.centroid[1])
                if dist < aggressive_merge_dist:
                    ang = abs(np.arctan2(p2.centroid[0] - p1.centroid[0], p2.centroid[1] - p1.centroid[1]))
                    if ang < np.pi/6 or ang > 5*np.pi/6 or (np.pi/3 < ang < 2*np.pi/3):
                        group.add(l2); used.add(l2)
            groups.append(group)
        for g in groups:
            if len(g) > 1:
                labels = list(g); main = labels[0]
                for l in labels[1:]:
                    merged[merged == l] = main
        return relabel_sequential(merged)[0]

    def _merge_fragments(self, labeled, area_per_px_m2):
        if labeled.max() < 2:
            return labeled
        lab = labeled.copy()
        props = {r.label: r for r in measure.regionprops(lab)}
        if np.isfinite(area_per_px_m2) and area_per_px_m2 > 0:
            px_per_m = 1.0 / np.sqrt(area_per_px_m2)
            max_dist_px = self.constraints["max_merge_distance_m"] * px_per_m
        else:
            max_dist_px = 50
        for label, prop in props.items():
            area_m2 = prop.area * area_per_px_m2 if np.isfinite(area_per_px_m2) else prop.area
            if area_m2 < self.constraints["fragment_threshold_m2"]:
                min_dist, best = float('inf'), None
                for other_label, other_prop in props.items():
                    if other_label == label:
                        continue
                    other_area_m2 = other_prop.area * area_per_px_m2 if np.isfinite(area_per_px_m2) else other_prop.area
                    if other_area_m2 >= self.constraints["min_transformer_m2"]:
                        dist = np.hypot(prop.centroid[0] - other_prop.centroid[0],
                                        prop.centroid[1] - other_prop.centroid[1])
                        if dist < min_dist and dist < max_dist_px:
                            min_dist, best = dist, other_label
                lab[lab == label] = best if best is not None else 0
        return lab

    def _final_cleanup(self, labeled, area_per_px_m2):
        min_area_m2 = max(0.25 * self.constraints["min_transformer_m2"], 20.0)
        min_area_px = self._m2_to_px(min_area_m2, area_per_px_m2)
        labeled = remove_small_objects(labeled, min_size=min_area_px)
        labeled, _, _ = relabel_sequential(labeled)
        return labeled


# ------------------------------ Batch Processor ------------------------------

class BatchTransformerProcessor:
    """Batch processor: preserves baseline areas; adds rich features and per-site stats."""

    def __init__(self, output_dir, primary_model, fallback_model,
                 primary_prep, fallback_prep, device):
        self.output_dir = output_dir
        self.overlay_dir = os.path.join(output_dir, "overlays")
        self.csv_path = os.path.join(output_dir, "transformer_detections.csv")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.overlay_dir, exist_ok=True)

        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.primary_prep = primary_prep
        self.fallback_prep = fallback_prep
        self.device = device

        self.pipeline = TransformerSegmentationPipeline()
        self.results = []

        # Model selection overrides (unchanged)
        self.force_model = {
            82773910: "fallback_unet", 1052023838: "fallback_unet",
            25676535: "fallback_unet", 641340358: "fallback_unet",
            23084417: "fallback_unet", 29417741: "fallback_unet",
            152973139: "fallback_unet", 169244940: "fallback_unet",
            60680313: "fallback_unet", 30239842: "fallback_unet",
            42486826: "fallback_unet",
        }

    # ---------- World file helpers ----------
    def find_world_file(self, image_path):
        root, _ = os.path.splitext(image_path)
        for ext in ['.pgw', '.pngw', '.wld']:
            wf_path = root + ext
            if os.path.exists(wf_path):
                return wf_path
        base = os.path.basename(root)
        image_dir = os.path.dirname(image_path)
        for ext in ['.pgw', '.pngw', '.wld']:
            matches = glob.glob(os.path.join(image_dir, f"{base}*{ext}"))
            if matches:
                return matches[0]
        return None

    def _deg_to_m_scales(self, lat_deg):
        lat_rad = np.deg2rad(lat_deg)
        mpp_lon = 111_320.0 * np.cos(lat_rad)  # meters per degree longitude
        mpp_lat = 110_574.0                   # meters per degree latitude
        return mpp_lon, mpp_lat

    def read_world_file(self, world_file):
        """Return (world_params, area_per_px_m2_baseline, meta_dict). Baseline uses |A|*|E| logic."""
        try:
            with open(world_file, 'r') as f:
                vals = [float(line.strip()) for line in f.readlines()[:6]]
            if len(vals) < 6:
                return None, None, {}

            A, D, B, E, C, F = vals
            looks_like_degrees = (
                abs(A) < 1e-3 and abs(E) < 1e-3 and
                -180 <= C <= 180 and -90 <= F <= 90
            )

            if looks_like_degrees:
                mpp_lon, mpp_lat = self._deg_to_m_scales(F)
                # Baseline pixel area (your previous approach)
                area_per_px_m2 = abs(A) * mpp_lon * abs(E) * mpp_lat
                # Determinant-based (alternate) pixel area in meters:
                A_m = A * mpp_lon
                E_m = E * mpp_lat
                B_m = B * mpp_lon
                D_m = D * mpp_lat
                pixel_area_det_m2 = abs(A_m * E_m - B_m * D_m)
                pixel_area_simple_m2 = abs(A) * mpp_lon * abs(E) * mpp_lat
                world_has_rotation = (abs(B) > 0 or abs(D) > 0)
            else:
                # Map units already meters
                area_per_px_m2 = abs(A) * abs(E)
                pixel_area_det_m2 = abs(A * E - B * D)
                pixel_area_simple_m2 = abs(A) * abs(E)
                world_has_rotation = (abs(B) > 0 or abs(D) > 0)

            meta = {
                "looks_like_degrees": bool(looks_like_degrees),
                "pixel_area_simple_m2": float(pixel_area_simple_m2),
                "pixel_area_det_m2": float(pixel_area_det_m2),
                "flag_scale_disagreement": bool(
                    pixel_area_simple_m2 > 0 and
                    abs(pixel_area_det_m2 - pixel_area_simple_m2) / pixel_area_simple_m2 > 0.10
                ),
                "world_has_rotation": bool(world_has_rotation),
                "C": C, "F": F, "A": A, "B": B, "D": D, "E": E
            }
            return (A, D, B, E, C, F), area_per_px_m2, meta

        except Exception as e:
            logger.error(f"Error reading world file {world_file}: {str(e)}")
            return None, None, {}

    def get_georeferenced_coords(self, world_params, x_px, y_px):
        if world_params is None:
            return None, None
        A, D, B, E, C, F = world_params
        x_geo = C + x_px * A + y_px * B
        y_geo = F + x_px * D + y_px * E
        return x_geo, y_geo

    # ---------- Utilities for features ----------
    @staticmethod
    def _edge_flags_for_region(mask_region, img_h, img_w):
        """Return (edge_cut_flag, edge_touch_ratio, edge_direction) for a single binary mask."""
        # mask_region is full-image boolean mask for this region
        from skimage.segmentation import find_boundaries
        bnd = find_boundaries(mask_region, mode='outer')
        if not bnd.any():
            return False, 0.0, None

        rr, cc = np.where(bnd)
        total = len(rr)
        hit_top = np.sum(rr == 0)
        hit_bottom = np.sum(rr == img_h - 1)
        hit_left = np.sum(cc == 0)
        hit_right = np.sum(cc == img_w - 1)
        touches = hit_top + hit_bottom + hit_left + hit_right
        edge_cut_flag = touches > 0
        if touches == 0 or total == 0:
            return False, 0.0, None
        edge_touch_ratio = touches / total
        dirs = {"N": hit_top, "S": hit_bottom, "W": hit_left, "E": hit_right}
        edge_direction = max(dirs, key=dirs.get) if max(dirs.values()) > 0 else None
        return bool(edge_cut_flag), float(edge_touch_ratio), edge_direction

    @staticmethod
    def _region_prob_stats(probs, coords):
        rr = np.clip(coords[:, 0], 0, probs.shape[0]-1)
        cc = np.clip(coords[:, 1], 0, probs.shape[1]-1)
        vals = probs[rr, cc]
        return float(vals.max()) if vals.size else 0.0, float(vals.mean()) if vals.size else 0.0, float(np.median(vals)) if vals.size else 0.0

    @staticmethod
    def _region_stability(probs, coords, base_area_m2, area_per_px_m2, t, delta=0.05):
        """Threshold-sweep stability (no mask growth; measured inside current region)."""
        rr = np.clip(coords[:, 0], 0, probs.shape[0]-1)
        cc = np.clip(coords[:, 1], 0, probs.shape[1]-1)
        vals = probs[rr, cc]
        t_minus = np.clip(t - delta, 0.05, 0.95)
        t_plus  = np.clip(t + delta, 0.05, 0.95)
        a_minus = float((vals > t_minus).sum()) * area_per_px_m2
        a_plus  = float((vals > t_plus).sum()) * area_per_px_m2
        area_slope = (a_plus - a_minus) / (2 * delta)  # m² per unit threshold
        stability = 1.0 / (1.0 + abs(area_slope) / max(base_area_m2, 1e-6))
        return a_minus, base_area_m2, a_plus, area_slope, stability

    @staticmethod
    def _subpixel_area_from_contour(binary_mask, area_per_px_m2):
        """Compute sub-pixel area using 0.5 contour and shoelace; returns m²."""
        try:
            contours = find_contours(binary_mask.astype(float), 0.5)
            if not contours:
                return float(binary_mask.sum()) * area_per_px_m2
            # choose longest contour
            c = max(contours, key=lambda x: x.shape[0])
            # Shoelace in pixel coords
            x = c[:, 1]; y = c[:, 0]
            area_px = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            return float(area_px) * area_per_px_m2
        except Exception:
            return float(binary_mask.sum()) * area_per_px_m2

    @staticmethod
    def _rectangularity_and_hull_area(coords, mpp):
        """Return (rectangularity, solidity). Fallbacks if shapely missing."""
        # area and perimeter computed elsewhere; here we compute min rotated rect area and convex hull area
        if _HAVE_SHAPELY:
            try:
                poly = Polygon([(c[1], c[0]) for c in coords])  # (x,y) in px
                if poly.is_valid and not poly.is_empty and poly.area > 0:
                    mrr = poly.minimum_rotated_rectangle
                    rect_area_px = mrr.area
                    hull_area_px = poly.convex_hull.area
                    return float(rect_area_px) * (mpp**2), float(hull_area_px) * (mpp**2)
            except Exception:
                pass
        # Fallback: axis-aligned bounding box + convex hull via numpy (rough)
        ys = coords[:, 0]; xs = coords[:, 1]
        rect_area_px = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
        # crude hull area via bounding box as fallback
        hull_area_px = rect_area_px
        return float(rect_area_px) * (mpp**2), float(hull_area_px) * (mpp**2)

    @staticmethod
    def _pca_orientation_and_spacing(centroids_px, mpp):
        """Return (theta_deg [0,180), anisotropy [0..1], median_nn_gap_m) for a set of (row, col) centroids."""
        n = len(centroids_px)
        if n < 2:
            return None, None, None
        pts = np.array([[c[1], c[0]] for c in centroids_px], dtype=float)  # (x,y)
        mean = pts.mean(axis=0, keepdims=True)
        X = pts - mean
        # 2x2 covariance
        cov = (X.T @ X) / (n - 1) if n > 1 else np.zeros((2, 2))
        w, v = np.linalg.eigh(cov)  # ascending
        vec = v[:, 1]               # principal direction
        theta = (np.degrees(np.arctan2(vec[1], vec[0])) + 180.0) % 180.0
        # anisotropy
        lam1, lam2 = float(w[1]), float(w[0])
        ani = (lam1 - lam2) / (lam1 + lam2 + 1e-9)
        # spacing along axis: project onto vec and take median nearest-neighbor gap
        proj = (X @ vec.reshape(2, 1)).ravel()
        if n >= 2:
            proj_sorted = np.sort(proj)
            gaps = np.diff(proj_sorted)
            med_gap_px = float(np.median(gaps)) if gaps.size > 0 else None
            med_gap_m = (med_gap_px * mpp) if med_gap_px is not None else None
        else:
            med_gap_m = None
        return float(theta), float(ani), (float(med_gap_m) if med_gap_m is not None else None)

    # ---------- Model plumbing ----------
    def extract_osm_id(self, filename):
        import re
        match = re.search(r'(\d{7,10})', filename)
        if match:
            return int(match.group(1))
        return None

    def select_model(self, osm_id, image_pil):
        if osm_id in self.force_model and self.force_model[osm_id] == "fallback_unet":
            return self.fallback_model, self.fallback_prep, "fallback_unet"
        return self.primary_model, self.primary_prep, "primary_unet"

    def run_inference(self, image_pil, model, preprocess, threshold=0.5):
        """Run model inference with size preservation and safety checks (unchanged)."""
        img_np = np.array(image_pil)
        H, W = img_np.shape[:2]
        img_pre = preprocess(img_np)
        inp = torch.from_numpy(img_pre.astype('float32')).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = model(inp)
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
        if probs.shape != (H, W):
            probs = sk_resize(
                probs, (H, W),
                order=1, anti_aliasing=False, preserve_range=True
            )
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        probs = np.clip(probs, 0.0, 1.0)
        return (probs > threshold).astype(np.uint8), probs

    def create_overlay(self, image_np, labeled_mask):
        overlay = image_np.copy()
        colors = np.array([
            [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163],
            [255, 127, 0], [255, 255, 51], [166, 86, 40], [247, 129, 191],
            [153, 153, 153], [31, 120, 180], [178, 223, 138], [51, 160, 44],
            [251, 154, 153], [227, 26, 28], [253, 191, 111], [255, 127, 0],
            [202, 178, 214], [106, 61, 154], [255, 255, 153], [177, 89, 40]
        ])
        for label_id in range(1, labeled_mask.max() + 1):
            mask = (labeled_mask == label_id)
            color_idx = (label_id - 1) % len(colors)
            overlay[mask] = overlay[mask] * 0.6 + colors[color_idx] * 0.4
        boundaries = find_boundaries(labeled_mask, mode='outer')
        overlay[boundaries] = [255, 255, 255]
        for region in measure.regionprops(labeled_mask):
            cy, cx = map(int, region.centroid)
            cy = max(0, min(cy, overlay.shape[0]-1))
            cx = max(0, min(cx, overlay.shape[1]-1))
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if abs(dy) + abs(dx) <= 2:
                        y, x = cy + dy, cx + dx
                        if 0 <= y < overlay.shape[0] and 0 <= x < overlay.shape[1]:
                            overlay[y, x] = [255, 255, 255]
        return overlay.astype(np.uint8)

    # ---------- Per-image processing ----------
    def process_single_image(self, image_path, metadata=None):
        try:
            filename = os.path.basename(image_path)
            osm_id = self.extract_osm_id(filename)

            # World file & scale (baseline area_per_px_m2 uses |A|*|E| logic)
            world_file = self.find_world_file(image_path)
            world_params = None
            area_per_px_m2 = 0.3329 * 0.3329  # baseline default
            meta = {}
            if world_file:
                world_params, area_per_px_m2, meta = self.read_world_file(world_file)
                if area_per_px_m2 is None or area_per_px_m2 <= 0:
                    area_per_px_m2 = 0.3329 * 0.3329
                    logger.warning(f"Invalid scale from world file for {filename}, using default")
            else:
                logger.warning(f"No world file for {filename}, using default scale")

            # Load image
            img_pil = Image.open(image_path).convert("RGB")
            img_np = np.array(img_pil)
            H, W = img_np.shape[:2]
            mpp = float(np.sqrt(area_per_px_m2)) if area_per_px_m2 else None

            # Select and run model
            model, preprocess, model_name = self.select_model(osm_id, img_pil)
            _, probs = self.run_inference(img_pil, model, preprocess)

            # Segmentation (baseline behavior)
            labeled, info = self.pipeline.process(probs, area_per_px_m2)
            if labeled.max() == 0:
                logger.info(f"No transformers detected in {filename}")
                return

            # Image-level PCA orientation (computed once)
            centroids_px = [r.centroid for r in measure.regionprops(labeled)]
            img_theta_deg, img_anisotropy, img_med_gap_m = self._pca_orientation_and_spacing(centroids_px, mpp if mpp else 1.0)

            # Overlay (optional)
            overlay = self.create_overlay(img_np, labeled)
            overlay_filename = f"{osm_id}_overlay.png" if osm_id else f"{os.path.splitext(filename)[0]}_overlay.png"
            Image.fromarray(overlay).save(os.path.join(self.overlay_dir, overlay_filename))

            # Per-component features
            for r in measure.regionprops(labeled):
                # Baseline hard area (kept intact)
                area_px = int(r.area)
                area_m2 = float(area_px * area_per_px_m2) if (area_per_px_m2 and np.isfinite(area_per_px_m2)) else None

                # Georeferenced centroid
                x_geo, y_geo = self.get_georeferenced_coords(world_params, r.centroid[1], r.centroid[0])

                # Prob/confidence
                peak_p, mean_p, median_p = self._region_prob_stats(probs, r.coords)

                # Stability (threshold sweep inside current region mask)
                a_minus, a_t, a_plus, area_slope, stability = self._region_stability(
                    probs, r.coords, area_m2 if area_m2 is not None else 0.0, area_per_px_m2, info.get('threshold', 0.15)
                )

                # Edge flags
                region_mask_full = (labeled == r.label)
                edge_cut_flag, edge_touch_ratio, edge_dir = self._edge_flags_for_region(region_mask_full, H, W)

                # Shape descriptors
                major_axis_m = (float(r.major_axis_length) * mpp) if (mpp and r.major_axis_length) else None
                minor_axis_m = (float(r.minor_axis_length) * mpp) if (mpp and r.minor_axis_length) else None
                axis_ratio = (major_axis_m / minor_axis_m) if (major_axis_m and minor_axis_m and minor_axis_m > 0) else None
                perimeter_m = (float(r.perimeter) * mpp) if (mpp and r.perimeter) else None
                compactness = (4.0 * np.pi * area_m2 / (perimeter_m ** 2)) if (area_m2 and perimeter_m and perimeter_m > 0) else None
                # rectangularity & solidity
                rect_area_m2, hull_area_m2 = self._rectangularity_and_hull_area(r.coords, mpp if mpp else 1.0)
                rectangularity = (area_m2 / rect_area_m2) if (area_m2 and rect_area_m2 and rect_area_m2 > 0) else None
                solidity = (area_m2 / hull_area_m2) if (area_m2 and hull_area_m2 and hull_area_m2 > 0) else None
                orientation_deg = (np.degrees(r.orientation) + 180.0) % 180.0 if hasattr(r, 'orientation') else None

                # Nearest-neighbor spacing
                nn_dist_m = None
                if len(centroids_px) >= 2 and mpp:
                    cy, cx = r.centroid
                    dists_px = [np.hypot(cy - yy, cx - xx) for (yy, xx) in centroids_px if (yy, xx) != (cy, cx)]
                    nn_dist_m = float(min(dists_px) * mpp) if dists_px else None

                # Alternate area estimates (reported, not replacing baseline)
                area_m2_subpx = self._subpixel_area_from_contour(region_mask_full, area_per_px_m2)

                result = {
                    # identity
                    'osm_id': osm_id,
                    'image_name': filename,
                    'image_path': image_path,
                    'component_id': int(r.label),

                    # baseline area (unchanged)
                    'area_px': area_px,
                    'area_m2': area_m2,

                    # centroid (px + geo)
                    'centroid_x_px': round(r.centroid[1], 2),
                    'centroid_y_px': round(r.centroid[0], 2),
                    'centroid_x_geo': x_geo,
                    'centroid_y_geo': y_geo,

                    # model / pipeline
                    'model_used': model_name,
                    'threshold': float(info.get('threshold', 0)),
                    'n_components': int(labeled.max()),
                    'meters_per_px': float(mpp) if mpp else None,
                    'world_file_found': world_file is not None,

                    # world meta & alternate pixel area estimate
                    'pixel_area_simple_m2': meta.get('pixel_area_simple_m2'),
                    'pixel_area_det_m2': meta.get('pixel_area_det_m2'),
                    'flag_scale_disagreement': meta.get('flag_scale_disagreement'),
                    'world_has_rotation': meta.get('world_has_rotation'),
                    'world_is_degrees': meta.get('looks_like_degrees'),

                    # confidence & stability
                    'region_peak_prob': peak_p,
                    'region_mean_prob': mean_p,
                    'region_median_prob': median_p,
                    'area_m2_tminus': a_minus,
                    'area_m2_t': a_t,
                    'area_m2_tplus': a_plus,
                    'area_slope': area_slope,
                    'area_stability': stability,

                    # shape
                    'major_axis_m': major_axis_m,
                    'minor_axis_m': minor_axis_m,
                    'axis_ratio': axis_ratio,
                    'perimeter_m': perimeter_m,
                    'compactness': compactness,
                    'rectangularity': rectangularity,
                    'solidity': solidity,
                    'orientation_deg': orientation_deg,

                    # edge
                    'edge_cut_flag': edge_cut_flag,
                    'edge_touch_ratio': edge_touch_ratio,
                    'edge_direction': edge_dir,

                    # spacing (per component + image-level PCA)
                    'nn_dist_m': nn_dist_m,
                    'image_row_theta_deg': img_theta_deg,
                    'image_row_anisotropy': img_anisotropy,
                    'image_row_nn_gap_median': img_med_gap_m,

                    # alternate area estimate (reported only)
                    'area_m2_subpx': area_m2_subpx,
                }

                if metadata:
                    result.update(metadata)

                self.results.append(result)

            logger.info(f"Processed {filename}: {labeled.max()} transformers detected")

        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
        except PermissionError:
            logger.error(f"Permission denied accessing: {image_path}")
        except Exception as e:
            logger.error(f"Unexpected error processing {image_path}: {type(e).__name__}: {str(e)}")

    # ---------- Directory ----------
    def process_directory(self, image_dir, pattern="*.png", metadata_json=None, save_interval=100):
        metadata_dict = {}
        if metadata_json and os.path.exists(metadata_json):
            try:
                with open(metadata_json, 'r') as f:
                    metadata_list = json.load(f)
                for item in metadata_list:
                    oid = self.extract_osm_id(str(item.get('Id', '')))
                    if oid:
                        metadata_dict[oid] = item
                logger.info(f"Loaded metadata for {len(metadata_dict)} substations")
            except Exception as e:
                logger.error(f"Error loading metadata JSON: {str(e)}")

        image_paths = glob.glob(os.path.join(image_dir, pattern))
        logger.info(f"Found {len(image_paths)} images to process")

        for i, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            oid = self.extract_osm_id(os.path.basename(image_path))
            metadata = metadata_dict.get(oid, {}) if oid else {}
            self.process_single_image(image_path, metadata)
            if (i + 1) % save_interval == 0:
                self.save_results(interim=True)
                logger.info(f"Interim save at {i+1} images processed")

        self.save_results(interim=False)

    # ---------- Save & per-site stats ----------
    def _attach_site_relative_stats(self, df, area_col='area_m2'):
        """Compute and attach per-osm_id site-relative stats to df."""
        if 'osm_id' not in df.columns or df['osm_id'].isna().all():
            return df

        def iqr(series):
            q75, q25 = np.nanpercentile(series, 75), np.nanpercentile(series, 25)
            return q75 - q25

        g = df.groupby('osm_id', dropna=False)[area_col]
        site_median = g.transform('median')
        site_iqr = g.transform(iqr)
        # rank (1 = largest)
        site_rank = df.groupby('osm_id', dropna=False)[area_col] \
                      .rank(method='dense', ascending=False)
        # percentile within site
        def pct_rank(s):
            return s.rank(pct=True) * 100.0
        site_pctl = df.groupby('osm_id', dropna=False)[area_col] \
                      .transform(pct_rank)
        # z-score via IQR
        z_iqr = (df[area_col] - site_median) / (site_iqr / 1.349 + 1e-9)

        df['site_area_median_osm'] = site_median
        df['site_area_iqr_osm'] = site_iqr
        df['site_rank_in_osm'] = site_rank.astype('Int64')
        df['site_pctl_in_osm'] = site_pctl
        df['z_iqr_osm'] = z_iqr
        return df

    def save_results(self, interim=False):
        if not self.results:
            logger.warning("No results to save")
            return

        df = pd.DataFrame(self.results)

        # Attach site-relative stats (per OSM id) on final save
        if not interim and 'area_m2' in df.columns:
            try:
                df = self._attach_site_relative_stats(df, area_col='area_m2')
            except Exception as e:
                logger.error(f"Failed to attach site-relative stats: {e}")

        # Write CSV (add a header comment only on final save)
        csv_path = self.csv_path.replace('.csv', '_interim.csv') if interim else self.csv_path
        if not interim:
            header_comment = (
                "# NOTE:\n"
                "# - area_m2 is the baseline hard area (binary mask × baseline pixel area).\n"
                "# - area_m2_subpx is an alternate area estimate using a sub-pixel contour (not used to replace area_m2).\n"
                "# - pixel_area_simple_m2 vs pixel_area_det_m2 (full affine determinant) are provided for scale QA only.\n"
                "#   area_m2 remains unchanged regardless of determinant.\n"
            )
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write(header_comment)
            df.to_csv(csv_path, mode='a', index=False)
        else:
            df.to_csv(csv_path, index=False)

        logger.info(f"Saved {len(self.results)} transformer detections to {csv_path}")

        if not interim:
            # Per-site summary CSV (one row per OSM id)
            try:
                site_cols = ['osm_id', 'area_m2', 'edge_cut_flag', 'nn_dist_m']
                sdf = df[site_cols].copy()
                sdf['edge_cut_flag'] = sdf['edge_cut_flag'].fillna(False)
                agg = sdf.groupby('osm_id', dropna=False).agg(
                    n_components=('area_m2', 'size'),
                    area_median=('area_m2', 'median'),
                    area_p25=('area_m2', lambda s: np.nanpercentile(s, 25)),
                    area_p75=('area_m2', lambda s: np.nanpercentile(s, 75)),
                    area_cv=('area_m2', lambda s: (np.nanstd(s) / (np.nanmean(s) + 1e-9))),
                    share_edge_cut=('edge_cut_flag', 'mean'),
                    nn_dist_median=('nn_dist_m', 'median')
                ).reset_index()
                site_csv = self.csv_path.replace('.csv', '_per_site_summary.csv')
                agg.to_csv(site_csv, index=False)
                logger.info(f"Wrote per-site summary to {site_csv}")
            except Exception as e:
                logger.error(f"Failed to write per-site summary: {e}")


# ==== Main execution function ====
def run_batch_processing(
    image_dir,
    output_dir,
    metadata_json=None,
    pattern="*.png",
    save_interval=100
):
    logger.info(f"Starting batch processing at {datetime.now()}")
    logger.info(f"Image directory: {image_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Pattern: {pattern}")

    if not os.path.exists(image_dir):
        logger.error(f"Image directory does not exist: {image_dir}")
        return

    processor = BatchTransformerProcessor(
        output_dir=output_dir,
        primary_model=primary_model,
        fallback_model=fallback_model,
        primary_prep=primary_prep,
        fallback_prep=fallback_prep,
        device=DEVICE
    )

    processor.process_directory(
        image_dir=image_dir,
        pattern=pattern,
        metadata_json=metadata_json,
        save_interval=save_interval
    )

    logger.info(f"Batch processing complete at {datetime.now()}")


# ==== Example usage ====
if __name__ == "__main__":
    # You must define: primary_model, fallback_model, primary_prep, fallback_prep, DEVICE
    IMAGE_DIR = "/path/to/images"
    OUTPUT_DIR = "/path/to/output"
    METADATA_JSON = None

    run_batch_processing(
        image_dir=IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        metadata_json=METADATA_JSON,
        pattern="*.png",
        save_interval=100
    )
