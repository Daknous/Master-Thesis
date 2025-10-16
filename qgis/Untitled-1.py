# osm_eval_tiles_adaptive_scale_missing.py — Enhanced for missing substations
import os, csv
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsCoordinateTransform, QgsRectangle, QgsGeometry,
    QgsMapSettings, QgsMapRendererParallelJob
)
from qgis.PyQt.QtGui import QImage
from qgis.PyQt.QtCore import QSize

# ─── ENHANCED PARAMS FOR MISSING SUBSTATIONS ─────────────────────────────
PARAMS = dict(
    OSM_CSV        = "/Users/zif/Documents/Substation_Master_thesis/Master-Thesis/final_results/missing_substations/missing_substations_footprint_results.csv",
    OUTPUT_DIR     = "/Users/zif/Documents/Substation_Master_thesis/Master-Thesis/final_results/missing_substations/missing_substations_images_v2",
    WMS_LAYER_NAME = "Google Satellite",
    IMG_SIZE_PX    = 1024,
    DPI            = 300,

    # MORE AGGRESSIVE adaptive framing for missing substations
    TARGET_COVERAGE = 0.45,   # Increased from 0.33 - make transformers more prominent
    SIDE_MIN_M      = 150.0,  # Reduced from 250 - allow tighter zoom
    SIDE_MAX_M      = 500.0,  # Reduced from 600 - stay closer
    PAD_PCT         = 0.10,   # Reduced from 0.15 - less padding

    # Multiple variants for missing substations
    EXPORT_TIGHT    = True,   # adaptive FOV
    EXPORT_CLOSE    = True,   # even tighter for small substations
    EXPORT_TRAIN    = False,  # skip fixed FOV for now

    CSV_FILENAME    = "missing_image_scales.csv"
)
# ──────────────────────────────────────────────────────────────────────────

os.makedirs(PARAMS['OUTPUT_DIR'], exist_ok=True)

# Load CSV and check for required columns
import pandas as pd
df_check = pd.read_csv(PARAMS['OSM_CSV'])
print(f"CSV columns: {list(df_check.columns)}")
print(f"Found {len(df_check)} substations, {df_check['found'].sum()} with footprints")

# Load footprints CSV as WKT layer (EPSG:4326)
uri = f"file:///{PARAMS['OSM_CSV']}?delimiter=,&crs=EPSG:4326&wktField=footprint_wkt"
v_foot = QgsVectorLayer(uri, "osm_footprints", "delimitedtext")
if not v_foot.isValid():
    raise Exception(f"Failed to load CSV: {PARAMS['OSM_CSV']}")
QgsProject.instance().addMapLayer(v_foot)

# Basemap/WMS
wms_list = QgsProject.instance().mapLayersByName(PARAMS['WMS_LAYER_NAME'])
if not wms_list:
    raise Exception(f"WMS layer '{PARAMS['WMS_LAYER_NAME']}' not found")
wms = wms_list[0]

# Transform footprints → WMS CRS (meters)
xf = QgsCoordinateTransform(v_foot.crs(), wms.crs(), QgsProject.instance())

# Map render settings
ms = QgsMapSettings()
ms.setLayers([wms])
ms.setOutputDpi(PARAMS['DPI'])

def square_extent(cx, cy, side_m):
    h = side_m/2.0
    return QgsRectangle(cx - h, cy - h, cx + h, cy + h)

def write_worldfile(png_path, extent, w, h):
    A = extent.width() / w
    E = -extent.height() / h
    C = extent.xMinimum() + A/2.0
    F = extent.yMaximum() + E/2.0
    with open(png_path + "w", "w") as wf:
        wf.write(f"{A:.12f}\n0.0\n0.0\n{E:.12f}\n{C:.12f}\n{F:.12f}\n")

def render(extent, out_png, size_px):
    ms.setExtent(extent)
    ms.setOutputSize(QSize(size_px, size_px))
    img = QImage(size_px, size_px, QImage.Format_ARGB32)
    img.fill(0)
    job = QgsMapRendererParallelJob(ms)
    job.start(); job.waitForFinished()
    job.renderedImage().save(out_png, "PNG")
def get_substation_info(feature):
    """Extract metadata for adaptive scaling"""
    # Get transformer count
    transformer_count = 0
    field_names = [f.name() for f in feature.fields()]
    
    if 'TransformerCount' in field_names:
        transformer_count = feature['TransformerCount']
        if transformer_count is None:
            transformer_count = 0
    
    # Get estimated size
    estimated_size = 200  # default
    if 'estimated_size_m' in field_names:
        estimated_size = feature['estimated_size_m']
        if estimated_size is None:
            estimated_size = 200
    
    # Get method used
    method = 'unknown'
    if 'method' in field_names:
        method = feature['method']
        if method is None:
            method = 'unknown'
    
    return transformer_count, estimated_size, method

# CSV tracking
csv_rows = []
fieldnames = ["image","variant","ground_width_m","ground_height_m","m_per_px_x","m_per_px_y","meters_per_px","transformer_count","method","footprint_area_m2"]

processed = 0
for f in v_foot.getFeatures():
    # Skip if no footprint found - fix this line too
    field_names = [field.name() for field in f.fields()]
    found_value = True  # default
    if 'found' in field_names:
        found_value = f['found']
        if found_value is None:
            found_value = True
    
    if not found_value:
        continue
        
    oid = str(f["osm_id"])
    g = f.geometry()
    if not g or g.isEmpty():
        print(f"⚠️ {oid}: empty geometry → skipped")
        continue

    # Get substation metadata
    transformer_count, estimated_size, method = get_substation_info(f)

    # Transform to WMS CRS
    g2 = QgsGeometry(g)
    g2.transform(xf)
    bb = g2.boundingBox()
    footprint_area = g2.area()

    # Adaptive padding based on substation characteristics
    base_pad_pct = PARAMS['PAD_PCT']
    
    # Less padding for fallback circles (they're already generous)
    if 'fallback' in str(method):
        base_pad_pct *= 0.5
    
    # More padding for very small footprints
    if footprint_area < 5000:  # < 5000 m²
        base_pad_pct *= 1.5

    # Calculate padded bbox
    pad = max(bb.width(), bb.height()) * base_pad_pct
    padded = QgsRectangle(bb.xMinimum()-pad, bb.yMinimum()-pad, bb.xMaximum()+pad, bb.yMaximum()+pad)
    padded_side = max(padded.width(), padded.height())
    cx, cy = padded.center().x(), padded.center().y()

    # ── TIGHT: Standard adaptive scaling
    if PARAMS['EXPORT_TIGHT']:
        desired_side = padded_side / max(1e-6, PARAMS['TARGET_COVERAGE'])
        tight_side = min(max(desired_side, PARAMS['SIDE_MIN_M']), PARAMS['SIDE_MAX_M'])
        tight_extent = square_extent(cx, cy, tight_side)
        
        base = f"osm_{oid}_tight_{PARAMS['IMG_SIZE_PX']}.png"
        out_png = os.path.join(PARAMS['OUTPUT_DIR'], base)
        render(tight_extent, out_png, PARAMS['IMG_SIZE_PX'])
        write_worldfile(out_png, tight_extent, PARAMS['IMG_SIZE_PX'], PARAMS['IMG_SIZE_PX'])
        
        gw, gh = tight_extent.width(), tight_extent.height()
        mpx, mpy = gw/PARAMS['IMG_SIZE_PX'], abs(gh/PARAMS['IMG_SIZE_PX'])
        csv_rows.append(dict(
            image=base, variant="tight",
            ground_width_m=gw, ground_height_m=gh,
            m_per_px_x=mpx, m_per_px_y=mpy, meters_per_px=(mpx+mpy)/2,
            transformer_count=transformer_count, method=method, footprint_area_m2=footprint_area
        ))
        print(f"✅ {oid}: TIGHT {gw:.1f}×{gh:.1f} m (area:{footprint_area:.0f}m², method:{method})")

    # ── CLOSE: Extra tight for small substations with few transformers
    if PARAMS['EXPORT_CLOSE'] and (footprint_area < 10000 or transformer_count <= 2):
        # Even more aggressive coverage for small substations
        close_coverage = 0.6  # Make footprint 60% of image
        desired_side = padded_side / max(1e-6, close_coverage)
        close_side = min(max(desired_side, 100.0), PARAMS['SIDE_MIN_M'])  # Allow very tight zoom
        close_extent = square_extent(cx, cy, close_side)
        
        base = f"osm_{oid}_close_{PARAMS['IMG_SIZE_PX']}.png"
        out_png = os.path.join(PARAMS['OUTPUT_DIR'], base)
        render(close_extent, out_png, PARAMS['IMG_SIZE_PX'])
        write_worldfile(out_png, close_extent, PARAMS['IMG_SIZE_PX'], PARAMS['IMG_SIZE_PX'])
        
        gw, gh = close_extent.width(), close_extent.height()
        mpx, mpy = gw/PARAMS['IMG_SIZE_PX'], abs(gh/PARAMS['IMG_SIZE_PX'])
        csv_rows.append(dict(
            image=base, variant="close",
            ground_width_m=gw, ground_height_m=gh,
            m_per_px_x=mpx, m_per_px_y=mpy, meters_per_px=(mpx+mpy)/2,
            transformer_count=transformer_count, method=method, footprint_area_m2=footprint_area
        ))
        print(f"✅ {oid}: CLOSE {gw:.1f}×{gh:.1f} m")

    processed += 1

# Save metrics CSV
csv_path = os.path.join(PARAMS['OUTPUT_DIR'], PARAMS['CSV_FILENAME'])
with open(csv_path, "w", newline="") as cf:
    w = csv.DictWriter(cf, fieldnames=fieldnames)
    w.writeheader(); w.writerows(csv_rows)

print(f"\nProcessed {processed} substations → {csv_path}")
print(f"Generated {len(csv_rows)} images total")

# Print scale statistics
if csv_rows:
    scales = [row['meters_per_px'] for row in csv_rows]
    print(f"Scale range: {min(scales):.3f} - {max(scales):.3f} m/px")
    print(f"Mean scale: {sum(scales)/len(scales):.3f} m/px")