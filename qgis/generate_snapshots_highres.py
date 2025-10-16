# regenerate_highres.py  ──────────────────────────────────────────────
"""
Render high-resolution satellite snapshots for any split (train / valid / test),
write a CSV with ground dimensions and meters-per-pixel for each exported image,
and optionally generate world-files.

Usage inside QGIS:
  Plugins → Python Console → Run script… → select this file
"""

import os
import csv
from pathlib import Path
from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsCoordinateTransform,
    QgsMapSettings,
    QgsMapRendererParallelJob,
)
from qgis.PyQt.QtGui import QImage
from qgis.PyQt.QtCore import QSize

# ─── 1) EDIT THESE ONLY ─────────────────────────────────────────────────
PARAMS = dict(
    SPLIT            = "valid",    # "train" / "valid" / "test"
    MAPPING_CSV      = "/Users/zif/Documents/Substation_Master_thesis/Master-Thesis/add_data/unspecified_locations.csv",
    OUTPUT_DIR       = "/Volumes/Seif_SSD/unspecified_substations_highres",  # external SSD mount point
    WMS_LAYER_NAME   = "Google Satellite",  # must already be loaded in QGIS
    BUFFER_METERS    = 300,       # buffer radius around point in meters
    IMG_SIZE_PX      = 1024,      # rendered width/height in pixels
    DPI              = 300,       # output DPI
    CSV_FILENAME     = "image_meters_per_px.csv"  # output CSV file name
)
# ────────────────────────────────────────────────────────────────────────────

print(f"\n=== Generating HIGH-RES snapshots and metrics for «{PARAMS['SPLIT']}» split ===")

# Ensure output directory exists (for external SSD or any path)
os.makedirs(PARAMS['OUTPUT_DIR'], exist_ok=True)

# Grab WMS layer
wms_layers = QgsProject.instance().mapLayersByName(PARAMS['WMS_LAYER_NAME'])
if not wms_layers:
    raise Exception(f"WMS layer '{PARAMS['WMS_LAYER_NAME']}' not found in project.")
wms = wms_layers[0]

# Load mapping CSV into a QGIS vector layer
uri = (
    f"file:///{PARAMS['MAPPING_CSV']}"
    "?delimiter=,&xField=Longitude&yField=Latitude&crs=EPSG:4326"
)
vlayer = QgsVectorLayer(uri, f"{PARAMS['SPLIT']}_mapping", "delimitedtext")
if not vlayer.isValid():
    raise Exception(f"Could not load mapping CSV: {PARAMS['MAPPING_CSV']}")
QgsProject.instance().addMapLayer(vlayer)

# Determine which field holds the image filename
fields = [f.name() for f in vlayer.fields()]
print(f"Loaded fields from mapping CSV: {fields}")
if 'image' in fields:
    image_field = 'image'
elif 'Image' in fields:
    image_field = 'Image'
else:
    image_field = fields[0]
print(f"Using '{image_field}' as the image field")

# Coordinate transform CSV CRS → WMS CRS
xf = QgsCoordinateTransform(vlayer.crs(), wms.crs(), QgsProject.instance())

# Shared map-render settings
ms = QgsMapSettings()
ms.setLayers([wms])
ms.setOutputSize(QSize(PARAMS['IMG_SIZE_PX'], PARAMS['IMG_SIZE_PX']))
ms.setOutputDpi(PARAMS['DPI'])

# Prepare CSV output
csv_path = os.path.join(PARAMS['OUTPUT_DIR'], PARAMS['CSV_FILENAME'])
fieldnames = ['image', 'ground_width_m', 'ground_height_m', 'm_per_px_x', 'm_per_px_y', 'meters_per_px']
csv_rows = []

for feat in vlayer.getFeatures():
    # Original image identifier from CSV
    img_val = feat[image_field]
    # Sanitize to create a valid filename (replace '/' with '_', drop extension)
    base = Path(img_val).stem.replace('/', '_')
    fname = f"{base}.png"
    out_png = os.path.join(PARAMS['OUTPUT_DIR'], fname)
    # Ensure directory exists
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    # Transform geometry and buffer extent
    geom = feat.geometry()
    geom.transform(xf)
    bbox = geom.buffer(PARAMS['BUFFER_METERS'], 50).boundingBox()

    # Compute ground dimensions
    ground_width_m  = bbox.xMaximum() - bbox.xMinimum()
    ground_height_m = bbox.yMaximum() - bbox.yMinimum()

    # Render snapshot
    ms.setExtent(bbox)
    img = QImage(QSize(PARAMS['IMG_SIZE_PX'], PARAMS['IMG_SIZE_PX']), QImage.Format_ARGB32)
    img.fill(0)
    job = QgsMapRendererParallelJob(ms)
    job.start(); job.waitForFinished()
    job.renderedImage().save(out_png, 'PNG')

    # Write world-file (.pngw)
    px_w = ground_width_m / PARAMS['IMG_SIZE_PX']
    px_h = -(ground_height_m / PARAMS['IMG_SIZE_PX'])
    ulx = bbox.xMinimum() + abs(px_w)/2
    uly = bbox.yMaximum() + px_h/2
    wld_lines = [
        f"{px_w:.10f}",
        '0.0',
        '0.0',
        f"{px_h:.10f}",
        f"{ulx:.10f}",
        f"{uly:.10f}"
    ]
    world_path = f"{out_png}w"
    with open(world_path, 'w') as wf:
        wf.write("\n".join(wld_lines))

    # Collect CSV metrics
    meters_per_px = (abs(px_w) + abs(px_h)) / 2
    csv_rows.append({
        'image': fname,
        'ground_width_m': ground_width_m,
        'ground_height_m': ground_height_m,
        'm_per_px_x': px_w,
        'm_per_px_y': abs(px_h),
        'meters_per_px': meters_per_px
    })

    print(f"✓ Exported {fname}: width={ground_width_m:.1f}m, height={ground_height_m:.1f}m")

# Save the CSV
with open(csv_path, 'w', newline='') as cf:
    writer = csv.DictWriter(cf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"\n=== Done – images & metrics CSV saved to: {csv_path} ===")
