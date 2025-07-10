#!/usr/bin/env python3
import json
import os
import sys

# ──────────────── CONFIG ────────────────
# Path to your original (low-res) COCO JSON
INPUT_JSON     = "/Users/zif/Documents/Substation_Master_thesis/Master-Thesis/Dataset/valid/_annotations.coco.json"

# Path or directory where you want to save the fixed high-res JSON.
# If this is a directory, the script will write:
#   <OUTPUT_JSON>/<basename(INPUT_JSON)>
OUTPUT_PATH     = "/Users/zif/Documents/Substation_Master_thesis/Master-Thesis/Dataset/highres_dataset/valid"  # can be a full filename or a directory

# Original and new image sizes (px)
OLD_SIZE     = 640
NEW_SIZE     = 1024

# Buffers used when cropping (in metres)
BUFFER_OLD   = 200    # original buffer 
BUFFER_NEW   = 300    # new buffer

# Fine-tune shifts (in pixels) if you're still a bit off:
#   +X shifts everything to the right; +Y shifts everything down.
#   Try small values like ±10, ±20 until your overlay is perfect.
SHIFT_X      = 60
SHIFT_Y      = -180
# ────────── END CONFIG (no edits below) ──────────

def resolve_output_path(inp, out):
    # If OUTPUT_PATH is a dir, write basename(INPUT_JSON) inside it
    if os.path.isdir(out) or out.endswith(os.sep):
        os.makedirs(out, exist_ok=True)
        return os.path.join(out, os.path.basename(inp))
    parent = os.path.dirname(out)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    return out

def main():
    # compute metres-per-pixel
    extent_old = 2 * BUFFER_OLD
    extent_new = 2 * BUFFER_NEW
    ps_old = extent_old / OLD_SIZE
    ps_new = extent_new / NEW_SIZE

    # scale & base-shift
    scale   = ps_old / ps_new
    shift0  = (BUFFER_NEW - BUFFER_OLD) / ps_new

    # resolve where to write
    out_json = resolve_output_path(INPUT_JSON, OUTPUT_PATH)

    # load
    try:
        coco = json.load(open(INPUT_JSON))
    except Exception as e:
        print(f"Error loading {INPUT_JSON}: {e}", file=sys.stderr)
        sys.exit(1)

    # update image sizes
    for img in coco.get("images", []):
        img["width"]  = NEW_SIZE
        img["height"] = NEW_SIZE

    # remap annotations
    for ann in coco.get("annotations", []):
        # polygons
        new_segs = []
        for poly in ann.get("segmentation", []):
            new_poly = []
            for x, y in zip(poly[0::2], poly[1::2]):
                x2 = x * scale + shift0 + SHIFT_X
                y2 = y * scale + shift0 + SHIFT_Y
                new_poly += [x2, y2]
            new_segs.append(new_poly)
        ann["segmentation"] = new_segs

        # bbox [x, y, w, h]
        bx, by, bw, bh = ann["bbox"]
        ann["bbox"] = [
            bx * scale + shift0 + SHIFT_X,
            by * scale + shift0 + SHIFT_Y,
            bw * scale,
            bh * scale
        ]

    # write out
    try:
        with open(out_json, "w") as f:
            json.dump(coco, f, indent=2)
        print(f"Wrote corrected annotations to {out_json}")
    except Exception as e:
        print(f"Error writing {out_json}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
