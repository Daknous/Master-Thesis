import json, cv2, numpy as np, pathlib

DATASET_ROOT = pathlib.Path("Dataset")          # adjust if needed
TRANS_CAT_ID = 6                               # transformer category id

json_paths = list(DATASET_ROOT.rglob("_annotations.coco.json"))
if not json_paths:
    print("❗ No COCO annotation files found under", DATASET_ROOT.resolve())
else:
    total_anns = 0
    for src in json_paths:
        coco = json.loads(src.read_text())
        mod = False
        for ann in coco["annotations"]:
            if ann["category_id"] != TRANS_CAT_ID:
                continue
            seg = ann.get("segmentation", [])
            if isinstance(seg, list) and seg:        # polygon exists
                pts = np.asarray(seg[0], dtype=np.float32).reshape(-1, 2)
                (cx, cy), (w, h), theta = cv2.minAreaRect(pts)
            else:                                    # fallback to AABB
                x, y, w, h = ann["bbox"]
                cx, cy, theta = x + w / 2, y + h / 2, 0.0
            ann["obb"] = [float(cx), float(cy), float(w), float(h), float(theta)]
            mod = True
            total_anns += 1

        if mod:
            dst = src.with_name(src.stem.replace("_annotations", "_annotations_obb") + src.suffix)
            dst.write_text(json.dumps(coco))
            print(f"✓ {src.relative_to(DATASET_ROOT)} → {dst.name}")
        else:
            print(f"– {src.relative_to(DATASET_ROOT)} (no transformer anns)")

    print(f"\n★ Added OBB to {total_anns} transformer annotations across {len(json_paths)} files.")
