# coco_obb_to_yolo.py  (append-safe)
import json, math, pathlib

DATASET = pathlib.Path("Dataset")   # root folder
TRANS_ID = 6                        # transformer category id in COCO

def convert_one(coco_json: pathlib.Path):
    coco = json.loads(coco_json.read_text())
    imgs = {im["id"]: im for im in coco["images"]}
    lbl_dir = coco_json.parent / "labels"
    lbl_dir.mkdir(exist_ok=True)

    # wipe old .txt files
    for p in lbl_dir.glob("*.txt"):
        p.unlink()

    for ann in coco["annotations"]:
        if ann["category_id"] != TRANS_ID or "obb" not in ann:
            continue
        cx, cy, w, h, theta = ann["obb"]
        im = imgs[ann["image_id"]]
        nx, ny = cx / im["width"],  cy / im["height"]
        nw, nh = w  / im["width"],  h  / im["height"]
        theta_r = math.radians(theta)
        cls = 0                                   # YOLO class id
        stem = pathlib.Path(im["file_name"]).stem
        with (lbl_dir / f"{stem}.txt").open("a") as f:
            f.write(f"{cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f} {theta_r:.6f}\n")

    print(f"✓ {coco_json.parent.name}: wrote labels → {lbl_dir}")

# run for every split that has an OBB JSON
for coco_path in DATASET.rglob("_annotations_obb.coco.json"):
    convert_one(coco_path)
