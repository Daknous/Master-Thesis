#!/usr/bin/env python
"""
Inference script for transformer segmentation.
Scale_csv iis optional and can be used to provide per-image metre-per-pixel values.
It will be used to compute the area in square metres for each segmented component.
Example
-------
python infer_transformers.py \
    --images_dir /Volumes/Seif_SSD/unspecified_substations_highres \
    --checkpoint /Users/zif/Documents/Substation_Master_thesis/Master-Thesis/runs/exp5/model_epoch100_valIoU0.6063.pth \
    --out_csv    predictions.csv \
    --out_dir    /Volumes/Seif_SSD/validation_overlays_dataset_v2 \
    --scale_csv  /Volumes/Seif_SSD/unspecified_substations_highres/image_meters_per_px.csv
"""
import os, csv, time, argparse, json
import numpy as np
from PIL import Image
from glob import glob
import pandas as pd
import torch, torchvision.transforms as T
import segmentation_models_pytorch as smp
from skimage.measure import label
from skimage.morphology import (remove_small_objects, opening,
                                closing, disk)
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# ─── post-processors ─────────────────────────────────────────────────────────
MIN_SIZE = 100

def watershed_split(m):
    dist = distance_transform_edt(m)
    peaks = peak_local_max(dist, labels=m, footprint=np.ones((3,3)))
    markers = np.zeros_like(m, dtype=int)
    for i,(r,c) in enumerate(peaks, start=1):
        markers[r,c] = i
    lab = watershed(-dist, markers, mask=m)
    return (lab > 0)

POST = {
    "raw": lambda m: m,
    "remove_small": lambda m: remove_small_objects(m, MIN_SIZE),
    #"open_then_close": lambda m: closing(opening(m, disk(3)), disk(7)),
    #"watershed": watershed_split,
}

# ─── helpers ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_mask(img_pil, model, tfm, thr, device):
    t = tfm(img_pil).unsqueeze(0).to(device)
    logits = model(t)          # [1,2,H,W]
    prob   = torch.softmax(logits, dim=1)[0, 1]   # foreground prob
    return (prob.cpu().numpy() > thr)

def comp_stats(mask_bin, mpp):
    lbl = label(mask_bin, connectivity=2)
    for cid in range(1, lbl.max()+1):
        px = int((lbl==cid).sum())
        yield px, px * mpp * mpp if mpp is not None else None

# ─── main ────────────────────────────────────────────────────────────────────
def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("🖥  device:", device)

    # load model (single-channel)
    model = smp.Unet(encoder_name=args.encoder,
                     encoder_weights=None,
                     in_channels=3, classes=2).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std =[0.229,0.224,0.225]),
    ])

    # optional scale CSV -> dict {image: metres_per_px}
    mpp_lookup = None
    if args.scale_csv:
        df_scale = pd.read_csv(args.scale_csv)
        mpp_lookup = dict(zip(df_scale['image'], df_scale['m_per_px_x']))

    # prepare outputs
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []

    img_paths = sorted(glob(os.path.join(args.images_dir, "*.png")) +
                       glob(os.path.join(args.images_dir, "*.jpg")))
    print(f"🔎 found {len(img_paths)} images")

    for ip in img_paths:
        name = os.path.basename(ip)
        img  = Image.open(ip).convert("RGB")
        mpp  = mpp_lookup.get(name) if mpp_lookup else None

        mask_raw = predict_mask(img, model, tfm, args.threshold, device)

        for tag, proc in POST.items():
            t0     = time.time()
            try: mask_pp = proc(mask_raw.copy())
            except Exception as e:
                print(f"⚠️  {tag} failed on {name}: {e}"); continue
            elapsed = time.time() - t0

            # component loop
            for idx,(px, m2) in enumerate(comp_stats(mask_pp, mpp), start=1):
                rows.append({
                    "image"  : name,
                    "method" : tag,
                    "cid"    : idx,
                    "area_px2": px,
                    "area_m2": None if m2 is None else round(m2,2),
                    "time_s"  : round(elapsed,4)
                })

            # save overlay
            if args.save_overlays:
                ov = np.array(img)
                ov[mask_pp] = [255,0,0]
                Image.fromarray(ov).save(
                    os.path.join(args.out_dir,
                                 f"{os.path.splitext(name)[0]}_{tag}.png"))

    # write CSV
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print("✅ saved", args.out_csv)

# ─── cli ─────────────────────────────────────────────────────────────────────
def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--images_dir', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out_csv',    default='predictions.csv')
    p.add_argument('--out_dir',    default='overlays')
    p.add_argument('--scale_csv',  default=None,
                   help='optional CSV with per-image metre-per-px columns '
                        '(image,m_per_px_x,m_per_px_y)')
    p.add_argument('--encoder',    default='resnet34')
    p.add_argument('--threshold',  type=float, default=0.7)
    p.add_argument('--save_overlays', action='store_true')
    return p.parse_args()

if __name__ == '__main__':
    run(parse())
