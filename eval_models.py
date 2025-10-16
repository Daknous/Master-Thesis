#!/usr/bin/env python
"""
Evaluate multiple checkpoints on the validation set.

Metrics per model & post-processor:
  • Count-accuracy  (= % images where predicted instance count == GT)
  • MAE-count       (= |pred – gt|  averaged)
  • MAE-area (m²)   (= absolute area error in square-metres)
  • MAPE-area       (= |pred-gt| / gt)
  • mean inference  + post-processing time (sec)

Extra columns can be added easily.

# ── example usage ──────────────────────────────────────────────────────────────
python eval_models.py \
    runs/exp5/model_best_epoch94_valIoU0.7238.pth \
    runs/final_experiments/model_best_epoch83_valIoU0.7325.pth \
    runs/final_experiments/model_best_epoch92_valIoU0.7404.pth \
    --val_coco_json  Dataset_v2_filtered/valid/_annotations.coco.json \
    --val_img_dir    Dataset_v2_filtered/valid \
    --threshold 0.90 \
    --device cpu

"""
import argparse, glob, os, sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label
from skimage.morphology import remove_small_objects
from pycocotools.coco import COCO
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# ──────────────── Post-processing registry ────────────────────────────────────
MIN_SIZE = 100          # px
def no_pp(mask, *a):         return mask
def remove_small(mask, *a):  return remove_small_objects(mask.astype(bool),
                                                         MIN_SIZE).astype(np.uint8)
POST_PROCS = {
    "raw"        : no_pp,
    "remove_small": remove_small,
}

# ──────────────── CLI ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model_paths", nargs="+",
                   help="paths to .pth checkpoints (encoder, weights saved with torch.save)")
    p.add_argument("--val_coco_json", required=True)
    p.add_argument("--val_img_dir",   required=True)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    p.add_argument("--img_size", type=int, default=1200)
    return p.parse_args()

# ──────────────── Helper: load SMP model from checkpoint ─────────────────────-
def load_model(pth, device):
    ckpt   = torch.load(pth, map_location=device)
    # cfg    = ckpt["config"]                       # saved by your training script
    # reproduce model exactly
    model  = smp.Unet(
        encoder_name   = "resnet34",
        encoder_weights= "imagenet",
        in_channels    = 3, classes = 1,
        #decoder_attention_type = cfg.get("decoder_attention","none") or None,
    )
    #model.load_state_dict(ckpt, strict=True)
    model.to(device).eval()
    preprocess = smp.encoders.get_preprocessing_fn("resnet34", "imagenet")
    return model, preprocess, Path(pth).stem   # short name for tables

# ──────────────── Inference (no grads) ────────────────────────────────────────
@torch.no_grad()
def infer_mask(pil_img, model, pre_fn, device, thr):
    arr = np.asarray(pil_img)                               # H,W,3 uint8
    arr = pre_fn(arr).transpose(2,0,1).astype("float32")    # C,H,W
    t   = torch.from_numpy(arr).unsqueeze(0).to(device)     # 1,C,H,W
    probs = torch.sigmoid(model(t)).squeeze().cpu().numpy() # H,W
    return (probs > thr).astype(np.uint8)

# ──────────────── Main --------------------------------------------------------
def main():
    args   = parse_args()
    device = torch.device(args.device)

    # constant scale you computed once
    mpp = 0.1317 * (3000 / 1200)          # metres per pixel
    coco = COCO(args.val_coco_json)
    img_infos = coco.loadImgs(coco.getImgIds())

    rows = []
    for pth in args.model_paths:
        model, pre_fn, tag = load_model(pth, device)
        print(f"[{tag}] loaded.")

        for proc_name, proc in POST_PROCS.items():
            tic = time.time()
            records = []                 # one per image

            for info in img_infos:
                base = Path(info["file_name"]).stem
                img_path = glob.glob(os.path.join(args.val_img_dir, f"{base}*"))[0]
                img  = Image.open(img_path).convert("RGB")

                mask_pred = infer_mask(img, model, pre_fn, device, args.threshold)
                mask_pred = proc(mask_pred)

                lbl  = label(mask_pred, connectivity=2)
                n_pred = lbl.max()
                area_pred_px = (lbl>0).sum()
                area_pred_m2 = area_pred_px * (mpp**2)

                # ground-truth
                anns   = coco.loadAnns(coco.getAnnIds(info["id"], iscrowd=False))
                gt_cnt = len(anns)
                gt_area_px = sum(coco.annToMask(ann).sum() for ann in anns
                                 if ann.get("segmentation"))
                gt_area_m2 = gt_area_px * (mpp**2)

                records.append((
                    abs(n_pred-gt_cnt)==0,
                    abs(n_pred-gt_cnt),
                    abs(area_pred_m2-gt_area_m2) if gt_area_px>0 else np.nan,
                    abs(area_pred_m2-gt_area_m2)/gt_area_m2 if gt_area_px>0 else np.nan
                ))

            rec = np.vstack(records)
            count_acc = rec[:,0].mean()
            mae_cnt   = rec[:,1].mean()
            mae_area  = np.nanmean(rec[:,2])
            mape_area = np.nanmean(rec[:,3])
            rows.append({
                "model": tag,
                "post":  proc_name,
                "count_acc": count_acc,
                "mae_cnt":   mae_cnt,
                "mae_area_m2": mae_area,
                "mape_area":  mape_area,
                "images": len(records),
                "avg_time_s": (time.time()-tic)/len(records)
            })

    df = pd.DataFrame(rows)
    print("\n─ Summary ─")
    print(df.sort_values(["model","post"]).to_string(index=False,
                                                    float_format=lambda x:f"{x:0.4f}"))

if __name__ == "__main__":
    main()
