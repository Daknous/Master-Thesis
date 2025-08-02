"""
Single script training pipeline for transformer segmentation using PyTorch and Albumentations.
Uses a combined BCE + Dice loss and runs a validation loop each epoch to track IoU.
Usage:
    python electrical_substation_segmentation.py \
        --train_images_dir /path/to/train/images \
        --train_coco_json  /path/to/train/_annotations.coco.json \
        --val_images_dir   /path/to/valid/images \
        --val_coco_json    /path/to/valid/_annotations.coco.json \
        --batch_size 8 \
        --lr 1e-4 \
        --epochs 50 \
        --num_workers 4 \
        [--device cuda] [--dry_run] \
        [--log_dir runs/exp2]
dry_run:
    python electrical_substation_segmentation.py \
        --train_images_dir /Users/zif/Documents/Substation_Master_thesis/Master-Thesis/Dataset_v2_filtered/train \
        --train_coco_json  /Users/zif/Documents/Substation_Master_thesis/Master-Thesis/Dataset_v2_filtered/train/_annotations.coco.json \
        --val_images_dir   /Users/zif/Documents/Substation_Master_thesis/Master-Thesis/Dataset_v2_filtered/valid \
        --val_coco_json    /Users/zif/Documents/Substation_Master_thesis/Master-Thesis/Dataset_v2_filtered/valid/_annotations.coco.json \
        --batch_size 2 \
        --dry_run \
        --one_cycle \
        --tta_flip \
        --loss_ft

"""

import os
import json
import argparse
import sys
import cv2
import numpy as np
from glob import glob
from collections import defaultdict

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A

# ---------------------------
# Configuration
# ---------------------------
IMG_SIZE = 1200
DEFAULT_ENCODER = 'resnet34'
PREPROCESS_FN   = smp.encoders.get_preprocessing_fn(DEFAULT_ENCODER, 'imagenet')

# ---------------------------
# Custom Augmentation Transforms
# ---------------------------
class RandomCropWithMask(A.DualTransform):
    def __init__(self, height, width, min_mask_frac=0.005, max_tries=5, p=1.0):
        super().__init__(always_apply=True, p=p)
        self.height = height
        self.width = width
        self.min_mask_frac = min_mask_frac
        self.max_tries = max_tries

    def apply(self, img, x=0, y=0, **params):
        return img[y:y+self.height, x:x+self.width]

    def apply_to_mask(self, mask, x=0, y=0, **params):
        return mask[y:y+self.height, x:x+self.width]

    def get_params(self):
        return {}

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params['image'].shape[:2]
        mask = (params['mask'] > 0).astype(np.uint8)
        # compute mask centroid
        ys, xs = np.where(mask)
        if len(ys) == 0:
            # no mask: center crop
            return {'x': (img_w - self.width)//2, 'y': (img_h - self.height)//2}
        cy, cx = ys.mean(), xs.mean()
        # try random crops
        area = self.height * self.width
        for _ in range(self.max_tries):
            x = np.random.randint(0, img_w - self.width)
            y = np.random.randint(0, img_h - self.height)
            patch = mask[y:y+self.height, x:x+self.width]
            if patch.sum() / area >= self.min_mask_frac:
                return {'x': x, 'y': y}
        # fallback: center on mask centroid
        x0 = int(np.clip(cx - self.width/2, 0, img_w - self.width))
        y0 = int(np.clip(cy - self.height/2, 0, img_h - self.height))
        return {'x': x0, 'y': y0}

    def get_transform_init_args_names(self):
        return ('height','width','min_mask_frac','max_tries')

class MaskAwareDropout(A.DualTransform):
    def __init__(self, max_holes=8, hole_frac=0.05, max_mask_overlap_frac=0.1, max_tries=10, p=0.3):
        super().__init__(always_apply=False, p=p)
        self.max_holes = max_holes
        self.hole_frac = hole_frac
        self.max_mask_overlap_frac = max_mask_overlap_frac
        self.max_tries = max_tries

    def apply(self, img, holes=(), **params):
        mean_px = tuple(map(int, img.mean(axis=(0,1))))
        for y1, x1, y2, x2 in holes:
            img[y1:y2, x1:x2] = mean_px
        return img

    def apply_to_mask(self, mask, **params):
        return mask

    def get_params(self):
        return {}

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params['image'].shape[:2]
        mask = (params['mask'] > 0)
        holes = []
        hole_h = int(self.hole_frac * img_h)
        hole_w = int(self.hole_frac * img_w)
        for _ in range(self.max_holes):
            for _ in range(self.max_tries):
                x1 = np.random.randint(0, img_w - hole_w)
                y1 = np.random.randint(0, img_h - hole_h)
                x2, y2 = x1 + hole_w, y1 + hole_h
                patch = mask[y1:y2, x1:x2]
                if patch.sum() / (hole_h * hole_w) <= self.max_mask_overlap_frac:
                    holes.append((y1, x1, y2, x2))
                    break
        return {'holes': holes}

    def get_transform_init_args_names(self):
        return ('max_holes','hole_frac','max_mask_overlap_frac','max_tries')

# ---------------------------
# Augmentation Pipelines
# ---------------------------
def get_training_augmentation():
    return A.Compose([
        # geometric
        A.Rotate(limit=360, p=0.5),
        A.HorizontalFlip(p=0.5),
        # A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_REFLECT),
        # final mask-aware crop
        # RandomCropWithMask(height=IMG_SIZE, width=IMG_SIZE, min_mask_frac=0.005, max_tries=5, p=1.0),
        # photometric
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4), # Increased intensity
        # A.GaussianBlur(blur_limit=7, p=0.2),
        # A.GaussNoise(p=0.2),
        # mask-safe dropout
        # MaskAwareDropout(max_holes=8, hole_frac=0.05, max_mask_overlap_frac=0.1, max_tries=10, p=0.3),
    ], additional_targets={'mask': 'mask'})


def get_validation_augmentation():
    return A.Compose([
        # A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_REFLECT, p=1),
        # A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE, p=1),
    ], additional_targets={'mask': 'mask'})


# ---------------------------
# Dataset Definition
# ---------------------------
class SubstationDataset(Dataset):
    """
    PyTorch Dataset for substation transformer segmentation.
    Expects:
      - images_dir: folder containing RGB images (*.png, *.jpg)
      - coco_json: path to COCO-format JSON listing only images/annotations for this split
    Returns (image_tensor, mask_tensor, filename).
    """
    def __init__(self, images_dir, coco_json, augmentation=None, preprocessing_fn=None):
        # 1) Grab all filtered PNGs
        self.image_paths = sorted(glob(os.path.join(images_dir, '*.jpg')))
        
        # 2) Load your prefiltered COCO JSON
        with open(coco_json, 'r') as f:
            coco = json.load(f)

        # 3) Find the transformer category ID
        transformer = next(
            c for c in coco['categories']
            if c['name'].lower() == 'transformer'
        )
        self.tid = transformer['id']

        # 4) Map filenames → image_id
        self.name2id = {img['file_name']: img['id'] for img in coco['images']}

        # 5) Group only true polygon annotations by image_id
        self.anns_by_image = defaultdict(list)
        for ann in coco['annotations']:
            if (ann['category_id'] == self.tid
                and isinstance(ann.get('segmentation'), list)
                and len(ann['segmentation']) > 0):
                self.anns_by_image[ann['image_id']].append(ann)

        # 6) Store transforms and preprocessing
        self.augmentation     = augmentation
        self.preprocessing_fn = preprocessing_fn

        print(f"Loaded {len(self.image_paths)} images for transformer segmentation")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # --- Load image ---
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        image_id = self.name2id[filename]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # --- Build mask from polygons ---
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in self.anns_by_image[image_id]:
            for poly in ann['segmentation']:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 1)

        # --- Apply augmentation ---
        if self.augmentation:
            data = self.augmentation(image=img, mask=mask)
            img, mask = data['image'], data['mask']

        # --- Preprocessing (e.g. normalization) ---
        if self.preprocessing_fn:
            img = self.preprocessing_fn(img)

        # --- To tensor CHW ---
        img  = img.astype('float32').transpose(2, 0, 1)
        # Add channel dimension for the mask -> [1, H, W] # <-- CHANGED
        mask = mask.astype('float32')[np.newaxis, :, :]

        return torch.from_numpy(img), torch.from_numpy(mask), filename

# ---------------------------------------------------------------------
# Training + validation
# ---------------------------------------------------------------------
def train_model(args):
    # 1) Device ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available()
                          and args.device == "cuda" else "cpu")
    print("Using device:", device)

    # 2) Data -----------------------------------------------------------
    global PREPROCESS_FN
    PREPROCESS_FN = smp.encoders.get_preprocessing_fn(args.encoder, "imagenet")

    train_ds = SubstationDataset(args.train_images_dir, args.train_coco_json,
                                 augmentation=get_training_augmentation(),
                                 preprocessing_fn=PREPROCESS_FN)
    val_ds   = SubstationDataset(args.val_images_dir,   args.val_coco_json,
                                 augmentation=get_validation_augmentation(),
                                 preprocessing_fn=PREPROCESS_FN)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,  shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # 3) Model / loss / optim ------------------------------------------
    model = smp.Unet(encoder_name=args.encoder, encoder_weights="imagenet",
                     in_channels=3, classes=1).to(device)

    if args.loss_ft:
        loss_fn = smp.losses.TverskyLoss(mode="binary", alpha=0.7, gamma=0.75)
        print("Loss: Focal-Tversky (α 0.7, γ 0.75)")
    else:
        bce  = nn.BCEWithLogitsLoss()
        dice = smp.losses.DiceLoss(mode="binary")
        loss_fn = lambda p, t: 0.5 * bce(p, t) + 0.5 * dice(p, t)
        print("Loss: 0.5 × BCE  +  0.5 × Dice")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if args.one_cycle:
        scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                               epochs=args.epochs,
                               steps_per_epoch=len(train_loader),
                               pct_start=0.3, anneal_strategy="cos")
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                      factor=0.5, patience=5, verbose=True)

    best_val_iou   = 0.0
    stagnant_epochs = 0
    os.makedirs(args.log_dir, exist_ok=True)

    # 4) Optional dry-run ----------------------------------------------
    if args.dry_run:
        imgs, masks, *_ = next(iter(train_loader))
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad():
            preds = model(imgs)
        print(f"Dry-run shapes  imgs:{imgs.shape}  masks:{masks.shape}  preds:{preds.shape}")
        sys.exit(0)

    # 5) Main loop ------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        print("-" * 30)
        # ---- Train ----------------------------------------------------
        model.train()
        running_loss = 0.0

        for imgs, masks, *_ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss  = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.one_cycle:          # advance One-Cycle LR *per batch*
                scheduler.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{args.epochs}  |  Train Loss: {avg_train_loss:.4f}")

        # ---- Validate -------------------------------------------------
        model.eval()
        val_iou = []

        with torch.no_grad():
            for imgs, masks, *_ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)

                logits = model(imgs)
                if args.tta_flip:
                    logits = (logits + model(torch.flip(imgs, dims=[3]))) / 2

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                inter = (preds * masks).sum()
                union = preds.sum() + masks.sum() - inter
                val_iou.append(((inter + 1e-6) / (union + 1e-6)).item())

        avg_val_iou = np.mean(val_iou)

        # LR step for Plateau schedule
        if not args.one_cycle:
            scheduler.step(avg_val_iou)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"           Val IoU: {avg_val_iou:.4f}  |  LR: {lr_now:.6f}")

        # ---- Checkpoint / early-stop ---------------------------------
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            stagnant_epochs = 0
            ckpt_name = f"model_best_epoch{epoch}_valIoU{avg_val_iou:.4f}.pth"
            torch.save(model.state_dict(), os.path.join(args.log_dir, ckpt_name))
            print(f"  → New best model saved: {ckpt_name}")
        else:
            stagnant_epochs += 1
            print(f"  → No improvement for {stagnant_epochs} epoch(s)")

        if args.early_stop and stagnant_epochs >= args.early_stop:
            print(f"Early stopping (patience = {args.early_stop})")
            break

    print(f"\nTraining complete. Best Val IoU: {best_val_iou:.4f}")



# ---------------------------
# Argument Parsing
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net for transformer segmentation.")

    # Data paths
    p.add_argument('--train_images_dir', type=str, required=True, help='Path to folder of training images.')
    p.add_argument('--train_coco_json',  type=str, required=True, help='Path to COCO JSON for training.')
    p.add_argument('--val_images_dir', type=str, required=True, help='Path to folder of validation images.')
    p.add_argument('--val_coco_json',  type=str, required=True, help='Path to COCO JSON for validation.')

    # Training hyperparameters
    p.add_argument('--batch_size',  type=int, default=8)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--epochs',      type=int, default=50)
    p.add_argument('--num_workers', type=int, default=4)

    # System and logging
    p.add_argument('--device',      type=str, default='cuda', choices=['cpu', 'cuda'])
    p.add_argument('--dry_run',     action='store_true', help='Perform a single batch test and exit.')
    p.add_argument('--log_dir',     type=str, default='runs', help='Directory to save best checkpoints.')
    p.add_argument('--encoder', type=str, default='resnet34', help='SMP backbone, e.g. resnet34, efficientnet-b3…')
    p.add_argument('--one_cycle', action='store_true', help='Use One-Cycle LR schedule')
    p.add_argument('--early_stop', type=int, default=0, help='Patience in epochs (0 = off, i.e. disabled)')
    p.add_argument('--tta_flip', action='store_true', help='Average logits of image and its horizontal flip during validation')
    p.add_argument('--loss_ft', action='store_true', help='Use Focal-Tversky loss (alpha 0.7, gamma 0.75) instead of BCE+Dice')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    PREPROCESS_FN = smp.encoders.get_preprocessing_fn(args.encoder, 'imagenet')
    train_model(args)