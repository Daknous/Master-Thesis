"""
Single script training pipeline for transformer segmentation using PyTorch and Albumentations.

Two modes:
1) Standard (default): train/val split with proper validation and best-val IoU checkpoint.
2) Production all-data mode: --train_on_all
   -> uses only train_images_dir/train_coco_json as ALL labelled data
   -> no validation, no early stopping, fixed number of epochs
   -> checkpoints like model_all_epoch{epoch}.pth
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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import segmentation_models_pytorch as smp
import albumentations as A

from scipy import ndimage
import torch.nn.functional as F

# ---------------------------
# Configuration
# ---------------------------
IMG_SIZE = 1200
DEFAULT_ENCODER = "resnet34"
PREPROCESS_FN = smp.encoders.get_preprocessing_fn(DEFAULT_ENCODER, "imagenet")


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
        return img[y : y + self.height, x : x + self.width]

    def apply_to_mask(self, mask, x=0, y=0, **params):
        return mask[y : y + self.height, x : x + self.width]

    def get_params(self):
        return {}

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]
        mask = (params["mask"] > 0).astype(np.uint8)

        ys, xs = np.where(mask)
        if len(ys) == 0:
            # no mask: center crop
            return {"x": (img_w - self.width) // 2, "y": (img_h - self.height) // 2}

        cy, cx = ys.mean(), xs.mean()
        area = self.height * self.width

        # try random crops
        for _ in range(self.max_tries):
            x = np.random.randint(0, img_w - self.width)
            y = np.random.randint(0, img_h - self.height)
            patch = mask[y : y + self.height, x : x + self.width]
            if patch.sum() / area >= self.min_mask_frac:
                return {"x": x, "y": y}

        # fallback: center on mask centroid
        x0 = int(np.clip(cx - self.width / 2, 0, img_w - self.width))
        y0 = int(np.clip(cy - self.height / 2, 0, img_h - self.height))
        return {"x": x0, "y": y0}

    def get_transform_init_args_names(self):
        return ("height", "width", "min_mask_frac", "max_tries")


class MaskAwareDropout(A.DualTransform):
    def __init__(self, max_holes=8, hole_frac=0.05, max_mask_overlap_frac=0.1, max_tries=10, p=0.3):
        super().__init__(always_apply=False, p=p)
        self.max_holes = max_holes
        self.hole_frac = hole_frac
        self.max_mask_overlap_frac = max_mask_overlap_frac
        self.max_tries = max_tries

    def apply(self, img, holes=(), **params):
        mean_px = tuple(map(int, img.mean(axis=(0, 1))))
        for y1, x1, y2, x2 in holes:
            img[y1:y2, x1:x2] = mean_px
        return img

    def apply_to_mask(self, mask, **params):
        return mask

    def get_params(self):
        return {}

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]
        mask = (params["mask"] > 0)
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
        return {"holes": holes}

    def get_transform_init_args_names(self):
        return ("max_holes", "hole_frac", "max_mask_overlap_frac", "max_tries")


# ---------------------------
# Augmentation Pipelines
# ---------------------------
def get_training_augmentation():
    return A.Compose(
        [
            A.Rotate(limit=360, p=0.5),
            A.HorizontalFlip(p=0.5),
            # A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_REFLECT),
            # RandomCropWithMask(height=IMG_SIZE, width=IMG_SIZE, min_mask_frac=0.005, max_tries=5, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            # A.GaussianBlur(blur_limit=7, p=0.2),
            # A.GaussNoise(p=0.2),
            # MaskAwareDropout(max_holes=8, hole_frac=0.05, max_mask_overlap_frac=0.1, max_tries=10, p=0.3),
        ],
        additional_targets={"mask": "mask"},
    )


def get_validation_augmentation():
    return A.Compose(
        [
            # A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_REFLECT, p=1),
            # A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE, p=1),
        ],
        additional_targets={"mask": "mask"},
    )


# ---------------------------
# Dataset Definition
# ---------------------------
class SubstationDataset(Dataset):
    """
    PyTorch Dataset for substation transformer segmentation.
    Expects:
      - images_dir: folder containing RGB images (*.jpg)
      - coco_json: path to COCO-format JSON listing images/annotations
    Returns (image_tensor, mask_tensor, filename).
    """

    def __init__(self, images_dir, coco_json, augmentation=None, preprocessing_fn=None):
        self.image_paths = sorted(glob(os.path.join(images_dir, "*.jpg")))

        with open(coco_json, "r") as f:
            coco = json.load(f)

        # Find transformer category
        transformer = next(c for c in coco["categories"] if c["name"].lower() == "transformer")
        self.tid = transformer["id"]

        # Map filenames → image_id
        self.name2id = {img["file_name"]: img["id"] for img in coco["images"]}

        # Group annotations by image_id
        self.anns_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            if (
                ann["category_id"] == self.tid
                and isinstance(ann.get("segmentation"), list)
                and len(ann["segmentation"]) > 0
            ):
                self.anns_by_image[ann["image_id"]].append(ann)

        self.augmentation = augmentation
        self.preprocessing_fn = preprocessing_fn

        print(f"Loaded {len(self.image_paths)} images for transformer segmentation from {images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        image_id = self.name2id[filename]

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in self.anns_by_image[image_id]:
            for poly in ann["segmentation"]:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 1)

        if self.augmentation:
            data = self.augmentation(image=img, mask=mask)
            img, mask = data["image"], data["mask"]

        if self.preprocessing_fn:
            img = self.preprocessing_fn(img)

        img = img.astype("float32").transpose(2, 0, 1)
        mask = mask.astype("float32")[np.newaxis, :, :]

        return torch.from_numpy(img), torch.from_numpy(mask), filename


# ---------------------------------------------------------------------
# Losses and helpers
# ---------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.view(-1)

        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt).pow(self.gamma)

        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        focal_loss = alpha_t * focal_weight * bce_loss
        return focal_loss.mean()


class BoundaryLoss(nn.Module):
    def __init__(self, theta=5):
        super().__init__()
        self.theta = theta
        self.lap = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, pred, mask):
        device = pred.device
        lap = self.lap.to(device)

        pred_prob = torch.sigmoid(pred)
        boundary_targets = F.conv2d(mask, lap, padding=1)
        boundary_targets = (boundary_targets > 0.1).float()

        pred_b = F.conv2d(pred_prob, lap, padding=1)
        boundary_loss = F.binary_cross_entropy_with_logits(pred_b * self.theta, boundary_targets, reduction="none")
        return boundary_loss.mean()


def multi_scale_tta(model, imgs, device, scales=[0.8, 1.0, 1.2]):
    """Multi-scale + flip TTA (used only in validation mode)."""
    all_preds = []
    h, w = imgs.shape[2:4]

    for scale in scales:
        new_h, new_w = int(h * scale), int(w * scale)
        if scale != 1.0:
            scaled_imgs = F.interpolate(imgs, size=(new_h, new_w), mode="bilinear", align_corners=False)
        else:
            scaled_imgs = imgs

        # Original
        pred = model(scaled_imgs)
        pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        all_preds.append(pred)

        # H-flip
        pred_hflip = model(torch.flip(scaled_imgs, dims=[3]))
        pred_hflip = torch.flip(pred_hflip, dims=[3])
        pred_hflip = F.interpolate(pred_hflip, size=(h, w), mode="bilinear", align_corners=False)
        all_preds.append(pred_hflip)

        # V-flip
        pred_vflip = model(torch.flip(scaled_imgs, dims=[2]))
        pred_vflip = torch.flip(pred_vflip, dims=[2])
        pred_vflip = F.interpolate(pred_vflip, size=(h, w), mode="bilinear", align_corners=False)
        all_preds.append(pred_vflip)

    return torch.stack(all_preds).mean(dim=0)


def find_optimal_threshold(probs, mask, thresholds=np.arange(0.3, 0.8, 0.05)):
    best_iou = 0.0
    best_t = 0.5
    for t in thresholds:
        pred = (probs > t).float()
        inter = (pred * mask).sum()
        union = pred.sum() + mask.sum() - inter
        iou = (inter + 1e-6) / (union + 1e-6)
        if iou > best_iou:
            best_iou = iou
            best_t = t
    return best_t


def post_process_predictions(pred_mask, min_size=100, merge_distance=20):
    mask = pred_mask.astype(np.uint8)
    labeled, num = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(num + 1))
    mask_clean = np.zeros_like(mask)
    for i in range(1, num + 1):
        if sizes[i] >= min_size:
            mask_clean[labeled == i] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_distance, merge_distance))
    mask_merged = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    return mask_merged


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def train_model(args):
    # 1) Device ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available()
                          and args.device == "cuda" else "cpu")
    print("Using device:", device)

    # 2) Data -----------------------------------------------------------
    global PREPROCESS_FN
    PREPROCESS_FN = smp.encoders.get_preprocessing_fn(args.encoder, "imagenet")

    # Always create the TRAIN dataset (train split)
    base_train_ds = SubstationDataset(
        args.train_images_dir,
        args.train_coco_json,
        augmentation=get_training_augmentation(),
        preprocessing_fn=PREPROCESS_FN,
    )

    if args.train_on_all:
        # -------- ALL-DATA TRAINING MODE --------
        datasets = [base_train_ds]

        # Optionally include VALID split
        if args.val_images_dir is not None and args.val_coco_json is not None:
            print("Including VALID split in all-data training.")
            val_train_ds = SubstationDataset(
                args.val_images_dir,
                args.val_coco_json,
                augmentation=get_training_augmentation(),  # training aug
                preprocessing_fn=PREPROCESS_FN,
            )
            datasets.append(val_train_ds)

        # Optionally include TEST split
        if args.test_images_dir is not None and args.test_coco_json is not None:
            print("Including TEST split in all-data training.")
            test_train_ds = SubstationDataset(
                args.test_images_dir,
                args.test_coco_json,
                augmentation=get_training_augmentation(),  # training aug
                preprocessing_fn=PREPROCESS_FN,
            )
            datasets.append(test_train_ds)

        if len(datasets) == 1:
            print("WARNING: --train_on_all set but only train split provided. "
                  "Training only on train split.")

        train_ds = ConcatDataset(datasets)
        val_loader = None  # no validation in this mode

    else:
        # -------- STANDARD TRAIN/VAL MODE --------
        if args.val_images_dir is None or args.val_coco_json is None:
            raise ValueError("val_images_dir and val_coco_json must be provided unless --train_on_all is set.")

        train_ds = base_train_ds
        val_ds = SubstationDataset(
            args.val_images_dir,
            args.val_coco_json,
            augmentation=get_validation_augmentation(),   # no heavy aug for val
            preprocessing_fn=PREPROCESS_FN,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 3) Model / loss / optim ------------------------------------------
    decoder_attention = args.decoder_attention if hasattr(args, "decoder_attention") else None
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        decoder_attention_type=decoder_attention,
    ).to(device)

    # Loss selection
    if args.use_focal:
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        print("Loss: Focal Loss (α=0.25, γ=2.0)")
    elif args.loss_ft:
        loss_fn = smp.losses.TverskyLoss(mode="binary", alpha=0.7, gamma=0.75)
        print("Loss: Focal-Tversky (α 0.7, γ 0.75)")
    else:
        bce = nn.BCEWithLogitsLoss()
        dice = smp.losses.DiceLoss(mode="binary")
        loss_fn = lambda p, t: 0.5 * bce(p, t) + 0.5 * dice(p, t)
        print("Loss: 0.5 × BCE  +  0.5 × Dice")

    bnd_loss = BoundaryLoss() if args.use_boundary else None

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if args.one_cycle:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy="cos",
        )
    else:
        # Plateau scheduler only makes sense with a validation metric.
        if args.train_on_all:
            scheduler = None
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, verbose=True)

    best_val_iou = 0.0
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
        print(f"Epoch {epoch}/{args.epochs}")
        # ---- Train ----------------------------------------------------
        model.train()
        running_loss = 0.0

        for imgs, masks, *_ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)

            # Hard mining (train-only)
            if args.hard_mining and epoch > 10:
                with torch.no_grad():
                    probs = torch.sigmoid(preds)
                    errors = torch.abs(probs - masks)
                    threshold = torch.quantile(errors.view(-1), 0.5)
                    hard_mask = errors > threshold

                loss = loss_fn(preds[hard_mask], masks[hard_mask])
            else:
                loss = loss_fn(preds, masks)

            if bnd_loss is not None:
                loss += 0.1 * bnd_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.one_cycle:
                scheduler.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"  Train Loss: {avg_train_loss:.4f}")

        # --- If training on all data: no validation, save checkpoints and move on ---
        if args.train_on_all:
            ckpt_name = f"model_all_epoch{epoch}.pth"
            ckpt_path = os.path.join(args.log_dir, ckpt_name)
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Saved all-data checkpoint: {ckpt_name}")
            # no early stopping, no scheduler.step on val metric
            continue

        # ---- Validate (standard mode only) ---------------------------
        model.eval()
        val_iou = []

        with torch.no_grad():
            for imgs, masks, *_ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)

                logits = model(imgs)
                if args.tta_flip:
                    logits_orig = model(imgs)
                    logits_flip = torch.flip(model(torch.flip(imgs, dims=[3])), dims=[3])
                    logits = (logits_orig + logits_flip) / 2
                elif args.multi_scale:
                    logits = multi_scale_tta(model, imgs, device)
                else:
                    logits = model(imgs)

                probs = torch.sigmoid(logits)
                if args.adaptive_thresh:
                    threshold_opt = find_optimal_threshold(probs, masks)
                    preds = (probs > threshold_opt).float()
                else:
                    preds = (probs > 0.5).float()

                if args.post_process:
                    pred_np = preds.squeeze().cpu().numpy()
                    if pred_np.ndim == 2:
                        pred_np = post_process_predictions(pred_np, args.min_size, args.merge_distance)
                        preds = torch.from_numpy(pred_np).to(device).float()
                        while preds.ndim < 4:
                            preds = preds.unsqueeze(0)

                inter = (preds * masks).sum()
                union = preds.sum() + masks.sum() - inter
                val_iou.append(((inter + 1e-6) / (union + 1e-6)).item())

        avg_val_iou = float(np.mean(val_iou)) if val_iou else 0.0

        if not args.one_cycle and scheduler is not None:
            scheduler.step(avg_val_iou)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Val IoU: {avg_val_iou:.4f}  |  LR: {lr_now:.6f}")

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

    if args.train_on_all:
        print("\nAll-data training complete.")
    else:
        print(f"\nTraining complete. Best Val IoU: {best_val_iou:.4f}")


# ---------------------------
# Argument Parsing
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net for transformer segmentation.")

    # Data paths
    p.add_argument('--train_images_dir', type=str, required=True,
                   help='Path to folder of training images.')
    p.add_argument('--train_coco_json',  type=str, required=True,
                   help='Path to COCO JSON for training.')

    # Validation paths (still needed for normal training mode)
    p.add_argument('--val_images_dir', type=str, default=None,
                   help='Path to folder of validation images.')
    p.add_argument('--val_coco_json',  type=str, default=None,
                   help='Path to COCO JSON for validation.')

    # NEW: optionally include test split in all-data training
    p.add_argument('--test_images_dir', type=str, default=None,
                   help='(Optional) Path to folder of test images (used only in --train_on_all).')
    p.add_argument('--test_coco_json',  type=str, default=None,
                   help='(Optional) Path to COCO JSON for test split (used only in --train_on_all).')

    # Training hyperparameters
    p.add_argument('--batch_size',  type=int, default=8)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--epochs',      type=int, default=50)
    p.add_argument('--num_workers', type=int, default=4)

    # System and logging
    p.add_argument('--device',      type=str, default='cuda', choices=['cpu', 'cuda'])
    p.add_argument('--dry_run',     action='store_true', help='Perform a single batch test and exit.')
    p.add_argument('--log_dir',     type=str, default='runs', help='Directory to save best checkpoints.')
    p.add_argument('--encoder', type=str, default='resnet34',
                   help='SMP backbone, e.g. resnet34, efficientnet-b3…')
    p.add_argument('--one_cycle', action='store_true', help='Use One-Cycle LR schedule')
    p.add_argument('--early_stop', type=int, default=0,
                   help='Patience in epochs (0 = off, i.e. disabled)')
    p.add_argument('--tta_flip', action='store_true',
                   help='Average logits of image and its horizontal flip during validation')
    p.add_argument('--loss_ft', action='store_true',
                   help='Use Focal-Tversky loss (alpha 0.7, gamma 0.75) instead of BCE+Dice')

    # --- New flags for advanced features ---
    p.add_argument('--use_focal', action='store_true', help='Use Focal Loss')
    p.add_argument('--hard_mining', action='store_true', help='Enable hard negative mining')
    p.add_argument('--use_boundary', action='store_true', help='Add boundary loss')
    p.add_argument('--adaptive_thresh', action='store_true',
                   help='Enable adaptive thresholding during validation')
    p.add_argument('--post_process', action='store_true',
                   help='Apply post-processing to predictions')
    p.add_argument('--min_size', type=int, default=100,
                   help='Min size for post-processing components')
    p.add_argument('--merge_distance', type=int, default=20,
                   help='Merge distance for post-processing')
    p.add_argument('--decoder_attention', type=str, default=None,
                   help='Decoder attention type (e.g. scse)')
    p.add_argument('--multi_scale', action='store_true',
                   help='Enable multi-scale TTA during validation')

    # NEW: all-data training mode
    p.add_argument('--train_on_all', action='store_true',
                   help='Train on train+valid(+test) as a single dataset, disable validation.')

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
