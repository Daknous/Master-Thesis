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
        --use_focal \
        --hard_mining \
        --use_boundary \
        --adaptive_thresh \
        --post_process \
        --min_size 100 \
        --merge_distance 20 \
        --decoder_attention scse

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

# --- Additional imports for new features ---
from scipy import ndimage
import torch.nn.functional as F

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
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits, targets):
        # Flatten the tensors
        logits = logits.view(-1)
        targets = targets.view(-1)
        
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt).pow(self.gamma)
        
        # Apply alpha weighting
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_t * focal_weight * bce_loss
        
        return focal_loss.mean()

# --- Boundary-aware loss ---
class BoundaryLoss(nn.Module):
    def __init__(self, theta=5):
        super().__init__()
        self.theta = theta
        self.lap = torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=torch.float32).view(1,1,3,3)
    def forward(self, pred, mask):
        device = pred.device
        lap = self.lap.to(device)
        pred_prob = torch.sigmoid(pred)
        boundary_targets = F.conv2d(mask, lap, padding=1)
        boundary_targets = (boundary_targets > 0.1).float()
        pred_b = F.conv2d(pred_prob, lap, padding=1)
        boundary_loss = F.binary_cross_entropy_with_logits(
            pred_b * self.theta, boundary_targets, reduction='none')
        return boundary_loss.mean()
    
def multi_scale_tta(model, imgs, device, scales=[0.8, 1.0, 1.2]):
    """Multi-scale + flip TTA"""
    all_preds = []
    
    for scale in scales:
        # Resize image
        h, w = imgs.shape[2:4]
        new_h, new_w = int(h * scale), int(w * scale)
        
        if scale != 1.0:
            scaled_imgs = torch.nn.functional.interpolate(
                imgs, size=(new_h, new_w), mode='bilinear', align_corners=False
            )
        else:
            scaled_imgs = imgs
        
        # Original
        pred = model(scaled_imgs)
        pred = torch.nn.functional.interpolate(
            pred, size=(h, w), mode='bilinear', align_corners=False
        )
        all_preds.append(pred)
        
        # H-flip
        pred_hflip = model(torch.flip(scaled_imgs, dims=[3]))
        pred_hflip = torch.flip(pred_hflip, dims=[3])
        pred_hflip = torch.nn.functional.interpolate(
            pred_hflip, size=(h, w), mode='bilinear', align_corners=False
        )
        all_preds.append(pred_hflip)
        
        # V-flip
        pred_vflip = model(torch.flip(scaled_imgs, dims=[2]))
        pred_vflip = torch.flip(pred_vflip, dims=[2])
        pred_vflip = torch.nn.functional.interpolate(
            pred_vflip, size=(h, w), mode='bilinear', align_corners=False
        )
        all_preds.append(pred_vflip)
    
    return torch.stack(all_preds).mean(dim=0)

# --- Adaptive threshold selection ---
def find_optimal_threshold(probs, mask, thresholds=np.arange(0.3, 0.8, 0.05)):
    best_iou = 0
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

# --- Post-processing: connected component filtering ---
def post_process_predictions(pred_mask, min_size=100, merge_distance=20):
    mask = pred_mask.astype(np.uint8)
    # Remove small components
    labeled, num = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(num + 1))
    mask_clean = np.zeros_like(mask)
    for i in range(1, num + 1):
        if sizes[i] >= min_size:
            mask_clean[labeled == i] = 1
    # Morphological closing to merge nearby components
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_distance, merge_distance))
    mask_merged = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    return mask_merged

def compute_detection_metrics(pred_mask, true_mask, iou_threshold=0.5):
    """
    Compute detection accuracy metrics (precision, recall, F1) at object level
    """
    from scipy import ndimage
    from scipy.spatial.distance import cdist
    
    # Find connected components (detected objects)
    pred_labeled, pred_num = ndimage.label(pred_mask)
    true_labeled, true_num = ndimage.label(true_mask)
    
    if pred_num == 0 and true_num == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    
    if pred_num == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": true_num}
    
    if true_num == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": pred_num, "fn": 0}
    
    # Compute IoU between all predicted and true objects
    ious = np.zeros((pred_num, true_num))
    
    for i in range(1, pred_num + 1):
        pred_obj = (pred_labeled == i)
        for j in range(1, true_num + 1):
            true_obj = (true_labeled == j)
            intersection = (pred_obj & true_obj).sum()
            union = (pred_obj | true_obj).sum()
            ious[i-1, j-1] = intersection / (union + 1e-6)
    
    # Match predictions to ground truth using Hungarian algorithm (greedy approximation)
    matched_pred = set()
    matched_true = set()
    tp = 0
    
    # Sort by IoU score (highest first)
    matches = []
    for i in range(pred_num):
        for j in range(true_num):
            if ious[i, j] > iou_threshold:
                matches.append((ious[i, j], i, j))
    
    matches.sort(reverse=True)
    
    for iou_score, pred_idx, true_idx in matches:
        if pred_idx not in matched_pred and true_idx not in matched_true:
            matched_pred.add(pred_idx)
            matched_true.add(true_idx)
            tp += 1
    
    fp = pred_num - tp
    fn = true_num - tp
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        "precision": precision,
        "recall": recall, 
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

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
    # Model with optional attention
    decoder_attention = args.decoder_attention if hasattr(args, 'decoder_attention') else None
    model = smp.Unet(encoder_name=args.encoder, encoder_weights="imagenet",
                     in_channels=3, classes=1,
                     decoder_attention_type=decoder_attention).to(device)

    # Loss selection
    if args.use_focal:
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        base_loss_msg = "Focal Loss (α=0.25, γ=2.0)"
    elif args.loss_ft:
        loss_fn = smp.losses.TverskyLoss(mode="binary", alpha=0.7, gamma=0.75)
        base_loss_msg = "Focal-Tversky (α 0.7, γ 0.75)"
    else:
        bce  = nn.BCEWithLogitsLoss()
        dice = smp.losses.DiceLoss(mode="binary")
        loss_fn = lambda p, t: 0.5 * bce(p, t) + 0.5 * dice(p, t)
        base_loss_msg = "0.5 × BCE  +  0.5 × Dice"

    bnd_loss = BoundaryLoss() if args.use_boundary else None

    # Print complete loss configuration
    if bnd_loss is not None:
        print(f"Loss: {base_loss_msg}  +  0.1 × Boundary")
    else:
        print(f"Loss: {base_loss_msg}")



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
            
            # Add boundary loss if enabled
            if bnd_loss is not None:
                boundary_loss = bnd_loss(preds, masks)
                loss += 0.1 * boundary_loss  # 10% weight for boundary loss

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
        val_metrics = {"precision": [], "recall": [], "f1": []}

        with torch.no_grad():
            for imgs, masks, *_ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)

                # Multi-scale TTA if enabled
                if args.multi_scale:
                    logits = multi_scale_tta(model, imgs, device)
                else:
                    logits = model(imgs)
                    
                    # Simple TTA flip if enabled
                    if args.tta_flip:
                        logits_flip = model(torch.flip(imgs, dims=[3]))
                        logits_flip = torch.flip(logits_flip, dims=[3])
                        logits = (logits + logits_flip) / 2

                probs = torch.sigmoid(logits)
                
                # Convert to numpy for post-processing
                probs_np = probs.squeeze().cpu().numpy()
                masks_np = masks.squeeze().cpu().numpy()
                
                # Adaptive thresholding if enabled
                if args.adaptive_thresh:
                    threshold = find_optimal_threshold(
                        torch.from_numpy(probs_np), 
                        torch.from_numpy(masks_np)
                    )
                else:
                    threshold = 0.5
                
                # Apply threshold
                preds_np = (probs_np > threshold).astype(np.float32)
                
                # Post-processing if enabled
                if args.post_process:
                    preds_np = post_process_predictions(
                        preds_np, 
                        min_size=args.min_size, 
                        merge_distance=args.merge_distance
                    )
                
                # Compute IoU
                intersection = (preds_np * masks_np).sum()
                union = preds_np.sum() + masks_np.sum() - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                val_iou.append(iou)
                
                # Compute detection metrics
                det_metrics = compute_detection_metrics(preds_np, masks_np, iou_threshold=0.5)
                val_metrics["precision"].append(det_metrics["precision"])
                val_metrics["recall"].append(det_metrics["recall"])
                val_metrics["f1"].append(det_metrics["f1"])

        avg_val_iou = np.mean(val_iou)
        avg_precision = np.mean(val_metrics["precision"])
        avg_recall = np.mean(val_metrics["recall"])
        avg_f1 = np.mean(val_metrics["f1"])

        # LR step for Plateau schedule
        if not args.one_cycle:
            scheduler.step(avg_val_iou)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"           Val IoU: {avg_val_iou:.4f}  |  P: {avg_precision:.3f}  R: {avg_recall:.3f}  F1: {avg_f1:.3f}  |  LR: {lr_now:.6f}")

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

    # --- New flags for advanced features ---
    p.add_argument('--use_focal', action='store_true', help='Use Focal Loss')
    p.add_argument('--hard_mining', action='store_true', help='Enable hard negative mining')
    p.add_argument('--use_boundary', action='store_true', help='Add boundary loss')
    p.add_argument('--adaptive_thresh', action='store_true', help='Enable adaptive thresholding during validation')
    p.add_argument('--post_process', action='store_true', help='Apply post-processing to predictions')
    p.add_argument('--min_size', type=int, default=100, help='Min size for post-processing components')
    p.add_argument('--merge_distance', type=int, default=20, help='Merge distance for post-processing')
    p.add_argument('--decoder_attention', type=str, default=None, help='Decoder attention type (e.g. scse)')
    p.add_argument('--multi_scale', action='store_true', help='Enable multi-scale TTA during validation')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_model(args)