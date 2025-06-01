"""
Single‐script training pipeline for transformer segmentation using PyTorch and Albumentations.
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
"""

import os
import json
import argparse
import sys
import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from glob import glob

# ---------------------------
# Configuration
# ---------------------------
IMG_SIZE = 512
ENCODER = 'resnet50'
PREPROCESS_FN = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet')


# ---------------------------
# Augmentation Pipelines
# ---------------------------
def get_training_augmentation():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Affine(rotate=(-30,30), scale=(0.9,1.1), translate_percent=(0.1,0.1), p=0.8, border_mode=0),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, p=1),
        A.RandomCrop(height=IMG_SIZE, width=IMG_SIZE, p=1),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.GridDistortion(p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
    ], additional_targets={'mask': 'mask'})


def get_validation_augmentation():
    return A.Compose([
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, p=1),
        A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE, p=1),
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
        # List all PNG/JPG files in images_dir
        self.image_paths = sorted(
            glob(os.path.join(images_dir, '*.png')) + glob(os.path.join(images_dir, '*.jpg'))
        )

        # Load COCO JSON for this split
        with open(coco_json, 'r') as f:
            coco = json.load(f)

        # Identify the “transformer” category
        transformer = next(c for c in coco['categories'] if c['name'].lower() == 'transformer')
        self.tid = transformer['id']

        # Group annotations by image_id
        self.anns_by_image = {}
        for ann in coco['annotations']:
            if ann['category_id'] == self.tid:
                self.anns_by_image.setdefault(ann['image_id'], []).append(ann)

        # Build a map: filename → image_id
        self.name2id = {img['file_name']: img['id'] for img in coco['images']}

        self.augmentation = augmentation
        self.preprocessing_fn = preprocessing_fn

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load RGB image
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        image_id = self.name2id[filename]  # Look up its ID in this split’s JSON

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # 2. Build a binary mask of size (H,W)
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in self.anns_by_image.get(image_id, []):
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                # Polygon‐based segmentation
                for seg in ann['segmentation']:
                    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
            else:
                # Fallback to bounding box [x,y,width,height]
                x, y, bw, bh = map(int, ann['bbox'])
                cv2.rectangle(mask, (x, y), (x + bw, y + bh), 1, -1)

        # 3. One‐hot encode: [background, transformer]
        mask = mask.astype('float32')
        bg = 1.0 - mask
        mask = np.stack([bg, mask], axis=-1)

        # 4. Apply augmentations (simultaneously to image & mask)
        if self.augmentation:
            data = self.augmentation(image=image, mask=mask)
            image, mask = data['image'], data['mask']

        # 5. Preprocess image (e.g. ResNet mean/std)
        if self.preprocessing_fn:
            image = self.preprocessing_fn(image)

        # 6. Convert to PyTorch format: CHW
        image = image.astype('float32').transpose(2, 0, 1)  # [3,H,W]
        mask = mask.astype('float32').transpose(2, 0, 1)    # [2,H,W]

        return torch.from_numpy(image), torch.from_numpy(mask), filename


# ---------------------------
# Training & Validation Routine
# ---------------------------
def train_model(args):
    # 1) Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # 2) Build Datasets & DataLoaders
    train_ds = SubstationDataset(
        images_dir       = args.train_images_dir,
        coco_json        = args.train_coco_json,
        augmentation     = get_training_augmentation(),
        preprocessing_fn = PREPROCESS_FN
    )
    val_ds = SubstationDataset(
        images_dir       = args.val_images_dir,
        coco_json        = args.val_coco_json,
        augmentation     = get_validation_augmentation(),
        preprocessing_fn = PREPROCESS_FN
    )

    train_loader = DataLoader(
        train_ds,
        batch_size   = args.batch_size,
        shuffle      = True,
        num_workers  = args.num_workers,
        pin_memory   = True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size   = 1,
        shuffle      = False,
        num_workers  = args.num_workers,
        pin_memory   = True
    )

    # 3) Model, loss, optimizer
    model = smp.Unet(
        encoder_name    = ENCODER,
        encoder_weights = 'imagenet',
        in_channels     = 3,
        classes         = 2
    )
    model.to(device)

    # Composite loss: BCEWithLogits + Dice
    bce_loss  = nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode='binary')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_iou = 0.0

    # 4) Dry‐run (optional)
    if args.dry_run:
        imgs, masks, _ = next(iter(train_loader))
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad():
            preds = model(imgs)
        print(f"Dry run shapes → imgs: {imgs.shape}, masks: {masks.shape}, preds: {preds.shape}")
        sys.exit(0)

    # 5) Main training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0

        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs)  # [B,2,512,512]
            loss = bce_loss(preds, masks) + dice_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_train_loss = total_loss / count
        print(f"Epoch {epoch}/{args.epochs} → Train Loss: {avg_train_loss:.4f}")

        # 6) Validation loop (compute IoU at threshold=0.5)
        model.eval()
        val_iou = 0.0
        val_count = 0

        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)

                logits = model(imgs)                      # [1,2,512,512]
                probs  = torch.sigmoid(logits)[0, 1]      # squeeze batch, pick channel=1
                true   = masks[0, 1]                      # ground‐truth transformer mask

                pred_bin = (probs > 0.5).float()
                inter    = (pred_bin * true).sum().item()
                union    = ((pred_bin + true) > 0).sum().item()
                iou      = inter / union if union > 0 else 1.0

                val_iou += iou
                val_count += 1

        avg_val_iou = val_iou / val_count
        print(f"Epoch {epoch}/{args.epochs} → Val IoU: {avg_val_iou:.4f}")

        # 7) Checkpoint if validation IoU improved
        ckpt_name = f"model_epoch{epoch}_valIoU{avg_val_iou:.4f}.pth"
        ckpt_path = os.path.join(args.log_dir, ckpt_name)
        os.makedirs(args.log_dir, exist_ok=True)

        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → New best model saved: {ckpt_name}")

    print(f"\nTraining complete. Best Val IoU: {best_val_iou:.4f}")


# ---------------------------
# Argument Parsing
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()

    # Train split
    p.add_argument('--train_images_dir', type=str, required=True,
                   help='Path to folder of training images (e.g. Dataset/train)')
    p.add_argument('--train_coco_json',  type=str, required=True,
                   help='Path to COCO JSON for training (e.g. Dataset/train/_annotations.coco.json)')

    # Validation split
    p.add_argument('--val_images_dir', type=str, required=True,
                   help='Path to folder of validation images (e.g. Dataset/valid)')
    p.add_argument('--val_coco_json',  type=str, required=True,
                   help='Path to COCO JSON for validation (e.g. Dataset/valid/_annotations.coco.json)')

    # Training hyperparameters
    p.add_argument('--batch_size',  type=int, default=8)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--epochs',      type=int, default=50)
    p.add_argument('--num_workers', type=int, default=4)

    # Device & logging
    p.add_argument('--device',      type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--dry_run',     action='store_true',
                   help='Perform a single batch test and exit')
    p.add_argument('--log_dir',     type=str, default='runs',
                   help='Directory in which to save best‐IoU checkpoints')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_model(args)
