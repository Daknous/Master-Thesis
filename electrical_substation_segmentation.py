"""
train_all.py

Single-script training pipeline for transformer segmentation using PyTorch and Albumentations.
Supports a dry-run mode for local testing.
Usage:
    python train_all.py \
        --images_dir /path/to/train/images \
        --coco_json /path/to/annotations.json \
        --batch_size 8 \
        --lr 1e-4 \
        --epochs 50 \
        --num_workers 4 \
        --val_split validation \
        [--device cuda] [--dry_run]
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
ENCODER = 'resnet34'

# Prepare preprocessing function
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


def get_validation_augmentation():
    return A.Compose([
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, p=1),
        A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE, p=1),
    ], additional_targets={'mask': 'mask'})

# ---------------------------
# Dataset Definition
# ---------------------------
class SubstationDataset(Dataset):
    def __init__(self, images_dir, coco_json, augmentation=None, preprocessing_fn=None):
        self.image_paths = sorted(
            glob(os.path.join(images_dir, '*.png')) + glob(os.path.join(images_dir, '*.jpg'))
        )
        # Load COCO JSON
        with open(coco_json, 'r') as f:
            coco = json.load(f)
        # Find transformer category
        transformer = next(c for c in coco['categories'] if c['name'].lower()=='transformer')
        self.transformer_id = transformer['id']
        # Group annotations by image id
        self.anns_by_image = {}
        for ann in coco['annotations']:
            if ann['category_id']==self.transformer_id:
                self.anns_by_image.setdefault(ann['image_id'], []).append(ann)
        self.imgname_to_id = {img['file_name']: img['id'] for img in coco['images']}
        self.augmentation = augmentation
        self.preprocessing_fn = preprocessing_fn

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        image_id = self.imgname_to_id[filename]
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        # Rasterize mask
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in self.anns_by_image.get(image_id, []):
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                for seg in ann['segmentation']:
                    pts = np.array(seg).reshape(-1,2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
            else:
                x,y,bw,bh = map(int, ann['bbox'])
                cv2.rectangle(mask, (x,y), (x+bw, y+bh), 1, -1)
        # One-hot encode [background, transformer]
        mask = mask.astype('float32')
        bg = 1.0 - mask
        mask = np.stack([bg, mask], axis=-1)
        # Augment
        if self.augmentation:
            data = self.augmentation(image=image, mask=mask)
            image, mask = data['image'], data['mask']
        # Preprocess
        if self.preprocessing_fn:
            image = self.preprocessing_fn(image)
        # Convert to tensor, CHW
        image = image.astype('float32').transpose(2,0,1)
        mask  = mask.astype('float32').transpose(2,0,1)
        return torch.from_numpy(image), torch.from_numpy(mask)

# ---------------------------
# Training Routine
# ---------------------------
def train_model(args):
    # Determine device
    if args.device=='auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Data loaders
    train_dir = args.images_dir
    val_dir   = train_dir.replace('train', args.val_split)
    train_ds = SubstationDataset(
        images_dir=train_dir,
        coco_json=args.coco_json,
        augmentation=get_training_augmentation(),
        preprocessing_fn=PREPROCESS_FN
    )
    val_ds = SubstationDataset(
        images_dir=val_dir,
        coco_json=args.coco_json,
        augmentation=get_validation_augmentation(),
        preprocessing_fn=PREPROCESS_FN
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model, loss, optimizer
    model = smp.Unet(encoder_name=ENCODER, encoder_weights='imagenet', in_channels=3, classes=2)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Dry run
    if args.dry_run:
        imgs, masks = next(iter(train_loader))
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        print(f"Dry run shapes -> imgs: {imgs.shape}, masks: {masks.shape}, preds: {preds.shape}")
        sys.exit(0)

    # Training loop
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{args.epochs} - Loss: {total_loss/len(train_loader):.4f}")

# ---------------------------
# Argument Parsing
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--images_dir',  required=True)
    p.add_argument('--coco_json',   required=True)
    p.add_argument('--batch_size',  type=int, default=8)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--epochs',      type=int, default=50)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--val_split',   type=str, default='validation')
    p.add_argument('--device',      type=str, default='auto', choices=['auto','cpu','cuda'])
    p.add_argument('--dry_run',     action='store_true')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train_model(args)
