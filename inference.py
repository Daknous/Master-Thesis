"""
inference.py

Inference and evaluation script for transformer segmentation.
Loads a trained checkpoint, runs on validation patches, computes IoU & Dice,
and saves overlay images.

Usage:
    python inference.py \
        --images_dir /path/to/validation/images \
        --coco_json /path/to/annotations.json \
        --checkpoint /path/to/checkpoint.pth \
        --output_dir /path/to/save/overlays \
        [--device cuda] [--normalize_output]
"""
import os
import argparse
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A

# ---------------------------
# Configuration
# ---------------------------
IMG_SIZE = 512
ENCODER = 'resnet34'
PREPROCESS_FN = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet')

# ---------------------------
# Validation Augmentation
# ---------------------------
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
        self.image_paths = sorted([
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg'))
        ])
        with open(coco_json, 'r') as f:
            coco = json.load(f)
        transformer = next(c for c in coco['categories'] if c['name'].lower()=='transformer')
        self.tid = transformer['id']
        self.imgid2anns = {}
        for ann in coco['annotations']:
            if ann['category_id'] == self.tid:
                self.imgid2anns.setdefault(ann['image_id'], []).append(ann)
        self.name2id = {img['file_name']: img['id'] for img in coco['images']}
        self.augmentation = augmentation
        self.preprocessing_fn = preprocessing_fn

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        fname = os.path.basename(path)
        imgid = self.name2id[fname]
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in self.imgid2anns.get(imgid, []):
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                for seg in ann['segmentation']:
                    pts = np.array(seg).reshape(-1,2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
            else:
                x,y,bw,bh = map(int, ann['bbox'])
                cv2.rectangle(mask, (x,y), (x+bw, y+bh), 1, -1)
        mask = mask.astype('float32')
        bg = 1.0 - mask
        mask = np.stack([bg, mask], axis=-1)
        if self.augmentation:
            data = self.augmentation(image=image, mask=mask)
            image, mask = data['image'], data['mask']
        if self.preprocessing_fn:
            image = self.preprocessing_fn(image)
        image = image.astype('float32').transpose(2,0,1)
        mask  = mask.astype('float32').transpose(2,0,1)
        return torch.from_numpy(image), torch.from_numpy(mask), fname

# ---------------------------
# Inference & Metrics
# ---------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # DataLoader
    dataset = SubstationDataset(
        args.images_dir,
        args.coco_json,
        augmentation=get_validation_augmentation(),
        preprocessing_fn=PREPROCESS_FN
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Model
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=2
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.eval()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)

    total_iou = 0.0
    total_dice = 0.0

    for img, mask, fname_tuple in loader:
        fname = fname_tuple[0]
        img, mask = img.to(device), mask.to(device)
        with torch.no_grad():
            logits = model(img)
            probs = torch.sigmoid(logits)[:,1]
            pred = (probs > 0.5).float()
        true = mask[:,1]

        # Metrics
        intersection = (pred * true).sum().item()
        union = ((pred + true) > 0).sum().item()
        iou = intersection / union if union>0 else 1.0
        dice = 2 * intersection / (pred.sum().item() + true.sum().item()) if (pred.sum()+true.sum())>0 else 1.0
        total_iou += iou
        total_dice += dice

        # Overlay
        np_img = img[0].cpu().numpy().transpose(1,2,0)
        # Optionally denormalize: if normalize_output flag
        if args.normalize_output:
            np_img = PREPROCESS_FN(np_img)
        np_img = (np_img * 255).clip(0,255).astype(np.uint8)
        overlay = np_img.copy()
        overlay[pred[0].cpu().bool()] = [255, 0, 0]
        out_path = os.path.join(args.output_dir, f"overlay_{fname}")
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    n = len(loader)
    print(f"Average IoU: {total_iou/n:.4f}")
    print(f"Average Dice: {total_dice/n:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--coco_json', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', default='overlays')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--normalize_output', action='store_true')
    args = parser.parse_args()
    main(args)
