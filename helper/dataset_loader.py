import os
import glob
import json
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A

from helper.preprocessing import get_preprocessing_fn
from settings.config import (
    TRAIN_IMAGES_DIR, TRAIN_COCO_JSON,
    VALID_IMAGES_DIR, VALID_COCO_JSON,
    TEST_IMAGES_DIR, TEST_COCO_JSON
)

# Image size constant
IMG_SIZE = 640

# === Custom Augmentation Transforms ===
class RandomCropWithMask(A.DualTransform):
    def __init__(self, height, width, min_mask_frac=0.005, max_tries=5, p=1.0):
        super().__init__(p=p)
        self.height = height
        self.width = width
        self.min_mask_frac = min_mask_frac
        self.max_tries = max_tries

    def apply(self, img, x=0, y=0, **params):
        return img[y:y+self.height, x:x+self.width]

    def apply_to_mask(self, mask, x=0, y=0, **params):
        return mask[y:y+self.height, x:x+self.width]

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params['image'].shape[:2]
        mask = (params['mask'] > 0).astype(np.uint8)
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return {'x': (img_w - self.width)//2, 'y': (img_h - self.height)//2}
        cy, cx = ys.mean(), xs.mean()
        area = self.height * self.width
        for _ in range(self.max_tries):
            x = np.random.randint(0, img_w - self.width)
            y = np.random.randint(0, img_h - self.height)
            patch = mask[y:y+self.height, x:x+self.width]
            if patch.sum() / area >= self.min_mask_frac:
                return {'x': x, 'y': y}
        x0 = int(np.clip(cx - self.width/2, 0, img_w - self.width))
        y0 = int(np.clip(cy - self.height/2, 0, img_h - self.height))
        return {'x': x0, 'y': y0}

    def get_transform_init_args_names(self):
        return ('height','width','min_mask_frac','max_tries')

class MaskAwareDropout(A.DualTransform):
    def __init__(self, max_holes=8, hole_frac=0.05, max_mask_overlap_frac=0.1, max_tries=10, p=0.3):
        super().__init__(p=p)
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

# === Augmentation Pipelines ===
def get_training_augmentation():
    return A.Compose([
        A.Rotate(limit=360, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_REFLECT),
        RandomCropWithMask(height=IMG_SIZE, width=IMG_SIZE, min_mask_frac=0.005, max_tries=5, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.GaussianBlur(blur_limit=7, p=0.2),
        A.GaussNoise(p=0.2),
        MaskAwareDropout(max_holes=8, hole_frac=0.05, max_mask_overlap_frac=0.1, max_tries=10, p=0.3),
    ], additional_targets={'mask': 'mask'})


def get_validation_augmentation():
    return A.Compose([
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_REFLECT, p=1),
        A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE, p=1),
    ], additional_targets={'mask': 'mask'})

# === Dataset Definition ===
class SubstationDataset(Dataset):
    """
    Dataset for transformer segmentation using COCO annotations.
    """
    def __init__(self, images_dir, coco_json, augmentation=None):
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, '*.png'))) + \
                           sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
        with open(coco_json, 'r') as f:
            coco = json.load(f)
        transformer_cat = next(c for c in coco['categories'] if c['name'].lower() == 'transformer')
        tid = transformer_cat['id']
        self.name2id = {img['file_name']: img['id'] for img in coco['images']}
        anns = defaultdict(list)
        for ann in coco['annotations']:
            if ann['category_id'] == tid and isinstance(ann.get('segmentation'), list) and ann['segmentation']:
                anns[ann['image_id']].append(ann['segmentation'])
        self.anns_by_image = anns
        self.augmentation = augmentation
        self.preprocess_fn = get_preprocessing_fn()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        image_id = self.name2id[filename]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for polys in self.anns_by_image[image_id]:
            for poly in polys:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 1)
        fg = mask.astype(np.float32)
        bg = 1.0 - fg
        mask = np.stack([bg, fg], axis=-1)
        # augmentation
        if self.augmentation:
            data = self.augmentation(image=img, mask=mask)
            img, mask = data['image'], data['mask']
        # preprocessing
        img = self.preprocess_fn(img)
        # to tensor CHW
        img = img.astype(np.float32).transpose(2, 0, 1)
        mask = mask.astype(np.float32).transpose(2, 0, 1)
        return torch.from_numpy(img), torch.from_numpy(mask)

# === DataLoader Factory ===
def get_dataloaders(batch_size=8, num_workers=4, pin_memory=True):
    train_ds = SubstationDataset(
        images_dir=TRAIN_IMAGES_DIR,
        coco_json=TRAIN_COCO_JSON,
        augmentation=get_training_augmentation()
    )
    valid_ds = SubstationDataset(
        images_dir=VALID_IMAGES_DIR,
        coco_json=VALID_COCO_JSON,
        augmentation=get_validation_augmentation()
    )
    test_ds = SubstationDataset(
        images_dir=TEST_IMAGES_DIR,
        coco_json=TEST_COCO_JSON,
        augmentation=get_validation_augmentation()
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(valid_ds, batch_size=1, shuffle=False,
                   num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(test_ds, batch_size=1, shuffle=False,
                   num_workers=num_workers, pin_memory=pin_memory)
    )
