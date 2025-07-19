# count_pixels.py
import torch
from torch.utils.data import DataLoader

from helper.dataset_loader import SubstationDataset   # same file you pasted
from settings.config   import TRAIN_IMAGES_DIR, TRAIN_COCO_JSON

ds = SubstationDataset(
    images_dir = TRAIN_IMAGES_DIR,
    coco_json  = TRAIN_COCO_JSON,
    augmentation = None          # raw images, *no* cropping/rotation
)
loader = DataLoader(ds, batch_size=1, num_workers=4)

fg, bg = 0, 0
with torch.no_grad():
    for _, mask in loader:       # current mask shape [1, 2, H, W]
        fg += mask[:, 1].sum().item()   # transformer channel
        bg += mask[:, 0].sum().item()   # background channel

total = fg + bg
print(f"Pixels  FG: {fg:,.0f}  ({fg/total:.4%})")
print(f"Pixels  BG: {bg:,.0f}  ({bg/total:.4%})")
print(f"Suggested pos_weight for BCEWithLogitsLoss: {bg/fg:.2f}")
