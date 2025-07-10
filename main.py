#!/usr/bin/env python
"""
Main training script: orchestrates the segmentation pipeline via modular helpers.
"""
import os
import sys
import time
import argparse
import torch

from helper.dataset_loader import get_dataloaders
from models.unet import get_model, get_criterion, get_optimizer, get_scheduler
from utils.metrics import iou_score
from utils.logger import init_wandb, log_metrics
from settings.config import METRIC_THRESHOLD



def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation model using config paths and helper modules.")
    # Training hyperparameters
    parser.add_argument('--batch_size',  type=int, default=8)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--epochs',      type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    # Device & logging
    parser.add_argument('--device',      type=str, default='auto', choices=['auto','cpu','cuda'])
    parser.add_argument('--dry_run',     action='store_true', help='Perform a single batch test and exit')
    parser.add_argument('--log_dir',     type=str, default='runs', help='Directory to save checkpoints')
    parser.add_argument('--wandb_project', type=str, default='substation-segmentation')
    parser.add_argument('--wandb_entity',  type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Initialize Weights & Biases
    init_wandb(
        project_name=args.wandb_project,
        config=vars(args),
        entity=args.wandb_entity,
        run_name=f"run-{int(time.time())}"
    )

    # DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Model and training components
    model     = get_model(device)
    criterion = get_criterion()
    optimizer = get_optimizer(model, lr=args.lr)
    scheduler = get_scheduler(optimizer)

    best_val_iou = 0.0
    os.makedirs(args.log_dir, exist_ok=True)

    # Dry run
    if args.dry_run:
        imgs, masks = next(iter(train_loader))
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad():
            preds = model(imgs)
        print(f"Dry run shapes → imgs: {imgs.shape}, masks: {masks.shape}, preds: {preds.shape}")
        sys.exit(0)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print("--- Training Epoch {:>3} ---".format(epoch))
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
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} → Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                # take transformer class channel if multi-class, else single
                if logits.shape[1] > 1:
                    preds = logits[:, 1, :, :]
                else:
                    preds = torch.sigmoid(logits[:, 0, :, :])
                # ground-truth mask: if channel, take channel 1, else 0
                if masks.ndim == 4:
                    truths = masks[:, 1, :, :]
                else:
                    truths = masks
                val_iou += iou_score(preds, truths, threshold=METRIC_THRESHOLD)
        avg_val_iou = val_iou / len(val_loader)
        scheduler.step(avg_val_iou)
        print(f"Epoch {epoch} → Val IoU: {avg_val_iou:.4f} | LR: {scheduler.get_last_lr():.6f}")

        # Log metrics
        log_metrics({'train/loss': avg_train_loss, 'val/iou': avg_val_iou}, step=epoch)

        # Checkpoint
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            ckpt_name = f"best_model_epoch{epoch}.pth"
            ckpt_path = os.path.join(args.log_dir, ckpt_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_iou': best_val_iou
            }, ckpt_path)
            print(f"Saved new best model: {ckpt_name}")

    print(f"Training complete. Best Val IoU: {best_val_iou:.4f}")

if __name__ == '__main__':
    main()
