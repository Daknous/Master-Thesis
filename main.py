#!/usr/bin/env python
"""
Main training script: orchestrates the segmentation pipeline via modular helpers.
"""
import os
import sys
import time
import argparse
import random
import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR

import wandb
from helper.dataset_loader import get_dataloaders
from models.unet import get_model, get_criterion, get_optimizer
from utils.metrics import iou_score
from utils.logger import init_wandb, log_metrics, log_image
from settings.config import METRIC_THRESHOLD


def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation model using config paths and helper modules.")
    # Training hyperparameters
    parser.add_argument('--batch_size',  type=int,   default=8,     help='Mini-batch size')
    parser.add_argument('--lr',          type=float, default=1e-4,  help='Initial learning rate')
    parser.add_argument('--epochs',      type=int,   default=50,    help='Maximum number of epochs')
    parser.add_argument('--num_workers', type=int,   default=4,     help='Number of dataloader workers')
    parser.add_argument('--dropout',     type=float, default=0.2,   help='Dropout probability in UNet decoder')
    # Device & logging
    parser.add_argument('--device',      type=str,   default='auto', choices=['auto','cpu','cuda'], help='Compute device')
    parser.add_argument('--dry_run',     action='store_true',        help='Perform a single batch test and exit')
    parser.add_argument('--log_dir',     type=str,   default='runs',  help='Directory to save experiment outputs')
    parser.add_argument('--wandb_project', type=str, default='substation-segmentation', help='W&B project name')
    parser.add_argument('--wandb_entity',  type=str, default=None,   help='W&B entity/team name')
    parser.add_argument('--seed',         type=int,   default=42,    help='Random seed for reproducibility')
    # W&B tags
    parser.add_argument('--tags',        nargs='+', default=[],    help='List of W&B tags for this run')
    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seeds for deterministic behavior
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # Device setup
    device = torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else args.device)
    print(f"Using device: {device}")
    print(f"Dropout probability: {args.dropout}")

    # Initialize W&B with tags
    run_name = f"exp.{int(time.time())}"
    init_wandb(
        project_name=args.wandb_project,
        config=vars(args),
        entity=args.wandb_entity,
        run_name=run_name,
        tags=args.tags
    )

    # Experiment directory under runs/
    exp_name = wandb.run.name
    experiment_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Saving checkpoints to: {experiment_dir}")

    # DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Model, criterion, optimizer, scheduler
    model     = get_model(device, dropout=args.dropout)
    criterion = get_criterion()
    optimizer = get_optimizer(model, lr=args.lr)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0
    )

    best_val_iou = 0.0

    # Dry-run check
    if args.dry_run:
        imgs, masks = next(iter(train_loader))
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad(): preds = model(imgs)
        print(f"Dry run shapes → imgs: {imgs.shape}, masks: {masks.shape}, preds: {preds.shape}")
        sys.exit(0)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"--- Training Epoch {epoch:>3} ---")
        model.train()
        total_loss = 0.0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss   = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # End-of-epoch Train IoU
        model.eval()
        train_iou_accum = 0.0
        with torch.no_grad():
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                probs  = torch.softmax(logits, dim=1)[:,1] if logits.shape[1]>1 else torch.sigmoid(logits[:,0])
                targets= masks[:,1] if masks.ndim==4 else masks
                train_iou_accum += iou_score(probs, targets, threshold=METRIC_THRESHOLD)
        avg_train_iou = train_iou_accum / len(train_loader)
        current_lr    = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch} → Train Loss: {avg_train_loss:.4f} | Train IoU: {avg_train_iou:.4f} | LR: {current_lr:.6f}")

        # Validation
        val_loss = 0.0
        val_iou  = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, masks).item()
                preds     = torch.softmax(logits, dim=1)[:,1] if logits.shape[1]>1 else torch.sigmoid(logits[:,0])
                truths    = masks[:,1] if masks.ndim==4 else masks
                val_iou  += iou_score(preds, truths, threshold=METRIC_THRESHOLD)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou  = val_iou  / len(val_loader)
        print(f"Epoch {epoch} → Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

        # Log metrics
        log_metrics(
            {
                'train/loss': avg_train_loss,
                'train/iou':  avg_train_iou,
                'val/loss':   avg_val_loss,
                'val/iou':    avg_val_iou
            },
            step=epoch
        )

        # Visual logging of validation examples (only after 10 epochs)
        if epoch > 10:
            imgs_vis, masks_vis = next(iter(val_loader))
            imgs_vis, masks_vis = imgs_vis.to(device), masks_vis.to(device)
            with torch.no_grad():
                logits_vis = model(imgs_vis)
                preds_vis  = torch.softmax(logits_vis, dim=1)[:,1] if logits_vis.shape[1]>1 else torch.sigmoid(logits_vis[:,0])
            truths_vis = masks_vis[:,1] if masks_vis.ndim==4 else masks_vis
            for idx in range(min(3, imgs_vis.size(0))):
                # Log original image, ground truth, and prediction mask
                overlay = torch.stack([
                    imgs_vis[idx].cpu(),
                    truths_vis[idx].unsqueeze(0).cpu(),
                    (preds_vis[idx] > METRIC_THRESHOLD).float().unsqueeze(0).cpu()
                ], dim=0)
                log_image(
                    image=overlay,
                    caption=f"GT vs Pred (idx={idx})",
                    step=epoch,
                    key=f"example_{idx}"
                )

        # Checkpoint best model
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            ckpt_name    = f"best_model_epoch{epoch}.pth"
            ckpt_path    = os.path.join(experiment_dir, ckpt_name)
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_iou': best_val_iou
                },
                ckpt_path
            )
            print(f"Saved new best model: {ckpt_name} to {experiment_dir}")

    print(f"Training complete. Best Val IoU: {best_val_iou:.4f}")


if __name__ == '__main__':
    main()
