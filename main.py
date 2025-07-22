#!/usr/bin/env python
"""
Main training script for binary segmentation (single channel output).
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
from utils.metrics import iou_score, dice_coefficient, pixel_accuracy
from utils.logger import init_wandb, log_metrics, log_image
from settings.config import METRIC_THRESHOLD


def parse_args():
    parser = argparse.ArgumentParser(description="Train binary segmentation model.")
    # General settings
    parser.add_argument('--seed',         type=int,   default=42,    help='Random seed for reproducibility')
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
   # W&B configuration
    parser.add_argument('--wandb_project', type=str, default='substation-segmentation', help='W&B project name')
    parser.add_argument('--wandb_entity',  type=str, default=None,   help='W&B entity/team name')
    parser.add_argument('--wandb_tags',        nargs='+', default=[],    help='List of W&B tags for this run')
    parser.add_argument('--wandb_group', type=str, default=None, help='W&B group name for grouping runs')

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
    run_name = f"binary_exp_{int(time.time())}"
    init_wandb(
        project_name=args.wandb_project,
        config=vars(args),
        entity=args.wandb_entity,
        run_name=run_name,
        tags=args.wandb_tags + ['binary_segmentation'],
        group=args.wandb_group
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
        print(f"Input shapes → imgs: {imgs.shape}, masks: {masks.shape}")
        print(f"Mask stats → min: {masks.min():.3f}, max: {masks.max():.3f}, mean: {masks.mean():.3f}")
        print(f"Positive pixels: {(masks > 0.5).sum().item()} / {masks.numel()}")
        
        with torch.no_grad(): 
            preds = model(imgs)
            print(f"Model output shape: {preds.shape}")
            print(f"Output range: {preds.min():.3f} to {preds.max():.3f}")
        sys.exit(0)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"--- Training Epoch {epoch:>3} ---")
        model.train()
        total_loss = 0.0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)  # [B, 1, H, W]
            loss   = criterion(logits, masks)  # masks: [B, H, W]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # End-of-epoch evaluation
        model.eval()
        train_metrics = evaluate_model(model, train_loader, device, METRIC_THRESHOLD)
        val_metrics = evaluate_model(model, val_loader, device, METRIC_THRESHOLD)
        
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch} → Train Loss: {avg_train_loss:.4f} | Train IoU: {train_metrics['iou']:.4f} | LR: {current_lr:.6f}")
        print(f"Epoch {epoch} → Val Loss: {val_metrics['loss']:.4f} | Val IoU: {val_metrics['iou']:.4f} | Val Dice: {val_metrics['dice']:.4f}")

        # Log metrics
        log_metrics({
            'train/loss': avg_train_loss,
            'train/iou': train_metrics['iou'],
            'train/dice': train_metrics['dice'],
            'train/pixel_acc': train_metrics['pixel_acc'],
            'val/loss': val_metrics['loss'],
            'val/iou': val_metrics['iou'],
            'val/dice': val_metrics['dice'],
            'val/pixel_acc': val_metrics['pixel_acc'],
            'lr': current_lr
        }, step=epoch)

        # Visual logging of validation examples
        if epoch % 5 == 0:
            log_predictions(model, val_loader, device, epoch, METRIC_THRESHOLD)

        # Checkpoint best model
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            ckpt_name = f"best_model_epoch{epoch}.pth"
            ckpt_path = os.path.join(experiment_dir, ckpt_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
                'config': vars(args)
            }, ckpt_path)
            print(f"Saved new best model: {ckpt_name} (IoU: {best_val_iou:.4f})")

    print(f"Training complete. Best Val IoU: {best_val_iou:.4f}")


def evaluate_model(model, dataloader, device, threshold):
    """Evaluate model on given dataloader."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_pixel_acc = 0.0
    criterion = get_criterion()
    
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)  # [B, 1, H, W]
            
            # Calculate loss
            loss = criterion(logits, masks)
            total_loss += loss.item()
            
            # Convert logits to probabilities
            probs = torch.sigmoid(logits.squeeze(1))  # [B, H, W]
            
            # Calculate metrics
            total_iou += iou_score(probs, masks, threshold)
            total_dice += dice_coefficient(probs, masks, threshold)
            total_pixel_acc += pixel_accuracy(probs, masks, threshold)
    
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'iou': total_iou / num_batches,
        'dice': total_dice / num_batches,
        'pixel_acc': total_pixel_acc / num_batches
    }


def log_predictions(model, dataloader, device, epoch, threshold, num_examples=3):
    """Log prediction examples to wandb."""
    model.eval()
    imgs, masks = next(iter(dataloader))
    imgs, masks = imgs.to(device), masks.to(device)
    
    with torch.no_grad():
        logits = model(imgs)
        probs = torch.sigmoid(logits.squeeze(1))  # [B, H, W]
        preds = (probs > threshold).float()
    
    for idx in range(min(num_examples, imgs.size(0))):
        # Convert to numpy for visualization
        img_np = imgs[idx].cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
        mask_np = masks[idx].cpu().numpy()  # [H, W]
        pred_np = preds[idx].cpu().numpy()  # [H, W]
        
        # Normalize image to [0, 1] for visualization
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        log_image(
            image=[
                wandb.Image(img_np, caption="Original"),
                wandb.Image(mask_np, caption="Ground Truth"),
                wandb.Image(pred_np, caption="Prediction")
            ],
            caption=f"Epoch {epoch} - Example {idx}",
            step=epoch,
            key=f"predictions_epoch_{epoch}_example_{idx}"
        )


if __name__ == '__main__':
    main()
    