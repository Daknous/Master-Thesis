import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# UNet for binary segmentation with single channel output
class UnetWithDecoderDropout(smp.Unet):
    def __init__(
        self,
        encoder_name: str,
        encoder_weights: str,
        in_channels: int,
        classes: int,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            **kwargs
        )
        # Dropout applied to the final mask logits
        self.dropout_final = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run the standard SMP UNet forward (encoder -> decoder -> segmentation_head)
        masks = super().forward(x)
        # Apply dropout to the final logits
        return self.dropout_final(masks)

# Factory functions for main.py
from helper.preprocessing import ENCODER

def get_model(device: torch.device, dropout: float = 0.0) -> nn.Module:
    """
    Create binary segmentation model with single channel output.
    """
    model = UnetWithDecoderDropout(
        encoder_name=ENCODER,
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,  # Single channel for binary segmentation
        dropout=dropout
    )
    return model.to(device)


class BinarySegmentationLoss(nn.Module):
    """
    Combined loss for binary segmentation.
    Uses BCE + Dice loss for single channel output.
    """
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: float = None
    ):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight

        # If pos_weight is provided, use it in BCEWithLogitsLoss
        if pos_weight is not None:
            pw = torch.tensor([pos_weight], dtype=torch.float)
            self.bce_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self.bce_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, 1, H, W] - raw logits from model
            targets: [B, H, W] - binary targets (0 for background, 1 for foreground)
        """
        # Squeeze logits to match target dimensions
        logits = logits.squeeze(1)  # [B, H, W]

        # BCE Loss (with optional pos_weight)
        bce_loss = self.bce_fn(logits, targets)

        # Dice Loss
        probs = torch.sigmoid(logits)
        smooth = 1e-6
        intersection = (probs * targets).sum()
        dice_coeff = (2.0 * intersection + smooth) / (
            probs.sum() + targets.sum() + smooth
        )
        dice_loss = 1.0 - dice_coeff

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def get_criterion(pos_weight: float = None) -> nn.Module:
    """
    Returns a BinarySegmentationLoss with optional pos_weight.
    """
    return BinarySegmentationLoss(
        bce_weight=1.0,
        dice_weight=1.0,
        pos_weight=pos_weight
    )


def get_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
