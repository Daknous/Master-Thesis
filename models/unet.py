import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# UNet for binary segmentation with single channel output
class UnetWithDecoderDropout(smp.Unet):
    def __init__(
        self,
        encoder_name: str,
        encoder_weights: str,
        in_channels: int,
        classes: int,
        decoder_attention_type: str = "none",
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            decoder_attention_type=None if decoder_attention_type=="none" else decoder_attention_type,
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

def get_model(device: torch.device, dropout: float = 0.0, decoder_attention: str = "none") -> nn.Module:
    """
    Create binary segmentation model with single channel output.
    """
    model = UnetWithDecoderDropout(
        encoder_name=ENCODER,
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,  # Single channel for binary segmentation
        dropout=dropout,
        decoder_attention_type=decoder_attention
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

class FocalTverskyLoss(nn.Module):
    """
    Focal-Tversky for one-channel logits/targets.
    Default alpha=0.7, beta=0.3 as in the paper; gamma=0.75 to add focal behaviour.
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits  = logits.squeeze(1)                 # [B,H,W]
        probs   = torch.sigmoid(logits)

        # flatten to (B, -1) for simplicity
        probs   = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1 - targets)).sum(dim=1)
        fn = ((1 - probs) * targets).sum(dim=1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth)

        focal_tversky = (1 - tversky) ** self.gamma
        return focal_tversky.mean()



def get_criterion(loss_name: str = "bce_dice", pos_weight: float = None) -> nn.Module:
    """
    Returns a BinarySegmentationLoss with optional pos_weight or FocalTverskyLoss.
    """
    if loss_name == "bce_dice":
        return BinarySegmentationLoss(
            bce_weight=1.0,
            dice_weight=1.0,
            pos_weight=pos_weight
        )
    elif loss_name == "focal_tversky":
        return FocalTverskyLoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


def get_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
