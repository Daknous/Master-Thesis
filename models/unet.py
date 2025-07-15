import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# A quick sanity-check UNet wrapper: applies final-layer dropout
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
from settings.config import INPUT_CHANNELS, NUM_CLASSES

def get_model(device: torch.device, dropout: float = 0.0) -> nn.Module:
    model = UnetWithDecoderDropout(
        encoder_name=ENCODER,
        encoder_weights='imagenet',
        in_channels=INPUT_CHANNELS,
        classes=NUM_CLASSES,
        dropout=dropout
    )
    return model.to(device)


def get_criterion() -> nn.Module:
    # Composite loss: BCEWithLogits + Dice
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode='binary')
    class CompositeLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.bce = bce_loss
            self.dice = dice_loss
        def forward(self, preds, targets):
            return self.bce(preds, targets) + self.dice(preds, targets)
    return CompositeLoss()


def get_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    # Use AdamW with a small weight decay
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
