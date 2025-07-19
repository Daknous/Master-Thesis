import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import FocalLoss



def get_model(
    device: torch.device,
    encoder_name: str,
    encoder_weights: str = 'imagenet',
    in_channels: int = 3,
    classes: int = 2,
    dropout: float = 0.0
) -> nn.Module:
    """
    Returns an SMP U-Net model with optional Dropout2d applied
    before the final segmentation head.

    Args:
        device: torch device to move the model to.
        encoder_name: name of backbone encoder (e.g. 'resnet34').
        encoder_weights: pretrained weights (e.g. 'imagenet' or None).
        in_channels: number of input channels (e.g. 3 for RGB).
        classes: number of output channels (e.g. 2 for background + mask).
        dropout: dropout probability to apply before segmentation head.
    """
    # Instantiate base U-Net
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None
    )

    # Inject Dropout2d before the final head if requested
    if dropout and dropout > 0.0:
        orig_head = model.segmentation_head
        dropout_layer = nn.Dropout2d(p=dropout)
        model.segmentation_head = nn.Sequential(dropout_layer, orig_head)

    # Move to device
    return model.to(device)


def get_criterion() -> nn.Module:
    """
    Composite loss combining BCEWithLogits, Dice, and Focal loss.
    """
    bce   = nn.BCEWithLogitsLoss()
    dice  = smp.losses.DiceLoss(mode='binary')
    focal = FocalLoss(mode='binary', alpha=0.5, gamma=2.0)

    class CompositeLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.bce   = bce
            self.dice  = dice
            self.focal = focal

        def forward(self, preds, targets):
            loss_bce   = self.bce(preds, targets)
            loss_dice  = self.dice(preds, targets)
            loss_focal = self.focal(preds, targets)
            return loss_bce + loss_dice + loss_focal

    return CompositeLoss()


def get_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    """
    Returns an AdamW optimizer with a small weight decay.
    """
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
