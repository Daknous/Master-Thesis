import torch
import segmentation_models_pytorch as smp
from torch import nn, optim
# Import encoder name from preprocessing to keep consistency
from helper.preprocessing import ENCODER


def get_model(device: torch.device):
    """
    Builds the U-Net model exactly as in electrical_substation_segmentation.py.

    Moves the model to the specified device.
    """
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights='imagenet',
        in_channels=3,
        classes=2
    )
    model.to(device)
    return model


def get_criterion() -> nn.Module:
    """
    Composite loss: BCEWithLogitsLoss + DiceLoss (binary mode), matching the original.
    """
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode='binary')

    class CompositeLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.bce = bce_loss
            self.dice = dice_loss

        def forward(self, preds, masks):
            return self.bce(preds, masks) + self.dice(preds, masks)

    return CompositeLoss()


def get_optimizer(model: torch.nn.Module, lr: float):
    """
    Adam optimizer with specified learning rate, as in the original script.
    """
    return optim.Adam(model.parameters(), lr=lr)


def get_scheduler(optimizer: optim.Optimizer):
    """
    ReduceLROnPlateau scheduler (mode='min', factor=0.7, patience=10, verbose=True), matching the original script.
    """
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.7,
        patience=10,
        verbose=True
    )
