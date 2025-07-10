import torch


def iou_score(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    Computes the Intersection over Union (IoU) score for binary segmentation.

    Args:
        preds: model output logits or probabilities, shape [B, H, W] or [B, ...].
        targets: ground truth binary masks, same shape as preds.
        threshold: threshold to binarize probabilities.
        eps: small epsilon to avoid division by zero.

    Returns:
        Mean IoU over batch.
    """
    # If logits, apply sigmoid
    if preds.dtype.is_floating_point and (preds.max() > 1 or preds.min() < 0):
        preds = torch.sigmoid(preds)

    # Binarize predictions
    preds_bin = (preds > threshold).float()
    targets_bin = (targets > 0.5).float()

    # Flatten
    preds_flat = preds_bin.view(preds_bin.size(0), -1)
    targets_flat = targets_bin.view(targets_bin.size(0), -1)

    # Compute intersection and union per sample
    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection

    # Compute IoU per sample and average
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()
