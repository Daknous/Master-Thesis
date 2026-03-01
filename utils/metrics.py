import torch


def iou_score(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    Computes the Intersection over Union (IoU) score for binary segmentation.

    Args:
        preds: model output probabilities, shape [B, H, W]. Should be in [0, 1] range.
        targets: ground truth binary masks, shape [B, H, W]. Should be in [0, 1] range.
        threshold: threshold to binarize probabilities.
        eps: small epsilon to avoid division by zero.

    Returns:
        Mean IoU over batch.
    """
    # Ensure inputs are on the same device
    preds = preds.float()
    targets = targets.float()
    
    # Binarize predictions using threshold
    preds_bin = (preds > threshold).float()
    targets_bin = (targets > 0.5).float()  # Binarize targets in case they're not exactly 0/1
    
    # Flatten spatial dimensions but keep batch dimension
    preds_flat = preds_bin.view(preds_bin.size(0), -1)  # [B, H*W]
    targets_flat = targets_bin.view(targets_bin.size(0), -1)  # [B, H*W]
    
    # Compute intersection and union per sample in batch
    intersection = (preds_flat * targets_flat).sum(dim=1)  # [B]
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection  # [B]
    
    # Handle edge case where both prediction and target are empty
    # In this case, IoU should be 1 (perfect match)
    perfect_match = (preds_flat.sum(dim=1) == 0) & (targets_flat.sum(dim=1) == 0)
    
    # Compute IoU per sample
    iou = torch.where(
        perfect_match,
        torch.ones_like(intersection),  # IoU = 1 for perfect empty matches
        (intersection + eps) / (union + eps)  # Standard IoU calculation
    )
    
    # Return mean IoU over batch
    return iou.mean().item()


def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Computes pixel-wise accuracy for binary segmentation.
    
    Args:
        preds: model output probabilities, shape [B, H, W]
        targets: ground truth binary masks, shape [B, H, W]
        threshold: threshold to binarize probabilities
        
    Returns:
        Mean pixel accuracy over batch
    """
    preds_bin = (preds > threshold).float()
    targets_bin = (targets > 0.5).float()
    
    correct = (preds_bin == targets_bin).float()
    accuracy = correct.mean()
    
    return accuracy.item()


def dice_coefficient(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    Computes Dice coefficient for binary segmentation.
    
    Args:
        preds: model output probabilities, shape [B, H, W]
        targets: ground truth binary masks, shape [B, H, W]
        threshold: threshold to binarize probabilities
        eps: small epsilon to avoid division by zero
        
    Returns:
        Mean Dice coefficient over batch
    """
    preds_bin = (preds > threshold).float()
    targets_bin = (targets > 0.5).float()
    
    # Flatten spatial dimensions but keep batch dimension
    preds_flat = preds_bin.view(preds_bin.size(0), -1)
    targets_flat = targets_bin.view(targets_bin.size(0), -1)
    
    intersection = (preds_flat * targets_flat).sum(dim=1)
    dice = (2.0 * intersection + eps) / (preds_flat.sum(dim=1) + targets_flat.sum(dim=1) + eps)
    
    return dice.mean().item()