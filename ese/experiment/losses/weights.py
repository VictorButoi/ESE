import torch
import numpy as np
from scipy import ndimage
from typing import Optional

# This file describes unique per loss weights that will be used for a few purposes:
# - Weighted loss functions
# - Weighted Brier Scores
# - Weighted Calibration Scores


def get_pixel_weights(
    y_true: torch.Tensor,
    loss_func: Optional[str] = None,
):
    if loss_func is None:
        return accuracy_weights(y_true)
    elif loss_func == "dice":
        return dice_weights(y_true)
    elif loss_func == "hausdorff":
        return hausdorff_weights(y_true)
    else:
        raise ValueError(f"Loss function {loss_func} not supported for pixel weights.")


def accuracy_weights(
    y_true
):
    """
    This function returns a tensor that is the same shape as
    y_true, which are all ones. This is because the accuracy
    score does not require any weights.
    """
    assert len(y_true.shape) == 3, "Inputs mut be (B, H, W)"
    ones_map = torch.ones_like(y_true)
    # Normalize by the number of samples
    return ones_map / ones_map.sum()


def dice_weights(
    y_true: torch.Tensor
):
    """
    This function returns a tensor that is the same shape as
    y_true, which each class is replaced by the inverse of the
    class frequency in the dataset. This is because the dice
    score is sensitive to class imbalance. This has to be done
    per item of the batch.

    args:
        y_true: torch.Tensor: The true labels, shape (B, H, W)

    returns:
        torch.Tensor: The weights for each class, shape (B, H, W)
    """
    assert len(y_true.shape) == 3, "Inputs mut be (B, H, W)"
    B, H, W = y_true.shape
    unique_classes = torch.unique(y_true)
    assert unique_classes.size(0) <= 2, "Weights currently only support binary segmentation"
    # Initialize the weights tensor, which is the same shape as y_true
    weights = torch.zeros_like(y_true)
    for c in unique_classes:
        # Get per item in the batch how much of the class is present
        class_freq = (y_true == c).sum(dim=(1, 2)) / (H * W)
        # Invert the class frequency
        class_freq = 1 / class_freq
        # Set the positions of the class to the class frequency per batch item.
        for batch_idx in range(B):
            weights[batch_idx][y_true[batch_idx] == c] = class_freq[batch_idx]
    return weights


def hausdorff_weights(
    y_true: torch.Tensor,
    distance_map: Optional[torch.Tensor] = None
):
    """
    This function returns a tensor that is the same shape as
    y_true, where pixels are replaced with their euclidean distance to the
    foreground class. This is because the Hausdorff distance is sensitive
    to the distance of the foreground class to the background class.

    args:
        y_true: torch.Tensor: The true labels, shape (B, H, W)
        distance_map: Optional[torch.Tensor]: The distance map, shape (B, H, W)

    returns:
        torch.Tensor: The weights for each class, shape (B, H, W)
    """
    assert len(y_true.shape) == 3, "Inputs mut be (B, H, W)"
    unique_classes = torch.unique(y_true)
    assert unique_classes.size(0) <= 2, "Weights currently only support binary segmentation"

    # The weights are going to be the normalized distance map
    if distance_map is None:
        # Calculate the distance transform.
        y_true_np = y_true.cpu().numpy()
        distance_map = np.zeros_like(y_true_np)
        for batch_idx in range(y_true_np.shape[0]):
            dist_to_boundary = ndimage.distance_transform_edt(y_true_np[batch_idx])
            background_dist_to_boundary = ndimage.distance_transform_edt(1 - y_true_np[batch_idx])
            distance_map[batch_idx] = (dist_to_boundary + background_dist_to_boundary)/2
        # Send to the same device as y_true
        distance_map = torch.from_numpy(distance_map).to(y_true.device)

    # Normalize the distance map (per item in the batch)
    weights = distance_map / distance_map.sum(dim=(1, 2))[..., None, None]

    return weights
    