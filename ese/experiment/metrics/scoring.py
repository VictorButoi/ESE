# torch imports
import torch
# misc imports
from pydantic import validate_arguments
from typing import Optional
# ionpy imports
from ionpy.metrics.util import (
    Reduction,
)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def brier_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    square_diff: bool = True,
    from_logits: bool = False,
    batch_reduction: Reduction = "mean",
    ignore_empty_labels: bool = False,
    ignore_index: Optional[int] = None,
):
    """
    Calculates the Brier Score for a predicted label map.
    """
    assert len(y_pred.shape) == 4 and len(y_true.shape) == 4,\
        "y_pred and y_true must be 4D tensors."
    # If the input is multi-channel for confidence, take the max across channels.
    if from_logits:
        y_pred = torch.softmax(y_pred, dim=1)
    B, C = y_pred.shape[:2]
    # Iterate through each label and calculate the brier score.
    unique_gt_labels = torch.unique(y_true)
    brier_map = torch.zeros_like(y_true).float()
    # Iterate through the possible label classes.
    for lab in range(C):
        if ignore_index is None or lab != ignore_index:
            if not ignore_empty_labels or lab in unique_gt_labels:
                binary_y_true = (y_true == lab).float()
                binary_y_pred = y_pred[:, lab:lab+1, ...]
                # Calculate the brier score.
                if square_diff:
                    pos_diff_per_pix = (binary_y_pred - binary_y_true).square()
                else:
                    pos_diff_per_pix = (binary_y_pred - binary_y_true).abs()
                # Sum across pixels.
                brier_map += pos_diff_per_pix 
    # Convert from loss to a score.
    brier_score_map = 1 - brier_map 
    # Reduce over the non-batch dimensions.
    brier_score = brier_score_map.mean(dim=(1, 2, 3))

    # Return the brier score.
    if batch_reduction == "mean":
        return brier_score.mean()
    else:
        return brier_score


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def cw_brier_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    square_diff: bool = True,
    from_logits: bool = False,
    batch_reduction: Reduction = "mean",
    ignore_index: Optional[int] = None,
):
    """
    Calculates the Brier Score for a predicted label map.
    """
    assert len(y_pred.shape) == 4 and len(y_true.shape) == 4,\
        "y_pred and y_true must be 4D tensors."
    # If the input is multi-channel for confidence, take the max across channels.
    if from_logits:
        y_pred = torch.softmax(y_pred, dim=1)
    B, C = y_pred.shape[:2]
    unique_gt_labels = torch.unique(y_true)

    # Determine the class weights as the inverse of the non-zero class frequencies
    # in the ground truth.
    balanced_class_weights = torch.zeros(B, C, device=y_pred.device)
    for lab in unique_gt_labels:
        balanced_class_weights[:, lab] = 1 / (y_true == lab).sum(dim=(1, 2, 3))
    # Normalize the class weights per batch item
    balanced_class_weights = balanced_class_weights / balanced_class_weights.sum(dim=1, keepdim=True)

    # Iterate through each label and calculate the brier score.
    brier_map = torch.zeros_like(y_true).float()
    # Iterate through the possible label classes.
    for lab in range(C):
        if (ignore_index is None or lab != ignore_index) and lab in unique_gt_labels:
            binary_y_true = (y_true == lab).float()
            binary_y_pred = y_pred[:, lab:lab+1, ...]
            # Calculate the brier score.
            if square_diff:
                pos_diff_per_pix = (binary_y_pred - binary_y_true).square()
            else:
                pos_diff_per_pix = (binary_y_pred - binary_y_true).abs()
            # Sum across pixels.
            brier_map += balanced_class_weights[:, lab] * pos_diff_per_pix 

    # Convert from loss to a score.
    brier_score_map = 1 - brier_map 
    # Reduce over the non-batch dimensions.
    brier_score = brier_score_map.mean(dim=(1, 2, 3))

    # Return the brier score.
    if batch_reduction == "mean":
        return brier_score.mean()
    else:
        return brier_score