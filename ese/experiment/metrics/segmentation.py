# torch imports
import torch
from torch import Tensor
# misc imports
from pydantic import validate_arguments
from typing import Optional, Union, List
# ionpy imports
from ionpy.metrics.util import (
    _metric_reduction,
    Reduction,
)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def brier_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    square_diff: bool = True,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, List]] = None,
    ignore_empty_labels: bool = True,
    from_logits: bool = False,
    ignore_index: Optional[int] = None,
):
    """
    Calculates the Brier Score for a predicted label map.
    """
    y_pred = y_pred.squeeze()
    y_true = y_true.squeeze()
    # If the input is multi-channel for confidence, take the max across channels.
    if from_logits:
        y_pred = torch.softmax(y_pred, dim=0)
    assert len(y_pred.shape) == 3 and len(y_true.shape) == 2,\
        f"y_pred and y_true must be 3D and 2D tensors respectively. Got {y_pred.shape} and {y_true.shape}."
    num_pred_classes = y_pred.shape[0]
    lab_brier_scores = torch.zeros(num_pred_classes, device=y_pred.device)

    # Iterate through each label and calculate the brier score.
    unique_gt_labels = torch.unique(y_true)
    for lab in unique_gt_labels:
        binary_y_true = (y_true == lab).float()
        # Calculate the brier score.
        if square_diff:
            pos_diff_per_pix = (y_pred[lab, ...] - binary_y_true).square()
        else:
            pos_diff_per_pix = (y_pred[lab, ...] - binary_y_true).abs()
        lab_brier_scores[lab] = pos_diff_per_pix.mean()
    
    # Don't include empty labels in the final score.
    if ignore_empty_labels:
        existing_label = torch.zeros(num_pred_classes, device=y_pred.device)
        existing_label[unique_gt_labels] = 1
        if weights is None:
            weights = existing_label
        else:
            weights = weights * existing_label

    # Get the mean across channels (and batch dim).
    brier_loss = _metric_reduction(
        lab_brier_scores[None], # Add dummy batch dim.
        reduction=reduction,
        weights=weights,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )

    # Return the brier score.
    return 1 - brier_loss 