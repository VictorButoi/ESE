# local imports
from .utils import get_edge_map
# torch imports
import torch
from torch import Tensor
# misc imports
from pydantic import validate_arguments
from typing import Optional, Union, List
# ionpy imports
from ionpy.metrics.util import (
    _inputs_as_longlabels,
    InputMode,
    _metric_reduction,
    Reduction,
)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def brier_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    square_diff: bool,
    ignore_empty_labels: bool = True,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, List]] = None,
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
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


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def labelwise_pixel_accuracy(
    y_pred: Tensor,
    y_true: Tensor,
    ignore_empty_labels: bool,
    mode: InputMode = "auto",
    from_logits: bool = False,
    weights: Optional[Union[Tensor, List]] = None,
    ignore_index: Optional[int] = None,
) -> Tensor:
    assert y_pred.shape[0] == y_true.shape[0] == 1, "y_pred and y_true must have the same batch size (1)."
    # Convert to onehot_long labels
    num_pred_classes = y_pred.shape[1]
    y_pred, y_true = _inputs_as_longlabels(
        y_pred, 
        y_true, 
        mode, 
        from_logits=from_logits, 
        discretize=True
    )
    # Get unique labels in y_true
    unique_labels = torch.unique(y_true)
    # Keep track of the accuracies per label.
    accuracies = torch.zeros(num_pred_classes, device=y_pred.device) 
    for label in unique_labels:
        # Create a mask for the current label
        label_mask = (y_true == label)
        label_pred = (y_pred == label)
        accuracies[label] = (label_pred == label_mask).float().mean()

    # If ignoring empty labels, make sure to set their weight to 0.
    if weights is None:
        if ignore_empty_labels:
            lab_present = torch.zeros(num_pred_classes, device=y_pred.device)
            lab_present[unique_labels] = 1 
            weights = lab_present
        else:
            weights = torch.ones(num_pred_classes, device=y_pred.device)
    # If ignore_index is not None, set the weight of the ignore_index to 0.
    if ignore_index is not None:
        weights[ignore_index] = 0
    # Renormalize the weights to sum to 1.
    weights = weights / weights.sum()
    return (accuracies * weights).sum()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def weighted_pixel_accuracy(
    y_pred: Tensor,
    y_true: Tensor,
    ignore_empty_labels: bool,
    mode: InputMode = "auto",
    from_logits: bool = False,
    weights: Optional[Union[Tensor, List]] = None,
    ignore_index: Optional[int] = None,
) -> Tensor:
    assert y_pred.shape[0] == y_true.shape[0] == 1, "y_pred and y_true must have the same batch size (1)."
    true_amount_per_label = torch.bincount(y_true.flatten())
    # Calculate each label's weight as the inverse amount.
    weights = 1 / true_amount_per_label
    # Return the labelwise pixel accuracy with the calculated weights.
    return labelwise_pixel_accuracy(
        y_pred, 
        y_true, 
        ignore_empty_labels=ignore_empty_labels,
        mode=mode,
        from_logits=from_logits,
        weights=weights,
        ignore_index=ignore_index
        )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def labelwise_edge_pixel_accuracy(
    y_pred: Tensor,
    y_true: Tensor,
    ignore_empty_labels: bool,
    mode: InputMode = "auto",
    from_logits: bool = False,
    weights: Optional[Union[Tensor, List]] = None,
    ignore_index: Optional[int] = None
) -> Tensor:
    assert y_pred.shape[0] == y_true.shape[0] == 1, "y_pred and y_true must have the same batch size (1)."
    # Get the edge map.
    y_true_squeezed = y_true.squeeze()
    y_true_edge_map = get_edge_map(y_true_squeezed)
    # Get the edge regions of both the prediction and the ground truth.
    y_pred_e_reg = y_pred[..., y_true_edge_map]
    y_true_e_reg = y_true[..., y_true_edge_map]
    # Return the mean of the accuracy.
    return labelwise_pixel_accuracy(
        y_pred_e_reg, 
        y_true_e_reg, 
        ignore_empty_labels=ignore_empty_labels,
        mode=mode,
        from_logits=from_logits,
        weights=weights,
        ignore_index=ignore_index
        )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def weighted_edge_pixel_accuracy(
    y_pred: Tensor,
    y_true: Tensor,
    ignore_empty_labels: bool,
    mode: InputMode = "auto",
    from_logits: bool = False,
    ignore_index: Optional[int] = None
) -> Tensor:
    assert y_pred.shape[0] == y_true.shape[0] == 1, "y_pred and y_true must have the same batch size (1)."
    true_amount_per_label = torch.bincount(y_true.flatten())
    # Calculate each label's weight as the inverse amount.
    weights = 1 / true_amount_per_label
    # Return the labelwise pixel accuracy with the calculated weights.
    return labelwise_edge_pixel_accuracy(
        y_pred, 
        y_true, 
        ignore_empty_labels=ignore_empty_labels,
        mode=mode,
        from_logits=from_logits,
        weights=weights,
        ignore_index=ignore_index
        )