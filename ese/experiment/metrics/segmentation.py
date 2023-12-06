# local imports
from .utils import get_edge_map
# torch imports
import torch
from torch import Tensor
# misc imports
import matplotlib.pyplot as plt
from pydantic import validate_arguments
from typing import Optional, Union, List
# ionpy imports
from ionpy.metrics.util import (
    _inputs_as_longlabels,
    InputMode,
)


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