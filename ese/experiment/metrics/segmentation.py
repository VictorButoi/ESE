# torch imports
import torch
from torch import Tensor
# misc imports
from pydantic import validate_arguments
from typing import Optional, Union, List
# ionpy imports
from ionpy.metrics.util import (
    _metric_reduction,
    _inputs_as_onehot,
    _inputs_as_longlabels,
    InputMode,
    Reduction,
)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def dice_calculation(
    y_pred: Tensor,
    y_true: Tensor,
    smooth: float = 1e-7,
    eps: float = 1e-7,
) -> Tensor:
    intersection = torch.logical_and(y_pred == 1.0, y_true == 1.0).sum(dim=-1)
    pred_amounts = (y_pred == 1.0).sum(dim=-1)
    true_amounts = (y_true == 1.0).sum(dim=-1)
    cardinalities = pred_amounts + true_amounts
    dice_scores = (2 * intersection + smooth) / (cardinalities + smooth).clamp_min(eps)
    return pred_amounts, true_amounts, dice_scores


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def avg_dice_score(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    smooth: float = 1e-7,
    eps: float = 1e-7,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
) -> Tensor:
    # Convert to one-hot labels
    y_pred, y_true = _inputs_as_onehot(
        y_pred, 
        y_true, 
        mode=mode, 
        from_logits=from_logits, 
        discretize=True
    )
    # Calculate dice score
    _, true_amounts, dice_scores = dice_calculation(
        y_pred, 
        y_true, 
        smooth=smooth, 
        eps=eps
    ) 
    # Get the weights by dividing the true amounts by the total number of pixels.
    weights = true_amounts / true_amounts.sum()
    # Return the metric reduction
    return _metric_reduction(
        dice_scores,
        reduction=reduction,
        weights=weights,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def labelwise_dice_score(
    y_pred: Tensor,
    y_true: Tensor,
    ignore_empty_labels: bool,
    mode: InputMode = "auto",
    smooth: float = 1e-7,
    eps: float = 1e-7,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
) -> Tensor:
    # Convert to one-hot labels
    y_pred, y_true = _inputs_as_onehot(
        y_pred, 
        y_true, 
        mode=mode, 
        from_logits=from_logits, 
        discretize=True
    )
    # Calculate dice score
    _, true_amounts, dice_scores = dice_calculation(
        y_pred, 
        y_true, 
        smooth=smooth, 
        eps=eps
    ) 
    # If ignoring empty labels, make sure to set their weight to 0.
    if ignore_empty_labels:
        label_weights = (true_amounts > 0).float().cpu()
    else:
        label_weights = torch.ones_like(true_amounts).float().cpu()
    # Return the metric reduction
    return _metric_reduction(
        dice_scores,
        reduction=reduction,
        weights=label_weights,
        ignore_empty_labels=ignore_empty_labels,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def avg_pixel_accuracy(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    from_logits: bool = False
):
    # Convert to long labels
    y_pred_long, y_true_long = _inputs_as_longlabels(
        y_pred, 
        y_true, 
        mode, 
        from_logits=from_logits, 
        discretize=True
    )
    return (y_pred_long == y_true_long).float().mean()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def labelwise_pixel_accuracy(
    y_pred: Tensor,
    y_true: Tensor,
    ignore_empty_labels: bool,
    mode: InputMode = "auto",
    from_logits: bool = False,
    ignore_index: Optional[int] = None,
) -> Tensor:
    # Convert to one-hot labels
    y_pred, y_true = _inputs_as_onehot(
        y_pred, 
        y_true, 
        mode=mode, 
        from_logits=from_logits, 
        discretize=True
    )
    labelwise_accuracy = (y_pred == y_true).float().mean(dim=-1)
    # If ignoring empty labels, make sure they aren't include in the
    # final calculation.
    label_indices = torch.arange(labelwise_accuracy.shape[-1])
    if ignore_empty_labels:
        true_amounts = (y_true == 1.0).sum(dim=-1)
        nonempty_label = (true_amounts > 0)
        if ignore_index is not None:
            valid_indices = nonempty_label & (label_indices != ignore_index)
        else:
            valid_indices = nonempty_label
        # Choose the valid indices.
        labelwise_accuracy = labelwise_accuracy[valid_indices] 
    else:
        if ignore_index is not None:
            valid_indices = (label_indices != ignore_index)
            labelwise_accuracy = labelwise_accuracy[valid_indices]
    # Finally, return the mean of the labelwise accuracies.
    return labelwise_accuracy.mean()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def avg_edge_pixel_accuracy(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    from_logits: bool = False
):
    # Convert to long labels
    y_pred, y_true = _inputs_as_longlabels(
        y_pred, 
        y_true, 
        mode, 
        from_logits=from_logits, 
        discretize=True
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def labelwise_edge_pixel_accuracy(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    from_logits: bool = False
):
    # Convert to long labels
    y_pred, y_true = _inputs_as_longlabels(
        y_pred, 
        y_true, 
        mode, 
        from_logits=from_logits, 
        discretize=True
    )