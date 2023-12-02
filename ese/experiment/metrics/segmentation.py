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
    weights: Optional[Union[Tensor, List]] = None,
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
    # If weights are defined, modulate the proportions by the weights.
    if weights is not None:
        true_amounts = true_amounts * weights
    # Get the weights by dividing the true amounts by the total number of pixels.
    label_weights = true_amounts / true_amounts.sum()
    # Return the metric reduction
    return _metric_reduction(
        dice_scores,
        reduction=reduction,
        weights=label_weights,
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
    weights: Optional[Union[Tensor, List]] = None,
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
        label_weights = (true_amounts > 0).float()
    else:
        label_weights = torch.ones_like(true_amounts).float()

    # If weights are defined, modulate the proportions by the weights.
    if weights is not None:
        label_weights = label_weights * weights

    # Return the metric reduction
    return _metric_reduction(
        dice_scores,
        reduction=reduction,
        weights=label_weights,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def avg_pixel_accuracy(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    from_logits: bool = False,
    ignore_index: Optional[int] = None
):
    # Convert to long labels
    y_pred_long, y_true_long = _inputs_as_longlabels(
        y_pred, 
        y_true, 
        mode, 
        from_logits=from_logits, 
        discretize=True
    )
    # Potentially ignore pixels with a certain label
    if ignore_index is not None:
        valid_mask = (y_true_long != ignore_index)
    else:
        valid_mask = torch.ones_like(y_true_long).bool()
    # Return accuracy of valid pixels.
    return (y_pred_long[valid_mask] == y_true_long[valid_mask]).float().mean()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def labelwise_pixel_accuracy(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    from_logits: bool = False,
    weights: Optional[Union[Tensor, List]] = None,
    ignore_index: Optional[int] = None,
) -> Tensor:
    # Convert to onehot_long labels
    y_pred, y_true = _inputs_as_longlabels(
        y_pred, 
        y_true, 
        mode, 
        from_logits=from_logits, 
        discretize=True
    )
    # Get unique labels in y_true
    unique_labels = torch.unique(y_true).tolist()
    # Remove labels to be ignored
    unique_labels = [label for label in unique_labels if not(ignore_index is not None and label == ignore_index)]
    accuracies = []
    for label in unique_labels:
        # Create a mask for the current label
        label_mask = (y_true == label).bool()
        # Extract predictions and ground truth for pixels belonging to the current label
        label_pred = y_pred[label_mask]
        label_true = y_true[label_mask]
        # Calculate accuracy for the current label
        label_accuracy = (label_pred == label_true).float().mean()
        accuracies.append(label_accuracy)
    acc_tensor = torch.tensor(accuracies, device=weights.device)
    # If weights are defined, modulate the proportions by the weights.
    if weights is not None:
        acc_tensor = acc_tensor * weights
        return acc_tensor.sum()
    else:
        # Calculate balanced accuracy by averaging individual label accuracies
        return acc_tensor.mean()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def avg_edge_pixel_accuracy(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    from_logits: bool = False,
    ignore_index: Optional[int] = None
) -> Tensor:
    # Get the edge map.
    y_true_squeezed = y_true.squeeze()
    edge_map = get_edge_map(y_true_squeezed)
    # Get the edge regions of both the prediction and the ground truth.
    y_pred_e_reg = y_pred[..., edge_map]
    y_true_e_reg = y_true[..., edge_map]
    # Return the mean of the accuracy.
    return avg_pixel_accuracy(
        y_pred_e_reg, 
        y_true_e_reg,
        mode=mode,
        from_logits=from_logits,
        ignore_index=ignore_index
        )

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def labelwise_edge_pixel_accuracy(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    from_logits: bool = False,
    weights: Optional[Union[Tensor, List]] = None,
    ignore_index: Optional[int] = None
) -> Tensor:
    # Get the edge map.
    y_true_squeezed = y_true.squeeze()
    edge_map = get_edge_map(y_true_squeezed)
    # Get the edge regions of both the prediction and the ground truth.
    y_pred_e_reg = y_pred[..., edge_map]
    y_true_e_reg = y_true[..., edge_map]
    # Return the mean of the accuracy.
    return labelwise_pixel_accuracy(
        y_pred_e_reg, 
        y_true_e_reg, 
        mode=mode,
        from_logits=from_logits,
        weights=weights,
        ignore_index=ignore_index
        )