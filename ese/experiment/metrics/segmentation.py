# torch imports
import torch
from torch import Tensor
from torch.nn import functional as F
# misc imports
from pydantic import validate_arguments
from typing import Optional, Union, List
from medpy.metric.binary import hd95 as HausdorffDist95
# local imports
from ionpy.metrics.util import (
    _metric_reduction,
    _inputs_as_onehot,
    InputMode,
    Reduction
)
from ionpy.loss.util import _loss_module_from_func


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def pixel_neighbor_uniform_w_cross_entropy(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    ignore_index: Optional[int] = -100,
    from_logits: bool = False,
) -> Tensor:
    assert len(y_pred.shape) > 2, "y_pred must have at least 3 dimensions."
    batch_size, num_classes = y_pred.shape[:2]
    y_true = y_true.long()

    if mode == "auto":
        if y_pred.shape == y_true.shape:
            mode = "binary" if num_classes == 1 else "onehot"
        else:
            mode = "multiclass"

    if mode == "binary":
        assert y_pred.shape == y_true.shape
        assert ignore_index is None
        if from_logits:
            loss = F.binary_cross_entropy_with_logits(
                y_pred, 
                y_true, 
                reduction="none"
                )
        else:
            loss = F.binary_cross_entropy(
                y_pred, 
                y_true, 
                reduction="none"
                )
        loss = loss.squeeze(dim=1)
    else:
        # Squeeze the label, (no need for channel dimension).
        if len(y_true.shape) == len(y_pred.shape):
            y_true = y_true.squeeze(1)

        if from_logits:
            loss = F.cross_entropy(
                y_pred,
                y_true,
                reduction="none",
                ignore_index=ignore_index,
            )
        else:
            loss = F.nll_loss(
                y_pred,
                y_true,
                reduction="none",
                ignore_index=ignore_index,
            )

    # Channels have been collapsed
    spatial_dims = list(range(1, len(y_pred.shape) - 1))
    if reduction == "mean":
        loss = loss.mean(dim=spatial_dims)
    if reduction == "sum":
        loss = loss.sum(dim=spatial_dims)

    if batch_reduction == "mean":
        loss = loss.mean(dim=0)
    if batch_reduction == "sum":
        loss = loss.sum(dim=0)

    return loss


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def dice_score(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    smooth: float = 1e-7,
    eps: float = 1e-7,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, List]] = None,
    ignore_empty_labels: bool = True,
    from_logits: bool = False,
    ignore_index: Optional[int] = None,
) -> Tensor:

    y_pred, y_true = _inputs_as_onehot(
        y_pred, 
        y_true, 
        mode=mode, 
        from_logits=from_logits, 
        discretize=True
    )

    intersection = torch.logical_and(y_pred == 1.0, y_true == 1.0).sum(dim=-1)
    pred_amounts = (y_pred == 1.0).sum(dim=-1)
    true_amounts = (y_true == 1.0).sum(dim=-1)
    cardinalities = pred_amounts + true_amounts

    dice_scores = (2 * intersection + smooth) / (cardinalities + smooth).clamp_min(eps)
    
    if ignore_empty_labels:
        existing_label = (true_amounts > 0).float()
        if weights is None:
            weights = existing_label
        else:
            weights = weights * existing_label
        
    return _metric_reduction(
        dice_scores,
        reduction=reduction,
        weights=weights,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def hd95(
    y_pred: Tensor,
    y_true: Tensor,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, List]] = None,
    ignore_empty_labels: bool = False,
    from_logits: bool = False,
    ignore_index: Optional[int] = None
):
    """
    Calculates the 95th percentile Hausdorff Distance for a predicted label map. 
    """
    assert len(y_pred.shape) == 4 and len(y_true.shape) == 4,\
        "y_pred and y_true must be 4D tensors."

    B, C = y_pred.shape[:2]
    if from_logits:
        y_pred = torch.softmax(y_pred, dim=1) # Label channels are 1 by default.
    
    # Get the preds with highest probs and the label map.
    y_pred = y_pred.argmax(dim=1)
    y_true = y_true.squeeze(dim=1)

    # Convert these to one hot tensors.
    y_pred_one_hot = F.one_hot(y_pred, num_classes=C).permute(0, 3, 1, 2)
    y_true_one_hot = F.one_hot(y_true, num_classes=C).permute(0, 3, 1, 2)

    # Unfortunately we have to convert these to numpy arrays to work with the medpy func.
    y_pred_one_hot = y_pred_one_hot.cpu().numpy()
    y_true_one_hot = y_true_one_hot.cpu().numpy()

    # Iterate through the labels, and set the batch scores corresponding to that label.
    hd_scores = torch.zeros(B, C) 
    for batch_idx in range(B):
        for lab_idx in range(C):
            label_pred = y_pred_one_hot[batch_idx, lab_idx, :, :]
            label_gt = y_true_one_hot[batch_idx, lab_idx, :, :]
            # If they both have pixels, calculate the hausdorff distance.
            if label_pred.sum() > 0 and label_gt.sum() > 0:
                hd_scores[batch_idx, lab_idx] = HausdorffDist95(
                    result=label_pred,
                    reference=label_gt
                    )
            # If neither have pixels, set the score to 0.
            elif label_pred.sum() == 0 and label_gt.sum() == 0:
                hd_scores[batch_idx, lab_idx] = 0.0
            # If one has pixels and the other doesn't, set the score to NaN
            else:
                hd_scores[batch_idx, lab_idx] = float('nan') 
        
    if ignore_empty_labels:
        true_amounts = torch.sum(torch.from_numpy(y_true_one_hot), dim=(-2, -1)) # B x C
        existing_label = (true_amounts > 0).float()
        if weights is None:
            weights = existing_label
        else:
            weights = weights * existing_label
    elif weights is None:
        weights = torch.ones_like(hd_scores) # Need to set weights to 1 for this in particular.
    
    # If we want to ignore a label, set its weight to 0.
    if ignore_index is not None:
        assert 0 <= ignore_index < C, "ignore_index must be in [0, channels)"
        weights[:, ignore_index] = 0.0
 
    # If the weight of a nan score is 0, then we want to set it to 0 instead of nan,
    # so that the reduction doesn't fail. This only is true if the weight is 0 and the
    # score is nan.
    nan_mask = torch.isnan(hd_scores) & (weights == 0.0)
    hd_scores[nan_mask] = 0.0

    return _metric_reduction(
        hd_scores,
        reduction=reduction,
        weights=weights,
        batch_reduction=batch_reduction,
    )

# Define the classwise versions.

PixelNUWCELoss = _loss_module_from_func("PixelNUWCELoss", pixel_neighbor_uniform_w_cross_entropy)