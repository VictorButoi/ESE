# torch imports
import torch
from torch import Tensor
from torch.nn import functional as F
# misc imports
import matplotlib.pyplot as plt
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
# local imports
from .utils import agg_neighbors_preds


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def dice_score(
    y_pred: Tensor,
    y_true: Tensor,
    eps: float = 1e-7,
    smooth: float = 1e-7,
    threshold: float = 0.5,
    mode: InputMode = "auto",
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    from_logits: bool = False,
    ignore_empty_labels: bool = True,
    ignore_index: Optional[int] = None,
    weights: Optional[Union[Tensor, List]] = None,
) -> Tensor:

    y_pred, y_true = _inputs_as_onehot(
        y_pred, 
        y_true, 
        mode=mode, 
        from_logits=from_logits, 
        discretize=True,
        threshold=threshold
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
    threshold: float = 0.5,
    from_logits: bool = False,
    ignore_empty_labels: bool = False,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, List]] = None,
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
    if y_pred.shape[1] > 1:
        if y_pred.shape[1] == 2 and threshold != 0.5:
            y_pred = (y_pred[:, 1, ...] > threshold).long()
        else:
            y_pred = y_pred.argmax(dim=1)
    else:
        y_pred = (y_pred > threshold).long()
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


def boundary_iou(
    y_pred: Tensor,
    y_true: Tensor,
    boundary_width: int = 1,
    mode: InputMode = "auto",
    smooth: float = 1e-7,
    eps: float = 1e-7,
    threshold: float = 0.5,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, List]] = None,
    ignore_empty_labels: bool = True,
    from_logits: bool = False,
    ignore_index: Optional[int] = None,
) -> Tensor:

    if from_logits:
        y_pred = torch.softmax(y_pred, dim=1) # Label channels are 1 by default.

    B, C, _, _ = y_pred.shape

    y_true = y_true.squeeze(1) # B x H x W
    if C == 1:
        y_hard = (y_pred > threshold).squeeze(1) # B x H x W
    else:
        y_hard = y_pred.argmax(dim=1) # B x H x W

    n_width = 2*boundary_width + 1 # The width of the neighborhood.
    neighb_args = {
        "class_wise": True,
        "binary": False, # Include background a having num neighbors.
        "neighborhood_width": n_width,
        "discrete": True,
        "num_classes": C
    }
    true_num_neighb_map = agg_neighbors_preds(
                            pred_map=y_true, # B x H x W
                            **neighb_args
                        ) # B x C x H x W
    pred_num_neighb_map = agg_neighbors_preds(
                            pred_map=y_hard, # B x H x W
                            **neighb_args
                        ) # B x C x H x W

    # Get the non-center pixels.
    max_matching_neighbors = (n_width**2 - 1) # The center pixel is not counted.
    boundary_pred = (pred_num_neighb_map < max_matching_neighbors) # B x C x H x W
    boundary_true = (true_num_neighb_map < max_matching_neighbors) # B x C x H x W

    # Get the one hot tensors. 
    y_pred = F.one_hot(y_hard, num_classes=C).permute(0, 3, 1, 2).float()
    y_true = F.one_hot(y_true.squeeze(1), num_classes=C).permute(0, 3, 1, 2).float()

    # Mask the true and pred tensors by the boundary tensors and then flatten the last
    # two dimensions.
    y_pred = (y_pred * boundary_pred).view(B, C, -1)
    y_true = (y_true * boundary_true).view(B, C, -1)

    intersection = torch.logical_and(y_pred == 1.0, y_true == 1.0).sum(dim=-1)
    true_amounts = (y_true == 1.0).sum(dim=-1)
    pred_amounts = (y_pred == 1.0).sum(dim=-1)
    cardinalities = true_amounts + pred_amounts
    union = cardinalities - intersection
    score = (intersection + smooth) / (union + smooth).clamp_min(eps)

    if ignore_empty_labels:
        existing_label = (true_amounts > 0).float()
        if weights is None:
            weights = existing_label
        else:
            weights = weights * existing_label

    return _metric_reduction(
        score,
        reduction=reduction,
        weights=weights,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )
