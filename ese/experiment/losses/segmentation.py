# torch imports
import torch
from torch import Tensor
from torch.nn import functional as F
# misc imports
import math
import matplotlib.pyplot as plt
from pydantic import validate_arguments
from typing import Optional, Union
# local imports
from .weights import get_pixel_weights
from ionpy.loss.util import _loss_module_from_func
from ionpy.metrics.segmentation import soft_dice_score
from ionpy.metrics.util import (
    InputMode,
    Reduction
)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def soft_dice_loss(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, list]] = None,
    ignore_index: Optional[int] = None,
    ignore_empty_labels: bool = False,
    from_logits: bool = False,
    smooth: float = 1e-7,
    eps: float = 1e-7,
    square_denom: bool = True,
    log_loss: bool = False,
) -> Tensor:
    # Quick check to see if we are dealing with binary segmentation
    if y_pred.shape[1] == 1:
        assert ignore_index is None, "ignore_index is not supported for binary segmentation."

    score = soft_dice_score(
        y_pred,
        y_true,
        mode=mode,
        reduction=reduction,
        batch_reduction=batch_reduction,
        weights=weights,
        ignore_empty_labels=ignore_empty_labels,
        ignore_index=ignore_index,
        from_logits=from_logits,
        smooth=smooth,
        eps=eps,
        square_denom=square_denom,
    )
    # Assert that everywhere the score is between 0 and 1 (batch many items)
    assert (score >= 0).all() and (score <= 1).all(), f"Score is not between 0 and 1: {score}"

    if log_loss:
        loss = -torch.log(score.clamp_min(eps))
    else:
        loss = 1.0 - score

    return loss


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def pixel_crossentropy_loss(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    loss_pix_weights: Optional[str] = None,
    weights: Optional[Union[Tensor, list]] = None,
    ignore_index: Optional[int] = -100,
    from_logits: bool = False,
):
    # Quick check to see if we are dealing with binary segmentation
    if y_pred.shape[1] == 1:
        assert ignore_index == -100, "ignore_index is not supported for binary segmentation."

    """One cross_entropy function to rule them all
    ---
    Pytorch has four CrossEntropy loss-functions
        1. Binary CrossEntropy
          - nn.BCELoss
          - F.binary_cross_entropy
        2. Sigmoid + Binary CrossEntropy (expects logits)
          - nn.BCEWithLogitsLoss
          - F.binary_cross_entropy_with_logits
        3. Categorical
          - nn.NLLLoss
          - F.nll_loss
        4. Softmax + Categorical (expects logits)
          - nn.CrossEntropyLoss
          - F.cross_entropy
    """
    assert len(y_pred.shape) > 2, "y_pred must have at least 3 dimensions."
    batch_size, num_classes = y_pred.shape[:2]
    y_true = y_true.long()

    if mode == "auto":
        if y_pred.shape == y_true.shape:
            mode = "binary" if num_classes == 1 else "onehot"
        else:
            mode = "multiclass"

    # If weights are a list turn them into a tensor
    if isinstance(weights, list):
        weights = torch.tensor(weights, device=y_pred.device, dtype=y_pred.dtype)

    if mode == "binary":
        assert y_pred.shape == y_true.shape
        assert weights is None
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
                weight=weights,
                ignore_index=ignore_index,
            )
        else:
            loss = F.nll_loss(
                y_pred,
                y_true,
                reduction="none",
                weight=weights,
                ignore_index=ignore_index,
            )
    
    if loss_pix_weights is not None and loss_pix_weights.lower() != "none":
        pix_weights = get_pixel_weights(
            y_true=y_true,
            y_pred=y_pred,
            loss_func=loss_pix_weights,
            from_logits=from_logits
        )
        # Multiply the loss by the pixel weights
        # print the range the loss tensor before and after
        loss = loss * pix_weights 

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


PixelCELoss = _loss_module_from_func("PixelCELoss", pixel_crossentropy_loss)
SoftDiceLoss = _loss_module_from_func("SoftDiceLoss", soft_dice_loss)