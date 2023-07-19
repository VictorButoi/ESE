import torch
from torch import Tensor
from typing import Optional, Union
from pydantic import validate_arguments

from IonPy.metrics.segmentation import soft_dice_score
from IonPy.metrics.util import (
    InputMode,
    Reduction,
)
from .util import _modified_loss_module_from_func


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def soft_dice_loss(
    y_pred: Tensor,
    y_true: Tensor,
    logits: Tensor = None, # probs is not used in the loss function.
    support_inds: Tensor = None, # support_inds is not used in the loss function.
    mode: InputMode = "auto",
    z_norm: bool = False,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, list]] = None,
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
    smooth: float = 1e-7,
    eps: float = 1e-7,
    square_denom: bool = True,
) -> Tensor:
    assert z_norm == False, "z_norm is not supported for soft_dice_loss."

    score = soft_dice_score(
        y_pred,
        y_true,
        mode=mode,
        reduction=None,
        batch_reduction=None,
        weights=weights,
        ignore_index=ignore_index,
        from_logits=from_logits,
        smooth=smooth,
        eps=eps,
        square_denom=square_denom,
    )

    loss = 1.0 - score

    if batch_reduction is not None:
        loss = loss.mean() if batch_reduction == "mean" else loss.sum()

    return loss


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def min_nll_with_dice(
    y_pred: Tensor,
    y_true: Tensor,
    logits: Tensor,
    support_inds: Tensor,
    mode: InputMode = "auto",
    z_norm: bool = False,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, list]] = None,
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
    smooth: float = 1e-7,
    eps: float = 1e-7,
    square_denom: bool = True,
) -> Tensor:

    score = soft_dice_score(
        y_pred,
        y_true,
        mode=mode,
        reduction=None,
        batch_reduction=None,
        weights=weights,
        ignore_index=ignore_index,
        from_logits=from_logits,
        smooth=smooth,
        eps=eps,
        square_denom=square_denom,
    )

    # Dice loss
    dice_loss = 1.0 - score

    # Get prob of support set
    probs = torch.softmax(logits, dim=1)
    neg_likelihood = -torch.log(torch.prod(probs[support_inds], dim=1) + eps)

    # Noramlize the losses if we take the optimization scheme as proposed in https://arxiv.org/pdf/2205.12548.pdf
    if z_norm:
        assert len(neg_likelihood.shape) == 2, "z_norm is only supported for batched losses."
        neg_likelihood = (neg_likelihood - neg_likelihood.mean()) / neg_likelihood.std()

    # Weight loss by likelihood of the support
    loss = neg_likelihood * dice_loss

    if batch_reduction is not None:
        loss = loss.mean() if batch_reduction == "mean" else loss.sum()

    return loss

SoftDiceLoss = _modified_loss_module_from_func("SoftDiceLoss", soft_dice_loss)
NllWithDice = _modified_loss_module_from_func("NllWithDice", min_nll_with_dice)
