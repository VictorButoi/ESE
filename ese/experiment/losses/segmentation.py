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
from ionpy.loss.util import _loss_module_from_func
from ionpy.metrics.util import (
    InputMode,
    Reduction
)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def bw_pixel_crossentropy_loss(
    y_pred: Tensor,
    y_true: Tensor,
    dist_to_boundary: Tensor,
    sigma: float = 8.0,
    mode: InputMode = "auto",
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, list]] = None,
    ignore_index: Optional[int] = -100,
    from_logits: bool = False,
):
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
        assert ignore_index is None
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
    
    def rbf_kernel(dist_tfm, sigma):
        constant = 1 / (sigma * torch.sqrt(2 * torch.tensor(math.pi, dtype=torch.float64)))
        return constant * torch.exp(-0.5 * (dist_tfm / sigma**2))

    # Calculate pixel weights, based on distance to boundary and smoothing factor.
    pixel_weights = rbf_kernel(dist_to_boundary, sigma)    

    # Weight the loss by the pixel weights.
    loss = loss * pixel_weights

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


BWPixelCELoss = _loss_module_from_func("BWPixelCELoss", bw_pixel_crossentropy_loss)