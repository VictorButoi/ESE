# torch imports
import torch
from torch import Tensor
from torch.nn import functional as F
# misc imports
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from pydantic import validate_arguments
from typing import Optional, Union
# local imports
from ionpy.loss.util import _loss_module_from_func
from ionpy.metrics.util import (
    InputMode,
    Reduction
)
# local imports
from ..metrics.utils import agg_neighbors_preds


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def bw_pixel_crossentropy_loss(
    y_pred: Tensor,
    y_true: Tensor,
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
    
    # Get the distance transform of the label.
    viz_y = y_true.cpu().numpy().squeeze()

    f, axarr = plt.subplots(1, 2, figsize=(16, 8))
    dist_to_boundary = ndimage.distance_transform_edt(viz_y)
    background_dist_to_boundary = ndimage.distance_transform_edt(1 - viz_y)
    combined_dist_to_boundary = (dist_to_boundary + background_dist_to_boundary)/2

    im1 = axarr[0].imshow(viz_y, interpolation='None')
    axarr[0].set_title('Label')
    axarr[0].grid(False)
    f.colorbar(im1, ax=axarr[0])

    # im2 = axarr[1].imshow(dist_to_boundary, interpolation='None', cmap='rocket_r')
    im2 = axarr[1].imshow(combined_dist_to_boundary, interpolation='None', cmap='rocket_r')
    axarr[1].set_title('Distance to Boundary')
    axarr[1].grid(False)
    f.colorbar(im2, ax=axarr[1])
    plt.show()

    def brb_w(dist_tfm, sigma):
        return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*(dist_tfm/sigma**2))
    
    f, axarr = plt.subplots(1, 5, figsize=(20, 4))
    for s_idx, sig in enumerate([1, 4, 8, 16, 32]):
        sig_im = axarr[s_idx].imshow(brb_w(combined_dist_to_boundary, sig), interpolation='None', cmap='rocket_r')
        axarr[s_idx].set_title(f'BRB Weight, sigma={sig}')
        axarr[s_idx].grid(False)
        f.colorbar(sig_im, ax=axarr[s_idx])
    plt.show()
    
    print(loss.shape)
    raise ValueError

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