# torch imports
import torch
from torch import Tensor
from torch.nn import functional as F
# misc imports
from typing import Optional
from pydantic import validate_arguments
import matplotlib.pyplot as plt
# local imports
from ionpy.metrics.util import Reduction


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def volume_error(
    y_pred: Tensor,
    y_true: Tensor,
    do_threshold: bool,
    from_logits: bool = False,
    vol_dim: Optional[int] = 1, # If None, we are considering all classes
    threshold: Optional[float] = 0.5,
    batch_reduction: Reduction = "mean",
):
    assert len(y_pred.shape) == len(y_true.shape) == 4, "Input tensors must be 4D"

    # Note this only really makes sense in non-binary contexts.
    if from_logits:
        if y_pred.shape[1] > 1:
            y_pred = F.softmax(y_pred, dim=1)
        else:
            y_pred = torch.sigmoid(y_pred)
        
    # If we have a vol_dim then we are only considering a single class
    if vol_dim is not None:
        y_pred = y_pred[:, vol_dim].unsqueeze(1)
    
    if do_threshold:
        y_pred = (y_pred > threshold).float()

    # Reshape the tensors to be 2D
    y_pred = y_pred.view(y_pred.size(0), -1)
    y_true = y_true.view(y_true.size(0), -1)

    # Compute the volume error
    vol_err = torch.sum(y_pred, dim=1) - torch.sum(y_true, dim=1)

    if batch_reduction == "mean":
        return vol_err.mean()
    else:
        return vol_err
