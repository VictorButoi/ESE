# torch imports
import torch
from torch import Tensor
from torch.nn import functional as F
# misc imports
from typing import Optional
from pydantic import validate_arguments
# local imports
from ionpy.metrics.util import Reduction


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def volume_error(
    y_pred: Tensor,
    y_true: Tensor,
    from_logits: bool = False,
    do_threshold: bool = False,
    threshold: Optional[float] = 0.5,
    batch_reduction: Reduction = "mean",
):
    # Note this only really makes sense in non-binary contexts.
    if from_logits:
        y_pred = F.softmax(y_pred, dim=1)
    
    # Flatten the tensors to be B x -1
    y_pred = y_pred.view(y_pred.size(0), -1)
    y_true = y_true.view(y_true.size(0), -1)
    
    if do_threshold:
        y_pred = (y_pred > threshold).float()
    
    # Compute the volume error
    vol_err = torch.sum(y_pred, dim=1) - torch.sum(y_true, dim=1)

    if batch_reduction == "mean":
        return vol_err.mean()
    else:
        return vol_err
