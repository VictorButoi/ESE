# torch imports
import torch
from torch import Tensor
# misc imports
from pydantic import validate_arguments
# ionpy imports
from ionpy.metrics.util import Reduction
import matplotlib.pyplot as plt


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def entropy_score(
    y_pred: Tensor,
    y_true: Tensor,
    eps: float = 1e-10,
    from_logits: bool = False,
    batch_reduction: Reduction = "mean",
):
    """
    Calculates the entropy of the y_pred.
    """
    if from_logits:
        if y_pred.shape[1] == 1:
            # If the input is multi-channel for confidence, take the max across channels.
            y_pred = torch.sigmoid(y_pred)
        else:
            y_pred = torch.softmax(y_pred, dim=1)

    entropy_map = - (y_pred * torch.log(y_pred + eps)).sum(dim=1)
    # Average over all spatial resolution dimensions to obtain one entropy value per batch item.
    # Identify all dimensions beyond the batch dimension (dimension 0)
    spatial_dims = list(range(1, entropy_map.dim()))
    batch_entropy = entropy_map.mean(dim=spatial_dims)
    # Return the brier score.
    if batch_reduction == "mean":
        return batch_entropy.mean()
    else:
        return batch_entropy 
    
