# torch imports
from torch import Tensor
from torch.nn import functional as F
# misc imports
from pydantic import validate_arguments
from typing import Optional, Union, List
# ionpy imports
from ionpy.loss.util import _loss_module_from_func 
from ionpy.metrics.util import (
    _metric_reduction,
    _inputs_as_onehot,
    InputMode,
    Reduction,
)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def nll_pixel_loss(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    from_logits: bool = False,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    ignore_index: Optional[int] = None,
    weights: Optional[Union[Tensor, List]] = None,
) -> Tensor:

    y_pred, y_true = _inputs_as_onehot(
        y_pred, y_true, mode=mode, from_logits=from_logits, discretize=False
    )
    nll_loss = F.nll_loss(y_pred, y_true)
    return _metric_reduction(
        nll_loss,
        reduction=reduction,
        weights=weights,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )

NLLPixelLoss = _loss_module_from_func("NLLPixelLoss", nll_pixel_loss)