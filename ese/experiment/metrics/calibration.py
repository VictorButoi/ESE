from pydantic import validate_arguments
from ionpy.metrics.util import _inputs_as_longlabels, InputMode
from torch import Tensor


validate_arguments(config=dict(arbitrary_types_allowed=True))
def ECE(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    from_logits: bool = False,
    bin_size: float = 0.1,
) -> Tensor:

    y_pred, y_true = _inputs_as_longlabels(
        y_pred, y_true, mode, from_logits=from_logits, discretize=True
    )


    return correct.float().mean()