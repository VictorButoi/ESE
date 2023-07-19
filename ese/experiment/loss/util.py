import torch.nn as nn

from IonPy.util.meta import delegates, get_default_kwargs
from IonPy.util.mapping import allbut


def _modified_loss_module_from_func(name, loss_func):
    class _LossWrapper(nn.Module):
        @delegates(to=loss_func)
        def __init__(self, **kwargs):
            super().__init__()
            self._func_kwargs = allbut(
                get_default_kwargs(loss_func), ["y_pred", "y_true", "logits", "support_inds"]
            )
            self._func_kwargs.update(kwargs)
            self.__dict__.update(self._func_kwargs)

        def forward(self, y_pred, y_true, logits=None, support_inds=None):
            return loss_func(y_pred, y_true, logits, support_inds, **self._func_kwargs)

    return type(name, (_LossWrapper,), {})