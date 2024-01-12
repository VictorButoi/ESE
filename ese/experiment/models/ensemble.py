# torch imports
import torch
from typing import Union
from pydantic import validate_arguments


def get_combine_fn(combine_fn: str):
    if combine_fn == "identity":
        return identity_combine_fn
    elif combine_fn == "mean":
        return mean_combine_fn
    elif combine_fn == "max":
        return max_combine_fn
    else:
        raise ValueError(f"Unknown combine function '{combine_fn}'.")


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def batch_ensemble_preds(model_outputs: dict):
    preds = list(model_outputs.values())
    pred_tensor = torch.stack(preds) # E, B, C, H, W
    batchwise_ensemble_tensor = pred_tensor.permute(1, 2, 0, 3, 4) # B, C, E, H, W
    return batchwise_ensemble_tensor


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def identity_combine_fn(
    ensemble_logits, # Either a dict or a tensor.
    pre_softmax: bool
):
    return batch_ensemble_preds(ensemble_logits)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def mean_combine_fn(
    ensemble_logits,
    pre_softmax: bool
):
    if isinstance(ensemble_logits, dict):
        ensemble_logits = batch_ensemble_preds(ensemble_logits) # B, C, E, H, W

    if pre_softmax:
        ensemble_logits = torch.softmax(ensemble_logits, dim=1)

    ensemble_mean_tensor = torch.mean(ensemble_logits, dim=2) # B, C, H, W

    if not pre_softmax:
        ensemble_mean_tensor = torch.softmax(ensemble_mean_tensor, dim=1)

    return ensemble_mean_tensor


def max_combine_fn(
    ensemble_logits, # Either a dict or a tensor.
    pre_softmax: bool
):
    if isinstance(ensemble_logits, dict):
        ensemble_logits = batch_ensemble_preds(ensemble_logits) # B, C, E, H, W

    if pre_softmax:
        ensemble_logits = torch.softmax(ensemble_logits, dim=1)

    ensemble_max_tensor = torch.max(ensemble_logits, dim=2)[0] # B, C, H, W, max returns max vals and indices

    if not pre_softmax:
        ensemble_max_tensor = torch.softmax(ensemble_max_tensor, dim=1)

    return ensemble_max_tensor