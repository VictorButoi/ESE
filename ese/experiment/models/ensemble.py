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
    ensemble_logits: dict, 
    pre_softmax: bool
):
    return batch_ensemble_preds(ensemble_logits)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def mean_combine_fn(
    ensemble_logits: dict, 
    pre_softmax: bool
):
    batch_ensemble_tensor = batch_ensemble_preds(ensemble_logits) # B, E, C, H, W

    if pre_softmax:
        batch_ensemble_tensor = torch.softmax(batch_ensemble_tensor, dim=2)

    batch_mean_tensors = torch.mean(batch_ensemble_tensor, dim=1) # B, C, H, W

    if not pre_softmax:
        batch_mean_tensors = torch.softmax(batch_mean_tensors, dim=1)

    return batch_mean_tensors

def max_combine_fn(
    ensemble_logits: dict, 
    pre_softmax: bool
):
    batch_ensemble_tensor = batch_ensemble_preds(ensemble_logits) # B, E, C, H, W

    if pre_softmax:
        batch_ensemble_tensor = torch.softmax(batch_ensemble_tensor, dim=2)

    batch_max_tensors = torch.max(batch_ensemble_tensor, dim=1)[0] # B, C, H, W, max returns max vals and indices

    if not pre_softmax:
        batch_max_tensors = torch.softmax(batch_max_tensors, dim=1)

    return batch_max_tensors