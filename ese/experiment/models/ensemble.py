# torch imports
import torch
from typing import Literal, Optional
from pydantic import validate_arguments


def get_combine_fn(combine_fn: str):
    if combine_fn == "identity":
        return identity_combine_fn
    elif combine_fn == "mean":
        return mean_combine_fn
    elif combine_fn == "product":
        return product_combine_fn
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
    ensemble_logits, 
    combine_quantity: Optional[str] = None
    ):
    return batch_ensemble_preds(ensemble_logits)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def mean_combine_fn(
    ensemble_logits,
    combine_quantity: Literal["probs", "logits"] 
):
    if isinstance(ensemble_logits, dict):
        ensemble_logits = batch_ensemble_preds(ensemble_logits) # B, C, E, H, W

    if combine_quantity == "probs":
        ensemble_logits = torch.softmax(ensemble_logits, dim=1)

    ensemble_mean_tensor = torch.mean(ensemble_logits, dim=2) # B, C, H, W

    if combine_quantity == "logits":
        ensemble_mean_tensor = torch.softmax(ensemble_mean_tensor, dim=1)

    return ensemble_mean_tensor


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def product_combine_fn(
    ensemble_logits,
    combine_quantity: Literal["probs"] 
):
    if isinstance(ensemble_logits, dict):
        ensemble_logits = batch_ensemble_preds(ensemble_logits) # B, C, E, H, W

    # We always need to softmax the logits before taking the product.
    ensemble_probs = torch.softmax(ensemble_probs, dim=1)

    # This is the geometric mean
    scaled_ensemble_logits = torch.pow(ensemble_logits, 1/ensemble_logits.shape[2]) # B, C, E, H, W
    ensemble_product_tensor = torch.prod(scaled_ensemble_logits, dim=2) # B, C, H, W

    return ensemble_product_tensor