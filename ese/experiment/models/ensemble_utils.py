# torch imports
import torch
import pandas as pd
from torch import Tensor
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
def batch_ensemble_preds(
    model_outputs: dict,
):
    sorted_paths = sorted(list(model_outputs.keys()))
    preds = [model_outputs[model_path] for model_path in sorted_paths]
    # Convert to tensors.
    raw_pred_tensor = torch.stack(preds) # E, B, C, H, W
    # Reshape to allow for broadcasting.
    pred_tensor = raw_pred_tensor.permute(1, 2, 0, 3, 4) # B, C, E, H, W
    # Return the reshaped tensors.
    return pred_tensor


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def identity_combine_fn(
    ensemble_logits: dict, 
    **kwargs
):
    return batch_ensemble_preds(ensemble_logits)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def mean_combine_fn(
    ensemble_logits,
    weights: Tensor,
    combine_quantity: Literal["probs", "logits"],
    **kwargs
):
    if isinstance(ensemble_logits, dict):
        ensemble_logits = batch_ensemble_preds(ensemble_logits)

    if combine_quantity == "probs":
        ensemble_logits = torch.softmax(ensemble_logits, dim=1)

    # Multiply the logits by the weights.
    weights = weights.reshape(1, 1, -1, 1, 1) # 1, 1, E, 1, 1
    w_ensemble_logits = weights * ensemble_logits # B, C, E, H, W
    ensemble_mean_tensor = torch.sum(w_ensemble_logits, dim=2) # B, C, H, W

    if combine_quantity == "logits":
        ensemble_mean_tensor = torch.softmax(ensemble_mean_tensor, dim=1)

    return ensemble_mean_tensor


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def product_combine_fn(
    ensemble_logits,
    weights: Tensor,
    **kwargs
):
    if isinstance(ensemble_logits, dict):
        ensemble_logits = batch_ensemble_preds(ensemble_logits)

    # We always need to softmax the logits before taking the product.
    ensemble_probs = torch.softmax(ensemble_logits, dim=1)

    # We want to calculate the weighted geometric mean of the probabilities.
    weights = weights.reshape(1, 1, -1, 1, 1) # 1, 1, E, 1, 1
    scaled_ensemble_probs = torch.pow(ensemble_probs, weights) # B, C, E, H, W
    ensemble_product_tensor = torch.prod(scaled_ensemble_probs, dim=2) # B, C, H, W

    return ensemble_product_tensor


def get_ensemble_member_weights(
    results_df: pd.DataFrame, 
    metric: str
) -> Tensor:
    if metric == "None":
        # Get the sorted path keys.
        exp_paths = results_df["path"].unique()
        sorted_path_keys = sorted(exp_paths)
        weights = Tensor([(1 / len(exp_paths)) for _ in sorted_path_keys])
    else:
        metric_components = metric.split("-")
        assert len(metric_components) == 2, "Metric must be of the form 'split-metric'."
        phase = metric_components[0]
        metric_name = metric_components[1]
        # Get the df corresponding to split.
        split_df = results_df[results_df["phase"] == phase]
        # Keep two columns, the path and the metric_name
        score_df = split_df[["path", metric_name]] 
        path_grouped_df = score_df.groupby("path")
        # If we are dealing with a 'loss' then take the min, if a 'score' take the max, otherwise default to min (with a print).
        if 'loss' in metric_name:
            best_df = path_grouped_df.min()
            metric_type = "min"
        elif 'score' in metric_name:
            best_df = path_grouped_df.max()
            metric_type = "max"
        else:
            print(f"Unknown metric type for '{metric_name}'. Defaulting to min.")
            best_df = path_grouped_df.min()
            metric_type = "min"
        # Get the weights in a dictionary
        weight_dict = best_df.to_dict()[metric_name]
        # If we are dealing with a 'min' metric_type we need to invert the weights.
        if metric_type == "min":
            weight_dict = {path: 1 / weight for path, weight in weight_dict.items()}
        total_weight = sum(weight_dict.values()) 
        # Get the sorted path keys.
        sorted_path_keys = sorted(list(weight_dict.keys()))
        weights = Tensor([(weight_dict[path] / total_weight) for path in sorted_path_keys])
    # Check that the weights sum to 1.
    assert torch.isclose(weights.sum(), Tensor([1.0])), "Weights do not sum to 1."
    # Return the weight dict.
    return weights 