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
    elif combine_fn == "upperbound":
        return upperbound_combine_fn
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
    ensemble_preds,
    combine_quantity: Literal["probs", "logits"],
    from_logits: bool,
    normalize: bool,
    weights: Optional[Tensor] = None,
    **kwargs
):
    if isinstance(ensemble_preds, dict):
        ensemble_preds = batch_ensemble_preds(ensemble_preds) # B, C, E, H, W

    if combine_quantity == "probs" and from_logits:
        ensemble_preds = torch.softmax(ensemble_preds, dim=1)

    # Multiply the logits by the weights.
    if weights is None:
        ensemble_mean_tensor = torch.mean(ensemble_preds, dim=2) # B, C, H, W
    else:
        weights = weights.reshape(1, 1, -1, 1, 1) # 1, 1, E, 1, 1
        w_ensemble_logits = weights * ensemble_preds # B, C, E, H, W
        ensemble_mean_tensor = torch.sum(w_ensemble_logits, dim=2) # B, C, H, W

    if (combine_quantity == "logits") and from_logits:
        ensemble_mean_tensor = torch.softmax(ensemble_mean_tensor, dim=1) # B, C, H, W
    
    # Make the output distribution a valid probability distribution.
    if normalize:
        # Get the sum across classes
        classwise_sum = ensemble_mean_tensor.sum(dim=1, keepdim=True) # B, 1, H, W
        classwise_sum[classwise_sum == 0] = 1
        ensemble_mean_tensor = ensemble_mean_tensor / classwise_sum

    return ensemble_mean_tensor


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def product_combine_fn(
    ensemble_preds,
    from_logits: bool,
    normalize: bool,
    weights: Optional[Tensor] = None,
    **kwargs
):
    if isinstance(ensemble_preds, dict):
        ensemble_preds = batch_ensemble_preds(ensemble_preds) # B, C, E, H, W

    # We always need to softmax the logits before taking the product.
    if from_logits:
        ensemble_preds = torch.softmax(ensemble_preds, dim=1)

    # We want to calculate the weighted geometric mean of the probabilities.
    if weights is None:
        prod = torch.prod(ensemble_preds, dim=2)
        E = ensemble_preds.shape[2]
        ensemble_product_tensor = torch.pow(prod, 1.0/E)
    else:
        weights = weights.reshape(1, 1, -1, 1, 1) # 1, 1, E, 1, 1
        prod = torch.prod(ensemble_preds, dim=2) # B, C, H, W
        ensemble_product_tensor  = torch.pow(prod, weights) # B, C, H, W

    # Make the output distribution a valid probability distribution.
    if normalize:
        # Get the sum across classes
        classwise_sum = ensemble_product_tensor.sum(dim=1, keepdim=True) # B, 1, H, W
        classwise_sum[classwise_sum == 0] = 1
        ensemble_product_tensor = ensemble_product_tensor / classwise_sum

    return ensemble_product_tensor


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def upperbound_combine_fn(
    ensemble_preds,
    y_true: Tensor,
    from_logits: bool,
    normalize: bool,
    **kwargs
):
    if isinstance(ensemble_preds, dict):
        ensemble_preds = batch_ensemble_preds(ensemble_preds)

    # Gather the individual predictions
    B, C, E, H, W = ensemble_preds.shape
    if from_logits:
        ensemble_probs = torch.softmax(ensemble_preds, dim=1) # B x C x E x H x W
    ensemble_hard_preds = torch.argmax(ensemble_probs, dim=1) # B x E x H x W
    # Get the upper bound prediction by going through and updating the prediction
    # by the pixels each model got right.
    ensemble_ub_pred = ensemble_probs[:, :, 0, ...] # B x C x H x W
    for ens_idx in range(E):
        correct_positions = (ensemble_hard_preds[:, ens_idx:ens_idx+1, ...] == y_true) # B x 1 x H x W
        correct_index = correct_positions.repeat(1, C, 1, 1) # B x C x H x W
        ensemble_ub_pred[correct_index] = ensemble_probs[:, :, ens_idx, ...][correct_index]
    # Here we need a check that if we sum across channels we get 1.
    assert torch.allclose(ensemble_ub_pred.sum(dim=1), torch.ones((B, H, W), device=ensemble_ub_pred.device)),\
        "The upper bound prediction does not sum to 1 across channels."
    return ensemble_ub_pred


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

