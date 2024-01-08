# torch imports
import torch

def get_combine_fn(combine_fn):
    if combine_fn == "identity":
        return identity_combine_fn
    elif combine_fn == "mean":
        return mean_combine_fn
    elif combine_fn == "max":
        return max_combine_fn
    else:
        raise ValueError(f"Unknown combine function '{combine_fn}'.")


def batch_ensemble_preds(model_outputs):
    preds = list(model_outputs.values())
    pred_tensor = torch.stack(preds) # E, B, C, H, W
    batchwise_ensemble_tensor = pred_tensor.permute(1, 0, 2, 3, 4) # B, E, C, H, W
    return batchwise_ensemble_tensor


def identity_combine_fn(model_outputs: dict):
    return batch_ensemble_preds(model_outputs)


def mean_combine_fn(model_outputs: dict, pre_softmax: bool):
    batch_ensemble_tensor = batch_ensemble_preds(model_outputs) # B, E, C, H, W

    if pre_softmax:
        batch_ensemble_tensor = torch.softmax(batch_ensemble_tensor, dim=2)

    batch_mean_tensors = torch.mean(batch_ensemble_tensor, dim=1) # B, C, H, W

    if not pre_softmax:
        batch_mean_tensors = torch.softmax(batch_mean_tensors, dim=1)

    return batch_mean_tensors

def max_combine_fn(model_outputs: dict, pre_softmax: bool):
    batch_ensemble_tensor = batch_ensemble_preds(model_outputs) # B, E, C, H, W

    if pre_softmax:
        batch_ensemble_tensor = torch.softmax(batch_ensemble_tensor, dim=2)

    batch_max_tensors = torch.max(batch_ensemble_tensor, dim=1) # B, C, H, W

    if not pre_softmax:
        batch_mean_tensors = torch.softmax(batch_mean_tensors, dim=1)

    return batch_max_tensors