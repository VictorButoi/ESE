import torch
import pandas as pd
from pydantic import validate_arguments 
from torch.utils.data import DataLoader
from ionpy.experiment.util import absolute_import
from typing import Optional
from ..metrics.utils import (
    get_edge_map,
    get_uni_pixel_weights,
    count_matching_neighbors
)

def reorder_splits(df):
    if 'split' in df.keys():
        train_logs = df[df['split'] == 'train']
        val_logs = df[df['split'] == 'val']
        cal_logs = df[df['split'] == 'cal']
        fixed_df = pd.concat([train_logs, val_logs, cal_logs])
        return fixed_df
    else:
        return df

# This function will take in a dictionary of pixel meters and a metadata dataframe
# from which to select the log_set corresponding to particular attributes, then 
# we index into the dictionary to get the corresponding pixel meters.
def select_pixel_dict(pixel_meter_logdict, metadata, kwargs):
    # Select the metadata
    metadata = metadata.select(**kwargs)
    # Get the log set
    assert len(metadata) == 1, f"Need exactly 1 log set, found: {len(metadata)}."
    log_set = metadata['log_set'].iloc[0]
    # Return the pixel dict
    return pixel_meter_logdict[log_set]
    

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def dataloader_from_exp(
        exp, 
        new_dset_options=None, 
        return_data_id=False,
        num_workers=1
        ):
    exp_data_cfg = exp.config['data'].to_dict()
    if new_dset_options is not None:
        for key in new_dset_options.keys():
            exp_data_cfg[key] = new_dset_options[key]
    # Make sure we aren't sampling for evaluation. 
    if "slicing" in exp_data_cfg.keys():
        assert exp_data_cfg["slicing"] not in ["central", "dense", "uniform"], "Sampling methods not allowed for evaluation."
    # Get the dataset class and build the transforms
    dataset_cls = exp_data_cfg.pop("_class")
    # Drop auxiliary information used for making the models.
    if "in_channels" in exp_data_cfg.keys():
        exp_data_cfg.pop("in_channels")
    if "out_channels" in exp_data_cfg.keys():
        exp_data_cfg.pop("out_channels")
    dataset_obj = absolute_import(dataset_cls)(**exp_data_cfg)
    exp_data_cfg["_class"] = dataset_cls        
    dataset_obj.return_data_id = return_data_id 
    # Build the dataset and dataloader.
    dataloader = DataLoader(
        dataset_obj, 
        batch_size=1, 
        num_workers=num_workers,
        shuffle=False
    )
    return dataloader, exp_data_cfg


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def binarize(
    label_tensor: torch.Tensor, 
    label: int,
    discretize: bool
    ):
    assert label_tensor.dim() == 4, f"Expected 4D tensor, found: {label_tensor.dim()}."
    if discretize:
        binary_label_tensor = (label_tensor == label).type(label_tensor.dtype)
    else:
        label_channel = label_tensor[:, label:label+1, :, :]
        background_channel = label_tensor.sum(dim=1, keepdim=True) - label_channel 
        binary_label_tensor = torch.cat([background_channel, label_channel], dim=1)
    return binary_label_tensor


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_image_aux_info(
        y_hard: torch.Tensor,
        y_true: torch.Tensor,
        neighborhood_width: int,
        ignore_index: Optional[int] = None
):
    # Get the pixelwise accuracy.
    accuracy_map = (y_hard == y_true).float().squeeze()

    # Keep track of different things for each bin.
    pred_labels = y_hard.unique().tolist()
    if ignore_index is not None and ignore_index in pred_labels:
        pred_labels.remove(ignore_index)

    # Get a map of which pixels match their neighbors and how often, and pixel-wise accuracy.
    # For both our prediction and the true label map.
    pred_matching_neighbors_map = count_matching_neighbors(
        lab_map=y_hard, 
        neighborhood_width=neighborhood_width
    )
    true_matching_neighbors_map = count_matching_neighbors(
        lab_map=y_true, 
        neighborhood_width=neighborhood_width
    )
    # Get the pixel-weights if we are using them.
    pixel_weights = get_uni_pixel_weights(
        y_hard, 
        uni_w_attributes=["labels", "neighbors"],
        neighborhood_width=neighborhood_width,
        ignore_index=ignore_index
        )
    return {
        "accuracy_map": accuracy_map,
        "pred_labels": pred_labels,
        "pred_matching_neighbors_map": pred_matching_neighbors_map,
        "true_matching_neighbors_map": true_matching_neighbors_map,
        "pixel_weights": pixel_weights
    }


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_edge_aux_info(
    y_hard: torch.Tensor,
    y_true: torch.Tensor,
    neighborhood_width: int,
    ignore_index: Optional[int] = None
):
    # Get the edge map.
    y_true_squeezed = y_true.squeeze()
    y_true_edge_map = get_edge_map(y_true_squeezed)

    # Get the edge regions of both the prediction and the ground truth.
    y_edge_hard = y_hard[..., y_true_edge_map].unsqueeze(-2)
    y_edge_true = y_true[..., y_true_edge_map].unsqueeze(-2)

    # Calculate the edge accuracy
    edge_acc_map = (y_edge_hard == y_edge_true).float().squeeze()

    # Keep track of different things for each bin.
    edge_pred_labels = y_edge_hard.unique().tolist()
    if ignore_index is not None and ignore_index in edge_pred_labels:
        edge_pred_labels.remove(ignore_index)

    # Get a map of which pixels match their neighbors and how often, and pixel-wise accuracy.
    # For both our prediction and the true label map.
    edge_pred_matching_neighbors_map = count_matching_neighbors(
        lab_map=y_hard, 
        neighborhood_width=neighborhood_width
    )[..., y_true_edge_map].unsqueeze(-2)

    #Get the true matching neighbors map
    edge_true_matching_neighbors_map = count_matching_neighbors(
        lab_map=y_true, 
        neighborhood_width=neighborhood_width
    )[..., y_true_edge_map].unsqueeze(-2)

    # Get the pixel-weights if we are using them.
    edge_pixel_weights = get_uni_pixel_weights(
        y_hard, 
        uni_w_attributes=["labels", "neighbors"],
        neighborhood_width=neighborhood_width,
        ignore_index=ignore_index
    )[..., y_true_edge_map].unsqueeze(-2)

    return {
        "accuracy_map": edge_acc_map,
        "pred_labels": edge_pred_labels,
        "pred_matching_neighbors_map": edge_pred_matching_neighbors_map,
        "true_matching_neighbors_map": edge_true_matching_neighbors_map,
        "pixel_weights": edge_pixel_weights 
    }