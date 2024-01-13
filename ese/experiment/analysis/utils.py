#misc imports
import os
import torch
import pandas as pd
from pathlib import Path
from typing import Optional
from pydantic import validate_arguments 
from torch.utils.data import DataLoader
# ionpy imports
from ionpy.analysis import ResultsLoader
from ionpy.util.ioutil import autosave
from ionpy.util.config import config_digest
from ionpy.experiment.util import absolute_import, generate_tuid, eval_config
# local imports
from ..models.ensemble import get_combine_fn
from ..callbacks.visualize import ShowPredictionsCallback
from ..experiment.utils import load_experiment, process_pred_map
from ..experiment import EnsembleInferenceExperiment
from ..metrics.utils import count_matching_neighbors 


def preload_calibrator_classes(
        base_calibration_cfg: dict, 
        cal_metrics_dict: dict
):
    cal_metrics = {}
    for c_met_cfg in cal_metrics_dict:
        c_metric_name = list(c_met_cfg.keys())[0]
        calibration_metric_options = c_met_cfg[c_metric_name]
        calibrator_config = base_calibration_cfg.copy()
        # Update with the inference set of calibration options.
        calibrator_config.update(calibration_metric_options)
        # Add the calibration metric to the dictionary.
        cal_metrics[c_metric_name] = {
            "name": c_metric_name,
            "_fn": eval_config(c_met_cfg[c_metric_name])
        }
    return cal_metrics


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
    inference_exp, 
    new_dset_options=None, 
    batch_size=1,
    num_workers=1
):
    inference_data_cfg = inference_exp.config['data'].to_dict()
    if new_dset_options is not None:
        inference_data_cfg.update(new_dset_options)
    # Make sure we aren't sampling for evaluation. 
    if "slicing" in inference_data_cfg.keys():
        assert inference_data_cfg["slicing"] not in ["central", "dense", "uniform"], "Sampling methods not allowed for evaluation."
    # Get the dataset class and build the transforms
    dataset_cls = inference_data_cfg.pop("_class")
    # Drop auxiliary information used for making the models.
    for drop_key in ["in_channels", "out_channels", "iters_per_epoch", "input_type"]:
        if drop_key in inference_data_cfg.keys():
            inference_data_cfg.pop(drop_key)
    # Ensure that we return the different data ids.
    inference_data_cfg["return_data_id"] = True
    # Load the dataset with modified arguments.
    dataset_obj = absolute_import(dataset_cls)(**inference_data_cfg)
    inference_data_cfg["_class"] = dataset_cls        
    # Build the dataset and dataloader.
    dataloader = DataLoader(
        dataset_obj, 
        batch_size=batch_size, 
        num_workers=num_workers,
        shuffle=False
    )
    return dataloader, inference_data_cfg


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def load_inference_exp_from_cfg(
    inference_cfg: dict
): 
    model_cfg = inference_cfg['model']
    pretrained_exp_root = model_cfg['pretrained_exp_root']
    is_exp_group = not ("config.yml" in os.listdir(pretrained_exp_root)) 
    # Get the configs of the experiment
    if model_cfg['ensemble']:
        assert is_exp_group, "Ensemble inference only works with experiment groups."
        assert 'ensemble_combine_fn' in model_cfg.keys(), "Ensemble inference requires a combine function."
        inference_exp = EnsembleInferenceExperiment.from_config(inference_cfg)
        save_root = Path(inference_exp.path)
    else:
        rs = ResultsLoader()
        # If the experiment is a group, then load the configs and build the experiment.
        if is_exp_group: 
            dfc = rs.load_configs(
                pretrained_exp_root,
                properties=False,
            )
            inference_exp = load_experiment(
                df=rs.load_metrics(dfc),
                checkpoint=model_cfg['checkpoint'],
                selection_metric=model_cfg['pretrained_select_metric'],
                load_data=False
            )
        # Load the experiment directly if you give a sub-path.
        else:
            inference_exp = load_experiment(
                path=pretrained_exp_root,
                checkpoint=model_cfg['checkpoint'],
                load_data=False
            )
        save_root = None
    # Make a new value for the pretrained seed, so we can differentiate between
    # members of ensemble
    old_inference_cfg = inference_exp.config.to_dict()
    inference_cfg['experiment']['pretrained_seed'] = old_inference_cfg['experiment']['seed']
    # Update the model cfg to include old model cfg.
    inference_cfg['model'].update(old_inference_cfg['model']) # Ideally everything the same but adding new keys.
    return inference_exp, save_root


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def save_inference_metadata(
    cfg_dict: dict,
    save_root: Optional[Path] = None
    ):
    if save_root is None:
        save_root = Path(cfg_dict['log']['root'])
        # Prepare the output dir for saving the results
        create_time, nonce = generate_tuid()
        digest = config_digest(cfg_dict)
        uuid = f"{create_time}-{nonce}-{digest}"
        path = save_root / uuid
        # Save the metadata if the path isn't defined yet.
        metadata = {"create_time": create_time, "nonce": nonce, "digest": digest}
        autosave(metadata, path / "metadata.json")
    else:
        path = save_root
    # Save the config.
    autosave(cfg_dict, path / "config.yml")
    return path
    

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
) -> dict:
    assert y_hard.dim() == 4, f"Expected 4D tensor for y_hard, found shape: {y_hard.shape}."
    assert y_true.dim() == 4, f"Expected 4D tensor for y_true, found shape: {y_true.shape}."
    # Get the pixelwise accuracy.
    accuracy_map = (y_hard == y_true).squeeze(1).float()

    # Keep track of different things for each bin.
    pred_labels = y_hard.unique().tolist()
    if ignore_index is not None and ignore_index in pred_labels:
        pred_labels.remove(ignore_index)

    # Get a map of which pixels match their neighbors and how often, and pixel-wise accuracy.
    # For both our prediction and the true label map.
    pred_matching_neighbors_map = count_matching_neighbors(
        lab_map=y_hard.squeeze(1), # Remove the channel dimension. 
        neighborhood_width=neighborhood_width
    )
    true_matching_neighbors_map = count_matching_neighbors(
        lab_map=y_true.squeeze(1), # Remove the channel dimension. 
        neighborhood_width=neighborhood_width
    )
    return {
        "accuracy_map": accuracy_map,
        "pred_labels": pred_labels,
        "pred_matching_neighbors_map": pred_matching_neighbors_map,
        "true_matching_neighbors_map": true_matching_neighbors_map,
    }


def show_inference_examples(
    output_dict: dict,
    inference_cfg: dict,
):
    ShowPredictionsCallback(
        output_dict, 
        softpred_dim=1,
        from_logits=inference_cfg["model"]["ensemble"] # If ensemble, need to do a softmax. over ensemble members.
        )
    # If we are showing examples with an ensemble, then we
    # returned initially the individual predictions.
    if inference_cfg["model"]["ensemble"]:
        # y_pred is the wrong shape, add back the channel dimension and shuffle the ensemble dimension.
        output_dict["y_pred"] = output_dict["y_pred"].permute(1, 0, 2, 3).unsqueeze(0) # B, C, E, H, W
        # Combine the outputs of the models.
        ensemble_prob_map = get_combine_fn(inference_cfg["model"]["ensemble_combine_fn"])(
            output_dict["y_pred"], 
            pre_softmax=inference_cfg["model"]["ensemble_pre_softmax"]
            )
        # Get the hard prediction and probabilities, if we are doing identity,
        # then we don't want to return probs.
        ensemble_prob_map, ensemble_pred_map = process_pred_map(
            ensemble_prob_map, 
            multi_class=True, 
            threshold=0.5,
            from_logits=False, # Ensemble methods already return probs.
            )
        # Place the ensemble predictions in the output dict.
        output_dict["y_pred"] = ensemble_prob_map
        output_dict["y_hard"] = ensemble_pred_map
        # Finally, show the ensemble combination.
        ShowPredictionsCallback(
            output_dict, 
            softpred_dim=1,
            from_logits=False
            )