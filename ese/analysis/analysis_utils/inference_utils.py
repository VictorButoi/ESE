''#misc imports
import re
import os
import ast
import yaml
import torch
import pickle
import voxynth
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint
from typing import Optional, List
from pydantic import validate_arguments 
from torch.utils.data import DataLoader
# ionpy imports
# ionpy imports
from ionpy.util import Config
from ionpy.util.ioutil import autosave
from ionpy.util.config import config_digest, HDict, valmap
from ionpy.experiment.util import absolute_import, generate_tuid, eval_config
# local imports
from ese.utils.general import save_records, save_dict
from ...augmentation.pipeline import build_aug_pipeline
from ...augmentation.gather import augmentations_from_config
from ...experiment.utils import load_experiment, get_exp_load_info


def list2tuple(val):
    if isinstance(val, list):
        return tuple(map(list2tuple, val))
    return val


def save_trackers(output_root, trackers):
    for key, tracker in trackers.items():
        if isinstance(tracker, dict):
            save_dict(tracker, output_root / f"{key}.pkl")
        elif isinstance(tracker, pd.DataFrame):
            tracker.to_pickle(output_root / f"{key}.pkl")
        else:
            save_records(tracker, output_root / f"{key}.pkl")


def verify_graceful_exit(log_path: str, log_root: str):
    submitit_dir = os.path.join(log_root, log_path, "submitit")
    # Check that the submitit directory exists if it doesnt then return.
    try:
        unique_logs = list(set([logfile.split("_")[0] for logfile in os.listdir(submitit_dir)]))
    except:
        print(f"Error loading submitit directory: {submitit_dir}")
        return
    # Check that all the logs have a success result.
    for log_name in unique_logs:
        result_log_file = os.path.join(submitit_dir, f"{log_name}_0_result.pkl")
        try:
            with open(result_log_file, 'rb') as f:
                result = pickle.load(f)[0]
            if result != 'success':
                raise ValueError(f"Found non-success result in file {result_log_file}: {result}.")
        except Exception as e:
            print(f"Error loading result log file: {e}")


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
def save_inference_metadata(cfg_dict, save_root: Optional[Path] = None):
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
    

def cal_stats_init(cfg_dict):

    ###################
    # BUILD THE MODEL #
    ###################
    inference_exp, inference_cfg = load_inference_exp_from_cfg(
        inference_cfg=cfg_dict
    )
    inference_exp.to_device()
    inference_exp_total_cfg_dict = inference_exp.config.to_dict()
    # Update important keys in the inference cfg.
    inference_cfg['train'] = inference_exp_total_cfg_dict['train']
    inference_cfg['loss_func'] = inference_exp_total_cfg_dict['loss_func']
    inference_cfg['training_data'] = inference_exp_total_cfg_dict['data'] 

    #####################
    # BUILD THE DATASET #
    #####################
    # Rebuild the experiments dataset with the new cfg modifications.
    inference_data_cfg = inference_cfg['inference_data'].copy()
    input_type = inference_data_cfg.pop("input_type", "image")
    assert input_type in ["volume", "image"], f"Data type {input_type} not supported."
    # Build the dataloaders.
    data_objs = dataloader_from_exp( 
        inf_data_cfg=inference_data_cfg, 
        dataloader_cfg=inference_cfg['dataloader'],
        aug_cfg_list=inference_cfg.get('support_augmentations', None)
    )
    #############################
    trackers = {
        "image_stats": [],
    }
    if cfg_dict['log']['log_pixel_stats']:
        trackers.update({
        "tl_pixel_meter_dict": {},
        "cw_pixel_meter_dict": {}
        })
        # Add trackers per split
        for data_cfg_opt in data_objs['dataloaders']:
            trackers['tl_pixel_meter_dict'][data_cfg_opt] = {}
            trackers['cw_pixel_meter_dict'][data_cfg_opt] = {}

    ###########################################################
    # Build the augmentation pipeline if we want augs on GPU. #
    ###########################################################
    inf_norm_augs = None
    if 'augmentations' in inference_exp_total_cfg_dict.keys():
        if 'visual' in inference_exp_total_cfg_dict['augmentations'].keys():
            visual_aug_cfg = inference_exp_total_cfg_dict['augmentations']['visual']
            inf_norm_augs = {
                "visual": {exp_key: exp_val for exp_key, exp_val in visual_aug_cfg.items() if 'normalize' in exp_key}
            }
            # It's possible we used other visual augs that weren't normalization.
            if len(inf_norm_augs['visual'].keys()) == 0:
                inf_norm_augs = None
    # Assemble the augmentation pipeline.
    if ('inference_augmentations' in inference_cfg.keys()) or (inf_norm_augs is not None):
        inference_augs = inference_cfg.get('inference_augmentations', {})
        inference_augs.update(inf_norm_augs)
        # Update the inference cfg with the new augmentations.
        inference_cfg['inference_augmentations'] = inference_augs
        # Place the augmentation function into our inference object.
        aug_pipeline = build_aug_pipeline(inference_augs)
    else:
        aug_pipeline = None

    #####################
    # SAVE THE METADATA #
    #####################
    task_root = save_inference_metadata(inference_cfg)
    print(f"Running:\n\n{str(yaml.safe_dump(Config(inference_cfg)._data, indent=0))}")

    # Compile everything into a dictionary.
    cal_init_obj_dict = {
        "data_counter": 0,
        "exp": inference_exp,
        "trackers": trackers,
        "aug_pipeline": aug_pipeline,
        "output_root": task_root,
        "support_transforms": None, # This is set later.
        "dloaders": data_objs['dataloaders'],
    }

    ##################################
    # INITIALIZE THE QUALITY METRICS #
    ##################################
    qual_metrics_dict = {}
    if 'qual_metrics' in inference_cfg.keys():
        for q_met_cfg in inference_cfg['qual_metrics']:
            q_metric_name = list(q_met_cfg.keys())[0]
            quality_metric_options = q_met_cfg[q_metric_name]
            metric_type = quality_metric_options.pop("metric_type")
            if 'from_logits' in quality_metric_options.keys():
                assert not quality_metric_options['from_logits'], "Quality metrics must be computed on probabilities."
            # Add the quality metric to the dictionary.
            qual_metrics_dict[q_metric_name] = {
                "name": q_metric_name,
                "_fn": eval_config(quality_metric_options),
                "_type": metric_type
            }
    # Place these dictionaries into the config dictionary.
    inference_cfg['qual_metrics'] = qual_metrics_dict

    ##################################
    # INITIALIZE CALIBRATION METRICS #
    ##################################
    # Image level metrics.
    if inference_cfg.get('image_cal_metrics', None) is not None:
        inference_cfg['image_cal_metrics'] = preload_calibration_metrics(
            base_cal_cfg=inference_cfg['local_calibration'],
            cal_metrics_dict=inference_cfg['image_cal_metrics']
        )
    # Global dataset level metrics. (Used for validation)
    if inference_cfg.get('global_cal_metrics', None) is not None:
        inference_cfg['global_cal_metrics'] = preload_calibration_metrics(
            base_cal_cfg=inference_cfg['global_calibration'],
            cal_metrics_dict=inference_cfg['global_cal_metrics']
        )
        
    # Return a dictionary of the components needed for the calibration statistics.
    return cal_init_obj_dict


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def load_inference_exp_from_cfg(inference_cfg): 
    inf_model_cfg = inference_cfg['model']

    # Get the configs of the experiment
    load_exp_args = {
        "checkpoint": inf_model_cfg['checkpoint'],
        "exp_kwargs": {
            "set_seed": False,
            "load_data": False,
            "load_aug_pipeline": False
        },
        **get_exp_load_info(inference_cfg['experiment']['model_dir']),
    }
    if "_attr" in inf_model_cfg:
        load_exp_args['attr_dict'] = inf_model_cfg['_attr']

    # Load the experiment directly if you give a sub-path.
    inference_exp = load_experiment(**load_exp_args)
    # Make a new value for the pretrained seed, so we can differentiate between
    # members of ensemble
    old_inference_cfg = inference_exp.config.to_dict()
    inference_cfg['experiment']['pretrained_seed'] = old_inference_cfg['experiment']['seed']
    # Update the model cfg to include old model cfg.
    inference_cfg['model'].update(old_inference_cfg['model']) # Ideally everything the same but adding new keys.
    return inference_exp, inference_cfg


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def dataloader_from_exp(
    inf_data_cfg, 
    dataloader_cfg,
    aug_cfg_list: Optional[List[dict]] = None,
    new_dset_options: Optional[dict] = None, # This is a dictionary of options to update the dataset with.
):
    if new_dset_options is not None:
        inf_data_cfg.update(new_dset_options)
    # Make sure we aren't sampling for evaluation. 
    if "slicing" in inf_data_cfg.keys():
        assert inf_data_cfg['slicing'] not in ['central', 'dense', 'uniform'], "Sampling methods not allowed for evaluation."
    # Get the dataset class and build the transforms
    dataset_cls = inf_data_cfg.pop('_class')

    # TODO: BACKWARDS COMPATIBILITY STOPGAP
    dataset_cls = dataset_cls.replace("ese.experiment", "ese")

    # TODO: Clean this up, way too hardcoded.
    # Drop auxiliary information used for making the models.
    for drop_key in [
        'in_channels', 
        'out_channels', 
        'iters_per_epoch', 
        'input_type',
        'add_aug',
        'return_dst_to_bdry',
        'train_splits',
        'train_datasets',
        'val_splits',
        'val_datasets'
    ]:
        if drop_key in inf_data_cfg.keys():
            inf_data_cfg.pop(drop_key)
    # Ensure that we return the different data ids.
    inf_data_cfg['return_data_id'] = True

    # We want to iterate through the keys of the inference data cfg, check which match the form of a tuple, and then use those
    # as options for the dataset.

    def is_tuple_string(s):
        # Regular expression to check if the string represents a tuple
        s = str(s)
        tuple_regex = re.compile(r'^\(\s*([^,]+,\s*)*[^,]+\s*\)$')
        return bool(tuple_regex.match(s))

    # Iterate through the keys and check if they are tuples.
    opt_dict = {}
    for key in list(inf_data_cfg.keys()):
        if is_tuple_string(inf_data_cfg[key]):
            opt_dict[key] = list(ast.literal_eval(inf_data_cfg.pop(key)))
    # Get the cartesian product of the options in the dictionary using itertools.
    data_cfg_vals = opt_dict.values()
    # Create a list of dictionaries from the Cartesian product
    data_cfgs = [dict(zip(opt_dict.keys(), prod)) for prod in itertools.product(*data_cfg_vals)]

    # Iterate through the configurations and construct both 
    # 1) The dataloaders corresponding to each set of examples for inference.
    # 2) The support sets for each run configuration.
    dataloaders = {}
    supports = {} # use for ICL
    for d_cfg_opt in data_cfgs:
        # Load the dataset with modified arguments.
        d_data_cfg = inf_data_cfg.copy()
        # Update the data cfg with the new options.
        d_data_cfg.update(d_cfg_opt)
        # Construct the dataset, either if it's incontext or standard.
        d_dataset_obj = absolute_import(dataset_cls)(
            transforms=augmentations_from_config(aug_cfg_list) if aug_cfg_list is not None else None, 
            **d_data_cfg
        )
        # We need to store these object by the contents of d_cfg_opt.
        opt_string = "^".join([f"{key}:{val}" for key, val in d_cfg_opt.items()])
        # Build the dataloader for this opt cfg and label.
        dataloaders[opt_string] = DataLoader(
            d_dataset_obj, 
            batch_size=dataloader_cfg['batch_size'], 
            num_workers=dataloader_cfg['num_workers'],
            shuffle=False,
            drop_last=False
        )
    # Build a dictionary of our data objs
    data_obj_dict = {
        "dataloaders": dataloaders,
        "supports": supports,
    }
    # Return the dataloaders and the modified data cfg.
    return data_obj_dict


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def preload_calibration_metrics(
    base_cal_cfg,
    cal_metrics_dict: List[dict] 
):
    cal_metrics = {}
    for c_met_cfg in cal_metrics_dict:
        c_metric_name = list(c_met_cfg.keys())[0]
        calibration_metric_options = c_met_cfg[c_metric_name]
        cal_base_cfg_copy = base_cal_cfg.copy()
        # Update with the inference set of calibration options.
        cal_base_cfg_copy.update(calibration_metric_options)
        # Add the calibration metric to the dictionary.
        cal_metrics[c_metric_name] = {
            "_fn": eval_config(cal_base_cfg_copy),
            "name": c_metric_name,
            "cal_type": c_met_cfg[c_metric_name]["cal_type"]
        }
    return cal_metrics


def save_preds(output_dict, output_root):
    # Make a copy of the output dict.
    np.savez(output_root / 'preds.npz', **output_dict)


