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
from ionpy.util.ioutil import autosave
from ionpy.util import Config, dict_product
from ionpy.util.config import config_digest
from ionpy.experiment.util import absolute_import, generate_tuid, eval_config
# local imports
from ese.utils.general import save_records, save_dict
from ...augmentation.pipeline import build_aug_pipeline
from ...augmentation.gather import augmentations_from_config
from ...experiment.utils import load_experiment, get_exp_load_info
from ...datasets.support import RandomSupport


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
        result_pickl_files = [logfile for logfile in os.listdir(submitit_dir) if logfile.endswith("_result.pkl")]
        unique_logs = list(set([logfile.split("_")[0] for logfile in result_pickl_files]))
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
    

def cal_stats_init(inference_cfg):

    ###################
    # BUILD THE MODEL #
    ###################
    inference_exp = load_inference_exp(
        model_dir=inference_cfg['experiment']['model_dir'],
        checkpoint=inference_cfg['model']['checkpoint'],
        to_device=True
    )

    # Update important keys in the inference cfg.
    inference_exp_total_cfg_dict = inference_exp.config.to_dict()
    inference_cfg['train'] = inference_exp_total_cfg_dict['train']
    inference_cfg['loss_func'] = inference_exp_total_cfg_dict['loss_func']
    inference_cfg['training_data'] = inference_exp_total_cfg_dict['data'] 
    inference_cfg['experiment']['pretrained_seed'] = inference_exp_total_cfg_dict['experiment']['seed']

    # Update the model cfg to include old model cfg.
    inference_cfg['model'].update(inference_exp_total_cfg_dict['model']) # Ideally everything the same but adding new keys.

    #####################
    # BUILD THE DATASET #
    #####################
    # Rebuild the experiments dataset with the new cfg modifications.
    inference_data_cfg = inference_cfg['inference_data'].copy()
    input_type = inference_data_cfg.pop("input_type", "image")
    assert input_type in ["volume", "image"], f"Data type {input_type} not supported."
    # Build the dataloaders.
    loaded_inf_data_cfg, dataobj_dict = dataobjs_from_exp( 
        inf_data_cfg=inference_data_cfg, 
        dataloader_cfg=inference_cfg['dataloader'],
        aug_cfg_list=inference_cfg.get('support_augmentations', None)
    )
    # Update the inference_data_cfg to reflect the data we are running on.
    inference_cfg['inference_data'] = loaded_inf_data_cfg 

    #############################
    trackers = {
        "image_stats": [],
    }
    if inference_cfg['log']['log_pixel_stats']:
        trackers.update({
        "tl_pixel_meter_dict": {},
        "cw_pixel_meter_dict": {}
        })
        # Add trackers per split
        for data_cfg_opt in dataobj_dict:
            trackers['tl_pixel_meter_dict'][data_cfg_opt] = {}
            trackers['cw_pixel_meter_dict'][data_cfg_opt] = {}

    ###########################################################
    # Build the augmentation pipeline if we want augs on GPU. #
    ###########################################################
    inf_norm_augs = None
    if 'augmentations' in inference_exp_total_cfg_dict.keys():
        aug_obj = inference_exp_total_cfg_dict['augmentations']
        if isinstance(aug_obj, dict) and 'visual' in aug_obj.keys():
            visual_aug_cfg = aug_obj['visual']
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
        "dataobjs": dataobj_dict,
        "output_root": task_root,
        "aug_pipeline": aug_pipeline,
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
def load_inference_exp(
    model_dir,
    checkpoint: str = "max-val-dice_score",
    to_device: bool = False,
    inf_kwargs: Optional[dict] = {},
): 
    # If we are passing to the device, we need to set the 'device' of
    # our init to 'gpu'.
    if to_device:
        inf_kwargs['device'] = 'cuda'

    # Get the configs of the experiment
    load_exp_args = {
        "checkpoint": checkpoint,
        "exp_kwargs": {
            "set_seed": False,
            "load_data": False,
            "load_aug_pipeline": False
        },
        **inf_kwargs,
        **get_exp_load_info(model_dir),
    }

    # Load the experiment directly if you give a sub-path.
    inference_exp = load_experiment(**load_exp_args)

    # Optionally, move the model to the device.
    if to_device:
        inference_exp.to_device()

    return inference_exp


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def dataobjs_from_exp(
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

    # Get the dataset class and sometimes the support size.
    dataset_cls_str = inf_data_cfg.pop('_class')
    dset_cls = absolute_import(dataset_cls_str)
    # Some information about supports.
    support_size = inf_data_cfg.pop('support_size', None)
    support_split = inf_data_cfg.pop('support_split', 'train')

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
        'val_datasets',
        'samples_per_epoch',
        'return_data_id' 
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
    d_cfg_data_objs = {}
    # Iterate through the data configurations.
    for d_cfg_opt in data_cfgs:
        # Load the dataset with modified arguments.
        d_data_cfg = inf_data_cfg.copy()
        # Update the data cfg with the new options.
        d_data_cfg.update(d_cfg_opt)
        # We need to store these object by the contents of d_cfg_opt.
        opt_string = "^".join([f"{key}:{val}" for key, val in d_cfg_opt.items()])
        d_cfg_data_objs[opt_string] = {}

        # Make a dictionary for how we construct the dataset. (important order because we want to be able to modify)
        dset_args = {
            "transforms": augmentations_from_config(aug_cfg_list) if aug_cfg_list is not None else None, 
            **d_data_cfg
        }
        # Build the dataloader for this opt cfg and label.
        d_cfg_data_objs[opt_string]['dloader'] = DataLoader(
            dset_cls(**dset_args), 
            batch_size=dataloader_cfg['batch_size'], 
            num_workers=dataloader_cfg['num_workers'],
            shuffle=False,
            drop_last=False
        )
        if dataset_cls_str == 'ese.datasets.Segment2D':
            support_dset_args = dset_args.copy()
            # Important settings for inference.
            support_dset_args['split'] = support_split
            support_dset_args['return_data_id'] = False
            # Build the support sampler.
            d_cfg_data_objs[opt_string]['support'] = RandomSupport(
                dset_cls(**support_dset_args), 
                support_size=support_size, 
                return_data_ids=False,
                replacement=True, 
            )
    
    # Modify the inference data cfg to reflect the new data objects.
    modified_inf_data_cfg = inf_data_cfg.copy()
    modified_inf_data_cfg['_class'] = dataset_cls_str
    if support_size is not None:
        modified_inf_data_cfg['support_size'] = support_size
        modified_inf_data_cfg['support_split'] = support_split

    # Return the dataloaders and the modified data cfg.
    return modified_inf_data_cfg, d_cfg_data_objs
    


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

def soft_abs_area_estimation_error(soft_volume, gt_volume):
    return np.abs(soft_volume - gt_volume)

def hard_abs_area_estimation_error(hard_volume, gt_volume):
    return np.abs(hard_volume - gt_volume)

def soft_RAVE(soft_volume, gt_volume):
    return np.abs(soft_volume - gt_volume) / gt_volume

def hard_RAVE(hard_volume, gt_volume):
    return np.abs(hard_volume - gt_volume) / gt_volume

def log_soft_RAVE(soft_RAVE):
    return np.log(soft_RAVE + 0.01)

def log_hard_RAVE(hard_RAVE):
    return np.abs(hard_RAVE + 0.01)

def add_vol_error_keys(inference_df):
    # Base Metrics
    inference_df.augment(soft_abs_area_estimation_error)
    inference_df.augment(hard_abs_area_estimation_error)
    # Relative Metrics
    inference_df.augment(soft_RAVE)
    inference_df.augment(hard_RAVE)
    # Log Metrics
    # RAVE
    inference_df.augment(log_soft_RAVE)
    inference_df.augment(log_hard_RAVE)


def resize_image(resize_cfg, image):
    # Get the original resolution of the image and write that we did resizing to the metadata.
    new_img_res = resize_cfg['size']
    # Resize the image
    interpolate_args = {
        "input": image, 
        "size": new_img_res, 
        "mode": resize_cfg['mode'], # this one is ok to be bilinear interpolation becuase this is what we trained on.
    }
    if resize_cfg['mode'] in ['linear', 'bilinear', 'bicubic', 'trilinear']:
        interpolate_args['align_corners'] = resize_cfg['align_corners']
    # Resize the image or return if it's already the right size.
    if new_img_res != image.shape[-2:]:
        return torch.nn.functional.interpolate(**interpolate_args)
    else:
        return image


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def select_batch_by_inds(
    batch, 
    valid_inds
):
    subselect_batch = {}
    # We want to iterate through the keys, and if the key is a torch.Tensor or np.ndarray, then we want to select
    # the indices that are valid.
    for ikey in batch.keys():
        if isinstance(batch[ikey], (torch.Tensor, np.ndarray)):
            if len(valid_inds) == 1:
                if valid_inds[0]:
                    subselect_batch[ikey] = batch[ikey]
                else:
                    subselect_batch[ikey] = np.array([])
            else:
                subselect_batch[ikey] = batch[ikey][valid_inds]
        else:
            subselect_batch[ikey] = batch[ikey]
    # Return the filtered batch. 
    return subselect_batch 


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def filter_by_min_lab(
    label_map, 
    min_pix: int = 0
):
    # We want to sum over everything except the batch dimension.
    valid_indices = torch.sum(label_map != 0, tuple(range(1, len(label_map.shape)))) >= min_pix
    #  Return the valid indices.
    return valid_indices.cpu()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_kwarg_sweep(
    inf_cfg_dict: dict
):
    # If there are inference kwargs, then we need to do a forward pass with those kwargs.
    inf_kwarg_opts = inf_cfg_dict['experiment'].get('inf_kwargs', None)
    if inf_kwarg_opts is not None:
        # Go through each, and if they are strings representing tuples, then we need to convert them to tuples.
        for inf_key, inf_val in inf_kwarg_opts.items(): 
            if isinstance(inf_val, str):
                lit_eval_val = ast.literal_eval(inf_val)
                if isinstance(lit_eval_val, tuple):
                    inf_kwarg_opts[inf_key] = list(lit_eval_val)
            # Ensure this is a list.
            new_inf_val = inf_kwarg_opts[inf_key]
            if not isinstance(new_inf_val, list):
                inf_kwarg_opts[inf_key] = [new_inf_val]
        # Now we need to do a grid of the options, similar to how we build configs.
        return list(dict_product(inf_kwarg_opts))
    else:
        return [{}]