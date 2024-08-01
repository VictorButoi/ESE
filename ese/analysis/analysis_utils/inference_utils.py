#misc imports
import re
import os
import ast
import yaml
import torch
import pickle
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint
from typing import Optional, List
from pydantic import validate_arguments 
from torch.utils.data import DataLoader
# ionpy imports
from ionpy.util import Config
from ionpy.util.ioutil import autosave
from ionpy.util.config import config_digest
from ionpy.experiment.util import absolute_import, generate_tuid, eval_config
# UniverSeg imports if universeg.experiment is installed
try:
    from universeg.experiment.datasets import Segment2D
    from universeg.experiment.datasets.support import RandomSupport
except:
    pass
# local imports
from ese.utils.general import save_records, save_dict
from ...augmentation.gather import augmentations_from_config
from ...experiment.utils import load_experiment, get_exp_load_info
from ...experiment import EnsembleInferenceExperiment, BinningInferenceExperiment


def save_trackers(output_root, trackers):
    for key, tracker in trackers.items():
        if isinstance(tracker, dict):
            save_dict(tracker, output_root / f"{key}.pkl")
        elif isinstance(tracker, pd.DataFrame):
            tracker.to_pickle(output_root / f"{key}.pkl")
        else:
            save_records(tracker, output_root / f"{key}.pkl")

def aug_support(sx_cpu, sy_cpu, inference_init_obj):
    sx_cpu_np = sx_cpu.numpy() # S 1 H W
    sy_cpu_np = sy_cpu.numpy() # S 1 H W
    # Augment each member of the support set individually.
    aug_sx = []    
    aug_sy = []
    for sup_idx in range(sx_cpu_np.shape[0]):
        img_slice = sx_cpu_np[sup_idx, 0, ...]
        lab_slice = sy_cpu_np[sup_idx, 0, ...]
        # Apply the augmentation to the support set.
        aug_pair = inference_init_obj['support_transforms'](
            image=img_slice,
            mask=lab_slice
        )
        aug_sx.append(aug_pair['image'][np.newaxis, ...])
        aug_sy.append(aug_pair['mask'][np.newaxis, ...])
    # Concatenate the augmented support set.
    sx_cpu_np = np.stack(aug_sx, axis=0)
    sy_cpu_np = np.stack(aug_sy, axis=0)
    # Convert to torch tensors.
    sx_cpu = torch.from_numpy(sx_cpu_np)
    sy_cpu = torch.from_numpy(sy_cpu_np)
    # Return the augmented support sets.
    return sx_cpu, sy_cpu


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


def add_dice_loss_rows(inference_df, opts_cfg):
    # Get the rows corresponding to the Dice metric.
    dice_rows = inference_df[inference_df['image_metric'] == 'Dice']
    # Make a copy of the rows.
    dice_loss_rows = dice_rows.copy()
    # Change the metric score to 1 - metric_score.
    dice_loss_rows['metric_score'] = 1 - dice_loss_rows['metric_score']
    # Change the image metric to Dice Loss.
    dice_loss_rows['image_metric'] = 'Dice Loss'
    # If groupavg metrics are present, then change those as well.
    if 'groupavg_image_metric' in dice_loss_rows.keys():
        if opts_cfg.get('load_groupavg_metrics', False):
            # First replace Dice in the groupavg metric with 'Dice Loss'
            dice_loss_rows['groupavg_image_metric'] = dice_loss_rows['groupavg_image_metric'].str.replace('Dice', 'Dice_Loss')
            dice_loss_rows['groupavg_metric_score'] = 1 - dice_loss_rows['groupavg_metric_score']
    # Add the new rows to the inference df.
    return pd.concat([inference_df, dice_loss_rows], axis=0, ignore_index=True)


# Load the pickled df corresponding to the upper-bound of the uncalibrated UNets
def load_upperbound_df(log_cfg):
    # Load the pickled df corresponding to the upper-bound of the uncalibrated UNets
    upperbound_dir = f"{log_cfg['root']}/{log_cfg['inference_group']}/ensemble_upper_bounds/"
    # Get the runs in the upper bound dir
    try:
        run_names = os.listdir(upperbound_dir)
        for remove_dir_name in ["submitit", "debug"]:
            if remove_dir_name in run_names:
                run_names.remove(remove_dir_name)
        assert len(run_names) == 1, f"Expected 1 run in upperbound dir, found: {len(run_names)}."
        # Get the run name
        upperbound_file = upperbound_dir + f"{run_names[0]}/image_stats.pkl"
        # load the pickle
        with open(upperbound_file, 'rb') as f:
            upperbound_df = pickle.load(f)
        # Fill the column corresponding to slice_idx with string 'None'
        upperbound_df['slice_idx'] = upperbound_df['slice_idx'].fillna('None')
        # If we have a minimum number of pixels, then filter out the slices that don't have enough pixels.
        if "min_fg_pixels" in log_cfg:
            # Get the names of all columns that have "num_lab" in them.
            num_lab_cols = [col for col in upperbound_df.columns if "num_lab" in col]
            # Remove num_lab_0_pixels because it is background
            num_lab_cols.remove("num_lab_0_pixels")
            # Make a new column that is the sum of all the num_lab columns.
            upperbound_df['num_fg_pixels'] = upperbound_df[num_lab_cols].sum(axis=1)
            upperbound_df = upperbound_df[upperbound_df['num_fg_pixels'] >= log_cfg["min_fg_pixels"]]
        # Add the dice loss rows if we want to.
        if log_cfg['add_dice_loss_rows']:
            upperbound_df = add_dice_loss_rows(upperbound_df)
        # Return the upperbound df
        return upperbound_df
    except Exception as e:
        print(f"Error loading upperbound df: {e}")
        return None


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
    

def get_average_unet_baselines(
    total_df: pd.DataFrame,
    per_calibrator: bool = True,
    num_seeds: Optional[int] = None,
    group_metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    # Collect all the individual networks.
    unet_info_df = total_df[total_df['ensemble'] == False].reset_index(drop=True)
    # These are the keys we want to group by.
    unet_group_keys = [
       'data_id',
       'slice_idx',
       'split',
       'normalize',
       'cal_stats_split',
       'joint_data_slice_id',
       'ensemble',
       'model_class',
       'model_type',
       'calibrator',
       '_pretrained_class',
       'image_metric',
       'metric_type',
       'groupavg_image_metric',
    ]
    # Only keep the keys that are actually in the columns of unet_info_df.
    unet_group_keys = [key for key in unet_group_keys if key in unet_info_df.keys()]
    # Assert that none of our grouping columns have NaNs in them.
    for unet_key in unet_group_keys:
        assert unet_info_df[unet_key].isna().sum() == 0, f"Found NaNs in {unet_key} column."
    # Run a check, that when you group by these keys, you get a unique row.
    num_rows_per_group = unet_info_df.groupby(unet_group_keys).size()
    # If num seeds is provided then we need to match it exactly (sanity check).
    if num_seeds is not None:
        assert (num_rows_per_group.max() == num_seeds) and (num_rows_per_group.min() == num_seeds),\
            f"Grouping by these keys does not give the required number of rows per seed ({num_seeds})."\
                + f" Got max {num_rows_per_group.max()} and min {num_rows_per_group.min()}. Rows look like: {num_rows_per_group}."
    # Group everything we need. 
    total_group_metrics = ['metric_score', 'groupavg_metric_score']
    if group_metrics is not None:
        total_group_metrics += group_metrics
    total_group_metrics = [gm for gm in total_group_metrics if gm in unet_info_df.keys()]
    
    average_seed_unet = unet_info_df.groupby(unet_group_keys).agg({met_name: 'mean' for met_name in total_group_metrics}).reset_index()
    # Set some useful variables.
    average_seed_unet['experiment.pretrained_seed'] = 'Average'
    average_seed_unet['ensemble_hash'] = 'Average' # Now this is a group of results
    average_seed_unet['pretrained_seed'] = 'Average'
    average_seed_unet['model_type'] = 'group' # Now this is a group of results
    average_seed_unet['method_name'] = 'Average UNet' # Now this is a group of results

    if not per_calibrator:
        average_seed_unet['calibrator'] = 'None'

    def configuration(method_name, calibrator):
        return f"{method_name}_{calibrator}"

    average_seed_unet.augment(configuration)

    return average_seed_unet


def cal_stats_init(
    cfg_dict,
    yaml_cfg_dir: Optional[str] = None
):
    cal_init_obj_dict = {}
    ###################
    # BUILD THE MODEL #
    ###################
    inference_exp, inference_cfg, save_root = load_inference_exp_from_cfg(
        inference_cfg=cfg_dict
    )
    inference_exp.to_device()
    cal_init_obj_dict['exp'] = inference_exp
    exp_total_config = inference_exp.config.to_dict()

    #####################
    # BUILD THE DATASET #
    #####################
    # Rebuild the experiments dataset with the new cfg modifications.
    inference_dset_options = inference_cfg['inference_data'].copy()
    input_type = inference_dset_options.pop("input_type")
    assert input_type in ["volume", "image"], f"Data type {input_type} not supported."
    # Build the dataloaders.
    training_data_cfg = exp_total_config['data']
    data_objs = dataloader_from_exp( 
        inf_data_cfg=inference_dset_options, 
        dataloader_cfg=inference_cfg['dataloader'],
        aug_cfg_list=inference_cfg.get('support_augmentations', None)
    )
    # Update important keys in the inference cfg.
    inference_cfg['train'] = exp_total_config['train']
    inference_cfg['loss_func'] = exp_total_config['loss_func']
    inference_cfg['training_data'] = training_data_cfg
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

    #####################
    # SAVE THE METADATA #
    #####################
    task_root = save_inference_metadata(
        cfg_dict=inference_cfg,
        save_root=save_root
    )

    # Compile everything into a dictionary.
    cal_init_obj_dict = {
        "data_counter": 0,
        "dloaders": data_objs['dataloaders'],
        "output_root": task_root,
        "supports": data_objs['supports'],
        "support_transforms": None, # This is set later.
        "trackers": trackers,
        **cal_init_obj_dict
    }

    # We can also add augmentation at inference to boost performance.
    support_aug_cfg = inference_cfg['experiment'].get('support_augs', None)
    if support_aug_cfg is not None and len(support_aug_cfg) > 0:
        # Open the yaml file corresponding to the augmentations
        with open(f"{yaml_cfg_dir}/ese/experiment/configs/inference/aug_cfg_bank.yaml", 'r') as f:
            aug_cfg_almanac = yaml.safe_load(f)
        aug_dict_list = []
        for sup_aug in support_aug_cfg:
            assert sup_aug in aug_cfg_almanac.keys(),\
                f"Augmentation must be defined in the yaml file. Got target aug: {sup_aug} and support is for keys: {aug_cfg_almanac.keys()}."   
            aug_dict_list.append(aug_cfg_almanac[sup_aug])
        cal_init_obj_dict['support_transforms'] = augmentations_from_config(aug_dict_list)
    
    print(f"Running:\n\n{str(yaml.safe_dump(Config(inference_cfg)._data, indent=0))}")
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
    if inf_model_cfg.get('ensemble', False) and inf_model_cfg['_type'] != "incontext":
        inference_exp = EnsembleInferenceExperiment.from_config(inference_cfg)
        save_root = Path(inference_exp.path)
    elif "Binning" in inference_cfg['calibrator']['_name']:
        inference_exp = BinningInferenceExperiment.from_config(inference_cfg)
        save_root = Path(inference_exp.path)
    else:
        load_exp_args = {
            "checkpoint": inf_model_cfg['checkpoint'],
            "load_data": False,
            "set_seed": False,
            **get_exp_load_info(inference_cfg['experiment']['model_dir']),
        }
        if "_attr" in inf_model_cfg:
            load_exp_args['attr_dict'] = inf_model_cfg['_attr']
        # Load the experiment directly if you give a sub-path.
        inference_exp = load_experiment(**load_exp_args)
        save_root = None

    # Make a new value for the pretrained seed, so we can differentiate between
    # members of ensemble
    old_inference_cfg = inference_exp.config.to_dict()
    inference_cfg['experiment']['pretrained_seed'] = old_inference_cfg['experiment']['seed']
    # Update the model cfg to include old model cfg.
    inference_cfg['model'].update(old_inference_cfg['model']) # Ideally everything the same but adding new keys.
    return inference_exp, inference_cfg, save_root


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
        lab_d_support_dict = None
        # Package these into dummy dictionaries. Not great, but it works.
        lab_d_dataset_dict = {
            -1 : d_dataset_obj
        }
        # We need to store these object by the contents of d_cfg_opt.
        opt_string = "^".join([f"{key}:{val}" for key, val in d_cfg_opt.items()])
        supports[opt_string] = lab_d_support_dict
        dataloaders[opt_string] = {}
        for lab in lab_d_dataset_dict.keys():
            # Build the dataloader for this opt cfg and label.
            dataloaders[opt_string][lab] = DataLoader(
                lab_d_dataset_dict[lab], 
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