#misc imports
import os
import ast
import yaml
import pickle
import pandas as pd
from pathlib import Path
from typing import Optional, List
from pydantic import validate_arguments 
from torch.utils.data import DataLoader
# ionpy imports
from ionpy.util import Config
from ionpy.util.ioutil import autosave
from ionpy.util.config import config_digest
from ionpy.experiment.util import absolute_import, fix_seed, generate_tuid, eval_config
# UniverSeg imports
from universeg.experiment.datasets import Segment2D
from universeg.experiment.datasets.support import RandomSupport
# local imports
from ...experiment.utils import load_experiment, get_exp_load_info
from ...augmentation.gather import augmentations_from_config
from ...experiment import EnsembleInferenceExperiment, BinningInferenceExperiment


def verify_graceful_exit(log_path: str, log_root: str):
    submitit_dir = os.path.join(log_root, log_path, "submitit")
    unique_logs = list(set([logfile.split("_")[0] for logfile in os.listdir(submitit_dir)]))
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


def cal_stats_init(cfg_dict):
    ###################
    # BUILD THE MODEL #
    ###################
    inference_exp, inference_cfg, save_root = load_inference_exp_from_cfg(
        inference_cfg=cfg_dict
    )
    inference_exp.to_device()
    # Ensure that inference seed is the same.
    fix_seed(inference_cfg['experiment']['seed'])

    #####################
    # BUILD THE DATASET #
    #####################
    # Rebuild the experiments dataset with the new cfg modifications.
    new_dset_options = inference_cfg['data'].copy()
    input_type = new_dset_options.pop("input_type")
    assert input_type in ["volume", "image"], f"Data type {input_type} not supported."
    assert inference_cfg['dataloader']['batch_size'] == 1, "Inference only configured for batch size of 1."
    # Get the inference augmentation options.
    aug_cfg_list = None if 'augmentations' not in inference_cfg.keys() else inference_cfg['augmentations']
    # Build the dataloaders.
    data_objs, modified_cfg = dataloader_from_exp( 
        inference_exp,
        inference_cfg=inference_cfg,
        new_dset_options=new_dset_options, 
        aug_cfg_list=aug_cfg_list
    )
    inference_cfg['dataset'] = modified_cfg 
    #############################
    trackers = {
        "image_level_records": [],
        "tl_pixel_meter_dict": {},
        "cw_pixel_meter_dict": {}
    }
    # Add trackers per split
    for data_split in data_objs['dataloaders']:
        trackers['tl_pixel_meter_dict'][data_split] = {}
        trackers['cw_pixel_meter_dict'][data_split] = {}

    #####################
    # SAVE THE METADATA #
    #####################
    task_root = save_inference_metadata(
        cfg_dict=inference_cfg,
        save_root=save_root
    )
    
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
    return {
        "exp": inference_exp,
        "dloaders": data_objs['dataloaders'],
        "supports": data_objs['supports'],
        "trackers": trackers,
        "output_root": task_root
    }


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def load_inference_exp_from_cfg(inference_cfg): 
    inf_model_cfg = inference_cfg['model']
    # Get the configs of the experiment
    if inf_model_cfg['ensemble'] and inf_model_cfg['_type'] != "incontext":
        inference_exp = EnsembleInferenceExperiment.from_config(inference_cfg)
        save_root = Path(inference_exp.path)
    elif "Binning" in inference_cfg['calibrator']['_name']:
        inference_exp = BinningInferenceExperiment.from_config(inference_cfg)
        save_root = Path(inference_exp.path)
    else:
        load_exp_args = {
            **get_exp_load_info(inf_model_cfg['pretrained_exp_root']),
            "checkpoint": inf_model_cfg['checkpoint'],
            "load_data": False,
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
    inference_exp, 
    inference_cfg,
    aug_cfg_list: Optional[List[dict]] = None,
    new_dset_options: Optional[dict] = None, # This is a dictionary of options to update the dataset with.
):
    total_config = inference_exp.config.to_dict()
    inference_data_cfg = total_config['data']
    if new_dset_options is not None:
        inference_data_cfg.update(new_dset_options)
    # Make sure we aren't sampling for evaluation. 
    if "slicing" in inference_data_cfg.keys():
        assert inference_data_cfg['slicing'] not in ['central', 'dense', 'uniform'], "Sampling methods not allowed for evaluation."
    # Get the dataset class and build the transforms
    dataset_cls = inference_data_cfg.pop('_class')
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
        if drop_key in inference_data_cfg.keys():
            inference_data_cfg.pop(drop_key)
    # Ensure that we return the different data ids.
    inference_data_cfg['return_data_id'] = True
    # If aug cfg list is not None, that means that we want to change the inference transforms.
    if aug_cfg_list is not None:
        inference_transforms = augmentations_from_config(aug_cfg_list)
    else:
        inference_transforms = None
    dset_splits = ast.literal_eval(inference_data_cfg.pop('splits'))
    dataloaders = {}
    supports = {} # use for ICL
    for split in dset_splits:
        # Load the dataset with modified arguments.
        split_data_cfg = inference_data_cfg.copy()
        split_data_cfg['split'] = split
        if inference_cfg['model']['_type'] == 'incontext':
            lab_split_dataset_dict, lab_split_support_dict = get_incontext_dataset(
                data_cfg=split_data_cfg, 
            )
        else: 
            split_dataset_obj = absolute_import(dataset_cls)(
                transforms=inference_transforms, 
                **split_data_cfg
            )
            lab_split_support_dict = None
            # Package these into dummy dictionaries. Not great, but it works.
            lab_split_dataset_dict = {
                -1 : split_dataset_obj
            }
        # Iterate through the labels and put in the dataloaders
        supports[split] = lab_split_support_dict
        dataloaders[split] = {}
        for lab in lab_split_dataset_dict.keys():
            # Build the dataloader for this split and label.
            dataloaders[split][lab] = DataLoader(
                lab_split_dataset_dict[lab], 
                batch_size=inference_cfg['dataloader']['batch_size'], 
                num_workers=inference_cfg['dataloader']['num_workers'],
                shuffle=False
            )
    # Add the augmentation information.
    inference_data_cfg['augmentations'] = aug_cfg_list
    inference_data_cfg['_class'] = dataset_cls        
    # Build a dictionary of our data objs
    data_obj_dict = {
        "dataloaders": dataloaders,
        "supports": supports,
    }
    # Return the dataloaders and the modified data cfg.
    return data_obj_dict, inference_data_cfg


def get_incontext_dataset(data_cfg):
    support_size = data_cfg.pop('support_size')
    # Build the target and support datasets.
    if "samples_per_epoch" in data_cfg.keys():
        data_cfg.pop("samples_per_epoch")
    # Make a target dataset for each label provided.
    target_dset_lab_dict = {}
    support_lab_dict = {}
    for label in data_cfg.pop('labels'):
        target_lab_dataset = Segment2D(label=label, **data_cfg)
        context_lab_dataset = target_lab_dataset.other_split("train")
        # Construct the support of examples to use.
        lab_support = RandomSupport(
            context_lab_dataset, support_size=support_size, replacement=True
        )
        # Place them in the dictionary
        target_dset_lab_dict[label] = target_lab_dataset
        support_lab_dict[label] = lab_support
    return target_dset_lab_dict, support_lab_dict 


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