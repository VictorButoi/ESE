#misc imports
import os
import ast
import yaml
import torch
import pickle
import pandas as pd
from pathlib import Path
from typing import Optional, List
from pydantic import validate_arguments 
from torch.utils.data import DataLoader
# ionpy imports
from ionpy.analysis import ResultsLoader
from ionpy.util import Config
from ionpy.util.ioutil import autosave
from ionpy.util.config import config_digest
from ionpy.experiment.util import absolute_import, fix_seed, generate_tuid, eval_config
# local imports
from ...augmentation.gather import augmentations_from_config
from ...experiment.utils import load_experiment
from ...experiment import EnsembleInferenceExperiment
from ...metrics.utils import (
    count_matching_neighbors,
    get_bins,
    find_bins
)


def add_dice_loss_rows(inference_df):
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
        # First replace Dice in the groupavg metric with 'Dice Loss'
        dice_loss_rows['groupavg_image_metric'] = dice_loss_rows['groupavg_image_metric'].str.replace('Dice', 'Dice_Loss')
        # Then flip the groupavg metric score.
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
    

def get_average_unet_baselines(
    total_df: pd.DataFrame,
    num_seeds: int,
    group_metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    # Collect all the individual networks.
    unet_info_df = total_df[total_df['ensemble'] == False].reset_index(drop=True)
    # These are the keys we want to group by.
    unet_group_keys = [
        'data_id',
        'slice_idx',
        'split',
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
    # They should have exactly 4, for four seeds.
    assert (num_rows_per_group.max() == num_seeds) and (num_rows_per_group.min() == num_seeds),\
        f"Grouping by these keys does not give the required number of rows per seed ({num_seeds})"\
             + f" with max {num_rows_per_group.max()} and min {num_rows_per_group.min()}. Got: {num_rows_per_group}."
    # Group everything we need. 
    total_group_metrics = ['metric_score', 'groupavg_metric_score']
    if group_metrics is not None:
        total_group_metrics += group_metrics
    average_seed_unet = unet_info_df.groupby(unet_group_keys).agg({met_name: 'mean' for met_name in total_group_metrics}).reset_index()
    # Set some useful variables.
    average_seed_unet['experiment.pretrained_seed'] = 'Average'
    average_seed_unet['pretrained_seed'] = 'Average'
    average_seed_unet['model_type'] = 'group' # Now this is a group of results

    def method_name(_pretrained_class, model_class):
        if model_class == "Vanilla":
            return "Average UNet"
        elif _pretrained_class == "None":
            return f"Average {model_class.split('.')[-1]}"
        else:
            return f"Average {_pretrained_class.split('.')[-1]}"

    def configuration(method_name, calibrator):
        return f"{method_name}_{calibrator}"

    average_seed_unet.augment(method_name)
    average_seed_unet.augment(configuration)

    return average_seed_unet


def cal_stats_init(cfg_dict):
    ###################
    # BUILD THE MODEL #
    ###################
    inference_exp, save_root = load_inference_exp_from_cfg(
        inference_cfg=cfg_dict
        )
    inference_exp.to_device()
    # Ensure that inference seed is the same.
    fix_seed(cfg_dict['experiment']['seed'])

    #####################
    # BUILD THE DATASET #
    #####################
    # Rebuild the experiments dataset with the new cfg modifications.
    new_dset_options = cfg_dict['data']
    input_type = new_dset_options.pop("input_type")
    assert input_type in ["volume", "image"], f"Data type {input_type} not supported."
    assert cfg_dict['dataloader']['batch_size'] == 1, "Inference only configured for batch size of 1."
    # Get the inference augmentation options.
    aug_cfg_list = None if 'augmentations' not in cfg_dict.keys() else cfg_dict['augmentations']
    # Build the dataloaders.
    dataloaders, modified_cfg = dataloader_from_exp( 
        inference_exp,
        new_dset_options=new_dset_options, 
        aug_cfg_list=aug_cfg_list,
        batch_size=cfg_dict['dataloader']['batch_size'],
        num_workers=cfg_dict['dataloader']['num_workers']
        )
    cfg_dict['dataset'] = modified_cfg 

    #####################
    # SAVE THE METADATA #
    #####################
    task_root = save_inference_metadata(
        cfg_dict=cfg_dict,
        save_root=save_root
    )
    
    print(f"Running:\n\n{str(yaml.safe_dump(Config(cfg_dict)._data, indent=0))}")
    ##################################
    # INITIALIZE THE QUALITY METRICS #
    ##################################
    qual_metrics = {}
    if 'qual_metrics' in cfg_dict.keys():
        for q_met_cfg in cfg_dict['qual_metrics']:
            q_metric_name = list(q_met_cfg.keys())[0]
            quality_metric_options = q_met_cfg[q_metric_name]
            metric_type = quality_metric_options.pop("metric_type")
            # Add the quality metric to the dictionary.
            qual_metrics[q_metric_name] = {
                "name": q_metric_name,
                "_fn": eval_config(quality_metric_options),
                "_type": metric_type
            }
    ##################################
    # INITIALIZE CALIBRATION METRICS #
    ##################################
    # Image level metrics.
    if 'image_cal_metrics' in cfg_dict.keys():
        image_cal_metrics = preload_calibration_metrics(
            base_cal_cfg=cfg_dict["calibration"],
            cal_metrics_dict=cfg_dict["image_cal_metrics"]
        )
    else:
        image_cal_metrics = {}
    # Global dataset level metrics. (Used for validation)
    if 'global_cal_metrics' in cfg_dict.keys():
        global_cal_metrics = preload_calibration_metrics(
            base_cal_cfg=cfg_dict["calibration"],
            cal_metrics_dict=cfg_dict["global_cal_metrics"]
        )
    else:
        global_cal_metrics = {}
    #############################
    trackers = {}
    # Setup trackers for both or either of image level statistics
    if cfg_dict['log']["log_image_stats"]:
        trackers["image_level_records"] = []
    # or pixel level statistics.
    if cfg_dict['log']["log_pixel_stats"]:
        trackers["tl_pixel_meter_dict"] = {}
        trackers["cw_pixel_meter_dict"] = {}
        # Add trackers per split
        for data_split in dataloaders:
            trackers["tl_pixel_meter_dict"][data_split] = {}
            trackers["cw_pixel_meter_dict"][data_split] = {}
    # Place these dictionaries into the config dictionary.
    cfg_dict["qual_metrics"] = qual_metrics 
    cfg_dict["image_cal_metrics"] = image_cal_metrics 
    cfg_dict["global_cal_metrics"] = global_cal_metrics
    # Return a dictionary of the components needed for the calibration statistics.
    return {
        "inference_exp": inference_exp,
        "input_type": input_type,
        "dataloaders": dataloaders,
        "trackers": trackers,
        "output_root": task_root
    }


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
        assert 'ensemble_cfg' in model_cfg.keys(), "Ensemble inference requires a combine function."
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
def dataloader_from_exp(
    inference_exp, 
    batch_size: int = 1,
    num_workers: int = 1,
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
    for drop_key in ['in_channels', 'out_channels', 'iters_per_epoch', 'input_type']:
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
    for split in dset_splits:
        split_data_cfg = inference_data_cfg.copy()
        split_data_cfg['split'] = split
        # Load the dataset with modified arguments.
        split_dataset_obj = absolute_import(dataset_cls)(transforms=inference_transforms, **split_data_cfg)
        # Build the dataset and dataloader.
        dataloaders[split] = DataLoader(
            split_dataset_obj, 
            batch_size=batch_size, 
            num_workers=num_workers,
            shuffle=False
        )
    # Add the augmentation information.
    inference_data_cfg['augmentations'] = aug_cfg_list
    inference_data_cfg['_class'] = dataset_cls        
    # Return the dataloaders and the modified data cfg.
    return dataloaders, inference_data_cfg


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def preload_calibration_metrics(
    base_cal_cfg: dict, 
    cal_metrics_dict: dict
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


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_image_aux_info(
    y_pred: torch.Tensor,
    y_hard: torch.Tensor,
    y_true: torch.Tensor,
    cal_cfg: dict,
) -> dict:
    assert y_hard.dim() == 4, f"Expected 4D tensor for y_hard, found shape: {y_hard.shape}."
    assert y_true.dim() == 4, f"Expected 4D tensor for y_true, found shape: {y_true.shape}."
    # Get the pixelwise frequency.
    frequency_map = (y_hard == y_true).squeeze(1).float()

    # Keep track of different things for each bin.
    pred_labels = y_hard.unique().tolist()
    if "ignore_index" in cal_cfg and cal_cfg["ignore_index"] in pred_labels:
        pred_labels.remove(cal_cfg["ignore_index"])

    # Get a map of which pixels match their neighbors and how often.
    # For both our prediction and the true label map.
    pred_matching_neighbors_map = count_matching_neighbors(
        lab_map=y_hard.squeeze(1), # Remove the channel dimension. 
        neighborhood_width=cal_cfg["neighborhood_width"]
    )

    true_matching_neighbors_map = count_matching_neighbors(
        lab_map=y_true.squeeze(1), # Remove the channel dimension. 
        neighborhood_width=cal_cfg["neighborhood_width"]
    ) 

    # Calculate the probability bin positions per pixel.
    y_max_prob_map = y_pred.max(dim=1).values # B x H x W

    # Define the confidence interval (if not provided).
    if "conf_interval" not in cal_cfg:
        C = y_pred.shape[1]
        if C == 0:
            lower_bound = 0
        else:
            lower_bound = 1 / C
        upper_bound = 1
        # Set the confidence interval.
        cal_cfg["conf_interval"] = (lower_bound, upper_bound)

    # Create the confidence bins.    
    conf_bins, conf_bin_widths = get_bins(
        num_bins=cal_cfg["num_bins"], 
        start=cal_cfg["conf_interval"][0], 
        end=cal_cfg["conf_interval"][1]
    )
    # Get the bin indices for each pixel.
    bin_ownership_map = find_bins(
        confidences=y_max_prob_map, 
        bin_starts=conf_bins,
        bin_widths=conf_bin_widths
    ) # B x H x W

    return {
        "frequency": frequency_map,
        "pred_labels": pred_labels,
        "bin_ownership_map": bin_ownership_map,
        "pred_matching_neighbors_map": pred_matching_neighbors_map,
        "true_matching_neighbors_map": true_matching_neighbors_map,
    }
