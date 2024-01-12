# Misc imports
import yaml
import pickle
import einops
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from pydantic import validate_arguments
from typing import Any, Optional, List
# torch imports
import torch
from torch.nn import functional as F
# ionpy imports
from ionpy.util import Config, StatsMeter
from ionpy.util.config import HDict, valmap
from ionpy.util.torchutils import to_device
from ionpy.experiment.util import fix_seed, eval_config
# local imports
from .utils import (
    get_image_aux_info, 
    dataloader_from_exp,
    show_inference_examples,
    save_inference_metadata,
    load_inference_exp_from_cfg
)
from ..metrics.utils import (
    get_bins, 
    find_bins, 
    count_matching_neighbors,
)


def list2tuple(val):
    if isinstance(val, list):
        return tuple(map(list2tuple, val))
    return val
    

def save_records(records, log_dir):
    # Save the items in a pickle file.  
    df = pd.DataFrame(records)
    # Save or overwrite the file.
    df.to_pickle(log_dir)


def save_dict(dict, log_dir):
    # save the dictionary to a pickl file at logdir
    with open(log_dir, 'wb') as f:
        pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def load_cal_inference_stats(
    log_dirs: List[Path],
    load_image_df: bool,
    load_pixel_meters_dict: bool
    ) -> dict:
    # Build a dictionary to store the inference info.
    cal_info_dict = {
        "pixel_meter_dicts": {},
        "image_info_df": pd.DataFrame([]),
        "metadata_df": pd.DataFrame([])
    }

    # Loop through every configuration in the log directory.
    for log_dir in log_dirs:
        for log_set in log_dir.iterdir():
            if log_set.name not in ["wandb", "submitit"]:
                # Load the metadata file (json) and add it to the metadata dataframe.
                config_dir = log_set / "config.yml"
                with open(config_dir, 'r') as stream:
                    cfg_yaml = yaml.safe_load(stream)
                cfg = HDict(cfg_yaml)
                flat_cfg = valmap(list2tuple, cfg.flatten())
                flat_cfg["log_set"] = log_set.name
                # Remove some columns we don't care about.
                for drop_key in [
                    "qual_metrics", 
                    "cal_metrics", 
                    "calibration.bin_weightings", 
                    "model.filters"
                    ]:
                    if drop_key in flat_cfg:
                        flat_cfg.pop(drop_key)
                # Convert the dictionary to a dataframe and concatenate it to the metadata dataframe.
                cfg_df = pd.DataFrame(flat_cfg, index=[0])
                cal_info_dict["metadata_df"] = pd.concat([cal_info_dict["metadata_df"], cfg_df])
    # Gather the columns that have unique values amongst the different configurations.
    unique_cols = []
    for col in cal_info_dict["metadata_df"].columns:
        if len(cal_info_dict["metadata_df"][col].unique()) > 1:
            unique_cols.append(col)
    # Loop through every configuration in the log directory.
    for log_dir in log_dirs:
        for log_set in log_dir.iterdir():
            if log_set.name not in ["wandb", "submitit"]:
                # Get the metadata corresponding to this log set.
                metadata_log_df = cal_info_dict["metadata_df"][cal_info_dict["metadata_df"]["log_set"] == log_set.name]
                # Optionally load the information from image-based metrics.
                if load_image_df:
                    log_image_df = pd.read_pickle(log_set / "image_stats.pkl")
                    log_image_df["log_set"] = log_set.name
                    # Add the columns from the metadata dataframe that have unique values.
                    for col in unique_cols:
                        assert len(metadata_log_df[col].unique()) == 1, \
                            f"Column {col} has more than one unique value in the metadata dataframe for log set {log_set}."
                        log_image_df[col] = metadata_log_df[col].values[0]
                    cal_info_dict["image_info_df"] = pd.concat([cal_info_dict["image_info_df"], log_image_df])
                # Optionally load the pixel stats.
                if load_pixel_meters_dict:
                    with open(log_set / "pixel_stats.pkl", 'rb') as f:
                        pixel_meter_dict = pickle.load(f)
                    # Set the pixel dict of the log set.
                    cal_info_dict["pixel_meter_dicts"][log_set.name] = pixel_meter_dict 
    # If we are loading the image stats df, make sure we check that all log_dirs have the same number of rows.
    if load_image_df:
        # Get the number of rows in image_info_df for each log set.
        num_rows_per_log_set = cal_info_dict["image_info_df"].groupby("log_set").size()
        # Make sure there is only one unique value in the above.
        assert len(num_rows_per_log_set.unique()) == 1, \
            "The number of rows in the image_info_df is not the same for all log sets."
    # Finally, return the dictionary of inference info.
    return cal_info_dict


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_cal_stats(
    cfg: Config,
    ) -> None:
    # Get the config dictionary
    cfg_dict = cfg.to_dict()

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
    dataloader, modified_cfg = dataloader_from_exp( 
        inference_exp,
        new_dset_options=new_dset_options, 
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
    cal_metrics = {}
    if 'cal_metrics' in cfg_dict.keys():
        for c_met_cfg in cfg_dict['cal_metrics']:
            c_metric_name = list(c_met_cfg.keys())[0]
            calibration_metric_options = c_met_cfg[c_metric_name]
            calibration_metric_options.update(cfg_dict['calibration'])
            # Add the calibration metric to the dictionary.
            cal_metrics[c_metric_name] = {
                "name": c_metric_name,
                "_fn": eval_config(c_met_cfg[c_metric_name])
            }
    #############################
    # Setup trackers for both or either of image level statistics and pixel level statistics.
    if cfg_dict["log"]["log_image_stats"]:
        image_level_records = []
    else:
        image_level_records = None
    if cfg_dict["log"]["log_pixel_stats"]:
        pixel_meter_dict = {}
    else:
        pixel_meter_dict = None
    # Place these dictionaries into the config dictionary.
    cfg_dict["qual_metrics"] = qual_metrics 
    cfg_dict["cal_metrics"] = cal_metrics 
    # Setup the log directories.
    image_level_dir = task_root / "image_stats.pkl"
    pixel_level_dir = task_root / "pixel_stats.pkl"
    # Set the looping function based on the input type.
    forward_loop_func = volume_forward_loop if (input_type == "volume") else image_forward_loop
    
    # Loop through the data, gather your stats!
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"Working on batch #{batch_idx} out of", len(dataloader), "({:.2f}%)".format(batch_idx / len(dataloader) * 100), end="\r")
            # Run the forward loop
            forward_loop_func(
                exp=inference_exp, 
                batch_idx=batch_idx,
                batch=batch, 
                inference_cfg=cfg_dict, 
                image_level_records=image_level_records,
                pixel_meter_dict=pixel_meter_dict
            )
            # Save the records every so often, to get intermediate results. Note, because of data_ids
            # this can contain fewer than 'log interval' many items.
            if batch_idx % cfg['log']['log_interval'] == 0:
                if image_level_records is not None:
                    save_records(image_level_records, image_level_dir)
                if pixel_meter_dict is not None:
                    save_dict(pixel_meter_dict, pixel_level_dir)
    # Save the records at the end too
    if image_level_records is not None:
        save_records(image_level_records, image_level_dir)
    if pixel_meter_dict is not None:
        save_dict(pixel_meter_dict, pixel_level_dir)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def volume_forward_loop(
    exp: Any,
    batch_idx: int,
    batch: Any,
    inference_cfg: dict,
    image_level_records: Optional[list] = None,
    pixel_meter_dict: Optional[dict] = None
):
    # Get the batch info
    image_vol, label_vol  = batch["img"], batch["label"]
    # Get your image label pair and define some regions.
    img_vol_cuda , label_vol_cuda = to_device((image_vol, label_vol), exp.device)
    # Go through each slice and predict the metrics.
    num_slices = img_vol_cuda.shape[1]
    for slice_idx in range(num_slices):
        print(f"-> Working on slice #{slice_idx} out of", num_slices, "({:.2f}%)".format((slice_idx / num_slices) * 100), end="\r")
        # Extract the slices from the volumes.
        image_cuda = img_vol_cuda[:, slice_idx:slice_idx+1, ...]
        label_map_cuda = label_vol_cuda[:, slice_idx:slice_idx+1, ...]
        # Get the prediction with no gradient accumulation.
        predict_args = {'multi_class': True}
        ensemble_show_preds = (inference_cfg["model"]["ensemble"] and inference_cfg["log"]["show_examples"])
        if ensemble_show_preds:
            predict_args["combine_fn"] = "identity"
        # Do a forward pass.
        with torch.no_grad():
            exp_output =  exp.predict(image_cuda, **predict_args)
        # Ensembling the preds and we want to show them we need to change the shape a bit.
        if ensemble_show_preds: 
            exp_output["ypred"] = einops.rearrange(exp_output["ypred"], "1 C E H W -> E C H W")
        # Wrap the outputs into a dictionary.
        output_dict = {
            "x": image_cuda,
            "ytrue": label_map_cuda.long(),
            "ypred": exp_output["ypred"],
            "yhard": exp_output["yhard"],
            "data_id": batch["data_id"][0], # Works because batchsize = 1
            "slice_idx": slice_idx 
        }
        # Get the calibration item info.  
        get_calibration_item_info(
            data_idx=(batch_idx*num_slices) + slice_idx,
            output_dict=output_dict,
            inference_cfg=inference_cfg,
            image_level_records=image_level_records,
            pixel_meter_dict=pixel_meter_dict
        )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_forward_loop(
    exp: Any,
    batch_idx: int,
    batch: Any,
    inference_cfg: dict,
    image_level_records: Optional[list],
    pixel_meter_dict: Optional[dict] = None
):
    # Get the batch info
    image, label_map  = batch["img"], batch["label"]
    # Get your image label pair and define some regions.
    image_cuda, label_map_cuda = to_device((image, label_map), exp.device)
    # Get the prediction with no gradient accumulation.
    predict_args = {'multi_class': True}
    ensemble_show_preds = (inference_cfg["model"]["ensemble"] and inference_cfg["log"]["show_examples"])
    if ensemble_show_preds:
        predict_args["combine_fn"] = "identity"
    # Do a forward pass.
    with torch.no_grad():
        exp_output =  exp.predict(image_cuda, **predict_args)
    # Ensembling the preds and we want to show them we need to change the shape a bit.
    if ensemble_show_preds: 
        exp_output["ypred"] = einops.rearrange(exp_output["ypred"], "1 C E H W -> E C H W")
        exp_output["yhard"] = einops.rearrange(exp_output["yhard"], "1 C E H W -> E C H W")
    # Wrap the outputs into a dictionary.
    output_dict = {
        "x": image_cuda,
        "ypred": exp_output["ypred"],
        "yhard": exp_output["yhard"],
        "ytrue": label_map_cuda.long(),
        "data_id": batch["data_id"][0],
        "slice_idx": None
    }
    # Get the calibration item info.  
    get_calibration_item_info(
        data_idx=batch_idx,
        output_dict=output_dict,
        inference_cfg=inference_cfg,
        image_level_records=image_level_records,
        pixel_meter_dict=pixel_meter_dict
    )

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_calibration_item_info(
    data_idx: int,
    output_dict: dict,
    inference_cfg: dict,
    image_level_records: Optional[list] = None,
    pixel_meter_dict: Optional[dict] = None
    ):
    ###########################
    # VISUALIZING IMAGE PREDS #
    ###########################
    if inference_cfg["log"]["show_examples"]:
        show_inference_examples(
            output_dict, 
            inference_cfg=inference_cfg
        )
    ########################
    # IMAGE LEVEL TRACKING #
    ########################
    check_image_stats = (image_level_records is not None)
    if check_image_stats:
        cal_metric_errors_dict = get_image_stats(
            output_dict=output_dict,
            inference_cfg=inference_cfg,
            image_level_records=image_level_records
        ) 
    ########################
    # PIXEL LEVEL TRACKING #
    ########################
    check_pixel_stats = (pixel_meter_dict is not None)
    if check_pixel_stats:
        update_pixel_meters(
            pixel_meter_dict=pixel_meter_dict,
            output_dict=output_dict,
            inference_cfg=inference_cfg
        )
    # Run a check on the image_level stats for a single image are the same as the pixel level stats.
    # This is just a sanity check. NOTE: data_idx has a small bug that if the inputs are volumes that 
    # have different numbers of slices, then the data_idx will not be correct but the first will be correct.
    if (data_idx == 0) and (check_image_stats and check_pixel_stats ): 
        global_cal_sanity_check(
            inference_cfg=inference_cfg, 
            cal_metric_errors_dict=cal_metric_errors_dict, 
            pixel_meter_dict=pixel_meter_dict
        )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_image_stats(
    output_dict: dict,
    inference_cfg: dict,
    image_level_records: list,
):
    # Define the cal config.
    qual_input_config = {
        "y_pred": output_dict["ypred"],
        "y_true": output_dict["ytrue"],
    }
    # Define the cal config.
    cal_input_config = {
        "y_pred": output_dict["ypred"],
        "y_true": output_dict["ytrue"],
        "stats_info_dict": get_image_aux_info(
            yhard=output_dict["yhard"],
            ytrue=output_dict["ytrue"],
            neighborhood_width=inference_cfg["calibration"]["neighborhood_width"],
            ignore_index=inference_cfg["calibration"]["ignore_index"]
        )
    }
    # Go through each calibration metric and calculate the score.
    qual_metric_scores_dict = {}
    for qual_metric_name, qual_metric_dict in inference_cfg["qual_metrics"].items():
        # Get the calibration error. 
        if qual_metric_dict['_type'] == 'calibration':
            # Higher is better for scores.
            qual_metric_scores_dict[qual_metric_name] = 1 - qual_metric_dict['_fn'](**cal_input_config).item() 
        else:
            qual_metric_scores_dict[qual_metric_name] = qual_metric_dict['_fn'](**qual_input_config).item()
        # If you're showing the predictions, also print the scores.
        if inference_cfg["log"]["show_examples"]:
            print(f"{qual_metric_name}: {qual_metric_scores_dict[qual_metric_name]}")
    # Go through each calibration metric and calculate the score.
    cal_metric_errors_dict = {}
    for cal_metric_name, cal_metric_dict in inference_cfg["cal_metrics"].items():
        # Get the calibration error. 
        cal_metric_errors_dict[cal_metric_name] = cal_metric_dict['_fn'](**cal_input_config).item() 
    
    assert not (len(qual_metric_scores_dict) == 0 and len(cal_metric_errors_dict) == 0), \
        "No metrics were specified in the config file."
    
    # Calculate the amount of present ground-truth there is in the image per label.
    num_classes = output_dict["ypred"].shape[1]
    y_true_one_hot = F.one_hot(output_dict["ytrue"], num_classes=num_classes) # B x 1 x H x W x C
    label_amounts = y_true_one_hot.sum(dim=(0, 1, 2, 3)) # C
    label_amounts_dict = {f"num_lab_{i}_pixels": label_amounts[i].item() for i in range(num_classes)}
    
    image_log_info = {
        "data_id": output_dict["data_id"],
        "slice_idx": output_dict["slice_idx"],
        **label_amounts_dict
    }
    if len(qual_metric_scores_dict) == 0:
        for cm_name in list(cal_metric_errors_dict.keys()):
            cal_metrics_record = {
                "cal_metric_type": cm_name.split("_")[-1],
                "cal_metric": cm_name.replace("_", " "),
                "cal_m_score": (1 - cal_metric_errors_dict[cm_name]),
                "cal_m_error": cal_metric_errors_dict[cm_name],
            }
            # Add the dataset info to the record
            record = {
                **image_log_info,
                **cal_metrics_record, 
                **inference_cfg["calibration"]
                }
            image_level_records.append(record)
    elif len(cal_metric_errors_dict) == 0:
        for qm_name in list(qual_metric_scores_dict.keys()):
            qual_metrics_record = {
                "qual_metric": qm_name,
                "qual_score": qual_metric_scores_dict[qm_name],
            }
            # Add the dataset info to the record
            record = {
                **image_log_info,
                **qual_metrics_record, 
                **inference_cfg["calibration"]
                }
            image_level_records.append(record)
    else:
        # Iterate through the cross product of calibration metrics and quality metrics.
        for qm_name, cm_name in list(product(qual_metric_scores_dict.keys(), cal_metric_errors_dict.keys())):
            combined_metrics_record = {
                "cal_metric_type": cm_name.split("_")[-1],
                "cal_metric": cm_name.replace("_", " "),
                "qual_metric": qm_name,
                "cal_m_score": (1 - cal_metric_errors_dict[cm_name]),
                "cal_m_error": cal_metric_errors_dict[cm_name],
                "qual_score": qual_metric_scores_dict[qm_name],
            }
            # Add the dataset info to the record
            record = {
                **image_log_info,
                **combined_metrics_record, 
                **inference_cfg["calibration"]
                }
            image_level_records.append(record)
    
    return cal_metric_errors_dict


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def update_pixel_meters(
    pixel_meter_dict: dict,
    output_dict: dict,
    inference_cfg: dict
):
    # Setup variables.
    H, W = output_dict["yhard"].shape[-2:]

    # If the confidence map is mulitclass, then we need to do some extra work.
    prob_map = output_dict["ypred"]
    if prob_map.shape[1] > 1:
        prob_map = torch.max(prob_map, dim=1, keepdim=True)[0]

    # Define the confidence bins and bin widths.
    conf_bins, conf_bin_widths = get_bins(
        num_bins=inference_cfg['calibration']['num_bins'], 
        start=inference_cfg['calibration']['conf_interval'][0], 
        end=inference_cfg['calibration']['conf_interval'][1]
    )

    # Figure out where each pixel belongs (in confidence)
    bin_ownership_map = find_bins(
        confidences=prob_map, 
        bin_starts=conf_bins,
        bin_widths=conf_bin_widths
        ).squeeze().cpu().numpy()

    # Get the pixel-wise number of matching neighbors map. Edge pixels have maximally 5 neighbors.
    pred_matching_neighbors_map = count_matching_neighbors(
        lab_map=output_dict["yhard"].squeeze(1), # Remove the channel dimension. 
        neighborhood_width=inference_cfg["calibration"]["neighborhood_width"],
        ).squeeze().cpu().numpy()

    # CPU-ize prob_map, yhard, and ytrue
    prob_map = prob_map.cpu().squeeze().numpy()
    ytrue = output_dict["ytrue"].cpu().squeeze().numpy()
    yhard = output_dict["yhard"].cpu().squeeze().numpy()

    # Calculate the accuracy map.
    acc_map = (yhard == ytrue).astype(np.float64)

    # Build the valid map from the ground truth pixels not containing our ignored index.
    ignore_index = inference_cfg["calibration"]["ignore_index"]
    if ignore_index is not None:
        assert isinstance(ignore_index, int)
        valid_idx_map = (ytrue != ignore_index)
    else:
        valid_idx_map = np.ones((H, W)).astype(np.bool)

    # Iterate through each pixel in the image.
    for (ix, iy) in np.ndindex((H, W)):
        # Only consider pixels that are valid (not ignored)
        if valid_idx_map[ix, iy]:
            # Create a unique key for the combination of label, neighbors, and confidence_bin
            true_label = ytrue[ix, iy]
            pred_label = yhard[ix, iy]
            num_matching_neighbors = pred_matching_neighbors_map[ix, iy]
            prob_bin = bin_ownership_map[ix, iy]
            # Define this dictionary prefix corresponding to a 'kind' of pixel.
            prefix = (true_label, pred_label, num_matching_neighbors, prob_bin)
            # Add bin specific keys to the dictionary if they don't exist.
            acc_key = prefix + ("accuracy",)
            conf_key = prefix + ("confidence",)
            # If this key doesn't exist in the dictionary, add it
            if conf_key not in pixel_meter_dict:
                for meter_key in [acc_key, conf_key]:
                    pixel_meter_dict[meter_key] = StatsMeter()
            # (acc , conf)
            acc = acc_map[ix, iy]
            conf = prob_map[ix, iy]
            # Finally, add the points to the meters.
            pixel_meter_dict[acc_key].add(acc) 
            pixel_meter_dict[conf_key].add(conf)


def global_cal_sanity_check(
        inference_cfg: dict, 
        cal_metric_errors_dict: dict, 
        pixel_meter_dict: dict
        ):
    # Iterate through all the calibration metrics and check that the pixel level calibration score
    # is the same as the image level calibration score (only true when we are working with a single
    # image.
    for cal_metric_name, cal_metric_dict in inference_cfg["cal_metrics"].items():
        # Get the calibration error. 
        pixel_level_cal_score = cal_metric_dict['_fn'](
            pixel_meters_dict=pixel_meter_dict,
            ).item() 
        assert cal_metric_errors_dict[cal_metric_name] == pixel_level_cal_score, \
            f"FAILED CAL EQUIVALENCE CHECK FOR CALIBRATION METRIC '{cal_metric_name}': "+\
                f"Pixel level calibration score ({pixel_level_cal_score}) does not match "+\
                    f"image level score ({cal_metric_errors_dict[cal_metric_name]})."

