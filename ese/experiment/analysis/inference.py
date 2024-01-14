# Misc imports
import os
import yaml
import pickle
import einops
import numpy as np
import pandas as pd
from pathlib import Path
from pydantic import validate_arguments
from typing import Any, Optional
# torch imports
import torch
from torch.nn import functional as F
# ionpy imports
from ionpy.util import Config, StatsMeter
from ionpy.util.config import HDict, valmap
from ionpy.util.torchutils import to_device
from ionpy.experiment.util import fix_seed, eval_config
# local imports
from .inference_utils import (
    get_image_aux_info, 
    dataloader_from_exp,
    save_inference_metadata,
    preload_calibration_metrics,
    load_inference_exp_from_cfg
)
from ..metrics.utils import (
    get_bins, 
    find_bins, 
    count_matching_neighbors,
)
from ..experiment.utils import show_inference_examples


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
    results_cfg: dict
) -> dict:
    # Build a dictionary to store the inference info.
    inference_df = pd.DataFrame([])
    metadata_df = pd.DataFrame([])
    # Loop through every configuration in the log directory.
    for log_path in results_cfg["log"]["inference_paths"]:
        log_dir = Path(os.path.join(results_cfg["log"]["root"], log_path))
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
                    "calibration.conf_interval",
                    "model.filters"
                    ]:
                    if drop_key in flat_cfg:
                        flat_cfg.pop(drop_key)
                # Convert the dictionary to a dataframe and concatenate it to the metadata dataframe.
                cfg_df = pd.DataFrame(flat_cfg, index=[0])
                metadata_df = pd.concat([metadata_df, cfg_df])
    # Gather the columns that have unique values amongst the different configurations.
    if results_cfg["log"]["remove_shared_columns"]:
        meta_cols = []
        for col in metadata_df.columns:
            if len(metadata_df[col].unique()) > 1:
                meta_cols.append(col)
    else:
        meta_cols = metadata_df.columns
    ##################################
    # INITIALIZE CALIBRATION METRICS #
    ##################################
    if 'cal_metrics' in results_cfg.keys():
        cal_metrics = preload_calibration_metrics(
            base_cal_cfg=results_cfg["calibration"],
            cal_metrics_dict=results_cfg["cal_metrics"]
        )
    else:
        cal_metrics = {}
    #############################
    # Loop through every configuration in the log directory.
    for log_path in results_cfg["log"]["inference_paths"]:
        log_dir = Path(os.path.join(results_cfg["log"]["root"], log_path))
        for log_set in log_dir.iterdir():
            if log_set.name not in ["wandb", "submitit"]:
                # Get the metadata corresponding to this log set.
                metadata_log_df = metadata_df[metadata_df["log_set"] == log_set.name]
                # Optionally load the information from image-based metrics.
                log_image_df = pd.read_pickle(log_set / "image_stats.pkl")
                log_image_df["log_set"] = log_set.name
                # Add the columns from the metadata dataframe that have unique values.
                for col in meta_cols:
                    assert len(metadata_log_df[col].unique()) == 1, \
                        f"Column {col} has more than one unique value in the metadata dataframe for log set {log_set}."
                    log_image_df[col] = metadata_log_df[col].values[0]
                # Optionally load the pixel stats.
                if results_cfg["log"]["load_pixel_meters"]:
                    with open(log_set / "pixel_stats.pkl", 'rb') as f:
                        pixel_meter_dict = pickle.load(f)
                    # Loop through the calibration metrics and add them to the dataframe.
                    for cal_metric_name, cal_metric_dict in cal_metrics.items():
                        log_image_df[cal_metric_name] = cal_metric_dict['_fn'](
                            pixel_meters_dict=pixel_meter_dict
                        ).item() 
                # Add this log to the dataframe.
                inference_df = pd.concat([inference_df, log_image_df])
    # Get the number of rows in image_info_df for each log set.
    num_rows_per_log_set = inference_df.groupby("log_set").size()
    # Make sure there is only one unique value in the above.
    assert len(num_rows_per_log_set.unique()) == 1, \
        "The number of rows in the image_info_df is not the same for all log sets."
    # Finally, return the dictionary of inference info.
    return inference_df


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
    cfg_dict["image_cal_metrics"] = image_cal_metrics 
    cfg_dict["global_cal_metrics"] = global_cal_metrics
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
    batch: Any,
    inference_cfg: dict,
    image_level_records: Optional[list] = None,
    pixel_meter_dict: Optional[dict] = None
):
    # Get the batch info
    image_vol_cpu, label_vol_cpu  = batch["img"], batch["label"]
    image_vol_cuda, label_vol_cuda = to_device((image_vol_cpu, label_vol_cpu), exp.device)
    # Go through each slice and predict the metrics.
    num_slices = image_vol_cuda.shape[1]
    for slice_idx in range(num_slices):
        print(f"-> Working on slice #{slice_idx} out of", num_slices, "({:.2f}%)".format((slice_idx / num_slices) * 100), end="\r")
        # Get the prediction with no gradient accumulation.
        slice_batch = {
            "img": image_vol_cuda[:, slice_idx:slice_idx+1, ...],
            "label": label_vol_cuda[:, slice_idx:slice_idx+1, ...],
            "data_id": batch["data_id"],
        } 
        image_forward_loop(
            exp=exp,
            batch=slice_batch,
            inference_cfg=inference_cfg,
            slice_idx=slice_idx,
            image_level_records=image_level_records,
            pixel_meter_dict=pixel_meter_dict
        )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_forward_loop(
    exp: Any,
    batch: Any,
    inference_cfg: dict,
    slice_idx: Optional[int] = None,
    image_level_records: Optional[list] = None,
    pixel_meter_dict: Optional[dict] = None
):
    # Get the batch info
    image, label_map  = batch["img"], batch["label"]
    # Get your image label pair and define some regions.
    if image.device != exp.device:
        image, label_map = to_device((image, label_map), exp.device)
    # Get the prediction with no gradient accumulation.
    predict_args = {'multi_class': True}
    ensemble_show_preds = (inference_cfg["model"]["ensemble"] and inference_cfg["log"]["show_examples"])
    if ensemble_show_preds:
        predict_args["combine_fn"] = "identity"
    # Do a forward pass.
    with torch.no_grad():
        exp_output =  exp.predict(image, **predict_args)
    # Ensembling the preds and we want to show them we need to change the shape a bit.
    if ensemble_show_preds: 
        exp_output["y_pred"] = einops.rearrange(exp_output["y_pred"], "1 C E H W -> E C H W")
    # Wrap the outputs into a dictionary.
    output_dict = {
        "x": image,
        "y_true": label_map.long(),
        "y_pred": exp_output["y_pred"],
        "y_hard": exp_output["y_hard"],
        "data_id": batch["data_id"][0], # Works because batchsize = 1
        "slice_idx": slice_idx 
    }
    # Get the calibration item info.  
    get_calibration_item_info(
        output_dict=output_dict,
        inference_cfg=inference_cfg,
        image_level_records=image_level_records,
        pixel_meter_dict=pixel_meter_dict
    )

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_calibration_item_info(
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
        image_cal_metrics_dict = get_image_stats(
            output_dict=output_dict,
            inference_cfg=inference_cfg,
            image_level_records=image_level_records
        ) 
    ########################
    # PIXEL LEVEL TRACKING #
    ########################
    check_pixel_stats = (pixel_meter_dict is not None)
    if check_pixel_stats:
        image_pixel_meter_dict = update_pixel_meters(
            pixel_meter_dict=pixel_meter_dict,
            output_dict=output_dict,
            inference_cfg=inference_cfg
        )
    ##################################################################
    # SANITY CHECK THAT THE CALIBRATION METRICS AGREE FOR THIS IMAGE #
    ##################################################################
    if check_image_stats and check_pixel_stats: 
        global_cal_sanity_check(
            data_id=output_dict["data_id"],
            slice_idx=output_dict["slice_idx"],
            inference_cfg=inference_cfg, 
            image_cal_metrics_dict=image_cal_metrics_dict, 
            image_pixel_meter_dict=image_pixel_meter_dict
        )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_image_stats(
    output_dict: dict,
    inference_cfg: dict,
    image_level_records: list,
):
    # Define the cal config.
    qual_input_config = {
        "y_pred": output_dict["y_pred"],
        "y_true": output_dict["y_true"],
    }
    # Define the cal config.
    cal_input_config = {
        "y_pred": output_dict["y_pred"],
        "y_true": output_dict["y_true"],
        "stats_info_dict": get_image_aux_info(
            y_pred=output_dict["y_pred"],
            y_hard=output_dict["y_hard"],
            y_true=output_dict["y_true"],
            cal_cfg=inference_cfg["calibration"]
        )
    }
    # Go through each calibration metric and calculate the score.
    qual_metric_scores_dict = {}
    for qual_metric_name, qual_metric_dict in inference_cfg["qual_metrics"].items():
        # Get the calibration error. 
        if qual_metric_dict['_type'] == 'calibration':
            # Higher is better for scores.
            qual_metric_scores_dict[qual_metric_name] = qual_metric_dict['_fn'](**cal_input_config).item() 
        else:
            qual_metric_scores_dict[qual_metric_name] = qual_metric_dict['_fn'](**qual_input_config).item()
        # If you're showing the predictions, also print the scores.
        if inference_cfg["log"]["show_examples"]:
            print(f"{qual_metric_name}: {qual_metric_scores_dict[qual_metric_name]}")
    # Go through each calibration metric and calculate the score.
    cal_metric_errors_dict = {}
    for cal_metric_name, cal_metric_dict in inference_cfg["image_cal_metrics"].items():
        # Get the calibration error. 
        cal_metric_errors_dict[cal_metric_name] = cal_metric_dict['_fn'](**cal_input_config).item() 
    
    assert not (len(qual_metric_scores_dict) == 0 and len(cal_metric_errors_dict) == 0), \
        "No metrics were specified in the config file."
    
    # Calculate the amount of present ground-truth there is in the image per label.
    num_classes = output_dict["y_pred"].shape[1]
    y_true_one_hot = F.one_hot(output_dict["y_true"], num_classes=num_classes) # B x 1 x H x W x C
    label_amounts = y_true_one_hot.sum(dim=(0, 1, 2, 3)) # C
    label_amounts_dict = {f"num_lab_{i}_pixels": label_amounts[i].item() for i in range(num_classes)}
    
    # Add our scores to the image level records.
    for metric_score_dict in [cal_metric_errors_dict, qual_metric_scores_dict]:
        for met_name in list(metric_score_dict.keys()):
            metrics_record = {
                "image_metric": met_name,
                "metric_score": metric_score_dict[met_name],
            }
            # Add the dataset info to the record
            record = {
                "data_id": output_dict["data_id"],
                "slice_idx": output_dict["slice_idx"],
                **metrics_record, 
                **label_amounts_dict,
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
    H, W = output_dict["y_hard"].shape[-2:]

    # If the confidence map is mulitclass, then we need to do some extra work.
    y_pred = output_dict["y_pred"]
    if y_pred.shape[1] > 1:
        y_pred = torch.max(y_pred, dim=1, keepdim=True)[0]

    # Define the confidence bins and bin widths.
    conf_bins, conf_bin_widths = get_bins(
        num_bins=inference_cfg['calibration']['num_bins'], 
        start=inference_cfg['calibration']['conf_interval'][0], 
        end=inference_cfg['calibration']['conf_interval'][1]
    )

    # Figure out where each pixel belongs (in confidence)
    bin_ownership_map = find_bins(
        confidences=y_pred, 
        bin_starts=conf_bins,
        bin_widths=conf_bin_widths
        ).squeeze().cpu().numpy()

    # Get the pixel-wise number of PREDICTED matching neighbors.
    pred_num_neighb_map = count_matching_neighbors(
        lab_map=output_dict["y_hard"].squeeze(1), # Remove the channel dimension. 
        neighborhood_width=inference_cfg["calibration"]["neighborhood_width"],
        ).squeeze().cpu().numpy()
    
    # Get the pixel-wise number of PREDICTED matching neighbors.
    true_num_neighb_map = count_matching_neighbors(
        lab_map=output_dict["y_true"].squeeze(1), # Remove the channel dimension. 
        neighborhood_width=inference_cfg["calibration"]["neighborhood_width"],
        ).squeeze().cpu().numpy()

    # CPU-ize prob_map, y_hard, and y_true
    y_pred = y_pred.cpu().squeeze().numpy().astype(np.float64) # REQUIRED FOR PRECISION
    y_hard = output_dict["y_hard"].cpu().squeeze().numpy()
    y_true = output_dict["y_true"].cpu().squeeze().numpy()

    # Calculate the accuracy map.
    acc_map = (y_hard == y_true).astype(np.float64)

    # Make a version of pixel meter dict for this image
    image_pixel_meter_dict = {}
    # Iterate through each pixel in the image.
    for (ix, iy) in np.ndindex((H, W)):
        # Create a unique key for the combination of label, neighbors, and confidence_bin
        true_label = y_true[ix, iy]
        pred_label = y_hard[ix, iy]
        pred_num_neighb = pred_num_neighb_map[ix, iy]
        true_num_neighb = true_num_neighb_map[ix, iy]
        prob_bin = bin_ownership_map[ix, iy]
        # Define this dictionary prefix corresponding to a 'kind' of pixel.
        prefix = (true_label, pred_label, true_num_neighb, pred_num_neighb, prob_bin)
        # Add bin specific keys to the dictionary if they don't exist.
        acc_key = prefix + ("accuracy",)
        conf_key = prefix + ("confidence",)
        # If this key doesn't exist in the dictionary, add it
        if conf_key not in pixel_meter_dict:
            for meter_key in [acc_key, conf_key]:
                pixel_meter_dict[meter_key] = StatsMeter()
        # Add the keys for the image level tracker.
        if conf_key not in image_pixel_meter_dict:
            for meter_key in [acc_key, conf_key]:
                image_pixel_meter_dict[meter_key] = StatsMeter()
        # (acc , conf)
        acc = acc_map[ix, iy]
        conf = y_pred[ix, iy]
        # Finally, add the points to the meters.
        pixel_meter_dict[acc_key].add(acc) 
        pixel_meter_dict[conf_key].add(conf)
        # Add to the local image meter dict.
        image_pixel_meter_dict[acc_key].add(acc)
        image_pixel_meter_dict[conf_key].add(conf)
    # Return the image pixel meter dict.
    return image_pixel_meter_dict


def global_cal_sanity_check(
        data_id: str,
        slice_idx: Any,
        inference_cfg: dict, 
        image_cal_metrics_dict: dict, 
        image_pixel_meter_dict: dict
        ):
    # Iterate through all the calibration metrics and check that the pixel level calibration score
    # is the same as the image level calibration score (only true when we are working with a single
    # image.
    for cal_metric_name, cal_metric_dict in inference_cfg["image_cal_metrics"].items():
        metric_base = cal_metric_name.split("_")[-1]
        if metric_base in inference_cfg["global_cal_metrics"]:
            global_metric_dict = inference_cfg["global_cal_metrics"][metric_base]
            # Get the calibration error in two views. 
            image_cal_score = np.round(image_cal_metrics_dict[cal_metric_name], 3)
            meter_cal_score = np.round(global_metric_dict['_fn'](pixel_meters_dict=image_pixel_meter_dict).item(), 3)
            if image_cal_score != meter_cal_score:
                raise ValueError(f"WARNING on data id {data_id}, slice {slice_idx}: CALIBRATION METRIC '{cal_metric_name}' DOES NOT MATCH FOR IMAGE AND PIXEL LEVELS."+\
                f" Pixel level calibration score ({meter_cal_score}) does not match image level score ({image_cal_score}).")

