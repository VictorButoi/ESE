# Misc imports
import yaml
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from pydantic import validate_arguments
from typing import Any, Optional, List
# torch imports
import torch
# ionpy imports
from ionpy.util import Config, StatsMeter
from ionpy.util.config import HDict, valmap
from ionpy.util.torchutils import to_device
from ionpy.experiment.util import fix_seed, eval_config
# local imports
from ..callbacks.visualize import ShowPredictionsCallback
from .utils import (
    get_image_aux_info, 
    dataloader_from_exp,
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
                if "qual_metrics" in flat_cfg:
                    flat_cfg.pop("qual_metrics")
                if "cal_metrics" in flat_cfg:
                    flat_cfg.pop("cal_metrics")
                if "calibration.bin_weightings" in flat_cfg:
                    flat_cfg.pop("calibration.bin_weightings")
                # Convert the dictionary to a dataframe and concatenate it to the metadata dataframe.
                cfg_df = pd.DataFrame(flat_cfg, index=[0])
                cal_info_dict["metadata_df"] = pd.concat([cal_info_dict["metadata_df"], cfg_df])
                # Loop through the different splits and load the image stats.
                if load_image_df:
                    log_image_df = pd.read_pickle(log_set / "image_stats.pkl")
                    log_image_df["log_set"] = log_set.name
                    cal_info_dict["image_info_df"] = pd.concat([cal_info_dict["image_info_df"], log_image_df])
                # Load the pixel stats.
                if load_pixel_meters_dict:
                    with open(log_set / "pixel_stats.pkl", 'rb') as f:
                        pixel_meter_dict = pickle.load(f)
                    # Set the pixel dict of the log set.
                    cal_info_dict["pixel_meter_dicts"][log_set.name] = pixel_meter_dict 
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
    # Make sure they are all evaluated in the same manner. This needs to go
    # below inference exp because loading the exp will overwrite the seed.
    fix_seed(cfg_dict['experiment']['seed'])
    # Make a new value for the pretrained seed, so we can differentiate between
    # members of ensemble
    old_inference_cfg = inference_exp.config.to_dict()
    cfg_dict['experiment']['pretrained_seed'] = old_inference_cfg['experiment']['seed']

    #####################
    # BUILD THE DATASET #
    #####################
    # Rebuild the experiments dataset with the new cfg modifications.
    new_dset_options = cfg_dict['dataset']
    input_type = new_dset_options.pop("input_type")
    assert input_type in ["volume", "image"], f"Data type {input_type} not supported."
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
    image_vol, label_vol  = batch
    # Get your image label pair and define some regions.
    img_vol_cuda , label_vol_cuda = to_device((image_vol, label_vol), exp.device)
    # Go through each slice and predict the metrics.
    num_slices = img_vol_cuda.shape[1]
    for slice_idx in range(num_slices):
        print(f"-> Working on slice #{slice_idx} out of", num_slices, "({:.2f}%)".format((slice_idx / num_slices) * 100), end="\r")
        # Extract the slices from the volumes.
        image_cuda = img_vol_cuda[:, slice_idx:slice_idx+1, ...]
        label_map_cuda = label_vol_cuda[:, slice_idx:slice_idx+1, ...]
        # Get the prediction and don't track gradients.
        with torch.no_grad():
            exp_output = exp.predict(image_cuda, multi_class=True)
        # Wrap the outputs into a dictionary.
        output_dict = {
            "x": image_cuda,
            "ytrue": label_map_cuda.long(),
            "ypred": exp_output["ypred"],
            "yhard": exp_output["yhard"],
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
    image, label_map  = batch
    # Get your image label pair and define some regions.
    image_cuda, label_map_cuda = to_device((image, label_map), exp.device)
    # Get the prediction with no gradient accumulation.
    with torch.no_grad():
        exp_output = exp.predict(image_cuda, multi_class=True)
    # Wrap the outputs into a dictionary.
    output_dict = {
        "x": image_cuda,
        "ypred": exp_output["ypred"],
        "yhard": exp_output["yhard"],
        "ytrue": label_map_cuda.long(),
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
    pixel_meter_dict: Optional[dict] = None,
    ignore_index: Optional[int] = None,
    ):
    if "show_examples" in inference_cfg["log"] and inference_cfg["log"]["show_examples"]:
        ShowPredictionsCallback(output_dict)
    raise ValueError
    # Setup some variables.
    if "ignore_index" in inference_cfg["log"]:
        ignore_index = inference_cfg["log"]["ignore_index"]
    ########################
    # IMAGE LEVEL TRACKING #
    ########################
    check_image_stats = (image_level_records is not None)
    if check_image_stats:
        cal_metric_errors_dict = update_image_records(
            image_level_records=image_level_records,
            output_dict=output_dict,
            inference_cfg=inference_cfg,
            ignore_index=ignore_index
        ) 
    ########################
    # PIXEL LEVEL TRACKING #
    ########################
    check_pixel_stats = (pixel_meter_dict is not None)
    if check_pixel_stats:
        update_pixel_meters(
            pixel_meter_dict=pixel_meter_dict,
            output_dict=output_dict,
            inference_cfg=inference_cfg,
            ignore_index=ignore_index
        )
    # Run a check on the image_level stats for a single image are the same as the pixel level stats.
    # This is just a sanity check. NOTE: data_idx has a small bug that if the inputs are volumes that 
    # have different numbers of slices, then the data_idx will not be correct but the first will be correct.
    if (data_idx == 0) and (check_image_stats and check_pixel_stats ): 
        global_cal_sanity_check(
            inference_cfg=inference_cfg, 
            cal_metric_errors_dict=cal_metric_errors_dict, 
            pixel_meter_dict=pixel_meter_dict,
            ignore_index=ignore_index
        )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def update_image_records(
    image_level_records: list,
    output_dict: dict,
    inference_cfg: dict,
    ignore_index: Optional[int] = None,
):
    # Setup the image stats config.
    image_stats_cfg = {
        "ypred": output_dict["ypred"],
        "yhard": output_dict["yhard"],
        "ytrue": output_dict["ytrue"],
        "slice_idx": output_dict["slice_idx"], # None if not a volume
        "inference_cfg": inference_cfg,
        "image_level_records": image_level_records,
        "ignore_index": ignore_index
    }
    return get_image_stats(**image_stats_cfg)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_image_stats(
    ypred: torch.Tensor,
    yhard: torch.Tensor,
    ytrue: torch.Tensor,
    inference_cfg: dict,
    image_level_records: list,
    slice_idx: Optional[int] = None,
    ignore_index: Optional[int] = None,
):
    # Define the cal config.
    qual_input_config = {
        "y_pred": ypred,
        "y_true": ytrue,
        "ignore_index": ignore_index
    }
    # Define the cal config.
    cal_input_config = {
        "y_pred": ypred,
        "y_true": ytrue,
        "num_bins": inference_cfg["calibration"]["num_bins"],
        "conf_interval":[
            inference_cfg["calibration"]["conf_interval_start"],
            inference_cfg["calibration"]["conf_interval_end"]
        ],
        "stats_info_dict": get_image_aux_info(
            yhard=yhard,
            ytrue=ytrue,
            neighborhood_width=inference_cfg["calibration"]["neighborhood_width"],
            ignore_index=ignore_index
        ),
        "square_diff": inference_cfg["calibration"]["square_diff"],
        "ignore_index": ignore_index
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
    if len(qual_metric_scores_dict) == 0:
        for cm_name in list(cal_metric_errors_dict.keys()):
            cal_record = {
                "cal_metric_type": cm_name.split("_")[-1],
                "cal_metric": cm_name.replace("_", " "),
                "cal_m_score": (1 - cal_metric_errors_dict[cm_name]),
                "cal_m_error": cal_metric_errors_dict[cm_name],
                "slice_idx": slice_idx
            }
            # Add the dataset info to the record
            record = {
                **cal_record, 
                **inference_cfg["calibration"]
                }
            image_level_records.append(record)
    elif len(cal_metric_errors_dict) == 0:
        for qm_name in list(qual_metric_scores_dict.keys()):
            cal_record = {
                "qual_metric": qm_name,
                "qual_score": qual_metric_scores_dict[qm_name],
                "slice_idx": slice_idx
            }
            # Add the dataset info to the record
            record = {
                **cal_record, 
                **inference_cfg["calibration"]
                }
            image_level_records.append(record)
    else:
        # Iterate through the cross product of calibration metrics and quality metrics.
        for qm_name, cm_name in list(product(qual_metric_scores_dict.keys(), cal_metric_errors_dict.keys())):
            cal_record = {
                "cal_metric_type": cm_name.split("_")[-1],
                "cal_metric": cm_name.replace("_", " "),
                "qual_metric": qm_name,
                "cal_m_score": (1 - cal_metric_errors_dict[cm_name]),
                "cal_m_error": cal_metric_errors_dict[cm_name],
                "qual_score": qual_metric_scores_dict[qm_name],
                "slice_idx": slice_idx
            }
            # Add the dataset info to the record
            record = {
                **cal_record, 
                **inference_cfg["calibration"]
                }
            image_level_records.append(record)
    
    return cal_metric_errors_dict


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def update_pixel_meters(
    pixel_meter_dict: dict,
    output_dict: dict,
    inference_cfg: dict,
    ignore_index: Optional[int] = None,
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
        start=inference_cfg['calibration']['conf_interval_start'], 
        end=inference_cfg['calibration']['conf_interval_end']
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

    # Build the valid map.
    if ignore_index is not None:
        valid_idx_map = (yhard != ignore_index)
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
        pixel_meter_dict: dict,
        ignore_index: Optional[int] = None
        ):
    # Iterate through all the calibration metrics and check that the pixel level calibration score
    # is the same as the image level calibration score (only true when we are working with a single
    # image.
    for cal_metric in inference_cfg["cal_metric_cfgs"]:
        # Get the calibration error. 
        cal_met_name = list(cal_metric.keys())[0]
        image_level_cal_score = cal_metric_errors_dict[cal_met_name]
        pixel_level_cal_score = cal_metric[cal_met_name]['func'](
            pixel_meters_dict=pixel_meter_dict,
            num_bins=inference_cfg["calibration"]["num_bins"],
            conf_interval=[
                inference_cfg["calibration"]["conf_interval_start"],
                inference_cfg["calibration"]["conf_interval_end"]
            ],
            square_diff=inference_cfg["calibration"]["square_diff"],
            ignore_index=ignore_index
            ).item() 
        assert cal_metric_errors_dict[cal_met_name] == pixel_level_cal_score, \
            f"FAILED CAL EQUIVALENCE CHECK FOR CALIBRATION METRIC '{cal_met_name}': "+\
                f"Pixel level calibration score ({pixel_level_cal_score}) does not match image level score ({image_level_cal_score})."

