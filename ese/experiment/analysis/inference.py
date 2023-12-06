# Misc imports
import yaml
import pickle
import einops
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pydantic import validate_arguments
from typing import Any, Optional, List
# torch imports
import torch
# ionpy imports
from ionpy.util import Config, StatsMeter
from ionpy.analysis import ResultsLoader
from ionpy.util.config import config_digest, HDict, valmap
from ionpy.util.torchutils import to_device
from ionpy.experiment.util import absolute_import, generate_tuid
# local imports
from .utils import dataloader_from_exp, binarize
from ..experiment.ese_exp import CalibrationExperiment
from ..metrics.image_cal import brier_score
from ..metrics.utils import (
    get_bins, 
    find_bins, 
    count_matching_neighbors, 
    get_uni_pixel_weights
)
from ..metrics.segmentation import (
    labelwise_dice_score,
    avg_pixel_accuracy, 
    labelwise_pixel_accuracy, 
    avg_edge_pixel_accuracy, 
    labelwise_edge_pixel_accuracy
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
    log_dir: pathlib.Path
    ) -> dict:
    # Load the calibration inference stats from the log directory.
    log_dir = pathlib.Path(log_dir)

    cal_info_dict = {}
    cal_info_dict["pixel_info_dicts"] = {}
    cal_info_dict["image_info_df"] = pd.DataFrame([])
    cal_info_dict["metadata"] = pd.DataFrame([])
    # Loop through every configuration in the log directory.
    for log_set in log_dir.iterdir():
        if log_set.name != "submitit":
            # Load the metadata file (json) and add it to the metadata dataframe.
            log_mdata_yaml = log_set / "metadata.yaml"
            with open(log_mdata_yaml, 'r') as stream:
                cfg_yaml = yaml.safe_load(stream)
            cfg = HDict(cfg_yaml)
            flat_cfg = valmap(list2tuple, cfg.flatten())
            flat_cfg["log_set"] = log_set.name
            # Remove some columns we don't care about.
            flat_cfg.pop("cal_metrics")
            if "calibration.bin_weightings" in flat_cfg.keys():
                flat_cfg.pop("calibration.bin_weightings")
            # Convert the dictionary to a dataframe and concatenate it to the metadata dataframe.
            cfg_df = pd.DataFrame(flat_cfg, index=[0])
            cal_info_dict["metadata"] = pd.concat([cal_info_dict["metadata"], cfg_df])

            # Loop through the different splits and load the image stats.
            image_stats_df = pd.DataFrame([])
            for image_stats_split in log_set.glob("image_stats_split*"):
                image_split_df = pd.read_pickle(image_stats_split)
                image_stats_df = pd.concat([image_stats_df, image_split_df])
            image_stats_df["log_set"] = log_set.name
            cal_info_dict["image_info_df"] = pd.concat([cal_info_dict["image_info_df"], image_stats_df])

            # Loop through each of the different splits, and accumulate the bin 
            # pixel data.
            running_meter_dict = None
            for pixel_split in log_set.glob("pixel_stats_split*"):
                # Load the pkl file
                with open(pixel_split, 'rb') as f:
                    pixel_meter_dict = pickle.load(f)
                # Combine the different data splits.
                if running_meter_dict is None:
                    running_meter_dict = pixel_meter_dict
                else:
                    # Go through all keys and combine the meters.
                    for key in pixel_meter_dict.keys():
                        if key not in running_meter_dict.keys():
                            running_meter_dict[key] = pixel_meter_dict[key]
                        else:
                            running_meter_dict[key] += pixel_meter_dict[key] 
            # Set the pixel dict of the log set.
            cal_info_dict["pixel_info_dicts"][log_set.name] = running_meter_dict

    # Finally, return the dictionary of inference info.
    return cal_info_dict


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_cal_stats(
    cfg: Config, 
    uuid: Optional[str] = None,
    data_split: Optional[int] = 0,
    split_data_ids: Optional[List[str]] = None
    ) -> None:
    ###################
    # BUILD THE MODEL #
    ###################
    # Get the config dictionary
    cfg_dict = cfg.to_dict()

    # Results loader object does everything
    save_root = pathlib.Path(cfg_dict['log']['root'])
    # Get the configs of the experiment
    rs = ResultsLoader()
    dfc = rs.load_configs(
        cfg_dict['model']['exp_root'],
        properties=False,
    )
    best_exp = rs.get_best_experiment(
        df=rs.load_metrics(dfc),
        exp_class=CalibrationExperiment,
        device="cuda"
    )

    #####################
    # BUILD THE DATASET #
    #####################
    # Rebuild the experiments dataset with the new cfg modifications.
    new_dset_options = cfg_dict['dataset']
    input_type = new_dset_options.pop("input_type")
    assert input_type in ["volume", "image"], f"Data type {input_type} not supported."
    dataloader, modified_cfg = dataloader_from_exp( 
        best_exp,
        new_dset_options=new_dset_options, 
        return_data_id=True,
        num_workers=cfg_dict['model']['num_workers']
        )
    cfg_dict['dataset'] = modified_cfg 
    # Set the looping function based on the input type.
    forward_loop_func = volume_forward_loop if input_type == "volume" else image_forward_loop

    #####################
    # DEFINE THE OUTPUT #
    #####################
    # Prepare the output dir for saving the results
    if uuid is None:
        create_time, nonce = generate_tuid()
        digest = config_digest(cfg_dict)
        uuid = f"{create_time}-{nonce}-{digest}"

    # make sure to add inference in front of the exp name (easy grep). We have multiple
    # data splits so that we can potentially parralelize the inference.
    task_root = save_root / uuid
    metadata_dir = task_root / "metadata.yaml"
    image_level_dir = task_root / f"image_stats_split:{data_split}.pkl"
    pixel_level_dir = task_root / f"pixel_stats_split:{data_split}.pkl"
    if not task_root.exists():
        task_root.mkdir(parents=True)
        with open(metadata_dir, 'w') as metafile:
            yaml.dump(cfg_dict, metafile, default_flow_style=False) 

    # Setup trackers for both or either of image level statistics and pixel level statistics.
    image_level_records = None
    pixel_meter_dict = None
    if cfg_dict["log"]["track_image_level"]:
        image_level_records = []
    if cfg_dict["log"]["track_pixel_level"]:
        pixel_meter_dict = {}
        
    ##################################
    # INITIALIZE CALIBRATION METRICS #
    ##################################
    metric_cfgs = cfg_dict['cal_metrics']
    for cal_metric in metric_cfgs:
        for metric_key in cal_metric.keys():
            cal_metric[metric_key]['func'] = absolute_import(cal_metric[metric_key]['func'])

    # Define the confidence bins and bin widths.
    conf_bins, conf_bin_widths = get_bins(
        num_bins=cfg_dict['calibration']['num_bins'], 
        start=cfg_dict['calibration']['conf_interval_start'], 
        end=cfg_dict['calibration']['conf_interval_end']
    )

    # Loop through the data, gather your stats!
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"Working on batch #{batch_idx} out of", len(dataloader), "({:.2f}%)".format(batch_idx / len(dataloader) * 100), end="\r")
            # Get the batch info
            _, _, batch_data_id = batch
            # Only run the loop if we are using all the data or if the batch_id is in the data_ids.
            # we do [0] because the batchsize is 1.
            if split_data_ids is None or batch_data_id[0] in split_data_ids:
                # Run the forward loop
                forward_loop_func(
                    exp=best_exp, 
                    batch=batch, 
                    inference_cfg=cfg_dict, 
                    metric_cfgs=metric_cfgs,
                    conf_bins=conf_bins,
                    conf_bin_widths=conf_bin_widths,
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
    exp: CalibrationExperiment,
    batch: Any,
    inference_cfg: dict,
    metric_cfgs: List[dict],
    conf_bins: torch.Tensor,
    conf_bin_widths: torch.Tensor,
    image_level_records: Optional[list] = None,
    pixel_meter_dict: Optional[dict] = None
):
    # Get the batch info
    image_vol, label_vol, batch_data_id = batch
    data_id = batch_data_id[0]
    # Get your image label pair and define some regions.
    img_vol_cuda , label_vol_cuda = to_device((image_vol, label_vol), exp.device)
    # Reshape so that we will like the shape.
    x_batch = einops.rearrange(img_vol_cuda, "b c h w -> (b c) 1 h w")
    y_batch = einops.rearrange(label_vol_cuda, "b c h w -> (b c) 1 h w")
    # Go through each slice and predict the metrics.
    num_slices = x_batch.shape[0]
    for slice_idx in range(num_slices):
        print(f"-> Working on slice #{slice_idx} out of", num_slices, "({:.2f}%)".format((slice_idx / num_slices) * 100), end="\r")
        # Extract the slices from the volumes.
        image_cuda = x_batch[slice_idx, ...][None]
        label_map_cuda = y_batch[slice_idx, ...][None]
        # Get the prediction
        conf_map, pred_map = exp.predict(image_cuda, multi_class=True)
        # Wrap the outputs into a dictionary.
        output_dict = {
            "image": image_cuda,
            "y_true": label_map_cuda.long(),
            "y_pred": conf_map,
            "y_hard": pred_map,
            "data_id": data_id,
            "slice_idx": slice_idx 
        }
        # Get the calibration item info.  
        get_calibration_item_info(
            output_dict=output_dict,
            inference_cfg=inference_cfg,
            metric_cfgs=metric_cfgs,
            conf_bins=conf_bins,
            conf_bin_widths=conf_bin_widths,
            image_level_records=image_level_records,
            pixel_meter_dict=pixel_meter_dict
        )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_forward_loop(
    exp: CalibrationExperiment,
    batch: Any,
    inference_cfg: dict,
    metric_cfgs: List[dict],
    conf_bins: torch.Tensor,
    conf_bin_widths: torch.Tensor,
    image_level_records: Optional[list],
    pixel_meter_dict: Optional[dict] = None
):
    # Get the batch info
    image, label_map, batch_data_id = batch
    data_id = batch_data_id[0]
    # Get your image label pair and define some regions.
    image_cuda, label_map_cuda = to_device((image, label_map), exp.device)
    # Get the prediction
    conf_map, pred_map = exp.predict(image_cuda, multi_class=True)
    # Wrap the outputs into a dictionary.
    output_dict = {
        "image": image_cuda,
        "y_true": label_map_cuda.long(),
        "y_pred": conf_map,
        "y_hard": pred_map,
        "data_id": data_id,
        "slice_idx": None
    }
    # Get the calibration item info.  
    get_calibration_item_info(
        output_dict=output_dict,
        inference_cfg=inference_cfg,
        metric_cfgs=metric_cfgs,
        conf_bins=conf_bins,
        conf_bin_widths=conf_bin_widths,
        image_level_records=image_level_records,
        pixel_meter_dict=pixel_meter_dict
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_quality_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    ignore_index: Optional[int] = None,
    square_diff: Optional[bool] = True, 
    ):
    # Get some metrics of these predictions.
    quality_metrics_dict = {
        "acc": labelwise_pixel_accuracy(
            y_pred=y_pred,
            y_true=y_true,
            ignore_index=ignore_index,
            ignore_empty_labels=True,
        ),
        "edge-acc": labelwise_edge_pixel_accuracy(
            y_pred=y_pred,
            y_true=y_true,
            ignore_index=ignore_index,
            ignore_empty_labels=True,
        ),
        "dice": labelwise_dice_score(
            y_pred=y_pred,
            y_true=y_true,
            ignore_index=ignore_index,
            ignore_empty_labels=True,
        ),
        "brier": brier_score(
            y_pred=y_pred,
            y_true=y_true,
            ignore_index=ignore_index,
            square_diff=square_diff,
            ignore_empty_labels=True,
        ),
    }
    return quality_metrics_dict


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_image_stats(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    data_id: int,
    inference_cfg: dict,
    metric_cfgs: List[dict],
    image_level_records: list,
    slice_idx: Optional[int] = None,
    label: Optional[torch.Tensor] = None,
    ignore_index: Optional[torch.Tensor] = None,
):
    # Binarize the prediction and label.
    quality_mets_conf = {
        "y_pred": y_pred,
        "y_true": y_true,
        "square_diff": inference_cfg["calibration"]["square_diff"],
        "ignore_index": ignore_index
    }

    # Get some metrics of these predictions.
    quality_metrics_dict = get_quality_metrics(**quality_mets_conf) 

    # Go through each calibration metric and calculate the score.
    for cal_metric in metric_cfgs:
        # Define the cal config.
        cal_config = {
            "y_pred": y_pred,
            "y_true": y_true,
            "num_bins": inference_cfg["calibration"]["num_bins"],
            "conf_interval":[
                inference_cfg["calibration"]["conf_interval_start"],
                inference_cfg["calibration"]["conf_interval_end"]
            ],
            "square_diff": inference_cfg["calibration"]["square_diff"],
            "ignore_index": ignore_index
        }

        # Get the calibration error. 
        cal_metric_name = list(cal_metric.keys())[0] # kind of hacky
        cal_m_error = cal_metric[cal_metric_name]['func'](**cal_config)['cal_error'] 

        # Wrap all image-level info in a record.
        for quality_metric in quality_metrics_dict.keys():
            cal_record = {
                "cal_metric_type": cal_metric_name.split("_")[-1],
                "cal_metric": cal_metric_name.replace("_", " "),
                "cal_m_score": (1 - cal_m_error),
                "cal_m_error": cal_m_error,
                "qual_metric": quality_metric,
                "qual_score": quality_metrics_dict[quality_metric].item(),
                "data_id": data_id,
                "slice_idx": slice_idx
            }
            # If we are binarizing, then we need to add the label info.
            if label is not None:
                cal_record["label"] = label.item()
                cal_record["true_lab_amount"] = (y_true == label).sum().item() 

            # Add the dataset info to the record
            record = {
                **cal_record, 
                **inference_cfg["calibration"]
                }
            image_level_records.append(record)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def update_image_records(
    image_level_records: list,
    output_dict: dict,
    inference_cfg: dict,
    metric_cfgs: List[dict],
    ignore_label: Optional[int] = None,
):
    # Setup some variables.
    if "ignore_label" in inference_cfg["log"]:
        ignore_label = inference_cfg["log"]["ignore_label"]

    # Setup the image stats config.
    image_stats_cfg = {
        "y_pred": output_dict["y_pred"],
        "y_true": output_dict["y_true"],
        "data_id": output_dict["data_id"],
        "slice_idx": output_dict["slice_idx"], # None if not a volume
        "inference_cfg": inference_cfg,
        "metric_cfgs": metric_cfgs,
        "image_level_records": image_level_records,
        "ignore_index": ignore_label
    }

    # Loop through each label in the prediction or just once if we are not binarizing.
    if inference_cfg["calibration"]["binarize"]:
        # Go through each label in the prediction.
        unique_labels = torch.unique(output_dict["y_hard"])
        if ignore_label is not None:
            unique_labels = unique_labels[unique_labels != ignore_label]
        # Loop through unique labels.
        for label in unique_labels:
            image_stats_cfg["y_pred"] = binarize(output_dict["y_pred"], label, discretize=False)
            image_stats_cfg["y_true"] = binarize(output_dict["y_true"], label, discretize=True)
            image_stats_cfg["label"] = label
            get_image_stats(**image_stats_cfg)
    else:
        # Loop through unique labels.
        get_image_stats(**image_stats_cfg)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def update_pixel_meters(
    pixel_meter_dict: dict,
    output_dict: dict,
    inference_cfg: dict,
    conf_bins: torch.Tensor,
    conf_bin_widths: torch.Tensor,
):
    # Setup variables.
    n_width = inference_cfg["calibration"]["neighborhood_width"]
    H, W = output_dict["y_hard"].shape[-2:]
    if "ignore_label" in inference_cfg["log"]:
        ignore_label = inference_cfg["log"]["ignore_label"]
    else:
        ignore_label = None

    # Calculate the pixel-wise accuracy map.
    acc_map = (output_dict["y_hard"] == output_dict["y_true"])

    # Get the pixel-wise number of matching neighbors map. Edge pixels have maximally 5 neighbors.
    matching_neighbors_0pad = count_matching_neighbors(
        output_dict["y_hard"], 
        neighborhood_width=n_width,
        )

    # Get the pixel-weightings by the number of neighbors in blobs. Edge pixels have minimum 1 neighbor.
    pix_weights = get_uni_pixel_weights(
        output_dict["y_hard"], 
        uni_w_attributes=["labels", "neighbors"],
        neighborhood_width=n_width,
        ignore_index=ignore_label
        )

    # If the confidence map is mulitclass, then we need to do some extra work.
    conf_map = output_dict["y_pred"]
    if conf_map.shape[1] > 1:
        conf_map = torch.max(conf_map, dim=1, keepdim=True)[0]

    # Figure out where each pixel belongs (in confidence)
    bin_ownership_map = find_bins(
        confidences=conf_map, 
        bin_starts=conf_bins,
        bin_widths=conf_bin_widths
        )

    # Build the valid map.
    if ignore_label is not None:
        valid_idx_map = (output_dict["y_hard"].squeeze() != ignore_label)
    else:
        valid_idx_map = torch.ones((H, W)).bool()

    # Iterate through each pixel in the image.
    for (ix, iy) in np.ndindex((H, W)):
        # Only consider pixels that are valid (not ignored)
        if valid_idx_map[ix, iy]:
            # Get the label, neighbors, neighbor_weighted proportion, and confidence bin for this pixel.
            pix_pred_label = output_dict["y_hard"][..., ix, iy].item()
            pix_c_bin = bin_ownership_map[ix, iy].item()
            pix_lab_neighbors = matching_neighbors_0pad[ix, iy].item()
            # Create a unique key for the combination of label, neighbors, and confidence_bin
            conf_key = (pix_pred_label, pix_lab_neighbors, pix_c_bin, "confidence")
            acc_key = (pix_pred_label, pix_lab_neighbors, pix_c_bin, "accuracy")
            weighted_acc_key = (pix_pred_label, pix_lab_neighbors, pix_c_bin, "weighted accuracy")
            weighted_conf_key = (pix_pred_label, pix_lab_neighbors, pix_c_bin, "weighted confidence")
            # If this key doesn't exist in the dictionary, add it
            if conf_key not in pixel_meter_dict:
                for meter_key in [acc_key, conf_key, weighted_acc_key, weighted_conf_key]:
                    pixel_meter_dict[meter_key] = StatsMeter()
            # Finally, add the points to the meters.
            pixel_meter_dict[acc_key].add(acc_map[..., ix, iy].item()) 
            pixel_meter_dict[conf_key].add(conf_map[..., ix, iy].item())
            # Add the weighted accuracy and confidence
            pixel_meter_dict[weighted_acc_key].add(acc_map[..., ix, iy].item(), weight=pix_weights[ix, iy].item()) 
            pixel_meter_dict[weighted_conf_key].add(conf_map[..., ix, iy].item(), weight=pix_weights[ix, iy].item()) 


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_calibration_item_info(
    output_dict: dict,
    inference_cfg: dict,
    metric_cfgs: List[dict],
    conf_bins: torch.Tensor,
    conf_bin_widths: torch.Tensor,
    image_level_records: Optional[list] = None,
    pixel_meter_dict: Optional[dict] = None,
    ):
    ########################
    # IMAGE LEVEL TRACKING #
    ########################
    if image_level_records is not None:
        update_image_records(
            image_level_records=image_level_records,
            output_dict=output_dict,
            inference_cfg=inference_cfg,
            metric_cfgs=metric_cfgs,
        ) 
    ########################
    # PIXEL LEVEL TRACKING #
    ########################
    if pixel_meter_dict is not None:
        update_pixel_meters(
            pixel_meter_dict=pixel_meter_dict,
            output_dict=output_dict,
            inference_cfg=inference_cfg,
            conf_bins=conf_bins,
            conf_bin_widths=conf_bin_widths,
        )