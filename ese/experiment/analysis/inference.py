# Misc imports
import yaml
import pickle
import einops
import pathlib
import numpy as np
import pandas as pd
from itertools import product
from pydantic import validate_arguments
from typing import Any, Optional, List
# torch imports
import torch
# ionpy imports
from ionpy.analysis import ResultsLoader
from ionpy.util import Config, StatsMeter
from ionpy.util.torchutils import to_device
from ionpy.util.config import config_digest, HDict, valmap
from ionpy.experiment.util import absolute_import, generate_tuid
# local imports
from ..experiment.ese_exp import CalibrationExperiment
from .utils import (
    binarize, 
    get_edge_aux_info,
    get_image_aux_info, 
    dataloader_from_exp
)
from ..metrics.utils import (
    get_bins, 
    find_bins, 
    count_matching_neighbors, 
    get_uni_pixel_weights
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
            flat_cfg.pop("qual_metrics")
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
    # INITIALIZE THE QUALITY METRICS #
    ##################################
    qual_metric_cfgs = cfg_dict['qual_metrics']
    for q_metric in qual_metric_cfgs:
        for q_key in q_metric.keys():
            q_metric[q_key]['func'] = absolute_import(q_metric[q_key]['func'])
    ##################################
    # INITIALIZE CALIBRATION METRICS #
    ##################################
    cal_metric_cfgs = cfg_dict['cal_metrics']
    for cal_metric in cal_metric_cfgs:
        for c_key in cal_metric.keys():
            cal_metric[c_key]['func'] = absolute_import(cal_metric[c_key]['func'])
    # Place these dictionaries into the config dictionary.
    cfg_dict["qual_metric_cfgs"] = qual_metric_cfgs
    cfg_dict["cal_metric_cfgs"] = cal_metric_cfgs

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
            image_level_records=image_level_records,
            pixel_meter_dict=pixel_meter_dict
        )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_forward_loop(
    exp: CalibrationExperiment,
    batch: Any,
    inference_cfg: dict,
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
        "y_pred": conf_map,
        "y_hard": pred_map,
        "y_true": label_map_cuda.long(),
        "data_id": data_id,
        "slice_idx": None
    }
    # Get the calibration item info.  
    get_calibration_item_info(
        output_dict=output_dict,
        inference_cfg=inference_cfg,
        image_level_records=image_level_records,
        pixel_meter_dict=pixel_meter_dict
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_image_stats(
    y_pred: torch.Tensor,
    y_hard: torch.Tensor,
    y_true: torch.Tensor,
    data_id: Any,
    inference_cfg: dict,
    image_level_records: list,
    slice_idx: Optional[int] = None,
    label: Optional[int] = None,
    ignore_index: Optional[int] = None,
):
    # Define the cal config.
    qual_input_config = {
        "y_pred": y_pred,
        "y_true": y_true,
        "ignore_index": ignore_index
    }

    # Calculate some qualities about the image, used for various bookeeping, that can be reused.
    image_info_dict = get_image_aux_info(
        y_hard=y_hard,
        y_true=y_true,
        neighborhood_width=inference_cfg["calibration"]["neighborhood_width"],
        ignore_index=ignore_index
    ) 
    edge_info_dict = get_edge_aux_info(
        y_hard=y_hard,
        y_true=y_true,
        neighborhood_width=inference_cfg["calibration"]["neighborhood_width"],
        ignore_index=ignore_index
    ) 

    # Define the cal config.
    cal_input_config = {
        "y_pred": y_pred,
        "y_true": y_true,
        "num_bins": inference_cfg["calibration"]["num_bins"],
        "conf_interval":[
            inference_cfg["calibration"]["conf_interval_start"],
            inference_cfg["calibration"]["conf_interval_end"]
        ],
        "stats_info_dict": {
            "image_info": image_info_dict,
            "edge_info": edge_info_dict
        },
        "square_diff": inference_cfg["calibration"]["square_diff"],
        "ignore_index": ignore_index
    }
    # Go through each calibration metric and calculate the score.
    qual_metric_scores_dict = {}
    for qual_metric in inference_cfg["qual_metric_cfgs"]:
        # Get the calibration error. 
        q_met_name = list(qual_metric.keys())[0] # kind of hacky
        if qual_metric[q_met_name]['metric_type'] == 'calibration':
            # Higher is better for scores.
            qual_metric_scores_dict[q_met_name] = 1 - qual_metric[q_met_name]['func'](**cal_input_config).item() 
        else:
            qual_metric_scores_dict[q_met_name] = qual_metric[q_met_name]['func'](**qual_input_config).item()


    # Get the amount of label, assuming 0 is a background class.
    if label is not None:
        true_lab_amount = (y_true > 0).sum().item()
        pred_lab_amount = (y_hard > 0).sum().item()
        assert not(pred_lab_amount == 0 and inference_cfg["calibration"]["binarize"]),\
            "Predicted label amount can not be 0 if we are binarizing."
        true_log_lab_amount = 0 if true_lab_amount == 0 else np.log(true_lab_amount)

    # Go through each calibration metric and calculate the score.
    cal_metric_errors_dict = {}
    for cal_metric in inference_cfg["cal_metric_cfgs"]:
        # Get the calibration error. 
        cal_met_name = list(cal_metric.keys())[0] # kind of hacky
        cal_metric_errors_dict[cal_met_name] = cal_metric[cal_met_name]['func'](**cal_input_config).item() 

    # Iterate through the cross product of calibration metrics and quality metrics.
    for qm_name, cm_name in list(product(qual_metric_scores_dict.keys(), cal_metric_errors_dict.keys())):
        cal_record = {
            "cal_metric_type": cm_name.split("_")[-1],
            "cal_metric": cm_name.replace("_", " "),
            "qual_metric": qm_name,
            "cal_m_score": (1 - cal_metric_errors_dict[cm_name]),
            "cal_m_error": cal_metric_errors_dict[cm_name],
            "qual_score": qual_metric_scores_dict[qm_name],
            "data_id": data_id,
            "slice_idx": slice_idx
        }
        # If we are binarizing, then we need to add the label info.
        if label is not None:
            cal_record["label"] = label
            cal_record["true_lab_amount"] = true_lab_amount
            cal_record["log_true_lab_amount"] = true_log_lab_amount
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
    ignore_index: Optional[int] = None,
):
    # Setup the image stats config.
    image_stats_cfg = {
        "y_pred": output_dict["y_pred"],
        "y_hard": output_dict["y_hard"],
        "y_true": output_dict["y_true"],
        "data_id": output_dict["data_id"],
        "slice_idx": output_dict["slice_idx"], # None if not a volume
        "inference_cfg": inference_cfg,
        "image_level_records": image_level_records,
        "ignore_index": ignore_index
    }

    # Loop through each label in the prediction or just once if we are not binarizing.
    if inference_cfg["calibration"]["binarize"]:
        # Go through each label in the prediction.
        unique_pred_labels = torch.unique(output_dict["y_hard"]).tolist()
        # Loop through unique labels.
        for label in unique_pred_labels:
            if ignore_index is None or label != ignore_index:
                image_stats_cfg["label"] = label
                image_stats_cfg["y_pred"] = binarize(
                    output_dict["y_pred"], 
                    label=label, 
                    discretize=False
                    )
                image_stats_cfg["y_hard"] = binarize(
                    output_dict["y_hard"], 
                    label=label, 
                    discretize=True
                    )
                image_stats_cfg["y_true"] = binarize(
                    output_dict["y_true"], 
                    label=label, 
                    discretize=True
                    )
                # run inner loop
                get_image_stats(**image_stats_cfg)
    else:
        # If no binarization, then just run the inner loop once.
        get_image_stats(**image_stats_cfg)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def update_pixel_meters(
    pixel_meter_dict: dict,
    output_dict: dict,
    inference_cfg: dict,
    ignore_index: Optional[int] = None,
):
    # Setup variables.
    H, W = output_dict["y_hard"].shape[-2:]

    # If the confidence map is mulitclass, then we need to do some extra work.
    conf_map = output_dict["y_pred"]
    if conf_map.shape[1] > 1:
        conf_map = torch.max(conf_map, dim=1, keepdim=True)[0]

    # Define the confidence bins and bin widths.
    conf_bins, conf_bin_widths = get_bins(
        num_bins=inference_cfg['calibration']['num_bins'], 
        start=inference_cfg['calibration']['conf_interval_start'], 
        end=inference_cfg['calibration']['conf_interval_end']
    )

    # Figure out where each pixel belongs (in confidence)
    bin_ownership_map = find_bins(
        confidences=conf_map, 
        bin_starts=conf_bins,
        bin_widths=conf_bin_widths
        ).cpu().numpy()

    # Get the pixel-wise number of matching neighbors map. Edge pixels have maximally 5 neighbors.
    pred_matching_neighbors_map = count_matching_neighbors(
        lab_map=output_dict["y_hard"], 
        neighborhood_width=inference_cfg["calibration"]["neighborhood_width"],
        ).cpu().numpy()

    # Get the pixel-weightings by the number of neighbors in blobs. Edge pixels have minimum 1 neighbor.
    pix_weights = get_uni_pixel_weights(
        lab_map=output_dict["y_hard"], 
        uni_w_attributes=["labels", "neighbors"],
        neighborhood_width=inference_cfg["calibration"]["neighborhood_width"],
        ignore_index=ignore_index
        ).cpu().numpy()

    # CPU-ize conf_map, y_hard, and y_true
    conf_map = conf_map.cpu().squeeze().numpy()
    y_true = output_dict["y_true"].cpu().squeeze().numpy()
    y_hard = output_dict["y_hard"].cpu().squeeze().numpy()

    # Calculate the accuracy map.
    acc_map = (y_hard == y_true).astype(np.float32)

    # Build the valid map.
    if ignore_index is not None:
        valid_idx_map = (y_hard != ignore_index)
    else:
        valid_idx_map = np.ones((H, W)).astype(np.bool)

    # Iterate through each pixel in the image.
    for (ix, iy) in np.ndindex((H, W)):
        # Only consider pixels that are valid (not ignored)
        if valid_idx_map[ix, iy]:
            # Create a unique key for the combination of label, neighbors, and confidence_bin
            true_label = y_true[ix, iy]
            pred_label = y_hard[ix, iy]
            num_matching_neighbors = pred_matching_neighbors_map[ix, iy]
            prob_bin = bin_ownership_map[ix, iy]
            # Define this dictionary prefix corresponding to a 'kind' of pixel.
            prefix = (true_label, pred_label, num_matching_neighbors, prob_bin)
            # Add bin specific keys to the dictionary if they don't exist.
            acc_key = prefix + ("accuracy",)
            conf_key = prefix + ("confidence",)
            weighted_acc_key = prefix + ("weighted accuracy",)
            weighted_conf_key = prefix + ("weighted confidence",)
            # If this key doesn't exist in the dictionary, add it
            if conf_key not in pixel_meter_dict:
                for meter_key in [acc_key, conf_key, weighted_acc_key, weighted_conf_key]:
                    pixel_meter_dict[meter_key] = StatsMeter()
            # (acc , conf)
            acc = acc_map[ix, iy]
            conf = conf_map[ix, iy]
            pix_w = pix_weights[ix, iy]
            # Finally, add the points to the meters.
            pixel_meter_dict[acc_key].add(acc) 
            pixel_meter_dict[conf_key].add(conf)
            # Add the weighted accuracy and confidence
            pixel_meter_dict[weighted_acc_key].add(acc, weight=pix_w) 
            pixel_meter_dict[weighted_conf_key].add(conf, weight=pix_w) 


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_calibration_item_info(
    output_dict: dict,
    inference_cfg: dict,
    image_level_records: Optional[list] = None,
    pixel_meter_dict: Optional[dict] = None,
    ignore_index: Optional[int] = None,
    ):
    # Setup some variables.
    if "ignore_index" in inference_cfg["log"]:
        ignore_index = inference_cfg["log"]["ignore_index"]
    ########################
    # IMAGE LEVEL TRACKING #
    ########################
    if image_level_records is not None:
        update_image_records(
            image_level_records=image_level_records,
            output_dict=output_dict,
            inference_cfg=inference_cfg,
            ignore_index=ignore_index
        ) 
    ########################
    # PIXEL LEVEL TRACKING #
    ########################
    if pixel_meter_dict is not None:
        update_pixel_meters(
            pixel_meter_dict=pixel_meter_dict,
            output_dict=output_dict,
            inference_cfg=inference_cfg,
            ignore_index=ignore_index
        )