# Misc imports
import yaml
import pickle
import einops
import pathlib
import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import validate_arguments
from typing import Any, Optional, List, Tuple
# torch imports
import torch
# ionpy imports
from ionpy.util import Config, StatsMeter
from ionpy.analysis import ResultsLoader
from ionpy.util.config import config_digest, HDict, valmap
from ionpy.util.torchutils import to_device
from ionpy.metrics import dice_score, pixel_accuracy
from ionpy.metrics.segmentation import balanced_pixel_accuracy
from ionpy.experiment.util import absolute_import, generate_tuid
# local imports
from .utils import dataloader_from_exp
from ..metrics.utils import get_bins, find_bins, count_matching_neighbors
from ..experiment.ese_exp import CalibrationExperiment


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


def load_cal_inference_stats(log_dir, return_metadata=True):
    log_dir = pathlib.Path(log_dir)

    inference_pixel_dicts = {}
    inference_image_df = pd.DataFrame([])
    metadata_df = pd.DataFrame([])
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
            metadata_df = pd.concat([metadata_df, cfg_df])

            # Loop through the different splits and load the image stats.
            image_stats_df = pd.DataFrame([])
            for image_stats_split in log_set.glob("image_stats_split*"):
                image_split_df = pd.read_pickle(image_stats_split)
                image_stats_df = pd.concat([image_stats_df, image_split_df])
            image_stats_df["log_set"] = log_set.name
            inference_image_df = pd.concat([inference_image_df, image_stats_df])

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
            # Add the running meter dict to the inference pixel dicts.
            inference_pixel_dicts[log_set.name] = running_meter_dict
    # Return all of the pixel dicts
    if return_metadata:
        return inference_pixel_dicts, inference_image_df, metadata_df 
    else:
        return inference_pixel_dicts, inference_image_df 


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_pixelinfo_df(
    data_points: List[dict],
    num_bins: int,
    conf_interval: List[float]
    ) -> pd.DataFrame:
    # Get the different confidence bins.
    conf_bins, conf_bin_widths = get_bins(
        num_bins=num_bins, 
        start=conf_interval[0], 
        end=conf_interval[1]
        )
    # Go through each datapoint and register information about the pixels.
    perppixel_records = []
    for dp in data_points:
        # Get the interesting info.
        accuracy_map = dp['perpix_accuracies']
        bin_ownership_map = find_bins(
            confidences=dp['conf_map'], 
            bin_starts=conf_bins,
            bin_widths=conf_bin_widths
            )
        # iterate through (x,y) positions from 0 - 255, 0 - 255
        for (ix, iy) in np.ndindex(dp['pred_map'].shape):
            # Record all important info.
            bin_n = int(bin_ownership_map[ix, iy].item())
            bin_ends = [np.round(conf_bins[bin_n].item(), 2), np.round(conf_bins[bin_n].item() + conf_bin_widths[bin_n].item(), 2)]
            for measure in ["confidence", "accuracy"]:
                val = dp['conf_map'][ix, iy] if measure == "confidence" else accuracy_map[ix, iy]
                record = {
                    "subject_id": dp['subject_id'],
                    "x": ix,
                    "y": iy,
                    "bin_num": bin_n,
                    "bin": f"{bin_n}, {bin_ends}",
                    "pred_label": dp['pred_map'][ix, iy],
                    "num_neighbors": dp['matching_neighbors'][ix, iy],
                    "pred_label,num_neighbors": f"{dp['pred_map'][ix, iy]},{dp['matching_neighbors'][ix, iy]}",
                    "measure": measure,
                    "value": val,
                }
                perppixel_records.append(record)
    return pd.DataFrame(perppixel_records) 


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_dataset_perf(
    exp, 
    split="val",
    dataset_cfg: dict = None,
    rearrange_channel_dim=False
    ) -> Tuple[List[dict], List[int]]:
    if dataset_cfg is None:
        # Choose the dataloader from the experiment.
        dataloader = exp.val_dl if split=="val" else exp.train_dl
    else:
        dataloader, _ = dataloader_from_exp(
            exp, 
            new_dset_options=dataset_cfg
            )
    items = []
    unique_labels = set([])
    with torch.no_grad():
        for subj_idx, batch in enumerate(dataloader):
            # Get your image label pair and define some regions.
            x, y = to_device(batch, exp.device)
            # Some datasets where we consider multi-slice we want to batch different slices.
            if rearrange_channel_dim:
                x = einops.rearrange(x, "b c h w -> (b c) 1 h w")
                y = einops.rearrange(y, "b c h w -> (b c) 1 h w")
            # Extract predictions
            pred_map, conf_map = exp.predict(x, multi_class=True, include_probs=True)
            # Squeeze our tensors
            image = x.squeeze().cpu().numpy()
            label_map = y.squeeze().cpu().numpy()
            conf_map = conf_map.squeeze().cpu().numpy()
            pred_map = pred_map.squeeze().cpu().numpy() 
            # Get the unique labels and merge using set union.
            new_unique_labels = set(np.unique(label_map))
            unique_labels = unique_labels.union(new_unique_labels)
            # Get some performance metrics
            accuracy_map = (pred_map == label_map).astype(np.float32)
            matching_neighbors = count_matching_neighbors(pred_map)
            # Wrap it in an item
            items.append({
                "subject_id": subj_idx,
                "image": image,
                "label_map": label_map,
                "conf_map": conf_map,
                "pred_map": pred_map,
                "matching_neighbors": matching_neighbors,
                "perpix_accuracies": accuracy_map,
            })
    # Return the subject info and labels.
    return items, list(unique_labels)


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
    root = save_root.parent
    # Get the configs of the experiment
    rs = ResultsLoader()
    dfc = rs.load_configs(
        root / cfg_dict['model']['exp_name'],
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
        pred_map, conf_map = exp.predict(image_cuda, multi_class=True, include_probs=True)
        get_calibration_item_info(
            conf_map=conf_map,
            pred_map=pred_map,
            label_map=label_map_cuda,
            data_id=data_id,
            inference_cfg=inference_cfg,
            metric_cfgs=metric_cfgs,
            conf_bins=conf_bins,
            conf_bin_widths=conf_bin_widths,
            image_level_records=image_level_records,
            pixel_meter_dict=pixel_meter_dict,
            slice_idx=slice_idx,
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
    pred_map, conf_map = exp.predict(image_cuda, multi_class=True, include_probs=True)
    get_calibration_item_info(
        conf_map=conf_map,
        pred_map=pred_map,
        label_map=label_map_cuda,
        data_id=data_id,
        inference_cfg=inference_cfg,
        metric_cfgs=metric_cfgs,
        conf_bins=conf_bins,
        conf_bin_widths=conf_bin_widths,
        image_level_records=image_level_records,
        pixel_meter_dict=pixel_meter_dict
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_calibration_item_info(
    conf_map: torch.Tensor,
    pred_map: torch.Tensor,
    label_map: torch.Tensor,
    data_id: str,
    inference_cfg: dict,
    metric_cfgs: List[dict],
    conf_bins: torch.Tensor,
    conf_bin_widths: torch.Tensor,
    image_level_records: Optional[list] = None,
    pixel_meter_dict: Optional[dict] = None,
    slice_idx: Optional[int] = None,
    ):
    # Get some metrics of these predictions
    dice = dice_score(pred_map, label_map).item()
    acc = pixel_accuracy(pred_map, label_map).item()
    balanced_acc = balanced_pixel_accuracy(pred_map, label_map).item()
    # Squeeze the tensors
    conf_map = conf_map.squeeze()
    pred_map = pred_map.squeeze()
    label_map = label_map.squeeze()
    ########################
    # IMAGE LEVEL TRACKING #
    ########################
    if image_level_records is not None:
        # Go through each calibration metric and calculate the score.
        for cal_metric in metric_cfgs:
            metric_name = list(cal_metric.keys())[0] # kind of hacky
            for bin_weighting in inference_cfg["calibration"]["bin_weightings"]:
                # Get the calibration metric
                cal_score = cal_metric[metric_name]['func'](
                    num_bins=inference_cfg["calibration"]["num_bins"],
                    conf_interval=[
                        inference_cfg["calibration"]["conf_interval_start"],
                        inference_cfg["calibration"]["conf_interval_end"]
                    ],
                    conf_map=conf_map,
                    pred_map=pred_map,
                    label_map=label_map,
                    weighting=bin_weighting,
                )['cal_score'] 
                # Wrap it in an item
                cal_record = {
                    "accuracy": acc,
                    "bin_weighting": bin_weighting,
                    "cal_metric": metric_name,
                    "cal_score": cal_score,
                    "data_id": data_id,
                    "dice": dice,
                    "w_accuracy": balanced_acc,
                    "slice_idx": slice_idx,
                }
                # Add the dataset info to the record
                record = {**cal_record, **inference_cfg["dataset"]}
                image_level_records.append(record)
    ########################
    # PIXEL LEVEL TRACKING #
    ########################
    if pixel_meter_dict is not None:
        # numpy-ize our tensors
        conf_map = conf_map.cpu().numpy()
        pred_map = pred_map.cpu().numpy()
        label_map = label_map.cpu().numpy()
        # Get the interesting info.
        acc_map = (pred_map == label_map).astype(np.float32)
        matching_neighbors = count_matching_neighbors(pred_map)
        # Figure out where each pixel belongs (in confidence)
        bin_ownership_map = find_bins(
            confidences=conf_map, 
            bin_starts=conf_bins,
            bin_widths=conf_bin_widths
            )
        # Iterate through each pixel in the image.
        for (ix, iy) in np.ndindex(pred_map.shape):
            # Calibration info for the pixel.
            pix_acc = acc_map[ix, iy].item()
            pix_conf = conf_map[ix, iy].item()
            # Get the label, neighbors, and confidence bin for this pixel.
            pix_pred_label = pred_map[ix, iy].item()
            pix_c_bin = bin_ownership_map[ix, iy].item()
            pix_lab_neighbors = matching_neighbors[ix, iy].item()
            # Create a unique key for the combination of label, neighbors, and confidence_bin
            acc_key = (pix_pred_label, pix_lab_neighbors, pix_c_bin, "accuracy")
            conf_key = (pix_pred_label, pix_lab_neighbors, pix_c_bin, "confidence")
            # If this key doesn't exist in the dictionary, add it
            if conf_key not in pixel_meter_dict:
                pixel_meter_dict[acc_key] = StatsMeter()
                pixel_meter_dict[conf_key] = StatsMeter()
            # Finally, add the points to the meters.
            pixel_meter_dict[acc_key].add(pix_acc) 
            pixel_meter_dict[conf_key].add(pix_conf)