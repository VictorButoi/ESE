# torch imports
import torch
# ionpy imports
from ionpy.util import Config
from ionpy.util.torchutils import to_device
# local imports
from .analysis_utils.inference_utils import cal_stats_init 
from ..experiment.utils import show_inference_examples, reduce_ensemble_preds
from .image_records import get_image_stats
from .pixel_records import (
    update_toplabel_pixel_meters,
    update_cw_pixel_meters
)
# Misc imports
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Optional
from pydantic import validate_arguments
import matplotlib.pyplot as plt
    

def save_records(records, log_dir):
    # Save the items in a pickle file.  
    df = pd.DataFrame(records)
    # Save or overwrite the file.
    df.to_pickle(log_dir)


def save_dict(dict, log_dir):
    # save the dictionary to a pickl file at logdir
    with open(log_dir, 'wb') as f:
        pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)


def save_trackers(output_root, trackers):
    save_records(trackers["image_level_records"], output_root / "image_stats.pkl")
    save_dict(trackers["cw_pixel_meter_dict"], output_root / "cw_pixel_meter_dict.pkl")
    save_dict(trackers["tl_pixel_meter_dict"], output_root / "tl_pixel_meter_dict.pkl")


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_cal_stats(
    cfg: Config,
) -> None:
    # Get the config dictionary
    inference_cfg_dict = cfg.to_dict()

    # initialize the calibration statistics.
    cal_stats_components = cal_stats_init(inference_cfg_dict)

    # setup the save dir.
    trackers = cal_stats_components["trackers"]
    output_root = cal_stats_components["output_root"]
    all_dataloaders = cal_stats_components["dataloaders"]
    all_supports = cal_stats_components["supports"] # Optional, None if not dealing with incontext data.

    # Loop through the data, gather your stats!
    if inference_cfg_dict["log"]["gether_inference_stats"]:
        data_counter = 0
        with torch.no_grad():
            for split in all_dataloaders:
                split_dataloader = all_dataloaders[split]
                for batch_idx, batch in enumerate(split_dataloader):
                    # if batch["data_id"][0] == "114":
                    # if batch["data_id"][0] == "munster_000143_000019":
                    print(f"Split: {split} | Working on batch #{batch_idx} out of", len(split_dataloader), "({:.2f}%)".format(batch_idx / len(split_dataloader) * 100), end="\r")
                    if isinstance(batch, list):
                        batch = {
                            "img": batch[0],
                            "label": batch[1],
                            "data_id": batch[2],
                        }
                    batch["split"] = split
                    batch["batch_idx"] = batch_idx
                    # Gather the forward item.
                    forward_item = {
                        "exp": cal_stats_components["inference_exp"],
                        "batch": batch,
                        "inference_cfg": inference_cfg_dict,
                        "trackers": trackers,
                        "data_counter": data_counter,
                        "output_root": output_root
                    }
                    # Run the forward loop
                    input_type = inference_cfg_dict['data']['input_type']
                    if input_type == 'volume':
                        volume_forward_loop(**forward_item)
                    elif input_type == 'image':
                        if inference_cfg_dict['model']['_type'] == 'incontext':
                            incontext_image_forward_loop(support=all_supports[split], **forward_item)
                        else:
                            standard_image_forward_loop(**forward_item)
                    else:
                        raise ValueError(f"Input type {cal_stats_components['input_type']} not recognized.")
        # Save the records at the end too
        if inference_cfg_dict["log"]["log_image_stats"]:
            save_trackers(output_root, trackers=trackers)

    # After the forward loop, we can calculate the global calibration metrics.
    if inference_cfg_dict["log"]["summary_compute_global_metrics"]:
        # After the final pixel_meters have been saved, we can calculate the global calibration metrics and
        # insert them into the saved image_level_record dataframe.
        image_stats_dir = output_root / "image_stats.pkl"
        log_image_df = pd.read_pickle(image_stats_dir)
        # Loop through the calibration metrics and add them to the dataframe.
        for cal_metric_name, cal_metric_dict in inference_cfg_dict["global_cal_metrics"].items():
            # Add a dummy column to the dataframe.
            log_image_df[cal_metric_name] = np.nan
            # Iterate through the splits, and replace the rows corresponding to the split with the cal losses.
            for split in all_dataloaders.keys():
                cal_type = cal_metric_dict['cal_type']
                if cal_type in ["classwise", "toplabel"]:
                    if cal_type == 'classwise':
                        cal_loss = cal_metric_dict['_fn'](
                            pixel_meters_dict=trackers["cw_pixel_meter_dict"][split]
                        ).item() 
                    else:
                        cal_loss = cal_metric_dict['_fn'](
                            pixel_meters_dict=trackers["tl_pixel_meter_dict"][split]
                        ).item() 
                    # Replace the rows of log_image_df with column 'split' 
                    log_image_df.loc[log_image_df["split"] == split, cal_metric_name] = cal_loss
                else:
                    raise ValueError(f"Calibration type {cal_metric_dict['cal_type']} not recognized.")
        # Save the dataframe again.
        if inference_cfg_dict["log"]["log_pixel_stats"]:
            log_image_df.to_pickle(image_stats_dir)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def volume_forward_loop(
    exp,
    batch,
    inference_cfg,
    trackers,
    data_counter,
    output_root: Path 
):
    # Get the batch info
    image_vol_cpu, label_vol_cpu  = batch["img"], batch["label"]
    image_vol_cuda, label_vol_cuda = to_device((image_vol_cpu, label_vol_cpu), exp.device)
    # Go through each slice and predict the metrics.
    num_slices = image_vol_cuda.shape[1]
    for slice_idx in range(num_slices):
        # if slice_idx == 10:
        print(f"-> Working on slice #{slice_idx} out of", num_slices, "({:.2f}%)".format((slice_idx / num_slices) * 100), end="\r")
        # Get the prediction with no gradient accumulation.
        slice_batch = {
            "img": image_vol_cuda[:, slice_idx:slice_idx+1, ...],
            "label": label_vol_cuda[:, slice_idx:slice_idx+1, ...],
            "data_id": batch["data_id"],
            "split": batch["split"]
        } 
        standard_image_forward_loop(
            exp=exp,
            batch=slice_batch,
            inference_cfg=inference_cfg,
            slice_idx=slice_idx,
            trackers=trackers,
            output_root=output_root,    
            data_counter=data_counter
        )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def standard_image_forward_loop(
    exp,
    batch,
    inference_cfg,
    trackers,
    data_counter,
    output_root: Path,
    slice_idx: Optional[int] = None,
):
    # Get the batch info
    image, label_map  = batch["img"], batch["label"]
    # If we have don't have a minimum number of foreground pixels, then we skip this image.
    if torch.sum(label_map != 0) >= inference_cfg["log"].get("min_fg_pixels", 0):

        # Get your image label pair and define some regions.
        if image.device != exp.device:
            image, label_map = to_device((image, label_map), exp.device)

        # Get the prediction with no gradient accumulation.
        predict_args = {'multi_class': True}
        if inference_cfg["model"]["ensemble"]:
            predict_args["combine_fn"] = "identity"

        # Do a forward pass.
        with torch.no_grad():
            exp_output =  exp.predict(image, **predict_args)

        # Wrap the outputs into a dictionary.
        output_dict = {
            "x": image,
            "y_true": label_map.long(),
            "y_logits": exp_output.get("y_logits", None),
            "y_probs": exp_output.get("y_probs", None),
            "y_hard": exp_output.get("y_hard", None),
            "data_id": batch["data_id"][0], # Works because batchsize = 1
            "split": batch["split"],
            "slice_idx": slice_idx
        }
        # Get the calibration item info.  
        get_calibration_item_info(
            output_dict=output_dict,
            inference_cfg=inference_cfg,
            trackers=trackers,
        )

        # Save the records every so often, to get intermediate results. Note, because of data_ids
        # this can contain fewer than 'log interval' many items.
        if inference_cfg["log"]["log_image_stats"] and (data_counter % inference_cfg['log']['log_interval'] == 0):
            save_trackers(output_root, trackers=trackers)

        data_counter += 1


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def incontext_image_forward_loop(
    exp,
    batch,
    support,
    inference_cfg,
    trackers,
    data_counter,
    output_root: Path,
    slice_idx: Optional[int] = None,
):
    # Get the batch info
    image, label_map  = batch["img"], batch["label"]
    # If we have don't have a minimum number of foreground pixels, then we skip this image.
    if torch.sum(label_map != 0) >= inference_cfg["log"].get("min_fg_pixels", 0):

        # Get your image label pair and define some regions.
        if image.device != exp.device:
            image, label_map = to_device((image, label_map), exp.device)

        # Get the prediction with no gradient accumulation.
        predict_args = {'multi_class': True}
        if inference_cfg["model"]["ensemble"]:
            predict_args["combine_fn"] = "identity"

        logit_list = []
        support_list = []
        with torch.no_grad():
            for j in range(inference_cfg['ensemble']['num_members']):
                # Note: different subjects will use different support sets
                # but different models will use the same support sets
                rng = inference_cfg['experiment']['seed'] * (j + 1) + batch['batch_idx'] 
                sx, sy = support[rng]
                # Send the support set to the device
                sx, sy = to_device((sx, sy), exp.device)
                # If exp has a .predict function, use it, otherwise use the forward function.
                support_args = {
                    "context_images": sx[None],
                    "context_labels": sy[None],
                }
                if hasattr(exp, "predict"):
                    y_logits = exp.predict(**support_args, x=image, multi_class=True)
                else:
                    y_logits = exp.model(**support_args, image=image)
                logit_list.append(y_logits)
                support_list.append((sx, sy))

        logit_ensemble_tensor = torch.stack(logit_list, dim=0).permute(1, 2, 0, 3, 4) # (B, 1, E, H, W)
        # Make the predictions (B, 2, E, H, W) by having the first channel be the background and second be the foreground.
        logit_ensemble_tensor = torch.cat([1 - logit_ensemble_tensor, logit_ensemble_tensor], dim=1) # (B, 2, E, H, W) 
        # Wrap the outputs into a dictionary.
        output_dict = {
            "x": image,
            "y_true": label_map.long(),
            "y_logits": logit_ensemble_tensor,
            "y_probs": None,
            "y_hard": None,
            "data_id": batch["data_id"][0], # Works because batchsize = 1
            "split": batch["split"],
            "slice_idx": slice_idx,
            "support_list": support_list
        }
        # Get the calibration item info.  
        get_calibration_item_info(
            output_dict=output_dict,
            inference_cfg=inference_cfg,
            trackers=trackers,
        )

        # Save the records every so often, to get intermediate results. Note, because of data_ids
        # this can contain fewer than 'log interval' many items.
        if inference_cfg["log"]["log_image_stats"] and (data_counter % inference_cfg['log']['log_interval'] == 0):
            save_trackers(output_root, trackers=trackers)

        data_counter += 1

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_calibration_item_info(
    output_dict,
    inference_cfg,
    trackers
):
    # Get the calibration item info.
    split = output_dict["split"]
    ###########################
    # VISUALIZING IMAGE PREDS #
    ###########################
    if inference_cfg["log"].get("show_examples", False):
        show_inference_examples(
            output_dict, 
            inference_cfg=inference_cfg
        )
    # start = time.time()
    # print("Started Image Level Records")
    ########################
    # IMAGE LEVEL TRACKING #
    ########################
    if "image_level_records" in trackers:
        image_cal_metrics_dict = get_image_stats(
            output_dict=output_dict,
            inference_cfg=inference_cfg,
            image_level_records=trackers["image_level_records"]
        ) 
    # end = time.time()
    # print("Finished Image Level Records. Took", end - start, "seconds.")
    ###############################################################################################
    # If we are ensembling, then we need to reduce the predictions of the individual predictions. #
    ###############################################################################################
    if inference_cfg["model"]["ensemble"]:
        # Get the reduced predictions
        output_dict = {
            **reduce_ensemble_preds(
                output_dict, 
                inference_cfg=inference_cfg,
            ),
            "y_true": output_dict["y_true"],
            "data_id": output_dict["data_id"],
            "split": output_dict["split"],
            "slice_idx": output_dict["slice_idx"]
        }
    ########################
    # PIXEL LEVEL TRACKING #
    ########################
    # start = time.time()
    # print("Started Pixel Records")
    if "tl_pixel_meter_dict" in trackers:
        image_tl_pixel_meter_dict = update_toplabel_pixel_meters(
            output_dict=output_dict,
            inference_cfg=inference_cfg,
            pixel_level_records=trackers["tl_pixel_meter_dict"][split]
        )
    ###########################
    # CW PIXEL LEVEL TRACKING #
    ###########################
    if "cw_pixel_meter_dict" in trackers:
        image_cw_pixel_meter_dict = update_cw_pixel_meters(
            output_dict=output_dict,
            inference_cfg=inference_cfg,
            pixel_level_records=trackers["cw_pixel_meter_dict"][split]
        )
    ##################################################################
    # SANITY CHECK THAT THE CALIBRATION METRICS AGREE FOR THIS IMAGE #
    ##################################################################
    if "image_level_records" in trackers and\
        "tl_pixel_meter_dict" in trackers and\
         "cw_pixel_meter_dict" in trackers: 
        global_cal_sanity_check(
            data_id=output_dict["data_id"],
            slice_idx=output_dict["slice_idx"],
            inference_cfg=inference_cfg, 
            image_cal_metrics_dict=image_cal_metrics_dict, 
            image_tl_pixel_meter_dict=image_tl_pixel_meter_dict,
            image_cw_pixel_meter_dict=image_cw_pixel_meter_dict
        )


def global_cal_sanity_check(
    data_id,
    slice_idx,
    inference_cfg,
    image_cal_metrics_dict,
    image_tl_pixel_meter_dict,
    image_cw_pixel_meter_dict
):
    # Iterate through all the calibration metrics and check that the pixel level calibration score
    # is the same as the image level calibration score (only true when we are working with a single
    # image.
    for cal_metric_name  in inference_cfg["image_cal_metrics"].keys():
        assert len(cal_metric_name.split("_")) == 2, f"Calibration metric name {cal_metric_name} not formatted correctly."
        metric_base = cal_metric_name.split("_")[-1]
        assert metric_base in inference_cfg["global_cal_metrics"], f"Metric base {metric_base} not found in global calibration metrics."
        global_metric_dict = inference_cfg["global_cal_metrics"][metric_base]
        # Get the calibration error in two views. 
        image_cal_score = image_cal_metrics_dict[cal_metric_name]
        # Choose which pixel meter dict to use.
        if global_metric_dict['cal_type'] == 'classwise':
            # Recalculate the calibration score using the pixel meter dict.
            meter_cal_score = global_metric_dict['_fn'](pixel_meters_dict=image_cw_pixel_meter_dict)
        elif global_metric_dict['cal_type'] == 'toplabel':
            # Recalculate the calibration score using the pixel meter dict.
            meter_cal_score = global_metric_dict['_fn'](pixel_meters_dict=image_tl_pixel_meter_dict)
        else:
            raise ValueError(f"Calibration type {global_metric_dict['cal_type']} not recognized.")
        if torch.abs(image_cal_score - meter_cal_score) >= 1e-3: # Allow for some numerical error.
            raise ValueError(f"WARNING on data id {data_id}, slice {slice_idx}: CALIBRATION METRIC '{cal_metric_name}' DOES NOT MATCH FOR IMAGE AND PIXEL LEVELS."+\
            f" Pixel level calibration score ({meter_cal_score}) does not match image level score ({image_cal_score}).")

