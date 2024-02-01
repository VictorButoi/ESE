# Misc imports
import pickle
import numpy as np
import pandas as pd
from typing import Any, Optional
from pydantic import validate_arguments
# torch imports
import torch
# ionpy imports
from ionpy.util import Config
from ionpy.util.torchutils import to_device
# local imports
from .analysis_utils.inference_utils import cal_stats_init 
from ..experiment.utils import show_inference_examples
from .image_records import get_image_stats
from .pixel_records import (
    update_toplabel_pixel_meters,
    update_cw_pixel_meters
)
    

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
def get_cal_stats(
    cfg: Config,
) -> None:
    # Get the config dictionary
    cfg_dict = cfg.to_dict()

    # Initialize the calibration statistics.
    cal_stats_components = cal_stats_init(cfg_dict)

    # Setup the save dir.
    output_root = cal_stats_components["output_root"]
    trackers = cal_stats_components["trackers"]
    # Loop through the data, gather your stats!
    with torch.no_grad():
        dataloader = cal_stats_components["dataloader"]
        for batch_idx, batch in enumerate(dataloader):
            print(f"Working on batch #{batch_idx} out of", len(dataloader), "({:.2f}%)".format(batch_idx / len(dataloader) * 100), end="\r")
            # Gather the forward item.
            forward_item = {
                "exp": cal_stats_components["inference_exp"],
                "batch": batch,
                "inference_cfg": cfg_dict,
                "trackers": trackers
            }
            # Run the forward loop
            if cal_stats_components["input_type"] == "volume":
                volume_forward_loop(**forward_item)
            else:
                image_forward_loop(**forward_item)
            # Save the records every so often, to get intermediate results. Note, because of data_ids
            # this can contain fewer than 'log interval' many items.
            if batch_idx % cfg['log']['log_interval'] == 0:
                if "image_level_records" in trackers:
                    save_records(trackers["image_level_records"], output_root / "image_stats.pkl")
                if "cw_pixel_meter_dict" in trackers:
                    save_dict(trackers["cw_pixel_meter_dict"], output_root / "cw_pixel_meter_dict.pkl")
                if "pixel_meter_dict" in trackers:
                    save_dict(trackers["pixel_meter_dict"], output_root / "pixel_meter_dict.pkl")
    # Save the records at the end too
    if "image_level_records" in trackers:
        save_records(trackers["image_level_records"], output_root / "image_stats.pkl")
    if "cw_pixel_meter_dict" in trackers:
        save_dict(trackers["cw_pixel_meter_dict"], output_root / "cw_pixel_meter_dict.pkl")
    if "pixel_meter_dict" in trackers:
        save_dict(trackers["pixel_meter_dict"], output_root / "pixel_meter_dict.pkl")
        # After the final pixel_meters have been saved, we can calculate the global calibration metrics and
        # insert them into the saved image_level_record dataframe.
        image_stats_dir = output_root + "/image_stats.pkl"
        log_image_df = pd.read_pickle(image_stats_dir)
        # Loop through the calibration metrics and add them to the dataframe.
        for cal_metric_name, cal_metric_dict in cfg_dict["global_cal_metrics"].items():
            log_image_df[cal_metric_name] = cal_metric_dict['_fn'](
                pixel_meters_dict=trackers["pixel_meter_dict"]
            ).item() 
        # Save the dataframe again.
        log_image_df.to_pickle(image_stats_dir)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def volume_forward_loop(
    exp: Any,
    batch: Any,
    inference_cfg: dict,
    trackers
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
            trackers=trackers,
        )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_forward_loop(
    exp: Any,
    batch: Any,
    inference_cfg: dict,
    trackers,
    slice_idx: Optional[int] = None,
):
    # Get the batch info
    image, label_map  = batch["img"], batch["label"]
    # Get your image label pair and define some regions.
    if image.device != exp.device:
        image, label_map = to_device((image, label_map), exp.device)
    # Get the prediction with no gradient accumulation.
    predict_args = {'multi_class': True}
    do_ensemble = inference_cfg["model"]["ensemble"]
    if do_ensemble:
        predict_args["combine_fn"] = "identity"
    # Do a forward pass.
    with torch.no_grad():
        exp_output =  exp.predict(image, **predict_args)
    # Wrap the outputs into a dictionary.
    output_dict = {
        "x": image,
        "y_true": label_map.long(),
        "y_pred": exp_output["y_pred"],
        "y_hard": exp_output["y_hard"],
        "data_id": batch["data_id"][0], # Works because batchsize = 1
        "slice_idx": slice_idx
    }
    if do_ensemble:
        output_dict["ens_weights"] = exp.ens_mem_weights
    
    # Get the calibration item info.  
    get_calibration_item_info(
        output_dict=output_dict,
        inference_cfg=inference_cfg,
        trackers=trackers,
    )

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_calibration_item_info(
    output_dict: dict,
    inference_cfg: dict,
    trackers
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
    if "image_level_records" in trackers:
        image_cal_metrics_dict = get_image_stats(
            output_dict=output_dict,
            inference_cfg=inference_cfg,
            image_level_records=trackers["image_level_records"]
        ) 

    ########################
    # PIXEL LEVEL TRACKING #
    ########################
    if "pixel_meter_dict" in trackers:
        image_tl_pixel_meter_dict = update_toplabel_pixel_meters(
            output_dict=output_dict,
            inference_cfg=inference_cfg,
            pixel_level_records=trackers["pixel_meter_dict"]
        )

    ###########################
    # CW PIXEL LEVEL TRACKING #
    ###########################
    if "cw_pixel_meter_dict" in trackers:
        image_cw_pixel_meter_dict = update_cw_pixel_meters(
            output_dict=output_dict,
            inference_cfg=inference_cfg,
            pixel_level_records=trackers["cw_pixel_meter_dict"]
        )
    
    ##################################################################
    # SANITY CHECK THAT THE CALIBRATION METRICS AGREE FOR THIS IMAGE #
    ##################################################################
    if "image_level_records" in trackers and\
        "pixel_meter_dict" in trackers and\
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
    data_id: str,
    slice_idx: Any,
    inference_cfg: dict, 
    image_cal_metrics_dict,
    image_tl_pixel_meter_dict,
    image_cw_pixel_meter_dict
):
    # Iterate through all the calibration metrics and check that the pixel level calibration score
    # is the same as the image level calibration score (only true when we are working with a single
    # image.
    for cal_metric_name  in inference_cfg["image_cal_metrics"].keys():
        metric_base = cal_metric_name.split("_")[-1]
        if metric_base in inference_cfg["global_cal_metrics"]:
            global_metric_dict = inference_cfg["global_cal_metrics"][metric_base]
            # Get the calibration error in two views. 
            image_cal_score = np.round(image_cal_metrics_dict[cal_metric_name], 3)
            # Choose which pixel meter dict to use.
            if "CW" in cal_metric_name:
                # Recalculate the calibration score using the pixel meter dict.
                meter_cal_score = np.round(global_metric_dict['_fn'](pixel_meters_dict=image_cw_pixel_meter_dict).item(), 3)
            else:
                # Recalculate the calibration score using the pixel meter dict.
                meter_cal_score = np.round(global_metric_dict['_fn'](pixel_meters_dict=image_tl_pixel_meter_dict).item(), 3)
            if image_cal_score != meter_cal_score:
                raise ValueError(f"WARNING on data id {data_id}, slice {slice_idx}: CALIBRATION METRIC '{cal_metric_name}' DOES NOT MATCH FOR IMAGE AND PIXEL LEVELS."+\
                f" Pixel level calibration score ({meter_cal_score}) does not match image level score ({image_cal_score}).")

