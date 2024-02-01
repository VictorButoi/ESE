# Misc imports
import pickle
import itertools
import numpy as np
import pandas as pd
from typing import Any, Optional
from pydantic import validate_arguments
# torch imports
import torch
from torch.nn import functional as F
# ionpy imports
from ionpy.util import Config, StatsMeter
from ionpy.util.torchutils import to_device
# local imports
from ..experiment.utils import show_inference_examples
from ..metrics.utils import (
    get_bins, 
    find_bins, 
    get_conf_region,
    count_matching_neighbors,
)
from .analysis_utils.inference_utils import (
    cal_stats_init,
    get_image_aux_info, 
    reduce_ensemble_preds
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

    # Setup the trackers
    image_level_records = cal_stats_components["image_level_records"]
    pixel_meter_dict = cal_stats_components["pixel_meter_dict"]

    # Setup the save directories.
    image_stats_save_dir = cal_stats_components["image_level_dir"]
    pixel_dicts_save_dir = cal_stats_components["pixel_level_dir"]
    
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
                "image_level_records": image_level_records,
                "pixel_meter_dict": pixel_meter_dict
            }
            # Run the forward loop
            if cal_stats_components["input_type"] == "volume":
                volume_forward_loop(**forward_item)
            else:
                image_forward_loop(**forward_item)
            # Save the records every so often, to get intermediate results. Note, because of data_ids
            # this can contain fewer than 'log interval' many items.
            if batch_idx % cfg['log']['log_interval'] == 0:
                if image_level_records is not None:
                    save_records(image_level_records, image_stats_save_dir)
                if pixel_meter_dict is not None:
                    save_dict(pixel_meter_dict, pixel_dicts_save_dir)

    # Save the records at the end too
    if image_level_records is not None:
        save_records(image_level_records, image_stats_save_dir)

    # Save the pixel dict.
    if pixel_meter_dict is not None:
        save_dict(pixel_meter_dict, pixel_dicts_save_dir)
        # After the final pixel_meters have been saved, we can calculate the global calibration metrics and
        # insert them into the saved image_level_record dataframe.
        log_image_df = pd.read_pickle(image_stats_save_dir)
        # Loop through the calibration metrics and add them to the dataframe.
        for cal_metric_name, cal_metric_dict in cfg_dict["global_cal_metrics"].items():
            log_image_df[cal_metric_name] = cal_metric_dict['_fn'](
                pixel_meters_dict=pixel_meter_dict
            ).item() 
        # Save the dataframe again.
        log_image_df.to_pickle(image_stats_save_dir)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def volume_forward_loop(
    exp: Any,
    batch: Any,
    inference_cfg: dict,
    image_level_records,
    pixel_meter_dict
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
    image_level_records,
    pixel_meter_dict,
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
        image_level_records=image_level_records,
        pixel_meter_dict=pixel_meter_dict
    )

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_calibration_item_info(
    output_dict: dict,
    inference_cfg: dict,
    image_level_records,
    pixel_meter_dict
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
            output_dict=output_dict,
            inference_cfg=inference_cfg,
            total_pixel_meter_dict=pixel_meter_dict
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
    image_level_records
):
    # Define the cal config.
    qual_input_config = {
        "y_pred": output_dict["y_pred"], # either (B, C, H, W) or (B, C, E, H, W), if ensembling
        "y_true": output_dict["y_true"], # (B, C, H, W)
    }
    # Define the cal config.
    cal_input_config = qual_input_config.copy() 
    # If not ensembling, we can cache information.
    if not inference_cfg["model"]["ensemble"]:
        cal_input_config["stats_info_dict"] = get_image_aux_info(
            y_pred=output_dict["y_pred"],
            y_hard=output_dict["y_hard"],
            y_true=output_dict["y_true"],
            cal_cfg=inference_cfg["calibration"]
        )
    # If we are ensembling, then we can precalulate the ensemble predictions.
    # (Both individual and reduced)
    if inference_cfg["model"]["ensemble"]:
        # Get the reduced predictions
        ensemble_input_config = {
            'y_pred': reduce_ensemble_preds(
                output_dict, 
                inference_cfg=inference_cfg)['y_pred'],
            'y_true': output_dict['y_true']
        }
        # Gather the individual predictions
        ensemble_member_preds = [
            output_dict["y_pred"][:, :, ens_mem_idx, ...]\
            for ens_mem_idx in range(output_dict["y_pred"].shape[2])
            ]
        # Construct the input cfgs used for calulating metrics.
        ensemble_member_input_cfgs = [
            {
                "y_pred": member_pred, 
                "y_true": output_dict["y_true"],
                "from_logits": True # IMPORTANT, we haven't softmaxed yet.
            } for member_pred in ensemble_member_preds
        ]
    # Dicts for storing ensemble scores.
    grouped_scores_dict = {
        "calibration": {},
        "quality": {}
    }
    #############################################################
    # CALCULATE QUALITY METRICS
    #############################################################
    qual_metric_scores_dict = {}
    for qual_metric_name, qual_metric_dict in inference_cfg["qual_metrics"].items():
        # If we are ensembling, then we need to go through eahc member of the ensemble and calculate individual metrics
        # so we can get group averages.
        if inference_cfg["model"]["ensemble"]:
            # First gather the quality scores per ensemble member.
            #######################################################
            individual_qual_scores = []
            for ens_mem_input_cfg in ensemble_member_input_cfgs:
                member_qual_score = qual_metric_dict['_fn'](**ens_mem_input_cfg).item()
                individual_qual_scores.append(member_qual_score)
            # Now place it in the dictionary.
            grouped_scores_dict['quality'][qual_metric_name] = np.mean(individual_qual_scores)
            # Now get the ensemble quality score.
            qual_metric_scores_dict[qual_metric_name] = qual_metric_dict['_fn'](**ensemble_input_config).item() 
        else:
            # Get the calibration error. 
            if qual_metric_dict['_type'] == 'calibration':
                # Higher is better for scores.
                qual_metric_scores_dict[qual_metric_name] = qual_metric_dict['_fn'](**cal_input_config).item() 
            else:
                qual_metric_scores_dict[qual_metric_name] = qual_metric_dict['_fn'](**qual_input_config).item()
            # If you're showing the predictions, also print the scores.
            if inference_cfg["log"]["show_examples"]:
                print(f"{qual_metric_name}: {qual_metric_scores_dict[qual_metric_name]}")

    #############################################################
    # CALCULATE CALIBRATION METRICS
    #############################################################
    cal_metric_errors_dict = {}
    for cal_metric_name, cal_metric_dict in inference_cfg["image_cal_metrics"].items():
        # If we are ensembling, then we need to go through eahc member of the ensemble and calculate individual metrics
        # so we can get group averages.
        if inference_cfg["model"]["ensemble"]:
            # First gather the calibration scores per ensemble member.
            #######################################################
            individual_cal_scores = []
            for ens_mem_input_cfg in ensemble_member_input_cfgs:
                member_cal_score = cal_metric_dict['_fn'](**ens_mem_input_cfg).item()
                individual_cal_scores.append(member_cal_score)
            # Now place it in the dictionary.
            grouped_scores_dict['calibration'][cal_metric_name] = np.mean(individual_cal_scores)
            # Now get the ensemble calibration error.
            cal_metric_errors_dict[cal_metric_name] = cal_metric_dict['_fn'](**ensemble_input_config).item() 
        else:
            # Get the calibration error. 
            cal_metric_errors_dict[cal_metric_name] = cal_metric_dict['_fn'](**cal_input_config).item() 
    
    assert not (len(qual_metric_scores_dict) == 0 and len(cal_metric_errors_dict) == 0), \
        "No metrics were specified in the config file."
    
    # Calculate the amount of present ground-truth there is in the image per label.
    if inference_cfg["log"]["track_label_amounts"]:
        num_classes = output_dict["y_pred"].shape[1]
        y_true_one_hot = F.one_hot(output_dict["y_true"], num_classes=num_classes) # B x 1 x H x W x C
        label_amounts = y_true_one_hot.sum(dim=(0, 1, 2, 3)) # C
        label_amounts_dict = {f"num_lab_{i}_pixels": label_amounts[i].item() for i in range(num_classes)}
    
    # Add our scores to the image level records.
    metrics_collection ={
        "quality": qual_metric_scores_dict,
        "calibration": cal_metric_errors_dict
    }

    for dict_type, metric_score_dict in metrics_collection.items():
        for met_name in list(metric_score_dict.keys()):
            metrics_record = {
                "image_metric": met_name,
                "metric_score": metric_score_dict[met_name],
            }
            if inference_cfg["model"]["ensemble"]:
                metrics_record["groupavg_image_metric"] = f"GroupAvg_{met_name}"
                metrics_record["groupavg_metric_score"] = grouped_scores_dict[dict_type][met_name]
            # Add the dataset info to the record
            record = {
                "data_id": output_dict["data_id"],
                "slice_idx": output_dict["slice_idx"],
                **metrics_record, 
                **inference_cfg["calibration"]
            }
            if inference_cfg["log"]["track_label_amounts"]:
                record = {**record, **label_amounts_dict}
            # Add the record to the list.
            image_level_records.append(record)
    
    return cal_metric_errors_dict


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def update_pixel_meters(
    output_dict: dict,
    inference_cfg: dict,
    total_pixel_meter_dict
):
    # If this is an ensembled prediction, then first we need to reduce the ensemble
    ####################################################################################
    if inference_cfg["model"]["ensemble"]:
        output_dict = {
            **reduce_ensemble_preds(
                output_dict, 
                inference_cfg=inference_cfg
            ),
            "y_true": output_dict["y_true"]
        }

    calibration_cfg = inference_cfg['calibration']
    y_pred = output_dict["y_pred"].cpu()
    y_hard = output_dict["y_hard"].cpu()
    y_true = output_dict["y_true"].cpu()

    # If the confidence map is mulitclass, then we need to do some extra work.
    if y_pred.shape[1] > 1:
        toplabel_prob_map = torch.max(y_pred, dim=1)[0]
    else:
        toplabel_prob_map = y_pred.squeeze(1) # Remove the channel dimension.

    # Define the confidence interval (if not provided).
    C = y_pred.shape[1]
    if "conf_interval" not in calibration_cfg:
        if C == 1:
            lower_bound = 0
        else:
            lower_bound = 1 / C
        upper_bound = 1
        # Set the confidence interval.
        calibration_cfg["conf_interval"] = (lower_bound, upper_bound)

    # Figure out where each pixel belongs (in confidence)
    # BIN CONFIDENCE 
    ############################################################################3
    # Define the confidence bins and bin widths.
    toplabel_conf_bins, toplabel_conf_bin_widths = get_bins(
        num_bins=calibration_cfg['num_bins'], 
        start=calibration_cfg['conf_interval'][0], 
        end=calibration_cfg['conf_interval'][1],
        device=None
    )
    toplabel_bin_ownership_map = find_bins(
        confidences=toplabel_prob_map, 
        bin_starts=toplabel_conf_bins,
        bin_widths=toplabel_conf_bin_widths,
        device=None
    )
    ############################################################################3
    # These are conv ops so they are done on the GPU.

    # Get the pixel-wise number of PREDICTED matching neighbors.
    pred_num_neighb_map = count_matching_neighbors(
        lab_map=output_dict["y_hard"].squeeze(1), # Remove the channel dimension. 
        neighborhood_width=calibration_cfg["neighborhood_width"],
    ).cpu()
    
    # Get the pixel-wise number of PREDICTED matching neighbors.
    true_num_neighb_map = count_matching_neighbors(
        lab_map=output_dict["y_true"].squeeze(1), # Remove the channel dimension. 
        neighborhood_width=calibration_cfg["neighborhood_width"],
    ).cpu()

    # Calculate the accuracy map.
    # FREQUENCY 
    ############################################################################3
    toplabel_freq_map = (y_hard == y_true)

    # Put all of our tensors on the cpu so we don't have to send them back and forth.
    toplabel_prob_map = toplabel_prob_map.unsqueeze(1) # Add the channel dimension back.
    # Make a cross product of the unique iterators using itertools
    unique_combinations = list(itertools.product(
        np.unique(y_true),
        np.unique(y_hard),
        np.unique(true_num_neighb_map),
        np.unique(pred_num_neighb_map),
        np.unique(toplabel_bin_ownership_map)
    ))
    # Make a version of pixel meter dict for this image (for error checking)
    image_tl_meter_dict = {}
    for bin_combo in unique_combinations:
        true_lab, pred_lab, true_num_neighb, pred_num_neighb, bin_idx = bin_combo
        # Get the region of image corresponding to the confidence
        bin_conf_region = get_conf_region(
            bin_idx=bin_idx, 
            bin_ownership_map=toplabel_bin_ownership_map,
            true_label=true_lab,
            true_lab_map=y_true, # Use ground truth to get the region.
            pred_label=pred_lab,
            pred_lab_map=y_hard, # Use ground truth to get the region.
            true_num_neighbors_map=true_num_neighb_map, # Note this is off ACTUAL neighbors.
            true_nn=true_num_neighb,
            pred_num_neighbors_map=pred_num_neighb_map,
            pred_nn=pred_num_neighb
        )
        if bin_conf_region.sum() > 0:
            # Add bin specific keys to the dictionary if they don't exist.
            acc_key = bin_combo + ("accuracy",)
            conf_key = bin_combo + ("confidence",)

            # If this key doesn't exist in the dictionary, add it
            if conf_key not in total_pixel_meter_dict:
                for meter_key in [acc_key, conf_key]:
                    total_pixel_meter_dict[meter_key] = StatsMeter()

            # Add the keys for the image level tracker.
            if conf_key not in image_tl_meter_dict:
                for meter_key in [acc_key, conf_key]:
                    image_tl_meter_dict[meter_key] = StatsMeter()

            # (acc , conf)
            tl_freq = toplabel_freq_map[bin_conf_region].numpy()
            tl_conf = toplabel_prob_map[bin_conf_region].numpy()
            # Finally, add the points to the meters.
            total_pixel_meter_dict[acc_key].addN(tl_freq) 
            total_pixel_meter_dict[conf_key].addN(tl_conf)
            # Add to the local image meter dict.
            image_tl_meter_dict[acc_key].addN(tl_freq)
            image_tl_meter_dict[conf_key].addN(tl_conf)
        
    return image_tl_meter_dict


def global_cal_sanity_check(
    data_id: str,
    slice_idx: Any,
    inference_cfg: dict, 
    image_cal_metrics_dict,
    image_pixel_meter_dict
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
            meter_cal_score = np.round(global_metric_dict['_fn'](pixel_meters_dict=image_pixel_meter_dict).item(), 3)
            if image_cal_score != meter_cal_score:
                raise ValueError(f"WARNING on data id {data_id}, slice {slice_idx}: CALIBRATION METRIC '{cal_metric_name}' DOES NOT MATCH FOR IMAGE AND PIXEL LEVELS."+\
                f" Pixel level calibration score ({meter_cal_score}) does not match image level score ({image_cal_score}).")

