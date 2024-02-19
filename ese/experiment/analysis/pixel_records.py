# Misc imports
import numpy as np
import matplotlib.pyplot as plt
from pydantic import validate_arguments
# torch imports
import torch
# ionpy imports
from ionpy.util import StatsMeter
# local imports
from ..metrics.utils import (
    get_bin_per_sample, 
    get_conf_region_np,
    agg_neighbors_preds
)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def update_toplabel_pixel_meters(
    output_dict: dict,
    inference_cfg: dict,
    pixel_level_records 
):
    calibration_cfg = inference_cfg['global_calibration']
    y_probs = output_dict["y_probs"]
    y_hard = output_dict["y_hard"]
    y_true = output_dict["y_true"]

    # If the confidence map is mulitclass, then we need to do some extra work.
    if y_probs.shape[1] > 1:
        toplabel_prob_map = torch.max(y_probs, dim=1)[0]
    else:
        toplabel_prob_map = y_probs.squeeze(1) # Remove the channel dimension.

    # Define the confidence interval (if not provided).
    C = y_probs.shape[1]
    if "conf_interval" not in calibration_cfg:
        if C == 1:
            lower_bound = 0
        else:
            lower_bound = 1 / C
        upper_bound = 1
        # Set the confidence interval.
        calibration_cfg["conf_interval"] = (lower_bound, upper_bound)

    # Figure out where each pixel belongs (in confidence)
    toplabel_bin_ownership_map = get_bin_per_sample(
        pred_map=toplabel_prob_map,
        class_wise=False,
        num_prob_bins=calibration_cfg['num_prob_bins'], 
        start=calibration_cfg['conf_interval'][0], 
        end=calibration_cfg['conf_interval'][1]
    )
    # Get the pixel-wise number of PREDICTED matching neighbors.
    pred_num_neighb_map = agg_neighbors_preds(
        pred_map=output_dict["y_hard"].squeeze(1), # Remove the channel dimension. 
        class_wise=False,
        binary=False,
        neighborhood_width=calibration_cfg["neighborhood_width"],
        discrete=True,
    )
    # Get the pixel-wise number of PREDICTED matching neighbors.
    true_num_neighb_map = agg_neighbors_preds(
        pred_map=output_dict["y_true"].squeeze(1), # Remove the channel dimension. 
        class_wise=False,
        binary=False,
        neighborhood_width=calibration_cfg["neighborhood_width"],
        discrete=True,
    )

    # Calculate the accuracy map.
    ############################################################################3
    toplabel_freq_map = (y_hard == y_true).squeeze(1).cpu().numpy() # Remove the channel dimension. B x H x W
    toplabel_prob_map = toplabel_prob_map.cpu().numpy()

    # Place all necessary tensors on the CPU as numpy arrays.
    y_true = y_true.squeeze(1).cpu().numpy() # Remove the channel dimension. B x H x W
    y_hard = y_hard.squeeze(1).cpu().numpy() # Remove the channel dimension. B x H x W
    toplabel_bin_ownership_map = toplabel_bin_ownership_map.cpu().numpy()
    true_num_neighb_map = true_num_neighb_map.cpu().numpy()
    pred_num_neighb_map = pred_num_neighb_map.cpu().numpy()

    # Make a cross product of the unique iterators using itertools
    combo_array = np.stack([
        y_true,
        y_hard,
        true_num_neighb_map,
        pred_num_neighb_map,
        toplabel_bin_ownership_map,
    ]) # 5 x B x H x W
    # Reshape the numpy array to be 5 x (B x H x W)
    combo_array = combo_array.reshape(combo_array.shape[0], -1)
    # Get the unique vectors along the pixel dimensions.
    unique_combinations = np.unique(combo_array, axis=1).T
    # Make a version of pixel meter dict for this image (for error checking)
    image_tl_meter_dict = {}
    for bin_combo in unique_combinations:
        bin_combo = tuple(bin_combo)
        true_lab, pred_lab, true_nn, pred_nn, bin_idx = bin_combo
        # Get the region of image corresponding to the confidence
        bin_conf_region = get_conf_region_np(
            conditional_region_dict={
                "bin_idx": (bin_idx, toplabel_bin_ownership_map),
                "true_label": (true_lab, y_true),
                "pred_label": (pred_lab, y_hard),
                "true_num_neighbors": (true_nn, true_num_neighb_map),
                "pred_num_neighbors": (pred_nn, pred_num_neighb_map)
            }
        )
        if bin_conf_region.sum() > 0:
            # Add bin specific keys to the dictionary if they don't exist.
            freq_key = bin_combo + ("accuracy",)
            conf_key = bin_combo + ("confidence",)

            # If this key doesn't exist in the dictionary, add it
            if conf_key not in pixel_level_records:
                for meter_key in [freq_key, conf_key]:
                    pixel_level_records[meter_key] = StatsMeter()

            # Add the keys for the image level tracker.
            if conf_key not in image_tl_meter_dict:
                for meter_key in [freq_key, conf_key]:
                    image_tl_meter_dict[meter_key] = StatsMeter()

            # (acc , conf)
            tl_freq = toplabel_freq_map[bin_conf_region]
            tl_conf = toplabel_prob_map[bin_conf_region]
            # Finally, add the points to the meters.
            pixel_level_records[freq_key].addN(tl_freq, batch=True) 
            pixel_level_records[conf_key].addN(tl_conf, batch=True)
            # Add to the local image meter dict.
            image_tl_meter_dict[freq_key].addN(tl_freq, batch=True)
            image_tl_meter_dict[conf_key].addN(tl_conf, batch=True)
        
    return image_tl_meter_dict


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def update_cw_pixel_meters(
    output_dict: dict,
    inference_cfg: dict,
    pixel_level_records 
):
    calibration_cfg = inference_cfg['global_calibration']
    num_neighbor_classes = calibration_cfg["neighborhood_width"]**2
    y_probs = output_dict["y_probs"]
    y_true = output_dict["y_true"]
    C = y_probs.shape[1]

    # BIN CONFIDENCE, both the actual bins and the smoothed local conf bins.
    ############################################################################3
    bin_args = {
        "start": 0.0,
        "end": 1.0,
        "class_wise": True
    }
    conf_bin_map = get_bin_per_sample(
        pred_map=y_probs,
        num_prob_bins=calibration_cfg['num_prob_bins'],
        **bin_args
    ).cpu().numpy()

    local_conf_bin_map = get_bin_per_sample(
        pred_map=agg_neighbors_preds(
                    pred_map=y_probs, # B x H x W
                    neighborhood_width=calibration_cfg["neighborhood_width"],
                    discrete=False,
                    class_wise=True
                ),
        num_prob_bins=num_neighbor_classes,
        **bin_args
    ).cpu().numpy()
    
    # NEIGHBORHOOD INFO. Get the predicted and actual number of label neighbors.
    ###########################################################################3
    agg_neighbor_args = {
        "class_wise": True,
        "discrete": True,
        "neighborhood_width": calibration_cfg["neighborhood_width"],
        "num_classes": C

    }
    true_nn_map = agg_neighbors_preds(
                    pred_map=output_dict["y_true"].squeeze(1), # B x H x W
                    **agg_neighbor_args
                ).cpu().numpy()

    pred_nn_map = agg_neighbors_preds(
                    pred_map=output_dict["y_hard"].long().squeeze(1), # B x H x W
                    **agg_neighbor_args
                ).cpu().numpy() 

    # CALIBRATION VARS.
    ###########################################################################3
    classwise_freq_map = torch.nn.functional.one_hot(
        y_true.squeeze(1).long(), C
    ).permute(0, 3, 1, 2).cpu().numpy() # (B x H x W x C) -> (B x C x H x W)
    # Place all necessary tensors on the CPU as numpy arrays.
    classwise_prob_map = y_probs.cpu().numpy()

    # Make a version of pixel meter dict for this image (for error checking)
    image_cw_meter_dict = {}
    for lab_idx in range(C):
        lab_conf_map = classwise_prob_map[:, lab_idx, ...]
        lab_freq_map = classwise_freq_map[:, lab_idx, ...]
        # Get the region of image corresponding to the confidence
        lab_true_nn_map = true_nn_map[:, lab_idx, ...]
        lab_pred_nn_map = pred_nn_map[:, lab_idx, ...]
        lab_bin_ownership_map = conf_bin_map[:, lab_idx, ...]
        lab_loc_bin_ownership_map = local_conf_bin_map[:, lab_idx, ...]
        # Calculate the unique combinations.
        lab_combo_array = np.stack([
            lab_true_nn_map,
            lab_pred_nn_map,
            lab_loc_bin_ownership_map,
            lab_bin_ownership_map,
        ]) # 4 x B x H x W
        # Reshape the numpy array to be 4 x (B x H x W)
        lab_combo_array = lab_combo_array.reshape(lab_combo_array.shape[0], -1)
        # Get the unique vectors along the pixel dimensions.
        unique_lab_combinations = np.unique(lab_combo_array, axis=1).T
        # Iterate through the unique combinations of the bin ownership map.
        for bin_combo in unique_lab_combinations:
            bin_combo = tuple(bin_combo)
            true_nn, pred_nn, loc_conf_bin_idx, bin_idx = bin_combo
            # Get the region of image corresponding to the confidence
            lab_bin_conf_region = get_conf_region_np(
                conditional_region_dict={
                    "bin_idx": (bin_idx, lab_bin_ownership_map),
                    "true_num_neighbors": (true_nn, lab_true_nn_map),
                    "pred_num_neighbors": (pred_nn, lab_pred_nn_map),
                    "loc_conf_bin_idx": (loc_conf_bin_idx, lab_loc_bin_ownership_map)
                }
            )
            if lab_bin_conf_region.sum() > 0:
                # Add bin specific keys to the dictionary if they don't exist.
                freq_key = (lab_idx,) + bin_combo + ("accuracy",)
                conf_key = (lab_idx,) + bin_combo + ("confidence",)

                # If this key doesn't exist in the dictionary, add it
                if conf_key not in pixel_level_records:
                    for meter_key in [freq_key, conf_key]:
                        pixel_level_records[meter_key] = StatsMeter()

                # Add the keys for the image level tracker.
                if conf_key not in image_cw_meter_dict:
                    for meter_key in [freq_key, conf_key]:
                        image_cw_meter_dict[meter_key] = StatsMeter()

                # (acc , conf)
                cw_freq = lab_freq_map[lab_bin_conf_region]
                cw_conf = lab_conf_map[lab_bin_conf_region]
                # Finally, add the points to the meters.
                pixel_level_records[freq_key].addN(cw_freq, batch=True) 
                pixel_level_records[conf_key].addN(cw_conf, batch=True)
                # Add to the local image meter dict.
                image_cw_meter_dict[freq_key].addN(cw_freq, batch=True)
                image_cw_meter_dict[conf_key].addN(cw_conf, batch=True)
    return image_cw_meter_dict