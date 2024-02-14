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
    get_bins, 
    find_bins, 
    get_conf_region_np,
    count_matching_neighbors,
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
    # BIN CONFIDENCE 
    ############################################################################3
    # Define the confidence bins and bin widths.
    toplabel_conf_bins, toplabel_conf_bin_widths = get_bins(
        num_bins=calibration_cfg['num_bins'], 
        start=calibration_cfg['conf_interval'][0], 
        end=calibration_cfg['conf_interval'][1]
    )
    toplabel_bin_ownership_map = find_bins(
        confidences=toplabel_prob_map, 
        bin_starts=toplabel_conf_bins,
        bin_widths=toplabel_conf_bin_widths
    )
    # Cast to int64
    ############################################################################3
    # These are conv ops so they are done on the GPU.

    # Get the pixel-wise number of PREDICTED matching neighbors.
    pred_num_neighb_map = count_matching_neighbors(
        lab_map=output_dict["y_hard"].squeeze(1), # Remove the channel dimension. 
        neighborhood_width=calibration_cfg["neighborhood_width"],
    )
    
    # Get the pixel-wise number of PREDICTED matching neighbors.
    true_num_neighb_map = count_matching_neighbors(
        lab_map=output_dict["y_true"].squeeze(1), # Remove the channel dimension. 
        neighborhood_width=calibration_cfg["neighborhood_width"],
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
        true_lab, pred_lab, true_num_neighb, pred_num_neighb, bin_idx = bin_combo
        # Get the region of image corresponding to the confidence
        bin_conf_region = get_conf_region_np(
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
    y_probs = output_dict["y_probs"]
    y_true = output_dict["y_true"]

    # Figure out where each pixel belongs (in confidence)
    # BIN CONFIDENCE 
    ############################################################################3
    classwise_conf_bins, classwise_conf_bin_widths = get_bins(
        num_bins=calibration_cfg['num_bins'], 
        start=0.0,
        end=1.0,
    )
    classwise_bin_ownership_map = torch.stack([
        find_bins(
            confidences=y_probs[:, l_idx, ...], 
            bin_starts=classwise_conf_bins,
            bin_widths=classwise_conf_bin_widths,
        ) # B x H x W
        for l_idx in range(y_probs.shape[1])], dim=0
    ).cpu().numpy() # C x B x H x W

    # Reshape to look like the y_probs.
    ###########################################################################3
    # These are conv ops so they are done on the GPU.

    # Get the pixel-wise number of PREDICTED matching neighbors.
    C = y_probs.shape[1]

    true_num_neighb_map = torch.stack([
        count_matching_neighbors(
            lab_map=(output_dict["y_true"] == lab_idx).long(),
            neighborhood_width=calibration_cfg["neighborhood_width"],
            binary=True # Ignore the background class.
        )
    for lab_idx in range(C)]) # C x B x H x W

    pred_num_neighb_map = torch.stack([
        count_matching_neighbors(
            lab_map=(output_dict["y_hard"] == lab_idx).long(),
            neighborhood_width=calibration_cfg["neighborhood_width"],
            binary=True # Ignore the background class.
        ) 
    for lab_idx in range(C)]) # C x B x H x W

    long_label_map = y_true.squeeze(1).long() # Squeeze out the channel dimension and convert to long.
    classwise_freq_map = torch.nn.functional.one_hot(
        long_label_map, C
    ).permute(0, 3, 1, 2) # B x C x H x W
    
    # Place all necessary tensors on the CPU as numpy arrays.
    classwise_freq_map = classwise_freq_map.cpu().numpy()
    true_nn_map = true_num_neighb_map.cpu().numpy()
    pred_nn_map = pred_num_neighb_map.cpu().numpy()
    y_probs = y_probs.cpu().numpy()

    # Make a version of pixel meter dict for this image (for error checking)
    image_cw_meter_dict = {}
    for lab_idx in range(C):
        lab_conf_map = y_probs[:, lab_idx, ...]
        lab_freq_map = classwise_freq_map[:, lab_idx, ...]
        lab_true_nn_map = true_nn_map[lab_idx, ...]
        lab_pred_nn_map = pred_nn_map[lab_idx, ...]
        lab_bin_ownership_map = classwise_bin_ownership_map[lab_idx, ...]
        # Calculate the unique combinations.
        lab_combo_array = np.stack([
            lab_true_nn_map,
            lab_pred_nn_map,
            lab_bin_ownership_map,
        ]) # 3 x B x H x W
        # Reshape the numpy array to be 2 x (B x H x W)
        lab_combo_array = lab_combo_array.reshape(lab_combo_array.shape[0], -1)
        # Get the unique vectors along the pixel dimensions.
        unique_lab_combinations = np.unique(lab_combo_array, axis=1).T
        # Iterate through the unique combinations of the bin ownership map.
        for bin_combo in unique_lab_combinations:
            bin_combo = tuple(bin_combo)
            true_nn, pred_nn, bin_idx = bin_combo
            # Get the region of image corresponding to the confidence
            lab_bin_conf_region = get_conf_region_np(
                bin_idx=bin_idx, 
                bin_ownership_map=lab_bin_ownership_map,
                true_num_neighbors_map=lab_true_nn_map, # Note this is off ACTUAL neighbors.
                true_nn=true_nn,
                pred_num_neighbors_map=lab_pred_nn_map,
                pred_nn=pred_nn
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