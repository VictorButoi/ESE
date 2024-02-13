# torch imports
import torch 
# misc imports
import numpy as np
from typing import Optional, Literal
from pydantic import validate_arguments


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def global_binwise_stats(
    pixel_meters_dict: dict,
    class_conditioned: bool,
    neighborhood_conditioned: bool,
    class_wise=False,
    device: Optional[Literal["cpu", "cuda"]] = None,
    **kwargs
    ) -> dict:
    # Return the calibration information.
    cal_info = {
        "pixel_meters_dict": pixel_meters_dict,
        "device": device,
        **kwargs
    }
    # If we are class or neighborhood conditioned, need to specify class_wise.
    if class_conditioned or neighborhood_conditioned:
        cal_info['class_wise'] = class_wise
    # Run the selected global function.
    if not class_conditioned and not neighborhood_conditioned:
        return prob_bin_stats(**cal_info)
    elif not neighborhood_conditioned:
        return class_wise_bin_stats(**cal_info)
    elif not class_conditioned:
        return neighbor_wise_bin_stats(**cal_info)
    else:
        return joint_class_neighbor_bin_stats(**cal_info)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def prob_bin_stats(
    pixel_meters_dict: dict,
    num_bins: int,
    square_diff: bool = False,
    edge_only: bool = False,
    neighborhood_width: Optional[int] = None,
    device: Optional[Literal["cpu", "cuda"]] = None,
) -> dict:
    accumulated_meters_dict, unique_values_dict = accumulate_pixel_preds(
        class_wise=False,
        pixel_meters_dict=pixel_meters_dict,
        key_1="prob_bin",
        edge_only=edge_only,
        neighborhood_width=neighborhood_width
    )
    unique_bins = unique_values_dict["prob_bin"]
    # Keep track of different things for each bin.
    cal_info = {
        "bin_confs": torch.zeros(num_bins, dtype=torch.float64),
        "bin_amounts": torch.zeros(num_bins, dtype=torch.float64),
        "bin_freqs": torch.zeros(num_bins, dtype=torch.float64),
        "bin_cal_errors": torch.zeros(num_bins, dtype=torch.float64),
    }
    # Get the regions of the prediction corresponding to each bin of confidence.
    for bin_idx in range(num_bins):
        if bin_idx in unique_bins:
            # Get the meter for the bin.
            bin_meter = accumulated_meters_dict[bin_idx]
            # Choose what key to use.
            bin_conf = bin_meter["confidence"].mean
            bin_freq = bin_meter["accuracy"].mean
            num_samples = bin_meter["accuracy"].n
            # Calculate the average calibration error for the regions in the bin.
            cal_info["bin_confs"][bin_idx] = bin_conf
            cal_info["bin_freqs"][bin_idx] = bin_freq
            cal_info["bin_amounts"][bin_idx] = num_samples
            # Choose whether or not to square for the cal error.
            if square_diff:
                cal_info["bin_cal_errors"][bin_idx] = np.power(bin_conf - bin_freq, 2)
            else:
                cal_info["bin_cal_errors"][bin_idx] = np.abs(bin_conf - bin_freq)
    if device is not None:
        for key, value in cal_info.items():
            cal_info[key] = value.to(device)
    # Return the calibration information.
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def class_wise_bin_stats(
    pixel_meters_dict: dict,
    num_bins: int,
    num_classes: int,
    class_wise: bool,
    square_diff: bool = False,
    edge_only: bool = False,
    neighborhood_width: Optional[int] = None,
    device: Optional[Literal["cpu", "cuda"]] = None,
) -> dict:
    stat_type = "true" if class_wise else "pred"
    accumulated_meters_dict, unique_values_dict = accumulate_pixel_preds(
        class_wise=class_wise,
        pixel_meters_dict=pixel_meters_dict,
        key_1=f"{stat_type}_label",
        key_2="prob_bin",
        edge_only=edge_only,
        neighborhood_width=neighborhood_width
    )
    unique_labels = unique_values_dict[f"{stat_type}_label"]
    unique_bins = unique_values_dict["prob_bin"]
    # Keep track of different things for each bin.
    cal_info = {
        "bin_confs": torch.zeros(num_classes, num_bins, dtype=torch.float64),
        "bin_amounts": torch.zeros(num_classes, num_bins, dtype=torch.float64),
        "bin_freqs": torch.zeros(num_classes, num_bins, dtype=torch.float64),
        "bin_cal_errors": torch.zeros(num_classes, num_bins, dtype=torch.float64),
    }
    for lab_idx in range(num_classes):
        for bin_idx in range(num_bins):
            if lab_idx in unique_labels and bin_idx in unique_bins:
                # Get the meter for the bin.
                bin_meter = accumulated_meters_dict[lab_idx][bin_idx]
                # Choose what key to use.
                bin_conf = bin_meter["confidence"].mean
                bin_freq = bin_meter["accuracy"].mean
                num_samples = bin_meter["accuracy"].n
                # Calculate the average calibration error for the regions in the bin.
                cal_info["bin_confs"][lab_idx, bin_idx] = bin_conf
                cal_info["bin_freqs"][lab_idx, bin_idx] = bin_freq
                cal_info["bin_amounts"][lab_idx, bin_idx] = num_samples
                # Choose whether or not to square for the cal error.
                if square_diff:
                    cal_info["bin_cal_errors"][lab_idx, bin_idx] = np.power(bin_conf - bin_freq, 2)
                else:
                    cal_info["bin_cal_errors"][lab_idx, bin_idx] = np.abs(bin_conf - bin_freq)
    if device is not None:
        for key, value in cal_info.items():
            cal_info[key] = value.to(device)
    # Return the calibration information.
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def neighbor_wise_bin_stats(
    pixel_meters_dict: dict,
    class_wise: bool,
    num_bins: int,
    neighborhood_width: int,
    square_diff: bool = False,
    edge_only: bool = False,
    device: Optional[Literal["cpu", "cuda"]] = None,
) -> dict:
    stat_type = "true" if class_wise else "pred"
    accumulated_meters_dict, unique_values_dict = accumulate_pixel_preds(
        class_wise=class_wise,
        pixel_meters_dict=pixel_meters_dict,
        key_1=f"{stat_type}_num_neighb",
        key_2="prob_bin",
        edge_only=edge_only,
        neighborhood_width=neighborhood_width,
    )
    unique_neighbor_classes = unique_values_dict[f"{stat_type}_num_neighb"]
    unique_prob_bins = unique_values_dict["prob_bin"]
    # Keep track of different things for each bin.
    num_neighb_classes = neighborhood_width**2
    cal_info = {
        "bin_confs": torch.zeros(num_neighb_classes, num_bins, dtype=torch.float64),
        "bin_amounts": torch.zeros(num_neighb_classes, num_bins, dtype=torch.float64),
        "bin_freqs": torch.zeros(num_neighb_classes, num_bins, dtype=torch.float64),
        "bin_cal_errors": torch.zeros(num_neighb_classes, num_bins, dtype=torch.float64),
    }
    for nn_idx in range(num_neighb_classes):
        for bin_idx in range(num_bins):
            if nn_idx in unique_neighbor_classes and bin_idx in unique_prob_bins:
                # Get the meter for the bin.
                bin_meter = accumulated_meters_dict[nn_idx][bin_idx]
                # Choose what key to use.
                bin_conf = bin_meter["confidence"].mean
                bin_freq = bin_meter["accuracy"].mean
                num_samples = bin_meter["accuracy"].n
                # Calculate the average calibration error for the regions in the bin.
                cal_info["bin_confs"][nn_idx, bin_idx] = bin_conf
                cal_info["bin_freqs"][nn_idx, bin_idx] = bin_freq
                cal_info["bin_amounts"][nn_idx, bin_idx] = num_samples
                # Choose whether or not to square for the cal error.
                if square_diff:
                    cal_info["bin_cal_errors"][nn_idx, bin_idx] = np.power(bin_conf - bin_freq, 2)
                else:
                    cal_info["bin_cal_errors"][nn_idx, bin_idx] = np.abs(bin_conf - bin_freq)
    if device is not None:
        for key, value in cal_info.items():
            cal_info[key] = value.to(device)
    # Return the calibration information.
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def joint_class_neighbor_bin_stats(
    pixel_meters_dict: dict,
    class_wise: bool,
    num_bins: int,
    num_classes: int,
    neighborhood_width: int,
    square_diff: bool = False,
    edge_only: bool = False,
    device: Optional[Literal["cpu", "cuda"]] = None,
) -> dict:
    stat_type = "true" if class_wise else "pred"
    accumulated_meters_dict, unique_values_dict = accumulate_pixel_preds(
        class_wise=class_wise,
        pixel_meters_dict=pixel_meters_dict,
        key_1=f"{stat_type}_label",
        key_2=f"pred_num_neighb",
        key_3="prob_bin",
        edge_only=edge_only,
        neighborhood_width=neighborhood_width
    )
    unique_labels = unique_values_dict[f"{stat_type}_label"]
    unique_num_neighb = unique_values_dict[f"pred_num_neighb"]
    unique_bins = unique_values_dict["prob_bin"]
    # Keep track of different things for each bin.
    num_nn_classes = neighborhood_width**2
    cal_info = {
        "bin_confs": torch.zeros(num_classes, num_nn_classes, num_bins, dtype=torch.float64),
        "bin_amounts": torch.zeros(num_classes, num_nn_classes, num_bins, dtype=torch.float64),
        "bin_freqs": torch.zeros(num_classes, num_nn_classes, num_bins, dtype=torch.float64),
        "bin_cal_errors": torch.zeros(num_classes, num_nn_classes, num_bins, dtype=torch.float64),
    }
    for label in range(num_classes):
        for nn_class in range(num_nn_classes):
            for prob_bin in range(num_bins):
                if label in unique_labels and nn_class in unique_num_neighb and prob_bin in unique_bins:
                    # Get the meter for the bin.
                    bin_meter = accumulated_meters_dict[label][nn_class][prob_bin]
                    # Gather the statistics.
                    bin_conf = bin_meter["confidence"].mean
                    bin_freq = bin_meter["accuracy"].mean
                    num_samples = bin_meter["accuracy"].n
                    # Calculate the average calibration error for the regions in the bin.
                    lab_idx = unique_labels.index(label)
                    nn_idx = unique_num_neighb.index(nn_class)
                    bin_idx = unique_bins.index(prob_bin)
                    # Stick the values in the bins.
                    cal_info["bin_confs"][lab_idx, nn_idx, bin_idx] = bin_conf
                    cal_info["bin_freqs"][lab_idx, nn_idx, bin_idx] = bin_freq
                    cal_info["bin_amounts"][lab_idx, nn_idx, bin_idx] = num_samples
                    # Choose whether or not to square for the cal error.
                    if square_diff:
                        cal_info["bin_cal_errors"][lab_idx, nn_idx, bin_idx] = np.power(bin_conf - bin_freq, 2)
                    else:
                        cal_info["bin_cal_errors"][lab_idx, nn_idx, bin_idx] = np.abs(bin_conf - bin_freq)
    if device is not None:
        for key, value in cal_info.items():
            cal_info[key] = value.to(device)
    # Return the calibration information.
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def accumulate_pixel_preds(
    class_wise: bool,
    pixel_meters_dict: dict,
    key_1: str,
    key_2: Optional[str] = None,
    key_3: Optional[str] = None,
    edge_only: bool = False,
    neighborhood_width: Optional[int] = None,
) -> dict:
    assert (not edge_only) or (edge_only and neighborhood_width is not None),\
        "If edge_only is True, neighborhood_width must be defined."
    # Accumulate the dictionaries corresponding to a single bin.
    accumulated_meters_dict = {}
    unique_key_1 = []
    unique_key_2 = []
    unique_key_3 = []
    # Iterate through the meters.
    for pix_dict_key, value in pixel_meters_dict.items():
        if class_wise:
            true_label, true_num_neighb, pred_num_neighb, prob_bin, measure = pix_dict_key
        else:
            true_label, pred_label, true_num_neighb, pred_num_neighb, prob_bin, measure = pix_dict_key
        # Get the total number of pixel classes for this neighborhood size.
        if neighborhood_width is not None:
            total_nearby_pixels = (neighborhood_width**2 - 1)
        # We track pixels if they are not edge pixels or if they are edge pixels and the edge_only flag is False.
        if (not edge_only) or (true_num_neighb < total_nearby_pixels):
            item = {
                "true_label": true_label,
                "true_num_neighb": true_num_neighb,
                "pred_num_neighb": pred_num_neighb,
                "prob_bin": prob_bin,
                "measure": measure,
            }
            if not class_wise:
                item["pred_label"] = pred_label
            # Keep track of unique values.
            if item[key_1] not in unique_key_1:
                unique_key_1.append(item[key_1])
            if key_2 is not None:
                if item[key_2] not in unique_key_2:
                    unique_key_2.append(item[key_2])
            if key_3 is not None:
                if item[key_3] not in unique_key_3:
                    unique_key_3.append(item[key_3])
            # El Monstro
            if item[key_1] not in accumulated_meters_dict:
                accumulated_meters_dict[item[key_1]] = {}
            level1_dict = accumulated_meters_dict[item[key_1]]
            if key_2 is None:
                if measure not in level1_dict:
                    level1_dict[measure] = value
                else:
                    level1_dict[measure] += value
            else:
                if item[key_2] not in level1_dict:
                    level1_dict[item[key_2]] = {}
                level2_dict = level1_dict[item[key_2]]
                if key_3 is None:                
                    if measure not in level2_dict:
                        level2_dict[measure] = value
                    else:
                        level2_dict[measure] += value
                else:
                    if item[key_3] not in level2_dict:
                        level2_dict[item[key_3]] = {}
                    level3_dict = level2_dict[item[key_3]]
                    if measure not in level3_dict:
                        level3_dict[measure] = value
                    else:
                        level3_dict[measure] += value
    # Wrap the unique values into a dictionary.
    unique_values_dict = {
        key_1: sorted(unique_key_1)
    }
    if key_2 is not None:
        unique_values_dict[key_2] = sorted(unique_key_2)
    if key_3 is not None:
        unique_values_dict[key_3] = sorted(unique_key_3)
    # Return the accumulated values and the unique keys.
    return accumulated_meters_dict, unique_values_dict
