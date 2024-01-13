# torch imports
import torch 
# misc imports
import numpy as np
from typing import Optional
from pydantic import validate_arguments


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def accumulate_pixel_preds(
    pixel_meters_dict: dict,
    key_1: str,
    key_2: Optional[str] = None,
    key_3: Optional[str] = None,
    edge_only: bool = False,
    neighborhood_width: Optional[int] = None,
    ignore_index: Optional[int] = None
) -> dict:
    assert (not edge_only) or (edge_only and neighborhood_width is not None),\
        "If edge_only is True, neighborhood_width must be defined."
    if neighborhood_width is not None:
        total_nearby_pixels = (neighborhood_width**2 - 1)
    # Accumulate the dictionaries corresponding to a single bin.
    accumulated_meters_dict = {}
    unique_key_1 = []
    unique_key_2 = []
    unique_key_3 = []
    # Iterate through the meters.
    for (true_label, pred_label, true_num_neighb, pred_num_neighb, prob_bin, measure), value in pixel_meters_dict.items():
        if ignore_index is None or true_label != ignore_index:
            if (not edge_only) or (true_num_neighb < total_nearby_pixels):
                item = {
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "pred_num_neighb": pred_num_neighb,
                    "true_num_neighb": true_num_neighb,
                    "prob_bin": prob_bin,
                    "measure": measure,
                }
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


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def global_bin_stats(
    pixel_meters_dict: dict,
    square_diff: bool = False,
    weighted: bool = False,
    edge_only: bool = False,
    neighborhood_width: Optional[int] = None,
    ignore_index: Optional[int] = None
    ) -> dict:
    accumulated_meters_dict, unique_values_dict = accumulate_pixel_preds(
        pixel_meters_dict,
        key_1="prob_bin",
        edge_only=edge_only,
        neighborhood_width=neighborhood_width,
        ignore_index=ignore_index
        )
    unique_bins = unique_values_dict["prob_bin"]
    # Get the num bins.
    num_bins = len(unique_bins) 
    # Keep track of different things for each bin.
    cal_info = {
        "bin_confs": torch.zeros(num_bins, dtype=torch.float64),
        "bin_amounts": torch.zeros(num_bins, dtype=torch.float64),
        "bin_accs": torch.zeros(num_bins, dtype=torch.float64),
        "bin_cal_errors": torch.zeros(num_bins, dtype=torch.float64),
    }
    # Either use the weighted or unweighted confidence and accuracy.
    conf_key = "confidence" if not weighted else "weighted confidence"
    acc_key = "accuracy" if not weighted else "weighted accuracy"
    # Get the regions of the prediction corresponding to each bin of confidence.
    for prob_bin in accumulated_meters_dict.keys():
        # Choose what key to use.
        bin_conf = accumulated_meters_dict[prob_bin][conf_key].mean
        bin_acc = accumulated_meters_dict[prob_bin][acc_key].mean
        num_samples = accumulated_meters_dict[prob_bin][acc_key].n
        # Calculate the average calibration error for the regions in the bin.
        bin_idx = unique_bins.index(prob_bin)
        cal_info["bin_confs"][bin_idx] = bin_conf
        cal_info["bin_accs"][bin_idx] = bin_acc
        cal_info["bin_amounts"][bin_idx] = num_samples
        # Choose whether or not to square for the cal error.
        if square_diff:
            cal_info["bin_cal_errors"][bin_idx] = np.power(bin_conf - bin_acc, 2)
        else:
            cal_info["bin_cal_errors"][bin_idx] = np.abs(bin_conf - bin_acc)
    # Return the calibration information.
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def global_label_bin_stats(
    pixel_meters_dict: dict,
    top_label: bool,
    square_diff: bool = False,
    weighted: bool = False,
    edge_only: bool = False,
    neighborhood_width: Optional[int] = None,
    ignore_index: Optional[int] = None
    ) -> dict:
    label_key = "pred_label" if top_label else "true_label"
    accumulated_meters_dict, unique_values_dict = accumulate_pixel_preds(
        pixel_meters_dict,
        key_1=label_key,
        key_2="prob_bin",
        edge_only=edge_only,
        neighborhood_width=neighborhood_width,
        ignore_index=ignore_index
        )
    unique_labels = unique_values_dict[label_key]
    unique_bins = unique_values_dict["prob_bin"]
    # Get the num bins.
    num_labels = len(unique_labels)
    num_bins = len(unique_bins) 
    # Keep track of different things for each bin.
    cal_info = {
        "bin_confs": torch.zeros(num_labels, num_bins, dtype=torch.float64),
        "bin_amounts": torch.zeros(num_labels, num_bins, dtype=torch.float64),
        "bin_accs": torch.zeros(num_labels, num_bins, dtype=torch.float64),
        "bin_cal_errors": torch.zeros(num_labels, num_bins, dtype=torch.float64),
    }
    # Either use the weighted or unweighted confidence and accuracy.
    conf_key = "confidence" if not weighted else "weighted confidence"
    acc_key = "accuracy" if not weighted else "weighted accuracy"
    for label in accumulated_meters_dict.keys():
        for prob_bin in accumulated_meters_dict[label].keys():
            # Choose what key to use.
            bin_conf = accumulated_meters_dict[label][prob_bin][conf_key].mean
            bin_acc = accumulated_meters_dict[label][prob_bin][acc_key].mean
            num_samples = accumulated_meters_dict[label][prob_bin][acc_key].n
            # Calculate the average calibration error for the regions in the bin.
            lab_idx = unique_labels.index(label)
            bin_idx = unique_bins.index(prob_bin)
            cal_info["bin_confs"][lab_idx, bin_idx] = bin_conf
            cal_info["bin_accs"][lab_idx, bin_idx] = bin_acc
            cal_info["bin_amounts"][lab_idx, bin_idx] = num_samples
            # Choose whether or not to square for the cal error.
            if square_diff:
                cal_info["bin_cal_errors"][lab_idx, bin_idx] = np.power(bin_conf - bin_acc, 2)
            else:
                cal_info["bin_cal_errors"][lab_idx, bin_idx] = np.abs(bin_conf - bin_acc)
    # Return the calibration information.
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def global_neighbors_bin_stats(
    pixel_meters_dict: dict,
    square_diff: bool = False,
    weighted: bool = False,
    neighborhood_width: Optional[int] = None,
    ignore_index: Optional[int] = None
    ) -> dict:
    accumulated_meters_dict, unique_values_dict = accumulate_pixel_preds(
        pixel_meters_dict,
        key_1="pred_num_neighb",
        key_2="prob_bin",
        neighborhood_width=neighborhood_width,
        ignore_index=ignore_index
        )
    unique_pred_neighbor_classes = unique_values_dict["pred_num_neighb"]
    unique_prob_bins = unique_values_dict["prob_bin"]
    # Get the num bins.
    num_pred_neighb_classes = len(unique_pred_neighbor_classes)
    num_bins = len(unique_prob_bins) 
    # Keep track of different things for each bin.
    cal_info = {
        "bin_confs": torch.zeros(num_pred_neighb_classes, num_bins, dtype=torch.float64),
        "bin_amounts": torch.zeros(num_pred_neighb_classes, num_bins, dtype=torch.float64),
        "bin_accs": torch.zeros(num_pred_neighb_classes, num_bins, dtype=torch.float64),
        "bin_cal_errors": torch.zeros(num_pred_neighb_classes, num_bins, dtype=torch.float64),
    }
    # Either use the weighted or unweighted confidence and accuracy.
    conf_key = "confidence" if not weighted else "weighted confidence"
    acc_key = "accuracy" if not weighted else "weighted accuracy"
    for neighbor_class in accumulated_meters_dict.keys():
        for prob_bin in accumulated_meters_dict[neighbor_class].keys():
            # Choose what key to use.
            bin_conf = accumulated_meters_dict[neighbor_class][prob_bin][conf_key].mean
            bin_acc = accumulated_meters_dict[neighbor_class][prob_bin][acc_key].mean
            num_samples = accumulated_meters_dict[neighbor_class][prob_bin][acc_key].n
            # Calculate the average calibration error for the regions in the bin.
            nn_idx = unique_pred_neighbor_classes.index(neighbor_class)
            bin_idx = unique_prob_bins.index(prob_bin)
            cal_info["bin_confs"][nn_idx, bin_idx] = bin_conf
            cal_info["bin_accs"][nn_idx, bin_idx] = bin_acc
            cal_info["bin_amounts"][nn_idx, bin_idx] = num_samples
            # Choose whether or not to square for the cal error.
            if square_diff:
                cal_info["bin_cal_errors"][nn_idx, bin_idx] = np.power(bin_conf - bin_acc, 2)
            else:
                cal_info["bin_cal_errors"][nn_idx, bin_idx] = np.abs(bin_conf - bin_acc)
    # Return the calibration information.
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def global_label_neighbors_bin_stats(
    pixel_meters_dict: dict,
    top_label: bool,
    square_diff: bool = False,
    weighted: bool = False,
    neighborhood_width: Optional[int] = None,
    ignore_index: Optional[int] = None
    ) -> dict:
    label_key = "pred_label" if top_label else "true_label"
    accumulated_meters_dict, unique_values_dict = accumulate_pixel_preds(
        pixel_meters_dict,
        key_1=label_key,
        key_2="pred_num_neighb",
        key_3="prob_bin",
        neighborhood_width=neighborhood_width,
        ignore_index=ignore_index
        )
    unique_labels = unique_values_dict[label_key] 
    unique_pred_neighbor_classes = unique_values_dict["pred_num_neighb"]
    unique_prob_bins = unique_values_dict["prob_bin"]
    # Get the num bins.
    num_labels = len(unique_labels) 
    num_pred_neighb_classes = len(unique_pred_neighbor_classes)
    num_bins = len(unique_prob_bins)
    # Keep track of different things for each bin.
    cal_info = {
        "bin_confs": torch.zeros(num_labels, num_pred_neighb_classes, num_bins, dtype=torch.float64),
        "bin_amounts": torch.zeros(num_labels, num_pred_neighb_classes, num_bins, dtype=torch.float64),
        "bin_accs": torch.zeros(num_labels, num_pred_neighb_classes, num_bins, dtype=torch.float64),
        "bin_cal_errors": torch.zeros(num_labels, num_pred_neighb_classes, num_bins, dtype=torch.float64),
    }
    # Either use the weighted or unweighted confidence and accuracy.
    conf_key = "confidence" if not weighted else "weighted confidence"
    acc_key = "accuracy" if not weighted else "weighted accuracy"
    for label in accumulated_meters_dict.keys():
        for neighbor_class in accumulated_meters_dict[label].keys():
            for prob_bin in accumulated_meters_dict[label][neighbor_class].keys():
                # Choose what key to use.
                bin_conf = accumulated_meters_dict[label][neighbor_class][prob_bin][conf_key].mean
                bin_acc = accumulated_meters_dict[label][neighbor_class][prob_bin][acc_key].mean
                num_samples = accumulated_meters_dict[label][neighbor_class][prob_bin][acc_key].n
                # Calculate the average calibration error for the regions in the bin.
                lab_idx = unique_labels.index(label)
                nn_idx = unique_pred_neighbor_classes.index(neighbor_class)
                bin_idx = unique_prob_bins.index(prob_bin)
                cal_info["bin_confs"][lab_idx, nn_idx, bin_idx] = bin_conf
                cal_info["bin_accs"][lab_idx, nn_idx, bin_idx] = bin_acc
                cal_info["bin_amounts"][lab_idx, nn_idx, bin_idx] = num_samples
                # Choose whether or not to square for the cal error.
                if square_diff:
                    cal_info["bin_cal_errors"][lab_idx, nn_idx, bin_idx] = np.power(bin_conf - bin_acc, 2)
                else:
                    cal_info["bin_cal_errors"][lab_idx, nn_idx, bin_idx] = np.abs(bin_conf - bin_acc)
    # Return the calibration information.
    return cal_info
