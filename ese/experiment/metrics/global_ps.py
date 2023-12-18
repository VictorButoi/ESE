# torch imports
import torch 
# misc imports
from typing import Optional
from pydantic import validate_arguments
from collections import defaultdict


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def accumulate_pixel_preds(
    pixel_preds: dict,
    ignore_index: Optional[int] = None
) -> dict:
    # Accumulate the dictionaries corresponding to a single bin.
    data_dict = defaultdict(lambda: defaultdict(list))
    for (true_label, pred_label, num_matching_neighbors, prob_bin, measure), value in pixel_preds.items():
        if ignore_index is None or true_label != ignore_index:
            data_dict[prob_bin][measure].append(value)
    return data_dict


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def global_bin_stats(
    pixel_preds_dict: dict,
    square_diff: bool,
    weighted: bool = False,
    ignore_index: Optional[int] = None
    ) -> dict:
    data_dict = accumulate_pixel_preds(
        pixel_preds_dict,
        ignore_index=ignore_index
        )
    # Get the num bins.
    num_bins = len(data_dict.keys()) 
    # Keep track of different things for each bin.
    cal_info = {
        "bin_confs": torch.zeros(num_bins),
        "bin_amounts": torch.zeros(num_bins),
        "bin_accs": torch.zeros(num_bins),
        "bin_cal_errors": torch.zeros(num_bins),
    }
    # Either use the weighted or unweighted confidence and accuracy.
    conf_key = "confidence" if not weighted else "weighted confidence"
    acc_key = "accuracy" if not weighted else "weighted accuracy"
    # Get the regions of the prediction corresponding to each bin of confidence.
    for bin_idx in data_dict.keys():
        # Choose what key to use.
        bin_conf = data_dict[conf_key].mean
        bin_acc = data_dict[acc_key].mean
        num_samples = data_dict[acc_key].n
        # Calculate the average calibration error for the regions in the bin.
        cal_info["bin_confs"][bin_idx] = bin_conf
        cal_info["bin_accs"][bin_idx] = bin_acc
        cal_info["bin_amounts"][bin_idx] = num_samples
        # Choose whether or not to square for the cal error.
        if square_diff:
            cal_info["bin_cal_errors"][bin_idx] = (bin_conf - bin_acc).pow(2)
        else:
            cal_info["bin_cal_errors"][bin_idx] = (bin_conf - bin_acc).abs()
    # Return the calibration information.
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def global_label_bin_stats(
    data_dict: dict,
    square_diff: bool,
    weighted: bool = False,
    ) -> dict:
    # num_labels = None
    # num_bins = None
    # # Setup the cal info tracker.
    # cal_info = {
    #     "bin_confs": torch.zeros((num_labels, num_bins)),
    #     "bin_amounts": torch.zeros((num_labels, num_bins)),
    #     "bin_accs": torch.zeros((num_labels, num_bins)),
    #     "bin_cal_errors": torch.zeros((num_labels, num_bins))
    # }
    # # Either use the weighted or unweighted confidence and accuracy.
    # conf_key = "confidence" if not weighted else "weighted confidence"
    # acc_key = "accuracy" if not weighted else "weighted accuracy"
    # # Go through each of the labels and bins.
    # for lab_idx in data_dict.keys():
    #     for bin_idx in data_dict[lab_idx].keys():
    #         # Choose what key to use.
    #         bin_conf = data_dict[lab_idx][bin_idx][conf_key].mean
    #         bin_acc = data_dict[lab_idx][bin_idx][acc_key].mean
    #         num_samples = data_dict[lab_idx][bin_idx][acc_key].n
    #         # Calculate the average calibration error for the regions in the bin.
    #         cal_info["bin_confs"][lab_idx, bin_idx] = bin_conf
    #         cal_info["bin_accs"][lab_idx, bin_idx] = bin_acc
    #         cal_info["bin_amounts"][lab_idx, bin_idx] = num_samples
    #         # Choose whether or not to square for the cal error.
    #         if square_diff:
    #             cal_info["bin_cal_errors"][lab_idx, bin_idx] = (bin_conf - bin_acc).pow(2)
    #         else:
    #             cal_info["bin_cal_errors"][lab_idx, bin_idx] = (bin_conf - bin_acc).abs()
    # # Return the label-wise calibration information.
    # return cal_info
    pass


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def global_neighbors_bin_stats(
    data_dict: dict,
    square_diff: bool,
    weighted: bool = False,
    ) -> dict:
    # num_labels = None
    # num_bins = None
    # # Setup the cal info tracker.
    # cal_info = {
    #     "bin_confs": torch.zeros((num_labels, num_bins)),
    #     "bin_amounts": torch.zeros((num_labels, num_bins)),
    #     "bin_accs": torch.zeros((num_labels, num_bins)),
    #     "bin_cal_errors": torch.zeros((num_labels, num_bins))
    # }
    # # Either use the weighted or unweighted confidence and accuracy.
    # conf_key = "confidence" if not weighted else "weighted confidence"
    # acc_key = "accuracy" if not weighted else "weighted accuracy"
    # # Go through each of the labels and bins.
    # for lab_idx in data_dict.keys():
    #     for bin_idx in data_dict[lab_idx].keys():
    #         # Choose what key to use.
    #         bin_conf = data_dict[lab_idx][bin_idx][conf_key].mean
    #         bin_acc = data_dict[lab_idx][bin_idx][acc_key].mean
    #         num_samples = data_dict[lab_idx][bin_idx][acc_key].n
    #         # Calculate the average calibration error for the regions in the bin.
    #         cal_info["bin_confs"][lab_idx, bin_idx] = bin_conf
    #         cal_info["bin_accs"][lab_idx, bin_idx] = bin_acc
    #         cal_info["bin_amounts"][lab_idx, bin_idx] = num_samples
    #         # Choose whether or not to square for the cal error.
    #         if square_diff:
    #             cal_info["bin_cal_errors"][lab_idx, bin_idx] = (bin_conf - bin_acc).pow(2)
    #         else:
    #             cal_info["bin_cal_errors"][lab_idx, bin_idx] = (bin_conf - bin_acc).abs()
    # # Return the label-wise calibration information.
    # return cal_info
    pass


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def global_label_neighbors_bin_stats(
    data_dict: dict,
    square_diff: bool,
    weighted: bool = False,
    ) -> dict:
    # # Init some things.
    # num_labels = None
    # # Get the num of neighbor classes.
    # unique_pred_matching_neighbors = None
    # num_neighbors = len(unique_pred_matching_neighbors)
    # # Define num bins
    # num_bins = None
    # # Init the cal info tracker.
    # cal_info = {
    #     "bin_cal_errors": torch.zeros((num_labels, num_neighbors, num_bins)),
    #     "bin_accs": torch.zeros((num_labels, num_neighbors, num_bins)),
    #     "bin_confs": torch.zeros((num_labels, num_neighbors, num_bins)),
    #     "bin_amounts": torch.zeros((num_labels, num_neighbors, num_bins))
    # }
    # for lab_idx, lab in enumerate(lab_info["unique_labels"]):
    #     for nn_idx, p_nn in enumerate(unique_pred_matching_neighbors):
    #         for bin_idx, conf_bin in enumerate(obj_dict["conf_bins"]):
    #             # Calculate the average score for the regions in the bin.
    #             bi = calc_bin_info(
    #                 conf_map=obj_dict["y_max_prob_map"],
    #                 bin_conf_region=bin_conf_region,
    #                 square_diff=square_diff,
    #                 pixelwise_accuracy=obj_dict["pixelwise_accuracy"],
    #                 pix_weights=obj_dict["pix_weights"]
    #             )
    #             # Calculate the average calibration error for the regions in the bin.
    #             cal_info["bin_confs"][lab_idx, nn_idx, bin_idx] = bi["avg_conf"] 
    #             cal_info["bin_accs"][lab_idx, nn_idx, bin_idx] = bi["avg_conf"] 
    #             cal_info["bin_amounts"][lab_idx, nn_idx, bin_idx] = bi["num_samples"] 
    #             cal_info["bin_cal_errors"][lab_idx, nn_idx, bin_idx] = bi["cal_error"] 
    # # Return the label-wise and neighborhood conditioned calibration information.
    # return cal_info
    pass