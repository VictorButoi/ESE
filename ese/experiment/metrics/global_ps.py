# torch imports
import torch 
# misc imports
from pydantic import validate_arguments
from collections import defaultdict


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def accumulate_pixel_preds(
    pixel_preds: dict,
) -> dict:
    # Accumulate the dictionaries corresponding to a single bin.
    data_dict = defaultdict(lambda: defaultdict(list))
    for (_, _, bin_num, measure), value in pixel_preds.items():
        data_dict[bin_num][measure].append(value)
    return data_dict


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def global_bin_stats(
    data_dict: dict,
    square_diff: bool,
    weighted: bool = False,
    ) -> dict:
    # Get the num bins.
    num_bins = len(data_dict.keys()) 
    # Keep track of different things for each bin.
    cal_info = {
        "bin_confs": torch.zeros(num_bins),
        "bin_amounts": torch.zeros(num_bins),
        "bin_accs": torch.zeros(num_bins),
        "bin_cal_errors": torch.zeros(num_bins),
    }
    # Get the regions of the prediction corresponding to each bin of confidence.
    for bin_idx in data_dict.keys():
        # Choose what key to use.
        conf_key = "confidence" if not weighted else "weighted confidence"
        acc_key = "accuracy" if not weighted else "weighted accuracy"
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