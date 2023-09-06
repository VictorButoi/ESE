import numpy as np
import torch
from ese.experiment.metrics import ESE, ReCE
from ionpy.metrics.segmentation import pixel_accuracy
from ionpy.util.islands import get_connected_components


def ECE_map(subj):
    calibration_image = np.zeros_like(subj['label'])
    foreground_accuracy = (subj['label'] == subj['hard_pred']).float()
    fore_regions = (subj['label'] == 1).bool()
    # Set the regions of the image corresponding to groundtruth label.
    calibration_image[fore_regions] = (foreground_accuracy - subj['soft_pred']).abs()[fore_regions]

    return calibration_image


def ESE_map(subj, conf_bins):

    ese_bin_scores, _, _ = ESE(
        conf_bins=conf_bins,
        pred=subj["soft_pred"],
        label=subj["label"],
    ) 

    pred = subj['soft_pred']
    calibration_image = np.zeros_like(pred)

    # Make sure bins are aligned.
    bin_width = conf_bins[1] - conf_bins[0]
    for b_idx, c_bin in enumerate(conf_bins):
        bin_conf_region = (pred >= c_bin) & (pred < (c_bin + bin_width))
        calibration_image[bin_conf_region] = ese_bin_scores[b_idx] 

    return calibration_image


def ReCE_map(subj, conf_bins):

    pred = subj['soft_pred']
    calibration_image = np.zeros_like(pred)

    # Make sure bins are aligned.
    bin_width = conf_bins[1] - conf_bins[0]
    for c_bin in conf_bins:
        # Get the binary region of this confidence interval
        bin_conf_region = (pred >= c_bin) & (pred < (c_bin + bin_width))
        # Break it up into islands
        conf_islands = get_connected_components(bin_conf_region)
        # Iterate through each island, and get the measure for each island.
        for island in conf_islands:
            # Get the label corresponding to the island and simulate ground truth and make the right shape.
            label_region = subj["label"][island][None, None, ...]
            # Calculate the accuracy and mean confidence for the island.
            region_acc_scores = pixel_accuracy(torch.ones_like(label_region), label_region)
            region_conf_scores = pred[island].mean()
            # Insert into this region of the image
            calibration_image[island] = (region_acc_scores - region_conf_scores).abs()

    return calibration_image