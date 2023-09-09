# misc imports
import numpy as np
import torch

# ionpy imports
from ionpy.metrics.segmentation import pixel_accuracy
from ionpy.util.islands import get_connected_components


def ECE_map(subj):
    conf_scores = subj['soft_pred']
    
    # Calculate the per-pixel accuracy and where the foreground regions are.
    foreground_accuracy = (subj['label'] == subj['hard_pred']).float()
    foreground_predicted_regions = (subj['hard_pred'] == 1).bool()

    # Set the regions of the image corresponding to groundtruth label.
    calibration_image = np.zeros_like(subj['label'])
    calibration_image[foreground_predicted_regions] = (conf_scores - foreground_accuracy)[foreground_predicted_regions]

    return calibration_image


def ESE_map(subj, conf_bins):

    pred = subj['soft_pred']
    calibration_image = np.zeros_like(pred)

    # Make sure bins are aligned.
    bin_width = conf_bins[1] - conf_bins[0]
    for c_bin in conf_bins:
        
        # Get the region of this confidence interval
        bin_conf_region = (pred >= c_bin) & (pred < (c_bin + bin_width))

        # Get the region of the label corresponding to this region.
        label_region = subj['label'][bin_conf_region][None, None, ...]

        # Calculate the accuracy and mean confidence for the island.
        avg_accuracy = pixel_accuracy(torch.ones_like(label_region), label_region)
        avg_confidence = pred[bin_conf_region].mean()

        # Calculate the calibration error for the pixels in the bin.
        calibration_image[bin_conf_region]  = (avg_confidence - avg_accuracy)

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
            calibration_image[island] = (region_conf_scores - region_acc_scores)

    return calibration_image
