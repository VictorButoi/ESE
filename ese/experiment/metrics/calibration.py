import numpy as np
import matplotlib.pyplot as plt
from ionpy.metrics import dice_score, pixel_accuracy
import torch


def ECE(
    bins,
    pred=None,
    label=None,
    confidences=None,
    accuracies=None,
    reduce=None
):
    """
    Calculates the Expected Calibration Error (ECE) for a model.
    Args:
        accuracies: numpy array of calibration accuracies for each bin
        confidences: numpy array of confidences outputted by the model
        num_bins (int): number of confidence interval bins
    Returns:
        float: Expected Calibration Error
    """
    assert not (pred is None and confidences is None), "Must provide either pred or confidences."
    assert not (label is None and accuracies is None), "Must provide either label or accuracies."

    # Get the confidence bin width
    bin_width = bins[1] - bins[0]

    # Calculate |confidence - accuracy| in each bin
    scores = np.zeros(len(bins))
    bin_amounts = np.zeros(len(bins))

    # Get the regions of the prediction corresponding to each bin of confidence.
    if pred is not None:
        confidences = pred.flatten()

    if label is not None:
        hard_pred = (pred >= 0.5)
        acc_image = (hard_pred == label).float()
        accuracies = acc_image.flatten()

    for bin_idx, bin in enumerate(bins):
        # Calculated |confidence - accuracy| in each bin
        in_bin = np.logical_and(confidences >= bin, confidences < (bin + bin_width)).bool()
        num_pix_in_bin = torch.sum(in_bin)

        if num_pix_in_bin > 0:
            all_bin_accs = accuracies[in_bin]
            all_bin_confs = confidences[in_bin]

            accuracy_in_bin = torch.mean(all_bin_accs) if torch.sum(all_bin_accs) > 0 else 0
            avg_confidence_in_bin = torch.mean(all_bin_confs)

            scores[bin_idx] = (avg_confidence_in_bin - accuracy_in_bin).abs()
            bin_amounts[bin_idx] = num_pix_in_bin 

    # Calculate ece as the weighted average, if there are any pixels in the bin.
    if reduce == "mean":
        if np.sum(bin_amounts) == 0:
            return 0
        else:
            props_per_bin = bin_amounts / np.sum(bin_amounts)
            return np.average(scores, weights=props_per_bin)
    else:
        return scores, bin_amounts


# Measure per bin.
def ESE(
    bins,
    pred=None, 
    label=None,
    confidences=None,
    accuracies=None,
    bin_weighting='proportional',
    conf_group="mean",
    reduce=None,
    ):
    """
    Calculates the Expected Semantic Error (ESE) for a predicted label map.

    Args:
        confidence_map: torch tensor of confidences outputted by the model (B x H x W)
        label_map: binary torch tensor of labels (B x H x W)
        measure: function that takes in two torch tensors and returns a float
        num_bins (int): number of confidence interval bins
        conf_group (str): how to group the confidences in each bin. Options: "mean"
        reduce Optional(str): whether to reduce the measure over the bins

    Returns:
        float: Bins Per Confidence
    """

    # Get the confidence bins
    bin_width = bins[1] - bins[0]

    # Get the regions of the prediction corresponding to each bin of confidence.
    confidence_regions = {bin: np.logical_and(pred >= bin, pred < (bin + bin_width)).bool() for bin in bins}

    # Iterate through the bins, and get the measure for each bin.
    measure_per_bin = np.zeros_like(bins)
    bin_amounts = np.zeros_like(bins)

    for b_idx, bin in enumerate(bins):
        label_region = label[confidence_regions[bin]][None, None, ...]

        # If there are no pixels in the region, then the measure is 0.
        bin_amounts[b_idx] = torch.sum(confidence_regions[bin])
        if bin_amounts[b_idx] == 0:
            measure_per_bin[b_idx] = 0
        else:
            simulated_ground_truth = torch.ones_like(label_region)
            bin_score = pixel_accuracy(simulated_ground_truth, label_region)

            # Need a way of aggregating the confidences in each bin. 
            if conf_group == "mean":
                bin_confidence = torch.mean(pred[confidence_regions[bin]]).item()
            else:
                raise NotImplementedError("Haven't implemented other confidence groups yet.")

            # Calculate the calibration error for the pixels in the bin.
            measure_per_bin[b_idx] = np.abs(bin_score - bin_confidence)

    if reduce == "mean":
        if bin_weighting == 'proportional':
            bin_weights = bin_amounts / np.sum(bin_amounts)
        elif bin_weighting == 'uniform':
            bin_weights = np.ones_like(bin_amounts) / len(bin_amounts)
        else:
            raise ValueError("Non-valid bin weighting scheme.")

        return np.average(measure_per_bin, weights=bin_weights)
    else:
        return measure_per_bin, bin_amounts 



