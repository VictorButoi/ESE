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
    bin_range=(0, 1),
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
    if isinstance(bins, int): 
        bins = np.linspace(bin_range[0], bin_range[1], bins+1)[:-1] # Chop off last bin edge

    bin_width = bins[1] - bins[0]

    # Calculate |confidence - accuracy| in each bin
    scores = np.zeros(len(bins))
    bin_amounts = np.zeros(len(bins))

    conf_a_thresh = confidences[confidences >= bin_range[0]] # If we want to ignore, then do so.
    acc_a_thresh = accuracies[confidences >= bin_range[0]]

    for bin_idx, bin in enumerate(bins):
        # Calculated |confidence - accuracy| in each bin
        in_bin = np.logical_and(conf_a_thresh >= bin, conf_a_thresh < (bin + bin_width))
        num_pix_in_bin = np.sum(in_bin)

        if num_pix_in_bin > 0:
            accuracy_in_bin = np.mean(acc_a_thresh[in_bin])
            avg_confidence_in_bin = np.mean(conf_a_thresh[in_bin])

            scores[bin_idx] = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            bin_amounts[bin_idx] = num_pix_in_bin 

    # Calculate ece as the weighted average, if there are any pixels in the bin.
    if reduce == "mean":
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



# ADRIAN's ALTERNATIVE CALIBRATION PLOT
def AACP(predictions, num_bins=40, round_to=5):
                       
    lsp = np.linspace(0, 1, num_bins+1)
    accs = []

    # For each bin look at the accuracies
    for bin in range(num_bins):
        dice_scores_per_sub = []
        # Iterate through each predictions
        for pred_dict in predictions:
            pixels = np.logical_and(pred_dict["soft_pred"] >= lsp[bin], pred_dict["soft_pred"] < lsp[bin+1])
            label = torch.from_numpy(pred_dict["label"]).float()[pixels][None]
            ones = torch.ones_like(label)
            if ones.shape[1] > 0:
                ds = np.round(dice_score(ones, label), round_to)
                # Add the dice score to running list
                dice_scores_per_sub.append(ds)
        # Place it in our array
        zero_pad = np.zeros((len(predictions) - len(dice_scores_per_sub)))
        accs.append(np.concatenate([np.array(dice_scores_per_sub), zero_pad]))
    
    plt.bar(lsp, accs, width=1/num_bins)



