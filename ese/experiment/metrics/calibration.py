import numpy as np
import matplotlib.pyplot as plt
from ionpy.metrics import dice_score
import torch


def ECE(accuracies, confidences, num_bins=40):
    """
    Calculates the Expected Calibration Error (ECE) for a model.
    Args:
        accuracies: numpy array of calibration accuracies for each bin
        confidences: numpy array of confidences outputted by the model
        num_bins (int): number of confidence interval bins
    Returns:
        float: Expected Calibration Error
    """
    bin_boundaries = np.linspace(0, 1, num_bins+1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = np.logical_and(confidences >= bin_lower, confidences < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


# Measure per bin.
def MPB(
    confidence_map, 
    label_map, 
    measure, 
    num_bins    
    ):
    """
    Calculates the Bins Per Confidence (BPC) for a model.
    Args:
        confidence_map: torch tensor of confidences outputted by the model (B x H x W)
        label_map: binary torch tensor of labels (B x H x W)
        measure: function that takes in two torch tensors and returns a float
        num_bins (int): number of confidence interval bins
    Returns:
        float: Bins Per Confidence
    """

    # Get the confidence bins
    bins = np.linspace(0, 1, num_bins+1)

    # Get the regions of the prediction corresponding to each bin of confidence.
    confidence_regions = {i: np.logical_and(confidence_map >= bins[i], confidence_map < bins[i+1]).bool() for i in range(num_bins)}

    # Iterate through the bins, and get the measure for each bin.
    measure_per_bin = np.zeros(num_bins+1)
    for i in range(num_bins):
        label_region = label_map[confidence_regions[i]][None, None, ...]
        # If there are no pixels in the region, then the measure is 0.
        if torch.sum(confidence_regions[i]) == 0:
            measure_per_bin[i] = 0
        else:
            simulated_ground_truth = torch.ones_like(label_region)
            measure_per_bin[i] = measure(simulated_ground_truth, label_region)

    return measure_per_bin



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



