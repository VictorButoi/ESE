import numpy as np
import matplotlib.pyplot as plt
from ionpy.metrics import dice_score
import torch


def ECE(accuracies, confidences, num_bins=40, round_to=5):
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

    return np.round(ece, round_to)


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



