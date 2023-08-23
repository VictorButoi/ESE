from pydantic import validate_arguments
import numpy as np
from ionpy.metrics.util import _inputs_as_longlabels, InputMode
from torch import Tensor


def ECE(accuracies, confidences, num_bins=10, round_to=5):
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
