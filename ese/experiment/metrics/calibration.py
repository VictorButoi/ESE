# misc imports
import numpy as np
import torch

# ionpy imports
from ionpy.metrics import dice_score, pixel_accuracy
from ionpy.util.validation import validate_arguments_init

#local imports
from .utils import reduce_scores


@validate_arguments_init
def ECE(
    bins: np.ndarray,
    pred: torch.Tensor = None, 
    label: torch.Tensor = None,
    confidences: torch.Tensor = None,
    accuracies: torch.Tensor = None,
    bin_weighting: str = 'proportional',
    from_logits: bool = False,
    reduce: str = None,
):
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    if from_logits:
        pred = torch.sigmoid(pred)

    # Get the confidence bin width
    bin_width = bins[1] - bins[0]

    # Calculate |confidence - accuracy| in each bin
    accuracy_per_bin = torch.zeros(len(bins))
    ece_per_bin = torch.zeros(len(bins))
    bin_amounts = torch.zeros(len(bins))

    # Get the regions of the prediction corresponding to each bin of confidence.
    if pred is not None:
        confidences = pred.flatten()

    if label is not None:
        hard_pred = (pred >= 0.5)
        acc_image = (hard_pred == label).float()
        accuracies = acc_image.flatten()

    for bin_idx, bin in enumerate(bins):
        # Calculated |confidence - accuracy| in each bin
        in_bin = torch.logical_and(confidences >= bin, confidences < (bin + bin_width)).bool()
        num_pix_in_bin = torch.sum(in_bin)

        if num_pix_in_bin > 0:
            all_bin_accs = accuracies[in_bin]
            all_bin_confs = confidences[in_bin]

            accuracy_in_bin = torch.mean(all_bin_accs) if torch.sum(all_bin_accs) > 0 else 0
            avg_confidence_in_bin = torch.mean(all_bin_confs)

            ece_per_bin[bin_idx] = (avg_confidence_in_bin - accuracy_in_bin).abs()
            accuracy_per_bin[bin_idx] = accuracy_in_bin
            bin_amounts[bin_idx] = num_pix_in_bin 

    # Calculate ece as the weighted average, if there are any pixels in the bin.
    if reduce == "mean":
        return reduce_scores(ece_per_bin, bin_amounts, bin_weighting)
    else:
        return ece_per_bin.cpu().numpy(), accuracy_per_bin.cpu().numpy(), bin_amounts.cpu().numpy()


# Measure per bin.
@validate_arguments_init
def ESE(
    bins: np.ndarray,
    pred: torch.Tensor = None, 
    label: torch.Tensor = None,
    bin_weighting: str = 'proportional',
    from_logits: bool = False,
    reduce: str = None,
    ):
    """
    Calculates the Expected Semantic Error (ESE) for a predicted label map.
    """
    if from_logits:
        pred = torch.sigmoid(pred)

    # Get the confidence bins
    bin_width = bins[1] - bins[0]

    # Get the regions of the prediction corresponding to each bin of confidence.
    confidence_regions = {bin: torch.logical_and(pred >= bin, pred < (bin + bin_width)).bool() for bin in bins}

    # Iterate through the bins, and get the measure for each bin.
    accuracy_per_bin = torch.zeros(len(bins))
    ese_per_bin = torch.zeros(len(bins))
    bin_amounts = torch.zeros(len(bins))

    for b_idx, bin in enumerate(bins):
        label_region = label[confidence_regions[bin]][None, None, ...]

        # If there are no pixels in the region, then the measure is 0.
        bin_amounts[b_idx] = torch.sum(confidence_regions[bin])
        if bin_amounts[b_idx] == 0:
            accuracy_per_bin[b_idx] = 0
            ese_per_bin[b_idx] = 0
        else:
            simulated_ground_truth = torch.ones_like(label_region)
            bin_score = pixel_accuracy(simulated_ground_truth, label_region)
            bin_confidence = torch.mean(pred[confidence_regions[bin]]).item()

            # Calculate the calibration error for the pixels in the bin.
            ese_per_bin[b_idx] = (bin_score - bin_confidence).abs()
            accuracy_per_bin[b_idx] = bin_score
    
    if reduce == "mean":
        return reduce_scores(ese_per_bin, bin_amounts, bin_weighting)
    else:
        return ese_per_bin.cpu().numpy(), accuracy_per_bin.cpu().numpy(), bin_amounts.cpu().numpy()



