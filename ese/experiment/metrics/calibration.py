# misc imports
from typing import Tuple
import numpy as np
import torch

# ionpy imports
from ionpy.metrics import pixel_accuracy
from ionpy.util.validation import validate_arguments_init
from ionpy.util.islands import get_connected_components


@validate_arguments_init
def ECE(
    bins: np.ndarray,
    pred: torch.Tensor = None, 
    label: torch.Tensor = None,
    confidences: torch.Tensor = None,
    accuracies: torch.Tensor = None,
    from_logits: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    assert len(pred.shape) == 2 and pred.shape == label.shape, f"pred and label must be 2D tensors of the same shape. Got {pred.shape} and {label.shape}."

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

    return ece_per_bin.cpu().numpy(), accuracy_per_bin.cpu().numpy(), bin_amounts.cpu().numpy()


# Measure per bin.
@validate_arguments_init
def ESE(
    bins: np.ndarray,
    pred: torch.Tensor = None, 
    label: torch.Tensor = None,
    from_logits: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the Expected Semantic Error (ESE) for a predicted label map.
    """
    assert len(pred.shape) == 2 and pred.shape == label.shape, f"pred and label must be 2D tensors of the same shape. Got {pred.shape} and {label.shape}."

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
    
    return ese_per_bin.cpu().numpy(), accuracy_per_bin.cpu().numpy(), bin_amounts.cpu().numpy()


# Measure per bin.
@validate_arguments_init
def ReCE(
    bins: np.ndarray,
    pred: torch.Tensor = None, 
    label: torch.Tensor = None,
    from_logits: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the ReCE: Region-wise Calibration Error
    """
    assert len(pred.shape) == 2 and pred.shape == label.shape, f"pred and label must be 2D tensors of the same shape. Got {pred.shape} and {label.shape}."

    if from_logits:
        pred = torch.sigmoid(pred)

    # Get the confidence bins
    bin_width = bins[1] - bins[0]

    # Get the regions of the prediction corresponding to each bin of confidence.
    confidence_regions = {bin: torch.logical_and(pred >= bin, pred < (bin + bin_width)).bool() for bin in bins}

    # Iterate through the bins, and get the measure for each bin.
    accuracy_per_bin = torch.zeros(len(bins))
    rece_per_bin = torch.zeros(len(bins))
    bin_amounts = torch.zeros(len(bins))

    # Setup a visited regions to speed up connected components and reverse bins to avoid running connected components on the 0 bin.
    reversed_bins = bins[::-1]
    visited_regions = torch.zeros_like(pred).bool()

    # Go through each bin, starting at the back so that we don't have to run connected components
    for rev_b_idx, bin in enumerate(reversed_bins):
        
        # Get the actual bin index
        b_idx = len(bins) - rev_b_idx - 1

        # Get the binary map of a particular conidence region.
        conf_region = confidence_regions[bin].int()

        if torch.sum(conf_region) == 0: # If there are no pixels, there are no regions
            accuracy_per_bin[b_idx] = 0
            rece_per_bin[b_idx] = 0
            bin_amounts[b_idx] = 0
        else:
            # If we are not the last bin, get the connected components and add it to the visited regions.
            if b_idx > 0:
                conf_islands = get_connected_components(
                    array=conf_region, 
                    visited=visited_regions
                    )
                visited_regions[confidence_regions[bin]] = True
            else:
                conf_islands = [conf_region.bool()]
            
            # Iterate through each island, and get the measure for each island.
            num_islands = len(conf_islands)
            region_ece_scores = torch.zeros(num_islands)
            region_acc_scores = torch.zeros(num_islands)

            for isl_idx, island in enumerate(conf_islands):
                # Get the label corresponding to the island and simulate ground truth.
                label_region = label[island][None, None, ...]
                simulated_ground_truth = torch.ones_like(label_region)
                
                # Calculate the accuracy and mean confidenc for the island..
                region_acc = pixel_accuracy(simulated_ground_truth, label_region)
                mean_region_confidence = torch.mean(pred[island]).item()

                # Calculate the calibration error for the pixels in the bin.
                region_ece_scores[isl_idx] = (region_acc - mean_region_confidence).abs()
                region_acc_scores[isl_idx] = region_acc 
            
            # Calculate the average calibration error for the regions in the bin.
            rece_per_bin[b_idx] = torch.mean(region_ece_scores)
            accuracy_per_bin[b_idx] = torch.mean(region_acc_scores)
            bin_amounts[b_idx] = num_islands # The number of islands is the number of regions.

    return rece_per_bin.cpu().numpy(), accuracy_per_bin.cpu().numpy(), bin_amounts.cpu().numpy()
    