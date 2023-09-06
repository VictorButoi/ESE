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
    conf_bins: torch.Tensor,
    pred: torch.Tensor = None, 
    label: torch.Tensor = None,
    from_logits: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    assert len(pred.shape) == 2 and pred.shape == label.shape, f"pred and label must be 2D tensors of the same shape. Got {pred.shape} and {label.shape}."

    if from_logits:
        pred = torch.sigmoid(pred)

    # Get the confidence bin width
    bin_width = conf_bins[1] - conf_bins[0]

    # Calculate |confidence - accuracy| in each bin
    num_bins = len(conf_bins)
    accuracy_per_bin = torch.zeros(num_bins)
    ece_per_bin = torch.zeros(num_bins)
    bin_amounts = torch.zeros(num_bins)

    # Get the regions of the prediction corresponding to each bin of confidence.
    if pred is not None:
        confidences = pred.flatten()

    if label is not None:
        hard_pred = (pred >= 0.5)
        acc_image = (hard_pred == label).float()
        accuracies = acc_image.flatten()

    for bin_idx, c_bin in enumerate(conf_bins):
        in_bin = torch.logical_and(confidences >= c_bin, confidences < (c_bin + bin_width)).bool()
        num_pix_in_bin = torch.sum(in_bin)

        if num_pix_in_bin > 0:
            all_bin_accs = accuracies[in_bin]

            accuracy_in_bin = torch.mean(all_bin_accs) if torch.sum(all_bin_accs) > 0 else 0
            avg_confidence_in_bin = confidences[in_bin].mean()

            ece_per_bin[bin_idx] = (avg_confidence_in_bin - accuracy_in_bin).abs()
            accuracy_per_bin[bin_idx] = accuracy_in_bin
            bin_amounts[bin_idx] = num_pix_in_bin 
        
    return ece_per_bin, accuracy_per_bin, bin_amounts


@validate_arguments_init
def ESE(
    conf_bins: torch.Tensor,
    pred: torch.Tensor,
    label: torch.Tensor,
    from_logits: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the Expected Semantic Error (ESE) for a predicted label map.
    """
    assert len(pred.shape) == 2 and pred.shape == label.shape, f"pred and label must be 2D tensors of the same shape. Got {pred.shape} and {label.shape}."

    if from_logits:
        pred = torch.sigmoid(pred)

    # Get the confidence bins
    bin_width = conf_bins[1] - conf_bins[0]

    # Get the regions of the prediction corresponding to each bin of confidence.
    confidence_regions = {c_bin.item(): torch.logical_and(pred >= c_bin, pred < (c_bin + bin_width)).bool() for c_bin in conf_bins}

    # Iterate through the bins, and get the measure for each bin.
    num_bins = len(conf_bins)
    ese_per_bin = torch.zeros(num_bins)
    accuracy_per_bin = torch.zeros(num_bins)
    bin_amounts = torch.zeros(num_bins)

    for b_idx, c_bin in enumerate(conf_bins):

        # Get the region of image corresponding to the confidence
        bin_conf_region = confidence_regions[c_bin.item()]
        
        # Get the region of the label corresponding to this region.
        label_region = label[bin_conf_region][None, None, ...]

        # If there are some pixels in this confidence bin.
        if bin_conf_region.sum() != 0:
            bin_score = pixel_accuracy(torch.ones_like(label_region), label_region)
            bin_confidence = pred[bin_conf_region].mean()

            # Calculate the calibration error for the pixels in the bin.
            ese_per_bin[b_idx] = (bin_score - bin_confidence).abs()
            accuracy_per_bin[b_idx] = bin_score
    
    return ese_per_bin, accuracy_per_bin, bin_amounts


@validate_arguments_init
def ReCE(
    conf_bins: torch.Tensor,
    pred: torch.Tensor,
    label: torch.Tensor,
    from_logits: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the ReCE: Region-wise Calibration Error
    """
    assert len(pred.shape) == 2 and pred.shape == label.shape, f"pred and label must be 2D tensors of the same shape. Got {pred.shape} and {label.shape}."

    if from_logits:
        pred = torch.sigmoid(pred)

    # Get the confidence bins
    bin_width = conf_bins[1] - conf_bins[0]

    # Get the regions of the prediction corresponding to each bin of confidence.
    confidence_regions = {c_bin.item(): torch.logical_and(pred >= c_bin, pred < (c_bin + bin_width)).bool() for c_bin in conf_bins}

    # Iterate through the bins, and get the measure for each bin.
    num_bins = len(conf_bins) 
    accuracy_per_bin = torch.zeros(num_bins)
    rece_per_bin = torch.zeros(num_bins)
    bin_amounts = torch.zeros(num_bins)

    # Setup a visited regions to speed up connected components and reverse bins to avoid running connected components on the 0 bin.
    reversed_confidence_bins = torch.flip(conf_bins, [0])

    # Go through each bin, starting at the back so that we don't have to run connected components
    for rev_b_idx, c_bin in enumerate(reversed_confidence_bins):
        
        # Get the actual bin index
        b_idx = (num_bins - 1) - rev_b_idx

        # Get the region of image corresponding to the confidence
        bin_conf_region = confidence_regions[c_bin.item()].bool()

        # If there are some pixels in this confidence bin.
        if bin_conf_region.sum() != 0:
            # If we are not the last bin, get the connected components.
            conf_islands = get_connected_components(bin_conf_region) if b_idx > 0 else [bin_conf_region]
            
            # Iterate through each island, and get the measure for each island.
            num_islands = len(conf_islands)
            region_acc_scores = torch.zeros(num_islands)
            region_conf_scores = torch.zeros(num_islands)

            # Iterate through each island, and get the measure for each island.
            for isl_idx, island in enumerate(conf_islands):
                # Get the label corresponding to the island and simulate ground truth and make the right shape.
                label_region = label[island][None, None, ...]
                # Calculate the accuracy and mean confidence for the island.
                region_acc_scores[isl_idx] = pixel_accuracy(torch.ones_like(label_region), label_region)
                region_conf_scores[isl_idx] = pred[island].mean()
            
            # Get the accumulate metrics from all the islands
            avg_region_acc = region_acc_scores.mean()
            avg_region_conf = region_conf_scores.mean()
            
            # Calculate the average calibration error for the regions in the bin.
            rece_per_bin[b_idx] = (avg_region_acc - avg_region_conf).abs()
            accuracy_per_bin[b_idx] = avg_region_acc
            bin_amounts[b_idx] = num_islands # The number of islands is the number of regions.

    return rece_per_bin, accuracy_per_bin, bin_amounts
    