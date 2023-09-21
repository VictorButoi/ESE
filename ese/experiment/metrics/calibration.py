# misc imports
import torch
from typing import Tuple
from pydantic import validate_arguments

# ionpy imports
from ionpy.metrics import pixel_accuracy, pixel_precision
from ionpy.util.islands import get_connected_components

measure_dict = {
    "Accuracy": pixel_accuracy,
    "Precision": pixel_precision
}


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ECE(
    conf_bins: torch.Tensor = None,
    num_bins: int = 10,
    pred: torch.Tensor = None, 
    label: torch.Tensor = None,
    measure: str = "Accuracy",
    include_background: bool = False,
    threshhold: float = 0.5,
    from_logits: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    assert len(pred.shape) == 2 and pred.shape == label.shape, f"pred and label must be 2D tensors of the same shape. Got {pred.shape} and {label.shape}."

    # If conf_bins is not predefined, create them. 
    if conf_bins is None:
        conf_bins = torch.linspace(0, 1, num_bins+1)[:-1] # Off by one error
        num_bins = len(conf_bins)
    if not include_background:
        conf_bins = conf_bins[conf_bins >= threshhold]

    if from_logits:
        pred = torch.sigmoid(pred)

    # Get the confidence bins
    bin_width = conf_bins[1] - conf_bins[0]

    # Get the regions of the prediction corresponding to each bin of confidence.
    confidence_regions = {c_bin.item(): torch.logical_and(pred >= c_bin, pred < (c_bin + bin_width)).bool() for c_bin in conf_bins}

    # Keep track of different things for each bin.
    ece_per_bin = torch.zeros(num_bins)
    measure_per_bin = torch.zeros(num_bins)
    bin_amounts = torch.zeros(num_bins)

    # Get the regions of the prediction corresponding to each bin of confidence.
    for bin_idx, c_bin in enumerate(conf_bins):

        # Get the region of image corresponding to the confidence
        bin_conf_region = confidence_regions[c_bin.item()]

        # If there are some pixels in this confidence bin.
        if bin_conf_region.sum() > 0:
            bin_pred = pred[bin_conf_region]
            bin_label = label[bin_conf_region]

            # Calculate the accuracy and mean confidence for the island.
            avg_metric = measure_dict[measure](bin_pred, bin_label)
            avg_confidence = bin_pred.mean()

            ece_per_bin[bin_idx] = (avg_metric - avg_confidence).abs()
            measure_per_bin[bin_idx] = avg_metric 
            bin_amounts[bin_idx] = bin_conf_region.sum() 
    
    return ece_per_bin, measure_per_bin, bin_amounts


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ReCE(
    conf_bins: torch.Tensor = None,
    num_bins: int = 10,
    pred: torch.Tensor = None,
    label: torch.Tensor = None,
    measure: str = "Accuracy",
    include_background: bool = False,
    threshhold: float = 0.5,
    from_logits: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the ReCE: Region-wise Calibration Error
    """
    assert len(pred.shape) == 2 and pred.shape == label.shape, f"pred and label must be 2D tensors of the same shape. Got {pred.shape} and {label.shape}."

    # If conf_bins is not predefined, create them. 
    if conf_bins is None:
        conf_bins = torch.linspace(0, 1, num_bins+1)[:-1] # Off by one error
        num_bins = len(conf_bins)
    if not include_background:
        conf_bins = conf_bins[conf_bins >= threshhold]

    if from_logits:
        pred = torch.sigmoid(pred)

    # Get the confidence bins
    bin_width = conf_bins[1] - conf_bins[0]

    # Get the regions of the prediction corresponding to each bin of confidence.
    confidence_regions = {c_bin.item(): torch.logical_and(pred >= c_bin, pred < (c_bin + bin_width)).bool() for c_bin in conf_bins}

    # Iterate through the bins, and get the measure for each bin.
    measure_per_bin = torch.zeros(num_bins)
    rece_per_bin = torch.zeros(num_bins)
    bin_amounts = torch.zeros(num_bins)

    # Go through each bin, starting at the back so that we don't have to run connected components
    for b_idx, c_bin in enumerate(conf_bins):
        
        # Get the region of image corresponding to the confidence
        bin_conf_region = confidence_regions[c_bin.item()].bool()

        # If there are some pixels in this confidence bin.
        if bin_conf_region.sum() != 0:
            # If we are not the last bin, get the connected components.
            conf_islands = get_connected_components(bin_conf_region)
            
            # Iterate through each island, and get the measure for each island.
            num_islands = len(conf_islands)
            region_metric_scores = torch.zeros(num_islands)
            region_conf_scores = torch.zeros(num_islands)

            # Iterate through each island, and get the measure for each island.
            for isl_idx, island in enumerate(conf_islands):
                # Get the island primitives
                bin_pred = pred[island]                
                bin_label = label[island]
                # Calculate the accuracy and mean confidence for the island.
                region_metric_scores[isl_idx] = measure_dict[measure](bin_pred, bin_label)
                region_conf_scores[isl_idx] = bin_pred.mean()
            
            # Get the accumulate metrics from all the islands
            avg_region_measure = region_metric_scores.mean()
            avg_region_conf = region_conf_scores.mean()
            
            # Calculate the average calibration error for the regions in the bin.
            rece_per_bin[b_idx] = (avg_region_measure - avg_region_conf).abs()
            measure_per_bin[b_idx] = avg_region_measure
            bin_amounts[b_idx] = num_islands # The number of islands is the number of regions.

    return rece_per_bin, measure_per_bin, bin_amounts
    
