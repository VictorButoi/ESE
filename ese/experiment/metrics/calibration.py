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
    num_bins: int = 10,
    pred: torch.Tensor = None, 
    label: torch.Tensor = None,
    measure: str = "Accuracy",
    include_background: bool = False,
    threshold: float = 0.5,
    from_logits: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    assert len(pred.shape) == 2 and pred.shape == label.shape, f"pred and label must be 2D tensors of the same shape. Got {pred.shape} and {label.shape}."

    # If conf_bins is not predefined, create them. 
    conf_bins = torch.linspace(0, 1, num_bins+1)[:-1] # Off by one error

    if not include_background:
        conf_bins = conf_bins[conf_bins >= threshold]
    
    if from_logits:
        pred = torch.sigmoid(pred)

    # Get the confidence bins
    bin_width = conf_bins[1] - conf_bins[0]
    conf_bin_widths = torch.ones(num_bins) * bin_width

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
    
    return conf_bins, conf_bin_widths, ece_per_bin, measure_per_bin, bin_amounts


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ACE(
    num_bins: int = 10,
    pred: torch.Tensor = None, 
    label: torch.Tensor = None,
    measure: str = "Accuracy",
    include_background: bool = False,
    threshold: float = 0.5,
    min_conf: float = 0.05,
    from_logits: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    assert len(pred.shape) == 2 and pred.shape == label.shape, f"pred and label must be 2D tensors of the same shape. Got {pred.shape} and {label.shape}."

    if from_logits:
        pred = torch.sigmoid(pred)

    def split_tensor(tensor, num_bins):
        """
        Split a tensor of shape [N] into num_bins smaller tensors such that
        the difference in size between any of the chunks is at most 1.

        Args:
        - tensor (torch.Tensor): Tensor of shape [N] to split
        - num_bins (int): Number of bins/tensors to split into

        Returns:
        - List of tensors
        """
        N = tensor.size(0)
        base_size = N // num_bins
        remainder = N % num_bins
        # This will give a list where the first `remainder` numbers are 
        # (base_size + 1) and the rest are `base_size`.
        split_sizes = [base_size + 1 if i < remainder else base_size for i in range(num_bins)]
        return torch.split(tensor, split_sizes)

    # If you don't want to include background pixels, remove them.
    if not include_background:
        pred = pred[pred >= threshold]

    # Eliminate the super small values, used when background is included typically
    if min_conf > 0:
        pred = pred[pred >= min_conf]

    # Create the adaptive confidence bins.    
    sorted_pix_values = torch.sort(pred.flatten())[0]
    conf_bins_chunks = split_tensor(sorted_pix_values, num_bins)
    # Get the ranges o the confidences bins.
    conf_bin_widths = [(chunk[-1] - chunk[0]) for chunk in conf_bins_chunks]
    conf_bins = torch.Tensor([chunk[0] for chunk in conf_bins_chunks])
    # Finally build the confidence regions.
    confidence_regions = {conf_bins[bin_idx].item(): torch.logical_and(pred >= conf_bins[bin_idx], 
                                                                          pred < conf_bins[bin_idx] + conf_bin_widths[bin_idx]) 
                                                                          for bin_idx in range(num_bins)}
    # Keep track of different things for each bin.
    ace_per_bin = torch.zeros(num_bins)
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

            ace_per_bin[bin_idx] = (avg_metric - avg_confidence).abs()
            measure_per_bin[bin_idx] = avg_metric 
            bin_amounts[bin_idx] = bin_conf_region.sum() 
    
    return conf_bins, conf_bin_widths, ace_per_bin, measure_per_bin, bin_amounts


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ReCE(
    num_bins: int = 10,
    pred: torch.Tensor = None,
    label: torch.Tensor = None,
    measure: str = "Accuracy",
    include_background: bool = False,
    threshold: float = 0.5,
    from_logits: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the ReCE: Region-wise Calibration Error
    """
    assert len(pred.shape) == 2 and pred.shape == label.shape, f"pred and label must be 2D tensors of the same shape. Got {pred.shape} and {label.shape}."

    # If conf_bins is not predefined, create them. 
    conf_bins = torch.linspace(0, 1, num_bins+1)[:-1] # Off by one error

    if not include_background:
        conf_bins = conf_bins[conf_bins >= threshold]

    if from_logits:
        pred = torch.sigmoid(pred)

    # Get the confidence bins
    bin_width = conf_bins[1] - conf_bins[0]
    conf_bin_widths = torch.ones(num_bins) * bin_width

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

    return conf_bins, conf_bin_widths, rece_per_bin, measure_per_bin, bin_amounts
    
