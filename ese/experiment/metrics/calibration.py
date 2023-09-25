# misc imports
import torch
from typing import Tuple
from pydantic import validate_arguments

# ionpy imports
from ionpy.metrics import pixel_accuracy, pixel_precision
from ionpy.util.islands import get_connected_components

measure_dict = {
    "Accuracy": pixel_accuracy,
    "Frequency": pixel_precision
}


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ECE(
    num_bins: int = 10,
    pred: torch.Tensor = None, 
    label: torch.Tensor = None,
    measure: str = "Accuracy",
    include_background: bool = False,
    min_prediction: float = 0.05,
    threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    assert len(pred.shape) == 2 and pred.shape == label.shape, f"pred and label must be 2D tensors of the same shape. Got {pred.shape} and {label.shape}."

    # Eliminate the super small predictions to get a better picture.
    label = label[pred >= min_prediction]
    pred = pred[pred >= min_prediction]

    # Define the confidence bins
    start = 0 if include_background else threshold
    conf_bins = torch.linspace(start, 1, num_bins+1)[:-1] # Off by one error
    
    # Get the confidence bins
    bin_width = conf_bins[1] - conf_bins[0]
    conf_bin_widths = torch.ones(num_bins) * bin_width

    # Keep track of different things for each bin.
    ece_per_bin = torch.zeros(num_bins)
    measure_per_bin = torch.zeros(num_bins)
    bin_amounts = torch.zeros(num_bins)

    # Get the regions of the prediction corresponding to each bin of confidence.
    for bin_idx, conf_bin in enumerate(conf_bins):

        # Get the region of image corresponding to the confidence
        if conf_bin_widths[bin_idx] == 0:
            bin_conf_region = (pred == conf_bin)
        else:
            bin_conf_region = torch.logical_and(pred >= conf_bin, pred < conf_bin + conf_bin_widths[bin_idx])

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
    
    return {
        "bins": conf_bins, 
        "bin_widths": conf_bin_widths, 
        "bin_amounts": bin_amounts,
        "scores_per_bin": ece_per_bin, 
        "accuracy_per_bin": measure_per_bin, 
    }


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ACE(
    num_bins: int = 10,
    pred: torch.Tensor = None, 
    label: torch.Tensor = None,
    measure: str = "Accuracy",
    include_background: bool = False,
    min_prediction: float = 0.05,
    threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    assert len(pred.shape) == 2 and pred.shape == label.shape, f"pred and label must be 2D tensors of the same shape. Got {pred.shape} and {label.shape}."

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
        split_tensors = torch.split(tensor, split_sizes)

        return split_tensors

    # Eliminate the super small predictions to get a better picture.
    label = label[pred >= min_prediction]
    pred = pred[pred >= min_prediction]

    # If you don't want to include background pixels, remove them.
    if not include_background:
        label = label[pred >= threshold]
        pred = pred[pred >= threshold]

    # Create the adaptive confidence bins.    
    sorted_pix_values = torch.sort(pred.flatten())[0]
    conf_bins_chunks = split_tensor(sorted_pix_values, num_bins)

    # Get the ranges of the confidences bins.
    bin_widths = []
    bin_starts = []
    for chunk in conf_bins_chunks:
        if len(chunk) > 0:
            bin_widths.append(chunk[-1] - chunk[0])
            bin_starts.append(chunk[0])
    conf_bin_widths = torch.Tensor(bin_widths)
    conf_bins = torch.Tensor(bin_starts)
    
    # Keep track of different things for each bin.
    ace_per_bin = torch.zeros(num_bins)
    measure_per_bin = torch.zeros(num_bins)
    bin_amounts = torch.zeros(num_bins)

    # Get the regions of the prediction corresponding to each bin of confidence.
    for bin_idx, conf_bin in enumerate(conf_bins):

        # Get the region of image corresponding to the confidence
        if conf_bin_widths[bin_idx] == 0:
            bin_conf_region = (pred == conf_bin)
        else:
            bin_conf_region = torch.logical_and(pred >= conf_bin, pred < conf_bin + conf_bin_widths[bin_idx])

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
    
    return {
        "bins": conf_bins, 
        "bin_widths": conf_bin_widths, 
        "bin_amounts": bin_amounts,
        "scores_per_bin": ace_per_bin, 
        "accuracy_per_bin": measure_per_bin, 
    }


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ReCE(
    num_bins: int = 10,
    pred: torch.Tensor = None,
    label: torch.Tensor = None,
    measure: str = "Accuracy",
    include_background: bool = False,
    min_prediction: float = 0.01,
    threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the ReCE: Region-wise Calibration Error
    """
    assert len(pred.shape) == 2 and pred.shape == label.shape, f"pred and label must be 2D tensors of the same shape. Got {pred.shape} and {label.shape}."

    # Eliminate the super small predictions to get a better picture.
    label = label[pred >= min_prediction]
    pred = pred[pred >= min_prediction]

    # Define the bins
    start = 0 if include_background else threshold
    conf_bins = torch.linspace(start, 1, num_bins+1)[:-1] # Off by one error

    # Get the confidence bins
    bin_width = conf_bins[1] - conf_bins[0]
    conf_bin_widths = torch.ones(num_bins) * bin_width

    # Iterate through the bins, and get the measure for each bin.
    measure_per_bin = torch.zeros(num_bins)
    rece_per_bin = torch.zeros(num_bins)
    bin_amounts = torch.zeros(num_bins)

    # Go through each bin, starting at the back so that we don't have to run connected components
    for bin_idx, conf_bin in enumerate(conf_bins):
        
        # Get the region of image corresponding to the confidence
        if conf_bin_widths[bin_idx] == 0:
            bin_conf_region = (pred == conf_bin)
        else:
            bin_conf_region = torch.logical_and(pred >= conf_bin, pred < conf_bin + conf_bin_widths[bin_idx])

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
            rece_per_bin[bin_idx] = (avg_region_measure - avg_region_conf).abs()
            measure_per_bin[bin_idx] = avg_region_measure
            bin_amounts[bin_idx] = num_islands # The number of islands is the number of regions.

    return {
        "bins": conf_bins, 
        "bin_widths": conf_bin_widths, 
        "bin_amounts": bin_amounts,
        "scores_per_bin": rece_per_bin, 
        "accuracy_per_bin": measure_per_bin, 
    }
    
