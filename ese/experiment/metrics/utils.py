# misc imports
import torch
from pydantic import validate_arguments
from typing import Optional, Literal, List
# local imports
from .utils import process_for_scoring, get_conf_region, init_stat_tracker
# ionpy imports
from ionpy.metrics import pixel_accuracy, pixel_precision
# misc imports
import torch
from typing import Literal
from pydantic import validate_arguments


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def gather_pixelwise_bin_stats(
    num_bins: int,
    conf_bins: torch.Tensor,
    conf_bin_widths: torch.Tensor,
    conf_map: torch.Tensor,
    pred_map: torch.Tensor,
    label_map: torch.Tensor,
    class_type: Literal["Binary", "Multi-class"],
    min_confidence: float = 0.001,
    include_background: bool = True,
    ) -> dict:
    # Process the inputs for scoring
    conf_map, pred_map, label_map = process_for_scoring(
        conf_map=conf_map, 
        pred_map=pred_map, 
        label_map=label_map, 
        class_type=class_type,
        min_confidence=min_confidence,
        include_background=include_background, 
    )
    # Keep track of different things for each bin.
    cal_info = init_stat_tracker(
        num_bins=num_bins,
        label_wise=False,
        ) 
    cal_info["bins"] = conf_bins
    cal_info["bin_widths"] = conf_bin_widths
    # Get the regions of the prediction corresponding to each bin of confidence.
    for bin_idx, conf_bin in enumerate(conf_bins):
        # Get the region of image corresponding to the confidence
        bin_conf_region = get_conf_region(bin_idx, conf_bin, conf_bin_widths, conf_map)
        # If there are some pixels in this confidence bin.
        if bin_conf_region.sum() > 0:
            # Calculate the average score for the regions in the bin.
            avg_bin_conf = conf_map[bin_conf_region].mean()
            avg_bin_measure, all_bin_measures = pixel_accuracy(
                pred_map[bin_conf_region], 
                label_map[bin_conf_region], 
                return_all=True
                )
            # Calculate the average calibration error for the regions in the bin.
            cal_info["bin_confs"][bin_idx] = avg_bin_conf 
            cal_info["bin_measures"][bin_idx] = avg_bin_measure 
            cal_info["bin_amounts"][bin_idx] = bin_conf_region.sum() 
            cal_info["bin_cal_scores"][bin_idx] = (avg_bin_conf - avg_bin_measure).abs()
            # Keep track of accumulate metrics over the bin.
            cal_info["measures_per_bin"][bin_idx] = all_bin_measures 
            cal_info["confs_per_bin"][bin_idx] = conf_map[bin_conf_region]

    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def gather_labelwise_pixelwise_bin_stats(
    num_bins: int,
    conf_bins: torch.Tensor,
    conf_bin_widths: torch.Tensor,
    conf_map: torch.Tensor,
    pred_map: torch.Tensor,
    label_map: torch.Tensor,
    class_type: Literal["Binary", "Multi-class"],
    min_confidence: float = 0.001,
    include_background: bool = True,
    ) -> dict:
    # Process the inputs for scoring
    conf_map, pred_map, label_map = process_for_scoring(
        conf_map=conf_map, 
        pred_map=pred_map, 
        label_map=label_map, 
        class_type=class_type,
        min_confidence=min_confidence,
        include_background=include_background, 
    )
    # Keep track of different things for each bin.
    pred_labels = pred_map.unique().tolist()
    cal_info = init_stat_tracker(
        num_bins=num_bins,
        label_wise=True,
        labels=pred_labels
        ) 
    cal_info["bins"] = conf_bins
    cal_info["bin_widths"] = conf_bin_widths
    # Get the regions of the prediction corresponding to each bin of confidence,
    # AND each prediction label.
    for bin_idx, conf_bin in enumerate(conf_bins):
        for lab_idx, p_label in enumerate(pred_labels):
            # Get the region of image corresponding to the confidence
            bin_conf_region = get_conf_region(
                bin_idx=bin_idx, 
                conf_bin=conf_bin, 
                conf_bin_widths=conf_bin_widths, 
                conf_map=conf_map,
                pred_map=pred_map,
                label=p_label
                )
            # If there are some pixels in this confidence bin.
            if bin_conf_region.sum() > 0:
                # Calculate the average score for the regions in the bin.
                avg_bin_conf = conf_map[bin_conf_region].mean()
                measure_func = pixel_accuracy if class_type == "Multi-class" else pixel_precision     
                avg_bin_measure, all_bin_measures = measure_func(
                    pred_map[bin_conf_region], 
                    label_map[bin_conf_region], 
                    return_all=True
                    )
                # Calculate the average calibration error for the regions in the bin.
                cal_info["lab_bin_confs"][lab_idx, bin_idx] = avg_bin_conf 
                cal_info["lab_bin_measures"][lab_idx, bin_idx] = avg_bin_measure 
                cal_info["lab_bin_amounts"][lab_idx, bin_idx] = bin_conf_region.sum() 
                cal_info["lab_bin_cal_scores"][lab_idx, bin_idx] = (avg_bin_conf - avg_bin_measure).abs()
                # Keep track of accumulate metrics over the bin.
                cal_info["lab_confs_per_bin"][p_label][bin_idx] = conf_map[bin_conf_region]
                cal_info["lab_measures_per_bin"][p_label][bin_idx] = all_bin_measures

    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def init_stat_tracker(
    num_bins: int,
    label_wise: bool,
    labels: Optional[List[int]] = None,
    ) -> dict:
    """
    Initialize the dictionary that will hold the statistics for each bin.
    """
    # Initialize the dictionary that will hold the statistics for each bin.
    if label_wise:
        num_labels = len(labels)
        bin_stats = {
            "lab_bin_cal_scores": torch.zeros((num_labels, num_bins)),
            "lab_bin_measures": torch.zeros((num_labels, num_bins)),
            "lab_bin_confs": torch.zeros((num_labels, num_bins)),
            "lab_bin_amounts": torch.zeros((num_labels, num_bins)),
            "lab_measures_per_bin": {lab: {} for lab in labels},
            "lab_confs_per_bin": {lab: {} for lab in labels},
        }
    else:
        bin_stats = {
            "bin_cal_scores": torch.zeros(num_bins),
            "bin_measures": torch.zeros(num_bins),
            "bin_confs": torch.zeros(num_bins),
            "bin_amounts": torch.zeros(num_bins),
            "measures_per_bin": {},
            "confs_per_bin": {},
        }
    # Track if the bins are label-wise or not.
    bin_stats["label-wise"] = label_wise
    return bin_stats


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def reduce_scores(
    score_per_bin: torch.Tensor, 
    amounts_per_bin: torch.Tensor, 
    weighting: str = "proportional",
    bin_weights: Optional[torch.Tensor] = None
    ) -> float:
    if bin_weights is None:
        if amounts_per_bin.sum() == 0:
            return 0.0
        elif weighting== 'proportional':
            bin_weights = amounts_per_bin / (amounts_per_bin).sum()
        elif weighting== 'uniform':
            bin_weights = torch.ones_like(amounts_per_bin) / len(amounts_per_bin)
        else:
            raise ValueError(f"Invalid bin weighting. Must be one of 'proportional' or 'uniform', got '{weighting}' instead.")
    # Multiply by the weights and sum.
    assert 1.0 - torch.sum(bin_weights) < 1e-5, f"Weights should approx. sum to 1.0, got {bin_weights.sum()} instead."
    return (score_per_bin * bin_weights).sum().item()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def split_tensor(
    tensor: torch.Tensor, 
    num_bins: int
    ):
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


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_conf_region(
    bin_idx: int, 
    conf_bin: torch.Tensor, 
    conf_bin_widths: torch.Tensor, 
    conf_map: torch.Tensor,
    pred_map: Optional[torch.Tensor] = None,
    label: Optional[int] = None,
    ):
    # Get the region of image corresponding to the confidence
    if conf_bin_widths[bin_idx] == 0:
        bin_conf_region = (conf_map == conf_bin) 
    else:
        bin_conf_region = torch.logical_and(conf_map >= conf_bin, conf_map < conf_bin + conf_bin_widths[bin_idx])
    if label is not None:
        bin_conf_region = torch.logical_and(bin_conf_region, pred_map==label)
    return bin_conf_region


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_bins(
    num_bins: int,
    start: float = 0.0,
    end: float = 1.0,
    adaptive: bool = False,
    conf_map: Optional[torch.Tensor] = None
    ):
    if adaptive:
        sorted_pix_values = torch.sort(conf_map.flatten())[0]
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
    else:
        conf_bins = torch.linspace(start, end, num_bins+1)[:-1] # Off by one error
        # Get the confidence bins
        bin_width = conf_bins[1] - conf_bins[0]
        conf_bin_widths = torch.ones(num_bins) * bin_width
    return conf_bins, conf_bin_widths


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def find_bins(confidences, bin_starts, bin_widths):
    """
    Given an array of confidence values, bin start positions, and individual bin widths, 
    find the bin index for each confidence.
    Args:
    - confidences (numpy.ndarray): A numpy array of confidence values.
    - bin_starts (torch.Tensor): A 1D tensor representing the start position of each confidence bin.
    - bin_widths (torch.Tensor): A 1D tensor representing the width of each confidence bin.
    Returns:
    - numpy.ndarray: A numpy array of bin indices corresponding to each confidence value. 
      If a confidence doesn't fit in any bin, its bin index is set to -1.
    """
    # Ensure that the bin_starts and bin_widths tensors have the same shape
    assert bin_starts.shape == bin_widths.shape, "bin_starts and bin_widths should have the same shape."
    # Convert the numpy confidences array to a PyTorch tensor
    confidences_tensor = torch.tensor(confidences)
    # Expand dimensions for broadcasting
    expanded_confidences = confidences_tensor.unsqueeze(-1)
    # Compare confidences against all bin ranges using broadcasting
    valid_bins = (expanded_confidences >= bin_starts) & (expanded_confidences < (bin_starts + bin_widths))
    # Get bin indices; if no valid bin is found for a confidence, the value will be -1
    bin_indices = torch.where(valid_bins, torch.arange(len(bin_starts)), -torch.ones_like(bin_starts)).max(dim=-1).values
    # Convert the resulting tensor back to a numpy array for the output
    return bin_indices.numpy() + 1 # Return + 1 so that we can talk about bun number #N


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def process_for_scoring(
    conf_map: torch.Tensor,
    pred_map: torch.Tensor,
    label_map: torch.Tensor,
    class_type: Literal["Binary", "Multi-class"],
    min_confidence: float,
    include_background: bool,
    ):
    if class_type == "Multi-class": 
        assert include_background, "Background must be included for multi-class."
    # Eliminate the super small predictions to get a better picture.
    label_map = label_map[conf_map >= min_confidence]
    pred_map = pred_map[conf_map >= min_confidence]
    conf_map = conf_map[conf_map >= min_confidence]
    if not include_background:
        foreground_pixels = (pred_map != 0)
        label_map = label_map[foreground_pixels]
        conf_map = conf_map[foreground_pixels]
        pred_map = pred_map[foreground_pixels]
    return conf_map, pred_map, label_map
