# misc imports
from typing import Optional, Literal
from pydantic import validate_arguments
import torch


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def reduce_scores(
    score_per_bin: torch.Tensor, 
    amounts_per_bin: torch.Tensor, 
    weighting: str = "proportional"
    ) -> float:

    if amounts_per_bin.sum() == 0:
        return 0.0
    elif weighting== 'proportional':
        bin_weights = amounts_per_bin / (amounts_per_bin).sum()
    elif weighting== 'uniform':
        bin_weights = torch.ones_like(amounts_per_bin) / len(amounts_per_bin)
    else:
        raise ValueError(f"Invalid bin weighting. Must be one of 'proportional' or 'uniform', got '{weighting}' instead.")
    assert 1.0 - torch.sum(bin_weights) < 1e-5, f"Weights should approx. sum to 1.0, got {bin_weights.sum()} instead."

    # Multiply by the weights and sum.
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
    conf_map: torch.Tensor
    ):
    # Get the region of image corresponding to the confidence
    if conf_bin_widths[bin_idx] == 0:
        bin_conf_region = (conf_map == conf_bin)
    else:
        bin_conf_region = torch.logical_and(conf_map >= conf_bin, conf_map < conf_bin + conf_bin_widths[bin_idx])
    return bin_conf_region


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_bins(
    num_bins: int,
    metric: str, 
    include_background: bool, 
    class_type: Literal["Binary", "Multi-class"],
    conf_map: Optional[torch.Tensor] = None
    ):
    if class_type == "Multi-class": 
        assert include_background, "Background must be included for multi-class."

    if metric == "ACE":
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
        # Define the bins
        if class_type == "Multi-class":
            start = 0
        else:
            start = 0 if include_background else 0.5 

        conf_bins = torch.linspace(start, 1, num_bins+1)[:-1] # Off by one error

        # Get the confidence bins
        bin_width = conf_bins[1] - conf_bins[0]
        conf_bin_widths = torch.ones(num_bins) * bin_width
    
    return conf_bins, conf_bin_widths


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def process_for_scoring(
    conf_map: torch.Tensor,
    pred_map: torch.Tensor,
    label: torch.Tensor,
    class_type: Literal["Binary", "Multi-class"],
    min_confidence: float,
    include_background: bool,
    ):
    if class_type == "Multi-class": 
        assert include_background, "Background must be included for multi-class."

    # Eliminate the super small predictions to get a better picture.
    label = label[conf_map >= min_confidence]
    pred_map = pred_map[conf_map >= min_confidence]
    conf_map = conf_map[conf_map >= min_confidence]

    if not include_background:
        foreground_pixels = (pred_map != 0)
        label = label[foreground_pixels]
        conf_map = conf_map[foreground_pixels]
        pred_map = pred_map[foreground_pixels]

    return conf_map, pred_map, label