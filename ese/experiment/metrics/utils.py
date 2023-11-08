# misc imports
import torch
import numpy as np
from typing import Optional, Union 
import torch.nn.functional as F
from scipy.signal import convolve2d
from pydantic import validate_arguments


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
    conf_map: torch.Tensor,
    conf_bin_widths: torch.Tensor, 
    label: Optional[int] = None,
    num_neighbors: Optional[int] = None,
    pred_map: Optional[torch.Tensor] = None
    ):
    # Get the region of image corresponding to the confidence
    if conf_bin_widths[bin_idx] == 0:
        bin_conf_region = (conf_map == conf_bin) 
    else:
        upper_condition = conf_map <= conf_bin + conf_bin_widths[bin_idx]
        if bin_idx == 0:
            lower_condition = conf_map >= conf_bin
        else:
            lower_condition = conf_map > conf_bin
        bin_conf_region = torch.logical_and(lower_condition, upper_condition)
    # If we want to only pick things which match the label.
    if label is not None:
        bin_conf_region = torch.logical_and(bin_conf_region, pred_map==label)
    # If we only want the pixels with this particular number of neighbords that match the label
    if num_neighbors is not None:
        matching_neighbors_map = count_matching_neighbors(pred_map)
        bin_conf_region = torch.logical_and(bin_conf_region, matching_neighbors_map==num_neighbors)
    # The final region is the intersection of the conditions.
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
    valid_bins = (expanded_confidences > bin_starts) & (expanded_confidences <= (bin_starts + bin_widths))
    # Get bin indices; if no valid bin is found for a confidence, the value will be -1
    bin_indices = torch.where(valid_bins, torch.arange(len(bin_starts)), -torch.ones_like(bin_starts)).max(dim=-1).values
    # Convert the resulting tensor back to a numpy array for the output
    assert torch.all(bin_indices >= 0), "All bin indices should be greater than 0."
    return bin_indices.numpy() # Return + 1 so that we can talk about bun number #N


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def count_matching_neighbors(
    label_map: Union[torch.Tensor, np.ndarray],
    reflect_boundaries: bool
):
    # Optionally take in numpy array, convert to torch tensor
    if isinstance(label_map, np.ndarray):
        label_map = torch.from_numpy(label_map)
        return_numpy = True
    else:
        return_numpy = False
    # Ensure label_map is on the correct device (e.g., CUDA if using GPU)
    device = label_map.device
    # Get unique labels (assuming label_map is already a long tensor with discrete labels)
    label_map = label_map.long()
    unique_labels = label_map.unique()
    # Create an array to store the counts
    count_array = torch.zeros_like(label_map)
    # Define a 3x3 kernel of ones for the convolution
    kernel = torch.ones((1, 1, 3, 3), device=device)
    # Reflective padding if reflect_boundaries is True
    padding_mode = 'reflect' if reflect_boundaries else 'constant'
    for label in unique_labels:
        # Create a binary mask for the current label
        mask = (label_map == label).float()
        # Unsqueeze masks to fit conv2d expected input (Batch Size, Channels, Height, Width)
        mask_unsqueezed = mask.unsqueeze(0).unsqueeze(0)
        # Apply padding
        padded_mask = F.pad(mask_unsqueezed, pad=(1, 1, 1, 1), mode=padding_mode)
        # Convolve the mask with the kernel to get the neighbor count using 2D convolution
        neighbor_count = F.conv2d(padded_mask, kernel, padding=0)  # No additional padding needed
        # Squeeze the result back to the original shape (Height x Width)
        neighbor_count_squeezed = neighbor_count.squeeze().long()
        # Update the count_array where the label_map matches the current label
        count_array[label_map == label] = neighbor_count_squeezed[label_map == label]
    # Subtract 1 because the center pixel is included in the 3x3 neighborhood count
    count_array -= 1
    # Return the count_array
    if return_numpy:
        return count_array.numpy()
    else:
        return count_array


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def threshold_min_conf(
    conf_map: torch.Tensor,
    pred_map: torch.Tensor,
    label_map: torch.Tensor,
    min_confidence: float,
    ):
    # Eliminate the super small predictions to get a better picture.
    label_map = label_map[conf_map >= min_confidence]
    pred_map = pred_map[conf_map >= min_confidence]
    conf_map = conf_map[conf_map >= min_confidence]
    return conf_map, pred_map, label_map
