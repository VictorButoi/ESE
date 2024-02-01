# torch imports
import torch
from torch import Tensor
import torch.nn.functional as F
# misc imports
import numpy as np
from pydantic import validate_arguments
from typing import Optional, Union, List
from scipy.ndimage import (
    distance_transform_edt, 
    binary_erosion, 
    label
)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_edge_pixels(
    y_pred: Tensor,
    y_true: Tensor,
    image_info_dict: dict
    ) -> Tensor:
    """
    Returns the edge pixels of the ground truth label map.
    """
    # Get the edge map.
    if "true_matching_neighbors_map" in image_info_dict:
        y_true_edge_map = (image_info_dict["true_matching_neighbors_map"] < 8)
    else:
        y_true_squeezed = y_true.squeeze()
        y_true_edge_map = get_edge_map(y_true_squeezed)
    # Get the edge regions of both the prediction and the ground truth.
    y_pred_e_reg = y_pred[..., y_true_edge_map]
    y_true_e_reg = y_true[..., y_true_edge_map]
    # Add a height dim.
    y_edge_pred = y_pred_e_reg.unsqueeze(-2)
    y_edge_true= y_true_e_reg.unsqueeze(-2)
    # Return the edge-ified values.
    return y_edge_pred, y_edge_true


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def reduce_bin_errors(
    error_per_bin: Tensor, 
    amounts_per_bin: Tensor, 
    weighting: str = "proportional",
    bin_weights: Optional[Tensor] = None
    ) -> float:
    if bin_weights is None:
        if amounts_per_bin.sum() == 0:
            return torch.tensor(0.0)
        elif weighting == 'proportional':
            bin_weights = amounts_per_bin / (amounts_per_bin).sum()
        else:
            raise ValueError(f"Invalid bin weighting. Must be 'proportional', got '{weighting}' instead.")
    # Multiply by the weights and sum.
    assert 1.0 - torch.sum(bin_weights) < 1e-5, f"Weights should approx. sum to 1.0, got {bin_weights.sum()} instead."
    reduced_error = (error_per_bin * bin_weights).sum()
    assert 1 >= reduced_error >= 0, f"Reduced error should be between 0 and 1, got {reduced_error} instead."

    return reduced_error


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def split_tensor(
    tensor: Tensor, 
    num_bins: int
    ):
    """
    Split a tensor of shape [N] into num_bins smaller tensors such that
    the difference in size between any of the chunks is at most 1.

    Args:
    - tensor (Tensor): Tensor of shape [N] to split
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
    bin_ownership_map: Tensor,
    true_label: Optional[int] = None,
    true_lab_map: Optional[Tensor] = None,
    pred_label: Optional[int] = None,
    pred_lab_map: Optional[Tensor] = None,
    true_nn: Optional[int] = None,
    true_num_neighbors_map: Optional[Tensor] = None,
    pred_nn: Optional[int] = None,
    pred_num_neighbors_map: Optional[Tensor] = None,
    edge_only: bool = False,
    neighborhood_width: Optional[int] = 3,
    ignore_index: Optional[int] = None,
    ):
    # We want to only pick things in the bin indicated.
    bin_conf_region = (bin_ownership_map == bin_idx)
    # If we want to only pick things which match the ground truth label.
    if true_label is not None:
        bin_conf_region = torch.logical_and(bin_conf_region, (true_lab_map==true_label))
    # If we want to only pick things which match the pred label.
    if pred_label is not None:
        bin_conf_region = torch.logical_and(bin_conf_region, (pred_lab_map==pred_label))
    # If we want to ignore a particular label, then we set it to 0.
    if ignore_index is not None:
        bin_conf_region = torch.logical_and(bin_conf_region, (true_lab_map != ignore_index))
    # If we only want the pixels with this particular number of neighbords that match the label
    if true_nn is not None:
        bin_conf_region = torch.logical_and(bin_conf_region, true_num_neighbors_map==true_nn)
    # If we only want the pixels with this particular number of neighbords that match the label
    if pred_nn is not None:
        bin_conf_region = torch.logical_and(bin_conf_region, pred_num_neighbors_map==pred_nn)
    # If we are doing edges only, then select those uses 
    if edge_only:
        n_neighbor_classes = (neighborhood_width**2 - 1)
        bin_conf_region = torch.logical_and(bin_conf_region, true_num_neighbors_map < n_neighbor_classes)
    # The final region is the intersection of the conditions.
    return bin_conf_region


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def threshold_min_conf(
    y_pred: Tensor,
    y_hard: Tensor,
    y_true: Tensor,
    min_confidence: float,
    ):
    # Eliminate the super small predictions to get a better picture.
    y_pred = y_pred[y_pred >= min_confidence]
    y_hard = y_hard[y_pred >= min_confidence]
    y_true = y_true[y_pred >= min_confidence]
    return y_pred, y_hard, y_true


# Get a distribution of per-pixel accuracy as a function of distance to a boundary for a 2D image.
# and this is done without bins.
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_perpix_boundary_dist(
    y_pred: np.ndarray,
) -> np.ndarray:
    # Expand the segmentation map with a 1-pixel border
    padded_y_pred = np.pad(y_pred, ((1, 1), (1, 1)), mode='constant', constant_values=-1)
    # Define a structuring element for the erosion
    struct_elem = np.ones((3, 3))
    boundaries = np.zeros_like(padded_y_pred, dtype=bool)
    for label in np.unique(y_pred):
        # Create a binary map of the current label
        binary_map = (padded_y_pred == label)
        # Erode the binary map
        eroded = binary_erosion(binary_map, structure=struct_elem)
        # Find boundary by XOR operation
        boundaries |= (binary_map ^ eroded)
    # Remove the padding
    boundary_image = boundaries[1:-1, 1:-1]
    # Compute distance to the nearest boundary
    distance_to_boundaries = distance_transform_edt(~boundary_image)
    return distance_to_boundaries


# Get a distribution of per-pixel accuracy as a function of the size of the instance that it was 
# predicted in.
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_perpix_group_size(
    y_pred: np.ndarray,
) -> np.ndarray:
    # Create an empty tensor with the same shape as the input
    size_map = np.zeros_like(y_pred)
    # Get unique labels in the segmentation map
    unique_labels = np.unique(y_pred)
    for label_val in unique_labels:
        # Create a mask for the current label
        mask = (y_pred == label_val)
        # Find connected components for the current mask
        labeled_array, num_features = label(mask)
        for i in range(1, num_features + 1):
            component_mask = (labeled_array == i)
            # Compute the size of the current component
            size = component_mask.sum().item()
            # Replace the pixels of the component with its size in the size_map tensor
            size_map[mask & component_mask] = size
    # If you want to return a tuple of per-pixel accuracies and the size map
    return size_map


# Get the size of each region of label in the label-map,
# and return it a s a dictionary: Label -> Sizes. A region
# of label is a contiguous set of pixels with the same label.
def get_label_region_sizes(y_true):
    # Get unique labels in the segmentation map
    unique_labels = np.unique(y_true)
    lab_reg_size_dict = {}
    for label_val in unique_labels:
        lab_reg_size_dict[label_val] = []
        # Create a mask for the current label
        mask = (y_true==label_val)
        # Find connected components for the current mask
        labeled_array, num_features = label(mask)
        for i in range(1, num_features + 1):
            component_mask = (labeled_array == i)
            # Compute the size of the current component
            size = component_mask.sum().item()
            # Replace the pixels of the component with its size in the size_map tensor
            lab_reg_size_dict[label_val].append(size)  
    # Return the dictionary of label region sizes
    return lab_reg_size_dict 


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_bins(
    num_bins: int,
    start: float = 0.0,
    end: float = 1.0,
    adaptive: bool = False,
    y_pred: Optional[Tensor] = None,
    device: Optional[torch.device] = "cuda"
    ):
    if adaptive:
        sorted_pix_values = torch.sort(y_pred.flatten())[0]
        conf_bins_chunks = split_tensor(sorted_pix_values, num_bins)
        # Get the ranges of the confidences bins.
        bin_widths = []
        bin_starts = []
        for chunk in conf_bins_chunks:
            if len(chunk) > 0:
                bin_widths.append(chunk[-1] - chunk[0])
                bin_starts.append(chunk[0])
        conf_bin_widths = Tensor(bin_widths)
        conf_bins = Tensor(bin_starts)
    else:
        conf_bins = torch.linspace(start, end, num_bins+1)[:-1] # Off by one error
        # Get the confidence bins
        bin_width = conf_bins[1] - conf_bins[0]
        conf_bin_widths = torch.ones(num_bins) * bin_width

    if device is not None:
        return conf_bins.to(device), conf_bin_widths.to(device)
    else:
        return conf_bins, conf_bin_widths


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def find_bins(
    confidences: Tensor, 
    bin_starts: Tensor, 
    bin_widths: Tensor,
    device: Optional[torch.device] = "cuda"
    ):
    """
    Given an array of confidence values, bin start positions, and individual bin widths, 
    find the bin index for each confidence.
    Args:
    - confidences (Tensor): A batch torch tensor of confidence values.
    - bin_starts (Tensor): A 1D tensor representing the start position of each confidence bin.
    - bin_widths (Tensor): A 1D tensor representing the width of each confidence bin.
    Returns:
    - numpy.ndarray: A numpy array of bin indices corresponding to each confidence value. 
      If a confidence doesn't fit in any bin, its bin index is set to -1.
    """
    # Ensure that the bin_starts and bin_widths tensors have the same shape
    assert len(confidences.shape) == 3, "Confidences must be (B, H, W)."
    assert bin_starts.shape == bin_widths.shape, "bin_starts and bin_widths should have the same shape."
    # Expand dimensions for broadcasting
    expanded_confidences = confidences.unsqueeze(-1)
    # Compare confidences against all bin ranges using broadcasting
    valid_bins = (expanded_confidences > bin_starts) & (expanded_confidences <= (bin_starts + bin_widths))
    # Get bin indices; if no valid bin is found for a confidence, the value will be -1
    if device is not None:
        bin_indices = torch.where(valid_bins, torch.arange(len(bin_starts)).to(device), -torch.ones_like(bin_starts)).max(dim=-1).values
    else:
        bin_indices = torch.where(valid_bins, torch.arange(len(bin_starts)), -torch.ones_like(bin_starts)).max(dim=-1).values
    # Place all things in bin -1 in bin 0, this can happen when stuff is perfectly the boundary of bin_starts.
    bin_indices[bin_indices == -1] = 0
    return bin_indices


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def count_matching_neighbors(
    lab_map: Union[Tensor, np.ndarray],
    neighborhood_width: int = 3,
):
    if len(lab_map.shape) == 4:
        lab_map = lab_map.squeeze(1) # Attempt to squeeze out the channel dimension.
    assert len(lab_map.shape) == 3,\
        f"Label map shape should be: (B, H, W), got shape: {lab_map.shape}."
    assert neighborhood_width % 2 == 1,\
        "Neighborhood width should be an odd number."
    assert neighborhood_width >= 3,\
        "Neighborhood width should be at least 3."
    # Optionally take in numpy array, convert to torch tensor
    if isinstance(lab_map, np.ndarray):
        lab_map = torch.from_numpy(lab_map)
        return_numpy = True
    else:
        return_numpy = False
    # Convert to long tensor
    lab_map = lab_map.long()
    count_array = torch.zeros_like(lab_map)
    # Define a 3x3 kernel of ones for the convolution
    kernel = torch.ones((1, 1, neighborhood_width, neighborhood_width), device=lab_map.device)
    for label in lab_map.unique():
        # Create a binary mask for the current label
        mask = (lab_map == label).float()
        # Unsqueeze masks to fit conv2d expected input (Batch Size, Channels, Height, Width)
        mask_unsqueezed = mask.unsqueeze(1)
        # Calculate the mask padding depending on the neighborhood width
        mask_padding = (neighborhood_width - 1) // 2
        # Apply padding
        padded_mask = F.pad(mask_unsqueezed, pad=(mask_padding, mask_padding, mask_padding, mask_padding), mode='constant')
        # Convolve the mask with the kernel to get the neighbor count using 2D convolution
        neighbor_count = F.conv2d(padded_mask, kernel, padding=0)  # No additional padding needed
        # Squeeze the result back to the original shape (B x H x W)
        neighbor_count_squeezed = neighbor_count.squeeze(1).long()
        # Update the count_array where the y_true matches the current label
        count_array[lab_map == label] = neighbor_count_squeezed[lab_map == label]
    # Subtract 1 because the center pixel is included in the 3x3 neighborhood count
    count_array -= 1
    # Return the count_array
    if return_numpy:
        return count_array.numpy()
    else:
        return count_array


# Get a distribution of per-pixel accuracy as a function of the size of the instance that it was 
# predicted in.
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_perpix_group_size(
    y_true: Union[Tensor, np.ndarray],
):
    # Optionally take in numpy array, convert to torch tensor
    if isinstance(y_true, Tensor):
        y_true = y_true.numpy()
        return_numpy = False 
    else:
        return_numpy = True
    # Create an empty tensor with the same shape as the input
    size_map = np.zeros_like(y_true)
    # Get unique labels in the segmentation map
    unique_labels = np.unique(y_true)
    for label_val in unique_labels:
        # Create a mask for the current label
        mask = (y_true == label_val)
        # Find connected components for the current mask
        labeled_array, num_features = label(mask)
        for i in range(1, num_features + 1):
            component_mask = (labeled_array == i)
            # Compute the size of the current component
            size = component_mask.sum().item()
            # Replace the pixels of the component with its size in the size_map tensor
            size_map[mask & component_mask] = size
    # Return the size_map 
    if return_numpy:
        return size_map
    else:
        return torch.from_numpy(size_map)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_uni_pixel_weights(
    lab_map: Union[Tensor, np.ndarray],
    uni_w_attributes: List[str],
    neighborhood_width: Optional[int] = None,
    num_neighbors_map: Optional[Union[Tensor, np.ndarray]] = None,
    ignore_index: Optional[int] = None
):
    """
    Get a map of pixel weights for each unique label in the label map. The weights are calculated
    based on the number of pixels with that label and the number of pixels with that label and
    a particular number of neighbors. The weights are normalized such that the sum of the weights
    for each label is 1.0.
    Args:
    - lab_map (Tensor): A 2D tensor of labels.
    - uni_w_attributes (List[str]): A list of unique label attributes to use for weighting.
    - neighborhood_width (int): The width of the neighborhood to use when counting neighbors.
    - num_neighbors_map (Tensor): A 2D tensor of the number of neighbors for each pixel in
        the label map.
    Returns:
    - Tensor: A 2D tensor of pixel weights for each pixel in the label map.
    """
    lab_map = lab_map.squeeze()
    assert len(lab_map.shape) == 2, "Pred map can only currently be (H, W)."
    # Optionally take in numpy array, convert to torch tensor
    if isinstance(lab_map, np.ndarray):
        lab_map = torch.from_numpy(lab_map)
        return_numpy = True
    else:
        return_numpy = False

    # Get a map where each pixel corresponds to the amount of pixels with that label who have 
    # that number of neighbors, and the total amount of pixels with that label.
    nn_balanced_weights_map = torch.zeros_like(lab_map).float()
    if ignore_index is not None:
        NUM_SAMPLES = lab_map[lab_map != ignore_index].numel()
    else:
        NUM_SAMPLES = lab_map.numel()

    # Get information about labels.
    unique_pred_labels = torch.unique(lab_map)
    if ignore_index is not None:
        unique_pred_labels = unique_pred_labels[unique_pred_labels != ignore_index]
    NUM_LAB = len(unique_pred_labels)

    # Choose how you uniformly condition. 
    neighbor_condition = "neighbors" in uni_w_attributes
    label_condition = "labels" in uni_w_attributes
    neighbor_and_label_condition = neighbor_condition and label_condition
    # If doing something with neighbors, get the neighbor map (if not passed in).
    if num_neighbors_map is None:
        num_neighbors_map = count_matching_neighbors(
            lab_map=lab_map, 
            neighborhood_width=neighborhood_width
            )
    if ignore_index is not None:
        num_neighbors_map[lab_map == ignore_index] = -1

    # If we are conditioning on both.
    if neighbor_and_label_condition:
        # Loop through each unique label and its number of neighbors.
        for label in unique_pred_labels:
            label_group = (lab_map == label)
            unique_label_nns = torch.unique(num_neighbors_map[label_group])
            NUM_NN = len(unique_label_nns)
            for nn in unique_label_nns:
                label_nn_group = (label_group) & (num_neighbors_map==nn)
                pix_weights = (1 / (NUM_NN * NUM_LAB)) * (NUM_SAMPLES / label_nn_group.sum().item())
                nn_balanced_weights_map[label_nn_group] = pix_weights

    # If we are conditioning ONLY on number of neighbors. 
    elif neighbor_condition:
        unique_nns = torch.unique(num_neighbors_map[num_neighbors_map != -1])
        # Loop through each number of neighbors.
        NUM_NN = len(unique_nns)
        for nn in unique_nns:
            nn_group = (num_neighbors_map == nn)
            pix_weights = (1 / NUM_NN) * (NUM_SAMPLES / nn_group.sum().item())
            nn_balanced_weights_map[nn_group] = pix_weights

    # If we are conditioning ONLY on amount of label.
    elif label_condition:
        # Loop through each label.
        for label in unique_pred_labels:
            label_group = (lab_map == label)
            nn_balanced_weights_map[label_group] = (NUM_SAMPLES / label_group.sum().item()) * (1 / NUM_LAB)

    else:
        raise ValueError(f"Uniform conditioning must be one of 'neighbors', 'labels', or both, got {uni_w_attributes} instead.")

    # Return the count_array
    if return_numpy:
        return nn_balanced_weights_map.numpy()
    else:
        return nn_balanced_weights_map
    

# Helpful for calculating edge accuracies.
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_edge_map(
    lab_map: Tensor,
    neighborhood_width: int = 3
    ) -> Tensor:
    # Neighbor map
    num_neighbor_map = count_matching_neighbors(
        lab_map=lab_map, 
        neighborhood_width=neighborhood_width
        )
    edge_map = (num_neighbor_map < (neighborhood_width**2 - 1))
    return edge_map