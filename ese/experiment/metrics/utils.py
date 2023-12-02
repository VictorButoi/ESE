# misc imports
import torch
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import label
from typing import Optional, Union, List
from pydantic import validate_arguments
from scipy.ndimage import distance_transform_edt, binary_erosion, label


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def reduce_bin_errors(
    error_per_bin: torch.Tensor, 
    amounts_per_bin: torch.Tensor, 
    weighting: str = "proportional",
    bin_weights: Optional[torch.Tensor] = None
    ) -> float:
    if bin_weights is None:
        if amounts_per_bin.sum() == 0:
            return 0.0
        elif weighting == 'proportional':
            bin_weights = amounts_per_bin / (amounts_per_bin).sum()
        else:
            raise ValueError(f"Invalid bin weighting. Must be 'proportional', got '{weighting}' instead.")
    # Multiply by the weights and sum.
    assert 1.0 - torch.sum(bin_weights) < 1e-5, f"Weights should approx. sum to 1.0, got {bin_weights.sum()} instead."
    reduced_error = (error_per_bin * bin_weights).sum().item()
    assert 1 >= reduced_error >= 0, f"Reduced error should be between 0 and 1, got {reduced_error} instead."
    return reduced_error


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
    pred_map: Optional[torch.Tensor] = None,
    num_neighbors: Optional[int] = None,
    num_neighbors_map: Optional[torch.Tensor] = None,
    ignore_index: Optional[int] = None,
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
        bin_conf_region = torch.logical_and(bin_conf_region, (pred_map==label))
    # If we want to ignore a particular label, then we set it to 0.
    if ignore_index is not None:
        assert pred_map is not None, "If ignore_index is not None, then must supply pred map."
        bin_conf_region = torch.logical_and(bin_conf_region, (pred_map != ignore_index))
    # If we only want the pixels with this particular number of neighbords that match the label
    if num_neighbors is not None:
        assert num_neighbors_map is not None, "If num_neighbors is not None, then must supply num neighbors map."
        bin_conf_region = torch.logical_and(bin_conf_region, num_neighbors_map==num_neighbors)
    # The final region is the intersection of the conditions.
    return bin_conf_region


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
def get_label_region_sizes(label_map):
    # Get unique labels in the segmentation map
    unique_labels = np.unique(label_map)
    lab_reg_size_dict = {}
    for label_val in unique_labels:
        lab_reg_size_dict[label_val] = []
        # Create a mask for the current label
        mask = (label_map==label_val)
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
    conf_map: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = "cuda"
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
    
    return conf_bins.to(device), conf_bin_widths.to(device)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def find_bins(
    confidences, 
    bin_starts, 
    bin_widths
    ):
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
    confidences = confidences.squeeze()
    # Ensure that the bin_starts and bin_widths tensors have the same shape
    assert len(confidences.shape) == 2, "Confidences can only currently be (H, W)."
    assert bin_starts.shape == bin_widths.shape, "bin_starts and bin_widths should have the same shape."
    # Expand dimensions for broadcasting
    expanded_confidences = confidences.unsqueeze(-1)
    # Compare confidences against all bin ranges using broadcasting
    valid_bins = (expanded_confidences > bin_starts) & (expanded_confidences <= (bin_starts + bin_widths))
    # Get bin indices; if no valid bin is found for a confidence, the value will be -1
    bin_indices = torch.where(valid_bins, torch.arange(len(bin_starts)).cuda(), -torch.ones_like(bin_starts)).max(dim=-1).values
    return bin_indices


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def count_matching_neighbors(
    label_map: Union[torch.Tensor, np.ndarray],
    neighborhood_width: int = 3,
):
    label_map = label_map.squeeze()
    assert len(label_map.shape) == 2, "Label map can only currently be (H, W)."
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
    kernel = torch.ones((1, 1, neighborhood_width, neighborhood_width), device=device)
    # Reflective padding if reflect_boundaries is True
    # padding_mode = 'reflect' if reflect_boundaries else 'constant'
    padding_mode = 'constant'
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


# Get a distribution of per-pixel accuracy as a function of the size of the instance that it was 
# predicted in.
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_perpix_group_size(
    label_map: Union[torch.Tensor, np.ndarray],
):
    # Optionally take in numpy array, convert to torch tensor
    if isinstance(label_map, torch.Tensor):
        label_map = label_map.numpy()
        return_numpy = False 
    else:
        return_numpy = True
    # Create an empty tensor with the same shape as the input
    size_map = np.zeros_like(label_map)
    # Get unique labels in the segmentation map
    unique_labels = np.unique(label_map)
    for label_val in unique_labels:
        # Create a mask for the current label
        mask = (label_map == label_val)
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
    pred_map: Union[torch.Tensor, np.ndarray],
    uni_w_attributes: List[str],
    neighborhood_width: Optional[int] = None,
    num_neighbors_map: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ignore_index: Optional[int] = None
):
    """
    Get a map of pixel weights for each unique label in the label map. The weights are calculated
    based on the number of pixels with that label and the number of pixels with that label and
    a particular number of neighbors. The weights are normalized such that the sum of the weights
    for each label is 1.0.
    Args:
    - pred_map (torch.Tensor): A 2D tensor of labels.
    - uni_w_attributes (List[str]): A list of unique label attributes to use for weighting.
    - neighborhood_width (int): The width of the neighborhood to use when counting neighbors.
    - num_neighbors_map (torch.Tensor): A 2D tensor of the number of neighbors for each pixel in
        the label map.
    Returns:
    - torch.Tensor: A 2D tensor of pixel weights for each pixel in the label map.
    """
    pred_map = pred_map.squeeze()
    assert len(pred_map.shape) == 2, "Pred map can only currently be (H, W)."
    # Optionally take in numpy array, convert to torch tensor
    if isinstance(pred_map, np.ndarray):
        pred_map = torch.from_numpy(pred_map)
        return_numpy = True
    else:
        return_numpy = False

    # Get a map where each pixel corresponds to the amount of pixels with that label who have 
    # that number of neighbors, and the total amount of pixels with that label.
    nn_balanced_weights_map = torch.zeros_like(pred_map).float()
    if ignore_index is not None:
        NUM_SAMPLES = pred_map[pred_map != ignore_index].numel()
    else:
        NUM_SAMPLES = pred_map.numel()

    # Get information about labels.
    unique_pred_labels = torch.unique(pred_map)
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
            label_map=pred_map, 
            neighborhood_width=neighborhood_width
            )
    if ignore_index is not None:
        num_neighbors_map[pred_map == ignore_index] = -1

    # If we are conditioning on both.
    if neighbor_and_label_condition:
        # Loop through each unique label and its number of neighbors.
        for label in unique_pred_labels:
            label_group = (pred_map == label)
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
            label_group = (pred_map == label)
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
    label_map: torch.Tensor
    ) -> torch.Tensor:
    # Neighbor map
    num_neighbor_map = count_matching_neighbors(
        label_map, 
        neighborhood_width=3
        )
    edge_map = (num_neighbor_map < 8)
    return edge_map