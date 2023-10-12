import torch
import numpy as np
from torch import Tensor
from pydantic import validate_arguments
from scipy.ndimage import distance_transform_edt, binary_erosion


# Get a distribution of per-pixel accuracy as a function of distance to a boundary for a 2D image.
# and this is done without bins.
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def accuracy_vs_boundary_dist(
    y_pred: Tensor,
    y_true: Tensor,
) -> Tensor:
    # Get the per-pixel accuracy
    perpix_accuracies = (y_pred == y_true).float()
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
    return torch.cat([perpix_accuracies, distance_to_boundaries]) 


# Get a distribution of per-pixel accuracy as a function of the size of the instance that it was 
# predicted in.
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def accuracy_vs_instancesize(
    y_pred: Tensor,
    y_true: Tensor,
) -> Tensor:
    perpix_accuracies = (y_pred == y_true).float()
    # Create an empty tensor with the same shape as the input
    size_map = torch.zeros_like(y_pred, dtype=torch.int)
    # Get unique labels in the segmentation map
    unique_labels = torch.unique(y_pred)
    for label in unique_labels:
        # Create a mask for the current label
        mask = (y_pred == label)
        # Find connected components for the current label
        connected_components, num_components = torch.connected_components(mask)
        for i in range(num_components):
            component_mask = connected_components == i
            # Compute the size of the current component
            size = component_mask.sum().item()
            # Replace the pixels of the component with its size
            size_map[component_mask] = size
    return torch.cat([perpix_accuracies, size_map]) 


# Get a distribution of per-pixel accuracy as a function of the label
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def accuracy_vs_label(
    y_pred: Tensor,
    y_true: Tensor,
) -> Tensor:
    perpix_accuracies = (y_pred == y_true).float()
    return torch.cat([perpix_accuracies, y_pred])