import torch
import numpy as np
from torch import Tensor
from pydantic import validate_arguments
from scipy.ndimage import distance_transform_edt, binary_erosion, label


# Get a distribution of per-pixel accuracy as a function of distance to a boundary for a 2D image.
# and this is done without bins.
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def accuracy_vs_boundary_dist(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    # Get the per-pixel accuracy
    perpix_accuracies = (y_pred == y_true).astype(np.float32)
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
    return np.concatenate([perpix_accuracies, distance_to_boundaries]) 


# Get a distribution of per-pixel accuracy as a function of the size of the instance that it was 
# predicted in.
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def accuracy_vs_instancesize(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    perpix_accuracies = (y_pred == y_true).astype(np.float32)
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
    return np.concatenate([perpix_accuracies, size_map])


# Get a distribution of per-pixel accuracy as a function of the label
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def accuracy_vs_label(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    perpix_accuracies = (y_pred == y_true).astype(np.float32)
    return np.concatenate([perpix_accuracies, y_pred])