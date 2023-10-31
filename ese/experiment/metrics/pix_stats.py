# local imports 
from .utils import get_conf_region
# ionpy imports
from ionpy.metrics import pixel_accuracy
# misc. imports
import torch
import numpy as np
from pydantic import validate_arguments
from scipy.ndimage import distance_transform_edt, binary_erosion, label


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def bin_stats(
    num_bins: int,
    conf_bins: torch.Tensor,
    conf_bin_widths: torch.Tensor,
    conf_map: torch.Tensor,
    pred_map: torch.Tensor,
    label_map: torch.Tensor,
    ) -> dict:
    # Keep track of different things for each bin.
    cal_info = {
        "bin_confs": torch.zeros(num_bins),
        "bin_amounts": torch.zeros(num_bins),
        "bin_measures": torch.zeros(num_bins),
        "bin_cal_scores": torch.zeros(num_bins),
    }
    # Get the regions of the prediction corresponding to each bin of confidence.
    for bin_idx, conf_bin in enumerate(conf_bins):
        # Get the region of image corresponding to the confidence
        bin_conf_region = get_conf_region(bin_idx, conf_bin, conf_bin_widths, conf_map)
        # If there are some pixels in this confidence bin.
        if bin_conf_region.sum() > 0:
            # Calculate the average score for the regions in the bin.
            avg_bin_conf = conf_map[bin_conf_region].mean()
            avg_bin_measure = pixel_accuracy(pred_map[bin_conf_region], label_map[bin_conf_region])
            # Calculate the average calibration error for the regions in the bin.
            cal_info["bin_confs"][bin_idx] = avg_bin_conf 
            cal_info["bin_measures"][bin_idx] = avg_bin_measure 
            cal_info["bin_amounts"][bin_idx] = bin_conf_region.sum() 
            cal_info["bin_cal_scores"][bin_idx] = (avg_bin_conf - avg_bin_measure).abs()
    # Return the calibration information.
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def label_bin_stats(
    num_bins: int,
    conf_bins: torch.Tensor,
    conf_bin_widths: torch.Tensor,
    conf_map: torch.Tensor,
    pred_map: torch.Tensor,
    label_map: torch.Tensor,
    ) -> dict:
    # Keep track of different things for each bin.
    pred_labels = pred_map.unique().tolist()
    num_labels = len(pred_labels)
    cal_info = {
        "bin_confs": torch.zeros((num_labels, num_bins)),
        "bin_amounts": torch.zeros((num_labels, num_bins)),
        "bin_measures": torch.zeros((num_labels, num_bins)),
        "bin_cal_scores": torch.zeros((num_labels, num_bins))
    }
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
                avg_bin_measure = pixel_accuracy(pred_map[bin_conf_region], label_map[bin_conf_region])
                # Calculate the average calibration error for the regions in the bin.
                cal_info["bin_confs"][lab_idx, bin_idx] = avg_bin_conf 
                cal_info["bin_amounts"][lab_idx, bin_idx] = bin_conf_region.sum() 
                cal_info["bin_measures"][lab_idx, bin_idx] = avg_bin_measure 
                cal_info["bin_cal_scores"][lab_idx, bin_idx] = (avg_bin_conf - avg_bin_measure).abs()
    # Return the label-wise calibration information.
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def label_neighbors_bin_stats(
    num_bins: int,
    conf_bins: torch.Tensor,
    conf_bin_widths: torch.Tensor,
    conf_map: torch.Tensor,
    pred_map: torch.Tensor,
    label_map: torch.Tensor,
    neighborhood_width: int
    ) -> dict:
    # Keep track of different things for each bin.
    pred_labels = pred_map.unique().tolist()
    num_labels = len(pred_labels)
    num_neighbors = neighborhood_width**2
    cal_info = {
        "bin_cal_scores": torch.zeros((num_labels, num_neighbors, num_bins)),
        "bin_measures": torch.zeros((num_labels, num_neighbors, num_bins)),
        "bin_confs": torch.zeros((num_labels, num_neighbors, num_bins)),
        "bin_amounts": torch.zeros((num_labels, num_neighbors, num_bins))
    }
    # Get the regions of the prediction corresponding to each bin of confidence,
    # AND each prediction label.
    for bin_idx, conf_bin in enumerate(conf_bins):
        for lab_idx, p_label in enumerate(pred_labels):
            for num_neighb in range(0, num_neighbors):
                # Get the region of image corresponding to the confidence
                bin_conf_region = get_conf_region(
                    bin_idx=bin_idx, 
                    conf_bin=conf_bin, 
                    conf_bin_widths=conf_bin_widths, 
                    conf_map=conf_map,
                    pred_map=pred_map,
                    label=p_label,
                    num_neighbors=num_neighb
                    )
                # If there are some pixels in this confidence bin.
                if bin_conf_region.sum() > 0:
                    # Calculate the average score for the regions in the bin.
                    avg_bin_conf = conf_map[bin_conf_region].mean()
                    avg_bin_measure = pixel_accuracy(pred_map[bin_conf_region], label_map[bin_conf_region])
                    # Calculate the average calibration error for the regions in the bin.
                    cal_info["bin_confs"][lab_idx, num_neighb, bin_idx] = avg_bin_conf 
                    cal_info["bin_measures"][lab_idx, num_neighb, bin_idx] = avg_bin_measure 
                    cal_info["bin_amounts"][lab_idx, num_neighb, bin_idx] = bin_conf_region.sum() 
                    cal_info["bin_cal_scores"][lab_idx, num_neighb, bin_idx] = (avg_bin_conf - avg_bin_measure).abs()
    # Return the label-wise and neighborhood conditioned calibration information.
    return cal_info


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