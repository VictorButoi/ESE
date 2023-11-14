# local imports 
from .utils import get_conf_region, count_matching_neighbors, get_uni_pixel_weights
# misc. imports
import torch
from typing import Optional, List
from pydantic import validate_arguments


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def bin_stats(
    num_bins: int,
    conf_bins: torch.Tensor,
    conf_bin_widths: torch.Tensor,
    conf_map: torch.Tensor,
    pred_map: torch.Tensor,
    label_map: torch.Tensor,
    neighborhood_width: Optional[int] = None,
    uni_w_attributes: Optional[List[str]] = None
    ) -> dict:
    # Keep track of different things for each bin.
    cal_info = {
        "bin_confs": torch.zeros(num_bins),
        "bin_amounts": torch.zeros(num_bins),
        "bin_accs": torch.zeros(num_bins),
        "bin_cal_errors": torch.zeros(num_bins),
    }
    # Get the pixel-weights if we are using them.
    if uni_w_attributes is not None:
        pix_weights = get_uni_pixel_weights(
            pred_map, 
            uni_w_attributes=uni_w_attributes,
            neighborhood_width=neighborhood_width,
            reflect_boundaries=True
            )
    else:
        pix_weights = None
    # Get the pixelwise accuracy.
    pixelwise_accuracy = (pred_map == label_map).float()
    # Get the regions of the prediction corresponding to each bin of confidence.
    for bin_idx, conf_bin in enumerate(conf_bins):
        # Get the region of image corresponding to the confidence
        bin_conf_region = get_conf_region(
            bin_idx=bin_idx, 
            conf_bin=conf_bin, 
            conf_bin_widths=conf_bin_widths, 
            conf_map=conf_map
            )
        # If there are some pixels in this confidence bin.
        if bin_conf_region.sum() > 0:
            # Calculate the average score for the regions in the bin.
            if pix_weights is None:
                avg_bin_confidence = conf_map[bin_conf_region].mean()
                avg_bin_accuracy = pixelwise_accuracy[bin_conf_region].mean()
                bin_num_samples = bin_conf_region.sum() 
            else:
                bin_num_samples = pix_weights[bin_conf_region].sum()
                avg_bin_confidence = (pix_weights[bin_conf_region] * conf_map[bin_conf_region]).sum() / bin_num_samples
                avg_bin_accuracy = (pix_weights[bin_conf_region] * pixelwise_accuracy[bin_conf_region]).sum() / bin_num_samples
            # Calculate the average calibration error for the regions in the bin.
            cal_info["bin_confs"][bin_idx] = avg_bin_confidence
            cal_info["bin_accs"][bin_idx] = avg_bin_accuracy
            cal_info["bin_amounts"][bin_idx] = bin_num_samples
            cal_info["bin_cal_errors"][bin_idx] = (avg_bin_confidence - avg_bin_accuracy).abs()

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
    neighborhood_width: Optional[int] = None,
    uni_w_attributes: Optional[List[str]] = None
    ) -> dict:
    # Keep track of different things for each bin.
    pred_labels = pred_map.unique().tolist()
    num_labels = len(pred_labels)
    cal_info = {
        "bin_confs": torch.zeros((num_labels, num_bins)),
        "bin_amounts": torch.zeros((num_labels, num_bins)),
        "bin_accs": torch.zeros((num_labels, num_bins)),
        "bin_cal_errors": torch.zeros((num_labels, num_bins))
    }
    # Get the pixel-weights if we are using them.
    if uni_w_attributes is not None:
        pix_weights = get_uni_pixel_weights(
            pred_map, 
            uni_w_attributes=uni_w_attributes,
            neighborhood_width=neighborhood_width,
            reflect_boundaries=True
            )
    else:
        pix_weights = None
    # Get the regions of the prediction corresponding to each bin of confidence,
    pixelwise_accuracy = (pred_map == label_map).float()
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
                if pix_weights is None:
                    avg_bin_confidence = conf_map[bin_conf_region].mean()
                    avg_bin_accuracy = pixelwise_accuracy[bin_conf_region].mean()
                    bin_num_samples = bin_conf_region.sum() 
                else:
                    bin_num_samples = pix_weights[bin_conf_region].sum()
                    avg_bin_confidence = (pix_weights[bin_conf_region] * conf_map[bin_conf_region]).sum() / bin_num_samples
                    avg_bin_accuracy = (pix_weights[bin_conf_region] * pixelwise_accuracy[bin_conf_region]).sum() / bin_num_samples
                # Calculate the average calibration error for the regions in the bin.
                cal_info["bin_amounts"][lab_idx, bin_idx] = bin_num_samples
                cal_info["bin_confs"][lab_idx, bin_idx] = avg_bin_confidence
                cal_info["bin_accs"][lab_idx, bin_idx] = avg_bin_accuracy
                cal_info["bin_cal_errors"][lab_idx, bin_idx] = (avg_bin_confidence - avg_bin_accuracy).abs()
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
    neighborhood_width: int,
    uni_w_attributes: Optional[List[str]] = None
    ) -> dict:
    # Keep track of different things for each bin.
    pred_labels = pred_map.unique().tolist()
    num_labels = len(pred_labels)
    num_neighbors = neighborhood_width**2
    cal_info = {
        "bin_cal_errors": torch.zeros((num_labels, num_neighbors, num_bins)),
        "bin_accs": torch.zeros((num_labels, num_neighbors, num_bins)),
        "bin_confs": torch.zeros((num_labels, num_neighbors, num_bins)),
        "bin_amounts": torch.zeros((num_labels, num_neighbors, num_bins))
    }
    # Get the pixel-weights if we are using them.
    if uni_w_attributes is not None:
        pix_weights = get_uni_pixel_weights(
            pred_map, 
            uni_w_attributes=uni_w_attributes,
            neighborhood_width=neighborhood_width,
            reflect_boundaries=True
            )
    else:
        pix_weights = None
    # Get a map of which pixels match their neighbors and how often, and pixel-wise accuracy.
    matching_neighbors_map = count_matching_neighbors(pred_map, reflect_boundaries=False)
    pixelwise_accuracy = (pred_map == label_map).float()
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
                    label=p_label,
                    pred_map=pred_map,
                    num_neighbors=num_neighb,
                    num_neighbors_map=matching_neighbors_map,
                    )
                # If there are some pixels in this confidence bin.
                if bin_conf_region.sum() > 0:
                    # Calculate the average score for the regions in the bin.
                    if pix_weights is None:
                        avg_bin_confidence = conf_map[bin_conf_region].mean()
                        avg_bin_accuracy = pixelwise_accuracy[bin_conf_region].mean()
                        bin_num_samples = bin_conf_region.sum() 
                    else:
                        bin_num_samples = pix_weights[bin_conf_region].sum()
                        avg_bin_confidence = (pix_weights[bin_conf_region] * conf_map[bin_conf_region]).sum() / bin_num_samples
                        avg_bin_accuracy = (pix_weights[bin_conf_region] * pixelwise_accuracy[bin_conf_region]).sum() / bin_num_samples
                    # Calculate the average calibration error for the regions in the bin.
                    cal_info["bin_confs"][lab_idx, num_neighb, bin_idx] = avg_bin_confidence
                    cal_info["bin_accs"][lab_idx, num_neighb, bin_idx] = avg_bin_accuracy
                    cal_info["bin_amounts"][lab_idx, num_neighb, bin_idx] = bin_num_samples
                    cal_info["bin_cal_errors"][lab_idx, num_neighb, bin_idx] = (avg_bin_confidence - avg_bin_accuracy).abs()
    # Return the label-wise and neighborhood conditioned calibration information.
    return cal_info