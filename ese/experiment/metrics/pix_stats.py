# local imports 
from .utils import (
    get_bins,
    get_conf_region, 
    get_uni_pixel_weights,
    count_matching_neighbors
)
# misc. imports
import torch
from typing import Optional, List, Tuple
from pydantic import validate_arguments


def bin_stats_init(y_pred,
                   y_true,
                   num_bins,
                   conf_interval):
    y_pred = y_pred.squeeze()
    y_true = y_true.squeeze()
    assert len(y_pred.shape) == 3 and len(y_true.shape) == 2,\
        f"y_pred and y_true must be 3D and 2D tensors, respectively. Got {y_pred.shape} and {y_true.shape}."
    y_hard = y_pred.argmax(dim=0)
    # Create the confidence bins.    
    conf_bins, conf_bin_widths = get_bins(
        num_bins=num_bins, 
        start=conf_interval[0], 
        end=conf_interval[1]
        )
    return y_pred, y_hard, y_true, conf_bins, conf_bin_widths


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def bin_stats(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_bins: int,
    conf_interval: Tuple[float, float],
    square_diff: bool,
    neighborhood_width: Optional[int] = None,
    uni_w_attributes: Optional[List[str]] = None,
    label: Optional[int] = None,
    ignore_index: Optional[int] = None
    ) -> dict:
    # Init some things.
    y_pred, y_hard, y_true, conf_bins, conf_bin_widths = bin_stats_init(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval
        )
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
            y_hard=y_hard, 
            uni_w_attributes=uni_w_attributes,
            neighborhood_width=neighborhood_width,
            ignore_index=ignore_index
            )
    else:
        pix_weights = None
    # Get the pixelwise accuracy.
    pixelwise_accuracy = (y_hard == y_true).float()
    # Get the regions of the prediction corresponding to each bin of confidence.
    for bin_idx, conf_bin in enumerate(conf_bins):
        # Get the region of image corresponding to the confidence
        bin_conf_region = get_conf_region(
            bin_idx=bin_idx, 
            conf_bin=conf_bin, 
            y_pred=y_pred,
            conf_bin_widths=conf_bin_widths, 
            y_hard=y_hard,
            label=label, # Focus on just one label (Optionally).
            ignore_index=ignore_index
            )
        # If there are some pixels in this confidence bin.
        if bin_conf_region.sum() > 0:
            # Calculate the average score for the regions in the bin.
            if pix_weights is None:
                avg_bin_confidence = y_pred[bin_conf_region].mean()
                avg_bin_accuracy = pixelwise_accuracy[bin_conf_region].mean()
                bin_num_samples = bin_conf_region.sum() 
            else:
                bin_num_samples = pix_weights[bin_conf_region].sum()
                avg_bin_confidence = (pix_weights[bin_conf_region] * y_pred[bin_conf_region]).sum() / bin_num_samples
                avg_bin_accuracy = (pix_weights[bin_conf_region] * pixelwise_accuracy[bin_conf_region]).sum() / bin_num_samples
            # Calculate the average calibration error for the regions in the bin.
            cal_info["bin_confs"][bin_idx] = avg_bin_confidence
            cal_info["bin_accs"][bin_idx] = avg_bin_accuracy
            cal_info["bin_amounts"][bin_idx] = bin_num_samples
            # Calculate the calibration error.
            if square_diff:
                cal_info["bin_cal_errors"][bin_idx] = (avg_bin_confidence - avg_bin_accuracy).square()
            else:
                cal_info["bin_cal_errors"][bin_idx] = (avg_bin_confidence - avg_bin_accuracy).abs()
    # Return the calibration information.
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def label_bin_stats(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_bins: int,
    conf_interval: Tuple[float, float],
    square_diff: bool,
    neighborhood_width: Optional[int] = None,
    uni_w_attributes: Optional[List[str]] = None,
    label: Optional[int] = None,
    ignore_index: Optional[int] = None
    ) -> dict:
    # Init some things.
    y_pred, y_hard, y_true, conf_bins, conf_bin_widths = bin_stats_init(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval
        )
    # Keep track of different things for each bin.
    if label is None:
        pred_labels = y_hard.unique().tolist()
    else:
        pred_labels = [label]
    # Remove the ignore index if it is in the list of labels.
    if ignore_index is not None and ignore_index in pred_labels:
        pred_labels.remove(ignore_index)
    num_labels = len(pred_labels)
    # Setup the cal info tracker.
    cal_info = {
        "bin_confs": torch.zeros((num_labels, num_bins)),
        "bin_amounts": torch.zeros((num_labels, num_bins)),
        "bin_accs": torch.zeros((num_labels, num_bins)),
        "bin_cal_errors": torch.zeros((num_labels, num_bins))
    }
    # Get the pixel-weights if we are using them.
    if uni_w_attributes is not None:
        pix_weights = get_uni_pixel_weights(
            y_hard, 
            uni_w_attributes=uni_w_attributes,
            neighborhood_width=neighborhood_width,
            ignore_index=ignore_index
            )
    else:
        pix_weights = None
    # Get the regions of the prediction corresponding to each bin of confidence,
    pixelwise_accuracy = (y_hard == y_true).float()
    # AND each prediction label.
    for bin_idx, conf_bin in enumerate(conf_bins):
        for lab_idx, p_label in enumerate(pred_labels):
            # Get the region of image corresponding to the confidence
            bin_conf_region = get_conf_region(
                bin_idx=bin_idx, 
                conf_bin=conf_bin, 
                y_pred=y_pred,
                conf_bin_widths=conf_bin_widths, 
                y_hard=y_hard,
                label=p_label,
                ignore_index=ignore_index
                )
            # If there are some pixels in this confidence bin.
            if bin_conf_region.sum() > 0:
                # Calculate the average score for the regions in the bin.
                if pix_weights is None:
                    avg_bin_confidence = y_pred[bin_conf_region].mean()
                    avg_bin_accuracy = pixelwise_accuracy[bin_conf_region].mean()
                    bin_num_samples = bin_conf_region.sum() 
                else:
                    bin_num_samples = pix_weights[bin_conf_region].sum()
                    avg_bin_confidence = (pix_weights[bin_conf_region] * y_pred[bin_conf_region]).sum() / bin_num_samples
                    avg_bin_accuracy = (pix_weights[bin_conf_region] * pixelwise_accuracy[bin_conf_region]).sum() / bin_num_samples
                # Calculate the average calibration error for the regions in the bin.
                cal_info["bin_confs"][lab_idx, bin_idx] = avg_bin_confidence
                cal_info["bin_accs"][lab_idx, bin_idx] = avg_bin_accuracy
                cal_info["bin_amounts"][lab_idx, bin_idx] = bin_num_samples
                # Calculate the calibration error.
                if square_diff:
                    cal_info["bin_cal_errors"][lab_idx, bin_idx] = (avg_bin_confidence - avg_bin_accuracy).square()
                else:
                    cal_info["bin_cal_errors"][lab_idx, bin_idx] = (avg_bin_confidence - avg_bin_accuracy).abs()
    # Return the label-wise calibration information.
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def neighbors_bin_stats(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_bins: int,
    conf_interval: Tuple[float, float],
    square_diff: bool,
    neighborhood_width: int,
    uni_w_attributes: Optional[List[str]] = None,
    label: Optional[int] = None,
    ignore_index: Optional[int] = None
    ) -> dict:
    # Init some things.
    y_pred, y_hard, y_true, conf_bins, conf_bin_widths = bin_stats_init(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval
        )
    # Set the cal info tracker.
    num_neighbors = neighborhood_width**2
    cal_info = {
        "bin_cal_errors": torch.zeros((num_neighbors, num_bins)),
        "bin_accs": torch.zeros((num_neighbors, num_bins)),
        "bin_confs": torch.zeros((num_neighbors, num_bins)),
        "bin_amounts": torch.zeros((num_neighbors, num_bins))
    }
    # Get the pixel-weights if we are using them.
    if uni_w_attributes is not None:
        pix_weights = get_uni_pixel_weights(
            y_hard, 
            uni_w_attributes=uni_w_attributes,
            neighborhood_width=neighborhood_width,
            ignore_index=ignore_index
            )
    else:
        pix_weights = None
    # Get a map of which pixels match their neighbors and how often, and pixel-wise accuracy.
    matching_neighbors_map = count_matching_neighbors(
        y_hard, 
        neighborhood_width=neighborhood_width
    )
    pixelwise_accuracy = (y_hard == y_true).float()
    # Get the regions of the prediction corresponding to each bin of confidence,
    # AND each prediction label.
    for num_neighb in range(0, num_neighbors):
        for bin_idx, conf_bin in enumerate(conf_bins):
            # Get the region of image corresponding to the confidence
            bin_conf_region = get_conf_region(
                y_pred=y_pred,
                y_hard=y_hard,
                bin_idx=bin_idx, 
                conf_bin=conf_bin, 
                conf_bin_widths=conf_bin_widths, 
                num_neighbors=num_neighb,
                num_neighbors_map=matching_neighbors_map,
                label=label, # Focus on just one label (Optionally).
                ignore_index=ignore_index
                )
            # If there are some pixels in this confidence bin.
            if bin_conf_region.sum() > 0:
                # Calculate the average score for the regions in the bin.
                if pix_weights is None:
                    avg_bin_confidence = y_pred[bin_conf_region].mean()
                    avg_bin_accuracy = pixelwise_accuracy[bin_conf_region].mean()
                    bin_num_samples = bin_conf_region.sum() 
                else:
                    bin_num_samples = pix_weights[bin_conf_region].sum()
                    avg_bin_confidence = (pix_weights[bin_conf_region] * y_pred[bin_conf_region]).sum() / bin_num_samples
                    avg_bin_accuracy = (pix_weights[bin_conf_region] * pixelwise_accuracy[bin_conf_region]).sum() / bin_num_samples
                # Calculate the average calibration error for the regions in the bin.
                cal_info["bin_confs"][num_neighb, bin_idx] = avg_bin_confidence
                cal_info["bin_accs"][num_neighb, bin_idx] = avg_bin_accuracy
                cal_info["bin_amounts"][num_neighb, bin_idx] = bin_num_samples

                # Calculate the calibration error.
                if square_diff:
                    cal_info["bin_cal_errors"][num_neighb, bin_idx] = (avg_bin_confidence - avg_bin_accuracy).square()
                else:
                    cal_info["bin_cal_errors"][num_neighb, bin_idx] = (avg_bin_confidence - avg_bin_accuracy).abs()
    # Return the label-wise and neighborhood conditioned calibration information.
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def label_neighbors_bin_stats(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_bins: int,
    conf_interval: Tuple[float, float],
    square_diff: bool,
    neighborhood_width: int,
    uni_w_attributes: Optional[List[str]] = None,
    label: Optional[int] = None,
    ignore_index: Optional[int] = None
    ) -> dict:
    # Init some things.
    y_pred, y_hard, y_true, conf_bins, conf_bin_widths = bin_stats_init(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval
        )
    # Keep track of different things for each bin.
    if label is None:
        pred_labels = y_hard.unique().tolist()
    else:
        pred_labels = [label]
    # Remove the ignore index if it is in the list of labels.
    if ignore_index is not None and ignore_index in pred_labels:
        pred_labels.remove(ignore_index)
    num_labels = len(pred_labels)
    # Set the cal info tracker.
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
            y_hard, 
            uni_w_attributes=uni_w_attributes,
            neighborhood_width=neighborhood_width,
            ignore_index=ignore_index
            )
    else:
        pix_weights = None
    # Get a map of which pixels match their neighbors and how often, and pixel-wise accuracy.
    matching_neighbors_map = count_matching_neighbors(
        y_hard, 
        neighborhood_width=neighborhood_width
    )
    pixelwise_accuracy = (y_hard == y_true).float()
    # Get the regions of the prediction corresponding to each bin of confidence,
    # AND each prediction label.
    for lab_idx, p_label in enumerate(pred_labels):
        for num_neighb in range(0, num_neighbors):
            for bin_idx, conf_bin in enumerate(conf_bins):
                # Get the region of image corresponding to the confidence
                bin_conf_region = get_conf_region(
                    bin_idx=bin_idx, 
                    conf_bin=conf_bin, 
                    y_pred=y_pred,
                    conf_bin_widths=conf_bin_widths, 
                    y_hard=y_hard,
                    num_neighbors=num_neighb,
                    num_neighbors_map=matching_neighbors_map,
                    label=p_label,
                    ignore_index=ignore_index
                    )
                # If there are some pixels in this confidence bin.
                if bin_conf_region.sum() > 0:
                    # Calculate the average score for the regions in the bin.
                    if pix_weights is None:
                        avg_bin_confidence = y_pred[bin_conf_region].mean()
                        avg_bin_accuracy = pixelwise_accuracy[bin_conf_region].mean()
                        bin_num_samples = bin_conf_region.sum() 
                    else:
                        bin_num_samples = pix_weights[bin_conf_region].sum()
                        avg_bin_confidence = (pix_weights[bin_conf_region] * y_pred[bin_conf_region]).sum() / bin_num_samples
                        avg_bin_accuracy = (pix_weights[bin_conf_region] * pixelwise_accuracy[bin_conf_region]).sum() / bin_num_samples
                    # Calculate the average calibration error for the regions in the bin.
                    cal_info["bin_confs"][lab_idx, num_neighb, bin_idx] = avg_bin_confidence
                    cal_info["bin_accs"][lab_idx, num_neighb, bin_idx] = avg_bin_accuracy
                    cal_info["bin_amounts"][lab_idx, num_neighb, bin_idx] = bin_num_samples
                    # Calculate the calibration error.
                    if square_diff:
                        cal_info["bin_cal_errors"][lab_idx, num_neighb, bin_idx] = (avg_bin_confidence - avg_bin_accuracy).square()
                    else:
                        cal_info["bin_cal_errors"][lab_idx, num_neighb, bin_idx] = (avg_bin_confidence - avg_bin_accuracy).abs()
    # Return the label-wise and neighborhood conditioned calibration information.
    return cal_info