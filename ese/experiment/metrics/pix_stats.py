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


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def calc_bin_info(
    y_pred: torch.Tensor,
    bin_conf_region: torch.Tensor,
    pixelwise_accuracy: torch.Tensor,
    square_diff: bool,
    pix_weights: Optional[torch.Tensor] = None
):
    if pix_weights is None:
        avg_bin_confidence = y_pred[bin_conf_region].mean()
        avg_bin_accuracy = pixelwise_accuracy[bin_conf_region].mean()
        bin_num_samples = bin_conf_region.sum() 
    else:
        bin_num_samples = pix_weights[bin_conf_region].sum()
        avg_bin_confidence = (pix_weights[bin_conf_region] * y_pred[bin_conf_region]).sum() / bin_num_samples
        avg_bin_accuracy = (pix_weights[bin_conf_region] * pixelwise_accuracy[bin_conf_region]).sum() / bin_num_samples
    # Calculate the calibration error.
    if square_diff:
        cal_error = (avg_bin_confidence - avg_bin_accuracy).square()
    else:
        cal_error = (avg_bin_confidence - avg_bin_accuracy).abs()
    return {
        "avg_conf": avg_bin_confidence,
        "avg_acc": avg_bin_accuracy,
        "cal_error": cal_error,
        "num_samples": bin_num_samples
    }


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def bin_stats_init(
    y_pred,
    y_true,
    num_bins,
    conf_interval,
    neighborhood_width,
    uni_w_attributes,
    ignore_index
):
    y_pred = y_pred.squeeze()
    y_true = y_true.squeeze()
    assert len(y_pred.shape) == 3 and len(y_true.shape) == 2,\
        f"y_pred and y_true must be 3D and 2D tensors, respectively. Got {y_pred.shape} and {y_true.shape}."
    # Keep track of everything in an obj dict
    obj_dict = {}
    
    # Get the hard predictions and the max confidences.
    y_hard = y_pred.argmax(dim=0)
    y_pred = y_pred.max(dim=0).values
    obj_dict["y_pred"] = y_pred
    obj_dict["y_hard"] = y_hard

    # Create the confidence bins.    
    conf_bins, conf_bin_widths = get_bins(
        num_bins=num_bins, 
        start=conf_interval[0], 
        end=conf_interval[1]
        )
    obj_dict["conf_bins"] = conf_bins
    obj_dict["conf_bin_widths"] = conf_bin_widths

    # Get the pixelwise accuracy.
    obj_dict["pixelwise_accuracy"]= (y_hard == y_true).float()

    # Keep track of different things for each bin.
    pred_labels = y_hard.unique().tolist()
    if ignore_index is not None and ignore_index in pred_labels:
        pred_labels.remove(ignore_index)
    obj_dict["pred_labels"] = pred_labels

    # Get a map of which pixels match their neighbors and how often, and pixel-wise accuracy.
    if neighborhood_width is not None:
        matching_neighbors_map = count_matching_neighbors(
            y_hard, 
            neighborhood_width=neighborhood_width
        )
        obj_dict["matching_neighbors_map"] = matching_neighbors_map

    # Get the pixel-weights if we are using them.
    if uni_w_attributes is not None:
        obj_dict["pix_weights"] = get_uni_pixel_weights(
            y_hard=y_hard, 
            uni_w_attributes=uni_w_attributes,
            neighborhood_width=neighborhood_width,
            ignore_index=ignore_index
            )
    else:
        obj_dict["pix_weights"] = None

    return obj_dict


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def bin_stats(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_bins: int,
    conf_interval: Tuple[float, float],
    square_diff: bool,
    neighborhood_width: Optional[int] = None,
    uni_w_attributes: Optional[List[str]] = None,
    ignore_index: Optional[int] = None
    ) -> dict:
    # Init some things.
    obj_dict = bin_stats_init(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        neighborhood_width=neighborhood_width,
        uni_w_attributes=uni_w_attributes,
        ignore_index=ignore_index
        )
    # Keep track of different things for each bin.
    cal_info = {
        "bin_confs": torch.zeros(num_bins),
        "bin_amounts": torch.zeros(num_bins),
        "bin_accs": torch.zeros(num_bins),
        "bin_cal_errors": torch.zeros(num_bins),
    }
    # Get the regions of the prediction corresponding to each bin of confidence.
    for bin_idx, conf_bin in enumerate(obj_dict["conf_bins"]):
        # Get the region of image corresponding to the confidence
        bin_conf_region = get_conf_region(
            y_pred=obj_dict["y_pred"],
            y_hard=obj_dict["y_hard"],
            bin_idx=bin_idx, 
            conf_bin=conf_bin, 
            conf_bin_widths=obj_dict["conf_bin_widths"], 
            ignore_index=ignore_index
            )
        # If there are some pixels in this confidence bin.
        if bin_conf_region.sum() > 0:
            # Calculate the average score for the regions in the bin.
            bi = calc_bin_info(
                y_pred=obj_dict["y_pred"],
                bin_conf_region=bin_conf_region,
                square_diff=square_diff,
                pixelwise_accuracy=obj_dict["pixelwise_accuracy"],
                pix_weights=obj_dict["pix_weights"]
            )
            # Calculate the average calibration error for the regions in the bin.
            cal_info["bin_confs"][bin_idx] = bi["avg_conf"] 
            cal_info["bin_accs"][bin_idx] = bi["avg_acc"] 
            cal_info["bin_amounts"][bin_idx] = bi["num_samples"] 
            cal_info["bin_cal_errors"][bin_idx] = bi["cal_error"]
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
    ignore_index: Optional[int] = None
    ) -> dict:
    # Init some things.
    obj_dict = bin_stats_init(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        neighborhood_width=neighborhood_width,
        uni_w_attributes=uni_w_attributes,
        ignore_index=ignore_index
        )
    num_labels = len(obj_dict["pred_labels"])
    # Setup the cal info tracker.
    cal_info = {
        "bin_confs": torch.zeros((num_labels, num_bins)),
        "bin_amounts": torch.zeros((num_labels, num_bins)),
        "bin_accs": torch.zeros((num_labels, num_bins)),
        "bin_cal_errors": torch.zeros((num_labels, num_bins))
    }
    for lab_idx, p_label in enumerate(obj_dict["pred_labels"]):
        for bin_idx, conf_bin in enumerate(obj_dict["conf_bins"]):
            # Get the region of image corresponding to the confidence
            bin_conf_region = get_conf_region(
                y_pred=obj_dict["y_pred"],
                y_hard=obj_dict["y_hard"],
                bin_idx=bin_idx, 
                conf_bin=conf_bin, 
                conf_bin_widths=obj_dict["conf_bin_widths"], 
                label=p_label,
                ignore_index=ignore_index
                )
            # If there are some pixels in this confidence bin.
            if bin_conf_region.sum() > 0:
                # Calculate the average score for the regions in the bin.
                bi = calc_bin_info(
                    y_pred=obj_dict["y_pred"],
                    bin_conf_region=bin_conf_region,
                    square_diff=square_diff,
                    pixelwise_accuracy=obj_dict["pixelwise_accuracy"],
                    pix_weights=obj_dict["pix_weights"]
                )
                # Calculate the average calibration error for the regions in the bin.
                cal_info["bin_confs"][lab_idx, bin_idx] = bi["avg_conf"] 
                cal_info["bin_accs"][lab_idx, bin_idx] = bi["avg_acc"] 
                cal_info["bin_amounts"][lab_idx, bin_idx] = bi["num_samples"] 
                cal_info["bin_cal_errors"][lab_idx, bin_idx] = bi["cal_error"] 
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
    ignore_index: Optional[int] = None
    ) -> dict:
    # Init some things.
    obj_dict = bin_stats_init(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        neighborhood_width=neighborhood_width,
        uni_w_attributes=uni_w_attributes,
        ignore_index=ignore_index
        )
    # Set the cal info tracker.
    num_neighbors = neighborhood_width**2
    cal_info = {
        "bin_cal_errors": torch.zeros((num_neighbors, num_bins)),
        "bin_accs": torch.zeros((num_neighbors, num_bins)),
        "bin_confs": torch.zeros((num_neighbors, num_bins)),
        "bin_amounts": torch.zeros((num_neighbors, num_bins))
    }
    for nn_idx in range(num_neighbors):
        for bin_idx, conf_bin in enumerate(obj_dict["conf_bins"]):
            # Get the region of image corresponding to the confidence
            bin_conf_region = get_conf_region(
                y_pred=obj_dict["y_pred"],
                y_hard=obj_dict["y_hard"],
                bin_idx=bin_idx, 
                conf_bin=conf_bin, 
                conf_bin_widths=obj_dict["conf_bin_widths"], 
                num_neighbors=nn_idx,
                num_neighbors_map=obj_dict["matching_neighbors_map"],
                ignore_index=ignore_index
                )
            # If there are some pixels in this confidence bin.
            if bin_conf_region.sum() > 0:
                # Calculate the average score for the regions in the bin.
                bi = calc_bin_info(
                    y_pred=obj_dict["y_pred"],
                    bin_conf_region=bin_conf_region,
                    square_diff=square_diff,
                    pixelwise_accuracy=obj_dict["pixelwise_accuracy"],
                    pix_weights=obj_dict["pix_weights"]
                )
                # Calculate the average calibration error for the regions in the bin.
                cal_info["bin_confs"][nn_idx, bin_idx] = bi["avg_conf"] 
                cal_info["bin_accs"][nn_idx, bin_idx] = bi["avg_acc"] 
                cal_info["bin_amounts"][nn_idx, bin_idx] = bi["num_samples"]
                cal_info["bin_amounts"][nn_idx, bin_idx] = bi["cal_error"] 
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
    ignore_index: Optional[int] = None
    ) -> dict:
    # Init some things.
    obj_dict = bin_stats_init(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        neighborhood_width=neighborhood_width,
        uni_w_attributes=uni_w_attributes,
        ignore_index=ignore_index
        )
    num_labels = len(obj_dict["pred_labels"])
    num_neighbors = neighborhood_width**2
    # Init the cal info tracker.
    cal_info = {
        "bin_cal_errors": torch.zeros((num_labels, num_neighbors, num_bins)),
        "bin_accs": torch.zeros((num_labels, num_neighbors, num_bins)),
        "bin_confs": torch.zeros((num_labels, num_neighbors, num_bins)),
        "bin_amounts": torch.zeros((num_labels, num_neighbors, num_bins))
    }
    for lab_idx, p_label in enumerate(obj_dict["pred_labels"]):
        for nn_idx in range(num_neighbors):
            for bin_idx, conf_bin in enumerate(obj_dict["conf_bins"]):
                # Get the region of image corresponding to the confidence
                bin_conf_region = get_conf_region(
                    y_pred=obj_dict["y_pred"],
                    y_hard=obj_dict["y_hard"],
                    bin_idx=bin_idx, 
                    conf_bin=conf_bin, 
                    conf_bin_widths=obj_dict["conf_bin_widths"], 
                    num_neighbors=nn_idx,
                    num_neighbors_map=obj_dict["matching_neighbors_map"],
                    label=p_label,
                    ignore_index=ignore_index
                    )
                # If there are some pixels in this confidence bin.
                if bin_conf_region.sum() > 0:
                    # Calculate the average score for the regions in the bin.
                    bi = calc_bin_info(
                        y_pred=obj_dict["y_pred"],
                        bin_conf_region=bin_conf_region,
                        square_diff=square_diff,
                        pixelwise_accuracy=obj_dict["pixelwise_accuracy"],
                        pix_weights=obj_dict["pix_weights"]
                    )
                    # Calculate the average calibration error for the regions in the bin.
                    cal_info["bin_confs"][lab_idx, nn_idx, bin_idx] = bi["avg_conf"] 
                    cal_info["bin_accs"][lab_idx, nn_idx, bin_idx] = bi["avg_conf"] 
                    cal_info["bin_amounts"][lab_idx, nn_idx, bin_idx] = bi["num_samples"] 
                    cal_info["bin_cal_errors"][lab_idx, nn_idx, bin_idx] = bi["cal_error"] 
    # Return the label-wise and neighborhood conditioned calibration information.
    return cal_info