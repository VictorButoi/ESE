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
        bin_num_samples = bin_conf_region.sum() 
        avg_bin_confidence = y_pred[bin_conf_region].sum() / bin_num_samples
        avg_bin_accuracy = pixelwise_accuracy[bin_conf_region].sum() / bin_num_samples
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
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_bins: int,
    conf_interval: Tuple[float, float],
    uniform_weighting: bool = False,
    neighborhood_width: Optional[int] = None,
    stats_info_dict: Optional[dict] = {},
    ignore_index: Optional[int] = None
):
    y_pred = y_pred.squeeze()
    y_true = y_true.squeeze()
    assert len(y_pred.shape) == 3 and len(y_true.shape) == 2,\
        f"y_pred and y_true must be 3D and 2D tensors, respectively. Got {y_pred.shape} and {y_true.shape}."

    # Get the hard predictions and the max confidences.
    y_hard = y_pred.argmax(dim=0)
    y_pred = y_pred.max(dim=0).values

    # Create the confidence bins.    
    conf_bins, conf_bin_widths = get_bins(
        num_bins=num_bins, 
        start=conf_interval[0], 
        end=conf_interval[1]
        )

    # Get the pixelwise accuracy.
    if "accuracy_map" in stats_info_dict:
        accuracy_map = stats_info_dict["accuracy_map"]
    else:
        accuracy_map = (y_hard == y_true).float()

    # Keep track of different things for each bin.
    if "pred_labels" in stats_info_dict:
        pred_labels = stats_info_dict["pred_labels"]
    else:
        pred_labels = y_hard.unique().tolist()
        if ignore_index is not None and ignore_index in pred_labels:
            pred_labels.remove(ignore_index)

    # Get a map of which pixels match their neighbors and how often, and pixel-wise accuracy.
    if neighborhood_width is not None:
        if "nn_neighbors_map" in stats_info_dict:
            nn_neighborhood_map = stats_info_dict["nn_neighbors_map"]
        else:
            nn_neighborhood_map = count_matching_neighbors(
                y_hard, 
                neighborhood_width=neighborhood_width
            )
    else:
        nn_neighborhood_map = None

    # Get the pixel-weights if we are using them.
    if uniform_weighting:
        if "pixel_weights" in stats_info_dict:
            pixel_weights = stats_info_dict["pixel_weights"]
        else:
            pixel_weights = get_uni_pixel_weights(
                y_hard=y_hard, 
                uni_w_attributes=["labels", "neighbors"],
                neighborhood_width=neighborhood_width,
                ignore_index=ignore_index
                )
    else:
        pixel_weights = None

    # Wrap this into a dictionary.
    return {
        "y_pred": y_pred,
        "y_hard": y_hard,
        "conf_bins": conf_bins,
        "conf_bin_widths": conf_bin_widths,
        "pixelwise_accuracy": accuracy_map,
        "pred_labels": pred_labels,
        "matching_neighbors_map": nn_neighborhood_map,
        "pix_weights": pixel_weights
    } 


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def bin_stats(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_bins: int,
    conf_interval: Tuple[float, float],
    square_diff: bool,
    uniform_weighting: bool = False,
    neighborhood_width: Optional[int] = None,
    stats_info_dict: Optional[dict] = {},
    ignore_index: Optional[int] = None
    ) -> dict:
    # Init some things.
    obj_dict = bin_stats_init(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        uniform_weighting=uniform_weighting,
        neighborhood_width=neighborhood_width,
        stats_info_dict=stats_info_dict,
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
    uniform_weighting: bool = False,
    neighborhood_width: Optional[int] = None,
    stats_info_dict: Optional[dict] = {},
    ignore_index: Optional[int] = None
    ) -> dict:
    # Init some things.
    obj_dict = bin_stats_init(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        uniform_weighting=uniform_weighting,
        neighborhood_width=neighborhood_width,
        stats_info_dict=stats_info_dict,
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
    uniform_weighting: bool = False,
    stats_info_dict: Optional[dict] = {},
    ignore_index: Optional[int] = None
    ) -> dict:
    # Init some things.
    obj_dict = bin_stats_init(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        uniform_weighting=uniform_weighting,
        neighborhood_width=neighborhood_width,
        stats_info_dict=stats_info_dict,
        ignore_index=ignore_index
        )
    # Set the cal info tracker.
    unique_num_neighbors = obj_dict["matching_neighbors_map"].unique()
    num_neighbors = len(unique_num_neighbors)
    cal_info = {
        "bin_cal_errors": torch.zeros((num_neighbors, num_bins)),
        "bin_accs": torch.zeros((num_neighbors, num_bins)),
        "bin_confs": torch.zeros((num_neighbors, num_bins)),
        "bin_amounts": torch.zeros((num_neighbors, num_bins))
    }
    for nn_idx, p_nn in enumerate(unique_num_neighbors):
        for bin_idx, conf_bin in enumerate(obj_dict["conf_bins"]):
            # Get the region of image corresponding to the confidence
            bin_conf_region = get_conf_region(
                y_pred=obj_dict["y_pred"],
                y_hard=obj_dict["y_hard"],
                bin_idx=bin_idx, 
                conf_bin=conf_bin, 
                conf_bin_widths=obj_dict["conf_bin_widths"], 
                num_neighbors=p_nn,
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
                cal_info["bin_cal_errors"][nn_idx, bin_idx] = bi["cal_error"] 
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
    uniform_weighting: bool = False,
    stats_info_dict: Optional[dict] = {},
    ignore_index: Optional[int] = None
    ) -> dict:
    # Init some things.
    obj_dict = bin_stats_init(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        uniform_weighting=uniform_weighting,
        neighborhood_width=neighborhood_width,
        stats_info_dict=stats_info_dict,
        ignore_index=ignore_index
        )
    num_labels = len(obj_dict["pred_labels"])
    unique_num_neighbors = obj_dict["matching_neighbors_map"].unique()
    num_neighbors = len(unique_num_neighbors)
    # Init the cal info tracker.
    cal_info = {
        "bin_cal_errors": torch.zeros((num_labels, num_neighbors, num_bins)),
        "bin_accs": torch.zeros((num_labels, num_neighbors, num_bins)),
        "bin_confs": torch.zeros((num_labels, num_neighbors, num_bins)),
        "bin_amounts": torch.zeros((num_labels, num_neighbors, num_bins))
    }
    for lab_idx, p_label in enumerate(obj_dict["pred_labels"]):
        for nn_idx, p_nn in enumerate(unique_num_neighbors):
            for bin_idx, conf_bin in enumerate(obj_dict["conf_bins"]):
                # Get the region of image corresponding to the confidence
                bin_conf_region = get_conf_region(
                    y_pred=obj_dict["y_pred"],
                    y_hard=obj_dict["y_hard"],
                    bin_idx=bin_idx, 
                    conf_bin=conf_bin, 
                    conf_bin_widths=obj_dict["conf_bin_widths"], 
                    num_neighbors=p_nn,
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