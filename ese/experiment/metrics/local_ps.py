# misc. imports
import torch
from torch import Tensor
from typing import Optional, Tuple
from pydantic import validate_arguments
# local imports 
from .utils import (
    get_bins,
    get_conf_region, 
    get_uni_pixel_weights,
    count_matching_neighbors
)


def get_lab_info(y_hard, y_true, top_label, ignore_index):
    if top_label:
        lab_map = y_hard
    else:
        lab_map = y_true
    unique_labels = torch.unique(lab_map)
    if ignore_index is not None:
        unique_labels = unique_labels[unique_labels != ignore_index]
    num_labels = len(unique_labels)
    return {
        "lab_map": lab_map, 
        "num_labels": num_labels,
        "unique_labels": unique_labels,
    }


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def calc_bin_info(
    conf_map: Tensor,
    bin_conf_region: Tensor,
    pixelwise_accuracy: Tensor,
    square_diff: bool,
    pix_weights: Optional[Tensor] = None
):
    if pix_weights is None:
        bin_num_samples = bin_conf_region.sum() 
        avg_bin_confidence = conf_map[bin_conf_region].sum() / bin_num_samples
        avg_bin_accuracy = pixelwise_accuracy[bin_conf_region].sum() / bin_num_samples
    else:
        bin_num_samples = pix_weights[bin_conf_region].sum()
        avg_bin_confidence = (pix_weights[bin_conf_region] * conf_map[bin_conf_region]).sum() / bin_num_samples
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
    y_pred: Tensor,
    y_true: Tensor,
    num_bins: int,
    conf_interval: Tuple[float, float],
    from_logits: bool = False,
    uniform_weighting: bool = False,
    neighborhood_width: int = 3,
    stats_info_dict: Optional[dict] = {},
    ignore_index: Optional[int] = None
):
    assert len(y_pred.shape) == len(y_true.shape) == 4,\
        f"y_pred and y_true must be 4D tensors. Got {y_pred.shape} and {y_true.shape}."
    
    # If from logits, apply softmax along channels of y pred.
    if from_logits:
        y_pred = torch.softmax(y_pred, dim=1)

    y_pred = y_pred.to(torch.float64) # Get precision for calibration.
    y_true = y_true.squeeze(1).to(torch.float64) # Remove the channel dimension.
    assert len(y_pred.shape) == 4 and len(y_true.shape) == 3,\
        f"After prep, y_pred and y_true must be 4D and 3D tensors, respectively. Got {y_pred.shape} and {y_true.shape}."

    # Get the hard predictions and the max confidences.
    y_hard = y_pred.argmax(dim=1)
    y_max_prob_map = y_pred.max(dim=1).values

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
        accuracy_map = (y_hard == y_true)
    
    # Get a map of which pixels match their neighbors and how often, and pixel-wise accuracy.
    if neighborhood_width is not None:
        if "pred_matching_neighbors_map" in stats_info_dict:
            pred_matching_neighbors_map = stats_info_dict["nn_neighbors_map"]
        else:
            pred_matching_neighbors_map = count_matching_neighbors(
                lab_map=y_hard, 
                neighborhood_width=neighborhood_width
            )
        if "true_matching_neighbors_map" in stats_info_dict:
            true_matching_neighbors_map = stats_info_dict["nn_neighbors_map"]
        else:
            true_matching_neighbors_map = count_matching_neighbors(
                lab_map=y_true, 
                neighborhood_width=neighborhood_width
            )
    else:
        pred_matching_neighbors_map = None
        true_matching_neighbors_map = None 

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
        "y_max_prob_map": y_max_prob_map.to(torch.float64),
        "y_hard": y_hard.to(torch.float64),
        "y_true": y_true.to(torch.float64),
        "pixelwise_accuracy": accuracy_map.to(torch.float64),
        "conf_bins": conf_bins,
        "conf_bin_widths": conf_bin_widths,
        "pred_matching_neighbors_map": pred_matching_neighbors_map,
        "true_matching_neighbors_map": true_matching_neighbors_map,
        "pix_weights": pixel_weights
    } 


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def bin_stats(
    y_pred: Tensor,
    y_true: Tensor,
    num_bins: int,
    conf_interval: Tuple[float, float],
    edge_only: bool = False,
    from_logits: bool = False,
    square_diff: bool = False,
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
        from_logits=from_logits,
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
            prob_map=obj_dict["y_max_prob_map"],
            bin_idx=bin_idx, 
            conf_bin=conf_bin, 
            conf_bin_widths=obj_dict["conf_bin_widths"], 
            lab_map=obj_dict["y_hard"],
            num_neighbors_map=obj_dict["pred_matching_neighbors_map"],
            edge_only=edge_only,
            ignore_index=ignore_index
            )
        # If there are some pixels in this confidence bin.
        if bin_conf_region.sum() > 0:
            # Calculate the average score for the regions in the bin.
            bi = calc_bin_info(
                conf_map=obj_dict["y_max_prob_map"],
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
    y_pred: Tensor,
    y_true: Tensor,
    top_label: bool,
    num_bins: int,
    conf_interval: Tuple[float, float],
    edge_only: bool = False,
    square_diff: bool = False,
    from_logits: bool = False,
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
        from_logits=from_logits,
        ignore_index=ignore_index
        )
    # If top label, then everything is done based on
    # predicted values, not ground truth. 
    lab_info = get_lab_info(
        y_hard=obj_dict["y_hard"],
        y_true=obj_dict["y_true"],
        top_label=top_label,
        ignore_index=ignore_index
        )
    num_labels = lab_info["num_labels"]
    # Setup the cal info tracker.
    cal_info = {
        "bin_confs": torch.zeros((num_labels, num_bins)),
        "bin_amounts": torch.zeros((num_labels, num_bins)),
        "bin_accs": torch.zeros((num_labels, num_bins)),
        "bin_cal_errors": torch.zeros((num_labels, num_bins))
    }
    for lab_idx, lab in enumerate(lab_info["unique_labels"]):
        for bin_idx, conf_bin in enumerate(obj_dict["conf_bins"]):
            # Get the region of image corresponding to the confidence
            bin_conf_region = get_conf_region(
                prob_map=obj_dict["y_max_prob_map"],
                bin_idx=bin_idx, 
                conf_bin=conf_bin, 
                conf_bin_widths=obj_dict["conf_bin_widths"], 
                label=lab,
                lab_map=lab_info["lab_map"],
                num_neighbors_map=obj_dict["pred_matching_neighbors_map"],
                edge_only=edge_only,
                ignore_index=ignore_index
                )
            # If there are some pixels in this confidence bin.
            if bin_conf_region.sum() > 0:
                # Calculate the average score for the regions in the bin.
                bi = calc_bin_info(
                    conf_map=obj_dict["y_max_prob_map"],
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
    y_pred: Tensor,
    y_true: Tensor,
    num_bins: int,
    conf_interval: Tuple[float, float],
    neighborhood_width: int,
    edge_only: bool = False,
    from_logits: bool = False,
    square_diff: bool = False,
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
        from_logits=from_logits,
        ignore_index=ignore_index
        )
    # Set the cal info tracker.
    unique_pred_matching_neighbors = obj_dict["pred_matching_neighbors_map"].unique()
    num_neighbors = len(unique_pred_matching_neighbors)
    cal_info = {
        "bin_cal_errors": torch.zeros((num_neighbors, num_bins)),
        "bin_accs": torch.zeros((num_neighbors, num_bins)),
        "bin_confs": torch.zeros((num_neighbors, num_bins)),
        "bin_amounts": torch.zeros((num_neighbors, num_bins))
    }
    for nn_idx, p_nn in enumerate(unique_pred_matching_neighbors):
        for bin_idx, conf_bin in enumerate(obj_dict["conf_bins"]):
            # Get the region of image corresponding to the confidence
            bin_conf_region = get_conf_region(
                prob_map=obj_dict["y_max_prob_map"],
                bin_idx=bin_idx, 
                conf_bin=conf_bin, 
                conf_bin_widths=obj_dict["conf_bin_widths"], 
                lab_map=obj_dict["y_hard"],
                num_neighbors=p_nn,
                num_neighbors_map=obj_dict["pred_matching_neighbors_map"],
                edge_only=edge_only,
                ignore_index=ignore_index
                )
            # If there are some pixels in this confidence bin.
            if bin_conf_region.sum() > 0:
                # Calculate the average score for the regions in the bin.
                bi = calc_bin_info(
                    conf_map=obj_dict["y_max_prob_map"],
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
    y_pred: Tensor,
    y_true: Tensor,
    top_label: bool,
    num_bins: int,
    conf_interval: Tuple[float, float],
    square_diff: bool,
    neighborhood_width: int,
    edge_only: bool = False,
    from_logits: bool = False,
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
        from_logits=from_logits,
        ignore_index=ignore_index
        )
    # Get the label information.
    lab_info = get_lab_info(
        y_hard=obj_dict["y_hard"],
        y_true=obj_dict["y_true"],
        top_label=top_label,
        ignore_index=ignore_index
        )
    num_labels = lab_info["num_labels"]
    # Get the num of neighbor classes.
    unique_pred_matching_neighbors = obj_dict["pred_matching_neighbors_map"].unique()
    num_neighbors = len(unique_pred_matching_neighbors)
    # Init the cal info tracker.
    cal_info = {
        "bin_cal_errors": torch.zeros((num_labels, num_neighbors, num_bins)),
        "bin_accs": torch.zeros((num_labels, num_neighbors, num_bins)),
        "bin_confs": torch.zeros((num_labels, num_neighbors, num_bins)),
        "bin_amounts": torch.zeros((num_labels, num_neighbors, num_bins))
    }
    for lab_idx, lab in enumerate(lab_info["unique_labels"]):
        for nn_idx, p_nn in enumerate(unique_pred_matching_neighbors):
            for bin_idx, conf_bin in enumerate(obj_dict["conf_bins"]):
                # Get the region of image corresponding to the confidence
                bin_conf_region = get_conf_region(
                    prob_map=obj_dict["y_max_prob_map"],
                    bin_idx=bin_idx, 
                    conf_bin=conf_bin, 
                    conf_bin_widths=obj_dict["conf_bin_widths"], 
                    label=lab,
                    lab_map=lab_info["lab_map"],
                    num_neighbors=p_nn,
                    num_neighbors_map=obj_dict["pred_matching_neighbors_map"],
                    edge_only=edge_only,
                    ignore_index=ignore_index
                    )
                # If there are some pixels in this confidence bin.
                if bin_conf_region.sum() > 0:
                    # Calculate the average score for the regions in the bin.
                    bi = calc_bin_info(
                        conf_map=obj_dict["y_max_prob_map"],
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