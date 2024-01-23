# misc. imports
import torch
from torch import Tensor
from typing import Optional, Tuple
from pydantic import validate_arguments
# local imports 
from .utils import (
    get_bins,
    find_bins,
    get_conf_region, 
    count_matching_neighbors
)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_lab_info(
    y_hard: Tensor, 
    y_true: Tensor, 
    top_label: bool, 
    ignore_index: Optional[int] = None
):
    if top_label:
        lab_map = y_hard
    else:
        lab_map = y_true

    unique_labels = torch.unique(lab_map)

    if ignore_index is not None:
        unique_labels = unique_labels[unique_labels != ignore_index]

    return {
        "lab_map": lab_map, 
        "num_labels": len(unique_labels),
        "unique_labels": unique_labels,
    }


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def calc_bin_info(
    prob_map: Tensor,
    bin_conf_region: Tensor,
    pixelwise_accuracy: Tensor,
    square_diff: bool,
    pix_weights: Optional[Tensor] = None
):
    if pix_weights is None:
        bin_num_samples = bin_conf_region.sum() 
        avg_bin_confidence = prob_map[bin_conf_region].sum() / bin_num_samples
        avg_bin_accuracy = pixelwise_accuracy[bin_conf_region].sum() / bin_num_samples
    else:
        bin_num_samples = pix_weights[bin_conf_region].sum()
        avg_bin_confidence = (pix_weights[bin_conf_region] * prob_map[bin_conf_region]).sum() / bin_num_samples
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
    neighborhood_width: Optional[int] = None,
    stats_info_dict: Optional[dict] = {}
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
    y_hard = y_pred.argmax(dim=1) # B x H x W
    y_max_prob_map = y_pred.max(dim=1).values # B x H x W

    # Figure out where each pixel belongs (in confidence)
    if "bin_ownership_map" in stats_info_dict:
        bin_ownership_map = stats_info_dict["bin_ownership_map"]
    else:
        # Create the confidence bins.    
        conf_bins, conf_bin_widths = get_bins(
            num_bins=num_bins, 
            start=conf_interval[0], 
            end=conf_interval[1]
        )
        bin_ownership_map = find_bins(
            confidences=y_max_prob_map, 
            bin_starts=conf_bins,
            bin_widths=conf_bin_widths
        ) # B x H x W
        assert bin_ownership_map.shape == y_hard.shape,\
            f"bin_ownership_map and y_hard must have the same shape. Got {bin_ownership_map.shape} and {y_hard.shape}."

    # Get the pixelwise accuracy.
    if "accuracy_map" in stats_info_dict:
        accuracy_map = stats_info_dict["accuracy_map"]
    else:
        accuracy_map = (y_hard == y_true).float()
    
    # Get a map of which pixels match their neighbors and how often, and pixel-wise accuracy.
    if neighborhood_width is not None:
        if "pred_matching_neighbors_map" in stats_info_dict:
            pred_matching_neighbors_map = stats_info_dict["pred_matching_neighbors_map"]
        else:
            pred_matching_neighbors_map = count_matching_neighbors(
                lab_map=y_hard, 
                neighborhood_width=neighborhood_width
            )
        if "true_matching_neighbors_map" in stats_info_dict:
            true_matching_neighbors_map = stats_info_dict["true_matching_neighbors_map"]
        else:
            true_matching_neighbors_map = count_matching_neighbors(
                lab_map=y_true, 
                neighborhood_width=neighborhood_width
            )
    else:
        pred_matching_neighbors_map = None
        true_matching_neighbors_map = None 

    # Wrap this into a dictionary.
    return {
        "y_max_prob_map": y_max_prob_map.to(torch.float64),
        "y_hard": y_hard.to(torch.float64),
        "y_true": y_true.to(torch.float64),
        "pixelwise_accuracy": accuracy_map.to(torch.float64),
        "bin_ownership_map": bin_ownership_map,
        "pred_matching_neighbors_map": pred_matching_neighbors_map,
        "true_matching_neighbors_map": true_matching_neighbors_map,
        "pix_weights": None 
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
        neighborhood_width=neighborhood_width,
        stats_info_dict=stats_info_dict,
        from_logits=from_logits
        )
    # Keep track of different things for each bin.
    cal_info = {
        "bin_confs": torch.zeros(num_bins, dtype=torch.float64),
        "bin_amounts": torch.zeros(num_bins, dtype=torch.float64),
        "bin_accs": torch.zeros(num_bins, dtype=torch.float64),
        "bin_cal_errors": torch.zeros(num_bins, dtype=torch.float64),
    }
    # Get the regions of the prediction corresponding to each bin of confidence.
    for bin_idx in range(num_bins):
        # Get the region of image corresponding to the confidence
        bin_conf_region = get_conf_region(
            bin_idx=bin_idx, 
            bin_ownership_map=obj_dict["bin_ownership_map"],
            lab_map=obj_dict["y_true"], # Use ground truth to get the region.
            pred_num_neighbors_map=obj_dict["pred_matching_neighbors_map"], # Note this is off PREDICTED neighbors.
            true_num_neighbors_map=obj_dict["true_matching_neighbors_map"], # Note this is off ACTUAL neighbors.
            edge_only=edge_only,
            ignore_index=ignore_index # Ignore index will ignore ground truth pixels with this value.
            )
        # If there are some pixels in this confidence bin.
        if bin_conf_region.sum() > 0:
            # Calculate the average score for the regions in the bin.
            bi = calc_bin_info(
                prob_map=obj_dict["y_max_prob_map"],
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
        neighborhood_width=neighborhood_width,
        stats_info_dict=stats_info_dict,
        from_logits=from_logits,
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
        "bin_confs": torch.zeros((num_labels, num_bins), dtype=torch.float64),
        "bin_amounts": torch.zeros((num_labels, num_bins), dtype=torch.float64),
        "bin_accs": torch.zeros((num_labels, num_bins), dtype=torch.float64),
        "bin_cal_errors": torch.zeros((num_labels, num_bins), dtype=torch.float64)
    }
    for lab_idx, lab in enumerate(lab_info["unique_labels"]):
        for bin_idx in range(num_bins):
            # Get the region of image corresponding to the confidence
            bin_conf_region = get_conf_region(
                bin_idx=bin_idx, 
                bin_ownership_map=obj_dict["bin_ownership_map"],
                label=lab,
                lab_map=lab_info["lab_map"],
                pred_num_neighbors_map=obj_dict["pred_matching_neighbors_map"], # Note this is off PREDICTED neighbors.
                true_num_neighbors_map=obj_dict["true_matching_neighbors_map"], # Note this is off ACTUAL neighbors.
                edge_only=edge_only,
                ignore_index=ignore_index
                )
            # If there are some pixels in this confidence bin.
            if bin_conf_region.sum() > 0:
                # Calculate the average score for the regions in the bin.
                bi = calc_bin_info(
                    prob_map=obj_dict["y_max_prob_map"],
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
    stats_info_dict: Optional[dict] = {},
    ignore_index: Optional[int] = None
    ) -> dict:
    # Init some things.
    obj_dict = bin_stats_init(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        neighborhood_width=neighborhood_width,
        stats_info_dict=stats_info_dict,
        from_logits=from_logits,
        )
    # Set the cal info tracker.
    unique_pred_matching_neighbors = obj_dict["pred_matching_neighbors_map"].unique()
    num_neighbors = len(unique_pred_matching_neighbors)
    cal_info = {
        "bin_cal_errors": torch.zeros((num_neighbors, num_bins), dtype=torch.float64),
        "bin_accs": torch.zeros((num_neighbors, num_bins), dtype=torch.float64),
        "bin_confs": torch.zeros((num_neighbors, num_bins), dtype=torch.float64),
        "bin_amounts": torch.zeros((num_neighbors, num_bins), dtype=torch.float64)
    }
    for nn_idx, p_nn in enumerate(unique_pred_matching_neighbors):
        for bin_idx in range(num_bins):
            # Get the region of image corresponding to the confidence
            bin_conf_region = get_conf_region(
                bin_idx=bin_idx, 
                bin_ownership_map=obj_dict["bin_ownership_map"],
                lab_map=obj_dict["y_true"], # Use ground truth to get the region.
                num_neighbors=p_nn,
                pred_num_neighbors_map=obj_dict["pred_matching_neighbors_map"], # Note this is off PREDICTED neighbors.
                true_num_neighbors_map=obj_dict["true_matching_neighbors_map"], # Note this is off ACTUAL neighbors.
                edge_only=edge_only,
                ignore_index=ignore_index
                )
            # If there are some pixels in this confidence bin.
            if bin_conf_region.sum() > 0:
                # Calculate the average score for the regions in the bin.
                bi = calc_bin_info(
                    prob_map=obj_dict["y_max_prob_map"],
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
    stats_info_dict: Optional[dict] = {},
    ignore_index: Optional[int] = None
    ) -> dict:
    # Init some things.
    obj_dict = bin_stats_init(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        neighborhood_width=neighborhood_width,
        stats_info_dict=stats_info_dict,
        from_logits=from_logits,
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
        "bin_cal_errors": torch.zeros((num_labels, num_neighbors, num_bins), dtype=torch.float64),
        "bin_accs": torch.zeros((num_labels, num_neighbors, num_bins), dtype=torch.float64),
        "bin_confs": torch.zeros((num_labels, num_neighbors, num_bins), dtype=torch.float64),
        "bin_amounts": torch.zeros((num_labels, num_neighbors, num_bins), dtype=torch.float64)
    }
    for lab_idx, lab in enumerate(lab_info["unique_labels"]):
        for nn_idx, p_nn in enumerate(unique_pred_matching_neighbors):
            for bin_idx in range(num_bins):
                # Get the region of image corresponding to the confidence
                bin_conf_region = get_conf_region(
                    bin_idx=bin_idx, 
                    bin_ownership_map=obj_dict["bin_ownership_map"],
                    label=lab,
                    lab_map=lab_info["lab_map"],
                    num_neighbors=p_nn,
                    pred_num_neighbors_map=obj_dict["pred_matching_neighbors_map"], # Note this is off PREDICTED neighbors.
                    true_num_neighbors_map=obj_dict["true_matching_neighbors_map"], # Note this is off ACTUAL neighbors.
                    edge_only=edge_only,
                    ignore_index=ignore_index
                    )
                # If there are some pixels in this confidence bin.
                if bin_conf_region.sum() > 0:
                    # Calculate the average score for the regions in the bin.
                    bi = calc_bin_info(
                        prob_map=obj_dict["y_max_prob_map"],
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