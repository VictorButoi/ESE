# torch imports
import torch
from torch import Tensor
# misc imports
from typing import Optional, Tuple
from pydantic import validate_arguments
# local imports 
from .utils import (
    get_bins,
    find_bins,
    get_conf_region, 
    count_matching_neighbors
)
# ionpy imports
from ionpy.metrics.util import _inputs_as_onehot


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def calc_bin_info(
    prob_map: Tensor,
    frequency_map: Tensor,
    bin_conf_region: Tensor,
    square_diff: bool,
    pix_weights: Optional[Tensor] = None
):
    if pix_weights is None:
        bin_num_samples = bin_conf_region.sum() 
        avg_bin_confidence = prob_map[bin_conf_region].sum() / bin_num_samples
        avg_bin_frequency = frequency_map[bin_conf_region].sum() / bin_num_samples
    else:
        bin_num_samples = pix_weights[bin_conf_region].sum()
        avg_bin_confidence = (pix_weights[bin_conf_region] * prob_map[bin_conf_region]).sum() / bin_num_samples
        avg_bin_frequency = (pix_weights[bin_conf_region] * frequency_map[bin_conf_region]).sum() / bin_num_samples

    # Calculate the calibration error.
    if square_diff:
        cal_error = (avg_bin_confidence - avg_bin_frequency).square()
    else:
        cal_error = (avg_bin_confidence - avg_bin_frequency).abs()
    return {
        "avg_conf": avg_bin_confidence,
        "avg_freq": avg_bin_frequency,
        "cal_error": cal_error,
        "num_samples": bin_num_samples
    }


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def bin_stats_init(
    y_pred: Tensor,
    y_true: Tensor,
    num_bins: int,
    class_wise: bool,
    from_logits: bool = False,
    conf_interval: Optional[Tuple[float, float]] = None,
    neighborhood_width: Optional[int] = None,
    stats_info_dict: Optional[dict] = {}
):
    if len(y_true.shape) == 3:
        y_true = y_true.unsqueeze(1) # Unsqueezing the channel dimension.
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

    # Define the confidence interval (if not provided).
    if conf_interval is None:
        if class_wise:
            conf_interval = (0, 1)
        else:
            C = y_pred.shape[1]
            if C == 0:
                lower_bound = 0
            else:
                lower_bound = 1 / C
            upper_bound = 1
            # Set the confidence interval.
            conf_interval = (lower_bound, upper_bound)

    # Figure out where each pixel belongs (in confidence)
    if "bin_ownership_map" in stats_info_dict and not class_wise:
        bin_ownership_map = stats_info_dict["bin_ownership_map"]
    else:
        # Create the confidence bins.    
        conf_bins, conf_bin_widths = get_bins(
            num_bins=num_bins, 
            start=conf_interval[0], 
            end=conf_interval[1]
        )
        if class_wise:
            bin_ownership_map = torch.stack([
                find_bins(
                    confidences=y_pred[:, l_idx, ...], 
                    bin_starts=conf_bins,
                    bin_widths=conf_bin_widths
                ) # B x H x W
                for l_idx in range(y_pred.shape[1])], dim=0
            ) # C x B x H x W
            # Reshape to look like the y_pred.
            bin_ownership_map = bin_ownership_map.permute(1, 0, 2, 3) # B x C x H x W
            assert bin_ownership_map.shape == y_pred.shape,\
                f"class-wise bin_ownership_map and y_pred must have the same shape. Got {bin_ownership_map.shape} and {y_pred.shape}."
        else:
            bin_ownership_map = find_bins(
                confidences=y_max_prob_map, 
                bin_starts=conf_bins,
                bin_widths=conf_bin_widths
            ) # B x H x W
            assert bin_ownership_map.shape == y_hard.shape,\
                f"bin_ownership_map and y_hard must have the same shape. Got {bin_ownership_map.shape} and {y_hard.shape}."

    # Get the pixelwise frequency.
    if "frequency_map" in stats_info_dict and not class_wise:
        frequency_map = stats_info_dict["frequency_map"]
    else:
        if class_wise:
            _, frequency_map = _inputs_as_onehot(y_pred, y_true, discretize=True)
            assert frequency_map.shape == y_pred.shape,\
                f"class-wise frequency_map and y_pred must have the same shape. Got {frequency_map.shape} and {y_pred.shape}."
        else:
            frequency_map = (y_hard == y_true).float()
    
    # Get a map of which pixels match their neighbors and how often.
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
        "y_pred": y_pred.to(torch.float64), # "to" is for precision.
        "y_max_prob_map": y_max_prob_map.to(torch.float64),
        "y_hard": y_hard.to(torch.float64),
        "y_true": y_true.to(torch.float64),
        "frequence_map": frequency_map.to(torch.float64),
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
    edge_only: bool = False,
    from_logits: bool = False,
    square_diff: bool = False,
    conf_interval: Optional[Tuple[float, float]] = None,
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
        class_wise=False
    )
    # Keep track of different things for each bin.
    cal_info = {
        "bin_confs": torch.zeros(num_bins, dtype=torch.float64),
        "bin_amounts": torch.zeros(num_bins, dtype=torch.float64),
        "bin_freqs": torch.zeros(num_bins, dtype=torch.float64),
        "bin_cal_errors": torch.zeros(num_bins, dtype=torch.float64),
    }
    # Get the regions of the prediction corresponding to each bin of confidence.
    for bin_idx in range(num_bins):
        # Get the region of image corresponding to the confidence
        bin_conf_region = get_conf_region(
            bin_idx=bin_idx, 
            bin_ownership_map=obj_dict["bin_ownership_map"],
            lab_map=obj_dict["y_true"], # Use ground truth to get the region.
            true_num_neighbors_map=obj_dict["true_matching_neighbors_map"], # Note this is off ACTUAL neighbors.
            edge_only=edge_only,
            ignore_index=ignore_index, # Ignore index will ignore ground truth pixels with this value.
            )
        # If there are some pixels in this confidence bin.
        if bin_conf_region.sum() > 0:
            # Calculate the average score for the regions in the bin.
            bi = calc_bin_info(
                prob_map=obj_dict["y_max_prob_map"],
                bin_conf_region=bin_conf_region,
                frequency_map=obj_dict["frequency_map"],
                pix_weights=obj_dict["pix_weights"],
                square_diff=square_diff
            )
            # Calculate the average calibration error for the regions in the bin.
            cal_info["bin_confs"][bin_idx] = bi["avg_conf"] 
            cal_info["bin_freqs"][bin_idx] = bi["avg_freq"] 
            cal_info["bin_amounts"][bin_idx] = bi["num_samples"] 
            cal_info["bin_cal_errors"][bin_idx] = bi["cal_error"]
    # Return the calibration information.
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def top_label_bin_stats(
    y_pred: Tensor,
    y_true: Tensor,
    num_bins: int,
    edge_only: bool = False,
    square_diff: bool = False,
    from_logits: bool = False,
    conf_interval: Optional[Tuple[float, float]] = None,
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
        class_wise=False
        )
    # If top label, then everything is done based on
    # predicted values, not ground truth. 
    unique_labels = torch.unique(obj_dict["y_hard"])

    if ignore_index is not None:
        unique_labels = unique_labels[unique_labels != ignore_index]

    num_labels = len(unique_labels)
    # Setup the cal info tracker.
    cal_info = {
        "bin_confs": torch.zeros((num_labels, num_bins), dtype=torch.float64),
        "bin_amounts": torch.zeros((num_labels, num_bins), dtype=torch.float64),
        "bin_freqs": torch.zeros((num_labels, num_bins), dtype=torch.float64),
        "bin_cal_errors": torch.zeros((num_labels, num_bins), dtype=torch.float64)
    }
    for lab_idx, lab in enumerate(unique_labels):
        for bin_idx in range(num_bins):
            # Get the region of image corresponding to the confidence
            bin_conf_region = get_conf_region(
                bin_idx=bin_idx, 
                bin_ownership_map=obj_dict["bin_ownership_map"],
                label=lab,
                lab_map=obj_dict["y_hard"], # Use ground truth to get the region.
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
                    frequency_map=obj_dict["frequency_map"],
                    pix_weights=obj_dict["pix_weights"],
                    square_diff=square_diff
                )
                # Calculate the average calibration error for the regions in the bin.
                cal_info["bin_confs"][lab_idx, bin_idx] = bi["avg_conf"] 
                cal_info["bin_freqs"][lab_idx, bin_idx] = bi["avg_freq"] 
                cal_info["bin_amounts"][lab_idx, bin_idx] = bi["num_samples"] 
                cal_info["bin_cal_errors"][lab_idx, bin_idx] = bi["cal_error"] 
    # Return the label-wise calibration information.
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def joint_label_bin_stats(
    y_pred: Tensor,
    y_true: Tensor,
    num_bins: int,
    edge_only: bool = False,
    square_diff: bool = False,
    from_logits: bool = False,
    conf_interval: Optional[Tuple[float, float]] = None,
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
        class_wise=True
    )
    
    # Unlike true labels we need to get the true unique labels.
    unique_true_labels = torch.unique(obj_dict["y_true"])

    if ignore_index is not None:
        unique_true_labels = unique_true_labels[unique_true_labels != ignore_index]

    num_true_labels = len(unique_true_labels)
    # Setup the cal info tracker.
    cal_info = {
        "bin_confs": torch.zeros((unique_true_labels, num_bins), dtype=torch.float64),
        "bin_amounts": torch.zeros((unique_true_labels, num_bins), dtype=torch.float64),
        "bin_freqs": torch.zeros((unique_true_labels, num_bins), dtype=torch.float64),
        "bin_cal_errors": torch.zeros((unique_true_labels, num_bins), dtype=torch.float64)
    }
    for lab_idx, lab in enumerate(num_true_labels):
        for bin_idx in range(num_bins):
            lab_bin_ownership_map = obj_dict["bin_ownership_map"][:, lab_idx, ...]
            # Get the region of image corresponding to the confidence
            bin_conf_region = get_conf_region(
                bin_idx=bin_idx, 
                bin_ownership_map=lab_bin_ownership_map,
                true_num_neighbors_map=obj_dict["true_matching_neighbors_map"], # Note this is off ACTUAL neighbors.
                edge_only=edge_only,
                ignore_index=ignore_index
            )
            # If there are some pixels in this confidence bin.
            if bin_conf_region.sum() > 0:
                lab_prob_map = obj_dict["y_pred"][:, lab_idx, ...]
                lab_frequency_map = obj_dict["frequency_map"][:, lab_idx, ...]
                # Calculate the average score for the regions in the bin.
                bi = calc_bin_info(
                    prob_map=lab_prob_map,
                    bin_conf_region=bin_conf_region,
                    frequency_map=lab_frequency_map,
                    pix_weights=obj_dict["pix_weights"],
                    square_diff=square_diff
                )
                # Calculate the average calibration error for the regions in the bin.
                cal_info["bin_confs"][lab_idx, bin_idx] = bi["avg_conf"] 
                cal_info["bin_freqs"][lab_idx, bin_idx] = bi["avg_freq"] 
                cal_info["bin_amounts"][lab_idx, bin_idx] = bi["num_samples"] 
                cal_info["bin_cal_errors"][lab_idx, bin_idx] = bi["cal_error"] 
    # Return the label-wise calibration information.
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def neighbor_bin_stats(
    y_pred: Tensor,
    y_true: Tensor,
    num_bins: int,
    neighborhood_width: int,
    edge_only: bool = False,
    from_logits: bool = False,
    square_diff: bool = False,
    conf_interval: Optional[Tuple[float, float]] = None,
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
        class_wise=False
    )
    # Set the cal info tracker.
    unique_pred_matching_neighbors = obj_dict["pred_matching_neighbors_map"].unique()
    num_neighbors = len(unique_pred_matching_neighbors)
    cal_info = {
        "bin_cal_errors": torch.zeros((num_neighbors, num_bins), dtype=torch.float64),
        "bin_freqs": torch.zeros((num_neighbors, num_bins), dtype=torch.float64),
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
                    frequency_map=obj_dict["frequency_map"],
                    pix_weights=obj_dict["pix_weights"],
                    square_diff=square_diff
                )
                # Calculate the average calibration error for the regions in the bin.
                cal_info["bin_confs"][nn_idx, bin_idx] = bi["avg_conf"] 
                cal_info["bin_freqs"][nn_idx, bin_idx] = bi["avg_freq"] 
                cal_info["bin_amounts"][nn_idx, bin_idx] = bi["num_samples"]
                cal_info["bin_cal_errors"][nn_idx, bin_idx] = bi["cal_error"] 
    # Return the label-wise and neighborhood conditioned calibration information.
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def neighbor_joint_label_bin_stats(
    y_pred: Tensor,
    y_true: Tensor,
    num_bins: int,
    square_diff: bool,
    neighborhood_width: int,
    edge_only: bool = False,
    from_logits: bool = False,
    conf_interval: Optional[Tuple[float, float]] = None,
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
        class_wise=False
    )
    # If top label, then everything is done based on
    # predicted values, not ground truth. 
    unique_labels = torch.unique(obj_dict["y_hard"])

    if ignore_index is not None:
        unique_labels = unique_labels[unique_labels != ignore_index]

    num_labels = len(unique_labels)
    # Get the num of neighbor classes.
    unique_pred_matching_neighbors = obj_dict["pred_matching_neighbors_map"].unique()
    num_neighbors = len(unique_pred_matching_neighbors)
    # Init the cal info tracker.
    cal_info = {
        "bin_cal_errors": torch.zeros((num_labels, num_neighbors, num_bins), dtype=torch.float64),
        "bin_freqs": torch.zeros((num_labels, num_neighbors, num_bins), dtype=torch.float64),
        "bin_confs": torch.zeros((num_labels, num_neighbors, num_bins), dtype=torch.float64),
        "bin_amounts": torch.zeros((num_labels, num_neighbors, num_bins), dtype=torch.float64)
    }
    for lab_idx, lab in enumerate(unique_labels):
        for nn_idx, p_nn in enumerate(unique_pred_matching_neighbors):
            for bin_idx in range(num_bins):
                # Get the region of image corresponding to the confidence
                bin_conf_region = get_conf_region(
                    bin_idx=bin_idx, 
                    bin_ownership_map=obj_dict["bin_ownership_map"],
                    label=lab,
                    lab_map=obj_dict["y_hard"], # use pred to get the region.
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
                        frequency_map=obj_dict["frequency_map"],
                        pix_weights=obj_dict["pix_weights"],
                        square_diff=square_diff
                    )
                    # Calculate the average calibration error for the regions in the bin.
                    cal_info["bin_confs"][lab_idx, nn_idx, bin_idx] = bi["avg_conf"] 
                    cal_info["bin_freqs"][lab_idx, nn_idx, bin_idx] = bi["avg_freq"] 
                    cal_info["bin_amounts"][lab_idx, nn_idx, bin_idx] = bi["num_samples"] 
                    cal_info["bin_cal_errors"][lab_idx, nn_idx, bin_idx] = bi["cal_error"] 
    # Return the label-wise and neighborhood conditioned calibration information.
    return cal_info

