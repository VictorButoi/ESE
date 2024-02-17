# torch imports
import torch
from torch import Tensor
# misc imports
from typing import Optional, Tuple
from pydantic import validate_arguments
# local imports 
from .utils import (
    get_conf_region, 
    get_bin_per_sample,
    agg_neighbors_preds 
)


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
    neighborhood_width: Optional[int] = None
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
    C = y_pred.shape[1]
    assert len(y_pred.shape) == 4 and len(y_true.shape) == 3,\
        f"After prep, y_pred and y_true must be 4D and 3D tensors, respectively. Got {y_pred.shape} and {y_true.shape}."

    # Get the hard predictions and the max confidences.
    y_hard = y_pred.argmax(dim=1) # B x H x W
    y_max_prob_map = y_pred.max(dim=1).values # B x H x W

    # Define the confidence interval (if not provided).
    if conf_interval is None:
        if class_wise:
            conf_interval = (0.0, 1.0)
        else:
            if C == 0:
                lower_bound = 0.0
            else:
                lower_bound = 1 / C
            upper_bound = 1.0
            # Set the confidence interval.
            conf_interval = (lower_bound, upper_bound)

    if class_wise:
        assert conf_interval == (0.0, 1.0),\
            f"Confidence interval must be (0, 1) for class-wise binning. Got {conf_interval}."
    conf_bin_map = get_bin_per_sample(
        pred_map=y_pred if class_wise else y_max_prob_map,
        num_bins=num_bins,
        start=conf_interval[0],
        end=conf_interval[1],
        class_wise=class_wise
    ) # B x H x W

    # Get a map of which pixels match their neighbors and how often.
    if neighborhood_width is not None:
        # Predicted map
        pred_neighbors_map = agg_neighbors_preds(
                                pred_map=y_hard,
                                neighborhood_width=neighborhood_width,
                                discrete=True,
                                class_wise=class_wise,
                                num_classes=C,
                                binary=True
                            )
        # True map
        true_neighbors_map = agg_neighbors_preds(
                                pred_map=y_true.long(),
                                neighborhood_width=neighborhood_width,
                                discrete=True,
                                class_wise=class_wise,
                                num_classes=C,
                                binary=True
                            )
    else:
        pred_neighbors_map = None
        true_neighbors_map = None 

    # Get the pixelwise frequency.
    if class_wise:
        C = y_pred.shape[1]
        frequency_map = torch.nn.functional.one_hot(y_true.long(), C).float()
        # Reshape it from B x H x W x C -> B x C x H x W
        frequency_map = frequency_map.permute(0, 3, 1, 2)
    else:
        frequency_map = (y_hard == y_true).float()
    
    # Wrap this into a dictionary.
    return {
        "y_pred": y_pred.to(torch.float64), # "to" is for precision.
        "y_max_prob_map": y_max_prob_map.to(torch.float64),
        "y_hard": y_hard.to(torch.float64),
        "y_true": y_true.to(torch.float64),
        "frequency_map": frequency_map.to(torch.float64),
        "bin_ownership_map": conf_bin_map,
        "pred_neighbors_map": pred_neighbors_map,
        "true_neighbors_map": true_neighbors_map,
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
    ) -> dict:
    # Init some things.
    obj_dict = bin_stats_init(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        neighborhood_width=neighborhood_width,
        from_logits=from_logits,
        class_wise=False
    )
    # Keep track of different things for each bin.
    cal_info = {
        "bin_confs": torch.zeros(num_bins, dtype=torch.float64),
        "bin_freqs": torch.zeros(num_bins, dtype=torch.float64),
        "bin_amounts": torch.zeros(num_bins, dtype=torch.float64),
        "bin_cal_errors": torch.zeros(num_bins, dtype=torch.float64),
    }
    # Get the regions of the prediction corresponding to each bin of confidence.
    for bin_idx in range(num_bins):
        # Get the region of image corresponding to the confidence
        bin_conf_region = get_conf_region(
            conditional_region_dict={
                "bin_idx": (bin_idx, obj_dict["bin_ownership_map"]),
            },
            gt_nn_map=obj_dict["true_neighbors_map"], # Note this is off ACTUAL neighbors.
            neighborhood_width=neighborhood_width,
            edge_only=edge_only,
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
            for k, v in bi.items():
                # Assert that v is not a torch NaN
                assert not torch.isnan(v).any(), f"Bin {bin_idx} has NaN in key: {k}."
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
) -> dict:
    # Init some things.
    obj_dict = bin_stats_init(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        neighborhood_width=neighborhood_width,
        from_logits=from_logits,
        class_wise=False
    )
    # If top label, then everything is done based on
    # predicted values, not ground truth. 
    unique_labels = torch.unique(obj_dict["y_hard"])

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
                conditional_region_dict={
                    "bin_idx": (bin_idx, obj_dict["bin_ownership_map"]),
                    "pred_label": (lab, obj_dict["y_hard"])
                },
                gt_nn_map=obj_dict["true_neighbors_map"], # Note this is off ACTUAL neighbors.
                neighborhood_width=neighborhood_width,
                edge_only=edge_only,
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
                for k, v in bi.items():
                    # Assert that v is not a torch NaN
                    assert not torch.isnan(v).any(), f"Lab {lab}, Bin {bin_idx} has NaN in key: {k}."
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
) -> dict:

    # Init some things.
    obj_dict = bin_stats_init(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        neighborhood_width=neighborhood_width,
        from_logits=from_logits,
        class_wise=True
    )
    
    # Unlike true labels we need to get the true unique labels.
    max_label = y_pred.shape[1]
    label_set = torch.arange(max_label)
    
    # Setup the cal info tracker.
    n_labs = len(label_set)
    cal_info = {
        "bin_confs": torch.zeros((n_labs, num_bins), dtype=torch.float64),
        "bin_freqs": torch.zeros((n_labs, num_bins), dtype=torch.float64),
        "bin_amounts": torch.zeros((n_labs, num_bins), dtype=torch.float64),
        "bin_cal_errors": torch.zeros((n_labs, num_bins), dtype=torch.float64)
    }
    for l_idx, lab in enumerate(label_set):
        lab_prob_map = obj_dict["y_pred"][:, lab, ...]
        lab_frequency_map = obj_dict["frequency_map"][:, lab, ...]
        lab_bin_ownership_map = obj_dict["bin_ownership_map"][:, lab, ...]
        lab_true_neighbors_map = obj_dict["true_neighbors_map"][:, lab, ...]
        # Cycle through the probability bins.
        for bin_idx in range(num_bins):
            # Get the region of image corresponding to the confidence
            bin_conf_region = get_conf_region(
                conditional_region_dict={
                    "bin_idx": (bin_idx, lab_bin_ownership_map)
                },
                gt_nn_map=lab_true_neighbors_map, # Note this is off ACTUAL neighbors.
                neighborhood_width=neighborhood_width,
                edge_only=edge_only,
            )
            # If there are some pixels in this confidence bin.
            if bin_conf_region.sum() > 0:
                # Calculate the average score for the regions in the bin.
                bi = calc_bin_info(
                    prob_map=lab_prob_map,
                    bin_conf_region=bin_conf_region,
                    frequency_map=lab_frequency_map,
                    pix_weights=obj_dict["pix_weights"],
                    square_diff=square_diff
                )
                for k, v in bi.items():
                    # Assert that v is not a torch NaN
                    assert not torch.isnan(v).any(), f"Lab {lab}, Bin {bin_idx} has NaN in key: {k}."
                # Calculate the average calibration error for the regions in the bin.
                cal_info["bin_confs"][l_idx, bin_idx] = bi["avg_conf"] 
                cal_info["bin_freqs"][l_idx, bin_idx] = bi["avg_freq"] 
                cal_info["bin_amounts"][l_idx, bin_idx] = bi["num_samples"] 
                cal_info["bin_cal_errors"][l_idx, bin_idx] = bi["cal_error"] 
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
    ) -> dict:
    # Init some things.
    obj_dict = bin_stats_init(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        neighborhood_width=neighborhood_width,
        from_logits=from_logits,
        class_wise=False
    )
    # Set the cal info tracker.
    unique_pred_matching_neighbors = obj_dict["pred_neighbors_map"].unique()
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
                conditional_region_dict={
                    "bin_idx": (bin_idx, obj_dict["bin_ownership_map"]),
                    "pred_nn": (p_nn, obj_dict["pred_neighbors_map"])
                },
                gt_lab_map=obj_dict["y_true"], # Use ground truth to get the region.
                gt_nn_map=obj_dict["true_neighbors_map"], # Note this is off ACTUAL neighbors.
                neighborhood_width=neighborhood_width,
                edge_only=edge_only,
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
                for k, v in bi.items():
                    # Assert that v is not a torch NaN
                    assert not torch.isnan(v).any(), f"Num-neighbors {p_nn}, Bin {bin_idx} has NaN in key: {k}."
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
    ) -> dict:
    # Init some things.
    obj_dict = bin_stats_init(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        neighborhood_width=neighborhood_width,
        from_logits=from_logits,
        class_wise=True
    )

    # Unlike true labels we need to get the true unique labels.
    max_label = y_pred.shape[1]
    label_set = torch.arange(max_label)
    
    # Setup the cal info tracker.
    n_labs = len(label_set)
    unique_pred_matching_neighbors = obj_dict["pred_neighbors_map"].unique()
    num_neighbors = len(unique_pred_matching_neighbors)
    # Init the cal info tracker.
    cal_info = {
        "bin_cal_errors": torch.zeros((n_labs, num_neighbors, num_bins), dtype=torch.float64),
        "bin_freqs": torch.zeros((n_labs, num_neighbors, num_bins), dtype=torch.float64),
        "bin_confs": torch.zeros((n_labs, num_neighbors, num_bins), dtype=torch.float64),
        "bin_amounts": torch.zeros((n_labs, num_neighbors, num_bins), dtype=torch.float64)
    }
    for l_idx, lab in enumerate(label_set):
        lab_prob_map = obj_dict["y_pred"][:, lab, ...]
        lab_frequency_map = obj_dict["frequency_map"][:, lab, ...]
        lab_bin_ownership_map = obj_dict["bin_ownership_map"][:, lab, ...]
        lab_pred_neighbors_map = obj_dict["pred_neighbors_map"][:, lab, ...]
        lab_true_neighbors_map = obj_dict["true_neighbors_map"][:, lab, ...]
        # Cycle through the neighborhood classes.
        for nn_idx, p_nn in enumerate(unique_pred_matching_neighbors):
            for bin_idx in range(num_bins):
                # Get the region of image corresponding to the confidence
                bin_conf_region = get_conf_region(
                    bin_idx=bin_idx, 
                    bin_ownership_map=lab_bin_ownership_map,
                    true_num_neighbors_map=lab_true_neighbors_map, # Note this is off ACTUAL neighbors.
                    pred_nn=p_nn,
                    pred_num_neighbors_map=lab_pred_neighbors_map, # Note this is off PREDICTED neighbors.
                    neighborhood_width=neighborhood_width,
                    edge_only=edge_only
                )
                # If there are some pixels in this confidence bin.
                if bin_conf_region.sum() > 0:
                    # Calculate the average score for the regions in the bin.
                    bi = calc_bin_info(
                        prob_map=lab_prob_map,
                        bin_conf_region=bin_conf_region,
                        frequency_map=lab_frequency_map,
                        pix_weights=obj_dict["pix_weights"],
                        square_diff=square_diff
                    )
                    for k, v in bi.items():
                        # Assert that v is not a torch NaN
                        assert not torch.isnan(v).any(), f"Label {lab}, Num-neighbors {p_nn}, Bin {bin_idx} has NaN in key: {k}."
                    # Calculate the average calibration error for the regions in the bin.
                    cal_info["bin_confs"][l_idx, nn_idx, bin_idx] = bi["avg_conf"] 
                    cal_info["bin_freqs"][l_idx, nn_idx, bin_idx] = bi["avg_freq"] 
                    cal_info["bin_amounts"][l_idx, nn_idx, bin_idx] = bi["num_samples"] 
                    cal_info["bin_cal_errors"][l_idx, nn_idx, bin_idx] = bi["cal_error"] 
    # Return the label-wise and neighborhood conditioned calibration information.
    return cal_info

