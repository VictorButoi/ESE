# local imports
from .utils import process_for_scoring, get_conf_region, init_stat_tracker
# ionpy imports
from ionpy.metrics import pixel_accuracy, pixel_precision
# misc imports
import torch
from typing import Literal
from pydantic import validate_arguments


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def gather_pixelwise_bin_stats(
    num_bins: int,
    conf_bins: torch.Tensor,
    conf_bin_widths: torch.Tensor,
    conf_map: torch.Tensor,
    pred_map: torch.Tensor,
    label_map: torch.Tensor,
    class_type: Literal["Binary", "Multi-class"],
    min_confidence: float = 0.001,
    include_background: bool = True,
    ) -> dict:
    # Process the inputs for scoring
    conf_map, pred_map, label_map = process_for_scoring(
        conf_map=conf_map, 
        pred_map=pred_map, 
        label_map=label_map, 
        class_type=class_type,
        min_confidence=min_confidence,
        include_background=include_background, 
    )
    # Keep track of different things for each bin.
    cal_info = init_stat_tracker(
        num_bins=num_bins,
        label_wise=False,
        ) 
    cal_info["bins"] = conf_bins
    cal_info["bin_widths"] = conf_bin_widths
    # Get the regions of the prediction corresponding to each bin of confidence.
    for bin_idx, conf_bin in enumerate(conf_bins):
        # Get the region of image corresponding to the confidence
        bin_conf_region = get_conf_region(bin_idx, conf_bin, conf_bin_widths, conf_map)
        # If there are some pixels in this confidence bin.
        if bin_conf_region.sum() > 0:
            # Calculate the average score for the regions in the bin.
            avg_bin_conf = conf_map[bin_conf_region].mean()
            avg_bin_measure, all_bin_measures = pixel_accuracy(
                pred_map[bin_conf_region], 
                label_map[bin_conf_region], 
                return_all=True
                )
            # Calculate the average calibration error for the regions in the bin.
            cal_info["bin_confs"][bin_idx] = avg_bin_conf 
            cal_info["bin_measures"][bin_idx] = avg_bin_measure 
            cal_info["bin_amounts"][bin_idx] = bin_conf_region.sum() 
            cal_info["bin_cal_scores"][bin_idx] = (avg_bin_conf - avg_bin_measure).abs()
            # Keep track of accumulate metrics over the bin.
            cal_info["measures_per_bin"][bin_idx] = all_bin_measures 
            cal_info["confs_per_bin"][bin_idx] = conf_map[bin_conf_region]

    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def gather_labelwise_pixelwise_bin_stats(
    num_bins: int,
    conf_bins: torch.Tensor,
    conf_bin_widths: torch.Tensor,
    conf_map: torch.Tensor,
    pred_map: torch.Tensor,
    label_map: torch.Tensor,
    class_type: Literal["Binary", "Multi-class"],
    min_confidence: float = 0.001,
    include_background: bool = True,
    ) -> dict:
    # Process the inputs for scoring
    conf_map, pred_map, label_map = process_for_scoring(
        conf_map=conf_map, 
        pred_map=pred_map, 
        label_map=label_map, 
        class_type=class_type,
        min_confidence=min_confidence,
        include_background=include_background, 
    )
    # Keep track of different things for each bin.
    pred_labels = pred_map.unique().tolist()
    cal_info = init_stat_tracker(
        num_bins=num_bins,
        label_wise=True,
        labels=pred_labels
        ) 
    cal_info["bins"] = conf_bins
    cal_info["bin_widths"] = conf_bin_widths
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
                measure_func = pixel_accuracy if class_type == "Multi-class" else pixel_precision     
                avg_bin_measure, all_bin_measures = measure_func(
                    pred_map[bin_conf_region], 
                    label_map[bin_conf_region], 
                    return_all=True
                    )
                # Calculate the average calibration error for the regions in the bin.
                cal_info["lab_bin_confs"][lab_idx, bin_idx] = avg_bin_conf 
                cal_info["lab_bin_measures"][lab_idx, bin_idx] = avg_bin_measure 
                cal_info["lab_bin_amounts"][lab_idx, bin_idx] = bin_conf_region.sum() 
                cal_info["lab_bin_cal_scores"][lab_idx, bin_idx] = (avg_bin_conf - avg_bin_measure).abs()
                # Keep track of accumulate metrics over the bin.
                cal_info["lab_confs_per_bin"][p_label][bin_idx] = conf_map[bin_conf_region]
                cal_info["lab_measures_per_bin"][p_label][bin_idx] = all_bin_measures

    return cal_info