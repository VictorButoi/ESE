# local imports
from .utils import process_for_scoring, get_conf_region
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
    bin_cal_scores = torch.zeros(num_bins)
    bin_measures = torch.zeros(num_bins)
    bin_confs = torch.zeros(num_bins)
    bin_amounts = torch.zeros(num_bins)
    metrics_per_bin, confs_per_bin = {}, {}

    # Get the regions of the prediction corresponding to each bin of confidence.
    for bin_idx, conf_bin in enumerate(conf_bins):

        # Get the region of image corresponding to the confidence
        bin_conf_region = get_conf_region(bin_idx, conf_bin, conf_bin_widths, conf_map)

        # If there are some pixels in this confidence bin.
        if bin_conf_region.sum() > 0:
            bin_probs = conf_map[bin_conf_region]
            bin_preds = pred_map[bin_conf_region]
            bin_label = label_map[bin_conf_region]

            # Calculate the average score for the regions in the bin.
            avg_bin_conf = bin_probs.mean()
            measure_func = pixel_accuracy if class_type == "Multi-class" else pixel_precision     
            avg_bin_measure, all_bin_measures = measure_func(bin_preds, bin_label, return_all=True)

            # Calculate the average calibration error for the regions in the bin.
            bin_confs[bin_idx] = avg_bin_conf 
            bin_measures[bin_idx] = avg_bin_measure 
            bin_amounts[bin_idx] = bin_conf_region.sum() 
            bin_cal_scores[bin_idx] = (avg_bin_conf - avg_bin_measure).abs()

            # Keep track of accumulate metrics over the bin.
            metrics_per_bin[bin_idx] = all_bin_measures 
            confs_per_bin[bin_idx] = bin_confs

    cal_info = {
        "bins": conf_bins, 
        "bin_measures": bin_measures,
        "bin_widths": conf_bin_widths, 
        "bin_amounts": bin_amounts,
        "bin_confs": bin_confs,
        "bin_cal_scores": bin_cal_scores,
        "measures_per_bin": metrics_per_bin,
        "confs_per_bin": confs_per_bin,
        "label-wise": False 
    }
    
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
    lab_bin_cal_scores = torch.zeros((len(pred_labels), num_bins))
    lab_bin_measures = torch.zeros((len(pred_labels), num_bins))
    lab_bin_confs = torch.zeros((len(pred_labels), num_bins))
    lab_bin_amounts = torch.zeros((len(pred_labels), num_bins))

    # Keep track of all "samples"
    lab_measures_per_bin = {lab: {} for lab in pred_labels}
    lab_confs_per_bin = {lab: {} for lab in pred_labels}

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
                bin_probs = conf_map[bin_conf_region]
                bin_preds = pred_map[bin_conf_region]
                bin_label = label_map[bin_conf_region]
                
                # Calculate the average score for the regions in the bin.
                avg_bin_conf = bin_probs.mean()
                measure_func = pixel_accuracy if class_type == "Multi-class" else pixel_precision     
                avg_bin_measure, all_bin_measures = measure_func(bin_preds, bin_label, return_all=True)

                # Calculate the average calibration error for the regions in the bin.
                lab_bin_confs[lab_idx, bin_idx] = avg_bin_conf 
                lab_bin_measures[lab_idx, bin_idx] = avg_bin_measure 
                lab_bin_amounts[lab_idx, bin_idx] = bin_conf_region.sum() 
                lab_bin_cal_scores[lab_idx, bin_idx] = (avg_bin_conf - avg_bin_measure).abs()

                # Keep track of accumulate metrics over the bin.
                lab_confs_per_bin[p_label][bin_idx] = bin_probs 
                lab_measures_per_bin[p_label][bin_idx] = all_bin_measures

    cal_info = {
        "bins": conf_bins, 
        "bin_widths": conf_bin_widths, 
        "lab_bin_cal_scores": lab_bin_cal_scores,
        "lab_bin_measures": lab_bin_measures,
        "lab_bin_confs": lab_bin_confs,
        "lab_bin_amounts": lab_bin_amounts,
        "lab_measures_per_bin": lab_measures_per_bin,
        "lab_confs_per_bin": lab_confs_per_bin,
        "label-wise": True 
    }
    
    return cal_info