# local imports
from .utils import process_for_scoring, get_conf_region
# ionpy imports
from ionpy.metrics import pixel_accuracy, pixel_precision
# misc imports
import torch
from typing import Optional, Literal
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
    bin_cal_scores, bin_avg_metric, bin_amounts = torch.zeros(num_bins), torch.zeros(num_bins), torch.zeros(num_bins)
    metrics_per_bin, confs_per_bin = {}, {}

    # Get the regions of the prediction corresponding to each bin of confidence.
    for bin_idx, conf_bin in enumerate(conf_bins):

        # Get the region of image corresponding to the confidence
        bin_conf_region = get_conf_region(bin_idx, conf_bin, conf_bin_widths, conf_map)

        # If there are some pixels in this confidence bin.
        if bin_conf_region.sum() > 0:
            bin_confs = conf_map[bin_conf_region]
            bin_preds = pred_map[bin_conf_region]
            bin_label = label_map[bin_conf_region]

            # Calculate the average score for the regions in the bin.
            if class_type == "Multi-class":
                avg_bin_metric, all_bin_metrics = pixel_accuracy(bin_preds, bin_label, return_all=True)
            else:
                avg_bin_metric, all_bin_metrics = pixel_precision(bin_preds, bin_label, return_all=True)
            # Record the confidences
            avg_bin_conf = bin_confs.mean()

            # Calculate the average calibration error for the regions in the bin.
            bin_cal_scores[bin_idx] = (avg_bin_conf - avg_bin_metric).abs()
            bin_avg_metric[bin_idx] = avg_bin_metric
            bin_amounts[bin_idx] = bin_conf_region.sum() 

            # Keep track of accumulate metrics over the bin.
            metrics_per_bin[bin_idx] = all_bin_metrics
            confs_per_bin[bin_idx] = bin_confs

    cal_info = {
        "bins": conf_bins, 
        "bin_widths": conf_bin_widths, 
        "bin_amounts": bin_amounts,
        "bin_cal_scores": bin_cal_scores,
        "confs_per_bin": confs_per_bin
    }

    if class_type == "Multi-class":
        cal_info["bin_accs"] = bin_avg_metric
        cal_info["accs_per_bin"] = metrics_per_bin
    else:
        cal_info["bin_freqs"] = bin_avg_metric
        cal_info["freqs_per_bin"] = metrics_per_bin
    
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
    bin_cal_scores = torch.zeros((len(pred_labels), num_bins))
    bin_avg_metric = torch.zeros((len(pred_labels), num_bins))
    bin_lab_amounts = torch.zeros((len(pred_labels), num_bins))

    # Keep track of all "samples"
    metrics_per_bin = {lab: {} for lab in pred_labels}
    confs_per_bin = {lab: {} for lab in pred_labels}

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
                bin_confs = conf_map[bin_conf_region]
                bin_preds = pred_map[bin_conf_region]
                bin_label = label_map[bin_conf_region]
                
                # Calculate the average score for the regions in the bin.
                if class_type == "Multi-class":
                    avg_bin_metric, all_bin_metrics = pixel_accuracy(bin_preds, bin_label, return_all=True)
                else:
                    avg_bin_metric, all_bin_metrics = pixel_precision(bin_preds, bin_label, return_all=True)
                # Record the confidences
                avg_bin_conf = bin_confs.mean()

                # Calculate the average calibration error for the regions in the bin.
                bin_cal_scores[lab_idx, bin_idx] = (avg_bin_conf - avg_bin_metric).abs()
                bin_avg_metric[lab_idx, bin_idx] = avg_bin_metric
                bin_lab_amounts[lab_idx, bin_idx] = bin_conf_region.sum() 

                # Keep track of accumulate metrics over the bin.
                metrics_per_bin[p_label][bin_idx] = all_bin_metrics
                confs_per_bin[p_label][bin_idx] = bin_confs

    cal_info = {
        "bins": conf_bins, 
        "bin_widths": conf_bin_widths, 
        "bin_amounts": bin_lab_amounts,
        "bin_cal_scores": bin_cal_scores,
        "confs_per_bin": confs_per_bin
    }

    if class_type == "Multi-class":
        cal_info["bin_accs"] = bin_avg_metric
        cal_info["accs_per_bin"] = metrics_per_bin
    else:
        cal_info["bin_freqs"] = bin_avg_metric
        cal_info["freqs_per_bin"] = metrics_per_bin
    
    return cal_info