# local imports
from ..utils.utils import get_bins, reduce_scores, process_for_scoring, get_conf_region, init_stat_tracker
from ..utils.bin_stats import gather_pixelwise_bin_stats, gather_labelwise_pixelwise_bin_stats
# ionpy imports
from ionpy.metrics import pixel_accuracy, pixel_precision
from ionpy.util.islands import get_connected_components
# misc imports
import torch
from typing import Literal
from pydantic import validate_arguments


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ECE(
    num_bins: int,
    conf_map: torch.Tensor, 
    pred_map: torch.Tensor, 
    label_map: torch.Tensor,
    class_type: Literal["Binary", "Multi-class"] = "Multi-class",
    include_background: bool = True,
    weighting: str = "proportional",
    min_confidence: float = 0.001,
    ) -> dict:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    assert len(conf_map.shape) == 2 and conf_map.shape == label_map.shape, f"conf_map and label_map must be 2D tensors of the same shape. Got {conf_map.shape} and {label_map.shape}."
    if class_type == "Multi-class": 
        assert include_background, "Background must be included for multi-class."
    # Create the confidence bins.    
    conf_bins, conf_bin_widths = get_bins(
        num_bins=num_bins,
        metric="ECE", 
        include_background=include_background, 
        class_type=class_type
        )
    # Keep track of different things for each bin.
    cal_info = gather_pixelwise_bin_stats(
        num_bins=num_bins,
        conf_bins=conf_bins,
        conf_bin_widths=conf_bin_widths,
        conf_map=conf_map,
        pred_map=pred_map,
        label_map=label_map,
        class_type=class_type,
        min_confidence=min_confidence,
        include_background=include_background
    )
    # Finally, get the calibration score.
    cal_info['cal_score'] = reduce_scores(
        score_per_bin=cal_info["bin_cal_scores"], 
        amounts_per_bin=cal_info["bin_amounts"], 
        weighting=weighting
        )
    # Return the calibration information
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def TL_ECE(
    num_bins: int,
    conf_map: torch.Tensor, 
    pred_map: torch.Tensor, 
    label_map: torch.Tensor,
    class_type: Literal["Binary", "Multi-class"] = "Multi-class",
    include_background: bool = True,
    weighting: str = "proportional",
    min_confidence: float = 0.001,
    ) -> dict:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    assert len(conf_map.shape) == 2 and conf_map.shape == label_map.shape, f"conf_map and label_map must be 2D tensors of the same shape. Got {conf_map.shape} and {label_map.shape}."
    if class_type == "Multi-class": 
        assert include_background, "Background must be included for multi-class."
    # Create the confidence bins.    
    conf_bins, conf_bin_widths = get_bins(
        num_bins=num_bins,
        metric="ECE", 
        include_background=include_background, 
        class_type=class_type
        )
    # Keep track of different things for each bin.
    cal_info = gather_labelwise_pixelwise_bin_stats(
        num_bins=num_bins,
        conf_bins=conf_bins,
        conf_bin_widths=conf_bin_widths,
        conf_map=conf_map,
        pred_map=pred_map,
        label_map=label_map,
        class_type=class_type,
        min_confidence=min_confidence,
    )
    # Finally, get the ECE score.
    num_labels = len(cal_info["lab_bin_cal_scores"])
    lab_amounts = cal_info['lab_bin_amounts'].sum(dim=1)
    w_ece_per_label = torch.zeros(num_labels)
    # Iterate through each label and calculate the weighted ece.
    for lab_idx in range(num_labels):
        lab_ece = reduce_scores(
            score_per_bin=cal_info['lab_bin_cal_scores'][lab_idx], 
            amounts_per_bin=cal_info['lab_bin_amounts'][lab_idx], 
            weighting=weighting
            )
        w_ece_per_label[lab_idx] = lab_amounts[lab_idx] * lab_ece
    # Finally, get the calibration score.
    cal_info['cal_score'] =  w_ece_per_label.sum() / lab_amounts.sum() 
    # Return the calibration information
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def CW_ECE(
    num_bins: int,
    conf_map: torch.Tensor, 
    pred_map: torch.Tensor, 
    label_map: torch.Tensor,
    class_type: Literal["Binary", "Multi-class"],
    include_background: bool,
    weighting: str = "proportional",
    min_confidence: float = 0.001,
    ) -> dict:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    assert len(conf_map.shape) == 2 and conf_map.shape == label_map.shape, f"conf_map and label_map must be 2D tensors of the same shape. Got {conf_map.shape} and {label_map.shape}."
    if class_type == "Multi-class": 
        assert include_background, "Background must be included for multi-class."
    # Create the confidence bins.    
    conf_bins, conf_bin_widths = get_bins(
        num_bins=num_bins,
        metric="ECE", 
        include_background=include_background, 
        class_type=class_type
        )
    # Keep track of different things for each bin.
    cal_info = gather_labelwise_pixelwise_bin_stats(
        num_bins=num_bins,
        conf_bins=conf_bins,
        conf_bin_widths=conf_bin_widths,
        conf_map=conf_map,
        pred_map=pred_map,
        label_map=label_map,
        class_type=class_type,
        min_confidence=min_confidence,
    )
    # Finally, get the ECE score.
    num_labels = len(cal_info["lab_bin_cal_scores"])
    ece_per_label = torch.zeros(num_labels)
    # Iterate through each label, calculating ECE
    for lab_idx in range(num_labels):
        ece_per_label[lab_idx] = reduce_scores(
            score_per_bin=cal_info["lab_bin_cal_scores"][lab_idx], 
            amounts_per_bin=cal_info["lab_bin_amounts"][lab_idx], 
            weighting=weighting
            )
    # Finally, get the calibration score.
    cal_info['cal_score'] = ece_per_label.sum() / num_labels
    # Return the calibration information
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ACE(
    num_bins: int,
    conf_map: torch.Tensor,
    pred_map: torch.Tensor,
    label_map: torch.Tensor,
    include_background: bool,
    class_type: Literal["Binary", "Multi-class"],
    weighting: str = "proportional",
    min_confidence: float = 0.001,
    ) -> dict:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    assert len(conf_map.shape) == 2 and conf_map.shape == label_map.shape, f"conf_map and label_map must be 2D tensors of the same shape. Got {conf_map.shape} and {label_map.shape}."
    if class_type == "Multi-class": 
        assert include_background, "Background must be included for multi-class."
    # Create the adaptive confidence bins.    
    conf_bins, conf_bin_widths = get_bins(
        num_bins=num_bins, 
        metric="ACE", 
        include_background=include_background, 
        class_type=class_type,
        conf_map=conf_map
        )
    # Keep track of different things for each bin.
    cal_info = gather_pixelwise_bin_stats(
        num_bins=num_bins,
        conf_bins=conf_bins,
        conf_bin_widths=conf_bin_widths,
        conf_map=conf_map,
        pred_map=pred_map,
        label_map=label_map,
        class_type=class_type,
        min_confidence=min_confidence,
        include_background=include_background
    )
    # Finally, get the calibration score.
    cal_info['cal_score'] = reduce_scores(
        score_per_bin=cal_info["bin_cal_scores"], 
        amounts_per_bin=cal_info["bin_amounts"], 
        weighting=weighting
        )
    # Return the calibration information
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def Island_ECE(
    num_bins: int,
    conf_map: torch.Tensor,
    pred_map: torch.Tensor,
    label_map: torch.Tensor,
    include_background: bool,
    class_type: Literal["Binary", "Multi-class"],
    weighting: str = "proportional",
    min_confidence: float = 0.001
    ) -> dict:
    """
    Calculates the ReCE: Region-wise Calibration Error
    """
    assert len(conf_map.shape) == 2 and conf_map.shape == label_map.shape, f"conf_map and label_map must be 2D tensors of the same shape. Got {conf_map.shape} and {label_map.shape}."
    if class_type == "Multi-class": 
        assert include_background, "Background must be included for multi-class."
    # Create the confidence bins.    
    conf_bins, conf_bin_widths = get_bins(
        num_bins=num_bins,
        metric="ReCE", 
        class_type=class_type,
        include_background=include_background, 
        )
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
    # Go through each bin, starting at the back so that we don't have to run connected components
    for bin_idx, conf_bin in enumerate(conf_bins):
        # Get the region of image corresponding to the confidence
        bin_conf_region = get_conf_region(bin_idx, conf_bin, conf_bin_widths, conf_map)
        # If there are some pixels in this confidence bin.
        if bin_conf_region.sum() > 0:
            # If we are not the last bin, get the connected components.
            conf_islands = get_connected_components(bin_conf_region)
            # Iterate through each island, and get the measure for each island.
            num_islands = len(conf_islands)
            region_metrics = torch.zeros(num_islands)
            region_confs = torch.zeros(num_islands)
            # Iterate through each island, and get the measure for each island.
            for isl_idx, island in enumerate(conf_islands):
                # Get the island primitives
                region_conf_map = conf_map[island]                
                region_pred_map = pred_map[island]
                region_label = label_map[island]
                # Calculate the average score for the regions in the bin.
                if class_type == "Multi-class":
                    region_metrics[isl_idx] = pixel_accuracy(region_pred_map, region_label)
                else:
                    region_metrics[isl_idx] = pixel_precision(region_pred_map, region_label)
                # Record the confidences
                region_confs[isl_idx] = region_conf_map.mean()
            # Get the accumulate metrics from all the islands
            avg_bin_conf = region_confs.mean()
            avg_bin_metric = region_metrics.mean()
            # Calculate the average calibration error for the regions in the bin.
            cal_info["bin_confs"][bin_idx] = avg_bin_conf 
            cal_info["bin_amounts"][bin_idx] = num_islands
            cal_info["bin_measures"][bin_idx] = avg_bin_metric
            cal_info["bin_cal_scores"][bin_idx] = (avg_bin_conf - avg_bin_metric).abs()
            # Now we put the ISLAND values for the accumulation
            cal_info["confs_per_bin"][bin_idx] = region_confs
            cal_info["measures_per_bin"][bin_idx] = region_metrics
    # Finally, get the ReCE score.
    cal_info["cal_score"] = reduce_scores(
        score_per_bin=cal_info["bin_cal_scores"], 
        amounts_per_bin=cal_info["bin_amounts"], 
        weighting=weighting
        )
    # Return the calibration information
    return cal_info
    