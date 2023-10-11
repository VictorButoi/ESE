# local imports
from .utils import get_bins, get_conf_region, process_for_scoring, reduce_scores, init_stat_tracker
# misc imports
import torch
from typing import Literal, Tuple
from pydantic import validate_arguments
# ionpy imports
from ionpy.metrics import pixel_accuracy, pixel_precision
from ionpy.util.islands import get_connected_components


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ReCE(
    num_bins: int,
    conf_map: torch.Tensor,
    pred_map: torch.Tensor,
    label_map: torch.Tensor,
    include_background: bool,
    class_type: Literal["Binary", "Multi-class"],
    weighting: str = "proportional",
    min_confidence: float = 0.001
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the ReCE: Region-wise Calibration Error
    """
    assert len(conf_map.shape) == 2 and conf_map.shape == label_map.shape, f"conf_map and label must be 2D tensors of the same shape. Got {conf_map.shape} and {label_map.shape}."
    if class_type == "Multi-class": 
        assert include_background, "Background must be included for multi-class."

    # Process the inputs for scoring
    conf_map, pred_map, label_map = process_for_scoring(
        conf_map=conf_map, 
        pred_map=pred_map, 
        label_map=label_map, 
        class_type=class_type,
        min_confidence=min_confidence,
        include_background=include_background, 
    )
    # Create the confidence bins.    
    conf_bins, conf_bin_widths = get_bins(
        num_bins=num_bins,
        metric="ReCE", 
        class_type=class_type,
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
            region_measures = torch.zeros(num_islands)
            region_confs = torch.zeros(num_islands)
            region_calibration = torch.zeros(num_islands)
            # Iterate through each island, and get the measure for each island.
            for isl_idx, island in enumerate(conf_islands):
                # Get the island primitives
                region_conf_map = conf_map[island]                
                region_pred_map = pred_map[island]
                region_label_map = label_map[island]
                # Calculate the average score for the regions in the bin.
                if class_type == "Multi-class":
                    region_measures [isl_idx] = pixel_accuracy(region_pred_map, region_label_map)
                else:
                    region_measures [isl_idx] = pixel_precision(region_pred_map, region_label_map)
                # Record the confidences
                region_confs[isl_idx] = region_conf_map.mean()
                # Calculate the calibration error WITHIN the island.
                region_calibration[isl_idx] = (region_confs[isl_idx] - region_measures[isl_idx]).abs()
            # Calculate the average calibration error for the regions in the bin.
            cal_info["bin_amounts"][bin_idx] = num_islands 
            cal_info["bin_confs"][bin_idx] = region_confs.mean()
            cal_info["bin_measures"][bin_idx] = region_measures.mean() 
            cal_info["bin_cal_scores"][bin_idx] = region_calibration.mean()
            # Now we put the ISLAND values for the accumulation
            cal_info["confs_per_bin"][bin_idx] = region_confs
            cal_info["measures_per_bin"][bin_idx] = region_measures
    # Finally, get the ReCE score.
    rece_score = reduce_scores(
        score_per_bin=cal_info["bin_cal_scores"], 
        amounts_per_bin=cal_info["bin_amounts"], 
        weighting=weighting
        )
    cal_info["cal_score"] = rece_score
    return cal_info
    