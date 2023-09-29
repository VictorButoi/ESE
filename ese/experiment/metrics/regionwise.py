# local imports
from .utils import get_bins, get_conf_region, process_for_scoring, reduce_scores

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
    label: torch.Tensor,
    include_background: bool,
    class_type: Literal["Binary", "Multi-class"],
    weighting: str = "proportional",
    min_confidence: float = 0.001
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the ReCE: Region-wise Calibration Error
    """
    assert len(conf_map.shape) == 2 and conf_map.shape == label.shape, f"conf_map and label must be 2D tensors of the same shape. Got {conf_map.shape} and {label.shape}."
    if class_type == "Multi-class": 
        assert include_background, "Background must be included for multi-class."
    raise ValueError("ReCE CURRENTLY BROKEN!")

    # Process the inputs for scoring
    conf_map, pred_map, label = process_for_scoring(
        conf_map=conf_map, 
        pred_map=pred_map, 
        label=label, 
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
    rece_per_bin, bin_avg_metric, bin_amounts = torch.zeros(num_bins), torch.zeros(num_bins), torch.zeros(num_bins)
    metrics_per_bin, confs_per_bin = {}, {}

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
                region_label = label[island]

                # Calculate the average score for the regions in the bin.
                if class_type == "Multi-class":
                    region_metrics[isl_idx] = pixel_accuracy(region_pred_map, region_label)
                else:
                    region_metrics[isl_idx] = pixel_precision(region_pred_map, region_label)
                # Record the confidences
                region_confs[isl_idx] = region_conf_map.mean()

                # Calculate the calibration error WITHIN the island.
                island_rece = (region_confs[isl_idx] - region_metrics[isl_idx]).abs()
            
            # Get the accumulate metrics from all the islands
            avg_bin_metric = region_metrics.mean()
            avg_bin_conf = region_confs.mean()
            
            # Calculate the average calibration error for the regions in the bin.
            rece_per_bin[bin_idx] = (avg_bin_conf - avg_bin_metric).abs()
            bin_avg_metric[bin_idx] = avg_bin_metric
            bin_amounts[bin_idx] = num_islands

            # Now we put the ISLAND values for the accumulation
            metrics_per_bin[bin_idx] = region_metrics
            confs_per_bin[bin_idx] = region_confs

    # Finally, get the ReCE score.
    rece_score = reduce_scores(
        score_per_bin=rece_per_bin, 
        amounts_per_bin=bin_amounts, 
        weighting=weighting
        )
    
    cal_info = {
        "score": rece_score,
        "bins": conf_bins, 
        "bin_widths": conf_bin_widths, 
        "bin_amounts": bin_amounts,
        "bin_scores": rece_per_bin,
        "confs_per_bin": confs_per_bin
    }

    if class_type == "Multi-class":
        cal_info["bin_accs"] = bin_avg_metric
        cal_info["accs_per_bin"] = metrics_per_bin
    else:
        cal_info["bin_freqs"] = bin_avg_metric
        cal_info["freqs_per_bin"] = metrics_per_bin

    return cal_info
    