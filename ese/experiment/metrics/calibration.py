# local imports
from .utils import get_bins, get_conf_region, reduce_scores

# misc imports
import torch
from typing import Tuple
from pydantic import validate_arguments

# ionpy imports
from ionpy.metrics import pixel_accuracy, pixel_precision
from ionpy.util.islands import get_connected_components


measure_dict = {
    "Accuracy": pixel_accuracy,
    "Frequency": pixel_precision
}


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ECE(
    num_bins: int,
    conf_map: torch.Tensor, 
    pred_map: torch.Tensor, 
    label: torch.Tensor,
    measure: str = "Frequency",
    weighting: str = "proportional",
    include_background: bool = False,
    min_confidence: float = 0.05,
    threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    assert len(conf_map.shape) == 2 and conf_map.shape == label.shape, f"conf_map and label must be 2D tensors of the same shape. Got {conf_map.shape} and {label.shape}."

    # Eliminate the super small predictions to get a better picture.
    label = label[conf_map >= min_confidence]
    pred_map = pred_map[conf_map >= min_confidence]
    conf_map = conf_map[conf_map >= min_confidence]

    # If you don't want to include background pixels, remove them.
    if not include_background:
        label = label[conf_map >= threshold]
        pred_map = pred_map[conf_map >= threshold]
        conf_map = conf_map[conf_map >= threshold]

    # Create the confidence bins.    
    conf_bins, conf_bin_widths = get_bins(
        metric="ECE", 
        include_background=include_background, 
        threshold=threshold, 
        num_bins=num_bins
        )

    # Keep track of different things for each bin, dicts are
    # for cummulation over multiple images, tensors are for one image.
    freqs_per_bin = {}
    confs_per_bin = {}

    ece_per_bin = torch.zeros(num_bins)
    avg_freq_per_bin = torch.zeros(num_bins)
    bin_amounts = torch.zeros(num_bins)

    # Get the regions of the prediction corresponding to each bin of confidence.
    for bin_idx, conf_bin in enumerate(conf_bins):

        # Get the region of image corresponding to the confidence
        bin_conf_region = get_conf_region(bin_idx, conf_bin, conf_bin_widths, conf_map)

        # If there are some pixels in this confidence bin.
        if bin_conf_region.sum() > 0:
            bin_confs = conf_map[bin_conf_region]
            bin_preds = pred_map[bin_conf_region]
            bin_label = label[bin_conf_region]

            # Calculate the accuracy and mean confidence for the island.
            avg_bin_metric = measure_dict[measure](bin_preds, bin_label)
            avg_bin_conf = bin_confs.mean()
            bin_freqs = avg_bin_metric

            # Calculate the average calibration error for the regions in the bin.
            ece_per_bin[bin_idx] = (avg_bin_metric - avg_bin_conf).abs()
            avg_freq_per_bin[bin_idx] = bin_freqs.mean()
            bin_amounts[bin_idx] = bin_conf_region.sum() 

            # Store the frequencies and confidences for each bin.
            freqs_per_bin[bin_idx] = bin_freqs 
            confs_per_bin[bin_idx] = bin_confs 
        else:
            freqs_per_bin[bin_idx] = torch.Tensor([]) 
            confs_per_bin[bin_idx] = torch.Tensor([]) 

    # Finally, get the ECE score.
    ece_score = reduce_scores(
        score_per_bin=ece_per_bin, 
        amounts_per_bin=bin_amounts, 
        weighting=weighting
        )
    
    return {
        "score": ece_score,
        "bins": conf_bins, 
        "bin_widths": conf_bin_widths, 
        "bin_amounts": bin_amounts,
        "score_per_bin": ece_per_bin, 
        "avg_freq_per_bin": avg_freq_per_bin,
        "freqs_per_bin": freqs_per_bin, 
        "confs_per_bin": confs_per_bin,
    }


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ACE(
    num_bins: int,
    conf_map: torch.Tensor,
    pred_map: torch.Tensor,
    label: torch.Tensor,
    measure: str = "Frequency",
    weighting: str = "proportional",
    include_background: bool = False,
    min_confidence: float = 0.05,
    threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    assert len(conf_map.shape) == 2 and conf_map.shape == label.shape, f"conf_map and label must be 2D tensors of the same shape. Got {conf_map.shape} and {label.shape}."

    # Eliminate the super small predictions to get a better picture.
    label = label[conf_map >= min_confidence]
    pred_map = pred_map[conf_map >= min_confidence]
    conf_map = conf_map[conf_map >= min_confidence]

    # If you don't want to include background pixels, remove them.
    if not include_background:
        label = label[conf_map >= threshold]
        pred_map = pred_map[conf_map >= threshold]
        conf_map = conf_map[conf_map >= threshold]

    # Create the adaptive confidence bins.    
    conf_bins, conf_bin_widths = get_bins(
        metric="ACE", 
        include_background=include_background, 
        threshold=threshold, 
        num_bins=num_bins, 
        conf_map=conf_map
        )
    
    # Keep track of different things for each bin, dicts are
    # for cummulation over multiple images, tensors are for one image.
    freqs_per_bin = {}
    confs_per_bin = {}

    ace_per_bin = torch.zeros(num_bins)
    avg_freq_per_bin = torch.zeros(num_bins)
    bin_amounts = torch.zeros(num_bins)

    # Get the regions of the prediction corresponding to each bin of confidence.
    for bin_idx, conf_bin in enumerate(conf_bins):

        # Get the region of image corresponding to the confidence
        bin_conf_region = get_conf_region(bin_idx, conf_bin, conf_bin_widths, conf_map)

        # If there are some pixels in this confidence bin.
        if bin_conf_region.sum() > 0:
            bin_confs = conf_map[bin_conf_region]
            bin_preds = pred_map[bin_conf_region]
            bin_label = label[bin_conf_region]

            # Calculate the accuracy and mean confidence for the island.
            avg_bin_metric = measure_dict[measure](bin_preds, bin_label)
            avg_bin_conf = bin_confs.mean()
            bin_freqs = avg_bin_metric

            # Calculate the average calibration error for the regions in the bin.
            ace_per_bin[bin_idx] = (avg_bin_metric - avg_bin_conf).abs()
            avg_freq_per_bin[bin_idx] = bin_freqs.mean()
            bin_amounts[bin_idx] = bin_conf_region.sum() 

            # Store the frequencies and confidences for each bin.
            freqs_per_bin[bin_idx] = bin_freqs 
            confs_per_bin[bin_idx] = bin_confs 
        else:
            freqs_per_bin[bin_idx] = torch.Tensor([]) 
            confs_per_bin[bin_idx] = torch.Tensor([]) 

    # Finally, get the ReCE score.
    ace_score = reduce_scores(
        score_per_bin=ace_per_bin, 
        amounts_per_bin=bin_amounts, 
        weighting=weighting
        )

    return {
        "score": ace_score,
        "bins": conf_bins, 
        "bin_widths": conf_bin_widths, 
        "bin_amounts": bin_amounts,
        "score_per_bin": ace_per_bin, 
        "avg_freq_per_bin": avg_freq_per_bin,
        "freqs_per_bin": freqs_per_bin, 
        "confs_per_bin": confs_per_bin, 
    }


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ReCE(
    num_bins: int,
    conf_map: torch.Tensor,
    pred_map: torch.Tensor,
    label: torch.Tensor,
    measure: str = "Frequency",
    weighting: str = "proportional",
    include_background: bool = False,
    min_confidence: float = 0.01,
    threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the ReCE: Region-wise Calibration Error
    """
    assert len(conf_map.shape) == 2 and conf_map.shape == label.shape, f"conf_map and label must be 2D tensors of the same shape. Got {conf_map.shape} and {label.shape}."

    # Eliminate the super small predictions to get a better picture.
    label = label[conf_map >= min_confidence]
    pred_map = pred_map[conf_map >= min_confidence]
    conf_map = conf_map[conf_map >= min_confidence]

    # If you don't want to include background pixels, remove them.
    if not include_background:
        label = label[conf_map >= threshold]
        pred_map = pred_map[conf_map >= threshold]
        conf_map = conf_map[conf_map >= threshold]

    # Create the confidence bins.    
    conf_bins, conf_bin_widths = get_bins(
        metric="ReCE", 
        include_background=include_background, 
        threshold=threshold, 
        num_bins=num_bins
        )

    # Keep track of different things for each bin, dicts are
    # for cummulation over multiple images, tensors are for one image.
    freqs_per_bin = {}
    confs_per_bin = {}

    rece_per_bin = torch.zeros(num_bins)
    avg_freq_per_bin = torch.zeros(num_bins)
    bin_amounts = torch.zeros(num_bins)

    # Go through each bin, starting at the back so that we don't have to run connected components
    for bin_idx, conf_bin in enumerate(conf_bins):
        
        # Get the region of image corresponding to the confidence
        bin_conf_region = get_conf_region(bin_idx, conf_bin, conf_bin_widths, conf_map)

        # If there are some pixels in this confidence bin.
        if bin_conf_region.sum() != 0:
            # If we are not the last bin, get the connected components.
            conf_islands = get_connected_components(bin_conf_region)

            # Iterate through each island, and get the measure for each island.
            num_islands = len(conf_islands)
            region_metric_scores = torch.zeros(num_islands)
            region_conf_scores = torch.zeros(num_islands)

            # Iterate through each island, and get the measure for each island.
            for isl_idx, island in enumerate(conf_islands):
                # Get the island primitives
                region_conf_map = conf_map[island]                
                region_pred_map = pred_map[island]
                region_label = label[island]

                # Calculate the accuracy and mean confidence for the island.
                region_metric_scores[isl_idx] = measure_dict[measure](region_pred_map, region_label)
                region_conf_scores[isl_idx] = region_conf_map.mean()
            
            # Get the accumulate metrics from all the islands
            avg_region_metric = region_metric_scores.mean()
            avg_region_conf = region_conf_scores.mean()
            
            # Calculate the average calibration error for the regions in the bin.
            rece_per_bin[bin_idx] = (avg_region_metric - avg_region_conf).abs()
            avg_freq_per_bin[bin_idx] = avg_region_metric 
            bin_amounts[bin_idx] = bin_conf_region.sum() 

            # Store the frequencies and confidences for each bin.
            freqs_per_bin[bin_idx] = region_metric_scores 
            confs_per_bin[bin_idx] = region_conf_scores 
        else:
            freqs_per_bin[bin_idx] = torch.Tensor([]) 
            confs_per_bin[bin_idx] = torch.Tensor([]) 

    # Finally, get the ReCE score.
    rece_score = reduce_scores(
        score_per_bin=rece_per_bin, 
        amounts_per_bin=bin_amounts, 
        weighting=weighting
        )
    
    return {
        "score": rece_score,
        "bins": conf_bins, 
        "bin_widths": conf_bin_widths, 
        "bin_amounts": bin_amounts,
        "score_per_bin": rece_per_bin, 
        "avg_freq_per_bin": avg_freq_per_bin,
        "freqs_per_bin": freqs_per_bin, 
        "confs_per_bin": confs_per_bin,
    }
    
