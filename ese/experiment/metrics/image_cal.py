# local imports
from .pix_stats import bin_stats, label_bin_stats
from .utils import get_bins, reduce_bin_errors
# misc imports
import numpy as np
import torch
from typing import Tuple, List, Optional
from pydantic import validate_arguments


def round_tensor(tensor, num_decimals=3):
    base_10 = 10 ** num_decimals
    r_tensor = torch.round(tensor * base_10) / base_10 
    return r_tensor


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ECE(
    num_bins: int,
    conf_map: torch.Tensor, 
    pred_map: torch.Tensor, 
    label_map: torch.Tensor,
    conf_interval: Tuple[float, float],
    weighting: str = "proportional",
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    assert len(conf_map.shape) == 2 and conf_map.shape == label_map.shape,\
        f"conf_map and label_map must be 2D tensors of the same shape. Got {conf_map.shape} and {label_map.shape}."
    # Create the confidence bins.    
    conf_bins, conf_bin_widths = get_bins(
        num_bins=num_bins, 
        start=conf_interval[0], 
        end=conf_interval[1]
        )
    # Keep track of different things for each bin.
    cal_info = bin_stats(
        num_bins=num_bins,
        conf_bins=conf_bins,
        conf_bin_widths=conf_bin_widths,
        conf_map=conf_map,
        pred_map=pred_map,
        label_map=label_map,
        ignore_index=ignore_index
    )
    # Finally, get the calibration score.
    cal_info['cal_error'] = reduce_bin_errors(
        error_per_bin=cal_info["bin_cal_errors"], 
        amounts_per_bin=cal_info["bin_amounts"], 
        weighting=weighting
        )
    # Return the calibration information
    assert 1 >= cal_info['cal_error'] >= 0,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def TL_ECE(
    num_bins: int,
    conf_map: torch.Tensor, 
    pred_map: torch.Tensor, 
    label_map: torch.Tensor,
    conf_interval: Tuple[float, float],
    weighting: str = "proportional",
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    assert len(conf_map.shape) == 2 and conf_map.shape == label_map.shape,\
        f"conf_map and label_map must be 2D tensors of the same shape. Got {conf_map.shape} and {label_map.shape}."
    # Create the confidence bins.    
    conf_bins, conf_bin_widths = get_bins(
        num_bins=num_bins, 
        start=conf_interval[0], 
        end=conf_interval[1]
        )
    # Keep track of different things for each bin.
    cal_info = label_bin_stats(
        num_bins=num_bins,
        conf_bins=conf_bins,
        conf_bin_widths=conf_bin_widths,
        conf_map=conf_map,
        pred_map=pred_map,
        label_map=label_map,
        ignore_index=ignore_index
    )
    # Finally, get the ECE score.
    num_labels, _ = cal_info["bin_cal_errors"].shape
    w_ece = torch.zeros(num_labels)
    # Iterate through each label and calculate the weighted ece.
    for lab_idx in range(num_labels):
        ece = reduce_bin_errors(
            error_per_bin=cal_info['bin_cal_errors'][lab_idx], 
            amounts_per_bin=cal_info['bin_amounts'][lab_idx], 
            weighting=weighting
            )
        w_ece[lab_idx] = ece * cal_info['bin_amounts'][lab_idx].sum()
    # Finally, get the calibration score.
    cal_info['cal_error'] =  (w_ece.sum() / cal_info['bin_amounts'].sum()).item()
    # Return the calibration information
    assert 1 >= cal_info['cal_error'] >= 0,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def CW_ECE(
    num_bins: int,
    conf_map: torch.Tensor, 
    pred_map: torch.Tensor, 
    label_map: torch.Tensor,
    conf_interval: Tuple[float, float],
    weighting: str = "proportional",
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    assert len(conf_map.shape) == 2 and conf_map.shape == label_map.shape,\
        f"conf_map and label_map must be 2D tensors of the same shape. Got {conf_map.shape} and {label_map.shape}."
    # Create the confidence bins.    
    conf_bins, conf_bin_widths = get_bins(
        num_bins=num_bins, 
        start=conf_interval[0], 
        end=conf_interval[1]
        )
    # Keep track of different things for each bin.
    cal_info = label_bin_stats(
        num_bins=num_bins,
        conf_bins=conf_bins,
        conf_bin_widths=conf_bin_widths,
        conf_map=conf_map,
        pred_map=pred_map,
        label_map=label_map,
        ignore_index=ignore_index
    )
    # Finally, get the ECE score.
    num_labels, _ = cal_info["bin_cal_errors"].shape
    w_ece = torch.zeros(num_labels)
    # Iterate through each label, calculating ECE
    for lab_idx in range(num_labels):
        w_ece[lab_idx] = reduce_bin_errors(
            error_per_bin=cal_info["bin_cal_errors"][lab_idx], 
            amounts_per_bin=cal_info["bin_amounts"][lab_idx], 
            weighting=weighting
            )
    # Finally, get the calibration score.
    cal_info['cal_error'] = (w_ece.sum() / num_labels).item()
    # Return the calibration information
    assert 1 >= cal_info['cal_error'] >= 0,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def SUME(
    num_bins: int,
    conf_map: torch.Tensor, 
    pred_map: torch.Tensor, 
    label_map: torch.Tensor,
    conf_interval: Tuple[float, float],
    neighborhood_width: int = 3,
    weighting: str = "proportional",
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the TENCE: Top-Label Expected Neighborhood-conditioned Calibration Error.
    """
    assert len(conf_map.shape) == 2 and conf_map.shape == label_map.shape,\
        f"conf_map and label_map must be 2D tensors of the same shape. Got {conf_map.shape} and {label_map.shape}."
    # Create the confidence bins.    
    conf_bins, conf_bin_widths = get_bins(
        num_bins=num_bins, 
        start=conf_interval[0], 
        end=conf_interval[1]
        )
    # Keep track of different things for each bin.
    cal_info = bin_stats(
        num_bins=num_bins,
        conf_bins=conf_bins,
        conf_bin_widths=conf_bin_widths,
        conf_map=conf_map,
        pred_map=pred_map,
        label_map=label_map,
        neighborhood_width=neighborhood_width,
        uni_w_attributes=["labels", "neighbors"],
        ignore_index=ignore_index
    )
    cal_info['cal_error'] = reduce_bin_errors(
        error_per_bin=cal_info["bin_cal_errors"], 
        amounts_per_bin=cal_info["bin_amounts"], 
        weighting=weighting
        )
    assert 1 >= cal_info['cal_error'] >= 0,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def TL_SUME(
    num_bins: int,
    conf_map: torch.Tensor, 
    pred_map: torch.Tensor, 
    label_map: torch.Tensor,
    conf_interval: Tuple[float, float],
    neighborhood_width: int = 3,
    weighting: str = "proportional",
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the TENCE: Top-Label Expected Neighborhood-conditioned Calibration Error.
    """
    assert len(conf_map.shape) == 2 and conf_map.shape == label_map.shape,\
        f"conf_map and label_map must be 2D tensors of the same shape. Got {conf_map.shape} and {label_map.shape}."
    # Create the confidence bins.    
    conf_bins, conf_bin_widths = get_bins(
        num_bins=num_bins, 
        start=conf_interval[0], 
        end=conf_interval[1]
        )
    # Keep track of different things for each bin.
    cal_info = label_bin_stats(
        num_bins=num_bins,
        conf_bins=conf_bins,
        conf_bin_widths=conf_bin_widths,
        conf_map=conf_map,
        pred_map=pred_map,
        label_map=label_map,
        neighborhood_width=neighborhood_width,
        uni_w_attributes=["neighbors"],
        ignore_index=ignore_index
    )
    # Finally, get the ECE score.
    num_labels, _ = cal_info["bin_cal_errors"].shape
    w_ece = torch.zeros(num_labels)
    # Iterate through each label and calculate the weighted ece.
    for lab_idx in range(num_labels):
        ece = reduce_bin_errors(
            error_per_bin=cal_info['bin_cal_errors'][lab_idx], 
            amounts_per_bin=cal_info['bin_amounts'][lab_idx], 
            weighting=weighting
            )
        # Calculate the empirical prob of the label.
        prob_l = cal_info['bin_amounts'][lab_idx].sum() / cal_info['bin_amounts'].sum()
        # Weight the ECE by the prob of the label.
        w_ece[lab_idx] = prob_l * ece
    # Finally, get the calibration score.
    cal_info['cal_error'] =  w_ece.sum().item()
    assert 1 >= cal_info['cal_error'] >= 0,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def CW_SUME(
    num_bins: int,
    conf_map: torch.Tensor, 
    pred_map: torch.Tensor, 
    label_map: torch.Tensor,
    conf_interval: Tuple[float, float],
    neighborhood_width: int = 3,
    weighting: str = "proportional",
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the TENCE: Top-Label Expected Neighborhood-conditioned Calibration Error.
    """
    assert len(conf_map.shape) == 2 and conf_map.shape == label_map.shape,\
        f"conf_map and label_map must be 2D tensors of the same shape. Got {conf_map.shape} and {label_map.shape}."
    # Create the confidence bins.    
    conf_bins, conf_bin_widths = get_bins(
        num_bins=num_bins, 
        start=conf_interval[0], 
        end=conf_interval[1]
        )
    # Keep track of different things for each bin.
    cal_info = label_bin_stats(
        num_bins=num_bins,
        conf_bins=conf_bins,
        conf_bin_widths=conf_bin_widths,
        conf_map=conf_map,
        pred_map=pred_map,
        label_map=label_map,
        neighborhood_width=neighborhood_width,
        uni_w_attributes=["neighbors"],
        ignore_index=ignore_index
    )
    # Finally, get the ECE score.
    num_labels, _ = cal_info["bin_cal_errors"].shape
    w_ece = torch.zeros(num_labels)
    # Iterate through each label, calculating ECE
    for lab_idx in range(num_labels):
        ece = reduce_bin_errors(
            error_per_bin=cal_info["bin_cal_errors"][lab_idx], 
            amounts_per_bin=cal_info["bin_amounts"][lab_idx], 
            weighting=weighting
            )
        w_ece[lab_idx] = ece
    # Finally, get the calibration score.
    cal_info['cal_error'] = (w_ece.sum() / num_labels).item()
    # Return the calibration information
    assert 1 >= cal_info['cal_error'] >= 0,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    return cal_info