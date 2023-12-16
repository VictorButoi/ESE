from .pix_stats import bin_stats, label_bin_stats
from .utils import reduce_bin_errors, get_edge_map
# misc imports
import torch
from typing import Tuple, Optional
from pydantic import validate_arguments


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ECE(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor,
    num_bins: int,
    conf_interval: Tuple[float, float],
    square_diff: bool,
    stats_info_dict: Optional[dict] = {},
    from_logits: bool = False,
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    # Keep track of different things for each bin.
    cal_info = bin_stats(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        square_diff=square_diff,
        stats_info_dict=stats_info_dict["image_info"] if "image_info" in stats_info_dict else {},
        from_logits=from_logits,
        ignore_index=ignore_index
    )
    # Finally, get the calibration score.
    cal_info['cal_error'] = reduce_bin_errors(
        error_per_bin=cal_info["bin_cal_errors"], 
        amounts_per_bin=cal_info["bin_amounts"], 
        )
    # Return the calibration information
    assert 0 <= cal_info['cal_error'] <= 1,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def TL_ECE(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor,
    num_bins: int,
    conf_interval: Tuple[float, float],
    square_diff: bool,
    stats_info_dict: Optional[dict] = {},
    from_logits: bool = False,
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    # Keep track of different things for each bin.
    cal_info = label_bin_stats(
        y_pred=y_pred,
        y_true=y_true,
        top_label=True,
        num_bins=num_bins,
        conf_interval=conf_interval,
        square_diff=square_diff,
        stats_info_dict=stats_info_dict["image_info"] if "image_info" in stats_info_dict else {},
        from_logits=from_logits,
        ignore_index=ignore_index
    )
    # Finally, get the ECE score.
    L, _ = cal_info["bin_cal_errors"].shape
    total_num_samples = cal_info['bin_amounts'].sum()
    ece_per_lab = torch.zeros(L)
    # Iterate through each label and calculate the weighted ece.
    for lab_idx in range(L):
        lab_ece = reduce_bin_errors(
            error_per_bin=cal_info['bin_cal_errors'][lab_idx], 
            amounts_per_bin=cal_info['bin_amounts'][lab_idx], 
            )
        lab_prob = cal_info['bin_amounts'][lab_idx].sum() / total_num_samples 
        # Weight the ECE by the prob of the label.
        ece_per_lab[lab_idx] = lab_prob * lab_ece
    # Finally, get the calibration score.
    cal_info['cal_error'] =  ece_per_lab.sum()
    # Return the calibration information
    assert 0 <= cal_info['cal_error'] <= 1,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def CW_ECE(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor,
    num_bins: int,
    conf_interval: Tuple[float, float],
    square_diff: bool,
    stats_info_dict: Optional[dict] = {},
    from_logits: bool = False,
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the LoMS.
    """
    # Keep track of different things for each bin.
    cal_info = label_bin_stats(
        y_pred=y_pred,
        y_true=y_true,
        top_label=False,
        num_bins=num_bins,
        conf_interval=conf_interval,
        square_diff=square_diff,
        stats_info_dict=stats_info_dict["image_info"] if "image_info" in stats_info_dict else {},
        from_logits=from_logits,
        ignore_index=ignore_index
    )
    # Finally, get the ECE score.
    L, _ = cal_info["bin_cal_errors"].shape
    ece_per_lab = torch.zeros(L)
    # Iterate through each label, calculating ECE
    for lab_idx in range(L):
        ece_per_lab[lab_idx] = reduce_bin_errors(
            error_per_bin=cal_info["bin_cal_errors"][lab_idx], 
            amounts_per_bin=cal_info["bin_amounts"][lab_idx], 
            )
    # Finally, get the calibration score.
    cal_info['cal_error'] = ece_per_lab.mean()
    # Return the calibration information
    assert 0 <= cal_info['cal_error'] <= 1,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    return cal_info


def get_edge_pixels(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        image_info_dict: dict
        ) -> torch.Tensor:
    """
    Returns the edge pixels of the ground truth label map.
    """
    # Get the edge map.
    if "true_matching_neighbors_map" in image_info_dict:
        y_true_edge_map = (image_info_dict["true_matching_neighbors_map"] < 8)
    else:
        y_true_squeezed = y_true.squeeze()
        y_true_edge_map = get_edge_map(y_true_squeezed)
    # Get the edge regions of both the prediction and the ground truth.
    y_pred_e_reg = y_pred[..., y_true_edge_map]
    y_true_e_reg = y_true[..., y_true_edge_map]
    # Add a height dim.
    y_edge_pred = y_pred_e_reg.unsqueeze(-2)
    y_edge_true= y_true_e_reg.unsqueeze(-2)
    # Return the edge-ified values.
    return y_edge_pred, y_edge_true


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def Edge_ECE(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor,
    num_bins: int,
    conf_interval: Tuple[float, float],
    square_diff: bool,
    stats_info_dict: Optional[dict] = {},
    from_logits: bool = False,
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    # Get the edge pixels.
    y_edge_pred, y_edge_true = get_edge_pixels(
        y_pred=y_pred, 
        y_true=y_true,
        image_info_dict=stats_info_dict["image_info"] if "image_info" in stats_info_dict else {}
    )
    # Return the calibration information
    return ECE(
        y_pred=y_edge_pred,
        y_true=y_edge_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        square_diff=square_diff,
        stats_info_dict=stats_info_dict["edge_info"] if "edge_info" in stats_info_dict else {},
        from_logits=from_logits,
        ignore_index=ignore_index
    ) 


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def TL_EdgeCE(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor,
    num_bins: int,
    conf_interval: Tuple[float, float],
    square_diff: bool,
    stats_info_dict: Optional[dict] = {},
    from_logits: bool = False,
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    # Get the edge pixels.
    y_edge_pred, y_edge_true = get_edge_pixels(
        y_pred=y_pred, 
        y_true=y_true,
        image_info_dict=stats_info_dict["image_info"] if "image_info" in stats_info_dict else {}
    )
    # Return the calibration information
    return TL_ECE(
        y_pred=y_edge_pred,
        y_true=y_edge_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        square_diff=square_diff,
        stats_info_dict=stats_info_dict["edge_info"] if "edge_info" in stats_info_dict else {},
        from_logits=from_logits,
        ignore_index=ignore_index
    ) 


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ECW_ECE(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor,
    num_bins: int,
    conf_interval: Tuple[float, float],
    square_diff: bool,
    stats_info_dict: Optional[dict] = {},
    from_logits: bool = False,
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    # Get the edge pixels.
    y_edge_pred, y_edge_true = get_edge_pixels(
        y_pred=y_pred, 
        y_true=y_true,
        image_info_dict=stats_info_dict["image_info"] if "image_info" in stats_info_dict else {}
    )
    # Return the calibration information
    return CW_ECE(
        y_pred=y_edge_pred,
        y_true=y_edge_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        square_diff=square_diff,
        stats_info_dict=stats_info_dict["edge_info"] if "edge_info" in stats_info_dict else {},
        from_logits=from_logits,
        ignore_index=ignore_index
    ) 