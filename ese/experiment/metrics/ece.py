# local imports
from .local_ps import (
    bin_stats, 
    label_bin_stats
)
from .global_ps import (
    global_bin_stats, 
    global_label_bin_stats
)
from .utils import (
    reduce_bin_errors, 
    get_edge_pixels, 
    get_edge_pixel_preds,
    cal_input_check 
)
# misc imports
import torch
from typing import Dict, Tuple, Optional
from pydantic import validate_arguments
# ionpy imports
from ionpy.loss.util import _loss_module_from_func
from ionpy.util.meter import Meter


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ece_loss(
    y_pred: torch.Tensor = None, 
    y_true: torch.Tensor = None,
    pixel_meters_dict: Dict[tuple, Meter] = None,
    num_bins: int = 10,
    square_diff: bool = False,
    conf_interval: Tuple[float, float] = (0.0, 1.0),
    stats_info_dict: Optional[dict] = {},
    from_logits: bool = False,
    return_dict: bool = False,
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    # Verify input.
    use_global = cal_input_check(y_pred, y_true, pixel_meters_dict)
    # Get the statistics either from images or pixel meter dict.
    if use_global:
        cal_info = global_bin_stats(
            pixel_meters_dict=pixel_meters_dict,
            square_diff=square_diff,
            weighted=False,
            ignore_index=ignore_index
        )
    else: 
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
    # Return the calibration information.
    assert 0 <= cal_info['cal_error'] <= 1,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    # Return the calibration information.
    if return_dict:
        return cal_info
    else:
        return cal_info['cal_error']


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def tl_ece_loss(
    y_pred: torch.Tensor = None, 
    y_true: torch.Tensor = None,
    pixel_meters_dict: Dict[tuple, Meter] = None,
    num_bins: int = 10,
    square_diff: bool = False,
    conf_interval: Tuple[float, float] = (0.0, 1.0),
    stats_info_dict: Optional[dict] = {},
    from_logits: bool = False,
    return_dict: bool = False,
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    # Verify input.
    use_global = cal_input_check(y_pred, y_true, pixel_meters_dict)
    # Get the statistics either from images or pixel meter dict.
    if use_global:
        cal_info = global_label_bin_stats(
            pixel_meters_dict=pixel_meters_dict,
            square_diff=square_diff,
            weighted=False,
            ignore_index=ignore_index
        )
    else: 
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
    # Return the calibration information.
    if return_dict:
        return cal_info
    else:
        return cal_info['cal_error']


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def cw_ece_loss(
    y_pred: torch.Tensor = None, 
    y_true: torch.Tensor = None,
    pixel_meters_dict: Dict[tuple, Meter] = None,
    num_bins: int = 10,
    square_diff: bool = False,
    conf_interval: Tuple[float, float] = (0.0, 1.0),
    stats_info_dict: Optional[dict] = {},
    from_logits: bool = False,
    return_dict: bool = False,
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the LoMS.
    """
    # Verify input.
    use_global = cal_input_check(y_pred, y_true, pixel_meters_dict)
    # Get the statistics either from images or pixel meter dict.
    if use_global:
        cal_info = global_label_bin_stats(
            pixel_meters_dict=pixel_meters_dict,
            square_diff=square_diff,
            weighted=False,
            ignore_index=ignore_index
        )
    else:
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
    # Return the calibration information.
    if return_dict:
        return cal_info
    else:
        return cal_info['cal_error']


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def edge_ece_loss(
    y_pred: torch.Tensor = None, 
    y_true: torch.Tensor = None,
    pixel_meters_dict: Dict[tuple, Meter] = None,
    num_bins: int = 10,
    square_diff: bool = False,
    conf_interval: Tuple[float, float] = (0.0, 1.0),
    stats_info_dict: Optional[dict] = {},
    from_logits: bool = False,
    return_dict: bool = False,
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    # Verify input.
    use_global = cal_input_check(y_pred, y_true, pixel_meters_dict)
    # Define the config for all common factors.
    ece_config = {
        "num_bins": num_bins,
        "conf_interval": conf_interval,
        "square_diff": square_diff,
        "stats_info_dict": stats_info_dict["edge_info"] if "edge_info" in stats_info_dict else {},
        "from_logits": from_logits,
        "return_dict": return_dict,
        "ignore_index": ignore_index
    }
    # Get the statistics either from images or pixel meter dict.
    if use_global:
        edge_pixel_preds = get_edge_pixel_preds(pixel_meters_dict)
        ece_config["pixel_preds_dict"] = edge_pixel_preds
    else:
        y_edge_pred, y_edge_true = get_edge_pixels(
            y_pred=y_pred, 
            y_true=y_true,
            image_info_dict=stats_info_dict["image_info"] if "image_info" in stats_info_dict else {}
        )
        ece_config["y_pred"] = y_edge_pred
        ece_config["y_true"] = y_edge_true
    # Return the calibration information
    return ece_loss(**ece_config)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def etl_ece_loss(
    y_pred: torch.Tensor = None, 
    y_true: torch.Tensor = None,
    pixel_meters_dict: Dict[tuple, Meter] = None,
    num_bins: int = 10,
    square_diff: bool = False,
    conf_interval: Tuple[float, float] = (0.0, 1.0),
    stats_info_dict: Optional[dict] = {},
    from_logits: bool = False,
    return_dict: bool = False,
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    # Verify input.
    use_global = cal_input_check(y_pred, y_true, pixel_meters_dict)
    # Define the config for all common factors.
    tl_ece_config = {
        "num_bins": num_bins,
        "conf_interval": conf_interval,
        "square_diff": square_diff,
        "stats_info_dict": stats_info_dict["edge_info"] if "edge_info" in stats_info_dict else {},
        "from_logits": from_logits,
        "return_dict": return_dict,
        "ignore_index": ignore_index
    }
    # Get the statistics either from images or pixel meter dict.
    if use_global:
        edge_pixel_preds = get_edge_pixel_preds(pixel_meters_dict)
        tl_ece_config["pixel_preds_dict"] = edge_pixel_preds
    else:
        y_edge_pred, y_edge_true = get_edge_pixels(
            y_pred=y_pred, 
            y_true=y_true,
            image_info_dict=stats_info_dict["image_info"] if "image_info" in stats_info_dict else {}
        )
        tl_ece_config["y_pred"] = y_edge_pred
        tl_ece_config["y_true"] = y_edge_true
    # Return the calibration information
    return tl_ece_loss(**tl_ece_config)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ecw_ece_loss(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor,
    pixel_meters_dict: Dict[tuple, Meter] = None,
    num_bins: int = 10,
    square_diff: bool = False,
    conf_interval: Tuple[float, float] = (0.0, 1.0),
    stats_info_dict: Optional[dict] = {},
    from_logits: bool = False,
    return_dict: bool = False,
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    # Verify input.
    use_global = cal_input_check(y_pred, y_true, pixel_meters_dict)
    # Define the config for all common factors.
    cw_ece_config = {
        "num_bins": num_bins,
        "conf_interval": conf_interval,
        "square_diff": square_diff,
        "stats_info_dict": stats_info_dict["edge_info"] if "edge_info" in stats_info_dict else {},
        "from_logits": from_logits,
        "return_dict": return_dict,
        "ignore_index": ignore_index
    }
    # Get the statistics either from images or pixel meter dict.
    if use_global:
        edge_pixel_preds = get_edge_pixel_preds(pixel_meters_dict)
        cw_ece_config["pixel_meters_dict"] = edge_pixel_preds
    else:
        y_edge_pred, y_edge_true = get_edge_pixels(
            y_pred=y_pred, 
            y_true=y_true,
            image_info_dict=stats_info_dict["image_info"] if "image_info" in stats_info_dict else {}
        )
        cw_ece_config["y_pred"] = y_edge_pred
        cw_ece_config["y_true"] = y_edge_true
    # Return the calibration information
    return cw_ece_loss(**cw_ece_config)


# Loss modules
ECE = _loss_module_from_func("ECE", ece_loss)
TL_ECE = _loss_module_from_func("TL_ECE", tl_ece_loss)
CW_ECE = _loss_module_from_func("CW_ECE", cw_ece_loss)
# Edge loss modules
Edge_ECE = _loss_module_from_func("Edge_ECE", edge_ece_loss)
ETL_ECE = _loss_module_from_func("ETL_ECE", etl_ece_loss)
ECW_ECE = _loss_module_from_func("ECW_ECE", ecw_ece_loss)