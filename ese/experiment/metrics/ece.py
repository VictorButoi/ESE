# local imports for:
# - pixel statistics
from .local_ps import (
    bin_stats, 
    label_bin_stats
)
# - global statistics
from .global_ps import (
    global_bin_stats, 
    global_label_bin_stats
)
# - misc utils.
from .utils import reduce_bin_errors
# torch imports
import torch
from torch import Tensor
# misc imports
from pydantic import validate_arguments
from typing import Dict, Optional, Union, List
# ionpy imports
from ionpy.util.meter import Meter
from ionpy.loss.util import _loss_module_from_func


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def cal_input_check(
    y_pred: Optional[Tensor] = None,
    y_true: Optional[Tensor] = None,
    pixel_preds_dict: Optional[dict] = None
):
    use_local_funcs = (y_pred is not None and y_true is not None)
    use_global_funcs = (pixel_preds_dict is not None)
    # xor images_defined pixel_preds_defined
    assert use_global_funcs ^ use_local_funcs,\
        "Exactly one of (y_pred and y_true) or pixel_preds_dict must be defined,"\
             + " but y_pred defined = {}, y_true defined = {}, pixel_preds_dict defined = {}.".format(\
            y_pred is not None, y_true is not None, pixel_preds_dict is not None)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ece_loss(
    num_bins: int,
    y_pred: Tensor = None, 
    y_true: Tensor = None,
    pixel_meters_dict: Dict[tuple, Meter] = None,
    neighborhood_width: int = 3,
    edge_only: bool = False,
    square_diff: bool = False,
    from_logits: bool = False,
    return_dict: bool = False,
    stats_info_dict: Optional[dict] = {},
    conf_interval: List[float] = [0.0, 1.0],
    ignore_index: Optional[int] = None
    ) -> Union[dict, Tensor]:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    # Verify input.
    cal_input_check(y_pred, y_true, pixel_meters_dict)
    # Get the statistics either from images or pixel meter dict.
    if pixel_meters_dict is not None:
        cal_info = global_bin_stats(
            pixel_meters_dict=pixel_meters_dict,
            square_diff=square_diff,
            edge_only=edge_only,
            neighborhood_width=neighborhood_width,
            ignore_index=ignore_index
        )
    else: 
        cal_info = bin_stats(
            y_pred=y_pred,
            y_true=y_true,
            num_bins=num_bins,
            conf_interval=conf_interval,
            square_diff=square_diff,
            neighborhood_width=neighborhood_width,
            edge_only=edge_only,
            stats_info_dict=stats_info_dict,
            from_logits=from_logits,
            ignore_index=ignore_index
        )

    # Finally, get the calibration score.
    cal_info['cal_error'] = reduce_bin_errors(
        error_per_bin=cal_info["bin_cal_errors"], 
        amounts_per_bin=cal_info["bin_amounts"]
        )
    # Return the calibration information.
    assert 0.0 <= cal_info['cal_error'] <= 1.0,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    # Return the calibration information.
    if return_dict:
        return cal_info
    else:
        return cal_info['cal_error']


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def tl_ece_loss(
    num_bins: int,
    y_pred: Tensor = None, 
    y_true: Tensor = None,
    pixel_meters_dict: Dict[tuple, Meter] = None,
    neighborhood_width: int = 3,
    edge_only: bool = False,
    square_diff: bool = False,
    from_logits: bool = False,
    return_dict: bool = False,
    stats_info_dict: dict = {},
    conf_interval: List[float] = [0.0, 1.0],
    ignore_index: Optional[int] = None
    ) -> Union[dict, Tensor]:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    # Verify input.
    cal_input_check(y_pred, y_true, pixel_meters_dict)
    # Get the statistics either from images or pixel meter dict.
    if pixel_meters_dict is not None:
        cal_info = global_label_bin_stats(
            pixel_meters_dict=pixel_meters_dict,
            top_label=True,
            square_diff=square_diff,
            neighborhood_width=neighborhood_width,
            edge_only=edge_only,
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
            neighborhood_width=neighborhood_width,
            edge_only=edge_only,
            stats_info_dict=stats_info_dict,
            from_logits=from_logits,
            ignore_index=ignore_index
        )
    # Finally, get the ECE score.
    total_num_samples = cal_info['bin_amounts'].sum()
    if total_num_samples == 0:
        cal_info['cal_error'] = torch.tensor(0.0)
    else:
        L, _ = cal_info["bin_cal_errors"].shape
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

    # Return the calibration information.
    assert 0.0 <= cal_info['cal_error'] <= 1.0,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    # Return the calibration information.
    if return_dict:
        return cal_info
    else:
        return cal_info['cal_error']


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def cw_ece_loss(
    num_bins: int,
    y_pred: Tensor = None, 
    y_true: Tensor = None,
    pixel_meters_dict: Dict[tuple, Meter] = None,
    neighborhood_width: int = 3,
    edge_only: bool = False,
    square_diff: bool = False,
    from_logits: bool = False,
    return_dict: bool = False,
    stats_info_dict: dict = {},
    conf_interval: List[float] = [0.0, 1.0],
    ignore_index: Optional[int] = None
    ) -> Union[dict, Tensor]:
    """
    Calculates the LoMS.
    """
    # Verify input.
    cal_input_check(y_pred, y_true, pixel_meters_dict)
    # Get the statistics either from images or pixel meter dict.
    if pixel_meters_dict is not None:
        cal_info = global_label_bin_stats(
            pixel_meters_dict=pixel_meters_dict,
            top_label=False,
            square_diff=square_diff,
            neighborhood_width=neighborhood_width,
            edge_only=edge_only,
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
            neighborhood_width=neighborhood_width,
            edge_only=edge_only,
            stats_info_dict=stats_info_dict,
            from_logits=from_logits,
            ignore_index=ignore_index
        )

    # Finally, get the ECE score.
    total_num_samples = cal_info['bin_amounts'].sum()
    if total_num_samples == 0:
        cal_info['cal_error'] = torch.tensor(0.0)
    else:
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

    # Return the calibration information.
    assert 0.0 <= cal_info['cal_error'] <= 1.0,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    # Return the calibration information.
    if return_dict:
        return cal_info
    else:
        return cal_info['cal_error']


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def edge_ece_loss(
    y_pred: Tensor = None, 
    y_true: Tensor = None,
    **kwargs
    ) -> Union[dict, Tensor]:
    """
    Calculates the Expected Semantic Error (ECE) of just the edges.
    """
    kwargs["y_pred"] = y_pred
    kwargs["y_true"] = y_true
    kwargs["edge_only"] = True
    # Return the calibration information
    return ece_loss(**kwargs)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def etl_ece_loss(
    y_pred: Tensor = None, 
    y_true: Tensor = None,
    **kwargs
    ) -> Union[dict, Tensor]:
    """
    Calculates the Top-label Expected Semantic Error (TL-ECE) for a predicted label map.
    """
    kwargs["y_pred"] = y_pred
    kwargs["y_true"] = y_true
    kwargs["edge_only"] = True
    # Return the calibration information
    return tl_ece_loss(**kwargs)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ecw_ece_loss(
    y_pred: Tensor = None, 
    y_true: Tensor = None,
    **kwargs
    ) -> Union[dict, Tensor]:
    """
    Calculates the Class-wise Expected Semantic Error (CW-ECE) for a predicted label map.
    """
    kwargs["y_pred"] = y_pred
    kwargs["y_true"] = y_true
    kwargs["edge_only"] = True
    # Return the calibration information
    return cw_ece_loss(**kwargs)


# Loss modules
ECE = _loss_module_from_func("ECE", ece_loss)
TL_ECE = _loss_module_from_func("TL_ECE", tl_ece_loss)
CW_ECE = _loss_module_from_func("CW_ECE", cw_ece_loss)

# Edge loss modules
Edge_ECE = _loss_module_from_func("Edge_ECE", edge_ece_loss)
ETL_ECE = _loss_module_from_func("ETL_ECE", etl_ece_loss)
ECW_ECE = _loss_module_from_func("ECW_ECE", ecw_ece_loss)