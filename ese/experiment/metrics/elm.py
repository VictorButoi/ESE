# get the processing function.
from .local_ps import (
    neighbors_bin_stats, 
    label_neighbors_bin_stats
)
from .global_ps import (
    global_neighbors_bin_stats,
    global_label_neighbors_bin_stats
)
from .utils import reduce_bin_errors
# torch imports
import torch
from torch import Tensor
# misc imports
from pydantic import validate_arguments
from typing import Dict, Tuple, Optional, Union
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
def elm_loss(
    y_pred: Tensor = None, 
    y_true: Tensor = None,
    pixel_meters_dict: Dict[tuple, Meter] = None,
    num_bins: int = 10,
    neighborhood_width: int = 3,
    square_diff: bool = False,
    from_logits: bool = False,
    return_dict: bool = False,
    conf_interval: Tuple[float, float] = (0.0, 1.0),
    stats_info_dict: Optional[dict] = {},
    ignore_index: Optional[int] = None
    ) -> Union[dict, Tensor]:
    """
    Calculates the TENCE: Top-Label Expected Neighborhood-conditioned Calibration Error.
    """
    # Verify input.
    cal_input_check(y_pred, y_true, pixel_meters_dict)
    if pixel_meters_dict is not None:
        cal_info = global_neighbors_bin_stats(
            pixel_meters_dict=pixel_meters_dict,
            square_diff=square_diff,
            ignore_index=ignore_index
        )
    else:
        cal_info = neighbors_bin_stats(
            y_pred=y_pred,
            y_true=y_true,
            num_bins=num_bins,
            conf_interval=conf_interval,
            square_diff=square_diff,
            neighborhood_width=neighborhood_width,
            stats_info_dict=stats_info_dict,
            from_logits=from_logits,
            ignore_index=ignore_index
        )
    # Finally, get the calibration score.
    cal_info['cal_error'] = reduce_bin_errors(
        error_per_bin=cal_info["bin_cal_errors"], 
        amounts_per_bin=cal_info["bin_amounts"], 
        )
    # Finally, get the ECE score.
    NN, _ = cal_info["bin_cal_errors"].shape
    total_num_samples = cal_info['bin_amounts'].sum()
    ece_per_nn = torch.zeros(NN)
    # Iterate through each label, calculating ECE
    for nn_idx in range(NN):
        nn_ece = reduce_bin_errors(
            error_per_bin=cal_info["bin_cal_errors"][nn_idx], 
            amounts_per_bin=cal_info["bin_amounts"][nn_idx], 
            )
        nn_prob = cal_info['bin_amounts'][nn_idx].sum() / total_num_samples
        # Weight the ECE by the prob of the num neighbors.
        ece_per_nn[nn_idx] = nn_prob * nn_ece 
    # Finally, get the calibration score.
    cal_info['cal_error'] = ece_per_nn.sum()
    # Return the calibration information
    assert 0 <= cal_info['cal_error'] <= 1,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    # Return the calibration information.
    if return_dict:
        return cal_info
    else:
        return cal_info['cal_error']


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def tl_elm_loss(
    y_pred: Tensor = None, 
    y_true: Tensor = None,
    pixel_meters_dict: Dict[tuple, Meter] = None,
    num_bins: int = 10,
    neighborhood_width: int = 3,
    square_diff: bool = False,
    from_logits: bool = False,
    return_dict: bool = False,
    conf_interval: Tuple[float, float] = (0.0, 1.0),
    stats_info_dict: Optional[dict] = {},
    ignore_index: Optional[int] = None
    ) -> Union[dict, Tensor]:
    """
    Calculates the LoMS.
    """
    # Verify input.
    cal_input_check(y_pred, y_true, pixel_meters_dict)
    if pixel_meters_dict is not None:
        cal_info = global_label_neighbors_bin_stats(
            pixel_meters_dict=pixel_meters_dict,
            top_label=True,
            square_diff=square_diff,
            ignore_index=ignore_index
        )
    else:
        cal_info = label_neighbors_bin_stats(
            y_pred=y_pred,
            y_true=y_true,
            top_label=True,
            num_bins=num_bins,
            conf_interval=conf_interval,
            square_diff=square_diff,
            neighborhood_width=neighborhood_width,
            stats_info_dict=stats_info_dict,
            from_logits=from_logits,
            ignore_index=ignore_index
        )
    # Finally, get the ECE score.
    L, NN, _ = cal_info["bin_cal_errors"].shape
    total_num_samples = cal_info['bin_amounts'].sum()
    ece_per_lab_nn = torch.zeros((L, NN))
    # Iterate through each label and calculate the weighted ece.
    for lab_idx in range(L):
        for nn_idx in range(NN):
            lab_nn_ece = reduce_bin_errors(
                error_per_bin=cal_info['bin_cal_errors'][lab_idx, nn_idx], 
                amounts_per_bin=cal_info['bin_amounts'][lab_idx, nn_idx], 
                )
            # Calculate the empirical prob of the label.
            lab_nn_prob = cal_info['bin_amounts'][lab_idx, nn_idx].sum() / total_num_samples
            # Weight the ECE by the prob of the label.
            ece_per_lab_nn[lab_idx, nn_idx] = lab_nn_prob * lab_nn_ece 
    # Finally, get the calibration score.
    cal_info['cal_error'] =  ece_per_lab_nn.sum()
    assert 0 <= cal_info['cal_error'] <= 1,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    # Return the calibration information.
    if return_dict:
        return cal_info
    else:
        return cal_info['cal_error']


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def cw_elm_loss(
    y_pred: Tensor = None, 
    y_true: Tensor = None,
    pixel_meters_dict: Dict[tuple, Meter] = None,
    num_bins: int = 10,
    neighborhood_width: int = 3,
    square_diff: bool = False,
    from_logits: bool = False,
    return_dict: bool = False,
    conf_interval: Tuple[float, float] = (0.0, 1.0),
    stats_info_dict: Optional[dict] = {},
    ignore_index: Optional[int] = None
    ) -> Union[dict, Tensor]:
    """
    Calculates the LoMS.
    """
    # Verify input.
    cal_input_check(y_pred, y_true, pixel_meters_dict)
    if pixel_meters_dict is not None:
        cal_info = global_label_neighbors_bin_stats(
            pixel_meters_dict=pixel_meters_dict,
            top_label=False,
            square_diff=square_diff,
            ignore_index=ignore_index
        )
    else:
        cal_info = label_neighbors_bin_stats(
            y_pred=y_pred,
            y_true=y_true,
            top_label=False,
            num_bins=num_bins,
            conf_interval=conf_interval,
            square_diff=square_diff,
            neighborhood_width=neighborhood_width,
            stats_info_dict=stats_info_dict,
            from_logits=from_logits,
            ignore_index=ignore_index
        )
    # Finally, get the ECE score.
    L, NN, _ = cal_info["bin_cal_errors"].shape
    ece_per_lab = torch.zeros(L)
    # Iterate through each label and calculate the weighted ece.
    for lab_idx in range(L):
        # Calculate the total amount of samples for the label.
        total_lab_samples = cal_info['bin_amounts'][lab_idx].sum()
        # Keep track of the ECE for each neighbor class.
        ece_per_nn = torch.zeros(NN)
        for nn_idx in range(NN):
            lab_nn_ece = reduce_bin_errors(
                error_per_bin=cal_info['bin_cal_errors'][lab_idx, nn_idx], 
                amounts_per_bin=cal_info['bin_amounts'][lab_idx, nn_idx], 
                )
            # Calculate the empirical prob of the label.
            lab_nn_prob = cal_info['bin_amounts'][lab_idx, nn_idx].sum() / total_lab_samples
            # Weight the ECE by the prob of the label.
            ece_per_nn[nn_idx] = lab_nn_prob * lab_nn_ece 
        # Place the weighted ECE for the label.
        ece_per_lab[lab_idx] = ece_per_nn.sum()
    # Finally, get the calibration score.
    cal_info['cal_error'] =  ece_per_lab.mean()
    assert 0 <= cal_info['cal_error'] <= 1,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    # Return the calibration information.
    if return_dict:
        return cal_info
    else:
        return cal_info['cal_error']


# Loss modules
ELM = _loss_module_from_func("ELM", elm_loss)
TL_ELM = _loss_module_from_func("TL_ELM", tl_elm_loss)
CW_ELM = _loss_module_from_func("CW_ELM", cw_elm_loss)