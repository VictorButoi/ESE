from .pix_stats import neighbors_bin_stats, label_neighbors_bin_stats
from .utils import reduce_bin_errors
# misc imports
import torch
from typing import Tuple, Optional
from pydantic import validate_arguments
# ionpy imports
from ionpy.loss.util import _loss_module_from_func


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def elm_loss(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor,
    num_bins: int = 10,
    square_diff: bool = False,
    conf_interval: Tuple[float, float] = (0.0, 1.0),
    neighborhood_width: int = 3,
    uniform_weighting: bool = False,
    stats_info_dict: Optional[dict] = {},
    from_logits: bool = False,
    return_dict: bool = False,
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the TENCE: Top-Label Expected Neighborhood-conditioned Calibration Error.
    """
    # Keep track of different things for each bin.
    cal_info = neighbors_bin_stats(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        square_diff=square_diff,
        neighborhood_width=neighborhood_width,
        uniform_weighting=uniform_weighting,
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
    y_pred: torch.Tensor, 
    y_true: torch.Tensor,
    num_bins: int = 10,
    square_diff: bool = False,
    conf_interval: Tuple[float, float] = (0.0, 1.0),
    neighborhood_width: int = 3,
    uniform_weighting: bool = False,
    stats_info_dict: Optional[dict] = {},
    from_logits: bool = False,
    return_dict: bool = False,
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the LoMS.
    """
    # Keep track of different things for each bin.
    cal_info = label_neighbors_bin_stats(
        y_pred=y_pred,
        y_true=y_true,
        top_label=True,
        num_bins=num_bins,
        conf_interval=conf_interval,
        square_diff=square_diff,
        neighborhood_width=neighborhood_width,
        uniform_weighting=uniform_weighting,
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
    y_pred: torch.Tensor, 
    y_true: torch.Tensor,
    num_bins: int = 10,
    square_diff: bool = False,
    conf_interval: Tuple[float, float] = (0.0, 1.0),
    neighborhood_width: int = 3,
    uniform_weighting: bool = False,
    stats_info_dict: Optional[dict] = {},
    from_logits: bool = False,
    return_dict: bool = False,
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the LoMS.
    """
    # Keep track of different things for each bin.
    cal_info = label_neighbors_bin_stats(
        y_pred=y_pred,
        y_true=y_true,
        top_label=False,
        num_bins=num_bins,
        conf_interval=conf_interval,
        square_diff=square_diff,
        neighborhood_width=neighborhood_width,
        uniform_weighting=uniform_weighting,
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