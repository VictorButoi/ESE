from .pix_stats import bin_stats, label_bin_stats, neighbors_bin_stats, label_neighbors_bin_stats
from .utils import reduce_bin_errors
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
    weighting: str = "proportional",
    stats_info_dict: Optional[dict] = {},
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
        stats_info_dict=stats_info_dict,
        ignore_index=ignore_index
    )
    # Finally, get the calibration score.
    cal_info['cal_error'] = reduce_bin_errors(
        error_per_bin=cal_info["bin_cal_errors"], 
        amounts_per_bin=cal_info["bin_amounts"], 
        weighting=weighting
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
    weighting: str = "proportional",
    stats_info_dict: Optional[dict] = {},
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    # Keep track of different things for each bin.
    cal_info = label_bin_stats(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        square_diff=square_diff,
        stats_info_dict=stats_info_dict,
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
            weighting=weighting
            )
        lab_prob = cal_info['bin_amounts'][lab_idx].sum() / total_num_samples 
        # Weight the ECE by the prob of the label.
        ece_per_lab[lab_idx] = lab_prob * lab_ece

    # Finally, get the calibration score.
    cal_info['cal_error'] =  ece_per_lab.sum().item()
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
    weighting: str = "proportional",
    stats_info_dict: Optional[dict] = {},
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the LoMS.
    """
    # Keep track of different things for each bin.
    cal_info = label_bin_stats(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        square_diff=square_diff,
        stats_info_dict=stats_info_dict,
        ignore_index=ignore_index
    )
    # Finally, get the ECE score.
    L, _ = cal_info["bin_cal_errors"].shape
    ece_per_lab = torch.zeros(L)
    # Iterate through each label, calculating ECE
    for lab_idx in range(L):
        lab_ece = reduce_bin_errors(
            error_per_bin=cal_info["bin_cal_errors"][lab_idx], 
            amounts_per_bin=cal_info["bin_amounts"][lab_idx], 
            weighting=weighting
            )
        ece_per_lab[lab_idx] = (1/L) * lab_ece

    # Finally, get the calibration score.
    cal_info['cal_error'] = ece_per_lab.sum().item()
    # Return the calibration information
    assert 0 <= cal_info['cal_error'] <= 1,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def LoMS(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor,
    num_bins: int,
    conf_interval: Tuple[float, float],
    square_diff: bool,
    neighborhood_width: int = 3,
    weighting: str = "proportional",
    stats_info_dict: Optional[dict] = {},
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
        uniform_weighting=True,
        stats_info_dict=stats_info_dict,
        ignore_index=ignore_index
    )
    # Finally, get the calibration score.
    cal_info['cal_error'] = reduce_bin_errors(
        error_per_bin=cal_info["bin_cal_errors"], 
        amounts_per_bin=cal_info["bin_amounts"], 
        weighting=weighting
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
            weighting=weighting
            )
        nn_prob = cal_info['bin_amounts'][nn_idx].sum() / total_num_samples
        # Weight the ECE by the prob of the num neighbors.
        ece_per_nn[nn_idx] = nn_prob * nn_ece 

    # Finally, get the calibration score.
    cal_info['cal_error'] = ece_per_nn.sum().item()
    # Return the calibration information
    assert 0 <= cal_info['cal_error'] <= 1,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def TL_LoMS(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor,
    num_bins: int,
    conf_interval: Tuple[float, float],
    square_diff: bool,
    neighborhood_width: int = 3,
    weighting: str = "proportional",
    stats_info_dict: Optional[dict] = {},
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the LoMS.
    """
    # Keep track of different things for each bin.
    cal_info = label_neighbors_bin_stats(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        square_diff=square_diff,
        neighborhood_width=neighborhood_width,
        uniform_weighting=True,
        stats_info_dict=stats_info_dict,
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
                weighting=weighting
                )
            # Calculate the empirical prob of the label.
            lab_nn_prob = cal_info['bin_amounts'][lab_idx, nn_idx].sum() / total_num_samples
            # Weight the ECE by the prob of the label.
            ece_per_lab_nn[lab_idx, nn_idx] = lab_nn_prob * lab_nn_ece 

    # Finally, get the calibration score.
    cal_info['cal_error'] =  ece_per_lab_nn.sum().item()
    assert 0 <= cal_info['cal_error'] <= 1,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def CW_LoMS(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor,
    num_bins: int,
    conf_interval: Tuple[float, float],
    square_diff: bool,
    neighborhood_width: int = 3,
    weighting: str = "proportional",
    stats_info_dict: Optional[dict] = {},
    ignore_index: Optional[int] = None
    ) -> dict:
    """
    Calculates the LoMS.
    """
    # Keep track of different things for each bin.
    cal_info = label_neighbors_bin_stats(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        square_diff=square_diff,
        neighborhood_width=neighborhood_width,
        uniform_weighting=True,
        stats_info_dict=stats_info_dict,
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
                weighting=weighting
                )
            # Calculate the empirical prob of the label.
            lab_nn_prob = cal_info['bin_amounts'][lab_idx, nn_idx].sum() / total_lab_samples
            # Weight the ECE by the prob of the label.
            ece_per_nn[nn_idx] = lab_nn_prob * lab_nn_ece 
        # Place the weighted ECE for the label.
        ece_per_lab[lab_idx] = (1 / L) * ece_per_nn.sum()

    # Finally, get the calibration score.
    cal_info['cal_error'] =  ece_per_lab.sum().item()
    assert 0 <= cal_info['cal_error'] <= 1,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    return cal_info