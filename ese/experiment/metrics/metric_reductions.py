# local imports for:
from .utils import reduce_bin_errors
# torch imports
import torch
from torch import Tensor
# misc imports
from pydantic import validate_arguments
from typing import Union


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ece_reduction(
    cal_info: dict,
    metric_type: str,
    return_dict: bool = False,
    ) -> Union[dict, Tensor]:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
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
        cal_info['metric_type'] = metric_type
        return cal_info
    else:
        return cal_info['cal_error']


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def tl_ece_reduction(
    cal_info: dict,
    metric_type: str,
    return_dict: bool = False,
    ) -> Union[dict, Tensor]:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
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
        cal_info['metric_type'] = metric_type
        return cal_info
    else:
        return cal_info['cal_error']


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def cw_ece_reduction(
    cal_info: dict,
    metric_type: str,
    return_dict: bool = False,
    ) -> Union[dict, Tensor]:
    """
    Calculates the LoMS.
    """
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
        cal_info['metric_type'] = metric_type
        return cal_info
    else:
        return cal_info['cal_error']


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def elm_reduction(
    cal_info: dict,
    metric_type: str,
    return_dict: bool = False,
) -> Union[dict, Tensor]:
    # Finally, get the ELM score.
    total_num_samples = cal_info['bin_amounts'].sum()
    if total_num_samples == 0:
        cal_info['cal_error'] = torch.tensor(0.0)
    else:
        NN, _ = cal_info["bin_cal_errors"].shape
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
    # Return the calibration information.
    assert 0.0 <= cal_info['cal_error'] <= 1.0,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    # Return the calibration information.
    if return_dict:
        cal_info['metric_type'] = metric_type
        return cal_info
    else:
        return cal_info['cal_error']


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def cw_elm_reduction(
    cal_info: dict,
    metric_type: str,
    return_dict: bool = False,
) -> Union[dict, Tensor]:
    total_num_samples = cal_info['bin_amounts'].sum()
    if total_num_samples == 0:
        cal_info['cal_error'] = torch.tensor(0.0)
    else:
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

    # Return the calibration information.
    assert 0.0 <= cal_info['cal_error'] <= 1.0,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    # Return the calibration information.
    if return_dict: 
        cal_info['metric_type'] = metric_type
        return cal_info
    else:
        return cal_info['cal_error']    