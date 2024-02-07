# local imports for:
from .utils import reduce_bin_errors
# torch imports
import torch
from torch import Tensor
# misc imports
from typing import Union, Literal
from pydantic import validate_arguments


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
def class_ece_reduction(
    cal_info: dict,
    metric_type: str,
    class_weighting: Literal['uniform', 'propotional'],
    ignore_empty_classes: bool,
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
        amounts_per_lab = torch.zeros(L)
        # Iterate through each label and calculate the weighted ece.
        for lab_idx in range(L):
            lab_ece = reduce_bin_errors(
                error_per_bin=cal_info['bin_cal_errors'][lab_idx], 
                amounts_per_bin=cal_info['bin_amounts'][lab_idx], 
                )
            lab_amount = cal_info['bin_amounts'][lab_idx].sum()
            amounts_per_lab[lab_idx] = lab_amount
            # If uniform then apply no weighting.
            if class_weighting == 'uniform':
                ece_per_lab[lab_idx] = lab_ece
            else:
                lab_prob = lab_amount / total_num_samples 
                # Weight the ECE by the prob of the label.
                ece_per_lab[lab_idx] = lab_prob * lab_ece
        # Finally, get the calibration score.
        if class_weighting == 'uniform':
            if ignore_empty_classes:
                cal_info['cal_error'] = ece_per_lab[amounts_per_lab > 0].mean()
            else:
                cal_info['cal_error'] = ece_per_lab.mean()
        else:
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