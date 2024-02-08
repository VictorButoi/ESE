# local imports for:
from .utils import reduce_bin_errors
# torch imports
import torch
from torch import Tensor
# misc imports
from pydantic import validate_arguments
from typing import Union, Literal, Optional


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
    ignore_index: Optional[int] = None,
    return_dict: bool = False,
) -> Union[dict, Tensor]:
    """
    Calculates the Expected Semantic Error (ECE) for a predicted label map.
    """
    # Finally, get the ECE score.
    total_num_samples = cal_info['bin_amounts'].sum()
    # If there are no samples, then the ECE is 0.
    if total_num_samples == 0:
        cal_info['cal_error'] = torch.tensor(0.0)
        if return_dict:
            cal_info['metric_type'] = metric_type
            return cal_info
        else:
            return cal_info['cal_error']
    # Go through each label and calculate the ECE.
    L, _ = cal_info["bin_cal_errors"].shape
    ece_per_lab = torch.zeros(L)
    amounts_per_lab = torch.zeros(L)
    # If we are ignoring an index, then the number of labels is reduced by 1.
    num_labs = L if ignore_index is None else max(L - 1, 1)
    # Iterate through each label and calculate the weighted ece.
    for lab_idx in range(L):
        # If we are ignoring an index, skip it in calculations.
        if ignore_index is not None and lab_idx != ignore_index:
            lab_ece = reduce_bin_errors(
                error_per_bin=cal_info['bin_cal_errors'][lab_idx], 
                amounts_per_bin=cal_info['bin_amounts'][lab_idx], 
                )
            lab_amount = cal_info['bin_amounts'][lab_idx].sum()
            amounts_per_lab[lab_idx] = lab_amount
            # If uniform then apply no weighting.
            if class_weighting == 'uniform':
                lab_prob = 1.0 / num_labs 
            else:
                lab_prob = lab_amount / total_num_samples 
            # Weight the ECE by the prob of the label.
            ece_per_lab[lab_idx] = lab_prob * lab_ece
    # Finally, get the calibration score.
    if ignore_empty_classes:
        if amounts_per_lab.sum() > 0:
            cal_info['cal_error'] = ece_per_lab[amounts_per_lab > 0].sum()
        else:
            cal_info['cal_error'] = torch.tensor(0.0) # If no samples, then ECE is 0.
    else:
        cal_info['cal_error'] = ece_per_lab.sum()
    # Return the calibration information.
    assert 0.0 <= cal_info['cal_error'] <= 1.0,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    # Return the calibration information.
    if return_dict:
        cal_info['metric_type'] = metric_type
        return cal_info
    else:
        return cal_info['cal_error']