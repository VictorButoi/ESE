# misc imports
import numpy as np
import torch
from typing import Literal

# ionpy imports
from ionpy.util.validation import validate_arguments_init

#local imports
from .utils import reduce_scores
from .calibration import ECE, ESE, ReCE



@validate_arguments_init
def ece_score(
    bins: np.ndarray,
    pred: torch.Tensor = None, 
    label: torch.Tensor = None,
    confidences: torch.Tensor = None,
    accuracies: torch.Tensor = None,
    bin_weighting: Literal["weighted", "uniform"] = 'weighted',
    from_logits: bool = False,
    ):
    
    ece_per_bin, _, bin_amounts = ECE(
        bins,
        pred,
        label,
        confidences,
        accuracies,
        from_logits
    )
    
    ece_score = reduce_scores(ece_per_bin, bin_amounts, bin_weighting) 
    
    return ece_score 
    

@validate_arguments_init
def ese_score(
    bins: np.ndarray,
    pred: torch.Tensor = None, 
    label: torch.Tensor = None,
    bin_weighting: Literal["weighted", "uniform"] = 'weighted',
    from_logits: bool = False
    ):
    
    ese_per_bin, _, bin_amounts = ESE(
        bins,
        pred,
        label,
        from_logits
    )
    
    ese_score = reduce_scores(ese_per_bin, bin_amounts, bin_weighting) 
    
    return ese_score 
    

@validate_arguments_init
def rece_score(
    bins: np.ndarray,
    pred: torch.Tensor = None, 
    label: torch.Tensor = None,
    bin_weighting: Literal["weighted", "uniform"] = 'weighted',
    from_logits: bool = False,
    ):
    
    rece_per_bin, _, bin_amounts = ReCE(
        bins,
        pred,
        label,
        from_logits
    )
    
    rece_score = reduce_scores(rece_per_bin, bin_amounts, bin_weighting) 
    
    return rece_score
    



