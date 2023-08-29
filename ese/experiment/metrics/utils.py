# misc imports
import numpy as np

# local imports
from ionpy.util.validation import validate_arguments_init


@validate_arguments_init
def reduce_scores(
        scores: np.ndarray, 
        bin_amounts: np.ndarray, 
        weighting='proportional' 
        ) -> float:

    if np.sum(bin_amounts) == 0:
        return 0.0
    elif weighting== 'proportional':
        bin_weights = bin_amounts / np.sum(bin_amounts)
    elif weighting== 'uniform':
        bin_weights = np.ones_like(bin_amounts) / len(bin_amounts)
    else:
        raise ValueError("Invalid bin weighting.")

    return np.average(scores, weights=bin_weights).item()


