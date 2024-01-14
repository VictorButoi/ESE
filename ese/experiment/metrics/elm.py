# get the processing function.
from .metric_reductions import (
    elm_reduction,
    tl_elm_reduction,
    cw_elm_reduction
)
from .local_ps import (
    neighbors_bin_stats, 
    label_neighbors_bin_stats
)
from .global_ps import (
    global_neighbors_bin_stats,
    global_label_neighbors_bin_stats
)
# torch imports
from torch import Tensor
# misc imports
from pydantic import validate_arguments
from typing import Dict, Optional, Union, List
# ionpy imports
from ionpy.util.meter import Meter
from ionpy.loss.util import _loss_module_from_func


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_elm_loss(
    y_pred: Tensor,
    y_true: Tensor,
    num_bins: int,
    conf_interval: List[float],
    neighborhood_width: int,
    square_diff: bool = False,
    from_logits: bool = False,
    stats_info_dict: Optional[dict] = {},
    ignore_index: Optional[int] = None,
    **kwargs
    ) -> Union[dict, Tensor]:
    """
    Calculates the TENCE: Top-Label Expected Neighborhood-conditioned Calibration Error.
    """
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
    kwargs['cal_info'] = cal_info
    return elm_reduction(**kwargs)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def elm_loss(
    pixel_meters_dict: Dict[tuple, Meter],
    neighborhood_width: int,
    square_diff: bool = False,
    ignore_index: Optional[int] = None,
    **kwargs
    ) -> Union[dict, Tensor]:
    """
    Calculates the TENCE: Top-Label Expected Neighborhood-conditioned Calibration Error.
    """
    cal_info = global_neighbors_bin_stats(
        pixel_meters_dict=pixel_meters_dict,
        neighborhood_width=neighborhood_width,
        square_diff=square_diff,
        ignore_index=ignore_index
    )
    kwargs['cal_info'] = cal_info
    return elm_reduction(**kwargs)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_tl_elm_loss(
    y_pred: Tensor,
    y_true: Tensor,
    num_bins: int,
    conf_interval: List[float],
    neighborhood_width: int,
    square_diff: bool = False,
    from_logits: bool = False,
    stats_info_dict: Optional[dict] = {},
    ignore_index: Optional[int] = None,
    **kwargs
    ) -> Union[dict, Tensor]:
    """
    Calculates the LoMS.
    """
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
    kwargs['cal_info'] = cal_info
    return tl_elm_reduction(**kwargs)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def tl_elm_loss(
    pixel_meters_dict: Dict[tuple, Meter],
    neighborhood_width: int,
    square_diff: bool = False,
    ignore_index: Optional[int] = None,
    **kwargs
    ) -> Union[dict, Tensor]:
    """
    Calculates the LoMS.
    """
    # Verify input.
    cal_info = global_label_neighbors_bin_stats(
        pixel_meters_dict=pixel_meters_dict,
        neighborhood_width=neighborhood_width,
        top_label=True,
        square_diff=square_diff,
        ignore_index=ignore_index
    )
    kwargs['cal_info'] = cal_info
    return tl_elm_reduction(**kwargs)
        

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_cw_elm_loss(
    y_pred: Tensor,
    y_true: Tensor,
    num_bins: int,
    conf_interval: List[float],
    neighborhood_width: int,
    square_diff: bool = False,
    from_logits: bool = False,
    stats_info_dict: Optional[dict] = {},
    ignore_index: Optional[int] = None,
    **kwargs
    ) -> Union[dict, Tensor]:
    """
    Calculates the LoMS.
    """
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
    kwargs['cal_info'] = cal_info
    return cw_elm_reduction(**kwargs)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def cw_elm_loss(
    pixel_meters_dict: Dict[tuple, Meter],
    neighborhood_width: int,
    square_diff: bool = False,
    ignore_index: Optional[int] = None,
    **kwargs
    ) -> Union[dict, Tensor]:
    """
    Calculates the LoMS.
    """
    cal_info = global_label_neighbors_bin_stats(
        pixel_meters_dict=pixel_meters_dict,
        neighborhood_width=neighborhood_width,
        top_label=False,
        square_diff=square_diff,
        ignore_index=ignore_index
    )
    kwargs['cal_info'] = cal_info
    return cw_elm_reduction(**kwargs)


#############################################################################
# Global metrics
#############################################################################

ELM = _loss_module_from_func("ELM", elm_loss)
TL_ELM = _loss_module_from_func("TL_ELM", tl_elm_loss)
CW_ELM = _loss_module_from_func("CW_ELM", cw_elm_loss)

#############################################################################
# Image-based metrics
#############################################################################

Image_ELM = _loss_module_from_func("Image_ELM", image_elm_loss)
Image_TL_ELM = _loss_module_from_func("Image_TL_ELM", image_tl_elm_loss)
Image_CW_ELM = _loss_module_from_func("Image_CW_ELM", image_cw_elm_loss)