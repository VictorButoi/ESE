# local imports for:
# - pixel statistics
from .metric_reductions import (
    ece_reduction,
    tl_ece_reduction,
    cw_ece_reduction
)
from .local_ps import (
    bin_stats, 
    top_label_bin_stats,
    joint_label_bin_stats
)
# - global statistics
from .global_ps import (
    global_bin_stats, 
    global_top_label_bin_stats,
    global_joint_label_bin_stats
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
def image_ece_loss(
    y_pred: Tensor,
    y_true: Tensor,
    num_bins: int,
    edge_only: bool = False,
    square_diff: bool = False,
    from_logits: bool = False,
    stats_info_dict: dict = {},
    conf_interval: Optional[List[float]] = None,
    neighborhood_width: Optional[int] = None,
    ignore_index: Optional[int] = None,
    **kwargs
    ) -> Union[dict, Tensor]:
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
    metric_dict = {
        "cal_info": cal_info,
        "return_dict": kwargs.get("return_dict", False) 
    }
    # Return the calibration information
    return ece_reduction(**metric_dict)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ece_loss(
    pixel_meters_dict: Dict[tuple, Meter],
    edge_only: bool = False,
    square_diff: bool = False,
    neighborhood_width: Optional[int] = None,
    ignore_index: Optional[int] = None,
    **kwargs
    ) -> Union[dict, Tensor]:
    cal_info = global_bin_stats(
        pixel_meters_dict=pixel_meters_dict,
        square_diff=square_diff,
        neighborhood_width=neighborhood_width,
        edge_only=edge_only,
        ignore_index=ignore_index
    )
    metric_dict = {
        "cal_info": cal_info,
        "return_dict": kwargs.get("return_dict", False) 
    }
    # Return the calibration information
    return ece_reduction(**metric_dict)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_tl_ece_loss(
    y_pred: Tensor,
    y_true: Tensor,
    num_bins: int,
    edge_only: bool = False,
    square_diff: bool = False,
    from_logits: bool = False,
    stats_info_dict: dict = {},
    conf_interval: Optional[List[float]] = None,
    neighborhood_width: Optional[int] = None,
    ignore_index: Optional[int] = None,
    **kwargs
    ) -> Union[dict, Tensor]:
    cal_info = top_label_bin_stats(
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
    metric_dict = {
        "cal_info": cal_info,
        "return_dict": kwargs.get("return_dict", False) 
    }
    # Return the calibration information
    return tl_ece_reduction(**metric_dict)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def tl_ece_loss(
    pixel_meters_dict: Dict[tuple, Meter],
    edge_only: bool = False,
    square_diff: bool = False,
    neighborhood_width: Optional[int] = None,
    ignore_index: Optional[int] = None,
    **kwargs
    ) -> Union[dict, Tensor]:
    cal_info = global_top_label_bin_stats(
        pixel_meters_dict=pixel_meters_dict,
        top_label=True,
        square_diff=square_diff,
        neighborhood_width=neighborhood_width,
        edge_only=edge_only,
        ignore_index=ignore_index
    )
    metric_dict = {
        "cal_info": cal_info,
        "return_dict": kwargs.get("return_dict", False) 
    }
    # Return the calibration information
    return tl_ece_reduction(**metric_dict)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_cw_ece_loss(
    y_pred: Tensor,
    y_true: Tensor,
    num_bins: int,
    edge_only: bool = False,
    square_diff: bool = False,
    from_logits: bool = False,
    stats_info_dict: dict = {},
    conf_interval: Optional[List[float]] = None,
    neighborhood_width: Optional[int] = None,
    ignore_index: Optional[int] = None,
    **kwargs
    ) -> Union[dict, Tensor]:
    cal_info = joint_label_bin_stats(
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
    metric_dict = {
        "cal_info": cal_info,
        "return_dict": kwargs.get("return_dict", False) 
    }
    # Return the calibration information
    return cw_ece_reduction(**metric_dict)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def cw_ece_loss(
    pixel_meters_dict: Dict[tuple, Meter],
    edge_only: bool = False,
    square_diff: bool = False,
    neighborhood_width: Optional[int] = None,
    ignore_index: Optional[int] = None,
    **kwargs
    ) -> Union[dict, Tensor]:
    # Get the statistics either from images or pixel meter dict.
    cal_info = global_joint_label_bin_stats(
        pixel_meters_dict=pixel_meters_dict,
        top_label=False,
        square_diff=square_diff,
        neighborhood_width=neighborhood_width,
        edge_only=edge_only,
        ignore_index=ignore_index
    )
    metric_dict = {
        "cal_info": cal_info,
        "return_dict": kwargs.get("return_dict", False) 
    }
    # Return the calibration information
    return cw_ece_reduction(**metric_dict)


# Edge only versions of the above functions.
##################################################################################################

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_edge_ece_loss(
    y_pred: Tensor,
    y_true: Tensor,
    **kwargs
    ) -> Union[dict, Tensor]:
    assert "neighborhood_width" in kwargs, "Must provide neighborhood width if doing an edge metric."
    kwargs["y_pred"] = y_pred
    kwargs["y_true"] = y_true
    kwargs["edge_only"] = True
    # Return the calibration information
    return image_ece_loss(**kwargs)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def edge_ece_loss(
    pixel_meters_dict: Dict[tuple, Meter],
    **kwargs
    ) -> Union[dict, Tensor]:
    assert "neighborhood_width" in kwargs, "Must provide neighborhood width if doing an edge metric."
    kwargs["pixel_meters_dict"] = pixel_meters_dict 
    kwargs["edge_only"] = True
    # Return the calibration information
    return ece_loss(**kwargs)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_etl_ece_loss(
    y_pred: Tensor = None, 
    y_true: Tensor = None,
    **kwargs
    ) -> Union[dict, Tensor]:
    assert "neighborhood_width" in kwargs, "Must provide neighborhood width if doing an edge metric."
    kwargs["y_pred"] = y_pred
    kwargs["y_true"] = y_true
    kwargs["edge_only"] = True
    # Return the calibration information
    return image_tl_ece_loss(**kwargs)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def etl_ece_loss(
    pixel_meters_dict: Dict[tuple, Meter],
    **kwargs
    ) -> Union[dict, Tensor]:
    assert "neighborhood_width" in kwargs, "Must provide neighborhood width if doing an edge metric."
    kwargs["pixel_meters_dict"] = pixel_meters_dict 
    kwargs["edge_only"] = True
    # Return the calibration information
    return tl_ece_loss(**kwargs)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_ecw_ece_loss(
    y_pred: Tensor = None, 
    y_true: Tensor = None,
    **kwargs
    ) -> Union[dict, Tensor]:
    assert "neighborhood_width" in kwargs, "Must provide neighborhood width if doing an edge metric."
    kwargs["y_pred"] = y_pred
    kwargs["y_true"] = y_true
    kwargs["edge_only"] = True
    # Return the calibration information
    return image_cw_ece_loss(**kwargs)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ecw_ece_loss(
    pixel_meters_dict: Dict[tuple, Meter],
    **kwargs
    ) -> Union[dict, Tensor]:
    assert "neighborhood_width" in kwargs, "Must provide neighborhood width if doing an edge metric."
    kwargs["pixel_meters_dict"] = pixel_meters_dict 
    kwargs["edge_only"] = True
    # Return the calibration information
    return cw_ece_loss(**kwargs)


#############################################################################
# Global metrics
#############################################################################

# Loss modules
ECE = _loss_module_from_func("ECE", ece_loss)
TL_ECE = _loss_module_from_func("TL_ECE", tl_ece_loss)
CW_ECE = _loss_module_from_func("CW_ECE", cw_ece_loss)

# Edge loss modules
Edge_ECE = _loss_module_from_func("Edge_ECE", edge_ece_loss)
ETL_ECE = _loss_module_from_func("ETL_ECE", etl_ece_loss)
ECW_ECE = _loss_module_from_func("ECW_ECE", ecw_ece_loss)

#############################################################################
# Image-based metrics
#############################################################################

# Loss modules
Image_ECE = _loss_module_from_func("Image_ECE", image_ece_loss)
Image_TL_ECE = _loss_module_from_func("Image_TL_ECE", image_tl_ece_loss)
Image_CW_ECE = _loss_module_from_func("Image_CW_ECE", image_cw_ece_loss)

# Edge loss modules
Image_Edge_ECE = _loss_module_from_func("Image_Edge_ECE", image_edge_ece_loss)
Image_ETL_ECE = _loss_module_from_func("Image_ETL_ECE", image_etl_ece_loss)
Image_ECW_ECE = _loss_module_from_func("Image_ECW_ECE", image_ecw_ece_loss)