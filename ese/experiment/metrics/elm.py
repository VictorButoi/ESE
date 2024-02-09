# get the processing function.
from .local_ps import neighbor_bin_stats 
from .metric_reductions import elm_reduction 
from .global_ps import global_binwise_stats  
# torch imports
from torch import Tensor
# misc imports
from pydantic import validate_arguments
from typing import Dict, Optional, Union, List, Literal
# ionpy imports
from ionpy.util.meter import Meter
from ionpy.loss.util import _loss_module_from_func


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_elm_loss(
    y_pred: Tensor,
    y_true: Tensor,
    num_bins: int,
    neighborhood_width: int,
    class_weighting: Literal["uniform", "proportional"],
    edge_only: bool = False,
    square_diff: bool = False,
    from_logits: bool = False,
    conf_interval: Optional[List[float]] = None,
    **kwargs
    ) -> Union[dict, Tensor]:
    cal_info = neighbor_bin_stats(
        y_pred=y_pred,
        y_true=y_true,
        num_bins=num_bins,
        conf_interval=conf_interval,
        square_diff=square_diff,
        neighborhood_width=neighborhood_width,
        edge_only=edge_only,
        from_logits=from_logits
    )
    metric_dict = {
        "metric_type": "local",
        "cal_info": cal_info,
        "class_weighting": class_weighting,
        "return_dict": kwargs.get("return_dict", False)
    }
    # print("Local Bin counts:\n", cal_info["bin_amounts"])
    # print("Local Bin cal errors:\n", cal_info["bin_cal_errors"])
    return elm_reduction(**metric_dict)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def elm_loss(
    pixel_meters_dict: Dict[tuple, Meter],
    neighborhood_width: int,
    class_weighting: Literal["uniform", "proportional"],
    edge_only: bool = False,
    square_diff: bool = False,
    **kwargs
    ) -> Union[dict, Tensor]:
    cal_info = global_binwise_stats(
        pixel_meters_dict=pixel_meters_dict,
        class_wise=False,
        class_conditioned=False,
        neighborhood_conditioned=True,
        square_diff=square_diff,
        neighborhood_width=neighborhood_width,
        edge_only=edge_only
    )
    metric_dict = {
        "metric_type": "global",
        "cal_info": cal_info,
        "class_weighting": class_weighting,
        "return_dict": kwargs.get("return_dict", False) 
    }
    # print("Global Bin counts:\n", cal_info["bin_amounts"])
    # print("Global Bin cal errors:\n", cal_info["bin_cal_errors"])
    return elm_reduction(**metric_dict)


# Edge only versions of the above functions.
##################################################################################################

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_edge_elm_loss(
    y_pred: Tensor,
    y_true: Tensor,
    **kwargs
    ) -> Union[dict, Tensor]:
    assert "neighborhood_width" in kwargs, "Must provide neighborhood width if doing an edge metric."
    kwargs["y_pred"] = y_pred
    kwargs["y_true"] = y_true
    kwargs["edge_only"] = True
    # Return the calibration information
    return image_elm_loss(**kwargs)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def edge_elm_loss(
    pixel_meters_dict: Dict[tuple, Meter],
    **kwargs
    ) -> Union[dict, Tensor]:
    assert "neighborhood_width" in kwargs, "Must provide neighborhood width if doing an edge metric."
    kwargs["pixel_meters_dict"] = pixel_meters_dict 
    kwargs["edge_only"] = True
    # Return the calibration information
    return elm_loss(**kwargs)


#############################################################################
# Global metrics
#############################################################################

ELM = _loss_module_from_func("ELM", elm_loss)
# Edge Metrics
Edge_ELM = _loss_module_from_func("Edge_ELM", edge_elm_loss)

#############################################################################
# Image-based metrics
#############################################################################

Image_ELM = _loss_module_from_func("Image_ELM", image_elm_loss)
# Edge Metrics
Image_Edge_ELM = _loss_module_from_func("Image_Edge_ELM", image_edge_elm_loss)