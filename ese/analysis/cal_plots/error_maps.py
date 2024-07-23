# random imports
import numpy as np
from typing import Any, Literal
from pydantic import validate_arguments

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_ece_map(
    subj: dict,
    class_type: Literal["Binary", "Multi-class"],
    fig: Any,
    ax: Any,
    include_background: bool
):
    if class_type == "Multi-class": 
        assert include_background, "Background must be included for multi-class."

    # Copy the soft and hard predictions
    conf_map = subj['conf_map'].clone()
    pred_map = subj['pred_map'].clone()

    # Calculate the per-pixel accuracy and where the foreground regions are.
    acc_per_pixel = (subj['label'] == pred_map).float()
    pred_foreground = pred_map.bool()

    # Set the regions of the image corresponding to groundtruth label.
    ece_map = np.zeros_like(subj['label'])

    # Fill in the areas where we want.
    ece_map[pred_foreground] = (conf_map - acc_per_pixel)[pred_foreground]
    if class_type == "Binary" and include_background:
        pred_background = ~pred_foreground
        background_conf_map = 1 - conf_map
        ece_map[pred_background] = (background_conf_map - acc_per_pixel)[pred_background]

    # Get the bounds for visualization
    ece_abs_max = np.max(np.abs(ece_map))
    ece_vmin, ece_vmax = -ece_abs_max, ece_abs_max

    # Show the ece map
    ce_im = ax.imshow(ece_map, cmap="RdBu_r", interpolation="None", vmax=ece_vmax, vmin=ece_vmin)
    ax.set_title("Pixel-wise Cal Error")
    fig.colorbar(ce_im, ax=ax)