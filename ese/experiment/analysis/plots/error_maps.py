# random imports
import numpy as np
from typing import Any, Literal
from pydantic import validate_arguments

# local imports
from ese.experiment.metrics.utils import get_bins, get_conf_region

# ionpy imports
from ionpy.metrics import pixel_accuracy, pixel_precision
from ionpy.util.islands import get_connected_components


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
    if include_background:
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


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_rece_map(
    subj: dict,
    num_bins: int,
    class_type: Literal["Binary", "Multi-class"],
    fig: Any,
    ax: Any,
    include_background: bool
):
    if class_type == "Multi-class": 
        assert include_background, "Background must be included for multi-class."

    # Create the confidence bins.    
    conf_bins, conf_bin_widths = get_bins(
        num_bins=num_bins,
        metric="ReCE", 
        class_type=class_type,
        include_background=include_background, 
        )

    conf_map = subj['conf_map']
    pred_map = subj['pred_map']
    label = subj['label']
    rece_map = np.zeros_like(conf_map)

    # Go through each bin, starting at the back so that we don't have to run connected components
    for bin_idx, conf_bin in enumerate(conf_bins):

        # Get the region of image corresponding to the confidence
        bin_conf_region = get_conf_region(bin_idx, conf_bin, conf_bin_widths, conf_map)

        # If there are some pixels in this confidence bin.
        if bin_conf_region.sum() != 0:
            # If we are not the last bin, get the connected components.
            conf_islands = get_connected_components(bin_conf_region)

            # Iterate through each island, and get the measure for each island.
            for island in conf_islands:
                # Get the island primitives
                region_conf_map = conf_map[island]                
                region_pred_map = pred_map[island]
                region_label_map = label[island]

                # Calculate the accuracy and mean confidence for the island.
                region_conf = region_conf_map.mean()

                if class_type == "Multi-class":
                    region_metric = pixel_accuracy(region_pred_map, region_label_map)
                else:
                    region_metric = pixel_precision(region_pred_map, region_label_map)

                # Place this score in the island.
                rece_map[island] = (region_conf - region_metric)

    # Get the bounds for visualization
    rece_abs_max = np.max(np.abs(rece_map))
    rece_vmin, rece_vmax = -rece_abs_max, rece_abs_max
    rece_im = ax.imshow(rece_map, cmap="RdBu_r", interpolation="None", vmax=rece_vmax, vmin=rece_vmin)

    # Set the title and add a color bar
    ax.set_title("Region-wise Cal Error")
    fig.colorbar(rece_im, ax=ax)
