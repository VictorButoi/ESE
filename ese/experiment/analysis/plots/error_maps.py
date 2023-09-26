# random imports
import torch
import numpy as np
from typing import Any
from pydantic import validate_arguments

# ionpy imports
from ionpy.metrics import pixel_accuracy
from ionpy.util.islands import get_connected_components


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_ece_map(
    subj: dict,
    fig: Any,
    ax: Any,
):
    # Copy the soft and hard predictions
    conf_map = subj['conf_map'].clone()
    pred_map = subj['pred_map'].clone()

    # Calculate the per-pixel accuracy and where the foreground regions are.
    acc_per_pixel = (subj['label'] == pred_map).float()
    pred_foreground = pred_map.bool()

    # Set the regions of the image corresponding to groundtruth label.
    ece_map = np.zeros_like(subj['label'])
    ece_map[pred_foreground] = (conf_map - acc_per_pixel)[pred_foreground]

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
    average: bool = False,
    fig: Any = None,
    ax: Any = None
):
    # Get the confidence bins
    conf_bins = torch.linspace(0, 1, num_bins+1)[:-1] # Off by one error

    conf_map = subj['conf_map']
    rece_map = np.zeros_like(conf_map)

    # Make sure bins are aligned.
    bin_width = conf_bins[1] - conf_bins[0]
    for c_bin in conf_bins:

        # Get the binary region of this confidence interval
        bin_conf_region = (conf_map >= c_bin) & (conf_map < (c_bin + bin_width))

        # Break it up into islands
        conf_islands = get_connected_components(bin_conf_region)

        # Iterate through each island, and get the measure for each island.
        for island in conf_islands:

            # Get the label corresponding to the island and simulate ground truth and make the right shape.
            label_region = subj["label"][island][None, None, ...]
            pseudo_pred = torch.ones_like(label_region)

            # If averaging, then everything in one island will get the same score, otherwise pixelwise.
            if average:
                # Calculate the accuracy and mean confidence for the island.
                region_accuracies  = pixel_accuracy(pseudo_pred , label_region)
                region_confidences = conf_map[island].mean()
            else:
                region_accuracies = (pseudo_pred == label_region).squeeze().float()
                region_confidences = conf_map[island]

            # Place the numbers in the island.
            rece_map[island] = (region_confidences - region_accuracies)

    # Get the bounds for visualization
    rece_abs_max = np.max(np.abs(rece_map))
    rece_vmin, rece_vmax = -rece_abs_max, rece_abs_max
    rece_im = ax.imshow(rece_map, cmap="RdBu_r", interpolation="None", vmax=rece_vmax, vmin=rece_vmin)

    if average:
        ax.set_title("Averaged Region-wise Cal Error")
    else:
        ax.set_title("Region-wise Cal Error")

    fig.colorbar(rece_im, ax=ax)
