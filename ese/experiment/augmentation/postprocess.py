# torch imports
import torch

# ionpy imports
from ionpy.util.islands import get_connected_components


def smooth_soft_pred(pred, num_bins):

    # Calculate the bins and spacing
    conf_bins = torch.linspace(0, 1, num_bins+1)[:-1] # Off by one error
    smoothed_pred = torch.zeros_like(pred)

    # Make sure bins are aligned.
    bin_width = conf_bins[1] - conf_bins[0]
    for c_bin in conf_bins:
        # Get the binary region of this confidence interval
        bin_conf_region = (pred >= c_bin) & (pred < (c_bin + bin_width))
        # Break it up into islands
        conf_islands = get_connected_components(bin_conf_region)
        # Iterate through each island, and get the measure for each island.
        for island in conf_islands:
            # Replace this region with the average confidence of the region.
            smoothed_pred[island] = pred[island].mean()

    return smoothed_pred
