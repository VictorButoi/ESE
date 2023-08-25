import numpy as np
import ionpy
import matplotlib.pyplot as plt

# ese imports
from ese.experiment.metrics import ECE, ESE
import ese.experiment.analysis.vis as vis

# Globally used for which metrics to plot for.
metric_dict = {
        "dice": ionpy.metrics.dice_score,
        "accuracy": ionpy.metrics.pixel_accuracy
    }


def plot_reliability_diagram(
    bins,
    subj=None,
    metric=None,
    bar_heights=None,
    title="",
    bin_weighting="proportional",
    bin_color='blue',
    ax=None
):

    if bar_heights is None:
        if metric == "ESE":
            # This returns a numpy array with the measure per confidence interval
            bar_heights, bin_amounts = ESE(
                bins=bins,
                pred=subj["soft_pred"],
                label=subj["label"],
            )
            bin_props = bin_amounts / np.sum(bin_amounts)

            if bin_weighting == "proportional":
                w_ese_score = np.average(bar_heights, weights=bin_props)
                title += f"wESE: {w_ese_score:.5f}"

            elif bin_weighting == "uniform":
                uniform_weights = np.ones(len(bar_heights)) / len(bar_heights)
                u_ese_score = np.average(bar_heights, weights=uniform_weights)

                title += f"uESE: {u_ese_score:.5f}"

            elif bin_weighting == "both":
                w_ese_score = np.average(bar_heights, weights=bin_props)

                uniform_weights = np.ones(len(bar_heights)) / len(bar_heights)
                u_ese_score = np.average(bar_heights, weights=uniform_weights)

                title += f"wESE: {w_ese_score:.5f}, uESE: {u_ese_score:.5f}"

        elif metric == "ECE":
            # This returns a numpy array with the measure per confidence interval
            bar_heights, bin_amounts = ECE(
                bins=bins,
                pred=subj["soft_pred"],
                label=subj["label"],
            )
            if np.sum(bin_amounts) == 0:
                ece_score = 0 
            else:
                bin_props = bin_amounts / np.sum(bin_amounts)
                ece_score = np.average(bar_heights, weights=bin_props)

            title += f"ECE: {ece_score:.5f}"

    interval_size = bins[1] - bins[0]
    aligned_bins = bins + (interval_size / 2) # shift bins to center

    ax.bar(aligned_bins, aligned_bins, width=interval_size, color='red', alpha=0.2)
    ax.bar(aligned_bins, bar_heights, width=interval_size, color=bin_color, alpha=0.5)

    # Plot diagonal line
    ax.plot([0, 1], [0, 1], linestyle='dotted', linewidth=3, color='gray')

    # Set title and axis labels
    ax.set_title(title)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Confidence")

    # Set x and y limits
    ax.set_xlim([0, 1])
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim([0, 1]) 

    return aligned_bins, bar_heights
