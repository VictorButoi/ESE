#i
import numpy as np
import ionpy

# ionpy imports
from ionpy.util.validation import validate_arguments_init

# ese imports
from ese.experiment.metrics import ECE, ESE

# Globally used for which metrics to plot for.
metric_dict = {
        "dice": ionpy.metrics.dice_score,
        "accuracy": ionpy.metrics.pixel_accuracy
    }

@validate_arguments_init
def plot_reliability_diagram(
    bins: np.ndarray,
    subj: dict = None,
    metric: str = None,
    bin_accs: np.ndarray = None,
    bin_amounts: np.ndarray = None,
    title: str = "",
    remove_empty_bins: bool = False,
    bin_weighting: str = "proportional",
    bin_color: str = 'blue',
    ax = None
) -> None:

    if bin_accs is None:
        if metric == "ESE":
            # This returns a numpy array with the measure per confidence interval
            _, bin_accs, bin_amounts = ESE(
                bins=bins,
                pred=subj["soft_pred"],
                label=subj["label"],
            )
            bin_props = bin_amounts / np.sum(bin_amounts)

            if bin_weighting == "proportional":
                w_ese_score = np.average(bin_accs, weights=bin_props)
                title += f"wESE: {w_ese_score:.5f}"

            elif bin_weighting == "uniform":
                uniform_weights = np.ones(len(bin_accs)) / len(bin_accs)
                u_ese_score = np.average(bin_accs, weights=uniform_weights)

                title += f"uESE: {u_ese_score:.5f}"

            elif bin_weighting == "both":
                w_ese_score = np.average(bin_accs, weights=bin_props)

                uniform_weights = np.ones(len(bin_accs)) / len(bin_accs)
                u_ese_score = np.average(bin_accs, weights=uniform_weights)

                title += f"wESE: {w_ese_score:.5f}, uESE: {u_ese_score:.5f}"

        elif metric == "ECE":
            # This returns a numpy array with the measure per confidence interval
            _, bin_accs, bin_amounts = ECE(
                bins=bins,
                pred=subj["soft_pred"],
                label=subj["label"],
            )
            if np.sum(bin_amounts) == 0:
                ece_score = 0 
            else:
                bin_props = bin_amounts / np.sum(bin_amounts)
                ece_score = np.average(bin_accs, weights=bin_props)

            title += f"ECE: {ece_score:.5f}"

    # Get rid of the empty bins.
    interval_size = bins[1] - bins[0]
    aligned_bins = bins + (interval_size / 2) # shift bins to center

    # Make sure to only use bins where the bin amounts are non-zero
    if remove_empty_bins:
        graph_bar_heights = bin_accs[bin_amounts != 0]
        graph_bins = aligned_bins[bin_amounts != 0]
    else:
        graph_bar_heights = bin_accs
        graph_bins = aligned_bins

    # Ideal boxs
    ax.bar(graph_bins, graph_bins, width=interval_size, color='red', alpha=0.2)
    ax.bar(graph_bins, graph_bar_heights, width=interval_size, color=bin_color, alpha=0.5)

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