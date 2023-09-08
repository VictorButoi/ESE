# misc imports
import numpy as np
from typing import List
import torch
import matplotlib.pyplot as plt

# ionpy imports
from ionpy.util.validation import validate_arguments_init

# ese imports
from ese.experiment.metrics import ECE, ESE, ReCE
from ese.experiment.metrics.utils import reduce_scores

# Globally used for which metrics to plot for.
metric_dict = {
        "ECE": ECE,
        "ESE": ESE,
        "ReCE": ReCE
    }

def build_title(title, metric, bin_scores, bin_amounts, bin_weightings):
    title_parts = []
    for weighting in bin_weightings:
        met_score = reduce_scores(bin_scores.numpy(), bin_amounts.numpy(), weighting)
        title_parts.append(f"{weighting[0]}{metric}: {met_score:.5f}")
    title += ", ".join(title_parts) + "\n"
    return title

@validate_arguments_init
def plot_reliability_diagram(
    bins: torch.Tensor,
    subj: dict = None,
    bin_info: str = None,
    metrics: List[str] = ["ECE", "ESE", "ReCE"],
    title: str = "",
    remove_empty_bins: bool = False,
    bin_weightings: List[str] = ["uniform", "weighted"],
    bin_color: str = 'blue',
    show_bin_amounts: bool = False,
    show_diagonal: bool = True,
    ax = None
) -> None:

    if bin_info is None:
        for met in metrics:
            bin_scores, bin_accs, bin_amounts = metric_dict[met](
                conf_bins=bins,
                pred=subj["soft_pred"],
                label=subj["label"],
            )
            title = build_title(
                title,
                met,
                bin_scores,
                bin_amounts,
                bin_weightings
            )
    else:
        bin_scores, bin_accs, bin_amounts = bin_info
        for met in metrics:
            title = build_title(
                title,
                met,
                bin_scores,
                bin_amounts,
                bin_weightings
            )

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

    # Pad the graph bar heights so it matches the bins
    graph_bins = graph_bins[len(graph_bins) - len(graph_bar_heights):]

    actual_bars = ax.bar(graph_bins, graph_bar_heights, width=interval_size, edgecolor=bin_color, color=bin_color, alpha=0.65, label='Predicted')
    ax.bar(graph_bins, graph_bins, width=interval_size, hatch='///', edgecolor='red', color='red', alpha=0.2, label='Ideal')

    # Display above the bars how many pixels are in the bar
    if show_bin_amounts:
        for b_idx, bar in enumerate(actual_bars):
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, "{:,}".format(int(bin_amounts[b_idx])), va='bottom', ha='center', rotation=90)

    # Plot diagonal line
    if show_diagonal:
        ax.plot([0, 1], [0, 1], linestyle='dotted', linewidth=3, color='gray', alpha=0.5)

    # Make sure ax is on
    ax.axis("on")

    # Set title and axis labels
    ax.set_title(title)
    ax.set_ylabel("Precision")
    ax.set_xlabel("Confidence")

    # Set x and y limits
    ax.set_xlim([0, 1])
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim([0, 1]) 

    # Add a legend
    ax.legend()