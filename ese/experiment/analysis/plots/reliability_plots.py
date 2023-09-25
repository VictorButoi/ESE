# misc imports
import numpy as np
from pydantic import validate_arguments
from typing import List

# ese imports
from ese.experiment.metrics import ECE, ACE, ReCE

# Globally used for which metrics to plot for.
metric_dict = {
        "ECE": ECE,
        "ACE": ACE,
        "ReCE": ReCE
    }

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def build_title(
    title: str, 
    metric: str, 
    met_score: float,
    bin_weightings: List[str]
) -> str:
    title_parts = []
    for weighting in bin_weightings:
        title_parts.append(f"{weighting[0]}{metric}: {met_score:.5f}")
    title += ", ".join(title_parts) + "\n"
    return title


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_subj_reliability_diagram(
    num_bins: int,
    y_axis: str,
    metric: str, 
    subj: dict = None,
    remove_empty_bins: bool = False,
    include_background: bool = False,
    threshold: float = 0.5,
    bin_weightings: List[str] = ["uniform", "weighted"],
    bar_color: str = 'blue',
    show_bin_amounts: bool = False,
    show_diagonal: bool = True,
    ax = None
) -> None:

    # Define the title
    title = f"{y_axis} Reliability Diagram w/ {num_bins} bins:\n"

    calibration_info = metric_dict[metric](
        num_bins=num_bins,
        pred=subj["soft_pred"],
        label=subj["label"],
        measure=y_axis,
        threshold=threshold,
        include_background=include_background
    )

    # Build the title
    title = build_title(
        title,
        metric=metric,
        met_score=calibration_info["score"],
        bin_weightings=bin_weightings
    )

    # Make sure to only use bins where the bin amounts are non-zero
    non_empty_bins = (calibration_info["bin_amounts"] != 0)

    # Get the bins, bin widths, and bin y values for the non-empty bins
    if len(calibration_info["bins"]) > 0:
        graph_bar_heights = calibration_info["avg_freq_per_bin"][non_empty_bins] if remove_empty_bins else calibration_info["avg_freq_per_bin"]
        graph_bin_widths = calibration_info["bin_widths"][non_empty_bins] if remove_empty_bins else calibration_info["bin_widths"]
        graph_bins = calibration_info["bins"][non_empty_bins] if remove_empty_bins else bin
    else:
        graph_bar_heights = np.zeros_like(calibration_info["avg_freq_per_bin"])
        graph_bin_widths = np.zeros_like(graph_bar_heights)
        graph_bins = np.zeros_like(graph_bar_heights)

    # Create the variable width bar plot
    for i in range(len(graph_bar_heights)):
        # Define the bars of the plots
        aligned_bar_position = graph_bins[i] + (graph_bin_widths[i] / 2)
        bar_height = graph_bar_heights[i]
        bar_width = graph_bin_widths[i]

        # Plot the real bars
        actual_bars = ax.bar(
            aligned_bar_position,
            bar_height, 
            width=bar_width,
            edgecolor=bar_color, 
            color=bar_color, 
            alpha=0.8
            )
        # Plot the ideal bars
        ax.bar(
            aligned_bar_position,
            aligned_bar_position,
            width=bar_width,
            hatch='///', 
            edgecolor='red', 
            color='red', 
            alpha=0.2, 
            )

    # Display above the bars how many pixels are in the bar
    if show_bin_amounts:
        for b_idx, bar in enumerate(actual_bars):
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, "{:,}".format(int(calibration_info["bin_amounts"][b_idx])), va='bottom', ha='center', rotation=90)

    # Plot diagonal line
    if show_diagonal:
        ax.plot([0, 1], [0, 1], linestyle='dotted', linewidth=3, color='gray', alpha=0.5)

    # Make sure ax is on
    ax.axis("on")

    # Set title and axis labels
    ax.set_title(title)
    ax.set_ylabel(y_axis)
    ax.set_xlabel("Confidence")

    # Set x and y limits
    ax.set_xlim([0, 1])
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim([0, 1]) 
