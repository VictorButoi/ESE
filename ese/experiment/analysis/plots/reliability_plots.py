# misc imports
import numpy as np
from typing import Any
from pydantic import validate_arguments

# ese imports
from ese.experiment.metrics import ECE, ACE, ReCE
from ese.experiment.metrics.utils import get_bins

# Globally used for which metrics to plot for.
metric_dict = {
    "ECE": ECE,
    "ACE": ACE,
    "ReCE": ReCE
}

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def reliability_diagram(
    title: str,
    calibration_info: dict,
    y_axis: str,
    metric: str, 
    num_bins: int,
    bin_weighting: str,
    ax: Any,
    threshold: float,
    remove_empty_bins: bool,
    include_background: bool,
    bar_color: str,
    show_bin_amounts: bool,
    show_diagonal: bool
):
    # Add the metric to the title
    title += f"{metric}: {calibration_info['score']:.5f} ({bin_weighting})"

    # Make sure to only use bins where the bin amounts are non-zero
    non_empty_bins = (calibration_info["bin_amounts"] != 0)

    # If bins are not defined, redefine them.
    if "bins" not in calibration_info:
        assert metric != "ACE", "ACE requires confidence values to be passed in."
        conf_bins, conf_bin_widths = get_bins(
            metric=metric,
            include_background=include_background,
            threshold=threshold,
            num_bins=num_bins
        )
        calibration_info["bins"] = conf_bins
        calibration_info["bin_widths"] = conf_bin_widths

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


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_subj_reliability_diagram(
    num_bins: int,
    y_axis: str,
    metric: str, 
    subj: dict,
    bin_weighting: str,
    include_background: bool,
    ax: Any,
    remove_empty_bins: bool = True,
    threshold: float = 0.5,
    bar_color: str = 'blue',
    show_bin_amounts: bool = False,
    show_diagonal: bool = True,
) -> None:

    # Define the title
    title = f"{y_axis} Reliability Diagram w/ {num_bins} bins:\n"

    calibration_info = metric_dict[metric](
        num_bins=num_bins,
        conf_map=subj["conf_map"],
        pred_map=subj["pred_map"],
        label=subj["label"],
        measure=y_axis,
        weighting=bin_weighting,
        threshold=threshold,
        include_background=include_background
    )
    
    reliability_diagram(
        title=title,
        calibration_info=calibration_info,
        y_axis=y_axis,
        metric=metric,
        num_bins=num_bins,
        bin_weighting=bin_weighting,
        remove_empty_bins=remove_empty_bins,
        include_background=include_background,
        threshold=threshold,
        bar_color=bar_color,
        show_bin_amounts=show_bin_amounts,
        show_diagonal=show_diagonal,
        ax=ax
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_cumulative_reliability_diagram(
    num_bins: int,
    calibration_info: dict,
    y_axis: str,
    metric: str, 
    include_background: bool,
    bin_weighting: str,
    ax: Any,
    threshold: float = 0.5,
    remove_empty_bins: bool = True,
    bar_color: str = 'blue',
    show_bin_amounts: bool = False,
    show_diagonal: bool = True,
) -> None:

    # Define the title
    title = f"{y_axis} Cumulative Reliability Diagram w/ {num_bins} bins:\n"

    reliability_diagram(
        title=title,
        calibration_info=calibration_info,
        y_axis=y_axis,
        metric=metric,
        num_bins=num_bins,
        bin_weighting=bin_weighting,
        remove_empty_bins=remove_empty_bins,
        include_background=include_background,
        threshold=threshold,
        bar_color=bar_color,
        show_bin_amounts=show_bin_amounts,
        show_diagonal=show_diagonal,
        ax=ax
    )
