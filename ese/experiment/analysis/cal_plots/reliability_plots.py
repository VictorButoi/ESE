# misc imports
import numpy as np
from typing import Any, Literal
from pydantic import validate_arguments
# Local imports
from ese.experiment.metrics.utils import get_bins


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def reliability_diagram(
    calibration_info: dict,
    title: str,
    num_prob_bins: int,
    class_type: Literal["Binary", "Multi-class"],
    plot_type: Literal["bar", "line"],
    ax: Any,
    bar_color: str,
) -> None:
    c_bins, c_bin_widths = get_bins(num_prob_bins=num_prob_bins, int_start=0.0, int_end=1.0)
    if plot_type == "bar":
        # Create the variable width bar plot
        for bin_idx in range(num_prob_bins):
            # Define the bars of the plots
            aligned_bar_position = c_bins[bin_idx] + (c_bin_widths[bin_idx] / 2)
            bar_width = c_bin_widths[bin_idx]
            bar_height = calibration_info["bin_freqs"][bin_idx]
            # Plot the real bars
            ax.bar(
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
    elif plot_type == "line":
        bin_confs = calibration_info["bin_confs"]
        bin_measures = calibration_info["bin_measures"]
        # Remove the empty bins
        nz_bin_confs = bin_confs[bin_confs != 0]
        nz_bin_measures = bin_measures[bin_confs != 0]
        # Plot the lines
        ax.plot(
            nz_bin_confs,
            nz_bin_measures,
            marker='o',
            color=bar_color,
        )
    else:
        raise NotImplementedError(f"Plot type '{plot_type}' not implemented.")

    # Make sure ax is on
    # Make sure ax is on
    ax.axis("on")
    y_label = "Frequency" if class_type == "Binary" else "Accuracy"
    ax.plot([0, 1], [0, 1], linestyle='dotted', linewidth=3, color='gray', alpha=0.5)
    # Set title and axis labels
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Confidence")
    # Set x and y limits
    ax.set_xlim([0, 1])
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim([0, 1]) 
    


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_subj_reliability_diagram(
    num_bins: int,
    metric_name: str,
    metric_dict: dict, 
    subj: dict,
    class_type: Literal["Binary", "Multi-class"],
    plot_type: Literal["bar", "line"],
    bin_weighting: str,
    include_background: bool,
    ax: Any,
) -> None:

    # Define the title
    title = f"{class_type} Reliability Diagram w/ {num_bins} bins:\n"

    calibration_info = metric_dict['func'](
        num_bins=num_bins,
        conf_map=subj["conf_map"],
        pred_map=subj["pred_map"],
        label_map=subj["label"],
        class_type=class_type,
        weighting=bin_weighting,
        include_background=include_background
    )
    
    reliability_diagram(
        title=title,
        calibration_info=calibration_info,
        class_type=class_type,
        plot_type=plot_type,
        metric_name=metric_name,
        bin_weighting=bin_weighting,
        bar_color=metric_dict["color"],
        ax=ax
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_cumulative_reliability_diagram(
    num_bins: int,
    calibration_info: dict,
    metric_name: str, 
    metric_dict: dict,
    class_type: Literal["Binary", "Multi-class"],
    include_background: bool,
    bin_weighting: str,
    ax: Any,
) -> None:

    # Define the title
    title = f"{class_type} Cumulative Reliability Diagram w/ {num_bins} bins:\n"

    reliability_diagram(
        title=title,
        calibration_info=calibration_info,
        class_type=class_type,
        metric_name=metric_name,
        num_bins=num_bins,
        bin_weighting=bin_weighting,
        bar_color=metric_dict["color"],
        ax=ax
    )