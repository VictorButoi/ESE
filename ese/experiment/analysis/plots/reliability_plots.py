# local imports
from .utils import build_rel_axes

# misc imports
import numpy as np
from typing import Any, Literal
from pydantic import validate_arguments


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def reliability_diagram(
    title: str,
    calibration_info: dict,
    class_type: Literal["Binary", "Multi-class"],
    plot_type: Literal["bar", "line"],
    metric_name: str, 
    bin_weighting: str,
    ax: Any,
    bar_color: str,
) -> None:
    # Add the metric to the title
    title += f"{metric_name}: {calibration_info['cal_score']:.5f} ({bin_weighting})"

    if plot_type == "bar":
        assert not calibration_info["label-wise"], "Label-wise reliability diagrams not implemented for bar plots."
        # Create the variable width bar plot
        for i in range(len(calibration_info["bins"])):
            # Define the bars of the plots
            aligned_bar_position = calibration_info["bins"][i] + (calibration_info["bin_widths"][i] / 2)
            bar_width = calibration_info["bin_widths"][i]
            bar_height = calibration_info["bin_measures"][i]
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
        if calibration_info["label-wise"]:
            num_labels = len(calibration_info["lab_bin_confs"])
            for i in range(num_labels):
                lab_bin_confs = calibration_info["lab_bin_confs"][i]
                lab_bin_measures = calibration_info["lab_bin_measures"][i]
                # Remove the empty bins
                nz_lab_bin_confs = lab_bin_confs[lab_bin_confs != 0] 
                nz_lab_bin_measures = lab_bin_measures[lab_bin_confs != 0]
                # Plot the lines
                ax.plot(
                    nz_lab_bin_confs,
                    nz_lab_bin_measures,
                    marker='o',
                    label=f"Label {i}"
                )
        else:
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
    build_rel_axes(title, class_type, ax)

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