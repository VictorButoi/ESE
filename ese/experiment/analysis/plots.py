# misc imports
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Any, List
from pydantic import validate_arguments
from sklearn.metrics import confusion_matrix

# ionpy imports
from ionpy.util.islands import get_connected_components

# ese imports
from ese.experiment.metrics import ECE, ACE, ReCE
from ese.experiment.metrics.utils import reduce_scores

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
    bin_scores: torch.Tensor, 
    bin_amounts: torch.Tensor, 
    bin_weightings: List[str]
) -> str:
    title_parts = []
    for weighting in bin_weightings:
        met_score = reduce_scores(bin_scores.numpy(), bin_amounts.numpy(), weighting)
        title_parts.append(f"{weighting[0]}{metric}: {met_score:.5f}")
    title += ", ".join(title_parts) + "\n"
    return title


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_reliability_diagram(
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
        bin_scores=calibration_info["scores_per_bin"],
        bin_amounts=calibration_info["bin_amounts"],
        bin_weightings=bin_weightings
    )

    # Make sure to only use bins where the bin amounts are non-zero
    non_empty_bins = (calibration_info["bin_amounts"] != 0)

    # Get the bins, bin widths, and bin y values for the non-empty bins
    if len(calibration_info["bins"]) > 0:
        graph_bar_heights = calibration_info["accuracy_per_bin"][non_empty_bins] if remove_empty_bins else calibration_info["accuracy_per_bin"]
        graph_bin_widths = calibration_info["bin_widths"][non_empty_bins] if remove_empty_bins else calibration_info["bin_widths"]
        graph_bins = calibration_info["bins"][non_empty_bins] if remove_empty_bins else bin
    else:
        graph_bar_heights = np.zeros_like(calibration_info["accuracy_per_bin"])
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
def plot_confusion_matrix(
    subj: dict,
    ax = None
    ):
    """
    This function prints and plots the confusion matrix.
    If normalize is set to True, the matrix will display percentages instead of counts.
    """
    
    # Convert PyTorch tensors to NumPy arrays
    ground_truth_np = subj['label'].cpu().numpy()
    predictions_np = subj['hard_pred'].cpu().numpy()

    # Calculate the confusion matrix
    cm = confusion_matrix(ground_truth_np.flatten(), predictions_np.flatten(), labels=[0, 1])

    # Define class labels
    class_labels = ['Background', 'Foreground']

    # Make sure ax is on and grid is off
    ax.axis("on")

    # Plot the confusion matrix using seaborn
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_error_vs_numbins(
    subj: dict,
    metrics: List[str],
    metric_colors: dict,
    bin_weightings: List[str],
    ax: Any = None
    ):

    # Define the number of bins to test.
    num_bins_set = np.arange(10, 110, 10, dtype=int)

    # Keep a dataframe of the error for each metric, bin weighting, and number of bins.
    error_list = []
    pred = subj["soft_pred"]
    label = subj["label"]

    # Go through each number of bins.
    for metric in metrics:
        for bin_weighting in bin_weightings:
            for num_bins in num_bins_set:
                
                # Compute the metrics.
                calibration_info = metric_dict[metric](
                    num_bins=num_bins,
                    pred=pred,
                    label=label
                )
                # Calculate the error.
                error = reduce_scores(
                    scores=calibration_info["scores_per_bin"].numpy(),
                    bin_amounts=calibration_info["bin_amounts"].numpy(),
                    weighting=bin_weighting
                )
                
                if error == 0:
                    print(calibration_info["bin_amounts"].numpy())
                    print(calibration_info["scores_per_bin"].numpy())
                assert error != 0, "Unlikely this happens with scores"
                # Add the error to the dataframe.
                error_list.append({
                    "# Bins": num_bins,
                    "Metric": metric,
                    "Weighting": bin_weighting,
                    "Calibration Error": error
                })

    # Convert list to a pandas dataframe
    error_df = pd.DataFrame(error_list)
    
    # Plot the number of bins vs the error using seaborn lineplot, with hue for each metric
    # and style for each weighting.
    ax.axis("on")
    multiple_bins = len(num_bins_set) > 1
    multiple_weighting = len(bin_weightings) > 1
    hue = None
    style = None

    # Check if you have multiple options
    if multiple_bins and multiple_weighting:
        hue = "Metric"
        style = "Weighting"
    elif multiple_bins:
        hue = "Metric"
    elif multiple_weighting:
        hue = "Weighting"

    sns.lineplot(
        data=error_df,
        x="# Bins",
        y="Calibration Error",
        hue=hue,
        style=style,
        palette=metric_colors,
        ax=ax
    )

    # Make the x ticks go every 10 bins, and set the x lim to be between the first and last number of bins.
    x_ticks = np.arange(0, 110, 10)
    ax.set_xticks(x_ticks)
    ax.set_title("Calibration Error vs. Number of Bins")
    ax.set_xlim([num_bins_set[0], num_bins_set[-1]])


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_avg_variance_vs_numbins(
    subj: dict,
    metrics: List[str],
    metric_colors: dict,
    bin_weightings: List[str],
    ax: Any = None
):
    # Define the number of bins to test.
    num_bins_set = np.arange(10, 110, 10, dtype=int)

    # Keep a dataframe of the error for each metric, bin weighting, and number of bins.
    var_list = []
    pred = subj["soft_pred"]
    label = subj["label"]

    # Go through each number of bins.
    for metric in metrics:
        for bin_weighting in bin_weightings:
            for num_bins in num_bins_set:
                # Compute the metrics.
                calibration_info = metric_dict[metric](
                    num_bins=num_bins,
                    pred=pred,
                    label=label
                )
                for c_bin, c_width in zip(calibration_info["bins"], calibration_info["bin_widths"]):
                    bin_confidences = torch.logical_and(pred >= c_bin, pred < (c_bin + c_width))

                    # Add the error to the dataframe.
                    if metric == "ReCE":
                        conf_islands = get_connected_components(bin_confidences)
                        region_conf_scores = torch.Tensor([pred[island].mean() for island in conf_islands])
                        bin_variance = region_conf_scores.var().item()
                    else:
                        bin_variance = pred[bin_confidences].var().item()

                    # Add the error to the dataframe.
                    var_list.append({
                        "# Bins": num_bins,
                        "Metric": metric,
                        "Weighting": bin_weighting,
                        "Bin-Wise Variance": bin_variance
                    })

    # Convert list to a pandas dataframe
    variances_df = pd.DataFrame(var_list)
    
    # Plot the number of bins vs the error using seaborn lineplot, with hue for each metric
    # and style for each weighting.
    ax.axis("on")
    multiple_bins = len(num_bins_set) > 1
    multiple_weighting = len(bin_weightings) > 1
    hue = None
    style = None

    # Check if you have multiple options
    if multiple_bins and multiple_weighting:
        hue = "Metric"
        style = "Weighting"
    elif multiple_bins:
        hue = "Metric"
    elif multiple_weighting:
        hue = "Weighting"

    sns.lineplot(
        data=variances_df,
        x="# Bins",
        y="Bin-Wise Variance",
        hue=hue,
        style=style,
        palette=metric_colors,
        ax=ax
    )

    # Make the x ticks go every 10 bins, and set the x lim to be between the first and last number of bins.
    x_ticks = np.arange(0, 110, 10)
    ax.set_xticks(x_ticks)
    ax.set_title("Average Bin Variance vs. Number of Bins")
    ax.set_xlim([num_bins_set[0], num_bins_set[-1]])



@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_avg_samplesize_vs_numbins(
    subj: dict,
    metrics: List[str],
    metric_colors: dict,
    bin_weightings: List[str],
    ax: Any = None
):
    # Define the number of bins to test.
    num_bins_set = np.arange(10, 110, 10, dtype=int)

    # Keep a dataframe of the error for each metric, bin weighting, and number of bins.
    amounts_list = []
    pred = subj["soft_pred"]
    label = subj["label"]

    # Go through each number of bins.
    for metric in metrics:
        for bin_weighting in bin_weightings:
            for num_bins in num_bins_set:
                # Compute the metrics.
                calibration_info = metric_dict[metric](
                    num_bins=num_bins,
                    pred=pred,
                    label=label
                )
                for c_bin, c_width in zip(calibration_info["bins"], calibration_info["bin_widths"]):
                    bin_confidences = torch.logical_and(pred >= c_bin, pred < (c_bin + c_width))
                    # Add the error to the dataframe.
                    if metric == "ReCE":
                        num_samples = len(get_connected_components(bin_confidences))
                    else:
                        num_samples = pred[bin_confidences].sum().item() 

                    amounts_list.append({
                        "# Bins": num_bins,
                        "Metric": metric,
                        "Weighting": bin_weighting,
                        "#Samples Per Bin": num_samples
                    })


    # Convert list to a pandas dataframe
    variances_df = pd.DataFrame(amounts_list)
    
    # Plot the number of bins vs the error using seaborn lineplot, with hue for each metric
    # and style for each weighting.
    ax.axis("on")
    multiple_bins = len(num_bins_set) > 1
    multiple_weighting = len(bin_weightings) > 1
    hue = None
    style = None

    # Check if you have multiple options
    if multiple_bins and multiple_weighting:
        hue = "Metric"
        style = "Weighting"
    elif multiple_bins:
        hue = "Metric"
    elif multiple_weighting:
        hue = "Weighting"

    sns.lineplot(
        data=variances_df,
        x="# Bins",
        y="#Samples Per Bin",
        hue=hue,
        style=style,
        palette=metric_colors,
        ax=ax
    )

    # Make the x ticks go every 10 bins, and set the x lim to be between the first and last number of bins.
    x_ticks = np.arange(0, 110, 10)
    ax.set_xticks(x_ticks)
    ax.set_title("# of Bin Samples vs. Number of Bins")
    ax.set_xlim([num_bins_set[0], num_bins_set[-1]])