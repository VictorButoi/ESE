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

# Globally used for which metrics to plot for.
metric_dict = {
    "ECE": ECE,
    "ACE": ACE,
    "ReCE": ReCE
}

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
    predictions_np = subj['pred_map'].cpu().numpy()

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
    bin_weighting: str,
    ax: Any = None
    ):
    try:
        # Define the number of bins to test.
        num_bins_set = np.arange(10, 60, 10, dtype=int)

        # Keep a dataframe of the error for each metric, bin weighting, and number of bins.
        error_list = []
        conf_map = subj["conf_map"]
        label = subj["label"]

        # Go through each number of bins.
        for metric in metrics:
            for num_bins in num_bins_set:
                # Compute the metrics.
                calibration_info = metric_dict[metric](
                    num_bins=num_bins,
                    conf_map=conf_map,
                    label=label,
                    weighting=bin_weighting
                )
                # Add the error to the dataframe.
                error_list.append({
                    "# Bins": num_bins,
                    "Metric": metric,
                    "Weighting": bin_weighting,
                    "Calibration Error": calibration_info["score"]
                })

        # Convert list to a pandas dataframe
        error_df = pd.DataFrame(error_list)
        
        # Plot the number of bins vs the error using seaborn lineplot, with hue for each metric
        # and style for each weighting.
        ax.axis("on")

        sns.lineplot(
            data=error_df,
            x="# Bins",
            y="Calibration Error",
            hue="Metric",
            palette=metric_colors,
            ax=ax
        )

        # Make the x ticks go every 10 bins, and set the x lim to be between the first and last number of bins.
        x_ticks = np.arange(0, 110, 10)
        ax.set_xticks(x_ticks)
        ax.set_title("Calibration Error vs. Number of Bins")
        ax.set_xlim([num_bins_set[0], num_bins_set[-1]])
    except Exception as e:
        print(e)
        pass


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_avg_variance_vs_numbins(
    subj: dict,
    metrics: List[str],
    metric_colors: dict,
    bin_weighting: str,
    ax: Any = None
):
    try:
        # Define the number of bins to test.
        num_bins_set = np.arange(10, 110, 10, dtype=int)

        # Keep a dataframe of the error for each metric, bin weighting, and number of bins.
        var_list = []
        conf_map = subj["conf_map"]
        label = subj["label"]

        # Go through each number of bins.
        for metric in metrics:
            for num_bins in num_bins_set:
                # Compute the metrics.
                calibration_info = metric_dict[metric](
                    num_bins=num_bins,
                    conf_map=conf_map,
                    label=label
                )
                for c_bin, c_width in zip(calibration_info["bins"], calibration_info["bin_widths"]):
                    # Get the region of image corresponding to the confidence
                    if c_width == 0:
                        bin_conf_region = (conf_map == c_bin)
                    else:
                        bin_conf_region = torch.logical_and(conf_map >= c_bin, conf_map < c_bin + c_width)

                    if bin_conf_region.sum() > 0:                
                        # Add the error to the dataframe.
                        if bin_conf_region.sum() == 1:
                            bin_variance = 0
                        else:
                            if metric == "ReCE":
                                conf_islands = get_connected_components(bin_conf_region)
                                region_conf_scores = torch.Tensor([conf_map[island].mean() for island in conf_islands])
                                bin_variance = region_conf_scores.var().item()
                            else:
                                bin_variance = conf_map[bin_conf_region].var().item()

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

        sns.lineplot(
            data=variances_df,
            x="# Bins",
            y="Bin-Wise Variance",
            hue="Metric",
            palette=metric_colors,
            ax=ax
        )

        # Make the x ticks go every 10 bins, and set the x lim to be between the first and last number of bins.
        x_ticks = np.arange(0, 110, 10)
        ax.set_xticks(x_ticks)
        ax.set_title("Average Bin Variance vs. Number of Bins")
        ax.set_xlim([num_bins_set[0], num_bins_set[-1]])
    except Exception as e:
        print(e)
        pass


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_avg_samplesize_vs_numbins(
    subj: dict,
    metrics: List[str],
    metric_colors: dict,
    bin_weighting: str,
    ax: Any = None
):
    try:   
        # Define the number of bins to test.
        num_bins_set = np.arange(10, 110, 10, dtype=int)

        # Keep a dataframe of the error for each metric, bin weighting, and number of bins.
        amounts_list = []
        conf_map = subj["conf_map"]
        label = subj["label"]

        # Go through each number of bins.
        for metric in metrics:
            for num_bins in num_bins_set:
                # Compute the metrics.
                calibration_info = metric_dict[metric](
                    num_bins=num_bins,
                    conf_map=conf_map,
                    label=label
                )
                for c_bin, c_width in zip(calibration_info["bins"], calibration_info["bin_widths"]):
                    # Get the region of image corresponding to the confidence
                    if c_width == 0:
                        bin_conf_region = (conf_map == c_bin)
                    else:
                        bin_conf_region = torch.logical_and(conf_map >= c_bin, conf_map < c_bin + c_width)

                    if bin_conf_region.sum() > 0:
                        # Add the error to the dataframe.
                        if metric == "ReCE":
                            num_samples = len(get_connected_components(bin_conf_region))
                        else:
                            num_samples = bin_conf_region.sum().item() 
                        
                        assert num_samples > 0, "Number of samples in bin should be greater than 0."
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
        
        # Plot line plot
        sns.lineplot(
            data=variances_df,
            x="# Bins",
            y="#Samples Per Bin",
            hue="Metric",
            palette=metric_colors,
            ax=ax
        )

        # Make the x ticks go every 10 bins, and set the x lim to be between the first and last number of bins.
        x_ticks = np.arange(0, 110, 10)
        ax.set_xticks(x_ticks)
        ax.set_title("# of Bin Samples vs. Number of Bins")
        ax.set_xlim([num_bins_set[0], num_bins_set[-1]])
    except Exception as e:
        print(e)
        pass