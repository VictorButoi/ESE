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
from ionpy.metrics.segmentation import dice_score, pixel_accuracy

# ese imports
from ese.experiment.metrics import ECE, ACE, ReCE
from ese.experiment.metrics.utils import reduce_scores
from ese.experiment.augmentation import smooth_soft_pred

# Globally used for which metrics to plot for.
metric_dict = {
        "ECE": ECE,
        "ACE": ACE,
        "ReCE": ReCE
    }

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def build_title(title, metric, bin_scores, bin_amounts, bin_weightings):
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
    bin_info: Any = None,
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

    if bin_info is None:
        bins, bin_widths, bin_scores, bin_y_vals, bin_amounts = metric_dict[metric](
            num_bins=num_bins,
            pred=subj["soft_pred"],
            label=subj["label"],
            measure=y_axis,
            threshold=threshold,
            include_background=include_background
        )
    else:
        bin_scores, bin_y_vals, bin_amounts = bin_info

    # Build the title
    title = build_title(
        title,
        metric,
        bin_scores,
        bin_amounts,
        bin_weightings
    )

    # Make sure to only use bins where the bin amounts are non-zero
    non_empty_bins = (bin_amounts != 0)

    graph_bar_heights = bin_y_vals[non_empty_bins] if remove_empty_bins else bin_y_vals
    graph_bin_widths = bin_widths[non_empty_bins] if remove_empty_bins else bin_widths
    graph_bins = bins[non_empty_bins] if remove_empty_bins else bins

    print("Graph Bins: ", graph_bins)
    print("Graph Bars: ", graph_bar_heights)
    print("Graph Widths: ", graph_bin_widths)
    print()

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
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, "{:,}".format(int(bin_amounts[b_idx])), va='bottom', ha='center', rotation=90)

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
def plot_subject_image(
    subj_name: str,
    subj: dict,
    fig: Any = None,
    ax: Any = None
):
    im = ax.imshow(subj["image"], cmap="gray")
    ax.set_title(f"{subj_name}, Image")
    fig.colorbar(im, ax=ax)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_subj_label(
    subj: dict,
    fig: Any = None,
    ax: Any = None
):
    lab = ax.imshow(subj["label"], cmap="gray")
    ax.set_title("Ground Truth")
    fig.colorbar(lab, ax=ax)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_prediction_probs(
    subj: dict,
    fig: Any = None,
    ax: Any = None
):
    pre = ax.imshow(subj["soft_pred"], cmap="gray")
    ax.set_title("Probabilities")
    fig.colorbar(pre, ax=ax)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_pred(
    subj: dict,
    fig: Any = None,
    ax: Any = None
):
    # Plot the pixel-wise prediction
    hard_im = ax.imshow(subj["hard_pred"], cmap="gray")

    # Expand extra dimensions for metrics
    pred = subj["hard_pred"][None, None, ...]
    label = subj["label"][None, None, ...]

    # Calculate the dice score
    dice = dice_score(pred, label)
    ax.set_title(f"Prediction, Dice: {dice:.3f}")

    fig.colorbar(hard_im, ax=ax)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_smoothed_pred(
    subj: dict,
    num_bins: int,
    fig: Any = None,
    ax: Any = None
):
    # Plot the processed region-wise prediction
    smoothed_prediction = smooth_soft_pred(subj["soft_pred"], num_bins)
    hard_smoothed_prediction = (smoothed_prediction > 0.5).float()
    smooth_im = ax.imshow(hard_smoothed_prediction, cmap="gray")

    # Expand extra dimensions for metrics
    smoothed_pred = hard_smoothed_prediction[None, None, ...]
    label = subj["label"][None, None, ...]

    # Calculate the dice score
    smoothed_dice = dice_score(smoothed_pred, label)
    ax.set_title(f"Smoothed Prediction, Dice: {smoothed_dice:.3f}")

    fig.colorbar(smooth_im, ax=ax)


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
                _, _, bin_scores, _, bin_amounts = metric_dict[metric](
                    num_bins=num_bins,
                    pred=pred,
                    label=label
                )
                # Calculate the error.
                error = reduce_scores(
                    scores=bin_scores.numpy(),
                    bin_amounts=bin_amounts.numpy(),
                    weighting=bin_weighting
                )
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
        ax=ax
    )

    # Make the x ticks go every 10 bins, and set the x lim to be between the first and last number of bins.
    x_ticks = np.arange(0, 110, 10)
    ax.set_xticks(x_ticks)
    ax.set_title("Calibration Error vs. Number of Bins")
    ax.set_xlim([num_bins_set[0], num_bins_set[-1]])


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_ece_map(
    subj: dict,
    fig: Any,
    ax: Any,
):
    # Copy the soft and hard predictions
    soft_pred = subj['soft_pred'].clone()
    hard_pred = subj['hard_pred'].clone()

    # Calculate the per-pixel accuracy and where the foreground regions are.
    acc_per_pixel = (subj['label'] == hard_pred).float()
    pred_foreground = hard_pred.bool()

    # Set the regions of the image corresponding to groundtruth label.
    ece_map = np.zeros_like(subj['label'])
    ece_map[pred_foreground] = (soft_pred - acc_per_pixel)[pred_foreground]

    # Get the bounds for visualization
    ece_abs_max = np.max(np.abs(ece_map))
    ece_vmin, ece_vmax = -ece_abs_max, ece_abs_max

    # Show the ece map
    ce_im = ax.imshow(ece_map, cmap="RdBu_r", interpolation="None", vmax=ece_vmax, vmin=ece_vmin)
    ax.set_title("Pixel-wise Cal Error")
    fig.colorbar(ce_im, ax=ax)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_rece_map(
    subj: dict,
    num_bins: int,
    average: bool = False,
    fig: Any = None,
    ax: Any = None
):
    # Get the confidence bins
    conf_bins = torch.linspace(0, 1, num_bins+1)[:-1] # Off by one error

    pred = subj['soft_pred']
    rece_map = np.zeros_like(pred)

    # Make sure bins are aligned.
    bin_width = conf_bins[1] - conf_bins[0]
    for c_bin in conf_bins:

        # Get the binary region of this confidence interval
        bin_conf_region = (pred >= c_bin) & (pred < (c_bin + bin_width))

        # Break it up into islands
        conf_islands = get_connected_components(bin_conf_region)

        # Iterate through each island, and get the measure for each island.
        for island in conf_islands:

            # Get the label corresponding to the island and simulate ground truth and make the right shape.
            label_region = subj["label"][island][None, None, ...]
            pseudo_pred = torch.ones_like(label_region)

            # If averaging, then everything in one island will get the same score, otherwise pixelwise.
            if average:
                # Calculate the accuracy and mean confidence for the island.
                region_accuracies  = pixel_accuracy(pseudo_pred , label_region)
                region_confidences = pred[island].mean()
            else:
                region_accuracies = (pseudo_pred == label_region).squeeze().float()
                region_confidences = pred[island]

            # Place the numbers in the island.
            rece_map[island] = (region_confidences - region_accuracies)

    # Get the bounds for visualization
    rece_abs_max = np.max(np.abs(rece_map))
    rece_vmin, rece_vmax = -rece_abs_max, rece_abs_max
    rece_im = ax.imshow(rece_map, cmap="RdBu_r", interpolation="None", vmax=rece_vmax, vmin=rece_vmin)

    if average:
        ax.set_title("Averaged Region-wise Cal Error")
    else:
        ax.set_title("Region-wise Cal Error")

    fig.colorbar(rece_im, ax=ax)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_variance_per_bin(
    subj: dict,
    num_bins: int,
    metrics: List[str],
    metric_colors: dict,
    ax: Any = None
):
    # If conf_bins is not predefined, create them. 
    conf_bins = torch.linspace(0, 1, num_bins+1)[:-1] # Off by one error

    pred = subj['soft_pred']

    # Get the confidence bins
    bin_width = conf_bins[1] - conf_bins[0]

    # Get the regions of the prediction corresponding to each bin of confidence.
    confidence_regions = {c_bin.item(): torch.logical_and(pred >= c_bin, pred < (c_bin + bin_width)).bool() for c_bin in conf_bins}

    conf_list = []
    # Go through each bin, starting at the back so that we don't have to run connected components
    for metric in metrics:
        for c_bin in conf_bins:
            # Get the region of image corresponding to the confidence
            bin_conf_region = confidence_regions[c_bin.item()].bool()
            # If there are some pixels in this confidence bin.
            if bin_conf_region.sum() != 0:
                if metric == "ReCE":
                    # Iterate through each island, and get the measure for each island
                    conf_islands = get_connected_components(bin_conf_region)
                    if len(conf_islands) == 1:
                        confidences_var = 0.0
                    else:
                        island_confidences = torch.stack([pred[island].mean() for island in conf_islands])
                        confidences_var = island_confidences.var().item()
                elif metric == "ECE":
                    confidences_var = pred[bin_conf_region].var().item()
                else:
                    raise ValueError(f"Metric {metric} not supported.")
                # Add the variance to the dictionary
                conf_list.append({
                    "Bin": np.round(c_bin.item(), 2),
                    "Variance": confidences_var,
                    "metric": metric
                })

    # Make a dataframe from the list
    conf_df = pd.DataFrame(conf_list)

    ax.axis("on")
    sns.barplot(x="Bin", y="Variance", data=conf_df, hue="metric", ax=ax, orient="v", palette=metric_colors)
    ax.set_title(f"Variance per Bin")