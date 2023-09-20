# misc imports
import numpy as np
from typing import List
import torch
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

# ionpy imports
from ionpy.metrics.segmentation import dice_score, pixel_accuracy
from ionpy.util.validation import validate_arguments_init
from ionpy.util.islands import get_connected_components

# ese imports
from ese.experiment.metrics import ECE, ESE, ReCE
from ese.experiment.metrics.utils import reduce_scores
from ese.experiment.augmentation import smooth_soft_pred

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
    num_bins: int,
    subj: dict = None,
    bin_info: str = None,
    metrics: List[str] = ["ECE", "ReCE"],
    remove_empty_bins: bool = False,
    bin_weightings: List[str] = ["uniform", "weighted"],
    y_axis: str = "Precision",
    bin_color: str = 'blue',
    show_bin_amounts: bool = False,
    show_diagonal: bool = True,
    ax = None
) -> None:

    # Setup the bins
    bins = torch.linspace(0, 1, num_bins+1)[:-1] # Off by one error

    # Define the title
    title = f"Reliability Diagram w/ {num_bins} bins:\n"

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

    actual_bars = ax.bar(graph_bins, graph_bar_heights, width=interval_size, edgecolor=bin_color, color=bin_color, alpha=0.8, label='Predicted')
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
    ax.set_ylabel(y_axis)
    ax.set_xlabel("Confidence")

    # Set x and y limits
    ax.set_xlim([0, 1])
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim([0, 1]) 

    # Add a legend
    ax.legend()


@validate_arguments_init
def plot_confusion_matrix(
    subj,
    ax=None
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


@validate_arguments_init
def plot_subject_image(
    subj_name,
    subj,
    fig,
    ax
):
    im = ax.imshow(subj["image"], cmap="gray")
    ax.set_title(f"{subj_name}, Image")
    fig.colorbar(im, ax=ax)


@validate_arguments_init
def plot_subj_label(
    subj,
    fig,
    ax
):
    lab = ax.imshow(subj["label"], cmap="gray")
    ax.set_title("Ground Truth")
    fig.colorbar(lab, ax=ax)


@validate_arguments_init
def plot_prediction_probs(
    subj,
    fig,
    ax
):
    pre = ax.imshow(subj["soft_pred"], cmap="gray")
    ax.set_title("Probabilities")
    fig.colorbar(pre, ax=ax)


@validate_arguments_init
def plot_pred(
    subj,
    fig,
    ax
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


@validate_arguments_init
def plot_smoothed_pred(
    subj,
    num_bins,
    fig,
    ax
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


@validate_arguments_init
def plot_error_vs_numbins(
    subj,
    metrics,
    bin_weightings,
    ax=None
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
                bin_scores, _, bin_amounts = metric_dict[metric](
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


def plot_ece_map(
    subj,
    fig,
    ax,
):
    # Calculate the per-pixel accuracy and where the foreground regions are.
    acc_per_pixel = (subj['label'] == subj['hard_pred']).float()
    foreground = subj['hard_pred'].bool()

    # Set the regions of the image corresponding to groundtruth label.
    ece_map = np.zeros_like(subj['label'])
    ece_map[foreground] = (subj['soft_pred'] - acc_per_pixel)[foreground]

    # Get the bounds for visualization
    ece_abs_max = np.max(np.abs(ece_map))
    ece_vmin, ece_vmax = -ece_abs_max, ece_abs_max

    # Show the ece map
    ce_im = ax.imshow(ece_map, cmap="RdBu_r", interpolation="None", vmax=ece_vmax, vmin=ece_vmin)
    ax.set_title("Pixel-wise Cal Error")
    fig.colorbar(ce_im, ax=ax)


def plot_rece_map(
    subj,
    num_bins,
    fig,
    ax,
    average=False
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
    ese_im = ax.imshow(rece_map, cmap="RdBu_r", interpolation="None", vmax=rece_vmax, vmin=rece_vmin)

    if average:
        ax.set_title("Averaged Region-wise Cal Error")
    else:
        ax.set_title("Region-wise Cal Error")

    fig.colorbar(ese_im, ax=ax)

