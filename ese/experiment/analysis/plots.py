# misc imports
import numpy as np
from typing import List
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ionpy imports
from ionpy.util.validation import validate_arguments_init
from ionpy.metrics import dice_score

# ese imports
from ese.experiment.metrics import ECE, ESE, ReCE
from ese.experiment.metrics.utils import reduce_scores
from ese.experiment.augmentation import smooth_soft_pred
import ese.experiment.analysis.vis as vis

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
    title: str = "",
    remove_empty_bins: bool = False,
    bin_weightings: List[str] = ["uniform", "weighted"],
    bin_color: str = 'blue',
    show_bin_amounts: bool = False,
    show_diagonal: bool = True,
    ax = None
) -> None:

    # Setup the bins
    print(num_bins)
    bins = torch.linspace(0, 1, num_bins+1)[:-1] # Off by one error

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
    ax.set_ylabel("Precision")
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
    ax.grid(False) 

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
def plot_pixelwise_pred(
    subj,
    fig,
    ax
):
    def_pred = ax.imshow(subj["hard_pred"], cmap="gray")

    dice = dice_score(subj["soft_pred"], subj["label"])
    ax.set_title(f"Pixel-wise Pred, Dice: {dice:.3f}")

    fig.colorbar(def_pred, ax=ax)


@validate_arguments_init
def plot_regionwise_pred(
    subj,
    num_bins,
    fig,
    ax
):
    smoothed_prediction = smooth_soft_pred(subj["soft_pred"], num_bins)
    hard_smoothed_prediction = (smoothed_prediction > 0.5).float()

    smooth_pred = ax.imshow(hard_smoothed_prediction, cmap="gray")

    smoothed_dice = dice_score(smoothed_prediction, subj["label"])
    ax.set_title(f"Region-wise Pred, Dice: {smoothed_dice:.3f}")

    fig.colorbar(smooth_pred, ax=ax)


@validate_arguments_init
def plot_error_vs_numbins(
    subj,
    metrics,
    bin_weightings,
    ax=None
    ):

    num_bins_set = np.linspace(10, 100, 10, dtype=int)


def plot_ece_map(
    subj,
    fig,
    ax,
):
    # Look at the pixelwise error.
    ece_map = vis.ECE_map(subj)

    # Get the bounds for visualization
    ece_abs_max = np.max(np.abs(ece_map))
    ece_vmin, ece_vmax = -ece_abs_max, ece_abs_max
    ce_im = ax.imshow(ece_map, cmap="RdBu_r", interpolation="None", vmax=ece_vmax, vmin=ece_vmin)
    ax.set_title("Pixel-wise Calibration Error")
    fig.colorbar(ce_im, ax=ax)


def plot_rece_map(
    subj,
    num_bins,
    fig,
    ax
):
    # Look at the regionwise error.
    rece_map = vis.ReCE_map(subj, num_bins)

    # Get the bounds for visualization
    rece_abs_max = np.max(np.abs(rece_map))
    rece_vmin, rece_vmax = -rece_abs_max, rece_abs_max
    ese_im = ax.imshow(rece_map, cmap="RdBu_r", interpolation="None", vmax=rece_vmax, vmin=rece_vmin)
    ax.set_title("Region-wise Calibration Error")
    fig.colorbar(ese_im, ax=ax)

