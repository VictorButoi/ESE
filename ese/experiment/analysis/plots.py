import numpy as np
import ionpy
import matplotlib.pyplot as plt

# ese imports
from ese.experiment.metrics import ESE

# Globally used for which metrics to plot for.
metric_dict = {
        "dice": ionpy.metrics.dice_score,
        "accuracy": ionpy.metrics.pixel_accuracy
    }


def plot_reliability_diagram(
    subj,
    metric,
    bins,
    sub_idx=0,
    bin_color='blue',
    ax=None
):

    if metric == "ESE":
        # This returns a numpy array with the measure per confidence interval
        ese_per_bin, bin_amounts = ESE(
            subj["soft_pred"],
            subj["label"],
            metric_dict[metric],
            bins
        )
        ese_score = np.average(ese_per_bin, weights=bin_amounts)
        title = f"Subject #{subj_idx}, ESE: {ese_score:.5f}"

    elif metric == "ECE":
        # This returns a numpy array with the measure per confidence interval
        ese_per_bin, bin_amounts = ESE(
            subj["soft_pred"],
            subj["label"],
            metric_dict[metric],
            bins
        )
        ece_score = np.average(ese_per_bin, weights=bin_amounts)
        title = f"Subject #{subj_idx}, ECE: {ece_score:.5f}"

    interval_size = bins[1] - bins[0]
    aligned_bins = bins + (interval_size / 2) # shift bins to center

    ax.bar(aligned_bins, aligned_bins, width=interval_size, color='red', alpha=0.2)
    ax.bar(aligned_bins, bar_heights, width=interval_size, color=bin_color, alpha=0.5)

    # Plot diagonal line
    ax.plot([0, 1], [0, 1], linestyle='dotted', linewidth=3, color='gray')

    # Set title and axis labels
    ax.set_title(title)
    ax.set_ylabel("Average Accuracy")
    ax.set_xlabel("Confidence")

    # Set x and y limits
    ax.set_xlim([0, 1])
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim([0, 1]) 


def subject_plot(
    subject_dict, 
    num_bins,
    metric="accuracy",
    bin_weighting="proportional",
    num_rows=3,
    show_subj=False,
    show_uncertainty_plots=False
    ):

    assert not(uncertainty_plots and not show_subj), \
    "If you're not looking at subjects you can't look at uncertainty plots."

    
    ese_prefix = "w" if bin_weighting == "proportional" else "u"
    # Calculate the bins and spacing
    bins = np.linspace(0, 1, num_bins+1)[:-1] # Off by one error

    # if you want to see the subjects and predictions
    if show_subj:
        num_cols = 7 if show_uncertainty_plots else 5

        plt.rcParams.update({'font.size': 12})  
        _, axarr = plt.subplots(
            nrows=len(subject_dict),
            ncols=num_cols,
            figsize=(6 * num_cols, len(subject_dict)*6)
        )
    # Otherwise clump the graphs together
    else:
        plt.rcParams.update({'font.size': 30})
        _, axarr = plt.subplots(
            nrows=num_rows,
            ncols=len(subject_dict) // num_rows,
            figsize=(len(subject_dict) * 6, num_rows*15)
        )
        
    for subj_idx, subj in enumerate(subject_dict):

        if show_subj:
            axarr[subj_idx, 0].imshow(subj["image"], cmap="gray")
            axarr[subj_idx, 0].axis("off")
            axarr[subj_idx, 0].set_title("Image")

            axarr[subj_idx, 1].imshow(subj["label"], cmap="gray")
            axarr[subj_idx, 1].axis("off")
            axarr[subj_idx, 1].set_title("Ground Truth")

            axarr[subj_idx, 2].imshow(subj["soft_pred"], cmap="gray")
            axarr[subj_idx, 2].axis("off")
            axarr[subj_idx, 2].set_title(f"Prediction, Dice: {subj['dice_score']:.3f}")

            subj_row = subj_idx
            subj_col = 3
        else:  
            num_per_row = len(subject_dict) // num_rows
            subj_row = subj_idx // num_per_row
            subj_col = subj_idx % num_per_row

        # Show different kinds of statistics about your subjects.
        plot_reliability_diagram(
            subj,
            subj_idx,
            metric="ECE",
            ax=axarr[subj_row, subj_col],
            bin_color="blue"
        )

        # Show different kinds of statistics about your subjects.
        plot_reliability_diagram(
            subj,
            subj_idx,
            metric="ESE",
            bin_color="green",
            ax=axarr[subj_row, subj_col],
        )

        # Show two more plots if you want to see the uncertainty.
        if show_uncertainty_plots:
            # Look at the pixelwise error.
            axarr[subj_row, 6].imshow(vis.pixelwise_unc_map(), cmap="plasma")
            axarr[subj_row, 6].axis("off")
            axarr[subj_row, 6].set_title("Pixel-wise Calibration Error")

            # Plot our region-wise uncertainty metric.
            axarr[subj_row, 7].imshow(vis.ese_unc_map(), cmap="plasma")
            axarr[subj_row, 7].axis("off")
            axarr[subj_row, 7].set_title("Region-wise Calibration Error")


    plt.show()


def aggregate_plot(
    plot_type,
    subject_dict,
    num_bins,
    color="blue",
    metric="accuracy",
    bin_weighting="proportional"
):
    _, axarr = plt.subplots(1, 1, figsize=(15, 10))

    ese_prefix = "w" if bin_weighting == "proportional" else "u"
    # Calculate the bins and spacing
    bins = np.linspace(0, 1, num_bins+1)[:-1] # Off by one error

    ese_info = [ESE(subj["soft_pred"], subj["label"], metric_dict[metric], bins) for subj in subject_dict]
    ese_per_bin = np.mean([ese[0] for ese in ese_info], axis=0)

    if bin_weighting == "proportional":
        bin_amounts = np.sum([ese[1] for ese in ese_info], axis=0)
        bin_weights = bin_amounts / np.sum(bin_amounts)
    elif bin_weighting == "uniform":
        bin_weights = np.ones(num_bins) / num_bins
    else:
        raise ValueError("Invalid bin weighting.")

    if plot_type == "ese_plot":
        plot_reliability_diagram(
            bins,
            ese_per_bin,
            ax=axarr,
            title=f"Total {ese_prefix}ESE: {np.average(ese_per_bin, weights=bin_weights):.5f}",
            x_label="Confidence",
            y_label=f"Average {metric}",
            bin_color=color
        )
    else:
        axarr.hist(
            bins,
            bin_proportions,
            ax=axarr,
            title=f"Pred Confidence Histogram",
            x_label="Confidence",
            y_label=f"Amount",
            bin_color=color
        )
