import numpy as np
import ionpy
import matplotlib.pyplot as plt

# ese imports
from ese.experiment.metrics import ECE, ESE

# Globally used for which metrics to plot for.
metric_dict = {
        "dice": ionpy.metrics.dice_score,
        "accuracy": ionpy.metrics.pixel_accuracy
    }


def plot_reliability_diagram(
    subj,
    metric,
    bins,
    title="",
    bin_weighting="proportional",
    bin_color='blue',
    ax=None
):

    if metric == "ESE":
        # This returns a numpy array with the measure per confidence interval
        bar_heights, bin_amounts = ESE(
            bins=bins,
            pred=subj["soft_pred"],
            label=subj["label"],
        )

        if bin_weighting == "proportional":
            ese_score = np.average(bar_heights, weights=bin_amounts)
            title += f"wESE: {ese_score:.5f}"

        elif bin_weighting == "uniform":
            ese_score = np.mean(bar_heights)
            title += f"uESE: {ese_score:.5f}"

        elif bin_weighting == "both":
            ese_score = np.average(bar_heights, weights=bin_amounts)
            ese_score = np.mean(bar_heights)
            title += f"wESE: {ese_score:.5f}, uESE: {ese_score:.5f}"

    elif metric == "ECE":
        # This returns a numpy array with the measure per confidence interval
        bar_heights, bin_amounts = ECE(
            bins=bins,
            pred=subj["soft_pred"],
            label=subj["label"]
        )
        ece_score = np.average(bar_heights, weights=bin_amounts)
        title += f"ECE: {ece_score:.5f}"

    interval_size = bins[1] - bins[0]
    aligned_bins = bins + (interval_size / 2) # shift bins to center

    ax.bar(aligned_bins, aligned_bins, width=interval_size, color='red', alpha=0.2)
    ax.bar(aligned_bins, bar_heights, width=interval_size, color=bin_color, alpha=0.5)

    # Plot diagonal line
    ax.plot([0, 1], [0, 1], linestyle='dotted', linewidth=3, color='gray')

    # Set title and axis labels
    ax.set_title(title)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Confidence")

    # Set x and y limits
    ax.set_xlim([0, 1])
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim([0, 1]) 


def subject_plot(
    subject_dict, 
    num_bins,
    plot_metric="ESE",
    bin_weighting="proportional",
    num_rows=3,
    show_subj=False,
    show_uncertainty_plots=False
    ):

    assert not(show_uncertainty_plots and not show_subj), \
    "If you're not looking at subjects you can't look at uncertainty plots."

    
    ese_prefix = "w" if bin_weighting == "proportional" else "u"
    # Calculate the bins and spacing
    bins = np.linspace(0, 1, num_bins+1)[:-1] # Off by one error

    # if you want to see the subjects and predictions
    if show_subj:
        num_cols = 7 if show_uncertainty_plots else 5
        plt.rcParams.update({'font.size': 12})  
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
            # Setup the plot for each subject.
            f, axarr = plt.subplots(
                nrows=1,
                ncols=num_cols,
                figsize=(6 * num_cols, 6)
            )

            im = axarr[0].imshow(subj["image"], cmap="gray")
            axarr[0].axis("off")
            axarr[0].set_title(f"Subject #{subj_idx}, Image")
            f.colorbar(im, ax=axarr[0])

            lab = axarr[1].imshow(subj["label"], cmap="gray")
            axarr[1].axis("off")
            axarr[1].set_title(f"Subject #{subj_idx}, Ground Truth")
            f.colorbar(lab, ax=axarr[1])

            pre = axarr[2].imshow(subj["soft_pred"], cmap="gray")
            axarr[2].axis("off")
            axarr[2].set_title(f"Subject #{subj_idx}, Prediction, Dice: {subj['dice_score']:.3f}")
            f.colorbar(pre, ax=axarr[2])

            # Show different kinds of statistics about your subjects.
            plot_reliability_diagram(
                subj,
                metric="ECE",
                bins=bins,
                bin_color="blue",
                ax=axarr[3]
            )

            # Show different kinds of statistics about your subjects.
            plot_reliability_diagram(
                subj,
                metric="ESE",
                bins=bins,
                bin_color="green",
                ax=axarr[4]
            )

            # Show two more plots if you want to see the uncertainty.
            if show_uncertainty_plots:
                # Look at the pixelwise error.
                axarr[subj_row, 5].imshow(vis.pixelwise_unc_map(), cmap="plasma")
                axarr[subj_row, 5].axis("off")
                axarr[subj_row, 5].set_title("Pixel-wise Calibration Error")

                # Plot our region-wise uncertainty metric.
                axarr[subj_row, 6].imshow(vis.ese_unc_map(), cmap="plasma")
                axarr[subj_row, 6].axis("off")
                axarr[subj_row, 6].set_title("Region-wise Calibration Error")

        else:  
            num_per_row = len(subject_dict) // num_rows

            subj_row = subj_idx // num_per_row
            subj_col = subj_idx % num_per_row

            plot_reliability_diagram(
                subj,
                metric=plot_metric,
                bins=bins,
                title=f"Subject #{subj_idx}",
                bin_color="blue",
                ax=axarr[subj_row, subj_col]
            )


    plt.show()


def aggregate_plot(
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
