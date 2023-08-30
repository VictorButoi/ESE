import numpy as np
import ionpy
import matplotlib.pyplot as plt

# ese imports
from ese.experiment.analysis.plots import plot_reliability_diagram
from ese.experiment.metrics import ESE
import ese.experiment.analysis.vis as vis

# Globally used for which metrics to plot for.
metric_dict = {
        "dice": ionpy.metrics.dice_score,
        "accuracy": ionpy.metrics.pixel_accuracy
    }

def subject_plot(
    subject_dict, 
    num_bins,
    bin_weighting="both",
    show_bin_amounts=False
    ):
    
    # Calculate the bins and spacing
    ece_bins = np.linspace(0.5, 1, (num_bins//2)+1)[:-1] # Off by one error
    ese_bins = np.linspace(0, 1, num_bins+1)[:-1] # Off by one error

    # if you want to see the subjects and predictions
    plt.rcParams.update({'font.size': 12})  
        
    for subj_idx, subj in enumerate(subject_dict):

        # Setup the plot for each subject.
        f, axarr = plt.subplots(
            nrows=2,
            ncols=4,
            figsize=(24, 12)
        )

        im = axarr[0, 0].imshow(subj["image"], cmap="gray")
        axarr[0, 0].axis("off")
        axarr[0, 0].set_title(f"#{subj_idx + 1}, Image")
        f.colorbar(im, ax=axarr[0,0])

        lab = axarr[0, 1].imshow(subj["label"], cmap="gray")
        axarr[0, 1].axis("off")
        axarr[0, 1].set_title(f"#{subj_idx + 1}, Ground Truth")
        f.colorbar(lab, ax=axarr[0,1])

        post = axarr[0, 2].imshow(subj["hard_pred"], cmap="gray")
        axarr[0, 2].axis("off")
        axarr[0, 2].set_title(f"#{subj_idx + 1}, Hard Pred, Dice: {subj['dice_score']:.3f}")
        f.colorbar(post, ax=axarr[0, 2])

        pre = axarr[0, 3].imshow(subj["soft_pred"], cmap="plasma")
        axarr[0, 3].axis("off")
        axarr[0, 3].set_title(f"#{subj_idx + 1}, Probabilities")
        f.colorbar(pre, ax=axarr[0, 3])

        # Show different kinds of statistics about your subjects.
        plot_reliability_diagram(
            bins=ece_bins,
            subj=subj,
            metric="ECE",
            remove_empty_bins=True,
            bin_color="blue",
            show_bin_amounts=show_bin_amounts,
            ax=axarr[1, 0]
        )

        # Show different kinds of statistics about your subjects.
        plot_reliability_diagram(
            bins=ese_bins,
            subj=subj,
            metric="ESE",
            remove_empty_bins=True,
            bin_weighting=bin_weighting,
            bin_color="green",
            show_bin_amounts=show_bin_amounts,
            ax=axarr[1, 1]
        )

        # Look at the pixelwise error.
        ce_im = axarr[1, 2].imshow(
            vis.pixelwise_unc_map(
                subj
            ), cmap="plasma"
        )
        axarr[1, 2].axis("off")
        axarr[1, 2].set_title("Pixel-wise Calibration Error")
        f.colorbar(ce_im, ax=axarr[1, 2])
        
        # Look at the regionwise error.
        ese_im = axarr[1,3].imshow(
            vis.ese_unc_map(
                subj,
                ese_bins,
            ), cmap="plasma"
        )
        axarr[1, 3].axis("off")
        axarr[1, 3].set_title("Region-wise Calibration Error")
        f.colorbar(ese_im, ax=axarr[1, 3])

        plt.show()


def aggregate_plot(
    subject_dict,
    num_bins,
    color="blue",
    bin_weighting="both"
):
    _, axarr = plt.subplots(1, 1, figsize=(15, 10))

    # Calculate the bins and spacing
    bins = np.linspace(0, 1, num_bins+1)[:-1] # Off by one error

    total_ese_info = [ESE(bins=bins,
                    pred=subj["soft_pred"],
                    label=subj["label"]) for subj in subject_dict]
    
    # Get the average score per bin and the amount of pixels that went into those.
    bar_heights = np.mean([ese[0] for ese in total_ese_info], axis=0)
    bin_amounts = np.sum([ese[2] for ese in total_ese_info], axis=0)
    bin_props = bin_amounts / np.sum(bin_amounts)

    if bin_weighting == "proportional":
        w_ese_score = np.average(bar_heights, weights=bin_props)
        title = f"wESE: {w_ese_score:.5f}"

    elif bin_weighting == "uniform":
        uniform_weights = np.ones(len(bar_heights)) / len(bar_heights)
        u_ese_score = np.average(bar_heights, weights=uniform_weights)

        title = f"uESE: {u_ese_score:.5f}"

    elif bin_weighting == "both":
        w_ese_score = np.average(bar_heights, weights=bin_props)

        uniform_weights = np.ones(len(bar_heights)) / len(bar_heights)
        u_ese_score = np.average(bar_heights, weights=uniform_weights)

        title = f"wESE: {w_ese_score:.5f}, uESE: {u_ese_score:.5f}"

    plot_reliability_diagram(
        bins,
        bin_accs=bar_heights,
        ax=axarr,
        title=title,
        bin_color=color
    )
