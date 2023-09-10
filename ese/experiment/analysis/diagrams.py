import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from typing import List

# ese imports
from ese.experiment.analysis.plots import plot_reliability_diagram, plot_confusion_matrix
from ese.experiment.metrics import ECE, ESE, ReCE
import ese.experiment.analysis.vis as vis
from ionpy.util.validation import validate_arguments_init

# Globally used for which metrics to plot for.
metric_dict = {
        "ECE": ECE,
        "ESE": ESE,
        "ReCE": ReCE
    }

@validate_arguments_init
def subject_plot(
    subject_dict: dict, 
    num_bins: int,
    metrics: List[str] = ["ECE", "ESE", "ReCE"],
    show_bin_amounts: bool = False
    ) -> None:
    
    # Calculate the bins and spacing
    bins = torch.linspace(0, 1, num_bins+1)[:-1] # Off by one error

    # if you want to see the subjects and predictions
    plt.rcParams.update({'font.size': 12})  
        
    for subj_idx, subj in enumerate(subject_dict):

        # Setup the plot for each subject.
        f, axarr = plt.subplots(
            nrows=2,
            ncols=4,
            figsize=(24, 12)
        )
        f.patch.set_facecolor('0.8')  

        # Turn the axes off for all plots
        for ax in axarr.ravel():
            ax.axis("off")

        # Define subject name
        subj_name = f"Subject #{subj_idx + 1}"
        # Show the image
        im = axarr[0, 0].imshow(subj["image"], cmap="gray")
        axarr[0, 0].set_title(f"{subj_name}, Image")
        f.colorbar(im, ax=axarr[0,0])

        # Show the groundtruth label
        lab = axarr[0, 1].imshow(subj["label"], cmap="gray")
        axarr[0, 1].set_title(f"{subj_name}, Ground Truth")
        f.colorbar(lab, ax=axarr[0,1])

        # Show the thresholded prediction
        post = axarr[0, 2].imshow(subj["hard_pred"], cmap="gray")
        axarr[0, 2].set_title(f"{subj_name}, Hard Pred, Dice: {subj['dice_score']:.3f}")
        f.colorbar(post, ax=axarr[0, 2])

        # Show the confidence map (which we interpret as probabilities)
        pre = axarr[0, 3].imshow(subj["soft_pred"], cmap="gray")
        axarr[0, 3].set_title(f"{subj_name}, Probabilities")
        f.colorbar(pre, ax=axarr[0, 3])

        # Show different kinds of statistics about your subjects.
        plot_reliability_diagram(
            bins=bins,
            subj=subj,
            metrics=metrics,
            remove_empty_bins=True,
            bin_color="blue",
            show_bin_amounts=show_bin_amounts,
            ax=axarr[1, 0]
        )

        # Show different kinds of statistics about your subjects.
        plot_confusion_matrix(
            subj=subj,
            ax=axarr[1, 1]
        )

        # Look at the pixelwise error.
        ece_map = vis.ECE_map(subj)
        # Get the bounds for visualization
        ece_abs_max = np.max(np.abs(ece_map))
        ece_vmin, ece_vmax = -ece_abs_max, ece_abs_max
        ce_im = axarr[1, 2].imshow(ece_map, cmap="RdBu_r", interpolation="None", vmax=ece_vmax, vmin=ece_vmin)
        axarr[1, 2].set_title("Pixel-wise Calibration Error")
        f.colorbar(ce_im, ax=axarr[1, 2])
        
        # Look at the regionwise error.
        rece_map = vis.ReCE_map(subj, bins)
        # Get the bounds for visualization
        rece_abs_max = np.max(np.abs(rece_map))
        rece_vmin, rece_vmax = -rece_abs_max, rece_abs_max
        ese_im = axarr[1, 3].imshow(rece_map, cmap="RdBu_r", interpolation="None", vmax=rece_vmax, vmin=rece_vmin)
        axarr[1, 3].set_title("Region-wise Calibration Error")
        f.colorbar(ese_im, ax=axarr[1, 3])
        
        # Adjust vertical spacing between the subplots
        plt.subplots_adjust(hspace=0.35)

        plt.show()


@validate_arguments_init
def aggregate_plot(
    subject_dict: dict,
    num_bins: int,
    metrics: List[str],
    color: str = "blue"
) -> None:
    
    # Consturct the subplot (just a single one)
    _, axarr = plt.subplots(nrows=1, ncols=len(metrics), figsize=(5 * len(metrics), 5))

    # Calculate the bins and spacing
    bins = torch.linspace(0, 1, num_bins+1)[:-1] # Off by one error

    for m_idx, metric in enumerate(metrics):
        aggregate_info = [metric_dict[metric](
            conf_bins=bins,
            pred=subj["soft_pred"],
            label=subj["label"]
        ) for subj in subject_dict]
        
        # Get the average score per bin and the amount of pixels that went into those.
        aggregated_scores = torch.stack([subj[0] for subj in aggregate_info])
        aggregated_accs = torch.stack([subj[1] for subj in aggregate_info])
        aggregated_amounts = torch.stack([subj[2] for subj in aggregate_info])

        # Average over the subjects
        bin_scores = torch.mean(aggregated_scores, dim=0)
        bin_accs = torch.mean(aggregated_accs, dim=0)
        bin_amounts = torch.sum(aggregated_amounts, dim=0)

        bin_info = [bin_scores, bin_accs, bin_amounts]
        plot_reliability_diagram(
            bins,
            bin_info=bin_info,
            metrics=[metric],
            ax=axarr[m_idx],
            bin_color=color
        )


@validate_arguments_init
def aggregate_confusion_matrix(
    subj_dict
):
    # Initialize an empty aggregate confusion matrix
    aggregate_cm = np.zeros((2, 2), dtype=int)  # Assuming binary segmentation

    # Define class labels
    class_labels = ['Background', 'Foreground']

    # Loop through each subject and calculate the confusion matrix
    for subj in subj_dict:
        ground_truth_np = subj['label'].cpu().numpy().flatten()
        predictions_np = subj['hard_pred'].cpu().numpy().flatten()
        cm = confusion_matrix(ground_truth_np, predictions_np, labels=[0, 1])
        aggregate_cm += cm

    # Plot the aggregate confusion matrix on the predefined axes using seaborn
    plt.figure(figsize=(12, 9))

    sns.heatmap(aggregate_cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Aggregate Confusion Matrix')

    # Display the plot
    plt.show()







