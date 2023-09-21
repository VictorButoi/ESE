import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from typing import List

# ese imports
from ese.experiment.analysis.plots import *
from ese.experiment.metrics import ECE, ReCE

# ionpy imports
from ionpy.util.validation import validate_arguments_init


# Globally used for which metrics to plot for.
metric_dict = {
        "ECE": ECE,
        "ReCE": ReCE
    }

@validate_arguments_init
def subject_plot(
    subject_dict: dict, 
    num_bins: int,
    metrics: List[str] = ["ECE", "ReCE"],
    bin_weightings: List[str] = ["uniform", "weighted"],
    show_bin_amounts: bool = False
    ) -> None:
    
    # if you want to see the subjects and predictions
    plt.rcParams.update({'font.size': 12})  
        
    # Dimensions of the plots
    width_per_plot = 7
    height_per_plot = 7
    num_rows = 3 
    num_cols = 4 

    # Go through each subject and plot a bunch of info about it.
    for subj_idx, subj in enumerate(subject_dict):

        # Setup the plot for each subject.
        f, axarr = plt.subplots(
            nrows=num_rows,
            ncols=num_cols,
            figsize=(width_per_plot * num_cols, height_per_plot * num_rows)
        )
        # 6 * 5, 6 * 2
        f.patch.set_facecolor('0.8')  

        # Turn the axes off for all plots
        for ax in axarr.ravel():
            ax.axis("off")

        #########################################################
        # ROW THREE
        #########################################################

        # Define subject name and plot the image
        subj_name = f"Subject #{subj_idx + 1}"
        plot_subject_image(
            subj_name=subj_name,
            subj=subj,
            fig=f,
            ax=axarr[0, 0]
        )

        # Show the groundtruth label
        plot_subj_label(
            subj=subj,
            fig=f,
            ax=axarr[0, 1]
        )

        # Show the confidence map (which we interpret as probabilities)
        plot_prediction_probs(
            subj=subj,
            fig=f,
            ax=axarr[0, 2]
        )

        # Show the pixelwise thresholded prediction
        plot_pred(
            subj=subj,
            fig=f,
            ax=axarr[0, 3]
        )

        #########################################################
        # ROW TwO 
        #########################################################

        # Show different kinds of statistics about your subjects.
        plot_confusion_matrix(
            subj=subj,
            ax=axarr[1, 0]
        )

        # Show different kinds of statistics about your subjects.
        plot_reliability_diagram(
            num_bins=num_bins,
            subj=subj,
            metrics=metrics,
            bin_weightings=bin_weightings,
            remove_empty_bins=True,
            bin_color="blue",
            show_bin_amounts=show_bin_amounts,
            ax=axarr[1, 1]
        )

        # Show different kinds of statistics about your subjects.
        plot_error_vs_numbins(
            subj=subj,
            metrics=metrics,
            bin_weightings=bin_weightings,
            ax=axarr[1, 2]
        )

        #########################################################
        # ROW THREE
        #########################################################

        # Display the pixelwise calibration error.
        plot_ece_map(
            subj=subj,
            fig=f,
            ax=axarr[2, 0]
        ) 

        # Display the regionwise calibration error averaging over the region.
        plot_rece_map(
            subj=subj,
            num_bins=num_bins,
            fig=f,
            ax=axarr[2, 1],
            average=True
        )

        # Show a more per-pixel calibration error for each region.
        plot_rece_map(
            subj=subj,
            num_bins=num_bins,
            fig=f,
            ax=axarr[2, 2],
            average=False
        )
        
        # Adjust vertical spacing between the subplots
        plt.subplots_adjust(hspace=0.3)

        # Display for the subject.
        plt.show()


@validate_arguments_init
def aggregate_reliability_plot(
    subject_dict: dict,
    num_bins: int,
    metrics: List[str],
    bin_weightings: List[str] = ["uniform", "weighted"],
    color: str = "blue"
) -> None:
    
    # Consturct the subplot (just a single one)
    _, axarr = plt.subplots(nrows=1, ncols=len(metrics), figsize=(7 * len(metrics), 7))

    # y-axis labels
    y_axes = {
        "ECE": "Avg. Accuracy",
        "ReCE": "Avg. Island-wise Precision"
    }

    for m_idx, metric in enumerate(metrics):

        aggregate_info = [
            metric_dict[metric](
            num_bins=num_bins,
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
            num_bins=num_bins,
            bin_info=bin_info,
            metrics=[metric],
            bin_weightings=bin_weightings,
            y_axis=y_axes[metric],
            ax=axarr[m_idx],
            bin_color=color
        )


@validate_arguments_init
def aggregate_cm_plot(
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


@validate_arguments_init
def aggregate_error_distribution_plot(
    subject_dict: dict,
    num_bins: int,
    metrics: List[str],
    bin_weightings: List[str] = ["uniform", "weighted"],
) -> None:

    error_list = []
    for metric in metrics:
        for bin_weighting in bin_weightings:
            for subj in subject_dict:

                # Compute the metrics.
                bin_scores, _, bin_amounts = metric_dict[metric](
                    num_bins=num_bins,
                    pred=subj['soft_pred'],
                    label=subj['label']
                )

                # Calculate the error.
                error = reduce_scores(
                    scores=bin_scores.numpy(),
                    bin_amounts=bin_amounts.numpy(),
                    weighting=bin_weighting
                )

                # Add the error to the dataframe.
                error_list.append({
                    "Metric": metric,
                    "Weighting": bin_weighting,
                    "Calibration Error": error
                })
    
    # Wrap into a pandas dataframe
    error_df = pd.DataFrame(error_list)

    # Create a box plot of the calibration errors, and set its height
    plt.figure(figsize=(6, 10))

    sns.set_theme(style="whitegrid")
    sns.boxplot(
        x="Metric",
        y="Calibration Error",
        data=error_df,
    )



            

