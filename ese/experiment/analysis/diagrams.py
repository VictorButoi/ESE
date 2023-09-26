import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List
from pydantic import validate_arguments
from sklearn.metrics import confusion_matrix

# ese imports
from ese.experiment.analysis.plots import analysis_plots, error_maps, reliability_plots, simple_vis
from ese.experiment.metrics import ECE, ReCE
import ese.experiment.metrics.utils as metric_utils


# Globally used for which metrics to plot for.
metric_dict = {
    "ECE": ECE,
    "ReCE": ReCE
}

metric_color_dict = {
    "ECE": "blue",
    "ACE": "goldenrod",
    "ReCE": "green" 
}

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def subject_diagram(
    subject_list: List[dict], 
    num_bins: int,
    reliability_y_axis: str = "Frequency",
    metrics: List[str] = ["ECE", "ReCE"],
    bin_weighting: str = "proportional",
    show_bin_amounts: bool = False,
    remove_empty_bins: bool = True,
    include_background: bool = True 
    ) -> None:
    assert not (reliability_y_axis == "Accuracy" and include_background), "Cannot include background when using accuracy as y-axis."
    
    # if you want to see the subjects and predictions
    plt.rcParams.update({'font.size': 12})  
        
    # Dimensions of the plots
    width_per_plot = 8
    height_per_plot = 7
    num_rows = 3 
    num_cols = 4 

    # Go through each subject and plot a bunch of info about it.
    for subj_idx, subj in enumerate(subject_list):

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
        simple_vis.plot_subject_image(
            subj_name=subj_name,
            subj=subj,
            fig=f,
            ax=axarr[0, 0]
        )
        # Show the groundtruth label
        simple_vis.plot_subj_label(
            subj=subj,
            fig=f,
            ax=axarr[0, 1]
        )
        # Show the confidence map (which we interpret as probabilities)
        simple_vis.plot_prediction_probs(
            subj=subj,
            fig=f,
            ax=axarr[0, 2]
        )
        # Show the pixelwise thresholded prediction
        simple_vis.plot_pred(
            subj=subj,
            fig=f,
            ax=axarr[0, 3]
        )

        #########################################################
        # ROW TwO 
        #########################################################

        # Show different kinds of statistics about your subjects.
        analysis_plots.plot_confusion_matrix(
            subj=subj,
            ax=axarr[1, 0]
        )
        for m_idx, metric in enumerate(metrics):
            # Plot reliability diagram with precision on y.
            reliability_plots.plot_subj_reliability_diagram(
                num_bins=num_bins,
                y_axis=reliability_y_axis,
                subj=subj,
                metric=metric,
                bin_weighting=bin_weighting,
                remove_empty_bins=remove_empty_bins,
                include_background=include_background,
                show_bin_amounts=show_bin_amounts,
                bar_color=metric_color_dict[metric],
                ax=axarr[1, 1 + m_idx]
            )
        #########################################################
        # ROW THREE
        #########################################################
        # Show the variance of the confidences for pixel samples in the bin.
        analysis_plots.plot_avg_samplesize_vs_numbins(
            subj=subj,
            metrics=metrics,
            metric_colors=metric_color_dict,
            bin_weighting=bin_weighting,
            ax=axarr[2, 0] 
        )
        # Show the variance of the confidences for pixel samples in the bin.
        analysis_plots.plot_avg_variance_vs_numbins(
            subj=subj,
            metrics=metrics,
            metric_colors=metric_color_dict,
            bin_weighting=bin_weighting,
            ax=axarr[2, 1] 
        )
        # Show different kinds of statistics about your subjects.
        analysis_plots.plot_error_vs_numbins(
            subj=subj,
            metrics=metrics,
            metric_colors=metric_color_dict,
            bin_weighting=bin_weighting,
            ax=axarr[2, 2]
        )
        # Show a more per-pixel calibration error for each region.
        error_maps.plot_rece_map(
            subj=subj,
            num_bins=num_bins,
            fig=f,
            ax=axarr[2, 3],
            average=False
        )
        
        # Adjust vertical spacing between the subplots
        plt.subplots_adjust(hspace=0.3)

        # Display for the subject.
        plt.show()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def aggregate_reliability_diagram(
    subject_list: List[dict],
    num_bins: int,
    metrics: List[str],
    reliability_y_axis: str = "Frequency",
    bin_weighting: str = "proportional",
    remove_empty_bins: bool = True,
    include_background: bool = True,
    threshold: float = 0.5,
) -> None:
    
    # Consturct the subplot (just a single one)
    _, axarr = plt.subplots(nrows=1, ncols=len(metrics), figsize=(7 * len(metrics), 7))

    for m_idx, metric in enumerate(metrics):

        conf_per_bin = {bin_idx: None for bin_idx in range(num_bins)}
        freq_per_bin = {bin_idx: None for bin_idx in range(num_bins)}
        amount_per_bin = {bin_idx: None for bin_idx in range(num_bins)}
        
        # Go through each subject, and get the information we need.
        for subj in subject_list:
            subj_calibration_info = metric_dict[metric](
                num_bins=num_bins,
                pred=subj["soft_pred"],
                label=subj["label"],
                measure=reliability_y_axis,
                weighting=bin_weighting,   
                threshold=threshold,
                include_background=include_background
            )
            # Information about the subject.
            subj_confs_per_bin = subj_calibration_info["confs_per_bin"]
            subj_freqs_per_bin = subj_calibration_info["freqs_per_bin"]
            subj_amounts_per_bin = subj_calibration_info["bin_amounts"]

            # Add the information to the aggregate.
            for bin_idx in range(num_bins):
                if conf_per_bin[bin_idx] is None:
                    conf_per_bin[bin_idx] = subj_confs_per_bin[bin_idx]
                    freq_per_bin[bin_idx] = subj_freqs_per_bin[bin_idx]
                    amount_per_bin[bin_idx] = subj_amounts_per_bin[bin_idx]
                else:
                    conf_per_bin[bin_idx] = torch.cat((conf_per_bin[bin_idx], subj_confs_per_bin[bin_idx]))
                    freq_per_bin[bin_idx] = torch.cat((freq_per_bin[bin_idx], subj_freqs_per_bin[bin_idx]))
                    amount_per_bin[bin_idx] += subj_amounts_per_bin[bin_idx]
        
        # Build the calibration info for each bin
        calibration_info = {
            "bin_amounts": torch.zeros(num_bins),
            "avg_conf_per_bin": torch.zeros(num_bins),
            "avg_freq_per_bin": torch.zeros(num_bins),
            "score_per_bin": torch.zeros(num_bins)
        }
        for b_idx in range(num_bins):
            # Reduce over the samples for confidences and frequencies.
            bin_amounts = amount_per_bin[b_idx]
            avg_frequency = freq_per_bin[b_idx].mean()
            avg_confidence = conf_per_bin[b_idx].mean()
            score_per_bin = (avg_confidence - avg_frequency).abs()
            # Store in our calibration info.
            calibration_info["bin_amounts"][b_idx] = bin_amounts
            calibration_info["avg_conf_per_bin"][b_idx] = avg_confidence
            calibration_info["avg_freq_per_bin"][b_idx] = avg_frequency
            calibration_info["score_per_bin"][b_idx] = score_per_bin
            # Get the calibration score
            calibration_info["score"] = metric_utils.reduce_scores(
                score_per_bin=score_per_bin,
                amounts_per_bin=bin_amounts, 
                weighting=bin_weighting
            )

        # Show the reliability plots.
        reliability_plots.plot_cumulative_reliability_diagram(
            num_bins=num_bins,
            calibration_info=calibration_info,
            metric=metric,
            bin_weighting=bin_weighting,
            y_axis=reliability_y_axis,
            remove_empty_bins=remove_empty_bins,
            include_background=include_background,
            ax=axarr[m_idx],
            bar_color=metric_color_dict[metric]
        )
    
    # Show the plot per metric.
    plt.show()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def score_histogram_diagram(
    subject_list: List[dict],
    num_bins: int,
    metrics: List[str],
    reliability_y_axis: str = "Frequency",
    bin_weighting: str = "proportional",
    include_background: bool = True,
    threshold: float = 0.5,
) -> None:
    
    # Go through each subject, and get the information we need.
    score_list = [] 
    for metric in metrics:
        # Go through each subject, and get the information we need.
        for subj in subject_list:
            subj_calibration_info = metric_dict[metric](
                num_bins=num_bins,
                pred=subj["soft_pred"],
                label=subj["label"],
                measure=reliability_y_axis,
                weighting=bin_weighting,   
                threshold=threshold,
                include_background=include_background
            )
            score_list.append({
                "error": subj_calibration_info["score"],
                "metric": metric
                })

    # Melting the DataFrame to long-form for compatibility with `hue`
    score_df = pd.DataFrame(score_list)

    # Set the size of the figure
    plt.figure(figsize=(12, 8))

    # Show the reliability plots.
    sns.kdeplot(
        data=score_df,
        x='error',
        hue='metric',
        common_norm=False,  # To plot each distribution independently
        palette=metric_color_dict,
        fill=True  # To fill under the density curve
    )

    # Calculating the mean of 'error' for each unique 'metric' and adding vertical lines
    for metric in score_df['metric'].unique():
        mean = score_df['error'][score_df['metric'] == metric].mean()
        plt.axvline(mean, color=metric_color_dict[metric], linestyle='--')
        plt.text(mean + 0.01, 1, f' {metric} mean', color=metric_color_dict[metric], rotation=90)

    plt.title(f"{metric} Histogram Over Subjects")
    plt.show()
