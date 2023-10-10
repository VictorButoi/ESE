import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Literal
from pydantic import validate_arguments

# ese imports
from ese.experiment.analysis.plots import analysis_plots, error_maps, reliability_plots, simple_vis
import ese.experiment.metrics.utils as metric_utils
from ionpy.experiment.util import absolute_import


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def subject_diagram(
    subject_list: List[dict], 
    num_bins: int,
    metric_cfg: dict,
    class_type: Literal["Binary", "Multi-class"],
    num_subjects: int = 10,
    bin_weighting: str = "proportional",
    include_background: bool = True,
    show_bin_amounts: bool = False,
    show_analysis_plots: bool = True,
    remove_empty_bins: bool = True,
    ) -> None:
    if class_type == "Multi-class": 
        assert include_background, "Background must be included for multi-class."
    
    # if you want to see the subjects and predictions
    plt.rcParams.update({'font.size': 12})  
        
    # Dimensions of the plots
    width_per_plot = 8
    height_per_plot = 7
    num_rows = 3 
    num_cols = 4 

    # Shorten the total subject list (to save time).
    subject_list = subject_list[:num_subjects]

    # Import the caibration metrics.
    metric_cfg = metric_cfg.copy()
    for cal_metric in metric_cfg.keys():
        metric_cfg[cal_metric]['func'] = absolute_import(metric_cfg[cal_metric]['func'])

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
        simple_vis.plot_conf_map(
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
        # if show_analysis_plots:
        #     analysis_plots.plot_error_vs_numbins(
        #         subj=subj,
        #         metric_cfg=metric_cfg,
        #         bin_weighting=bin_weighting,
        #         ax=axarr[1, 0]
        #     )
        # Plot the different reliability diagrams for our different
        # metrics.
        for m_idx, cal_metric in enumerate(metric_cfg):
            # Plot reliability diagram with precision on y.
            reliability_plots.plot_subj_reliability_diagram(
                num_bins=num_bins,
                subj=subj,
                metric_name=cal_metric,
                metric_dict=metric_cfg[cal_metric],
                class_type=class_type,
                bin_weighting=bin_weighting,
                remove_empty_bins=remove_empty_bins,
                include_background=include_background,
                show_bin_amounts=show_bin_amounts,
                ax=axarr[1, m_idx]
            )

        #########################################################
        # ROW THREE
        #########################################################
        # Show some statistics about what happens when number of bins change.
        if show_analysis_plots:
            # Show the variance of the confidences for pixel samples in the bin.
            analysis_plots.plot_avg_samplesize_vs_numbins(
                subj=subj,
                metric_cfg=metric_cfg,
                bin_weighting=bin_weighting,
                ax=axarr[2, 0] 
            )
            # Show the variance of the confidences for pixel samples in the bin.
            analysis_plots.plot_avg_variance_vs_numbins(
                subj=subj,
                metric_cfg=metric_cfg,
                bin_weighting=bin_weighting,
                ax=axarr[2, 1] 
            )
        # Show a more per-pixel calibration error for each region.
        error_maps.plot_ece_map(
            subj=subj,
            include_background=include_background,
            class_type=class_type,
            fig=f,
            ax=axarr[2, 2],
        )
        # Show a more per-pixel calibration error for each region.
        error_maps.plot_rece_map(
            subj=subj,
            num_bins=num_bins,
            include_background=include_background,
            class_type=class_type,
            fig=f,
            ax=axarr[2, 3],
        )
        
        # Adjust vertical spacing between the subplots
        plt.subplots_adjust(hspace=0.3)

        # Display for the subject.
        plt.show()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def aggregate_reliability_diagram(
    subject_list: List[dict],
    num_bins: int,
    metric_cfg: dict,
    class_type: Literal["Binary", "Multi-class"],
    bin_weighting: str = "proportional",
    remove_empty_bins: bool = True,
    include_background: bool = True,
) -> None:
    if class_type == "Multi-class": 
        assert include_background, "Background must be included for multi-class."

    # Consturct the subplot (just a single one)
    _, axarr = plt.subplots(nrows=1, ncols=len(metric_cfg), figsize=(7 * len(metric_cfg), 7))

    # Import the caibration metrics.
    metric_cfg = metric_cfg.copy()
    for cal_metric in metric_cfg.keys():
        metric_cfg[cal_metric]['func'] = absolute_import(metric_cfg[cal_metric]['func'])

    for m_idx, cal_metric in enumerate(metric_cfg):

        conf_per_bin = {bin_idx: None for bin_idx in range(num_bins)}
        metric_per_bin = {bin_idx: None for bin_idx in range(num_bins)}
        amount_per_bin = {bin_idx: None for bin_idx in range(num_bins)}
        
        # Go through each subject, and get the information we need.
        for subj in subject_list:
            subj_calibration_info = metric_cfg[cal_metric]['func'](
                num_bins=num_bins,
                conf_map=subj["conf_map"],
                pred_map=subj["pred_map"],
                label=subj["label"],
                class_type=class_type,
                weighting=bin_weighting,
                include_background=include_background
            )
            # Information about the subject.
            subj_confs_per_bin = subj_calibration_info["confs_per_bin"]
            subj_freqs_per_bin = subj_calibration_info["freqs_per_bin"]
            subj_amounts_per_bin = subj_calibration_info["bin_amounts"]

            # Add the information to the aggregate.
            for bin_idx in range(num_bins):
                if bin_idx in subj_confs_per_bin.keys():
                    if conf_per_bin[bin_idx] is None:
                        conf_per_bin[bin_idx] = subj_confs_per_bin[bin_idx]
                        metric_per_bin[bin_idx] = subj_freqs_per_bin[bin_idx]
                        amount_per_bin[bin_idx] = subj_amounts_per_bin[bin_idx]
                    else:
                        conf_per_bin[bin_idx] = torch.cat([conf_per_bin[bin_idx], subj_confs_per_bin[bin_idx]])
                        metric_per_bin[bin_idx] = torch.cat([metric_per_bin[bin_idx], subj_freqs_per_bin[bin_idx]])
                        amount_per_bin[bin_idx] += subj_amounts_per_bin[bin_idx]
        
        # Build the calibration info for each bin
        calibration_info = {
            "bin_amounts": torch.zeros(num_bins),
        }
        if class_type == "Multi-class":
            calibration_info["bin_accs"] = torch.zeros(num_bins)
        else:
            calibration_info["bin_freqs"] = torch.zeros(num_bins)

        for b_idx in range(num_bins):
            # Reduce over the samples for confidences and frequencies.
            bin_amounts = amount_per_bin[b_idx]
            calibration_info["bin_amounts"][b_idx] = bin_amounts

            # Store based on if binary of multiclass
            avg_metric = metric_per_bin[b_idx].mean()
            if class_type == "Multi-class":
                calibration_info["bin_accs"][b_idx] = avg_metric 
            else:
                calibration_info["bin_freqs"][b_idx] = avg_metric

            # Get the average confidence
            avg_confidence = conf_per_bin[b_idx].mean()

            # Get the scores.
            score_per_bin = (avg_confidence - avg_metric).abs()

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
            metric_name=cal_metric,
            metric_dict=metric_cfg[cal_metric],
            class_type=class_type,
            bin_weighting=bin_weighting,
            remove_empty_bins=remove_empty_bins,
            include_background=include_background,
            ax=axarr[m_idx],
        )
    
    # Show the plot per metric.
    plt.show()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def score_histogram_diagram(
    subject_list: List[dict],
    num_bins: int,
    metric_cfg: dict,
    class_type: Literal["Binary", "Multi-class"],
    bin_weighting: str = "proportional",
    include_background: bool = True
) -> None:
    if class_type == "Multi-class": 
        assert include_background, "Background must be included for multi-class."
    
    # Import the caibration metrics.
    metric_cfg = metric_cfg.copy()
    for cal_metric in metric_cfg.keys():
        metric_cfg[cal_metric]['func'] = absolute_import(metric_cfg[cal_metric]['func'])
        
    # Go through each subject, and get the information we need.
    score_list = [] 
    for cal_metric in metric_cfg:
        # Go through each subject, and get the information we need.
        for subj in subject_list:
            subj_calibration_info = metric_cfg[cal_metric]['func'](
                num_bins=num_bins,
                conf_map=subj["conf_map"],
                pred_map=subj["pred_map"],
                label=subj["label"],
                class_type=class_type,
                weighting=bin_weighting,
                include_background=include_background
            )
            score_list.append({
                "error": subj_calibration_info["score"],
                "metric": cal_metric
                })

    # Melting the DataFrame to long-form for compatibility with `hue`
    score_df = pd.DataFrame(score_list)

    # Set the size of the figure
    plt.figure(figsize=(12, 8))

    metric_color_dict = {metric: metric_cfg[metric]['color'] for metric in metric_cfg.keys()}

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
    for cal_metric in score_df['metric'].unique():
        mean = score_df['error'][score_df['metric'] == cal_metric].mean()
        plt.axvline(mean, color=metric_cfg[cal_metric]['color'], linestyle='--')
        plt.text(mean + 0.01, 1, f' {cal_metric} mean = {np.round(mean, 4)}', color=metric_cfg[cal_metric]['color'], rotation=90)

    plt.title(f"{cal_metric} Histogram Over Subjects")
    plt.show()
