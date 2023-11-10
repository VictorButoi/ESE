# local imports
from .utils import reorder_splits

# ionpy imports
from ionpy.util import StatsMeter

# misc. imports
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from pydantic import validate_arguments
from typing import List, Optional, Union, Literal, Any


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def viz_region_size_distribution(
    data_points: List[dict],
    hue: Optional[str] = None,
    col: Optional[str] = None,
    row: Optional[str] = None,
    row_split: Optional[str] = None,
    ) -> None:
    # Combine all the subject label dictionaries of region sizes 
    # in a single dictionary.    
    region_size_records = [] 
    for subject in data_points:
        for labmap_type in ["pred", "gt"]:
            lab_sizes_dict_key = "gt_lab_region_sizes" if (labmap_type=="gt") else "pred_lab_region_sizes"
            lab_size_dict = subject[lab_sizes_dict_key]
            for label in lab_size_dict.keys():
                for lab_reg_size in lab_size_dict[label]:
                    record = {
                        "subject_id": subject['subject_id'],
                        "label": label,
                        "label_map_type": labmap_type,
                        "region_size": lab_reg_size
                    } 
                    region_size_records.append(record)
    # Combine all the subject label dictionaries of region sizes
    regions_df = pd.DataFrame(region_size_records)
    # Define what a positive KDE looks like.
    def positive_kde(*args, **kwargs):
        sns.kdeplot(*args, clip=(0, None), **kwargs)
    # Creating a FacetGrid with KDE plots
    if row_split is None:
        g = sns.FacetGrid(
            regions_df, 
            hue=hue, 
            col=col, 
            row=row, 
            height=4, 
            aspect=1.5)
        g = g.map(positive_kde, "region_size", fill=True)
        g.add_legend()
    else:
        for value in regions_df[row_split].unique():
            print("For {row_split} = {value}".format(row_split=row_split, value=value))
            g = sns.FacetGrid(regions_df[regions_df[row_split]==value], 
                              hue=hue, 
                              col=col, 
                              row=row, 
                              height=4, 
                              aspect=1.5)
            g = g.map(positive_kde, "region_size", common_norm=False, fill=True)
            g.add_legend()
            plt.title(f"Label {value}")
            plt.show()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def viz_accuracy_vs_confidence(
    pixel_preds: Any,
    title: str,
    x: str,
    kind: Literal["bar", "line"],
    add_proportion: bool = True,
    add_weighted_metrics: bool = True,
    x_labels: bool = True,
    col: Optional[Literal["bin_num"]] = None,
    facet_kws: Optional[dict] = None,
):
    # Organize data into a structure for plotting
    # Structure: data_dict[bin_num][pred_label][measure] = list of values
    data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (pred_label, num_neighbors, bin_num, measure), value in pixel_preds.items():
        if x == "pred_label":
            data_dict[bin_num][pred_label][measure].append(value)
        elif x == "num_neighbors":
            data_dict[bin_num][num_neighbors][measure].append(value)
        elif x == "pred_label,num_neighbors":
            data_dict[bin_num][f"{pred_label},{num_neighbors}"][measure].append(value)
        else:
            raise NotImplementedError(f"Haven't configured {x} for bar plot.")

    # Calculate the average for each pred_label and measure within each bin_num
    avg_data_dict = defaultdict(lambda: defaultdict(dict))

    # These are the metrics we care about for each x group.
    measure_names = ["confidence", "accuracy"]
    if add_weighted_metrics:
        measure_names.append("weighted accuracy")
    
    # Helpful trackers for counting samples in each bin.
    bin_samples = {
        "uniform": {},
        "weighted": {}
    }
    # Go through each bin_num and each x variable. 
    for bin_num, x_var_set in data_dict.items():
        # SETUP A METER FOR EACH X GROUP, THIS WILL TRACK ONLY THE MEAN, STD, AND N
        # FOR EACH GROUP IN THIS BIN.
        avg_bin_met_meters = {measure_group: StatsMeter() for measure_group in measure_names}
        for x_var, measures in x_var_set.items():
            for measure, values in measures.items():
                bin_x_group_stats = StatsMeter()
                # Accumulate all of the values in this group, and at to our total bin trackers.
                for value in values:
                    bin_x_group_stats += value
                    avg_bin_met_meters[measure] += value
                # Record the mean, std, and n for the bin.
                avg_data_dict[int(bin_num)][str(x_var)][measure] = {
                    "mean": bin_x_group_stats.mean, 
                    "std": bin_x_group_stats.std, 
                    "n_samples": bin_x_group_stats.n
                } 
        #  GET THE AVERAGE FOR EACH CONFIDENCE BIN.
        for mes in measure_names:
            avg_data_dict[int(bin_num)]["avg"][mes] = {
                "mean": avg_bin_met_meters[mes].mean, 
                "std": avg_bin_met_meters[mes].std, 
                "n_samples": avg_bin_met_meters[mes].n
                }
            # Add the number of samples, both unwieghted and weighted. NOTE: We only do this for 
            # confidence because otherwise we would be double counting the samples.
            if "accuracy" in mes:
                if "weighted" in mes:
                    bin_samples_key = "weighted"
                else:  
                    bin_samples_key = "uniform"
                # Add the number of samples in this bin.
                bin_samples[bin_samples_key][bin_num] = avg_bin_met_meters[mes].n

    # NORMALIZE THE PROPORTION BY THE TOTAL NUMBER OF SAMPLES IN THE EXPERIMENT.
    total_uniform = sum(bin_samples["uniform"].values())
    total_weighted = int(sum(bin_samples["weighted"].values()))
    assert total_uniform == total_weighted,\
        f"Total uniform samples: {total_uniform} and Total weighted samples: {total_weighted} should be the same."
    # Loop through every bin and x_var
    for bin_num, x_var_set in avg_data_dict.items():
        for x_var in x_var_set.keys():
            # Normalize by number of samples in the bin.
            bin_x_samples = avg_data_dict[bin_num][x_var]["accuracy"]["n_samples"]
            avg_data_dict[bin_num][x_var]["proportion"] = bin_x_samples / bin_samples["uniform"][bin_num]
            # Normalize by the total weighted samples of the bin.
            weighted_x_samples = avg_data_dict[bin_num][x_var]["weighted accuracy"]["n_samples"]
            avg_data_dict[bin_num][x_var]["weighted proportion"] = weighted_x_samples / bin_samples["weighted"][bin_num]
        # Finally, normalize by the total number of samples.
        avg_data_dict[bin_num]["avg"]["proportion"] = bin_samples["uniform"][bin_num] / total_uniform 
        avg_data_dict[bin_num]["avg"]["weighted proportion"] = bin_samples["weighted"][bin_num] / total_weighted 

    # Add the proportion to the measures list.
    if add_proportion:
        measure_names += ["proportion", "weighted proportion"]

    # Set a bunch of information for plotting the graphs.
    num_bins = len(avg_data_dict)
    num_rows, num_cols = math.ceil(num_bins/5), 5
    sharex, sharey = True, True
    if facet_kws is not None:
        sharex, sharey = facet_kws["sharex"], facet_kws["sharey"]
    # Setup the subplot array.
    fig, axes = plt.subplots(
        nrows=num_rows, 
        ncols=num_cols, 
        figsize=(num_cols*6, num_rows*6), 
        sharex=sharex, 
        sharey=sharey
        )
    # Define the colors for the plot.
    metric_colors = {
        "confidence": "blue",
        "avg confidence": "dodgerblue",
        "accuracy": "darkorange",
        "weighted accuracy": "orange",
        "avg accuracy": "sandybrown",
        "avg weighted accuracy": "peachpuff",
        "proportion": "darkgreen",
        "weighted proportion": "seagreen",
        "avg proportion": "darkseagreen",
        "avg weighted proportion": "lightgreen",
    }

    # Loop through the axes, and if there isn't a bin number for that axis, remove it first sorting the avg_data_dict indices by col.
    data_sorted_by_bin_num = sorted(avg_data_dict.items(), key=lambda x: x[0])
    bin_nums = [bin_num for bin_num, _ in data_sorted_by_bin_num] 
    for ax_idx, ax in enumerate(axes.flat):
        if ax_idx not in bin_nums:
            fig.delaxes(ax)

    # Loop through the subplots and plot the data.
    for bin_num, x_var_set in data_sorted_by_bin_num:
        ax = axes[bin_num // 5, bin_num % 5]
        sorted_x_vars = sorted(list(x_var_set.keys()))
        # Bar plots so we can see the difference in heights.
        if kind == "bar":
            inds = np.arange(len(sorted_x_vars))  # the x locations for the groups
            width = 0.8 / len(measure_names)  # the widths of these x groups
            # Loop through  measure_names and plot them.
            for x_idx, x_var in enumerate(sorted_x_vars):
                for m_idx, measure in enumerate(measure_names):
                    # Determine the color and label based on whether the x_var is 'avg' or not
                    if x_var == 'avg':
                        bar_color = metric_colors[f'avg {measure}']
                        bar_label = f'avg {measure}'
                    else:
                        bar_color = metric_colors[measure]
                        bar_label = measure if (x_idx == 0) else ""  # Label only once for other measures
                    # If plotting the porportions, plot the proportion, otherwise plot the mean and std.        
                    if  "proportion" in measure:
                        ax.bar(
                            inds[x_idx] + m_idx * width,
                            x_var_set[x_var][measure],
                            width,
                            label=bar_label,
                            color=bar_color,
                        )
                    else:
                        ax.bar(
                            inds[x_idx] + m_idx * width,
                            x_var_set[x_var][measure]["mean"],
                            width,
                            label=bar_label,
                            capsize=3,  # Customize the cap size of error bars
                            color=bar_color,
                        )
        else:
            raise NotImplementedError("Haven't configured that kind of plot yet.") 
        # Add some text for labels, title and axes ticks
        ax.set_title(f'{col} = {bin_num}')
        ax.set_ylabel('value')
        # Optionally remove the x labels
        if x_labels:
            ax.set_xlabel(x)
            if kind == "bar":
                ax.set_xticks(inds + width * (len(measures) - 1) / 2)
                ax.set_xticklabels(sorted_x_vars)
        else:
            ax.set_xticklabels([])
    # Adjusting the titles
    plt.legend(
        title='measure',
        bbox_to_anchor=(1.6, 1),  # Adjust the position of the legend
        loc='center right',  # Specify the legend location
        frameon=False,  # Remove the legend background
    )
    plt.subplots_adjust(top=0.9)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def viz_cal_metric_corr(
    pixel_preds: pd.DataFrame,
    title: str,
    col: Optional[str] = None,
):
    # Group the metrics by important factors
    if col is None:
        grouped_preds = pixel_preds.groupby(['cal_metric'])
    else:
        grouped_preds = pixel_preds.groupby([col, 'cal_metric'])
    
    # Accuracy correlations 
    acc_correlations = grouped_preds.apply(lambda x: x['accuracy'].corr(x['cal_score'])).reset_index(name='correlation')
    acc_correlations['eval_metric'] = 'accuracy'

    # Dice correlations
    dice_correlations = grouped_preds.apply(lambda x: x['dice'].corr(x['cal_score'])).reset_index(name='correlation')
    dice_correlations['eval_metric'] = 'dice'

    if col == 'split':
        acc_correlations = reorder_splits(acc_correlations)
        dice_correlations = reorder_splits(dice_correlations)

    # Combine the two
    subject_correlations = pd.concat([acc_correlations, dice_correlations])

    # Plot the correlations
    g = sns.catplot(data=subject_correlations, 
                    x="eval_metric", 
                    y="correlation", 
                    hue='cal_metric', 
                    kind="bar", 
                    col=col,
                    height=8, 
                    aspect=1)

    # Set the y lim between - 1 and 1
    g.set(ylim=(-1, 1))
    # Adjusting the titles
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(title, fontsize=16)
    plt.show()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def viz_cal_metric_corr(
    pixel_preds: pd.DataFrame,
    title: str,
    col: Optional[str] = None,
):
    # Group the metrics by important factors
    if col is None:
        grouped_preds = pixel_preds.groupby(['cal_metric'])
    else:
        grouped_preds = pixel_preds.groupby([col, 'cal_metric'])
    
    # Accuracy correlations 
    acc_correlations = grouped_preds.apply(lambda x: x['accuracy'].corr(x['cal_score'])).reset_index(name='correlation')
    acc_correlations['eval_metric'] = 'accuracy'

    # Dice correlations
    dice_correlations = grouped_preds.apply(lambda x: x['dice'].corr(x['cal_score'])).reset_index(name='correlation')
    dice_correlations['eval_metric'] = 'dice'

    if col == 'split':
        acc_correlations = reorder_splits(acc_correlations)
        dice_correlations = reorder_splits(dice_correlations)

    # Combine the two
    subject_correlations = pd.concat([acc_correlations, dice_correlations])

    # Plot the correlations
    g = sns.catplot(data=subject_correlations, 
                    x="eval_metric", 
                    y="correlation", 
                    hue='cal_metric', 
                    kind="bar", 
                    col=col,
                    height=8, 
                    aspect=1)

    # Set the y lim between - 1 and 1
    g.set(ylim=(-1, 1))
    # Adjusting the titles
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(title, fontsize=16)
    plt.show()