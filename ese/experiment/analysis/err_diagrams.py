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
from typing import List, Optional, Literal, Any


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def viz_quality_metric_distributions(
    image_stats_df: pd.DataFrame,
    title: str,
    col_wrap: int = 3
    ) -> None:
    # Now using seaborn's FacetGrid to create the KDE plots for the 'accuracy' column for each 'split'.
    g = sns.FacetGrid(
        image_stats_df, 
        col="qual_metric", 
        col_wrap=col_wrap, 
        hue="qual_metric",
        sharex=False,
        sharey=False
        )
    g = g.map(sns.kdeplot, "qual_score", fill=True)
    # g.set(xlim=(0, 1))
    # Adjusting the layout
    g.fig.tight_layout()
    # Set the title for the entire FacetGrid
    plt.suptitle(title, fontsize=16)
    # Adjust layout to make room for the title
    plt.subplots_adjust(top=0.85)
    # Show plot
    plt.show()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def viz_calibration_metric_distributions(
    image_stats_df: pd.DataFrame,
    title: str,
    col_wrap: int = 3
    ) -> None:
    # Now using seaborn's FacetGrid to create the KDE plots for the 'accuracy' column for each 'split'.
    g = sns.FacetGrid(
        image_stats_df, 
        hue="cal_metric", 
        col="cal_metric", 
        col_wrap=col_wrap, 
        sharex=False,
        sharey=False
        )
    g = g.map(sns.kdeplot, "cal_m_score", fill=True)
    # g.set(xlim=(0, 1))
    # Adjusting the layout
    g.fig.tight_layout()
    # Set the title for the entire FacetGrid
    plt.suptitle(title, fontsize=16)
    # Adjust layout to make room for the title
    plt.subplots_adjust(top=0.85)
    # Show plot
    plt.show()


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
    pixel_preds: dict,
    title: str,
    x: str,
    kind: Literal["bar", "line"] = "bar",
    relative_props: bool = False,
    add_weighted: bool = False,
    add_edge_props: bool = False,
    label: Optional[int] = None,
    show_x_labels: bool = True,
    col: Optional[Literal["bin_num"]] = None,
    facet_kws: Optional[dict] = None,
):
    # Structure: data_dict[bin_num][pred_label][measure] = list of values
    # Organize data into a structure for plotting
    data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (pred_label, num_neighbors, bin_num, measure), value in pixel_preds.items():
        if (label is None) or (pred_label == label):
            pred_dict = {
                "pred_label": pred_label,
                "num_neighbors": num_neighbors,
                "bin_num": bin_num,
            }
            data_dict[bin_num][pred_dict[x]][measure].append(value)

    # Calculate the average for each pred_label and measure within each bin_num
    avg_data_dict = defaultdict(lambda: defaultdict(dict))
    # Helpful trackers for counting samples in each bin.
    bin_samples = {
        "edge": {},
        "uniform": {},
        "weighted": {}
    }
    # Go through each bin_num and each x variable. 
    for bin_num, x_var_set in data_dict.items():
        # SETUP A METER FOR EACH X GROUP, THIS WILL TRACK ONLY THE MEAN, STD, AND N
        # FOR EACH GROUP IN THIS BIN.
        bin_metric_meters = {mg: StatsMeter() for mg in ["accuracy", "weighted accuracy", "confidence", "weighted confidence"]}
        edge_pixels = 0
        # Loop through each x group and measure.
        for x_group, measures in x_var_set.items():
            for measure, values in measures.items():
                x_group_meter = StatsMeter()
                # Accumulate all of the values in this group, and at to our total bin trackers.
                for val in values:
                    x_group_meter += val
                    bin_metric_meters[measure] += val
                # Record the mean, std, and n for the bin.
                avg_data_dict[int(bin_num)][str(x_group)][measure] = {
                    "mean": x_group_meter.mean, 
                    "std": x_group_meter.std, 
                    "n_samples": x_group_meter.n
                } 
                if x == "num_neighbors" and measure == "accuracy" and x_group < 8:
                    edge_pixels += x_group_meter.n
        # Add the number of samples in this bin.
        bin_samples["edge"][bin_num] = edge_pixels
        bin_samples["uniform"][bin_num] = bin_metric_meters["accuracy"].n
        bin_samples["weighted"][bin_num] = bin_metric_meters["weighted accuracy"].n

    # NORMALIZE THE PROPORTION BY THE TOTAL NUMBER OF SAMPLES IN THE EXPERIMENT.
    total_samples = sum(bin_samples["uniform"].values())
    total_edge_samples = sum(bin_samples["edge"].values())
    total_weighted_samples = sum(bin_samples["weighted"].values())
    if label is None:
        assert np.abs(total_samples - total_weighted_samples) < 5,\
            f"Total uniform samples: {total_samples} and Total weighted samples: {total_weighted_samples} should be ~basically the same."
    # Loop through every bin and x_var
    for bin_num, x_var_set in avg_data_dict.items():
        for x_group in x_var_set.keys():
            x_group_dict = avg_data_dict[bin_num][x_group]
            # NORMALIZE THE STANDARD PROPORTIONS
            bin_x_samples = x_group_dict["accuracy"]["n_samples"]
            u_denom_samples = bin_samples["uniform"][bin_num] if relative_props else total_samples
            avg_data_dict[bin_num][x_group]["proportion"] = bin_x_samples / u_denom_samples 

            # NORMALIZE THE WEIGHTED PROPORTIONS
            weighted_bin_x_samples = x_group_dict["weighted accuracy"]["n_samples"]
            w_denom_samples = bin_samples["weighted"][bin_num] if relative_props else total_weighted_samples
            avg_data_dict[bin_num][x_group]["weighted proportion"] = weighted_bin_x_samples / w_denom_samples 

            # NORMALIZE THE EDGE PROPORTIONS
            if add_edge_props:
                edge_x_samples = x_group_dict["accuracy"]["n_samples"] if int(x_group) < 8 else 0 
                e_denom_samples = bin_samples["edge"][bin_num] if relative_props else total_edge_samples
                avg_data_dict[bin_num][x_group]["edge proportion"] = edge_x_samples / e_denom_samples 

    # These are the metrics we care about for each x group.
    measure_names = [
        "confidence", 
        "accuracy", 
        "weighted confidence", 
        "weighted accuracy",
        "proportion",
        "weighted proportion"
        ]
    if add_edge_props:
        measure_names += ["edge proportion"]

    # Set a bunch of information for plotting the graphs.
    num_bins = len(avg_data_dict)
    num_rows, num_cols = math.ceil(num_bins/5), 5
    sharex, sharey = True, True
    if facet_kws is not None:
        if "sharex" in facet_kws.keys():
            sharex = facet_kws["sharex"]
        if "sharey" in facet_kws.keys():
            sharey = facet_kws["sharey"]

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
        "confidence": "cornflowerblue",
        "weighted confidence": "mediumblue",
        "accuracy": "sandybrown",
        "weighted accuracy": "darkorange",
        "proportion": "darkgreen",
        "edge proportion": "mediumorchid",
        "weighted proportion": "lightgreen",
    }

    # Loop through the axes, and if there isn't a bin number for that axis, remove it first sorting the avg_data_dict indices by col.
    data_sorted_by_bin_num = sorted(avg_data_dict.items(), key=lambda x: x[0])
    bin_nums = [bin_num for bin_num, _ in data_sorted_by_bin_num] 
    for ax_idx, ax in enumerate(axes.flat):
        if ax_idx not in bin_nums:
            fig.delaxes(ax)
        
    # If not adding weighting, remove the weighted measures.
    if not add_weighted:
        for measure in [
            "weighted confidence", 
            "weighted accuracy", 
            "weighted proportion", 
        ]:
            measure_names.remove(measure)

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
                            edgecolor='none'
                        )
                    else:
                        ax.bar(
                            inds[x_idx] + m_idx * width,
                            x_var_set[x_var][measure]["mean"],
                            width,
                            label=bar_label,
                            capsize=3,  # Customize the cap size of error bars
                            color=bar_color,
                            edgecolor='none'
                        )
        else:
            raise NotImplementedError("Haven't configured that kind of plot yet.") 
        # Add some text for labels, title and axes ticks
        ax.set_title(f'{col} = {bin_num}')
        ax.set_ylabel('value')
        # Optionally remove the x labels
        if show_x_labels:
            ax.set_xlabel(x)
            if kind == "bar":
                ax.set_xticks(inds + width * (len(measures) - 1) / 2)
                ax.set_xticklabels(sorted_x_vars)
        else:
            ax.set_xticklabels([])
    # Adjusting the titles
    plt.legend(
        title='measure',
        bbox_to_anchor=(1.4, 1),  # Adjust the position of the legend
        loc='center right',  # Specify the legend location
        frameon=False,  # Remove the legend background
    )
    plt.subplots_adjust(top=0.5)
    plt.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.show()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def viz_cal_metric_corr(
    pixel_preds: pd.DataFrame,
    title: str,
    heatmap_row: str,
    heatmap_col: str,
    height: int = 5,
    aspect: float = 1,
    col: Optional[str] = None,
    row: Optional[str] = None,
) -> None:
    
    if 'split' in pixel_preds.keys(): 
        # Group by split, cal_metric, and qual_metric, then compute correlation
        grouped = pixel_preds.groupby(['split', 'cal_metric', 'qual_metric'])
    else:
        grouped = pixel_preds.groupby(['cal_metric', 'qual_metric'])

    correlations = grouped.apply(lambda g: g['cal_m_score'].corr(g['qual_score']))
    # Reset index to convert Series to DataFrame and name the correlation column
    correlations = correlations.reset_index(name='correlation')
    # Pivot the data.
    if 'split' in pixel_preds.keys():
        correlations_pivoted = correlations.pivot_table(index=['cal_metric', 'qual_metric'], columns='split', values='correlation').reset_index()
        correlations_melted = correlations_pivoted.melt(id_vars=['cal_metric', 'qual_metric'], var_name='split', value_name='correlation')
        # Reorder the splits in the melted dataframe.
        correlations_melted = reorder_splits(correlations_melted)
    else:
        correlations_melted = correlations.pivot_table(index=['cal_metric', 'qual_metric'], values='correlation').reset_index()
    # Add a new column for cal_metric_type if row is specified
    correlations_melted['cal_metric_type'] = correlations_melted['cal_metric'].str.contains('ECE').map({True: 'ECE', False: 'ELM'})
    # Initialize the FacetGrid with the reshaped DataFrame
    grid_kwargs = {
        'col': col, 
        'row': row, 
        'height': height, 
        'aspect': aspect, 
        'sharex': False, 
        'sharey': False
        }
    # Initialize a FacetGrid with kwargs.
    g = sns.FacetGrid(correlations_melted, **grid_kwargs)
    # Define the plot_heatmap function with annotations
    def plot_heatmap(data, **kwargs):
        pivot_data = data.pivot(index=heatmap_row, columns=heatmap_col, values='correlation')
        # Create a custom diverging colormap with red for -1 and green for 1
        custom_cmap = sns.diverging_palette(
            h_neg=10, 
            h_pos=120, 
            s=90, 
            l=40, 
            as_cmap=True
            )
        # Annotate each cell with the numeric value using `annot=True`
        sns.heatmap(pivot_data, 
                    annot=True, 
                    fmt=".2f",
                    cmap=custom_cmap,
                    center=0,
                    vmin=-1, 
                    vmax=1,
                    annot_kws={"color": "black", "weight": "bold", "fontsize":10},
                    **kwargs)
    # Use map_dataframe to draw heatmaps
    g.map_dataframe(plot_heatmap)
    # Rotate the y-labels 90 degrees clockwise
    g.set_yticklabels(rotation=0)
    # Adjust layout
    g.fig.tight_layout()
    # Set the title for the entire FacetGrid
    plt.suptitle(title, fontsize=16)
    # Adjust layout to make room for the title
    plt.subplots_adjust(top=0.9)
    # Show plot
    plt.show()


