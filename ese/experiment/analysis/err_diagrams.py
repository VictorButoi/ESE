import math
import numpy as np
import pandas as pd
import seaborn as sns
from ionpy.util import StatsMeter
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
    hue: Union[str, List[str]],
    kind: Literal["bar", "line"],
    add_average: bool = False,
    add_proportion: bool = False,
    x_labels: bool = True,
    style: Optional[str] = None,
    col: Optional[Literal["bin_num"]] = None,
    facet_kws: Optional[dict] = None,
):
    if isinstance(pixel_preds, pd.core.frame.DataFrame):
        assert not add_proportion, "add_proportion not implemented for DataFrame input."
        # Make a clone of the dataframe to not overwrite the original.
        pixel_preds_df = pixel_preds.copy()
        if add_average:
            # Calculate average conf and accuracy for each bin
            avg_bin_values = pixel_preds_df.groupby(['bin_num', 'measure']).agg({'value': 'mean'}).reset_index()
            # Prepare a list to store the new rows
            new_rows = []
            # Create new rows for the averages with a special pred label 'avg'
            for _, row in avg_bin_values.iterrows():
                new_rows.append({
                    'bin_num': row['bin_num'],
                    'measure': row['measure'],
                    'value': row['value'], 
                    'pred_label': 'avg',
                    'num_neighbors': 'avg',
                    'pred_label,num_neighbors': 'avg'
                    })
            average_df = pd.DataFrame(new_rows)
            # Concatenate the original DataFrame with the new rows if box/bar plot
            if kind in ['bar', 'box']:
                pixel_preds_df = pd.concat([pixel_preds_df, average_df], ignore_index=True)
        # Concatenate the original DataFrame with the new rows
        pixel_preds_df_sorted = pixel_preds_df.sort_values(by=[x, 'bin_num'], ascending=[True, True])
        # Using relplot to create a FacetGrid of bar plots
        if kind in ['bar']:
            sharex = True
            sharey = True
            if facet_kws is not None:
                sharex = facet_kws["sharex"]
                sharey = facet_kws["sharey"]
            # Plot the bar plots
            g = sns.catplot(data=pixel_preds_df_sorted, 
                            x=x, 
                            y='value', 
                            hue=hue,
                            col=col, 
                            col_wrap=5, 
                            kind='bar', 
                            height=5,
                            errorbar='sd',  # Add standard deviation bars
                            sharex=sharex,
                            sharey=sharey,
                            facet_kws=facet_kws)
        elif kind in ["line"]:
            g = sns.relplot(data=pixel_preds_df_sorted, 
                            x=x, 
                            y='value', 
                            hue=hue,
                            col=col, 
                            col_wrap=5, 
                            style=style,
                            kind='line', 
                            height=5,
                            facet_kws=facet_kws)
            if add_average:
                for ax, (bin_num_info, _) in zip(g.axes.flat, g.facet_data()):
                    # Get the bin num
                    bin_num = bin_num_info[1] + 1
                    # Filter the average_df for this particular 'col' (or 'bin_num')
                    subset_avg = average_df[average_df['bin_num'] == bin_num]
                    # Get average confidence for this 'bin_num'
                    avg_conf_value = subset_avg[subset_avg['measure'] == 'avg conf']['value'].mean()
                    ax.axhline(avg_conf_value, color='red', linestyle='--', label='avg conf', zorder=0)
                    # Get average accuracy for this 'bin_num'
                    avg_acc_value = subset_avg[subset_avg['measure'] == 'avg accuracy']['value'].mean()
                    ax.axhline(avg_acc_value, color='green', linestyle='--', label='avg accuracy', zorder=0)
        else:
            raise NotImplementedError("Haven't configured that kind of plot yet.") 
        # Optionally remove the x labels
        if not x_labels:
            g.set_xticklabels([])
        # Adjusting the titles
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(title, fontsize=16)
        plt.show()
    else:
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
        # Loop through the bins storing information.
        bin_samples = {}
        for bin_num, x_var_set in data_dict.items():
            bin_meters = {
                "confidence": StatsMeter(),
                "accuracy": StatsMeter()
            }
            for x_var, measures in x_var_set.items():
                for measure, values in measures.items():
                    y = StatsMeter()
                    for value in values:
                        y += value
                        bin_meters[measure] += value
                    # Place the average of the averages in the avg_data_dict.
                    avg_data_dict[int(bin_num)][str(x_var)][measure] = (y.mean, y.std) 
                if add_proportion:
                    avg_data_dict[int(bin_num)][str(x_var)]["proportion"] = y.n
            # Place the average of the averages in the avg_data_dict.
            if add_average:
                for measure in ["confidence", "accuracy"]:
                    avg_data_dict[int(bin_num)]["avg"][measure] = (bin_meters[measure].mean, bin_meters[measure].std)
                if add_proportion:
                    num_bin_samples = bin_meters["confidence"].n
                    avg_data_dict[int(bin_num)]["avg"]["proportion"] = num_bin_samples
                    bin_samples[int(bin_num)] = num_bin_samples
        # Keep track of the total number of samples in the experiment.
        total_samples = sum(bin_samples.values())
        # if we added the proportions, now we have to normalize them by the total, both
        # by bin and by x variable.
        if add_proportion:
            for bin_num, x_var_set in avg_data_dict.items():
                for x_var in x_var_set.keys():
                    if x_var != "avg":
                        avg_data_dict[bin_num][x_var]["proportion"] /= bin_samples[bin_num]
                avg_data_dict[bin_num]["avg"]["proportion"] /= total_samples
        # Sort the avg_data_dict indices by col
        sorted_data = sorted(avg_data_dict.items(), key=lambda x: x[0])
        # Plotting
        num_bins = len(avg_data_dict)
        num_rows = math.ceil(num_bins/5)
        num_cols = 5
        sharex = True
        sharey = True
        if facet_kws is not None:
            sharex = facet_kws["sharex"]
            sharey = facet_kws["sharey"]
        # Setup the subplot array.
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols*6, num_rows*6), sharex=sharex, sharey=sharey)
        # Define the colors for the plot.
        metric_colors = {
            "confidence": "blue",
            "accuracy": "darkorange",
            "proportion": "forestgreen",
            "avg confidence": "royalblue",
            "avg accuracy": "sandybrown",
            "avg proportion": "darkseagreen",
        }
        # Loop through the axes, and if there isn't a bin number for that axis, remove it. 
        bin_nums = [bin_num for bin_num, _ in sorted_data] 
        for ax_idx, ax in enumerate(axes.flat):
            if ax_idx not in bin_nums:
                fig.delaxes(ax)
        # Loop through the subplots and plot the data.
        for bin_num, x_var_set in sorted_data:
            ax = axes[bin_num // 5, bin_num % 5]
            sorted_x_vars = sorted(list(x_var_set.keys()))
            if kind == "bar":
                measures = ["confidence", "accuracy"]
                if add_proportion:
                    measures.append("proportion")
                width = 0.8 / len(measures)  # Width of bars, distributed over the number of measures
                # Loop through both measures and plot them.
                for i, measure in enumerate(measures):
                    inds = np.arange(len(sorted_x_vars))  # the x locations for the groups
                    for j, x_var in enumerate(sorted_x_vars):
                        # Determine the color and label based on whether the x_var is 'avg' or not
                        if x_var == 'avg':
                            bar_color = metric_colors[f'avg {measure}']
                            bar_label = f'avg {measure}'
                        else:
                            bar_color = metric_colors[measure]
                            bar_label = measure if j == 0 else ""  # Label only once for other measures
                        
                        if measure == "proportion":
                            value = x_var_set[x_var][measure]
                            ax.bar(
                                inds[j] + i * width,
                                value,
                                width,
                                label=bar_label,
                                color=bar_color,
                            )
                        else:
                            value, std_value = x_var_set[x_var][measure]
                            ax.bar(
                                inds[j] + i * width,
                                value,
                                width,
                                label=bar_label,
                                yerr=std_value,  # Add standard deviation bars
                                capsize=3,  # Customize the cap size of error bars
                                color=bar_color,
                            )
            elif kind == "line":
                assert not add_proportion, "add_proportion not implemented for line plot."
                measures = ["confidence", "accuracy"] 
                ax.axhline(x_var_set["avg"]["confidence"][0], color=metric_colors["avg confidence"], linestyle='--', label='avg confidence', zorder=0)
                ax.axhline(x_var_set["avg"]["accuracy"][0], color=metric_colors["avg accuracy"], linestyle='--', label='avg accuracy', zorder=0)
                for i, measure in enumerate(measures):
                    x_vars = [x_var for x_var in sorted_x_vars if x_var != "avg"]
                    values = [x_var_set[x_var][measure][0] for x_var in sorted_x_vars if x_var != "avg"]
                    inds = np.arange(len(sorted_x_vars) - 1)  # the x vars without the average.
                    ax.plot(
                        x_vars,
                        values,
                        label=f"{measure}",
                        marker='o',
                        color=metric_colors[measure],
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
            bbox_to_anchor=(1.4, 1),  # Adjust the position of the legend
            loc='center right',  # Specify the legend location
            frameon=False,  # Remove the legend background
        )
        plt.subplots_adjust(top=0.9)
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()