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
    x_labels: bool = True,
    style: Optional[str] = None,
    col: Optional[Literal["bin_num"]] = None,
    facet_kws: Optional[dict] = None,
):
    if isinstance(pixel_preds, pd.core.frame.DataFrame):
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
        # We are dealing with a dictionary of pixel predictions as meters
        # for different criteria.
        key = ("pred_label", "num_neighbors", "bin_num", "measure")

        # Organize data into a structure for plotting
        # Structure: data_dict[bin_num][pred_label][measure] = list of values
        data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for (pred_label, _, bin_num, measure), value in pixel_preds.items():
            data_dict[bin_num][pred_label][measure].append(value)

        # Calculate the average for each pred_label and measure within each bin_num
        avg_data_dict = defaultdict(lambda: defaultdict(dict))

        for bin_num, pred_labels in data_dict.items():
            bin_conf_meter = StatsMeter()
            bin_acc_meter = StatsMeter()
            for pred_label, measures in pred_labels.items():
                for measure, values in measures.items():
                    y = StatsMeter()
                    for value in values:
                        y += value
                        if measure == "confidence":
                            bin_conf_meter += value
                        else:
                            bin_acc_meter += value
                    # Place the average of the averages in the avg_data_dict.
                    avg_data_dict[int(bin_num)][str(pred_label)][measure] = (y.mean, y.std) 
            # Place the average of the averages in the avg_data_dict.
            avg_data_dict[bin_num]["avg_label"]["confidence"] = (bin_conf_meter.mean, bin_conf_meter.std)
            avg_data_dict[bin_num]["avg_label"]["accuracy"] = (bin_acc_meter.mean, bin_acc_meter.std)
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
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols*6, num_rows*6), sharex=sharex, sharey=sharey)
        # Loop through the subplots and plot the data.
        for ax_idx, (bin_num, pred_labels) in enumerate(sorted_data):
            ax = axes[ax_idx // 5, ax_idx % 5]
            labels = sorted(list(pred_labels.keys()))
            measures = ["confidence", "accuracy"] 
            num_measures = len(measures)
            width = 0.8 / num_measures  # Width of bars, distributed over the number of measures
            ind = np.arange(len(labels))  # the x locations for the groups
            # Loop through both measures and plot them.
            for i, measure in enumerate(measures):
                values = [pred_labels[label][measure][0] for label in labels]
                std_values = [pred_labels[label][measure][1] for label in labels]
                ax.bar(
                    ind + i * width,
                    values,
                    width,
                    label=f"{measure}",
                    yerr=std_values,  # Add standard deviation bars
                    capsize=5,  # Customize the cap size of error bars
                )

            # Add some text for labels, title and axes ticks
            ax.set_title(f'{col} = {bin_num}')
            ax.set_ylabel('value')
            ax.set_xlabel(x)
            # Optionally remove the x labels
            if x_labels:
                ax.set_xticks(ind + width * (num_measures - 1) / 2)
                ax.set_xticklabels(labels)

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