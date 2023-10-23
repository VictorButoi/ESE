import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pydantic import validate_arguments
from typing import List, Optional, Union, Literal


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
    preds_df: pd.DataFrame,
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
    # Make a clone of the dataframe to not overwrite the original.
    pixel_preds_df = preds_df.copy()

    if add_average:
        # Calculate average conf and accuracy for each bin
        avg_bin_values = pixel_preds_df.groupby(['bin_num', 'bin', 'measure']).agg({'value': 'mean'}).reset_index()
        # Prepare a list to store the new rows
        new_rows = []
        # Create new rows for the averages with a special label 'avg'
        for _, row in avg_bin_values.iterrows():
            new_rows.append({
                'bin': row['bin'], 
                'bin_num': row['bin_num'],
                'measure': f"avg {row['measure']}",
                'value': row['value'], 
                'label': 'avg',
                'num_neighbors': 'avg',
                'label,num_neighbors': 'avg'
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
            if not facet_kws["sharex"]:
                sharex = False
            if not facet_kws["sharey"]:
                sharey = False
        # Plot the bar plots
        g = sns.catplot(data=pixel_preds_df_sorted, 
                        x=x, 
                        y='value', 
                        hue=hue,
                        col=col, 
                        col_wrap=5, 
                        kind='bar', 
                        height=5,
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
        # Disable the x-ticks
        g.set_xticklabels([])

    # Adjusting the titles
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(title, fontsize=16)
    plt.show()