import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional
from pydantic import validate_arguments


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


def viz_accuracy_vs_confidence(

):
    # Calculate average conf and accuracy for each bin
    avg_values = pixel_preds_df.groupby('bin').agg({'conf': 'mean', 'accuracy': 'mean'}).reset_index()

    # Prepare a list to store the new rows
    new_rows = []

    # Create new rows for the averages with a special label 'avg'
    for _, row in avg_values.iterrows():
        new_rows.append({'bin': row['bin'], 'conf': row['conf'], 'accuracy': row['accuracy'], 'label': 'avg'})

    # Concatenate the original DataFrame with the new rows
    pixel_preds_df = pd.concat([pixel_preds_df, pd.DataFrame(new_rows)], ignore_index=True)

    # Melt the DataFrame
    pixel_preds_df_melted = pixel_preds_df.melt(id_vars=['bin', 'bin_num', 'label'], value_vars=['conf', 'accuracy'], var_name='metric', value_name='value')
    pixel_preds_df_melted = pixel_preds_df_melted.sort_values('bin_num')

    # Define a custom palette
    unique_labels = pixel_preds_df_melted['label'].unique()
    palette_colors = sns.color_palette('viridis', n_colors=len(unique_labels) - 1)  # -1 because we'll assign a distinct color to 'avg'
    palette_dict = {label: color for label, color in zip(unique_labels, palette_colors)}
    palette_dict['avg'] = 'gray'  # Assigning red color for 'avg' label

    # Using relplot to create a FacetGrid of scatter plots
    g = sns.catplot(data=pixel_preds_df_melted, 
                    x='metric', 
                    y='value', 
                    hue='label',
                    col='bin', 
                    col_wrap=5, 
                    kind='bar', 
                    height=5,
                    palette=palette_dict)
    # Adjusting the titles
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('WMH Confidence vs. Accuracy per Bin', fontsize=16)