# misc imports 
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pydantic import validate_arguments
import pickle
import seaborn as sns


def visualize_dice_per_experiment(
    x: str, 
    y: str, 
    data: pd.DataFrame,
    hue: str, 
    color_map: dict, 
    col: str=None, 
    kind: str="bar", 
    show_group_legend: bool=False,
    sort_value: str='dice_score'
    ):

    # x here is often split/query_split
    if sort_value is not None:
        sorted_cat_df = data.groupby(['experiment', x,'group'])[sort_value].mean().reset_index().sort_values(sort_value, ascending=False)
    else:
        sorted_cat_df = data

    combined_color_map = {}

    for group_name in sorted_cat_df['group'].unique():
        group_experiments = data.loc[data['group'] == group_name, 'experiment'].unique()
        # Get the colors, which can be a palette
        group_color = color_map[group_name]
        if "sns" in group_color:
            group_palette = sns.color_palette(group_color.split(":")[1], len(group_experiments))
        else:
            group_palette = [group_color] * len(group_experiments)
        # Build the color map for this group
        group_color_map = dict(zip(group_experiments, group_palette))
        combined_color_map = {**combined_color_map, **group_color_map}

    show_legend = not show_group_legend
    g = sns.catplot(x=x,
                    y=y, 
                    col=col,
                    data=sorted_cat_df, 
                    hue=hue, 
                    kind=kind, 
                    height=7.5, 
                    aspect=2,
                    dodge=True,
                    palette=combined_color_map,
                    legend=show_legend)
    
    # create a custom legend using plt.legend
    if show_group_legend:
        handles = [plt.Rectangle((0,0),1,1, color=color_map[label]) for label in color_map]
        labels = list(color_map.keys())
        plt.legend(handles, labels, loc='center right', bbox_to_anchor=(1.23, 0.5), borderaxespad=0., title="Group")

    return g


def compute_support_distribution(
        df: pd.DataFrame,
        percentile: float=0.95, 
        threshold_up: bool=True
        ):
    
    prop_used_dfs = []
    for subject, subject_df in df.groupby('subject'):
        # Calculate the dice score threshold for this subject
        threshold = subject_df['dice_score'].quantile(percentile)
        # Filter the dataframe to include only rows with dice scores >= threshold for this subject
        if threshold_up:
            filtered_df = subject_df.loc[subject_df['dice_score'] >= threshold]
        else:
            filtered_df = subject_df.loc[subject_df['dice_score'] <= threshold]

        # Drop the dice_score because now it doesn't matter
        filtered_df = filtered_df.drop(['dice_score'], axis=1)
        # Count the number of times each datapoint was used in predictions for this subject
        used_counts = filtered_df.iloc[:, 1:].sum()
        # Calculate the proportion of times each datapoint was used in predictions for this subject
        prop_used = used_counts / used_counts.sum()
        prop_used = prop_used.astype(float)
        assert np.abs(prop_used.sum() - 1.0) < 1e-6, "proportions must be very close to 1.0."
        # Create a new dataframe with the proportion used for each datapoint, and the subject and datapoint labels
        prop_used_df = pd.DataFrame({'proportion_used': prop_used})
        prop_used_df['datapoint'] = used_counts.index
        prop_used_df['subject'] = subject

        # Drop any rows where the datapoint label is "dice_score"
        prop_used_df = prop_used_df[prop_used_df['datapoint'] != 'dice_score']
        # Re-order the columns of the dataframe and sort by proportion used (in descending order)
        prop_used_df = prop_used_df[['subject', 'datapoint', 'proportion_used']]
        # Append the resulting dataframe to the list of dataframes for all subjects
        prop_used_dfs.append(prop_used_df)
    # Concatenate all the resulting dataframes into a single dataframe
    result_df = pd.concat(prop_used_dfs, axis=0)

    # Pivot the dataframe to create a matrix of proportion used values, with subjects on the rows and datapoints on the columns
    result_matrix = result_df.pivot(index='subject', columns='datapoint', values='proportion_used')
    # Return the matrix of proportion used values
    return result_matrix


def insert_baseline_and_bounds(
    graph, 
    b_and_b: pd.DataFrame, 
    colors: dict, 
    x_lim: list=None, 
    y_lim: list=None, 
    alpha: float=0.1
    ):

    x_data = graph.axes.flat[0].lines[0].get_xdata()

    experiments = b_and_b['experiment'].unique()
    splits = b_and_b['split'].unique()
    exp_and_split = [(i, j) for i in experiments for j in splits]

    # Iterate through experiments
    for (exp, split) in exp_and_split:

        # Get the dataframe for this experiment and split
        exp_df = b_and_b.select(experiment=exp, split=split)
        value_set_dice = exp_df['dice_score']

        # Val plot first, test second
        pos = 0 if exp_df['split'].iloc[0] == 'val' else 1

        # Compute the mean and std of the dice scores
        mean = value_set_dice.mean()
        std = value_set_dice.var()

        # Plot the mean and the std
        group = exp_df["group"].unique()[0]
        graph.axes.flat[pos].axhline(y=mean, color=colors[group], linestyle='--', label=exp)
        graph.axes.flat[pos].fill_between(x_data, mean - std, mean + std, interpolate=True, color=colors[group], alpha=alpha, zorder=-1)

    for ax in graph.axes.flat:
        if x_lim:
            ax.set(xlim=x_lim)
        if y_lim:
            ax.set(ylim=y_lim)
    
    # Get the handles and labels for the original legend
    handles, labels = graph.axes.flat[0].get_legend_handles_labels()

    # Add the baseline and bounds to the legend by inserting a space and then the name.
    handles.insert(-len(experiments), plt.Line2D([], [], color='white', marker='None', linestyle='None'))
    labels.insert(-len(experiments), "")
    handles.insert(-len(experiments), plt.Line2D([], [], color='white', marker='None', linestyle='None'))
    labels.insert(-len(experiments), "Baselines and Bounds")

    # Create new legend with white background and no outline
    legend_tile = graph._legend.get_title().get_text()
    graph._legend.remove()
    legend = graph.fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.05, 0.5), facecolor='white', frameon=False)

    # Set legend background color to white
    legend.set_frame_on(False)
    legend.get_frame().set_facecolor('white')

    # Add the title to the legend based on hue variable
    legend.set_title(legend_tile)
    legend.get_title().set_fontsize('11')


@validate_arguments
def load_baseline_and_bounds(
    task: str,
    use_cached: bool=True
    ):
    
    cache_file = f"/storage/vbutoi/scratch/DisegStuff/{task}_b_and_b.pickle"
    if use_cached and os.path.exists(cache_file):
        print("USING CACHED BASELINES AND BOUNDS.")
        with open(cache_file, 'rb') as f:
            b_and_b = pickle.load(f)
    else:
        df_file = f"/storage/vbutoi/scratch/DisegStuff/{task}_random_predictions.pickle"

        with open(df_file, 'rb') as f:
            baseline_df = pickle.load(f)
        cols_to_keep = ['subject', 'split', 'dice_score', 'support_indices']
        baseline_df = baseline_df[cols_to_keep]

        def get_best_supports(ind_df):
            subj_df = ind_df.groupby(['subject', 'split'])
            best_supports = subj_df.apply(lambda x: x.loc[x['dice_score'].idxmax()])
            return best_supports.reset_index(drop=True)

        # Get the best supports
        best_support_df = get_best_supports(baseline_df)
        best_support_df['experiment'] = 'upper-bound'
        best_support_df['group'] = 'upper-bound'

        # Give name to baseline 
        baseline_df['experiment'] = 'random'
        baseline_df['group'] = 'random'

        # Load the distance based baselines
        dist_baselines = load_distance_baselines(task)
    
        def get_best_dist_baseline(dist_df):
            # Calculate the average dice_score for each experiment_name and split
            average_scores = dist_df.groupby(['experiment', 'comp_pair', 'split'])['dice_score'].mean().to_frame().reset_index()
            # Find the experiment_name with the best average dice_score in val and test split
            val_experiments = average_scores.select(split='val')
            # Get by comp pair
            x_to_x_val_experiments = val_experiments[val_experiments['comp_pair'] == 'xx']
            y_to_y_val_experiments = val_experiments[val_experiments['comp_pair'] == 'yy']
            # Get the best experiments
            best_x_to_x_val_experiment = x_to_x_val_experiments.loc[x_to_x_val_experiments['dice_score'].idxmax(), 'experiment']
            best_y_to_y_val_experiment = y_to_y_val_experiments.loc[y_to_y_val_experiments['dice_score'].idxmax(), 'experiment']
            # Find the experiment_name with the best average dice_score in val split
            best_xx_val_df = dist_df.select(experiment=best_x_to_x_val_experiment, comp_pair='xx')
            best_xx_val_df['group'] = 'xx'
            best_yy_val_df = dist_df.select(experiment=best_y_to_y_val_experiment, comp_pair='yy')
            best_yy_val_df['group'] = 'yy'

            return pd.concat([best_xx_val_df, best_yy_val_df], axis=0)

        # Get the best supports
        best_distance_baseline_df = get_best_dist_baseline(dist_baselines)

        # Combine our two dataframes
        b_and_b = pd.concat([best_support_df, best_distance_baseline_df, baseline_df], axis=0)

        # Save b_and_b to pickle file
        b_and_b.to_pickle(cache_file)

    return b_and_b


@validate_arguments
def load_distance_baselines(
    task: str
    ):

    file_root = f"/storage/vbutoi/scratch/DisegStuff/{task}_predictions_by_metric.pickle"

    with open(file_root, 'rb') as f:
        baseline_df = pickle.load(f)

    def experiment(loss_func, comp_pair, sample, replace, probabilistic, exp_scale, top_k):
        if sample == False:
            return f"{loss_func}_{comp_pair}_topk:{top_k}"
        else:
            return f"{loss_func}_{comp_pair}_repl:{replace}_prob:{probabilistic}_exp:{exp_scale}_tk:{top_k}"

    # add experiment_column
    baseline_df.augment(experiment)
    cols_to_keep = ['subject', 'experiment', 'comp_pair', 'split', 'dice_score']
    dist_df = baseline_df[cols_to_keep]

    return dist_df



