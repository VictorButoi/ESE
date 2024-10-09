import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt 


def plot_upperbound_line(graph, plot_df, y, num_calibrators, col=None):
    if col is None:
        # Put the upper bound line on the plot
        metric_col = plot_df[y]
        mean_upper_bound = metric_col.mean()
        confidence_interval = stats.t.interval(0.95, len(metric_col)-1, loc=mean_upper_bound, scale=stats.sem(metric_col))
        #Plot the upper bound line on ax
        plt.axhline(y=mean_upper_bound, color='magenta', linestyle='--')
        # Plot area around the upper bound corresponding to 95 percent confidence interval
        plt.fill_between(
            x=[-1, num_calibrators],
            y1=confidence_interval[0],
            y2=confidence_interval[1],
            color='magenta',
            alpha=0.2
        )
    else:
        # Get the unique possible col values
        col_values = plot_df[col].unique()
        for ax in graph.axes.flat:
            # Get the row and column indices of the current subplot
            row_index, col_index = ax.get_subplotspec().rowspan.start, ax.get_subplotspec().colspan.start
            col_val = col_values[col_index]
            print(col_index, col_val)
            # Extract data for the current subplot
            metric_col = plot_df[plot_df[col] == col_val][y]
            mean_upper_bound = metric_col.mean()
            confidence_interval = stats.t.interval(0.95, len(metric_col)-1, loc=mean_upper_bound, scale=stats.sem(metric_col))
            #Plot the upper bound line on ax
            ax.axhline(y=mean_upper_bound, color='magenta', linestyle='--')
            # Plot area around the upper bound corresponding to 95 percent confidence interval
            ax.fill_between(
                x=[-1, num_calibrators],
                y1=confidence_interval[0],
                y2=confidence_interval[1],
                color='magenta',
                alpha=0.2
            )


def build_ensemble_vs_individual_cmap(dice_image_df):
    # Build a custom color palette where each configuration is mapped to a color_map
    # corresponding to if it is an ensemble or individual model.
    num_individual_configurations = len(dice_image_df[dice_image_df['ensemble'] == False]['configuration'].unique())
    num_ensemble_configurations = len(dice_image_df[dice_image_df['ensemble'] == True]['configuration'].unique())
    # Define the palettes
    individual_palette = sns.color_palette("rocket", num_individual_configurations)
    ensemble_palette = sns.color_palette("mako", num_ensemble_configurations) 
    # Build the color map
    individual_colors = {}
    for i, configuration in enumerate(dice_image_df[dice_image_df['ensemble'] == False]['configuration'].unique()):
        individual_colors[configuration] = individual_palette[i]
    ensemble_colors = {}
    for i, configuration in enumerate(dice_image_df[dice_image_df['ensemble'] == True]['configuration'].unique()):
        ensemble_colors[configuration] = ensemble_palette[i]
    # Combine the two color maps
    return {
        **individual_colors,
        **ensemble_colors
    }


def add_corr_coefficients(g, data, x, y, row, col):
    # Calculate and display correlation coefficient in each subplot title
    for ax in g.axes.flat:
        # Get the row and column indices of the current subplot
        row_index, col_index = ax.get_subplotspec().rowspan.start, ax.get_subplotspec().colspan.start
        row_method = g.row_names[row_index]
        col_calibrator = g.col_names[col_index]
        # Extract data for the current subplot
        x_data = data[data[row] == row_method][x]
        y_data = data[data[col] == col_calibrator][y]
        # Calculate correlation coefficient
        correlation_coefficient = x_data.corr(y_data)
        # Check if correlation coefficient is NaN
        if pd.isnull(correlation_coefficient):
            correlation_coefficient = 0
        # Get the existing title and append the correlation coefficient
        existing_title = ax.get_title()
        new_title = f'{existing_title}\nCorrelation: {correlation_coefficient:.2f}'
        # Set the updated title
        ax.set_title(new_title)


def add_axis_lines(g, color, linewidth, zorder):
    # Add x and y axis lines to each subplot using Matplotlib
    for ax in g.axes.flat:
        ax.axhline(0, color=color, linewidth=linewidth, zorder=zorder)  # Horizontal line
        ax.axvline(0, color=color, linewidth=linewidth, zorder=zorder)  # Vertical line


def plot_method_vs_calibrator_scatterplots(df, x, y, sharex=False, sharey=False, height=4):
    # Plot the relationship between the two metrics
    g = sns.relplot(
        data=df,
        x=x, 
        y=y,
        row='method_name',
        col='calibrator',
        hue='method_name',
        style='calibrator',
        kind='scatter',
        height=height,
        facet_kws={
            "margin_titles": True,
            "sharex": sharex,
            "sharey": sharey
        }
    )
    g.set_titles("")  # Set titles to empty string
    # Show the plot
    g.fig.subplots_adjust(hspace=0.2, wspace=0.2)
    # Add correlation coefficients
    add_corr_coefficients(
        g, 
        data=df, 
        x=x, 
        y=y,
        row='method_name',
        col='calibrator'
    )
    add_axis_lines(g, color='darkgrey', linewidth=1, zorder=1)
    # Add a title to the entire figure, and make it slightly bigger than the default
    g.fig.suptitle(f'{x} vs {y}', size=20)
    g.fig.subplots_adjust(top=0.9)
    # Show the plot
    plt.show()


def clump_df_datapoints(df: pd.DataFrame, num_bins: int, x: str, y: str, x_metric: str, y_metric: str) -> pd.DataFrame:
    # Make a copy of the dataframe
    df_copy = df.copy()
    # Iterate through the combination of unique method names and calibrators and determine the bins for each.
    for x_val, y_val in df_copy[[x, y]].drop_duplicates().values:
        # Get the rows corresponding to the method and calibrator
        rows = df_copy[
            (df_copy[x] == x_val) & 
            (df_copy[y] == y_val)
        ]
        # Use pandas qcut to create quantile-based bins and calculate average x and y values within each bin
        df_copy.loc[rows.index, 'bin'] = pd.qcut(rows[x].rank(method='first'), q=num_bins, labels=False)
    # Collapse the points in the bins.
    return df_copy.groupby([x, y, 'bin']).agg({
        x_metric: 'mean', 
        y_metric: 'mean'
        }).reset_index()

    
def get_prop_color_palette(df, hue_key, magnitude_key):
    # Step 1: Create a DataFrame mapping 'data_id' to 'gt_volume' (unique pairs)
    data_id_to_volume = df[[hue_key, magnitude_key]].drop_duplicates()
    # Step 2: Normalize the 'gt_volume' values to the range [0, 1]
    norm = plt.Normalize(vmin=data_id_to_volume[magnitude_key].min(), vmax=data_id_to_volume[magnitude_key].max())
    # Step 3: Get the 'magma' colormap
    cmap = plt.cm.get_cmap('magma')
    # Step 4: Map the normalized 'gt_volume' values to colors
    data_id_to_volume['color'] = data_id_to_volume[magnitude_key].apply(lambda x: cmap(norm(x)))
    # Step 5: Create a palette mapping 'data_id' to colors
    return dict(zip(data_id_to_volume[hue_key], data_id_to_volume['color']))