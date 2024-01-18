import pandas as pd
import seaborn as sns


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