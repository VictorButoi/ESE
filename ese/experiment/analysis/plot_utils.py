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