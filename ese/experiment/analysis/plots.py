import numpy as np


def plot_reliability_diagram(
    bins,
    bar_heights,
    ax,
    title,
    x_label,
    y_label,
    bin_color='blue'
):
    interval_size = bins[1] - bins[0]
    aligned_bins = bins + (interval_size / 2) # shift bins to center
    ax.bar(aligned_bins, aligned_bins, width=interval_size, color='red', alpha=0.2)
    ax.bar(aligned_bins, bar_heights, width=interval_size, color=bin_color, alpha=0.5)

    # Plot diagonal line
    ax.plot([0, 1], [0, 1], linestyle='dotted', linewidth=3, color='gray')

    # Set title and axis labels
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Confidence")

    # Set x and y limits
    ax.set_xlim([0, 1])
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim([0, 1]) 
