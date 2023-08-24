def reliability_diagram(
    bins,
    bar_heights,
    ax,
    title,
    x_label,
    y_label,
    bin_color='blue'
):
    interval_size = bins[1] - bins[0]
    ax.bar(bins, bins, width=interval_size, color='red', alpha=0.2)
    ax.bar(bins, bar_heights, width=interval_size, color=bin_color, alpha=0.5)
    ax.plot([0, 1], [0, 1], linestyle='dotted', linewidth=3, color='gray')
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Confidence")
    ax.set_xlim([0, 1])
