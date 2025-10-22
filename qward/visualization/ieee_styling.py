"""
IEEE styling utilities for QWARD visualizations.

This module contains utility functions for applying IEEE publication standards
to matplotlib plots. These functions are separated from ieee_config.py to
avoid circular imports with base.py.
"""

# IEEE font size constants (from analy.py)
IEEE_FONT_SIZES = {"title_size": 18, "label_size": 16, "tick_size": 14, "legend_size": 12}

# IEEE styling constants
IEEE_STYLING = {
    "marker_size": 80,
    "line_width": 2,
    "alpha": 0.7,
    "grid_alpha": 0.7,
    "edge_color": "black",
    "edge_linewidth": 0.5,
}


def apply_ieee_rcparams_styling():
    """
    Apply IEEE styling to matplotlib rcParams for global settings.

    This function sets up the global matplotlib parameters for IEEE publication style.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "grid.color": "#cccccc",
            "grid.linestyle": "--",
            "grid.alpha": IEEE_STYLING["grid_alpha"],
            "text.color": "black",
            "font.size": 12,
            "axes.labelsize": IEEE_FONT_SIZES["label_size"],
            "axes.titlesize": IEEE_FONT_SIZES["title_size"],
            "xtick.labelsize": IEEE_FONT_SIZES["tick_size"],
            "ytick.labelsize": IEEE_FONT_SIZES["tick_size"],
            "legend.fontsize": IEEE_FONT_SIZES["legend_size"],
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "lines.linewidth": IEEE_STYLING["line_width"],
            "lines.markersize": IEEE_STYLING["marker_size"] / 10,  # rcParams expects smaller values
        }
    )


def apply_ieee_styling_to_axes(ax, title: str = None, xlabel: str = None, ylabel: str = None):
    """
    Apply IEEE styling directly to matplotlib axes.

    This function applies IEEE-specific styling to individual axes, complementing
    the global rcParams settings.

    Args:
        ax: Matplotlib axes object
        title: Optional title for the plot (None means no title)
        xlabel: Optional x-axis label
        ylabel: Optional y-axis label
    """
    # Apply font sizes and weights (using IEEE constants)
    if title is not None:
        ax.set_title(title, fontsize=IEEE_FONT_SIZES["title_size"], fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=IEEE_FONT_SIZES["label_size"], fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=IEEE_FONT_SIZES["label_size"], fontweight="bold")

    # Apply tick styling
    ax.tick_params(
        axis="both", which="major", labelsize=IEEE_FONT_SIZES["tick_size"], width=1.0, length=4
    )

    # Apply grid styling (using IEEE constants)
    ax.grid(True, linestyle="--", alpha=IEEE_STYLING["grid_alpha"], color="#cccccc", linewidth=0.8)

    # Apply legend styling if legend exists
    legend = ax.get_legend()
    if legend:
        legend.set_fontsize(IEEE_FONT_SIZES["legend_size"])
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(0.8)

    # Set axes properties
    ax.spines["top"].set_linewidth(1.0)
    ax.spines["right"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["left"].set_linewidth(1.0)

    # Set background
    ax.set_facecolor("white")
