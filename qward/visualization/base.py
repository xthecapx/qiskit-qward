"""
Base classes for QWARD visualization system.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd


class PlotType(Enum):
    """Enumeration of plot types for metadata."""

    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    RADAR_CHART = "radar_chart"
    STACKED_BAR = "stacked_bar"
    GROUPED_BAR = "grouped_bar"
    LINE_CHART = "line_chart"
    SCATTER_PLOT = "scatter_plot"


@dataclass
class PlotMetadata:
    """Metadata for individual plots."""

    name: str
    method_name: str
    description: str
    plot_type: PlotType
    filename: str
    dependencies: List[str] = None
    category: str = None


# Type definitions for the new schema
PlotResult = Dict[str, plt.Figure]
PlotRegistry = Dict[str, PlotMetadata]


@dataclass
class PlotConfig:
    """Configuration for plot appearance and saving."""

    figsize: Tuple[int, int] = (10, 6)
    dpi: int = 300
    style: str = "default"
    color_palette: List[str] = None
    save_format: str = "png"
    grid: bool = True
    alpha: float = 0.7

    def __post_init__(self):
        """Set default color palette if not provided."""
        if self.color_palette is None:
            # ColorBrewer-inspired palette for better visualization
            self.color_palette = [
                "#1f77b4",  # Blue
                "#ff7f0e",  # Orange
                "#2ca02c",  # Green
                "#d62728",  # Red
                "#9467bd",  # Purple
                "#8c564b",  # Brown
                "#e377c2",  # Pink
                "#7f7f7f",  # Gray
                "#bcbd22",  # Olive
                "#17becf",  # Cyan
            ]


class VisualizationStrategy(ABC):
    """Base class for all QWARD visualization strategies."""

    def __init__(
        self,
        metrics_dict: Dict[str, pd.DataFrame],
        output_dir: str = "img",
        config: Optional[PlotConfig] = None,
    ):
        """
        Initialize the visualization strategy.

        Args:
            metrics_dict: Dictionary containing metrics data for this strategy
            output_dir: Directory to save plots
            config: Plot configuration settings
        """
        self.metrics_dict = metrics_dict
        self.output_dir = output_dir
        self.config = config or PlotConfig()
        self._setup_output_dir()
        self._apply_style()

    def _setup_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)

    def _apply_style(self) -> None:
        """Apply the specified plotting style."""
        if self.config.style == "quantum":
            plt.style.use("seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default")
            plt.rcParams.update(
                {
                    "figure.facecolor": "white",
                    "axes.facecolor": "#f8f9fa",
                    "axes.edgecolor": "#dee2e6",
                    "grid.color": "#e9ecef",
                    "text.color": "#212529",
                }
            )
        elif self.config.style == "minimal":
            plt.style.use(
                "seaborn-v0_8-whitegrid"
                if "seaborn-v0_8-whitegrid" in plt.style.available
                else "default"
            )
        else:
            plt.style.use("default")

    def save_plot(self, fig: plt.Figure, filename: str, **kwargs: Any) -> str:
        """
        Save a plot with consistent settings.

        Args:
            fig: Matplotlib figure to save
            filename: Name of the file (without extension)
            **kwargs: Additional arguments for savefig

        Returns:
            Full path to saved file
        """
        filepath = os.path.join(self.output_dir, f"{filename}.{self.config.save_format}")

        save_kwargs: Dict[str, Any] = {
            "dpi": self.config.dpi,
            "bbox_inches": "tight",
            "facecolor": "white",
            "edgecolor": "none",
        }
        save_kwargs.update(kwargs)

        fig.savefig(filepath, **save_kwargs)
        return filepath

    def show_plot(self, fig: plt.Figure) -> None:
        """Display a plot."""
        plt.show()

    # =============================================================================
    # Common Utility Methods
    # =============================================================================

    def _validate_required_columns(
        self, df: pd.DataFrame, required_cols: List[str], data_name: str = "data"
    ) -> None:
        """
        Validate that required columns exist in the DataFrame.

        Args:
            df: DataFrame to validate
            required_cols: List of required column names
            data_name: Name of the data for error messages

        Raises:
            ValueError: If required columns are missing
        """
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{data_name} missing required columns: {missing_cols}")

    def _extract_metrics_from_columns(
        self,
        df: pd.DataFrame,
        column_patterns: List[str],
        prefix_to_remove: str = "",
        row_index: int = 0,
    ) -> Dict[str, Any]:
        """
        Extract metrics from DataFrame columns based on patterns.

        Args:
            df: DataFrame containing the metrics
            column_patterns: List of column name patterns to match
            prefix_to_remove: Prefix to remove from column names for display
            row_index: Row index to extract values from

        Returns:
            Dictionary mapping display names to values
        """
        metrics_data = {}

        for pattern in column_patterns:
            if pattern in df.columns:
                # Create display name by removing prefix and formatting
                display_name = pattern.replace(prefix_to_remove, "").replace("_", " ").title()
                metrics_data[display_name] = df[pattern].iloc[row_index]

        return metrics_data

    def _create_bar_plot_with_labels(
        self,
        *,
        data: Union[pd.Series, Dict[str, Any]],
        ax: plt.Axes,
        title: str,
        xlabel: str = "Metrics",
        ylabel: str = "Value",
        value_format: str = "auto",
    ) -> None:
        """
        Create a bar plot with value labels on top of bars.

        Args:
            data: Data to plot (Series or dict)
            ax: Matplotlib axes to plot on
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            value_format: Format for value labels ("auto", "int", "float", or custom format string)
        """
        if isinstance(data, dict):
            data = pd.Series(data)

        # Check if data is empty - handle both dict and Series cases
        if len(data) == 0:
            self._show_no_data_message(ax, title)
            return

        # Create bar plot
        data.plot(
            kind="bar",
            ax=ax,
            color=self.config.color_palette[: len(data)],
            alpha=self.config.alpha,
        )

        # Set labels and styling
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if self.config.grid:
            ax.grid(True, alpha=0.3)

        ax.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        self._add_value_labels_to_bars(ax, data.values, value_format)

    def _add_value_labels_to_bars(
        self, ax: plt.Axes, values: List[float], value_format: str = "auto"
    ) -> None:
        """
        Add value labels on top of bars.

        Args:
            ax: Matplotlib axes
            values: List of bar values
            value_format: Format for labels ("auto", "int", "float", or format string)
        """
        # Convert to list if it's a numpy array or pandas series
        if hasattr(values, "tolist"):
            values = values.tolist()
        elif hasattr(values, "values"):
            values = values.values.tolist()

        # Check if values is empty using len() to avoid array truth value issues
        if len(values) == 0:
            return

        max_value = max(values)

        for i, v in enumerate(values):
            # Ensure v is a scalar
            if hasattr(v, "item"):
                v = v.item()

            # Determine format
            if value_format == "auto":
                if isinstance(v, int) or (isinstance(v, float) and v.is_integer()):
                    label = str(int(v))
                else:
                    label = f"{v:.3f}" if abs(v) < 1 else f"{v:.1f}"
            elif value_format == "int":
                label = str(int(v))
            elif value_format == "float":
                label = f"{v:.3f}"
            else:
                label = value_format.format(v)

            ax.text(i, v + max_value * 0.01, label, ha="center", va="bottom", fontweight="bold")

    def _show_no_data_message(self, ax: plt.Axes, title: str, message: str = None) -> None:
        """
        Show a "no data available" message on the axes.

        Args:
            ax: Matplotlib axes
            title: Plot title
            message: Custom message (default: "No data available")
        """
        if message is None:
            message = "No data available"

        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgray", "alpha": 0.8},
        )
        ax.set_title(title)

    def _setup_plot_axes(
        self, fig_ax_override: Optional[Tuple[plt.Figure, plt.Axes]] = None
    ) -> Tuple[plt.Figure, plt.Axes, bool]:
        """
        Set up plot figure and axes, handling the override pattern.

        Args:
            fig_ax_override: Optional tuple of (figure, axes) to use instead of creating new

        Returns:
            Tuple of (figure, axes, is_override) where is_override indicates if override was used
        """
        if fig_ax_override:
            fig, ax = fig_ax_override
            return fig, ax, True
        else:
            fig, ax = plt.subplots(figsize=self.config.figsize)
            return fig, ax, False

    def _finalize_plot(
        self,
        *,
        fig: plt.Figure,
        is_override: bool,
        save: bool,
        show: bool,
        filename: str,
    ) -> plt.Figure:
        """
        Finalize plot with tight layout, saving, and showing.

        Args:
            fig: Matplotlib figure
            is_override: Whether this plot is part of a larger figure
            save: Whether to save the plot
            show: Whether to show the plot
            filename: Filename for saving (without extension)

        Returns:
            The figure object
        """
        if not is_override:
            plt.tight_layout()

        if save and not is_override:
            self.save_plot(fig, filename)
        if show and not is_override:
            self.show_plot(fig)

        return fig

    def _format_column_name_for_display(self, column_name: str, prefix_to_remove: str = "") -> str:
        """
        Format a column name for display by removing prefix and formatting.

        Args:
            column_name: Original column name
            prefix_to_remove: Prefix to remove

        Returns:
            Formatted display name
        """
        display_name = column_name.replace(prefix_to_remove, "")
        # Handle nested prefixes (e.g., "metrics.sub_metrics.value" -> "value")
        if "." in display_name:
            display_name = display_name.split(".")[-1]
        return display_name.replace("_", " ").title()

    @classmethod
    @abstractmethod
    def get_available_plots(cls) -> List[str]:
        """Return list of available plot names for this strategy."""
        pass

    @classmethod
    @abstractmethod
    def get_plot_metadata(cls, plot_name: str) -> PlotMetadata:
        """Get metadata for a specific plot."""
        pass

    def generate_plot(
        self, plot_name: str, save: bool = False, show: bool = False, **kwargs
    ) -> plt.Figure:
        """
        Generate a single plot by name.

        Args:
            plot_name: Name of the plot to generate
            save: Whether to save the plot
            show: Whether to display the plot
            **kwargs: Additional arguments passed to the plot method

        Returns:
            matplotlib Figure object

        Raises:
            ValueError: If plot_name is not available
        """
        if plot_name not in self.get_available_plots():
            available = self.get_available_plots()
            raise ValueError(f"Plot '{plot_name}' not available. Available plots: {available}")

        metadata = self.get_plot_metadata(plot_name)
        method = getattr(self, metadata.method_name)
        return method(save=save, show=show, **kwargs)

    def generate_plots(
        self, plot_names: List[str] = None, save: bool = False, show: bool = False, **kwargs
    ) -> PlotResult:
        """
        Generate multiple plots and return as dictionary.

        Args:
            plot_names: List of plot names to generate. If None, generates all available plots.
            save: Whether to save the plots
            show: Whether to display the plots
            **kwargs: Additional arguments passed to plot methods

        Returns:
            Dictionary mapping plot names to Figure objects
        """
        if plot_names is None:
            plot_names = self.get_available_plots()

        results = {}
        for plot_name in plot_names:
            try:
                results[plot_name] = self.generate_plot(plot_name, save=save, show=show, **kwargs)
            except Exception as e:
                print(f"Warning: Failed to generate plot '{plot_name}': {e}")
                continue

        return results

    def generate_all_plots(self, save: bool = False, show: bool = False, **kwargs) -> PlotResult:
        """
        Generate all available plots.

        Args:
            save: Whether to save the plots
            show: Whether to display the plots
            **kwargs: Additional arguments passed to plot methods

        Returns:
            Dictionary mapping plot names to Figure objects
        """
        return self.generate_plots(save=save, show=show, **kwargs)

    @abstractmethod
    def create_dashboard(self, save: bool = False, show: bool = False) -> plt.Figure:
        """Create a comprehensive dashboard for this strategy's metrics."""
        pass
