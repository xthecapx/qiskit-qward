"""
Base classes for QWARD visualization system.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


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


class BaseVisualizer(ABC):
    """Base class for all QWARD visualizers."""

    def __init__(self, output_dir: str = "img", config: Optional[PlotConfig] = None):
        """
        Initialize the base visualizer.

        Args:
            output_dir: Directory to save plots
            config: Plot configuration settings
        """
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

    def save_plot(self, fig: plt.Figure, filename: str, **kwargs) -> str:
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

        save_kwargs = {
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

    @abstractmethod
    def create_plot(self) -> plt.Figure:
        """Create the main plot. Must be implemented by subclasses."""
        pass


class MetricVisualizer(BaseVisualizer):
    """Base class for metric-specific visualizers."""

    def __init__(
        self,
        metrics_dict: Dict[str, pd.DataFrame],
        output_dir: str = "img",
        config: Optional[PlotConfig] = None,
    ):
        """
        Initialize the metric visualizer.

        Args:
            metrics_dict: Dictionary of metric DataFrames
            output_dir: Directory to save plots
            config: Plot configuration settings
        """
        super().__init__(output_dir, config)
        self.metrics_dict = metrics_dict

    def get_metric_data(self, metric_name: str) -> Optional[pd.DataFrame]:
        """
        Get data for a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            DataFrame for the metric or None if not found
        """
        return self.metrics_dict.get(metric_name)

    def validate_columns(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate that a DataFrame has required columns.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Returns:
            True if all columns are present
        """
        return all(col in df.columns for col in required_columns)

    def add_value_labels(self, ax: plt.Axes, bars, format_str: str = "{:.3f}") -> None:
        """
        Add value labels on top of bars.

        Args:
            ax: Matplotlib axes
            bars: Bar plot objects
            format_str: Format string for labels
        """
        for bar_obj in bars:
            height = bar_obj.get_height()
            if height > 0:  # Only add labels for positive values
                ax.text(
                    bar_obj.get_x() + bar_obj.get_width() / 2.0,
                    height + 0.01,
                    format_str.format(height),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    def add_stacked_bar_summary(
        self, ax: plt.Axes, data: pd.DataFrame, position: str = "outside"
    ) -> None:
        """
        Add summary information for stacked bar charts in a readable format.

        Args:
            ax: Matplotlib axes
            data: DataFrame with the stacked bar data
            position: Where to place the summary ('outside', 'bottom_left', 'top_right')
        """
        # Calculate totals and create summary text
        summary_lines = []
        for idx, (job_name, row) in enumerate(data.iterrows()):
            total = row.sum()
            summary_lines.append(f"{job_name}: {int(total)} total")
            for col_name, value in row.items():
                percentage = (value / total * 100) if total > 0 else 0
                summary_lines.append(f"  {col_name}: {int(value)} ({percentage:.1f}%)")

        summary_text = "\n".join(summary_lines)

        # Position the summary text
        if position == "outside":
            # Place outside the plot area (to the right)
            ax.text(
                1.02,
                0.5,
                summary_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
            )
        elif position == "bottom_left":
            # Place in bottom left corner of plot
            ax.text(
                0.02,
                0.02,
                summary_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            )
        elif position == "top_right":
            # Place in top right corner of plot
            ax.text(
                0.98,
                0.98,
                summary_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            )
