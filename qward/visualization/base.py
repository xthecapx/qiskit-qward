"""
Base classes for QWARD visualization system.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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

    @abstractmethod
    def create_plot(self) -> plt.Figure:
        """Create the main plot. Must be implemented by subclasses."""
        pass
