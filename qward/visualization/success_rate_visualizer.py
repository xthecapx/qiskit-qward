"""
SuccessRate visualization module for QWARD.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .base import BaseVisualizer, PlotConfig


# --- Strategy Interface ---
class PlotStrategy(ABC):
    """
    Abstract base class for a plot generation strategy.
    Each concrete strategy encapsulates the algorithm to create a specific plot.
    """
    def __init__(self, visualizer: 'SuccessRateVisualizer'):
        self.visualizer = visualizer
        self.config = visualizer.config  # Access config from the visualizer

    @abstractmethod
    def plot(self, ax: plt.Axes, is_dashboard_context: bool = False) -> None:
        """
        Generates the plot on the given axes.

        Args:
            ax: Matplotlib axes to draw the plot on.
            is_dashboard_context: True if the plot is part of a larger dashboard,
                                  which might influence layout choices (e.g., summary text position).
        """
        pass

    def _get_individual_df(self) -> Optional[pd.DataFrame]:
        # Helper to access data via the visualizer instance
        return self.visualizer.individual_df

    def _get_aggregate_df(self) -> Optional[pd.DataFrame]:
        # Helper to access data via the visualizer instance
        return self.visualizer.aggregate_df


# --- Concrete Strategies ---


class SuccessErrorPlotStrategy(PlotStrategy):
    """Strategy for plotting success vs. error rates."""
    def plot(self, ax: plt.Axes, is_dashboard_context: bool = False) -> None:
        individual_df = self._get_individual_df()
        required_cols = ["success_rate", "error_rate"]
        if individual_df is None or not all(col in individual_df.columns for col in required_cols):
            raise ValueError(f"Individual jobs data missing required columns for SuccessErrorPlotStrategy: {required_cols}")

        plot_df = individual_df.copy()
        plot_df.index = [f"Job {i+1}" for i in range(len(plot_df))]
        plot_df[["success_rate", "error_rate"]].plot(
            kind="bar",
            ax=ax,
            color=[self.config.color_palette[0], self.config.color_palette[1]],
            alpha=self.config.alpha,
        )
        ax.set_title("Success vs Error Rates by Job")
        ax.set_xlabel("Jobs")
        ax.set_ylabel("Rate")
        ax.legend(["Success Rate", "Error Rate"])
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=0)


class FidelityPlotStrategy(PlotStrategy):
    """Strategy for plotting fidelity comparisons."""
    def plot(self, ax: plt.Axes, is_dashboard_context: bool = False) -> None:
        individual_df = self._get_individual_df()
        if individual_df is None or "fidelity" not in individual_df.columns:
            raise ValueError("Individual jobs data missing 'fidelity' column for FidelityPlotStrategy")

        plot_df = individual_df.copy()
        plot_df.index = [f"Job {i+1}" for i in range(len(plot_df))]
        plot_df["fidelity"].plot(
            kind="bar",
            ax=ax,
            color=self.config.color_palette[2],
            alpha=self.config.alpha,
        )
        ax.set_title("Fidelity by Job")
        ax.set_xlabel("Jobs")
        ax.set_ylabel("Fidelity")
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=0)


class ShotDistributionPlotStrategy(PlotStrategy):
    """Strategy for plotting shot distributions (successful vs. failed)."""
    def plot(self, ax: plt.Axes, is_dashboard_context: bool = False) -> None:
        individual_df = self._get_individual_df()
        required_cols = ["total_shots", "successful_shots"]
        if individual_df is None or not all(col in individual_df.columns for col in required_cols):
            raise ValueError(f"Individual jobs data missing required columns for ShotDistributionPlotStrategy: {required_cols}")

        plot_df = individual_df.copy()
        plot_df.index = [f"Job {i+1}" for i in range(len(plot_df))]
        shot_data = pd.DataFrame({
            "Successful Shots": plot_df["successful_shots"],
            "Failed Shots": plot_df["total_shots"] - plot_df["successful_shots"],
        })
        shot_data.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[self.config.color_palette[0], self.config.color_palette[1]],
            alpha=self.config.alpha,
        )
        ax.set_title("Shot Distribution by Job")
        ax.set_xlabel("Jobs")
        ax.set_ylabel("Number of Shots")
        ax.legend(["Successful Shots", "Failed Shots"])
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=0)
        
        self.visualizer.add_stacked_bar_labels(ax, shot_data) # Use public method
        summary_pos = "bottom_left" if is_dashboard_context else "outside"
        self.visualizer.add_stacked_bar_summary(ax, shot_data, position=summary_pos)


class AggregateSummaryPlotStrategy(PlotStrategy):
    """Strategy for plotting aggregate statistics summary."""
    def plot(self, ax: plt.Axes, is_dashboard_context: bool = False) -> None:
        aggregate_df = self._get_aggregate_df()
        required_cols = [
            "mean_success_rate", "std_success_rate", "min_success_rate",
            "max_success_rate", "fidelity", "error_rate",
        ]
        if aggregate_df is None or not all(col in aggregate_df.columns for col in required_cols):
            raise ValueError(f"Aggregate data missing required columns for AggregateSummaryPlotStrategy: {required_cols}")

        mean_success = aggregate_df["mean_success_rate"].iloc[0] if not aggregate_df["mean_success_rate"].empty else 0
        std_success = aggregate_df["std_success_rate"].iloc[0] if not aggregate_df["std_success_rate"].empty else 0
        min_success = aggregate_df["min_success_rate"].iloc[0] if not aggregate_df["min_success_rate"].empty else 0
        max_success = aggregate_df["max_success_rate"].iloc[0] if not aggregate_df["max_success_rate"].empty else 0
        fidelity = aggregate_df["fidelity"].iloc[0] if not aggregate_df["fidelity"].empty else 0
        error_rate = aggregate_df["error_rate"].iloc[0] if not aggregate_df["error_rate"].empty else 0
        
        aggregate_stats = pd.Series({
            "Mean Success Rate": mean_success,
            "Std Success Rate": std_success,
            "Min Success Rate": min_success,
            "Max Success Rate": max_success,
            "Mean Fidelity": fidelity,
            "Mean Error Rate": error_rate,
        })
        aggregate_stats.plot(
            kind="bar",
            ax=ax,
            color=self.config.color_palette[:len(aggregate_stats)],
            alpha=self.config.alpha,
        )
        ax.set_title("Aggregate Statistics Summary")
        ax.set_xlabel("Statistics")
        ax.set_ylabel("Value")
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45, ha="right")


# --- SuccessRateVisualizer (Context) ---
class SuccessRateVisualizer(BaseVisualizer):
    """Visualizer for SuccessRate metrics using the Strategy pattern for different plots."""

    def __init__(
        self,
        metrics_dict: Dict[str, pd.DataFrame],
        output_dir: str = "img",
        config: Optional[PlotConfig] = None,
    ):
        """
        Initialize the SuccessRate visualizer.

        Args:
            metrics_dict: Dictionary containing SuccessRate metrics.
                          Expected keys: "SuccessRate.individual_jobs", "SuccessRate.aggregate".
            output_dir: Directory to save plots.
            config: Plot configuration settings.
        """
        super().__init__(output_dir, config)
        self.metrics_dict = metrics_dict

        # Fetch data once. Strategies will access these via self.visualizer
        self.individual_df = self.metrics_dict.get("SuccessRate.individual_jobs")
        self.aggregate_df = self.metrics_dict.get("SuccessRate.aggregate")

        if self.individual_df is None:
            raise ValueError("'SuccessRate.individual_jobs' data not found in metrics_dict.")
        if self.aggregate_df is None:
            raise ValueError("'SuccessRate.aggregate' data not found in metrics_dict.")

        # Validate core columns early for individual_df
        core_individual_cols = [
            "success_rate",
            "error_rate",
            "fidelity",
            "total_shots",
            "successful_shots",
        ]
        if not all(col in self.individual_df.columns for col in core_individual_cols):
            missing_cols = [
                col for col in core_individual_cols if col not in self.individual_df.columns
            ]
            raise ValueError(f"Individual jobs data missing core columns: {missing_cols}")

        # Validate core columns early for aggregate_df
        core_aggregate_cols = [
            "mean_success_rate",
            "std_success_rate",
            "min_success_rate",
            "max_success_rate",
            "fidelity",
            "error_rate",
        ]
        if not all(col in self.aggregate_df.columns for col in core_aggregate_cols):
            missing_cols = [
                col for col in core_aggregate_cols if col not in self.aggregate_df.columns
            ]
            raise ValueError(f"Aggregate data missing core columns: {missing_cols}")

    def add_stacked_bar_labels(self, ax: plt.Axes, data: pd.DataFrame) -> None:
        """Add value labels on top of and within stacked bars."""
        for i, (_, row) in enumerate(data.iterrows()):
            total_height = row.sum()
            if total_height > 0:
                ax.text(
                    i,
                    total_height + total_height * 0.02,
                    f"{int(total_height)}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
            cumulative_height = 0
            for value in row.values:
                if value > 0:
                    segment_center = cumulative_height + value / 2
                    ax.text(
                        i,
                        segment_center,
                        f"{int(value)}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=("white" if value > total_height * 0.1 else "black"),
                        fontweight="bold",
                    )
                cumulative_height += value

    def add_stacked_bar_summary(self, ax: plt.Axes, data: pd.DataFrame, position: str = "outside") -> None:
        """Add summary information for stacked bar charts."""
        summary_lines = []
        for item_name, row in data.iterrows():  # Changed from job_name to item_name for generality
            total = row.sum()
            summary_lines.append(f"{item_name}: {int(total)} total")
            for col_name, value in row.items():
                percentage = (value / total * 100) if total > 0 else 0
                summary_lines.append(f"  {col_name}: {int(value)} ({percentage:.1f}%)")
        summary_text = "\n".join(summary_lines)

        bbox_props = {"boxstyle": "round,pad=0.3", "alpha": 0.8}
        if position == "outside":
            bbox_props["facecolor"] = "lightgray"
            ax.text(
                1.02,
                0.5,
                summary_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="center",
                bbox=bbox_props,
            )
        else:  # "bottom_left" or "top_right" for dashboard contexts
            bbox_props["facecolor"] = "white"
            bbox_props["alpha"] = 0.9
            if position == "bottom_left":
                ax.text(
                    0.02,
                    0.02,
                    summary_text,
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="bottom",
                    bbox=bbox_props,
                )
            elif position == "top_right":
                ax.text(
                    0.98,
                    0.98,
                    summary_text,
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=bbox_props,
                )

    def _execute_plot_strategy(
        self,
        strategy_class: type[PlotStrategy],
        filename: str,
        save: bool,
        show: bool,
        *,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None
    ) -> plt.Figure:
        """Helper to execute a plotting strategy."""
        is_dashboard_ctx = fig_ax_override is not None
        
        if fig_ax_override:
            fig, ax = fig_ax_override
        else:
            # Adjust figsize for shot distribution if it's a standalone plot and needs more space
            current_figsize = self.config.figsize
            if strategy_class == ShotDistributionPlotStrategy and not is_dashboard_ctx:
                current_figsize = (self.config.figsize[0] + 3, self.config.figsize[1])
            fig, ax = plt.subplots(figsize=current_figsize)

        strategy = strategy_class(self)
        strategy.plot(ax, is_dashboard_context=is_dashboard_ctx)

        if not fig_ax_override:  # Only do tight_layout if we created the fig/ax here
            plt.tight_layout()

        if save and not fig_ax_override:  # Don't save if part of a dashboard managed externally
            self.save_plot(fig, filename)
        if show and not fig_ax_override:  # Don't show if part of a dashboard
            self.show_plot(fig)
        return fig

    def create_plot(self) -> plt.Figure:
        """
        Creates the default plot for this visualizer, which is the dashboard.
        This method is required by BaseVisualizer.
        It does not save or show the plot by default when called this way.
        """
        return self.create_dashboard(save=False, show=False)

    def plot_success_error_comparison(self, save: bool = True, show: bool = True) -> plt.Figure:
        """Plots success vs error rates for individual jobs."""
        return self._execute_plot_strategy(
            SuccessErrorPlotStrategy, "success_error_rates", save, show
        )

    def plot_fidelity_comparison(self, save: bool = True, show: bool = True) -> plt.Figure:
        """Plots fidelity comparison across jobs."""
        return self._execute_plot_strategy(FidelityPlotStrategy, "fidelity_comparison", save, show)

    def plot_shot_distribution(self, save: bool = True, show: bool = True) -> plt.Figure:
        """Plots shot distribution (successful vs failed) as stacked bars."""
        return self._execute_plot_strategy(
            ShotDistributionPlotStrategy, "shot_distribution", save, show
        )

    def plot_aggregate_summary(self, save: bool = True, show: bool = True) -> plt.Figure:
        """Plots aggregate statistics summary."""
        return self._execute_plot_strategy(
            AggregateSummaryPlotStrategy, "aggregate_statistics", save, show
        )

    def create_dashboard(self, save: bool = True, show: bool = True) -> plt.Figure:
        """Creates a comprehensive dashboard with all plots in subplots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("SuccessRate Analysis Dashboard", fontsize=16)

        # Instantiate and run each strategy on its designated axes
        SuccessErrorPlotStrategy(self).plot(ax1, is_dashboard_context=True)
        FidelityPlotStrategy(self).plot(ax2, is_dashboard_context=True)
        ShotDistributionPlotStrategy(self).plot(ax3, is_dashboard_context=True)
        AggregateSummaryPlotStrategy(self).plot(ax4, is_dashboard_context=True)

        plt.tight_layout()

        if save:
            self.save_plot(fig, "success_rate_dashboard")
        if show:
            self.show_plot(fig)
        return fig

    def plot_all(self, save: bool = True, show: bool = True) -> List[plt.Figure]:
        """
        Generates all individual plots defined by the strategies.

        Args:
            save: Whether to save the plots.
            show: Whether to display the plots.

        Returns:
            List of matplotlib figures.
        """
        figures = []
        print("Creating SuccessRate visualizations...")

        figures.append(self.plot_success_error_comparison(save=save, show=show))
        figures.append(self.plot_fidelity_comparison(save=save, show=show))
        figures.append(self.plot_shot_distribution(save=save, show=show))
        figures.append(self.plot_aggregate_summary(save=save, show=show))

        if save:
            print(f"✅ All SuccessRate plots saved to '{self.output_dir}/' directory.")

        return figures
