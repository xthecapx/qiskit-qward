"""
CircuitPerformance visualization strategy for QWARD.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .base import VisualizationStrategy, PlotConfig, PlotMetadata, PlotType, PlotRegistry
from .constants import Plots


class CircuitPerformanceVisualizer(VisualizationStrategy):
    """Visualization strategy for CircuitPerformance metrics with performance analysis."""

    # Class-level plot registry
    PLOT_REGISTRY: PlotRegistry = {
        Plots.CircuitPerformance.SUCCESS_ERROR_COMPARISON: PlotMetadata(
            name=Plots.CircuitPerformance.SUCCESS_ERROR_COMPARISON,
            method_name="plot_success_error_comparison",
            description="Compares success rates and error rates across different circuit executions",
            plot_type=PlotType.GROUPED_BAR,
            filename="success_error_comparison",
            dependencies=["success_metrics.success_rate", "success_metrics.error_rate"],
            category="Performance Analysis",
        ),
        Plots.CircuitPerformance.FIDELITY_COMPARISON: PlotMetadata(
            name=Plots.CircuitPerformance.FIDELITY_COMPARISON,
            method_name="plot_fidelity_comparison",
            description="Visualizes fidelity metrics and quantum state quality measures",
            plot_type=PlotType.BAR_CHART,
            filename="fidelity_comparison",
            dependencies=["fidelity_metrics.state_fidelity", "fidelity_metrics.process_fidelity"],
            category="Fidelity Analysis",
        ),
        Plots.CircuitPerformance.SHOT_DISTRIBUTION: PlotMetadata(
            name=Plots.CircuitPerformance.SHOT_DISTRIBUTION,
            method_name="plot_shot_distribution",
            description="Shows the distribution of measurement outcomes across shots",
            plot_type=PlotType.STACKED_BAR,
            filename="shot_distribution",
            dependencies=["execution_metrics.total_shots", "execution_metrics.unique_outcomes"],
            category="Execution Analysis",
        ),
        Plots.CircuitPerformance.AGGREGATE_SUMMARY: PlotMetadata(
            name=Plots.CircuitPerformance.AGGREGATE_SUMMARY,
            method_name="plot_aggregate_summary",
            description="Comprehensive summary of all performance metrics in a dashboard format",
            plot_type=PlotType.BAR_CHART,
            filename="aggregate_summary",
            dependencies=["success_metrics", "fidelity_metrics", "execution_metrics"],
            category="Summary",
        ),
    }

    @classmethod
    def get_available_plots(cls) -> List[str]:
        """Return list of available plot names for this strategy."""
        return list(cls.PLOT_REGISTRY.keys())

    @classmethod
    def get_plot_metadata(cls, plot_name: str) -> PlotMetadata:
        """Get metadata for a specific plot."""
        if plot_name not in cls.PLOT_REGISTRY:
            available = list(cls.PLOT_REGISTRY.keys())
            raise ValueError(f"Plot '{plot_name}' not found. Available plots: {available}")
        return cls.PLOT_REGISTRY[plot_name]

    def __init__(
        self,
        metrics_dict: Dict[str, pd.DataFrame],
        output_dir: str = "img",
        config: Optional[PlotConfig] = None,
    ):
        """
        Initialize the CircuitPerformance visualization strategy.

        Args:
            metrics_dict: Dictionary containing CircuitPerformance metrics.
                          Expected keys: "CircuitPerformance.individual_jobs", "CircuitPerformance.aggregate".
            output_dir: Directory to save plots.
            config: Plot configuration settings.
        """
        super().__init__(metrics_dict, output_dir, config)

        # Fetch data - support both old and new key names
        self.individual_df = self.metrics_dict.get("CircuitPerformance.individual_jobs")
        if self.individual_df is None:
            self.individual_df = self.metrics_dict.get("SuccessRate.individual_jobs")

        self.aggregate_df = self.metrics_dict.get("CircuitPerformance.aggregate")
        if self.aggregate_df is None:
            self.aggregate_df = self.metrics_dict.get("SuccessRate.aggregate")

        if self.individual_df is None:
            raise ValueError(
                "'CircuitPerformance.individual_jobs' or 'SuccessRate.individual_jobs' data not found in metrics_dict."
            )

        if self.individual_df.empty:
            raise ValueError("CircuitPerformance individual jobs DataFrame is empty.")

        # Validate core columns
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate that required columns exist in the data."""
        # Validate individual jobs data
        required_individual_cols = [
            "success_metrics.success_rate",
            "success_metrics.error_rate",
            "fidelity_metrics.fidelity",
            "success_metrics.total_shots",
            "success_metrics.successful_shots",
        ]

        self._validate_required_columns(
            self.individual_df, required_individual_cols, "CircuitPerformance individual jobs data"
        )

        # Validate aggregate data if it exists
        if self.aggregate_df is not None and not self.aggregate_df.empty:
            required_aggregate_cols = [
                "success_metrics.mean_success_rate",
                "success_metrics.std_success_rate",
                "success_metrics.min_success_rate",
                "success_metrics.max_success_rate",
                "fidelity_metrics.mean_fidelity",
                "success_metrics.error_rate",
            ]

            self._validate_required_columns(
                self.aggregate_df, required_aggregate_cols, "CircuitPerformance aggregate data"
            )

    def plot_success_error_comparison(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
    ) -> plt.Figure:
        """Plot success vs error rates for individual jobs."""
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        # Prepare data with job labels
        plot_df = self.individual_df.copy()
        plot_df.index = [f"Job {i+1}" for i in range(len(plot_df))]

        # Create grouped bar chart
        success_error_data = pd.DataFrame(
            {
                "success_rate": plot_df["success_metrics.success_rate"],
                "error_rate": plot_df["success_metrics.error_rate"],
            }
        )
        success_error_data.plot(
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

        return self._finalize_plot(
            fig=fig, is_override=is_override, save=save, show=show, filename="success_error_rates"
        )

    def plot_fidelity_comparison(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
    ) -> plt.Figure:
        """Plot fidelity comparison across jobs."""
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        # Extract fidelity data
        fidelity_data = {}
        for i, fidelity in enumerate(self.individual_df["fidelity_metrics.fidelity"]):
            fidelity_data[f"Job {i+1}"] = fidelity

        self._create_bar_plot_with_labels(
            data=fidelity_data,
            ax=ax,
            title="Fidelity by Job",
            xlabel="Jobs",
            ylabel="Fidelity",
            value_format="{:.3f}",
        )

        return self._finalize_plot(
            fig=fig, is_override=is_override, save=save, show=show, filename="fidelity_comparison"
        )

    def plot_shot_distribution(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
    ) -> plt.Figure:
        """Plot shot distribution (successful vs failed) as stacked bars."""
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        # Prepare shot distribution data
        plot_df = self.individual_df.copy()
        plot_df.index = [f"Job {i+1}" for i in range(len(plot_df))]

        shot_data = pd.DataFrame(
            {
                "Successful Shots": plot_df["success_metrics.successful_shots"],
                "Failed Shots": plot_df["success_metrics.total_shots"]
                - plot_df["success_metrics.successful_shots"],
            }
        )

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

        # Add stacked bar labels
        self._add_stacked_bar_labels(ax, shot_data)

        return self._finalize_plot(
            fig=fig, is_override=is_override, save=save, show=show, filename="shot_distribution"
        )

    def plot_aggregate_summary(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
    ) -> plt.Figure:
        """Plot aggregate statistics summary."""
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        # Get aggregate data (compute if not available)
        if self.aggregate_df is not None and not self.aggregate_df.empty:
            # Use existing aggregate data
            aggregate_data = {
                "Mean Success Rate": self.aggregate_df["success_metrics.mean_success_rate"].iloc[0],
                "Std Success Rate": self.aggregate_df["success_metrics.std_success_rate"].iloc[0],
                "Min Success Rate": self.aggregate_df["success_metrics.min_success_rate"].iloc[0],
                "Max Success Rate": self.aggregate_df["success_metrics.max_success_rate"].iloc[0],
                "Mean Fidelity": self.aggregate_df["fidelity_metrics.mean_fidelity"].iloc[0],
                "Mean Error Rate": self.aggregate_df["success_metrics.error_rate"].iloc[0],
            }
        else:
            # Compute aggregate from individual jobs
            aggregate_data = {
                "Mean Success Rate": self.individual_df["success_metrics.success_rate"].mean(),
                "Std Success Rate": (
                    self.individual_df["success_metrics.success_rate"].std()
                    if len(self.individual_df) > 1
                    else 0
                ),
                "Min Success Rate": self.individual_df["success_metrics.success_rate"].min(),
                "Max Success Rate": self.individual_df["success_metrics.success_rate"].max(),
                "Mean Fidelity": self.individual_df["fidelity_metrics.fidelity"].mean(),
                "Mean Error Rate": self.individual_df["success_metrics.error_rate"].mean(),
            }

        self._create_bar_plot_with_labels(
            data=aggregate_data,
            ax=ax,
            title="Aggregate Statistics Summary",
            xlabel="Statistics",
            ylabel="Value",
            value_format="{:.3f}",
        )

        # Rotate x-axis labels for better readability
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

        return self._finalize_plot(
            fig=fig, is_override=is_override, save=save, show=show, filename="aggregate_statistics"
        )

    def create_dashboard(self, save: bool = False, show: bool = False) -> plt.Figure:
        """Create a comprehensive dashboard with all CircuitPerformance plots."""
        if self.aggregate_df is not None and not self.aggregate_df.empty:
            # Full dashboard with 4 plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle("CircuitPerformance Analysis Dashboard", fontsize=16)

            # Create each plot on its designated axes
            self.plot_success_error_comparison(save=False, show=False, fig_ax_override=(fig, ax1))
            self.plot_fidelity_comparison(save=False, show=False, fig_ax_override=(fig, ax2))
            self.plot_shot_distribution(save=False, show=False, fig_ax_override=(fig, ax3))
            self.plot_aggregate_summary(save=False, show=False, fig_ax_override=(fig, ax4))
        else:
            # Limited dashboard with 3 plots (no separate aggregate plot needed)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle("CircuitPerformance Analysis Dashboard (Single Job)", fontsize=16)

            # Create each plot on its designated axes
            self.plot_success_error_comparison(save=False, show=False, fig_ax_override=(fig, ax1))
            self.plot_fidelity_comparison(save=False, show=False, fig_ax_override=(fig, ax2))
            self.plot_shot_distribution(save=False, show=False, fig_ax_override=(fig, ax3))

        plt.tight_layout()

        if save:
            self.save_plot(fig, "circuit_performance_dashboard")
        if show:
            self.show_plot(fig)
        return fig

    def _add_stacked_bar_labels(self, ax: plt.Axes, data: pd.DataFrame) -> None:
        """Add value labels on stacked bars."""
        for i, (_, row) in enumerate(data.iterrows()):
            total_height = row.sum()
            if total_height > 0:
                # Add total label on top
                ax.text(
                    i,
                    total_height + total_height * 0.02,
                    f"{int(total_height)}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

                # Add labels within each segment
                cumulative_height = 0
                for value in row.values:
                    if value > 0:
                        segment_center = cumulative_height + value / 2
                        # Only show label if segment is large enough
                        if value > total_height * 0.1:
                            ax.text(
                                i,
                                segment_center,
                                f"{int(value)}",
                                ha="center",
                                va="center",
                                fontsize=9,
                                color="white",
                                fontweight="bold",
                            )
                    cumulative_height += value
