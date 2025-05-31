"""
CircuitPerformance visualization module for QWARD.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .base import BaseVisualizer, PlotConfig


class CircuitPerformanceVisualizer(BaseVisualizer):
    """Visualizer for CircuitPerformance metrics with performance analysis."""

    def __init__(
        self,
        metrics_dict: Dict[str, pd.DataFrame],
        output_dir: str = "img",
        config: Optional[PlotConfig] = None,
    ):
        """
        Initialize the CircuitPerformance visualizer.

        Args:
            metrics_dict: Dictionary containing CircuitPerformance metrics.
                          Expected keys: "CircuitPerformance.individual_jobs", "CircuitPerformance.aggregate".
            output_dir: Directory to save plots.
            config: Plot configuration settings.
        """
        super().__init__(output_dir, config)
        self.metrics_dict = metrics_dict

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
        required_individual_cols = [
            "success_rate",
            "error_rate",
            "fidelity",
            "total_shots",
            "successful_shots",
        ]

        self._validate_required_columns(
            self.individual_df, required_individual_cols, "CircuitPerformance individual jobs data"
        )

        # Validate aggregate data if it exists
        if self.aggregate_df is not None and not self.aggregate_df.empty:
            required_aggregate_cols = [
                "mean_success_rate",
                "std_success_rate",
                "min_success_rate",
                "max_success_rate",
                "mean_fidelity",
                "error_rate",
            ]

            self._validate_required_columns(
                self.aggregate_df, required_aggregate_cols, "CircuitPerformance aggregate data"
            )

    def create_plot(self) -> plt.Figure:
        """
        Creates the default plot for this visualizer, which is the dashboard.
        This method is required by BaseVisualizer.
        """
        return self.create_dashboard(save=False, show=False)

    def plot_success_error_comparison(
        self,
        save: bool = True,
        show: bool = True,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
    ) -> plt.Figure:
        """Plot success vs error rates for individual jobs."""
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        # Prepare data with job labels
        plot_df = self.individual_df.copy()
        plot_df.index = [f"Job {i+1}" for i in range(len(plot_df))]

        # Create grouped bar chart
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

        return self._finalize_plot(
            fig=fig, is_override=is_override, save=save, show=show, filename="success_error_rates"
        )

    def plot_fidelity_comparison(
        self,
        save: bool = True,
        show: bool = True,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
    ) -> plt.Figure:
        """Plot fidelity comparison across jobs."""
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        # Extract fidelity data
        fidelity_data = {}
        for i, fidelity in enumerate(self.individual_df["fidelity"]):
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
        save: bool = True,
        show: bool = True,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
    ) -> plt.Figure:
        """Plot shot distribution (successful vs failed) as stacked bars."""
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        # Prepare shot distribution data
        plot_df = self.individual_df.copy()
        plot_df.index = [f"Job {i+1}" for i in range(len(plot_df))]

        shot_data = pd.DataFrame(
            {
                "Successful Shots": plot_df["successful_shots"],
                "Failed Shots": plot_df["total_shots"] - plot_df["successful_shots"],
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
        save: bool = True,
        show: bool = True,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
    ) -> plt.Figure:
        """Plot aggregate statistics summary."""
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        # Get aggregate data (compute if not available)
        if self.aggregate_df is not None and not self.aggregate_df.empty:
            # Use existing aggregate data
            aggregate_data = {
                "Mean Success Rate": self.aggregate_df["mean_success_rate"].iloc[0],
                "Std Success Rate": self.aggregate_df["std_success_rate"].iloc[0],
                "Min Success Rate": self.aggregate_df["min_success_rate"].iloc[0],
                "Max Success Rate": self.aggregate_df["max_success_rate"].iloc[0],
                "Mean Fidelity": self.aggregate_df["mean_fidelity"].iloc[0],
                "Mean Error Rate": self.aggregate_df["error_rate"].iloc[0],
            }
        else:
            # Compute aggregate from individual jobs
            aggregate_data = {
                "Mean Success Rate": self.individual_df["success_rate"].mean(),
                "Std Success Rate": (
                    self.individual_df["success_rate"].std() if len(self.individual_df) > 1 else 0
                ),
                "Min Success Rate": self.individual_df["success_rate"].min(),
                "Max Success Rate": self.individual_df["success_rate"].max(),
                "Mean Fidelity": self.individual_df["fidelity"].mean(),
                "Mean Error Rate": self.individual_df["error_rate"].mean(),
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

    def create_dashboard(self, save: bool = True, show: bool = True) -> plt.Figure:
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

    def plot_all(self, save: bool = True, show: bool = True) -> List[plt.Figure]:
        """
        Generate all individual plots.

        Args:
            save: Whether to save the plots.
            show: Whether to display the plots.

        Returns:
            List of matplotlib figures.
        """
        figures = []
        print("Creating CircuitPerformance visualizations...")

        figures.append(self.plot_success_error_comparison(save=save, show=show))
        figures.append(self.plot_fidelity_comparison(save=save, show=show))
        figures.append(self.plot_shot_distribution(save=save, show=show))
        figures.append(self.plot_aggregate_summary(save=save, show=show))

        if save:
            print(f"✅ All CircuitPerformance plots saved to '{self.output_dir}/' directory.")

        return figures
