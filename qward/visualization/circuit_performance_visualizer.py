"""
CircuitPerformance visualization module for QWARD.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .base import BaseVisualizer, PlotConfig


class CircuitPerformanceVisualizer(BaseVisualizer):
    """Visualizer for CircuitPerformance metrics with simplified implementation."""

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

        # Fetch data once - support both old and new key names
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
        if self.aggregate_df is None:
            # For single job cases, aggregate_df might not exist
            print(
                "Note: Aggregate data not found. Some visualizations may be limited to individual job data only."
            )

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

        # Validate core columns early for aggregate_df (if it exists)
        if self.aggregate_df is not None:
            core_aggregate_cols = [
                "mean_success_rate",
                "std_success_rate",
                "min_success_rate",
                "max_success_rate",
                "mean_fidelity",
                "error_rate",
            ]
            if not all(col in self.aggregate_df.columns for col in core_aggregate_cols):
                missing_cols = [
                    col for col in core_aggregate_cols if col not in self.aggregate_df.columns
                ]
                raise ValueError(f"Aggregate data missing core columns: {missing_cols}")

    def create_plot(self) -> plt.Figure:
        """
        Creates the default plot for this visualizer, which is the dashboard.
        This method is required by BaseVisualizer.
        It does not save or show the plot by default when called this way.
        """
        return self.create_dashboard(save=False, show=False)

    def plot_success_error_comparison(
        self,
        save: bool = True,
        show: bool = True,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
    ) -> plt.Figure:
        """Plots success vs error rates for individual jobs."""
        required_cols = ["success_rate", "error_rate"]
        if not all(col in self.individual_df.columns for col in required_cols):
            raise ValueError(
                f"Individual jobs data missing required columns for success/error plot: {required_cols}"
            )

        if fig_ax_override:
            fig, ax = fig_ax_override
        else:
            fig, ax = plt.subplots(figsize=self.config.figsize)

        plot_df = self.individual_df.copy()
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

        if not fig_ax_override:
            plt.tight_layout()

        if save and not fig_ax_override:
            self.save_plot(fig, "success_error_rates")
        if show and not fig_ax_override:
            self.show_plot(fig)
        return fig

    def plot_fidelity_comparison(
        self,
        save: bool = True,
        show: bool = True,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
    ) -> plt.Figure:
        """Plots fidelity comparison across jobs."""
        if "fidelity" not in self.individual_df.columns:
            raise ValueError("Individual jobs data missing 'fidelity' column for fidelity plot")

        if fig_ax_override:
            fig, ax = fig_ax_override
        else:
            fig, ax = plt.subplots(figsize=self.config.figsize)

        plot_df = self.individual_df.copy()
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

        if not fig_ax_override:
            plt.tight_layout()

        if save and not fig_ax_override:
            self.save_plot(fig, "fidelity_comparison")
        if show and not fig_ax_override:
            self.show_plot(fig)
        return fig

    def plot_shot_distribution(
        self,
        save: bool = True,
        show: bool = True,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
    ) -> plt.Figure:
        """Plots shot distribution (successful vs failed) as stacked bars."""
        required_cols = ["total_shots", "successful_shots"]
        if not all(col in self.individual_df.columns for col in required_cols):
            raise ValueError(
                f"Individual jobs data missing required columns for shot distribution plot: {required_cols}"
            )

        if fig_ax_override:
            fig, ax = fig_ax_override
        else:
            # Adjust figsize for shot distribution if it's a standalone plot and needs more space
            current_figsize = self.config.figsize
            if not fig_ax_override:
                current_figsize = (self.config.figsize[0] + 3, self.config.figsize[1])
            fig, ax = plt.subplots(figsize=current_figsize)

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

        self._add_stacked_bar_labels(ax, shot_data)
        summary_pos = "bottom_left" if fig_ax_override is not None else "outside"
        self._add_stacked_bar_summary(ax, shot_data, position=summary_pos)

        if not fig_ax_override:
            plt.tight_layout()

        if save and not fig_ax_override:
            self.save_plot(fig, "shot_distribution")
        if show and not fig_ax_override:
            self.show_plot(fig)
        return fig

    def plot_aggregate_summary(
        self,
        save: bool = True,
        show: bool = True,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
    ) -> plt.Figure:
        """Plots aggregate statistics summary."""
        if self.aggregate_df is None:
            # Create aggregate summary from individual jobs data
            if "success_rate" in self.individual_df.columns:
                mean_success = self.individual_df["success_rate"].mean()
                std_success = (
                    self.individual_df["success_rate"].std() if len(self.individual_df) > 1 else 0
                )
                min_success = self.individual_df["success_rate"].min()
                max_success = self.individual_df["success_rate"].max()
            else:
                mean_success = std_success = min_success = max_success = 0

            if "fidelity" in self.individual_df.columns:
                fidelity = self.individual_df["fidelity"].mean()
            else:
                fidelity = 0

            if "error_rate" in self.individual_df.columns:
                error_rate = self.individual_df["error_rate"].mean()
            else:
                error_rate = 1.0 - mean_success
        else:
            # Use existing aggregate data
            required_cols = [
                "mean_success_rate",
                "std_success_rate",
                "min_success_rate",
                "max_success_rate",
                "mean_fidelity",
                "error_rate",
            ]
            if not all(col in self.aggregate_df.columns for col in required_cols):
                raise ValueError(
                    f"Aggregate data missing required columns for aggregate summary plot: {required_cols}"
                )

            mean_success = (
                self.aggregate_df["mean_success_rate"].iloc[0]
                if not self.aggregate_df["mean_success_rate"].empty
                else 0
            )
            std_success = (
                self.aggregate_df["std_success_rate"].iloc[0]
                if not self.aggregate_df["std_success_rate"].empty
                else 0
            )
            min_success = (
                self.aggregate_df["min_success_rate"].iloc[0]
                if not self.aggregate_df["min_success_rate"].empty
                else 0
            )
            max_success = (
                self.aggregate_df["max_success_rate"].iloc[0]
                if not self.aggregate_df["max_success_rate"].empty
                else 0
            )
            fidelity = (
                self.aggregate_df["mean_fidelity"].iloc[0]
                if not self.aggregate_df["mean_fidelity"].empty
                else 0
            )
            error_rate = (
                self.aggregate_df["error_rate"].iloc[0]
                if not self.aggregate_df["error_rate"].empty
                else 0
            )

        if fig_ax_override:
            fig, ax = fig_ax_override
        else:
            fig, ax = plt.subplots(figsize=self.config.figsize)

        aggregate_stats = pd.Series(
            {
                "Mean Success Rate": mean_success,
                "Std Success Rate": std_success,
                "Min Success Rate": min_success,
                "Max Success Rate": max_success,
                "Mean Fidelity": fidelity,
                "Mean Error Rate": error_rate,
            }
        )
        aggregate_stats.plot(
            kind="bar",
            ax=ax,
            color=self.config.color_palette[: len(aggregate_stats)],
            alpha=self.config.alpha,
        )
        ax.set_title("Aggregate Statistics Summary")
        ax.set_xlabel("Statistics")
        ax.set_ylabel("Value")
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

        if not fig_ax_override:
            plt.tight_layout()

        if save and not fig_ax_override:
            self.save_plot(fig, "aggregate_statistics")
        if show and not fig_ax_override:
            self.show_plot(fig)
        return fig

    def create_dashboard(self, save: bool = True, show: bool = True) -> plt.Figure:
        """Creates a comprehensive dashboard with all plots in subplots."""
        if self.aggregate_df is not None:
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

    def plot_all(self, save: bool = True, show: bool = True) -> List[plt.Figure]:
        """
        Generates all individual plots.

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

    def _add_stacked_bar_labels(self, ax: plt.Axes, data: pd.DataFrame) -> None:
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

    def _add_stacked_bar_summary(
        self, ax: plt.Axes, data: pd.DataFrame, position: str = "outside"
    ) -> None:
        """Add summary information for stacked bar charts."""
        summary_lines = []
        for item_name, row in data.iterrows():
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
