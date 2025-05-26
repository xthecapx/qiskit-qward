"""
SuccessRate visualization module for QWARD.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .base import MetricVisualizer, PlotConfig


class SuccessRateVisualizer(MetricVisualizer):
    """Visualizer for SuccessRate metrics."""

    def __init__(
        self,
        metrics_dict: Dict[str, pd.DataFrame],
        output_dir: str = "img",
        config: Optional[PlotConfig] = None,
    ):
        """
        Initialize the SuccessRate visualizer.

        Args:
            metrics_dict: Dictionary containing SuccessRate metrics
            output_dir: Directory to save plots
            config: Plot configuration settings
        """
        super().__init__(metrics_dict, output_dir, config)

        # Validate that we have SuccessRate data
        self.individual_df = self.get_metric_data("SuccessRate.individual_jobs")
        self.aggregate_df = self.get_metric_data("SuccessRate.aggregate")

        if self.individual_df is None:
            raise ValueError("SuccessRate.individual_jobs data not found in metrics_dict")
        if self.aggregate_df is None:
            raise ValueError("SuccessRate.aggregate data not found in metrics_dict")

    def _add_stacked_bar_labels(self, ax: plt.Axes, data: pd.DataFrame) -> None:
        """
        Add value labels on top of stacked bars.

        Args:
            ax: Matplotlib axes
            data: DataFrame with the stacked bar data
        """
        # Get the bar containers from the axes
        containers = ax.containers

        # For each job (x-position)
        for i, (job_name, row) in enumerate(data.iterrows()):
            total_height = row.sum()

            # Add total label on top of the stacked bar
            ax.text(
                i,  # x position (job index)
                total_height + total_height * 0.02,  # y position (slightly above the bar)
                f"{int(total_height)}",  # total value
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

            # Add individual segment labels within each segment
            cumulative_height = 0
            for j, (segment_name, value) in enumerate(row.items()):
                if value > 0:  # Only add labels for non-zero segments
                    segment_center = cumulative_height + value / 2
                    ax.text(
                        i,  # x position
                        segment_center,  # y position (center of segment)
                        f"{int(value)}",  # segment value
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=(
                            "white" if value > total_height * 0.1 else "black"
                        ),  # White text for large segments
                        fontweight="bold",
                    )
                cumulative_height += value

    def create_plot(self) -> plt.Figure:
        """Create the default dashboard plot."""
        return self.create_dashboard()

    def plot_success_error_comparison(self, save: bool = True, show: bool = True) -> plt.Figure:
        """
        Plot success vs error rates for individual jobs.

        Args:
            save: Whether to save the plot
            show: Whether to display the plot

        Returns:
            Matplotlib figure
        """
        # Validate required columns
        required_cols = ["success_rate", "error_rate"]
        if not self.validate_columns(self.individual_df, required_cols):
            raise ValueError(f"Individual jobs data missing required columns: {required_cols}")

        # Prepare data
        plot_df = self.individual_df.copy()
        plot_df.index = [f"Job {i+1}" for i in range(len(plot_df))]

        # Create plot
        fig, ax = plt.subplots(figsize=self.config.figsize)

        plot_df[["success_rate", "error_rate"]].plot(
            kind="bar",
            ax=ax,
            color=[self.config.color_palette[0], self.config.color_palette[1]],
            alpha=self.config.alpha,
            title="Success vs Error Rates by Job",
        )

        ax.set_xlabel("Jobs")
        ax.set_ylabel("Rate")
        ax.legend(["Success Rate", "Error Rate"])
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()

        if save:
            self.save_plot(fig, "success_error_rates")
        if show:
            self.show_plot(fig)

        return fig

    def plot_fidelity_comparison(self, save: bool = True, show: bool = True) -> plt.Figure:
        """
        Plot fidelity comparison across jobs.

        Args:
            save: Whether to save the plot
            show: Whether to display the plot

        Returns:
            Matplotlib figure
        """
        # Validate required columns
        if "fidelity" not in self.individual_df.columns:
            raise ValueError("Individual jobs data missing 'fidelity' column")

        # Prepare data
        plot_df = self.individual_df.copy()
        plot_df.index = [f"Job {i+1}" for i in range(len(plot_df))]

        # Create plot
        fig, ax = plt.subplots(figsize=self.config.figsize)

        plot_df["fidelity"].plot(
            kind="bar",
            ax=ax,
            color=self.config.color_palette[2],
            alpha=self.config.alpha,
            title="Fidelity by Job",
        )

        ax.set_xlabel("Jobs")
        ax.set_ylabel("Fidelity")
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()

        if save:
            self.save_plot(fig, "fidelity_comparison")
        if show:
            self.show_plot(fig)

        return fig

    def plot_shot_distribution(self, save: bool = True, show: bool = True) -> plt.Figure:
        """
        Plot shot distribution (successful vs failed) as stacked bars.

        Args:
            save: Whether to save the plot
            show: Whether to display the plot

        Returns:
            Matplotlib figure
        """
        # Validate required columns
        required_cols = ["total_shots", "successful_shots"]
        if not self.validate_columns(self.individual_df, required_cols):
            raise ValueError(f"Individual jobs data missing required columns: {required_cols}")

        # Prepare data
        plot_df = self.individual_df.copy()
        plot_df.index = [f"Job {i+1}" for i in range(len(plot_df))]

        shot_data = pd.DataFrame(
            {
                "Successful Shots": plot_df["successful_shots"],
                "Failed Shots": plot_df["total_shots"] - plot_df["successful_shots"],
            }
        )

        # Create plot with extra space for summary
        fig, ax = plt.subplots(figsize=(self.config.figsize[0] + 3, self.config.figsize[1]))

        bars = shot_data.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[self.config.color_palette[0], self.config.color_palette[1]],
            alpha=self.config.alpha,
            title="Shot Distribution by Job",
        )

        ax.set_xlabel("Jobs")
        ax.set_ylabel("Number of Shots")
        ax.legend(["Successful Shots", "Failed Shots"])
        if self.config.grid:
            ax.grid(True, alpha=0.3)

        # Add value labels on top of each stacked bar
        self._add_stacked_bar_labels(ax, shot_data)

        # Add summary information outside the plot
        self.add_stacked_bar_summary(ax, shot_data, position="outside")

        plt.xticks(rotation=0)
        plt.tight_layout()

        if save:
            self.save_plot(fig, "shot_distribution")
        if show:
            self.show_plot(fig)

        return fig

    def plot_aggregate_summary(self, save: bool = True, show: bool = True) -> plt.Figure:
        """
        Plot aggregate statistics summary.

        Args:
            save: Whether to save the plot
            show: Whether to display the plot

        Returns:
            Matplotlib figure
        """
        # Validate required columns
        required_cols = [
            "mean_success_rate",
            "std_success_rate",
            "min_success_rate",
            "max_success_rate",
            "fidelity",
            "error_rate",
        ]
        if not self.validate_columns(self.aggregate_df, required_cols):
            raise ValueError(f"Aggregate data missing required columns: {required_cols}")

        # Prepare data
        aggregate_stats = pd.Series(
            {
                "Mean Success Rate": self.aggregate_df["mean_success_rate"].iloc[0],
                "Std Success Rate": self.aggregate_df["std_success_rate"].iloc[0],
                "Min Success Rate": self.aggregate_df["min_success_rate"].iloc[0],
                "Max Success Rate": self.aggregate_df["max_success_rate"].iloc[0],
                "Mean Fidelity": self.aggregate_df["fidelity"].iloc[0],
                "Mean Error Rate": self.aggregate_df["error_rate"].iloc[0],
            }
        )

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        aggregate_stats.plot(
            kind="bar",
            ax=ax,
            color=self.config.color_palette[: len(aggregate_stats)],
            alpha=self.config.alpha,
            title="Aggregate Statistics Summary",
        )

        ax.set_xlabel("Statistics")
        ax.set_ylabel("Value")
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save:
            self.save_plot(fig, "aggregate_statistics")
        if show:
            self.show_plot(fig)

        return fig

    def create_dashboard(self, save: bool = True, show: bool = True) -> plt.Figure:
        """
        Create a comprehensive dashboard with all plots in subplots.

        Args:
            save: Whether to save the plot
            show: Whether to display the plot

        Returns:
            Matplotlib figure
        """
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("SuccessRate Analysis Dashboard", fontsize=16)

        # Prepare data
        plot_df = self.individual_df.copy()
        plot_df.index = [f"Job {i+1}" for i in range(len(plot_df))]

        # Plot 1: Success vs Error Rates
        plot_df[["success_rate", "error_rate"]].plot(
            kind="bar",
            ax=ax1,
            color=[self.config.color_palette[0], self.config.color_palette[1]],
            alpha=self.config.alpha,
        )
        ax1.set_title("Success vs Error Rates by Job")
        ax1.set_xlabel("Jobs")
        ax1.set_ylabel("Rate")
        ax1.legend(["Success Rate", "Error Rate"])
        if self.config.grid:
            ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis="x", rotation=0)

        # Plot 2: Fidelity
        plot_df["fidelity"].plot(
            kind="bar", ax=ax2, color=self.config.color_palette[2], alpha=self.config.alpha
        )
        ax2.set_title("Fidelity by Job")
        ax2.set_xlabel("Jobs")
        ax2.set_ylabel("Fidelity")
        if self.config.grid:
            ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis="x", rotation=0)

        # Plot 3: Shot Distribution
        shot_data = pd.DataFrame(
            {
                "Successful": plot_df["successful_shots"],
                "Failed": plot_df["total_shots"] - plot_df["successful_shots"],
            }
        )
        shot_data.plot(
            kind="bar",
            stacked=True,
            ax=ax3,
            color=[self.config.color_palette[0], self.config.color_palette[1]],
            alpha=self.config.alpha,
        )
        ax3.set_title("Shot Distribution by Job")
        ax3.set_xlabel("Jobs")
        ax3.set_ylabel("Number of Shots")
        ax3.legend(["Successful", "Failed"])
        if self.config.grid:
            ax3.grid(True, alpha=0.3)

        # Add value labels on top of each stacked bar
        self._add_stacked_bar_labels(ax3, shot_data)

        # Add summary information in bottom left for dashboard view
        self.add_stacked_bar_summary(ax3, shot_data, position="bottom_left")

        ax3.tick_params(axis="x", rotation=0)

        # Plot 4: Aggregate Statistics
        aggregate_stats = pd.Series(
            {
                "Mean Success": self.aggregate_df["mean_success_rate"].iloc[0],
                "Std Success": self.aggregate_df["std_success_rate"].iloc[0],
                "Min Success": self.aggregate_df["min_success_rate"].iloc[0],
                "Max Success": self.aggregate_df["max_success_rate"].iloc[0],
                "Mean Fidelity": self.aggregate_df["fidelity"].iloc[0],
                "Mean Error": self.aggregate_df["error_rate"].iloc[0],
            }
        )

        aggregate_stats.plot(
            kind="bar",
            ax=ax4,
            color=self.config.color_palette[: len(aggregate_stats)],
            alpha=self.config.alpha,
        )
        ax4.set_title("Aggregate Statistics")
        ax4.set_xlabel("Statistics")
        ax4.set_ylabel("Value")
        if self.config.grid:
            ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save:
            self.save_plot(fig, "success_rate_dashboard")
        if show:
            self.show_plot(fig)

        return fig

    def plot_all(self, save: bool = True, show: bool = True) -> List[plt.Figure]:
        """
        Generate all individual plots.

        Args:
            save: Whether to save the plots
            show: Whether to display the plots

        Returns:
            List of matplotlib figures
        """
        figures = []

        print("Creating SuccessRate visualizations...")

        figures.append(self.plot_success_error_comparison(save=save, show=show))
        figures.append(self.plot_fidelity_comparison(save=save, show=show))
        figures.append(self.plot_shot_distribution(save=save, show=show))
        figures.append(self.plot_aggregate_summary(save=save, show=show))

        if save:
            print(f"✅ All SuccessRate plots saved to '{self.output_dir}/' directory")

        return figures
