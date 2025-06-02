"""
Complexity visualization strategy for QWARD.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .base import VisualizationStrategy, PlotConfig, PlotMetadata, PlotType, PlotRegistry
from .constants import Plots


class ComplexityVisualizer(VisualizationStrategy):
    """Visualization strategy for ComplexityMetrics with comprehensive complexity analysis."""

    # Class-level plot registry
    PLOT_REGISTRY: PlotRegistry = {
        Plots.Complexity.GATE_BASED_METRICS: PlotMetadata(
            name=Plots.Complexity.GATE_BASED_METRICS,
            method_name="plot_gate_based_metrics",
            description="Visualizes gate-based complexity metrics including gate counts and distributions",
            plot_type=PlotType.BAR_CHART,
            filename="gate_based_metrics",
            dependencies=[
                "gate_based_metrics.gate_count",
                "gate_based_metrics.cnot_count",
                "gate_based_metrics.single_qubit_gate_count",
                "gate_based_metrics.multi_qubit_gate_count",
            ],
            category="Gate Analysis",
        ),
        Plots.Complexity.COMPLEXITY_RADAR: PlotMetadata(
            name=Plots.Complexity.COMPLEXITY_RADAR,
            method_name="plot_complexity_radar",
            description="Radar chart showing multiple complexity dimensions for comprehensive analysis",
            plot_type=PlotType.RADAR_CHART,
            filename="complexity_radar",
            dependencies=[
                "advanced_metrics.parallelism_factor",
                "derived_metrics.weighted_complexity",
                "derived_metrics.efficiency_ratio",
            ],
            category="Comprehensive Analysis",
        ),
        Plots.Complexity.EFFICIENCY_METRICS: PlotMetadata(
            name=Plots.Complexity.EFFICIENCY_METRICS,
            method_name="plot_efficiency_metrics",
            description="Shows efficiency-related metrics and derived complexity measures",
            plot_type=PlotType.BAR_CHART,
            filename="efficiency_metrics",
            dependencies=[
                "derived_metrics.efficiency_ratio",
                "derived_metrics.weighted_complexity",
            ],
            category="Efficiency Analysis",
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
        Initialize the Complexity visualization strategy.

        Args:
            metrics_dict: Dictionary containing ComplexityMetrics data.
                          Expected key: "ComplexityMetrics".
            output_dir: Directory to save plots.
            config: Plot configuration settings.
        """
        super().__init__(metrics_dict, output_dir, config)

        # Fetch data
        self.complexity_df = self.metrics_dict.get("ComplexityMetrics")
        if self.complexity_df is None:
            raise ValueError("'ComplexityMetrics' data not found in metrics_dict.")

        if self.complexity_df.empty:
            raise ValueError("ComplexityMetrics DataFrame is empty.")

        # Validate core columns
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate that required columns exist in the data."""
        required_cols = [
            "gate_based_metrics.gate_count",
            "gate_based_metrics.circuit_depth",
            "standardized_metrics.circuit_volume",
            "standardized_metrics.gate_density",
        ]

        self._validate_required_columns(self.complexity_df, required_cols, "ComplexityMetrics data")

    def plot_gate_based_metrics(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
    ) -> plt.Figure:
        """Plot gate-based complexity metrics."""
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        # Extract gate-based metrics using base class utility
        gate_cols = [
            "gate_based_metrics.gate_count",
            "gate_based_metrics.circuit_depth",
            "gate_based_metrics.t_count",
            "gate_based_metrics.cnot_count",
            "gate_based_metrics.two_qubit_count",
        ]

        gate_data = self._extract_metrics_from_columns(
            self.complexity_df, gate_cols, prefix_to_remove="gate_based_metrics."
        )

        if not gate_data:
            self._show_no_data_message(ax, "Gate-Based Metrics", "No gate-based metrics available")
            return self._finalize_plot(
                fig=fig,
                is_override=is_override,
                save=save,
                show=show,
                filename="complexity_gate_based_metrics",
            )

        self._create_bar_plot_with_labels(
            data=gate_data,
            ax=ax,
            title="Gate-Based Complexity Metrics",
            xlabel="Metrics",
            ylabel="Count",
            value_format="int",
        )

        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="complexity_gate_based_metrics",
        )

    def plot_complexity_radar(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
    ) -> plt.Figure:
        """Plot complexity metrics as a radar chart."""
        if fig_ax_override:
            fig, ax = fig_ax_override
            is_override = True
        else:
            fig, ax = plt.subplots(figsize=self.config.figsize, subplot_kw={"projection": "polar"})
            is_override = False

        # Extract normalized complexity metrics (0-1 scale)
        metrics_data = {}

        # Get multi-qubit ratio (already normalized)
        if "gate_based_metrics.multi_qubit_ratio" in self.complexity_df.columns:
            metrics_data["Multi-Qubit\nRatio"] = self.complexity_df[
                "gate_based_metrics.multi_qubit_ratio"
            ].iloc[0]

        # Get entangling gate density (already normalized)
        if "entanglement_metrics.entangling_gate_density" in self.complexity_df.columns:
            metrics_data["Entangling\nDensity"] = self.complexity_df[
                "entanglement_metrics.entangling_gate_density"
            ].iloc[0]

        # Get gate density (normalize by dividing by 1.0 - assuming max density is 1)
        if "standardized_metrics.gate_density" in self.complexity_df.columns:
            gate_density = self.complexity_df["standardized_metrics.gate_density"].iloc[0]
            metrics_data["Gate\nDensity"] = min(gate_density, 1.0)  # Cap at 1.0

        # Get parallelism efficiency
        if "advanced_metrics.parallelism_efficiency" in self.complexity_df.columns:
            metrics_data["Parallelism\nEfficiency"] = self.complexity_df[
                "advanced_metrics.parallelism_efficiency"
            ].iloc[0]

        # Get circuit efficiency
        if "advanced_metrics.circuit_efficiency" in self.complexity_df.columns:
            metrics_data["Circuit\nEfficiency"] = self.complexity_df[
                "advanced_metrics.circuit_efficiency"
            ].iloc[0]

        # Get square ratio
        if "derived_metrics.square_ratio" in self.complexity_df.columns:
            metrics_data["Square\nRatio"] = self.complexity_df["derived_metrics.square_ratio"].iloc[
                0
            ]

        if len(metrics_data) < 3:
            self._show_no_data_message(
                ax,
                "Complexity Radar Chart",
                "Insufficient data for radar chart\n(need at least 3 metrics)",
            )
            return self._finalize_plot(
                fig=fig,
                is_override=is_override,
                save=save,
                show=show,
                filename="complexity_radar_chart",
            )

        # Prepare data for radar chart
        categories = list(metrics_data.keys())
        values = list(metrics_data.values())

        # Number of variables
        num_vars = len(categories)

        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]  # Complete the circle

        # Add first value at the end to close the polygon
        values += values[:1]

        # Plot
        ax.plot(angles, values, "o-", linewidth=2, color=self.config.color_palette[0])
        ax.fill(angles, values, alpha=self.config.alpha, color=self.config.color_palette[0])

        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title("Complexity Metrics Radar Chart", pad=20)

        # Add grid
        ax.grid(True)

        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="complexity_radar_chart",
        )

    def plot_efficiency_metrics(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
    ) -> plt.Figure:
        """Plot circuit efficiency and utilization metrics."""
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        # Extract efficiency metrics using base class utility
        efficiency_cols = [
            "standardized_metrics.gate_density",
            "advanced_metrics.parallelism_factor",
            "advanced_metrics.parallelism_efficiency",
            "advanced_metrics.circuit_efficiency",
            "advanced_metrics.quantum_resource_utilization",
        ]

        # Use custom formatting for display names since these span multiple categories
        efficiency_data = {}
        for col in efficiency_cols:
            if col in self.complexity_df.columns:
                metric_name = col.rsplit(".", maxsplit=1)[-1].replace("_", " ").title()
                efficiency_data[metric_name] = self.complexity_df[col].iloc[0]

        if not efficiency_data:
            self._show_no_data_message(ax, "Efficiency Metrics", "No efficiency metrics available")
            return self._finalize_plot(
                fig=fig,
                is_override=is_override,
                save=save,
                show=show,
                filename="complexity_efficiency_metrics",
            )

        self._create_bar_plot_with_labels(
            data=efficiency_data,
            ax=ax,
            title="Circuit Efficiency Metrics",
            xlabel="Metrics",
            ylabel="Value",
            value_format="{:.3f}",
        )

        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="complexity_efficiency_metrics",
        )

    def create_dashboard(self, save: bool = False, show: bool = False) -> plt.Figure:
        """Creates a comprehensive dashboard with all ComplexityMetrics plots."""
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle("ComplexityMetrics Analysis Dashboard", fontsize=16)

        # Create a 2x2 grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], projection="polar")  # Radar chart
        ax3 = fig.add_subplot(gs[1, 0])  # Efficiency metrics
        ax4 = fig.add_subplot(gs[1, 1])  # Summary

        # Create each plot on its designated axes
        self.plot_gate_based_metrics(save=False, show=False, fig_ax_override=(fig, ax1))
        self.plot_complexity_radar(save=False, show=False, fig_ax_override=(fig, ax2))
        self.plot_efficiency_metrics(save=False, show=False, fig_ax_override=(fig, ax3))

        # Add a summary text box in the last subplot
        ax4.axis("off")
        summary_text = self._generate_summary_text()
        ax4.text(
            0.1,
            0.9,
            summary_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgray", "alpha": 0.8},
        )

        if save:
            self.save_plot(fig, "complexity_metrics_dashboard")
        if show:
            self.show_plot(fig)
        return fig

    def _generate_summary_text(self) -> str:
        """Generate summary text for the dashboard."""
        try:
            gate_count = self.complexity_df["gate_based_metrics.gate_count"].iloc[0]
            depth = self.complexity_df["gate_based_metrics.circuit_depth"].iloc[0]

            # Get efficiency if available
            efficiency = self.complexity_df.get(
                "advanced_metrics.circuit_efficiency", pd.Series([0])
            ).iloc[0]

            # Get parallelism factor if available
            parallelism = self.complexity_df.get(
                "advanced_metrics.parallelism_factor", pd.Series([0])
            ).iloc[0]

            summary = f"""Circuit Complexity Summary
            
Gate Count: {gate_count}
Circuit Depth: {depth}
Parallelism Factor: {parallelism:.2f}
Circuit Efficiency: {efficiency:.3f}

This dashboard shows various
complexity metrics that help
understand the computational
requirements and efficiency
of the quantum circuit."""

        except Exception:
            summary = """Circuit Complexity Summary
            
Unable to generate detailed
summary due to missing data.
Please check that all required
complexity metrics are available."""

        return summary
