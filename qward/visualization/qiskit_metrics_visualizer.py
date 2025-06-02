"""
Qiskit visualization strategy for QWARD.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .base import VisualizationStrategy, PlotConfig, PlotMetadata, PlotType, PlotRegistry
from .constants import Plots


class QiskitVisualizer(VisualizationStrategy):
    """Visualization strategy for QiskitMetrics with circuit structure and instruction analysis."""

    # Class-level plot registry
    PLOT_REGISTRY: PlotRegistry = {
        Plots.Qiskit.CIRCUIT_STRUCTURE: PlotMetadata(
            name=Plots.Qiskit.CIRCUIT_STRUCTURE,
            method_name="plot_circuit_structure",
            description="Visualizes the structure of quantum circuits including depth, width, and gate composition",
            plot_type=PlotType.BAR_CHART,
            filename="circuit_structure",
            dependencies=["basic_metrics.depth", "basic_metrics.num_qubits"],
            category="Structure Analysis",
        ),
        Plots.Qiskit.GATE_DISTRIBUTION: PlotMetadata(
            name=Plots.Qiskit.GATE_DISTRIBUTION,
            method_name="plot_gate_distribution",
            description="Shows the distribution of different gate types used in the circuit",
            plot_type=PlotType.PIE_CHART,
            filename="gate_distribution",
            dependencies=["gate_metrics.gate_types"],
            category="Gate Analysis",
        ),
        Plots.Qiskit.INSTRUCTION_METRICS: PlotMetadata(
            name=Plots.Qiskit.INSTRUCTION_METRICS,
            method_name="plot_instruction_metrics",
            description="Displays detailed instruction-level metrics and statistics",
            plot_type=PlotType.BAR_CHART,
            filename="instruction_metrics",
            dependencies=[
                "instruction_metrics.instruction_count",
                "instruction_metrics.instruction_types",
            ],
            category="Instruction Analysis",
        ),
        Plots.Qiskit.CIRCUIT_SUMMARY: PlotMetadata(
            name=Plots.Qiskit.CIRCUIT_SUMMARY,
            method_name="plot_circuit_summary",
            description="Comprehensive overview combining multiple circuit metrics",
            plot_type=PlotType.BAR_CHART,
            filename="circuit_summary",
            dependencies=["basic_metrics", "gate_metrics", "instruction_metrics"],
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
        Initialize the Qiskit visualization strategy.

        Args:
            metrics_dict: Dictionary containing QiskitMetrics data.
                          Expected key: "QiskitMetrics".
            output_dir: Directory to save plots.
            config: Plot configuration settings.
        """
        super().__init__(metrics_dict, output_dir, config)

        # Fetch data
        self.qiskit_df = self.metrics_dict.get("QiskitMetrics")
        if self.qiskit_df is None:
            raise ValueError("'QiskitMetrics' data not found in metrics_dict.")

        if self.qiskit_df.empty:
            raise ValueError("QiskitMetrics DataFrame is empty.")

        # Validate core columns
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate that required columns exist in the data."""
        required_basic_cols = [
            "basic_metrics.depth",
            "basic_metrics.width",
            "basic_metrics.size",
            "basic_metrics.num_qubits",
            "basic_metrics.num_clbits",
        ]

        self._validate_required_columns(self.qiskit_df, required_basic_cols, "QiskitMetrics data")

    def plot_circuit_structure(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
    ) -> plt.Figure:
        """Plot basic circuit structure metrics."""
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        # Get basic structure metrics
        structure_data = {
            "Depth": self.qiskit_df["basic_metrics.depth"].iloc[0],
            "Width": self.qiskit_df["basic_metrics.width"].iloc[0],
            "Size": self.qiskit_df["basic_metrics.size"].iloc[0],
            "Qubits": self.qiskit_df["basic_metrics.num_qubits"].iloc[0],
            "Classical Bits": self.qiskit_df["basic_metrics.num_clbits"].iloc[0],
        }

        self._create_bar_plot_with_labels(
            data=structure_data,
            ax=ax,
            title="Circuit Structure",
            xlabel="Metrics",
            ylabel="Count",
            value_format="int",
        )

        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="qiskit_circuit_structure",
        )

    def plot_gate_distribution(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
    ) -> plt.Figure:
        """Plot distribution of gate types."""
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        # Extract gate counts from count_ops columns
        gate_counts = {}
        for col in self.qiskit_df.columns:
            if col.startswith("basic_metrics.count_ops."):
                gate_name = col.replace("basic_metrics.count_ops.", "")
                gate_counts[gate_name] = self.qiskit_df[col].iloc[0]

        if not gate_counts:
            self._show_no_data_message(ax, "Gate Distribution", "No gate count data available")
            return self._finalize_plot(
                fig=fig,
                is_override=is_override,
                save=save,
                show=show,
                filename="qiskit_gate_distribution",
            )

        gate_series = pd.Series(gate_counts)

        # Create pie chart for gate distribution
        pie_result = ax.pie(
            gate_series.values,
            labels=gate_series.index,
            autopct="%1.1f%%",
            colors=self.config.color_palette[: len(gate_series)],
            startangle=90,
        )

        # Extract autopct text objects if available
        if len(pie_result) > 2:
            autotexts = pie_result[2]
            # Enhance text readability
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")

        ax.set_title("Gate Type Distribution")

        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="qiskit_gate_distribution",
        )

    def plot_instruction_metrics(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
    ) -> plt.Figure:
        """Plot instruction-related metrics."""
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        # Get instruction metrics using the base class utility
        instruction_cols = [
            "instruction_metrics.num_connected_components",
            "instruction_metrics.num_nonlocal_gates",
            "instruction_metrics.num_tensor_factors",
            "instruction_metrics.num_unitary_factors",
        ]

        instruction_data = self._extract_metrics_from_columns(
            self.qiskit_df, instruction_cols, prefix_to_remove="instruction_metrics."
        )

        if not instruction_data:
            self._show_no_data_message(
                ax, "Instruction Metrics", "No instruction metrics available"
            )
            return self._finalize_plot(
                fig=fig,
                is_override=is_override,
                save=save,
                show=show,
                filename="qiskit_instruction_metrics",
            )

        self._create_bar_plot_with_labels(
            data=instruction_data,
            ax=ax,
            title="Instruction Metrics",
            xlabel="Metrics",
            ylabel="Count",
            value_format="int",
        )

        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="qiskit_instruction_metrics",
        )

    def plot_circuit_summary(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
    ) -> plt.Figure:
        """Plot a summary view of key circuit characteristics."""
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        # Calculate derived metrics
        depth = self.qiskit_df["basic_metrics.depth"].iloc[0]
        width = self.qiskit_df["basic_metrics.width"].iloc[0]
        size = self.qiskit_df["basic_metrics.size"].iloc[0]

        # Circuit density (gates per qubit-time unit)
        circuit_volume = depth * width if depth > 0 and width > 0 else 1
        gate_density = size / circuit_volume

        # Parallelism factor (average gates per time step)
        parallelism = size / depth if depth > 0 else 0

        # Circuit efficiency metrics
        summary_data = {
            "Gate Density": round(gate_density, 3),
            "Parallelism": round(parallelism, 3),
            "Depth/Width Ratio": round(depth / width if width > 0 else 0, 3),
            "Circuit Volume": circuit_volume,
        }

        self._create_bar_plot_with_labels(
            data=summary_data,
            ax=ax,
            title="Circuit Summary Metrics",
            xlabel="Metrics",
            ylabel="Value",
            value_format="{:.3f}",
        )

        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="qiskit_circuit_summary",
        )

    def create_dashboard(self, save: bool = False, show: bool = False) -> plt.Figure:
        """Creates a comprehensive dashboard with all QiskitMetrics plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("QiskitMetrics Analysis Dashboard", fontsize=16)

        # Create each plot on its designated axes
        self.plot_circuit_structure(save=False, show=False, fig_ax_override=(fig, ax1))
        self.plot_gate_distribution(save=False, show=False, fig_ax_override=(fig, ax2))
        self.plot_instruction_metrics(save=False, show=False, fig_ax_override=(fig, ax3))
        self.plot_circuit_summary(save=False, show=False, fig_ax_override=(fig, ax4))

        plt.tight_layout()

        if save:
            self.save_plot(fig, "qiskit_metrics_dashboard")
        if show:
            self.show_plot(fig)
        return fig
