"""
Structural metrics visualization strategy for QWARD.

This strategy visualiza métricas estructurales (LOC, Halstead y estructura
del circuito) calculadas por `StructuralMetrics`.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .base import VisualizationStrategy, PlotConfig, PlotMetadata, PlotType, PlotRegistry
from .constants import Plots


class StructuralMetricsVisualizer(VisualizationStrategy):
    """Estrategia de visualización para StructuralMetrics."""

    PLOT_REGISTRY: PlotRegistry = {
        Plots.Structural.LOC_BREAKDOWN: PlotMetadata(
            name=Plots.Structural.LOC_BREAKDOWN,
            method_name="plot_loc_breakdown",
            description="Desglose de métricas LOC (total, gates, measure, quantum total y qubits)",
            plot_type=PlotType.BAR_CHART,
            filename="struct_loc_breakdown",
            dependencies=[
                "loc_phi1_total_loc",
                "loc_phi2_gate_loc",
                "loc_phi3_measure_loc",
                "loc_phi4_quantum_total_loc",
                "loc_phi5_num_qubits",
                "loc_phi6_num_gate_types",
            ],
            category="LOC",
        ),
        Plots.Structural.HALSTEAD_BASIC: PlotMetadata(
            name=Plots.Structural.HALSTEAD_BASIC,
            method_name="plot_halstead_basic",
            description="Métricas básicas de Halstead (operadores/operandos únicos y totales)",
            plot_type=PlotType.BAR_CHART,
            filename="struct_halstead_basic",
            dependencies=[
                "unique_operators",
                "unique_operands",
                "total_operators",
                "total_operands",
                "program_length",
                "vocabulary",
            ],
            category="Halstead",
        ),
        Plots.Structural.HALSTEAD_DERIVED: PlotMetadata(
            name=Plots.Structural.HALSTEAD_DERIVED,
            method_name="plot_halstead_derived",
            description="Métricas derivadas de Halstead (estimated_length, volume, difficulty, effort)",
            plot_type=PlotType.BAR_CHART,
            filename="struct_halstead_derived",
            dependencies=[
                "estimated_length",
                "volume",
                "difficulty",
                "effort",
            ],
            category="Halstead",
        ),
        Plots.Structural.STRUCTURE_DIMENSIONS: PlotMetadata(
            name=Plots.Structural.STRUCTURE_DIMENSIONS,
            method_name="plot_structure_dimensions",
            description="Dimensiones del circuito (width, depth, size)",
            plot_type=PlotType.BAR_CHART,
            filename="struct_structure_dimensions",
            dependencies=[
                "width",
                "depth",
                "size",
            ],
            category="Structure",
        ),
        Plots.Structural.DENSITY_METRICS: PlotMetadata(
            name=Plots.Structural.DENSITY_METRICS,
            method_name="plot_density_metrics",
            description="Métricas de densidad por capa (máxima y promedio)",
            plot_type=PlotType.BAR_CHART,
            filename="struct_density_metrics",
            dependencies=[
                "max_dens",
                "avg_dens",
            ],
            category="Structure",
        ),
        Plots.Structural.SUMMARY: PlotMetadata(
            name=Plots.Structural.SUMMARY,
            method_name="plot_summary",
            description="Resumen de métricas estructurales clave",
            plot_type=PlotType.BAR_CHART,
            filename="struct_summary",
            dependencies=[
                "loc_phi1_total_loc",
                "loc_phi2_gate_loc",
                "loc_phi3_measure_loc",
                "width",
                "depth",
                "size",
            ],
            category="Summary",
        ),
    }

    @classmethod
    def get_available_plots(cls) -> List[str]:
        return list(cls.PLOT_REGISTRY.keys())

    @classmethod
    def get_plot_metadata(cls, plot_name: str) -> PlotMetadata:
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
        super().__init__(metrics_dict, output_dir, config)

        self.struct_df = self.metrics_dict.get("StructuralMetrics")
        if self.struct_df is None:
            raise ValueError("'StructuralMetrics' data not found in metrics_dict.")
        if self.struct_df.empty:
            raise ValueError("StructuralMetrics DataFrame is empty.")

    # ----------------------- Individual plots -----------------------
    def plot_loc_breakdown(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        cols = [
            "loc_phi1_total_loc",
            "loc_phi2_gate_loc",
            "loc_phi3_measure_loc",
            "loc_phi4_quantum_total_loc",
            "loc_phi5_num_qubits",
            "loc_phi6_num_gate_types",
        ]
        data = {}
        for col in cols:
            if col in self.struct_df.columns:
                label = col.replace("loc_", "").replace("_", " ").title()
                data[label] = self.struct_df[col].iloc[0]

        self._create_bar_plot_with_labels(
            data=data,
            ax=ax,
            title=title or "LOC breakdown",
            xlabel="Métrica",
            ylabel="Valor",
            value_format="auto",
        )

        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="struct_loc_breakdown",
        )

    def plot_halstead_basic(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        cols = [
            "unique_operators",
            "unique_operands",
            "total_operators",
            "total_operands",
            "program_length",
            "vocabulary",
        ]
        data = {}
        for col in cols:
            if col in self.struct_df.columns:
                data[col.replace("_", " ").title()] = self.struct_df[col].iloc[0]

        self._create_bar_plot_with_labels(
            data=data,
            ax=ax,
            title=title or "Halstead - básicos",
            xlabel="Métrica",
            ylabel="Valor",
            value_format="auto",
        )

        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="struct_halstead_basic",
        )

    def plot_halstead_derived(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        cols = ["estimated_length", "volume", "difficulty", "effort"]
        data = {}
        for col in cols:
            if col in self.struct_df.columns:
                data[col.replace("_", " ").title()] = self.struct_df[col].iloc[0]

        self._create_bar_plot_with_labels(
            data=data,
            ax=ax,
            title=title or "Halstead - derivadas",
            xlabel="Métrica",
            ylabel="Valor",
            value_format="auto",
        )

        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="struct_halstead_derived",
        )

    def plot_structure_dimensions(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        data = {
            "Width": self.struct_df.get("width", pd.Series([0])).iloc[0],
            "Depth": self.struct_df.get("depth", pd.Series([0])).iloc[0],
            "Size": self.struct_df.get("size", pd.Series([0])).iloc[0],
        }

        self._create_bar_plot_with_labels(
            data=data,
            ax=ax,
            title=title or "Dimensiones del circuito",
            xlabel="Métrica",
            ylabel="Valor",
            value_format="auto",
        )

        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="struct_structure_dimensions",
        )

    def plot_density_metrics(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        data = {
            "Densidad máxima": self.struct_df.get("max_dens", pd.Series([0])).iloc[0],
            "Densidad promedio": self.struct_df.get("avg_dens", pd.Series([0])).iloc[0],
        }

        self._create_bar_plot_with_labels(
            data=data,
            ax=ax,
            title=title or "Densidad por capa",
            xlabel="Métrica",
            ylabel="Valor",
            value_format="auto",
        )

        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="struct_density_metrics",
        )

    def plot_summary(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        data = {
            "Total LOC": self.struct_df.get("loc_phi1_total_loc", pd.Series([0])).iloc[0],
            "LOC Gates": self.struct_df.get("loc_phi2_gate_loc", pd.Series([0])).iloc[0],
            "LOC Measure": self.struct_df.get("loc_phi3_measure_loc", pd.Series([0])).iloc[0],
            "Width": self.struct_df.get("width", pd.Series([0])).iloc[0],
            "Depth": self.struct_df.get("depth", pd.Series([0])).iloc[0],
            "Size": self.struct_df.get("size", pd.Series([0])).iloc[0],
        }

        self._create_bar_plot_with_labels(
            data=data,
            ax=ax,
            title=title or "Resumen estructural",
            xlabel="Métrica",
            ylabel="Valor",
            value_format="auto",
        )

        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="struct_summary",
        )

    # ----------------------- Dashboard -----------------------
    def create_dashboard(
        self, save: bool = False, show: bool = False, title: Optional[str] = None
    ) -> plt.Figure:
        fig = plt.figure(figsize=(16, 12))

        final_title = self._get_final_title(title or "StructuralMetrics Dashboard")
        if final_title is not None:
            fig.suptitle(final_title, fontsize=16)

        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])

        self.plot_loc_breakdown(save=False, show=False, fig_ax_override=(fig, ax1))
        self.plot_halstead_basic(save=False, show=False, fig_ax_override=(fig, ax2))
        self.plot_halstead_derived(save=False, show=False, fig_ax_override=(fig, ax3))
        self.plot_structure_dimensions(save=False, show=False, fig_ax_override=(fig, ax4))
        self.plot_density_metrics(save=False, show=False, fig_ax_override=(fig, ax5))
        self.plot_summary(save=False, show=False, fig_ax_override=(fig, ax6))

        if save:
            self.save_plot(fig, "structural_metrics_dashboard")
        if show:
            self.show_plot(fig)
        return fig
