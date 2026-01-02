"""
Element metrics visualization strategy for QWARD.

This strategy visualiza métricas a nivel de elementos (puertas, oráculos,
mediciones) calculadas por `ElementMetrics`.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .base import VisualizationStrategy, PlotConfig, PlotMetadata, PlotType, PlotRegistry
from .constants import Plots


class ElementMetricsVisualizer(VisualizationStrategy):
    """Estrategia de visualización para ElementMetrics."""

    # Registro de plots disponibles para esta estrategia
    PLOT_REGISTRY: PlotRegistry = {
        Plots.Element.PAULI_GATES: PlotMetadata(
            name=Plots.Element.PAULI_GATES,
            method_name="plot_pauli_gates",
            description="Conteo de compuertas Pauli (X, Y, Z) y total",
            plot_type=PlotType.BAR_CHART,
            filename="element_pauli_gates",
            dependencies=["no_p_x", "no_p_y", "no_p_z", "t_no_p"],
            category="Gates",
        ),
        Plots.Element.SINGLE_QUBIT_BREAKDOWN: PlotMetadata(
            name=Plots.Element.SINGLE_QUBIT_BREAKDOWN,
            method_name="plot_single_qubit_breakdown",
            description="Distribución de compuertas de un qubit (H, Pauli, otras, controladas)",
            plot_type=PlotType.BAR_CHART,
            filename="element_single_qubit_breakdown",
            dependencies=["no_h", "t_no_p", "no_other_sg", "t_no_csqg", "t_no_sqg"],
            category="Gates",
        ),
        Plots.Element.CONTROLLED_GATES: PlotMetadata(
            name=Plots.Element.CONTROLLED_GATES,
            method_name="plot_controlled_gates",
            description="Conteo de compuertas controladas (CNOT, Toffoli, SWAP, oráculos controlados)",
            plot_type=PlotType.BAR_CHART,
            filename="element_controlled_gates",
            dependencies=["no_cnot", "no_toff", "no_swap", "no_c_or", "no_c_any_g"],
            category="Controlled",
        ),
        Plots.Element.ORACLE_USAGE: PlotMetadata(
            name=Plots.Element.ORACLE_USAGE,
            method_name="plot_oracle_usage",
            description="Uso de oráculos (simples y controlados)",
            plot_type=PlotType.BAR_CHART,
            filename="element_oracle_usage",
            dependencies=["no_or", "no_c_or"],
            category="Oracles",
        ),
        Plots.Element.ORACLE_RATIOS: PlotMetadata(
            name=Plots.Element.ORACLE_RATIOS,
            method_name="plot_oracle_ratios",
            description="Proporción de qubits afectados por oráculos (simples/ctrl) y profundidad media/máxima",
            plot_type=PlotType.BAR_CHART,
            filename="element_oracle_ratios",
            dependencies=["percent_q_in_or", "percent_q_in_c_or", "avg_or_d", "max_or_d"],
            category="Oracles",
        ),
        Plots.Element.CNOT_TOFFOLI_STATS: PlotMetadata(
            name=Plots.Element.CNOT_TOFFOLI_STATS,
            method_name="plot_cnot_toffoli_stats",
            description="Estadísticas de CNOT y Toffoli (ratio qubits afectados, promedio y máximo)",
            plot_type=PlotType.BAR_CHART,
            filename="element_cnot_toffoli_stats",
            dependencies=[
                "percent_q_in_cnot",
                "avg_cnot",
                "max_cnot",
                "percent_q_in_toff",
                "avg_toff",
                "max_toff",
            ],
            category="Controlled",
        ),
        Plots.Element.MEASUREMENT_ANCILLA: PlotMetadata(
            name=Plots.Element.MEASUREMENT_ANCILLA,
            method_name="plot_measurement_ancilla",
            description="Métricas de medición y ancillas",
            plot_type=PlotType.BAR_CHART,
            filename="element_measurement_ancilla",
            dependencies=["no_qm", "percent_qm", "percent_anc"],
            category="Measurement",
        ),
        Plots.Element.SUMMARY: PlotMetadata(
            name=Plots.Element.SUMMARY,
            method_name="plot_summary",
            description="Resumen general: totales y proporciones clave",
            plot_type=PlotType.BAR_CHART,
            filename="element_summary",
            dependencies=["no_gates", "no_c_gates", "percent_single_gates"],
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

        self.element_df = self.metrics_dict.get("ElementMetrics")
        if self.element_df is None:
            raise ValueError("'ElementMetrics' data not found in metrics_dict.")
        if self.element_df.empty:
            raise ValueError("ElementMetrics DataFrame is empty.")

    # ----------------------- Individual plots -----------------------
    def plot_pauli_gates(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        data = {
            "X": self.element_df["no_p_x"].iloc[0],
            "Y": self.element_df["no_p_y"].iloc[0],
            "Z": self.element_df["no_p_z"].iloc[0],
            "Total": self.element_df["t_no_p"].iloc[0],
        }

        self._create_bar_plot_with_labels(
            data=data,
            ax=ax,
            title=title or "Pauli Gates",
            xlabel="Tipo",
            ylabel="Cantidad",
            value_format="int",
        )

        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="element_pauli_gates",
        )

    def plot_single_qubit_breakdown(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        data = {
            "Hadamard": self.element_df["no_h"].iloc[0],
            "Pauli (X+Y+Z)": self.element_df["t_no_p"].iloc[0],
            "Otras 1Q": self.element_df["no_other_sg"].iloc[0],
            "Ctrl 1Q": self.element_df["t_no_csqg"].iloc[0],
            "Total 1Q": self.element_df["t_no_sqg"].iloc[0],
        }

        self._create_bar_plot_with_labels(
            data=data,
            ax=ax,
            title=title or "Distribución compuertas de 1 qubit",
            xlabel="Categoría",
            ylabel="Cantidad",
            value_format="int",
        )

        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="element_single_qubit_breakdown",
        )

    def plot_controlled_gates(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        data = {
            "CNOT": self.element_df["no_cnot"].iloc[0],
            "Toffoli": self.element_df["no_toff"].iloc[0],
            "SWAP": self.element_df["no_swap"].iloc[0],
            "Oráculos ctrl": self.element_df["no_c_or"].iloc[0],
            "Ctrl (total)": self.element_df["no_c_any_g"].iloc[0],
        }

        self._create_bar_plot_with_labels(
            data=data,
            ax=ax,
            title=title or "Compuertas controladas",
            xlabel="Tipo",
            ylabel="Cantidad",
            value_format="int",
        )

        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="element_controlled_gates",
        )

    def plot_oracle_usage(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        data = {
            "Oráculos": self.element_df["no_or"].iloc[0],
            "Oráculos ctrl": self.element_df["no_c_or"].iloc[0],
        }

        self._create_bar_plot_with_labels(
            data=data,
            ax=ax,
            title=title or "Uso de oráculos",
            xlabel="Tipo",
            ylabel="Cantidad",
            value_format="int",
        )

        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="element_oracle_usage",
        )

    def plot_oracle_ratios(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        data = {
            "% qubits en oráculos": self.element_df["percent_q_in_or"].iloc[0],
            "% qubits en oráculos ctrl": self.element_df["percent_q_in_c_or"].iloc[0],
            "Prof. media oráculo": self.element_df["avg_or_d"].iloc[0],
            "Prof. máx oráculo": self.element_df["max_or_d"].iloc[0],
        }

        self._create_bar_plot_with_labels(
            data=data,
            ax=ax,
            title=title or "Proporciones/Profundidad de oráculos",
            xlabel="Métrica",
            ylabel="Valor",
            value_format="auto",
        )

        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="element_oracle_ratios",
        )

    def plot_cnot_toffoli_stats(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        # Creamos una figura con dos subplots para CNOT y Toffoli
        if fig_ax_override:
            fig, base_ax = fig_ax_override
            is_override = True
            # Si nos pasan ejes, usamos uno solo (combinamos métricas)
            axs = [base_ax]
        else:
            fig, axs = plt.subplots(
                1, 2, figsize=(self.config.figsize[0] + 2, self.config.figsize[1])
            )
            is_override = False

        # Datos CNOT
        cnot_data = {
            "% qubits en CNOT": self.element_df["percent_q_in_cnot"].iloc[0],
            "Promedio CNOT por qubit": self.element_df["avg_cnot"].iloc[0],
            "Máximo CNOT por qubit": self.element_df["max_cnot"].iloc[0],
        }

        # Datos Toffoli
        toff_data = {
            "% qubits en Toffoli": self.element_df["percent_q_in_toff"].iloc[0],
            "Promedio Toffoli por qubit": self.element_df["avg_toff"].iloc[0],
            "Máximo Toffoli por qubit": self.element_df["max_toff"].iloc[0],
        }

        if len(axs) == 1:
            # Combinar ambos conjuntos si solo tenemos un eje
            combined = {
                **{f"CNOT - {k}": v for k, v in cnot_data.items()},
                **{f"Toffoli - {k}": v for k, v in toff_data.items()},
            }
            self._create_bar_plot_with_labels(
                data=combined,
                ax=axs[0],
                title=title or "CNOT & Toffoli - estadísticas",
                xlabel="Métrica",
                ylabel="Valor",
                value_format="auto",
            )
        else:
            self._create_bar_plot_with_labels(
                data=cnot_data,
                ax=axs[0],
                title="CNOT - estadísticas",
                xlabel="Métrica",
                ylabel="Valor",
                value_format="auto",
            )
            self._create_bar_plot_with_labels(
                data=toff_data,
                ax=axs[1],
                title="Toffoli - estadísticas",
                xlabel="Métrica",
                ylabel="Valor",
                value_format="auto",
            )

        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="element_cnot_toffoli_stats",
        )

    def plot_measurement_ancilla(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        data = {
            "Qubits medidos": self.element_df["no_qm"].iloc[0],
            "% qubits medidos": self.element_df["percent_qm"].iloc[0],
            "% ancillas": self.element_df["percent_anc"].iloc[0],
        }

        self._create_bar_plot_with_labels(
            data=data,
            ax=ax,
            title=title or "Mediciones y ancillas",
            xlabel="Métrica",
            ylabel="Valor",
            value_format="auto",
        )

        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="element_measurement_ancilla",
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
            "Total compuertas": self.element_df["no_gates"].iloc[0],
            "Compuertas controladas": self.element_df["no_c_gates"].iloc[0],
            "% compuertas 1Q": self.element_df["percent_single_gates"].iloc[0],
        }

        self._create_bar_plot_with_labels(
            data=data,
            ax=ax,
            title=title or "Resumen de elementos",
            xlabel="Métrica",
            ylabel="Valor",
            value_format="auto",
        )

        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="element_summary",
        )

    # ----------------------- Dashboard -----------------------
    def create_dashboard(
        self, save: bool = False, show: bool = False, title: Optional[str] = None
    ) -> plt.Figure:
        fig = plt.figure(figsize=(16, 12))

        final_title = self._get_final_title(title or "ElementMetrics Dashboard")
        if final_title is not None:
            fig.suptitle(final_title, fontsize=16)

        # 2 filas x 3 columnas
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])

        self.plot_pauli_gates(save=False, show=False, fig_ax_override=(fig, ax1))
        self.plot_single_qubit_breakdown(save=False, show=False, fig_ax_override=(fig, ax2))
        self.plot_controlled_gates(save=False, show=False, fig_ax_override=(fig, ax3))
        self.plot_oracle_usage(save=False, show=False, fig_ax_override=(fig, ax4))
        self.plot_oracle_ratios(save=False, show=False, fig_ax_override=(fig, ax5))
        self.plot_measurement_ancilla(save=False, show=False, fig_ax_override=(fig, ax6))

        if save:
            self.save_plot(fig, "element_metrics_dashboard")
        if show:
            self.show_plot(fig)
        return fig
