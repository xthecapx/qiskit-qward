"""
Quantum Specific metrics visualization strategy for QWARD.

Visualiza métricas cuánticas específicas calculadas por QuantumSpecificMetrics:
- %SpposQ (spposq_ratio)
- Magic
- Coherence
- Sensitivity
- Entanglement Ratio
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .base import VisualizationStrategy, PlotConfig, PlotMetadata, PlotType, PlotRegistry
from .constants import Plots


class QuantumSpecificMetricsVisualizer(VisualizationStrategy):
    """Estrategia de visualización para QuantumSpecificMetrics (compacta)."""

    PLOT_REGISTRY: PlotRegistry = {
        Plots.QuantumSpecific.ALL_METRICS_BAR: PlotMetadata(
            name=Plots.QuantumSpecific.ALL_METRICS_BAR,
            method_name="plot_all_metrics_bar",
            description="Barras con todas las métricas cuánticas: %SpposQ, Magic, Coherence, Sensitivity, Entanglement Ratio",
            plot_type=PlotType.BAR_CHART,
            filename="quantum_specific_all_metrics",
            dependencies=[
                "spposq_ratio",
                "magic",
                "coherence",
                "sensitivity",
                "entanglement_ratio",
            ],
            category="QuantumSpecific",
        ),
        Plots.QuantumSpecific.QUANTUM_RADAR: PlotMetadata(
            name=Plots.QuantumSpecific.QUANTUM_RADAR,
            method_name="plot_quantum_radar",
            description="Radar con métricas normalizadas (relativas al máximo): %SpposQ, Magic, Coherence, Sensitivity, Entanglement",
            plot_type=PlotType.RADAR_CHART,
            filename="quantum_specific_radar",
            dependencies=[
                "magic",
                "coherence",
                "sensitivity",
                "entanglement_ratio",
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

        self.qs_df = self.metrics_dict.get("QuantumSpecificMetrics")
        if self.qs_df is None:
            raise ValueError("'QuantumSpecificMetrics' data not found in metrics_dict.")
        if self.qs_df.empty:
            raise ValueError("QuantumSpecificMetrics DataFrame is empty.")

    # ----------------------- Individual Plots -----------------------
    def plot_all_metrics_bar(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)

        spposq_ratio = float(self.qs_df.get("spposq_ratio", pd.Series([0.0])).iloc[0])
        magic = float(self.qs_df.get("magic", pd.Series([0.0])).iloc[0])
        coherence = float(self.qs_df.get("coherence", pd.Series([0.0])).iloc[0])
        sensitivity = float(self.qs_df.get("sensitivity", pd.Series([0.0])).iloc[0])
        entanglement_ratio = float(self.qs_df.get("entanglement_ratio", pd.Series([0.0])).iloc[0])

        data = {
            "%SpposQ": spposq_ratio,
            "Magic": magic,
            "Coherence": coherence,
            "Sensitivity": sensitivity,
            "Entanglement Ratio": entanglement_ratio,
        }

        self._create_bar_plot_with_labels(
            data=data,
            ax=ax,
            title=title or "Quantum Specific: All Metrics",
            xlabel="Métrica",
            ylabel="Valor",
            value_format="auto",
        )
        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="quantum_specific_all_metrics",
        )

    def plot_quantum_radar(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        # Preparar figura/axes (radar)
        if fig_ax_override:
            fig, ax = fig_ax_override
            is_override = True
        else:
            fig, ax = plt.subplots(figsize=self.config.figsize, subplot_kw={"projection": "polar"})
            is_override = False

        metrics = {
            "Magic": float(self.qs_df.get("magic", pd.Series([0.0])).iloc[0]),
            "Coherence": float(self.qs_df.get("coherence", pd.Series([0.0])).iloc[0]),
            "Sensitivity": float(self.qs_df.get("sensitivity", pd.Series([0.0])).iloc[0]),
            "Entanglement": float(self.qs_df.get("entanglement_ratio", pd.Series([0.0])).iloc[0]),
        }

        categories = list(metrics.keys())
        values = list(metrics.values())
        max_val = max(values) if len(values) > 0 else 1.0
        norm_values = [v / max_val if max_val > 0 else 0.0 for v in values]

        num_vars = len(categories)
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]
        norm_values += norm_values[:1]

        ax.plot(angles, norm_values, "o-", linewidth=2, color=self.config.color_palette[0])
        ax.fill(angles, norm_values, alpha=self.config.alpha, color=self.config.color_palette[0])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)

        final_title = self._get_final_title(title or "Quantum Specific Metrics (Radar)")
        if final_title is not None:
            ax.set_title(final_title, pad=20)
        ax.grid(True)

        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="quantum_specific_radar",
        )

    # ----------------------- Dashboard -----------------------
    def create_dashboard(
        self, save: bool = False, show: bool = False, title: Optional[str] = None
    ) -> plt.Figure:
        # Dashboard compacto: 1x2 (Radar + Barras)
        fig = plt.figure(figsize=(12, 6))

        final_title = self._get_final_title(title or "QuantumSpecificMetrics Dashboard (Compact)")
        if final_title is not None:
            fig.suptitle(final_title, fontsize=14)

        gs = fig.add_gridspec(1, 2, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0], projection="polar")
        ax2 = fig.add_subplot(gs[0, 1])

        self.plot_quantum_radar(save=False, show=False, fig_ax_override=(fig, ax1))
        self.plot_all_metrics_bar(save=False, show=False, fig_ax_override=(fig, ax2))

        if save:
            self.save_plot(fig, "quantum_specific_dashboard_compact")
        if show:
            self.show_plot(fig)
        return fig
