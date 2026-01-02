"""
Behavioral metrics visualization strategy for QWARD.

Visualiza métricas de comportamiento del circuito (comunicación, ruta crítica,
medición, liveness, paralelismo y depth normalizado) calculadas por BehavioralMetrics.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .base import VisualizationStrategy, PlotConfig, PlotMetadata, PlotType, PlotRegistry
from .constants import Plots


class BehavioralMetricsVisualizer(VisualizationStrategy):
    """Estrategia de visualización para BehavioralMetrics (versión compacta, sin gráficos redundantes o de una sola barra)."""

    # Mantener solo gráficos con varias dimensiones o comparativos relevantes
    PLOT_REGISTRY: PlotRegistry = {
        Plots.Behavioral.MEASUREMENT_LIVENESS: PlotMetadata(
            name=Plots.Behavioral.MEASUREMENT_LIVENESS,
            method_name="plot_measurement_liveness",
            description="Barras con todas las métricas calculadas: normalized_depth, program_communication, critical_depth, measurement, liveness, parallelism",
            plot_type=PlotType.BAR_CHART,
            filename="behavioral_all_metrics",
            dependencies=[
                "normalized_depth",
                "program_communication",
                "critical_depth",
                "measurement",
                "liveness",
                "parallelism",
            ],
            category="Behavior",
        ),
        Plots.Behavioral.BEHAVIORAL_RADAR: PlotMetadata(
            name=Plots.Behavioral.BEHAVIORAL_RADAR,
            method_name="plot_behavioral_radar",
            description="Radar con las métricas normalizadas (comunicación, ruta crítica, medida, liveness, paralelismo)",
            plot_type=PlotType.RADAR_CHART,
            filename="behavioral_radar",
            dependencies=[
                "program_communication",
                "critical_depth",
                "measurement",
                "liveness",
                "parallelism",
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

        self.beh_df = self.metrics_dict.get("BehavioralMetrics")
        if self.beh_df is None:
            raise ValueError("'BehavioralMetrics' data not found in metrics_dict.")
        if self.beh_df.empty:
            raise ValueError("BehavioralMetrics DataFrame is empty.")

    def plot_measurement_liveness(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)
        # Recoger todas las métricas calculadas por BehavioralMetrics
        normalized_depth = float(self.beh_df.get("normalized_depth", pd.Series([0.0])).iloc[0])
        program_communication = float(
            self.beh_df.get("program_communication", pd.Series([0.0])).iloc[0]
        )
        critical_depth = float(self.beh_df.get("critical_depth", pd.Series([0.0])).iloc[0])
        measurement = float(self.beh_df.get("measurement", pd.Series([0.0])).iloc[0])
        liveness = float(self.beh_df.get("liveness", pd.Series([0.0])).iloc[0])
        parallelism = float(self.beh_df.get("parallelism", pd.Series([0.0])).iloc[0])

        data = {
            "Normalized Depth": normalized_depth,
            "Communication": program_communication,
            "Critical Depth": critical_depth,
            "Measurement": measurement,
            "Liveness": liveness,
            "Parallelism": parallelism,
        }
        self._create_bar_plot_with_labels(
            data=data,
            ax=ax,
            title=title or "All Behavioral Metrics",
            xlabel="Métrica",
            ylabel="Valor",
            value_format="auto",
        )
        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="behavioral_all_metrics",
        )

    # (Se eliminan gráficos de una sola barra y redundantes como parallelism, program_communication, critical_depth, normalized_depth)

    def plot_behavioral_radar(
        self,
        save: bool = False,
        show: bool = False,
        fig_ax_override: Optional[tuple[plt.Figure, plt.Axes]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        # radar en proyección polar
        if fig_ax_override:
            fig, ax = fig_ax_override
            is_override = True
        else:
            fig, ax = plt.subplots(figsize=self.config.figsize, subplot_kw={"projection": "polar"})
            is_override = False

        metrics = {
            "Communication": float(
                self.beh_df.get("program_communication", pd.Series([0.0])).iloc[0]
            ),
            "Critical Depth": float(self.beh_df.get("critical_depth", pd.Series([0.0])).iloc[0]),
            "Measurement": float(self.beh_df.get("measurement", pd.Series([0.0])).iloc[0]),
            "Liveness": float(self.beh_df.get("liveness", pd.Series([0.0])).iloc[0]),
            "Parallelism": float(self.beh_df.get("parallelism", pd.Series([0.0])).iloc[0]),
        }

        categories = list(metrics.keys())
        values = list(metrics.values())
        num_vars = len(categories)

        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2, color=self.config.color_palette[0])
        ax.fill(angles, values, alpha=self.config.alpha, color=self.config.color_palette[0])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)

        final_title = self._get_final_title(title or "Behavioral Metrics (Radar)")
        if final_title is not None:
            ax.set_title(final_title, pad=20)
        ax.grid(True)

        return self._finalize_plot(
            fig=fig, is_override=is_override, save=save, show=show, filename="behavioral_radar"
        )

    # (Se elimina el plot de summary de barras para evitar duplicar la información del radar)

    # ----------------------- Dashboard -----------------------
    def create_dashboard(
        self, save: bool = False, show: bool = False, title: Optional[str] = None
    ) -> plt.Figure:
        # Dashboard compacto: 1x2 (Radar + Barras con todas las métricas)
        fig = plt.figure(figsize=(12, 6))

        final_title = self._get_final_title(title or "BehavioralMetrics Dashboard (Compact)")
        if final_title is not None:
            fig.suptitle(final_title, fontsize=14)

        gs = fig.add_gridspec(1, 2, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0], projection="polar")
        ax2 = fig.add_subplot(gs[0, 1])

        self.plot_behavioral_radar(save=False, show=False, fig_ax_override=(fig, ax1))
        self.plot_measurement_liveness(save=False, show=False, fig_ax_override=(fig, ax2))

        if save:
            self.save_plot(fig, "behavioral_metrics_dashboard_compact")
        if show:
            self.show_plot(fig)
        return fig
