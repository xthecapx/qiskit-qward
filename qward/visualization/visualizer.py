"""
Main Visualizer class for QWARD metrics.

This module provides the unified Visualizer class that acts as the main entry point
for visualizing metrics from Scanner or custom data sources.
"""

from typing import Dict, List, Optional, Type, Union, Any
import pandas as pd
import matplotlib.pyplot as plt

from qward.scanner import Scanner
from .base import BaseVisualizer, PlotConfig
from .circuit_performance_visualizer import CircuitPerformanceVisualizer


class Visualizer:
    """
    Unified visualizer for QWARD metrics.

    This class provides a single entry point for visualizing metrics from Scanner
    or custom data sources. It automatically detects available metrics and provides
    appropriate visualizations.
    """

    def __init__(
        self,
        scanner: Optional[Scanner] = None,
        metrics_data: Optional[Dict[str, pd.DataFrame]] = None,
        config: Optional[PlotConfig] = None,
        output_dir: str = "img",
    ):
        """
        Initialize the Visualizer.

        Args:
            scanner: Scanner instance containing metrics strategies
            metrics_data: Custom metrics data as DataFrames
            config: Global plot configuration for all visualizations
            output_dir: Directory to save plots

        Raises:
            ValueError: If neither scanner nor metrics_data is provided
        """
        if scanner is None and metrics_data is None:
            raise ValueError("Either scanner or metrics_data must be provided")

        self.scanner = scanner
        self.config = config or PlotConfig()
        self.output_dir = output_dir
        self.registered_visualizers: Dict[str, Type[BaseVisualizer]] = {}

        # Get metrics data
        if metrics_data is not None:
            self.metrics_data = metrics_data
        else:
            self.metrics_data = self.scanner.calculate_metrics()

        # Auto-register default visualizers
        self._auto_register_default_visualizers()

    def _auto_register_default_visualizers(self) -> None:
        """Auto-register default visualizers based on available metrics."""
        # Register CircuitPerformance visualizer if data is available
        if any(key.startswith("CircuitPerformance") for key in self.metrics_data.keys()):
            self.register_visualizer("CircuitPerformance", CircuitPerformanceVisualizer)

        # Register QiskitMetrics visualizer if data is available
        if "QiskitMetrics" in self.metrics_data:
            from .qiskit_metrics_visualizer import QiskitMetricsVisualizer

            self.register_visualizer("QiskitMetrics", QiskitMetricsVisualizer)

        # Register ComplexityMetrics visualizer if data is available
        if "ComplexityMetrics" in self.metrics_data:
            from .complexity_metrics_visualizer import ComplexityMetricsVisualizer

            self.register_visualizer("ComplexityMetrics", ComplexityMetricsVisualizer)

    def register_visualizer(self, metric_name: str, visualizer_class: Type[BaseVisualizer]) -> None:
        """
        Register a visualizer for a specific metric.

        Args:
            metric_name: Name of the metric (e.g., "QiskitMetrics", "ComplexityMetrics")
            visualizer_class: Visualizer class that inherits from BaseVisualizer
        """
        self.registered_visualizers[metric_name] = visualizer_class

    def get_available_metrics(self) -> List[str]:
        """
        Get list of available metrics that can be visualized.

        Returns:
            List[str]: List of metric names available for visualization
        """
        available = []
        for metric_name in self.registered_visualizers:
            if self._has_metric_data(metric_name):
                available.append(metric_name)
        return available

    def _has_metric_data(self, metric_name: str) -> bool:
        """Check if data is available for a specific metric."""
        if metric_name == "CircuitPerformance":
            return any(key.startswith("CircuitPerformance") for key in self.metrics_data.keys())
        return metric_name in self.metrics_data

    def _get_metric_data(self, metric_name: str) -> Dict[str, pd.DataFrame]:
        """Get data for a specific metric."""
        if metric_name == "CircuitPerformance":
            # Return all CircuitPerformance-related data
            return {
                key: df
                for key, df in self.metrics_data.items()
                if key.startswith("CircuitPerformance")
            }
        else:
            # Return single metric data
            return {metric_name: self.metrics_data[metric_name]}

    def visualize_metric(
        self, metric_name: str, save: bool = True, show: bool = True, **kwargs
    ) -> List[plt.Figure]:
        """
        Visualize a specific metric.

        Args:
            metric_name: Name of the metric to visualize
            save: Whether to save the plots
            show: Whether to display the plots
            **kwargs: Additional arguments passed to the visualizer

        Returns:
            List[plt.Figure]: List of generated figures

        Raises:
            ValueError: If metric is not available or not registered
        """
        if metric_name not in self.registered_visualizers:
            raise ValueError(f"No visualizer registered for metric '{metric_name}'")

        if not self._has_metric_data(metric_name):
            raise ValueError(f"No data available for metric '{metric_name}'")

        # Get the visualizer class and create instance
        visualizer_class = self.registered_visualizers[metric_name]
        metric_data = self._get_metric_data(metric_name)

        # Create visualizer instance
        visualizer = visualizer_class(  # type: ignore[call-arg]
            metrics_dict=metric_data, output_dir=self.output_dir, config=self.config
        )

        # Generate all plots for this metric
        return visualizer.plot_all(save=save, show=show, **kwargs)  # type: ignore[attr-defined]

    def create_dashboard(
        self, save: bool = True, show: bool = True, **kwargs
    ) -> Dict[str, plt.Figure]:
        """
        Create a comprehensive dashboard with all available metrics.

        Args:
            save: Whether to save the dashboard
            show: Whether to display the dashboard
            **kwargs: Additional arguments passed to visualizers

        Returns:
            Dict[str, plt.Figure]: Dictionary mapping metric names to dashboard figures
        """
        dashboards: Dict[str, plt.Figure] = {}
        available_metrics = self.get_available_metrics()

        if not available_metrics:
            print("No metrics available for visualization")
            return dashboards

        print(f"Creating dashboard for metrics: {', '.join(available_metrics)}")

        for metric_name in available_metrics:
            try:
                # Get the visualizer class and create instance
                visualizer_class = self.registered_visualizers[metric_name]
                metric_data = self._get_metric_data(metric_name)

                # Create visualizer instance
                visualizer = visualizer_class(  # type: ignore[call-arg]
                    metrics_dict=metric_data, output_dir=self.output_dir, config=self.config
                )

                # Create dashboard for this metric
                dashboard_fig = visualizer.create_dashboard(save=save, show=show, **kwargs)  # type: ignore[attr-defined]
                dashboards[metric_name] = dashboard_fig

            except Exception as e:
                print(f"Warning: Failed to create dashboard for {metric_name}: {e}")
                continue

        return dashboards

    def visualize_all(
        self, save: bool = True, show: bool = True, **kwargs
    ) -> Dict[str, List[plt.Figure]]:
        """
        Create individual plots for all available metrics.

        Args:
            save: Whether to save the plots
            show: Whether to display the plots
            **kwargs: Additional arguments passed to visualizers

        Returns:
            Dict[str, List[plt.Figure]]: Dictionary mapping metric names to lists of figures
        """
        all_figures: Dict[str, List[plt.Figure]] = {}
        available_metrics = self.get_available_metrics()

        if not available_metrics:
            print("No metrics available for visualization")
            return all_figures

        print(f"Creating visualizations for metrics: {', '.join(available_metrics)}")

        for metric_name in available_metrics:
            try:
                figures = self.visualize_metric(metric_name, save=save, show=show, **kwargs)
                all_figures[metric_name] = figures
            except Exception as e:
                print(f"Warning: Failed to create visualizations for {metric_name}: {e}")
                continue

        return all_figures

    def get_metric_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of available metrics and their key statistics.

        Returns:
            Dict[str, Dict[str, Any]]: Summary information for each metric
        """
        summary = {}

        for metric_name, df in self.metrics_data.items():
            summary[metric_name] = {
                "shape": df.shape,
                "columns": list(df.columns),
                "has_data": not df.empty,
                "sample_values": df.iloc[0].to_dict() if not df.empty else {},
            }

        return summary

    def print_available_metrics(self) -> None:
        """Print information about available metrics and visualizations."""
        print("=== QWARD Visualizer Summary ===")
        print(f"Output directory: {self.output_dir}")
        print(f"Total metrics datasets: {len(self.metrics_data)}")

        available_metrics = self.get_available_metrics()
        print(f"Available visualizations: {len(available_metrics)}")

        for metric_name in available_metrics:
            metric_data = self._get_metric_data(metric_name)
            total_rows = sum(df.shape[0] for df in metric_data.values())
            print(f"  - {metric_name}: {len(metric_data)} dataset(s), {total_rows} total rows")

        if not available_metrics:
            print("  No visualizations available. Register visualizers or check data.")
