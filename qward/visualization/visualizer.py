"""
Main Visualizer class for QWARD metrics using strategy pattern.

This module provides the unified Visualizer class that acts as the main entry point
for visualizing metrics from Scanner or custom data sources using pluggable strategies.
"""

from typing import Dict, List, Optional, Type, Union, Any
import pandas as pd
import matplotlib.pyplot as plt

from qward.scanner import Scanner
from .base import VisualizationStrategy, PlotConfig


class Visualizer:
    """
    Unified visualizer for QWARD metrics using strategy pattern.

    This class provides a single entry point for visualizing metrics from Scanner
    or custom data sources. It automatically detects available metrics and provides
    appropriate visualizations using pluggable strategies.
    """

    def __init__(
        self,
        scanner: Optional[Scanner] = None,
        metrics_data: Optional[Dict[str, pd.DataFrame]] = None,
        config: Optional[PlotConfig] = None,
        output_dir: str = "qward/examples/img",
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
        self.registered_strategies: Dict[str, Type[VisualizationStrategy]] = {}

        # Get metrics data
        if metrics_data is not None:
            self.metrics_data = metrics_data
        else:
            self.metrics_data = self.scanner.calculate_metrics()

        # Auto-register default strategies
        self._auto_register_default_strategies()

    def _auto_register_default_strategies(self) -> None:
        """Auto-register default visualization strategies based on available metrics."""
        # Register CircuitPerformance strategy if data is available
        if any(key.startswith("CircuitPerformance") for key in self.metrics_data.keys()):
            from .circuit_performance_visualizer import CircuitPerformanceVisualizer

            self.register_strategy("CircuitPerformance", CircuitPerformanceVisualizer)

        # Register Qiskit strategy if data is available
        if "QiskitMetrics" in self.metrics_data:
            from .qiskit_metrics_visualizer import QiskitVisualizer

            self.register_strategy("QiskitMetrics", QiskitVisualizer)

        # Register Complexity strategy if data is available
        if "ComplexityMetrics" in self.metrics_data:
            from .complexity_metrics_visualizer import ComplexityVisualizer

            self.register_strategy("ComplexityMetrics", ComplexityVisualizer)

    def register_strategy(
        self, metric_name: str, strategy_class: Type[VisualizationStrategy]
    ) -> None:
        """
        Register a visualization strategy for a specific metric.

        Args:
            metric_name: Name of the metric (e.g., "QiskitMetrics", "ComplexityMetrics")
            strategy_class: Strategy class that inherits from VisualizationStrategy
        """
        self.registered_strategies[metric_name] = strategy_class

    def get_available_metrics(self) -> List[str]:
        """
        Get list of available metrics that can be visualized.

        Returns:
            List[str]: List of metric names available for visualization
        """
        available = []
        for metric_name in self.registered_strategies:
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
        self, metric_name: str, save: bool = False, show: bool = False, **kwargs
    ) -> List[plt.Figure]:
        """
        Visualize a specific metric using its registered strategy.

        Args:
            metric_name: Name of the metric to visualize
            save: Whether to save the plots
            show: Whether to display the plots
            **kwargs: Additional arguments passed to the strategy

        Returns:
            List[plt.Figure]: List of generated figures

        Raises:
            ValueError: If metric is not available or not registered
        """
        if metric_name not in self.registered_strategies:
            raise ValueError(f"No strategy registered for metric '{metric_name}'")

        if not self._has_metric_data(metric_name):
            raise ValueError(f"No data available for metric '{metric_name}'")

        # Get the strategy class and create instance
        strategy_class = self.registered_strategies[metric_name]
        metric_data = self._get_metric_data(metric_name)

        # Create strategy instance
        strategy = strategy_class(
            metrics_dict=metric_data, output_dir=self.output_dir, config=self.config
        )

        # Generate all plots for this metric
        return strategy.plot_all(save=save, show=show, **kwargs)

    def create_dashboard(
        self, save: bool = False, show: bool = False, **kwargs
    ) -> Dict[str, plt.Figure]:
        """
        Create a comprehensive dashboard with all available metrics.

        Args:
            save: Whether to save the dashboard
            show: Whether to display the dashboard
            **kwargs: Additional arguments passed to strategies

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
                # Get the strategy class and create instance
                strategy_class = self.registered_strategies[metric_name]
                metric_data = self._get_metric_data(metric_name)

                # Create strategy instance
                strategy = strategy_class(
                    metrics_dict=metric_data, output_dir=self.output_dir, config=self.config
                )

                # Create dashboard for this metric
                dashboard_fig = strategy.create_dashboard(save=save, show=show, **kwargs)
                dashboards[metric_name] = dashboard_fig

            except Exception as e:
                print(f"Warning: Failed to create dashboard for {metric_name}: {e}")
                continue

        return dashboards

    def visualize_all(
        self, save: bool = False, show: bool = False, **kwargs
    ) -> Dict[str, List[plt.Figure]]:
        """
        Create individual plots for all available metrics.

        Args:
            save: Whether to save the plots
            show: Whether to display the plots
            **kwargs: Additional arguments passed to strategies

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
            print("  No visualizations available. Register strategies or check data.")

    def list_registered_strategies(self) -> Dict[str, str]:
        """
        List all registered visualization strategies.

        Returns:
            Dict[str, str]: Dictionary mapping metric names to strategy class names
        """
        return {
            metric_name: strategy_class.__name__
            for metric_name, strategy_class in self.registered_strategies.items()
        }
