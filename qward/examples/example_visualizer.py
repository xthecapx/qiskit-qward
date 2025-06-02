"""
Example demonstrating the new unified Visualizer system for QWARD.
"""

from .utils import get_display, create_example_circuit
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics
from qward.visualization import Visualizer, PlotConfig
from qward.visualization.constants import Metrics, Plots
from qiskit_aer import AerSimulator

display = get_display()


def example_basic_visualizer():
    """Example: Basic usage of the unified Visualizer."""
    print("\n=== Example: Basic Visualizer Usage ===")

    # Create a circuit and scanner
    circuit = create_example_circuit()
    scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics])

    # Create visualizer from scanner
    visualizer = Visualizer(scanner=scanner, output_dir="qward/examples/img")

    # Print available metrics
    visualizer.print_available_metrics()

    # Create dashboard for all available metrics
    print("\nCreating dashboard...")
    dashboards = visualizer.create_dashboard(save=True, show=False)

    print(f"Created {len(dashboards)} dashboards:")
    for metric_name, fig in dashboards.items():
        print(f"  - {metric_name}: {fig}")


def example_with_circuit_performance():
    """Example: Using all three visualizers including CircuitPerformanceVisualizer."""
    print("\n=== Example: All Visualizers with CircuitPerformanceVisualizer ===")

    # Create a circuit
    circuit = create_example_circuit()

    # Run circuit on simulator to get jobs for CircuitPerformanceMetrics
    simulator = AerSimulator()
    jobs = [simulator.run(circuit, shots=500) for _ in range(3)]

    # Create CircuitPerformanceMetrics with multiple jobs
    circuit_performance = CircuitPerformanceMetrics(circuit=circuit, jobs=jobs)

    # Create scanner with all metrics
    scanner = Scanner(
        circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics, circuit_performance]
    )

    # Create visualizer
    visualizer = Visualizer(scanner=scanner, output_dir="qward/examples/img")

    # Print available metrics
    visualizer.print_available_metrics()

    # Create dashboard for all available metrics
    print("\nCreating comprehensive dashboard...")
    dashboards = visualizer.create_dashboard(save=True, show=False)

    print(f"Created {len(dashboards)} dashboards:")
    for metric_name, fig in dashboards.items():
        print(f"  - {metric_name}: {fig}")


def example_custom_config():
    """Example: Using custom plot configuration."""
    print("\n=== Example: Custom Plot Configuration ===")

    # Create custom config
    custom_config = PlotConfig(
        figsize=(12, 8),
        style="quantum",
        dpi=300,
        color_palette=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
        alpha=0.8,
        grid=True,
    )

    # Create circuit and scanner
    circuit = create_example_circuit()
    scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics])

    # Create visualizer with custom config
    visualizer = Visualizer(
        scanner=scanner, config=custom_config, output_dir="qward/examples/custom_plots"
    )

    # Generate specific QiskitMetrics plots with custom config
    print("Creating QiskitMetrics visualizations with custom config...")
    figures = visualizer.generate_plots(
        {
            Metrics.QISKIT: [
                Plots.Qiskit.CIRCUIT_STRUCTURE,
                Plots.Qiskit.GATE_DISTRIBUTION,
                Plots.Qiskit.CIRCUIT_SUMMARY,
            ]
        },
        save=True,
        show=False,
    )

    qiskit_plot_count = len(figures[Metrics.QISKIT])
    print(f"Created {qiskit_plot_count} QiskitMetrics plots")


def example_custom_data():
    """Example: Using custom metrics data instead of Scanner."""
    print("\n=== Example: Custom Metrics Data ===")

    # Create some example data (normally this would come from your own calculations)
    import pandas as pd

    custom_qiskit_data = pd.DataFrame(
        [
            {
                "basic_metrics.depth": 5,
                "basic_metrics.width": 6,
                "basic_metrics.size": 8,
                "basic_metrics.num_qubits": 3,
                "basic_metrics.num_clbits": 3,
                "basic_metrics.count_ops.h": 2,
                "basic_metrics.count_ops.cx": 2,
                "basic_metrics.count_ops.measure": 3,
                "instruction_metrics.num_connected_components": 1,
                "instruction_metrics.num_nonlocal_gates": 2,
                "instruction_metrics.num_tensor_factors": 1,
                "instruction_metrics.num_unitary_factors": 1,
            }
        ]
    )

    custom_metrics_dict = {Metrics.QISKIT: custom_qiskit_data}

    # Create visualizer with custom data
    visualizer = Visualizer(metrics_data=custom_metrics_dict, output_dir="qward/examples/img")

    # Show what's available
    visualizer.print_available_metrics()

    # Create visualizations using new API
    print("Creating visualizations from custom data...")
    all_figures = visualizer.generate_plots(
        {Metrics.QISKIT: None}, save=True, show=False  # None = all plots
    )
    print(f"Created visualizations for {len(all_figures)} metric types")


def example_individual_strategies():
    """Example: Using individual visualization strategies directly."""
    print("\n=== Example: Individual Visualization Strategies ===")

    from qward.visualization import QiskitVisualizer, ComplexityVisualizer
    from qward.visualization import CircuitPerformanceVisualizer

    # Create circuit and get metrics
    circuit = create_example_circuit()

    # Create scanner with all metrics including CircuitPerformanceMetrics
    simulator = AerSimulator()
    jobs = [simulator.run(circuit, shots=500) for _ in range(2)]
    circuit_performance_metrics = CircuitPerformanceMetrics(circuit=circuit, jobs=jobs)

    scanner = Scanner(
        circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics, circuit_performance_metrics]
    )
    metrics_data = scanner.calculate_metrics()

    # Use QiskitVisualizer strategy directly
    if Metrics.QISKIT in metrics_data:
        print("Creating QiskitMetrics visualizations using QiskitVisualizer strategy...")
        qiskit_strategy = QiskitVisualizer(
            metrics_dict={Metrics.QISKIT: metrics_data[Metrics.QISKIT]},
            output_dir="qward/examples/img",
        )
        qiskit_figures = qiskit_strategy.generate_all_plots(save=True, show=False)
        print(f"Created {len(qiskit_figures)} QiskitMetrics plots")

    # Use ComplexityVisualizer strategy directly
    if Metrics.COMPLEXITY in metrics_data:
        print("Creating ComplexityMetrics visualizations using ComplexityVisualizer strategy...")
        complexity_strategy = ComplexityVisualizer(
            metrics_dict={Metrics.COMPLEXITY: metrics_data[Metrics.COMPLEXITY]},
            output_dir="qward/examples/img",
        )
        complexity_figures = complexity_strategy.generate_all_plots(save=True, show=False)
        print(f"Created {len(complexity_figures)} ComplexityMetrics plots")

    # Use CircuitPerformanceVisualizer strategy directly
    circuit_perf_data = {
        k: v for k, v in metrics_data.items() if k.startswith("CircuitPerformance")
    }
    if circuit_perf_data:
        print(
            "Creating CircuitPerformance visualizations using CircuitPerformanceVisualizer strategy..."
        )
        perf_strategy = CircuitPerformanceVisualizer(
            metrics_dict=circuit_perf_data, output_dir="qward/examples/img"
        )
        perf_figures = perf_strategy.generate_all_plots(save=True, show=False)
        print(f"Created {len(perf_figures)} CircuitPerformance plots")


def example_custom_strategy():
    """Example: Creating and registering a custom visualization strategy."""
    print("\n=== Example: Custom Visualization Strategy ===")

    from qward.visualization import VisualizationStrategy, PlotMetadata, PlotType
    import matplotlib.pyplot as plt

    class CustomMetricsStrategy(VisualizationStrategy):
        """Custom visualization strategy example."""

        # Class-level plot registry
        PLOT_REGISTRY = {
            "custom_plot": PlotMetadata(
                name="custom_plot",
                method_name="plot_custom_metric",
                description="Custom metric visualization",
                plot_type=PlotType.BAR_CHART,
                filename="custom_metric_plot",
                dependencies=["custom.metric"],
                category="custom",
            )
        }

        @classmethod
        def get_available_plots(cls):
            return list(cls.PLOT_REGISTRY.keys())

        @classmethod
        def get_plot_metadata(cls, plot_name):
            if plot_name not in cls.PLOT_REGISTRY:
                raise ValueError(f"Plot '{plot_name}' not found")
            return cls.PLOT_REGISTRY[plot_name]

        def create_dashboard(self, save=False, show=False):
            fig, ax = plt.subplots(figsize=self.config.figsize)
            ax.text(0.5, 0.5, "Custom Dashboard", ha="center", va="center", fontsize=16)
            ax.set_title("Custom Metrics Dashboard")

            if save:
                self.save_plot(fig, "custom_dashboard")
            if show:
                self.show_plot(fig)
            return fig

        def plot_custom_metric(self, save=False, show=False):
            fig, ax = plt.subplots(figsize=self.config.figsize)
            ax.bar(["A", "B", "C"], [1, 2, 3], color=self.config.color_palette[:3])
            ax.set_title("Custom Metric Plot")

            if save:
                self.save_plot(fig, "custom_metric")
            if show:
                self.show_plot(fig)
            return fig

    # Create custom data
    import pandas as pd

    custom_data = pd.DataFrame([{"custom.metric": 42}])
    custom_metrics_dict = {"CustomMetrics": custom_data}

    # Create visualizer and register custom strategy
    visualizer = Visualizer(metrics_data=custom_metrics_dict, output_dir="qward/examples/img")
    visualizer.register_strategy("CustomMetrics", CustomMetricsStrategy)

    # Use the custom strategy
    print("Creating custom visualizations...")
    custom_figures = visualizer.generate_plots(
        {"CustomMetrics": None}, save=True, show=False  # Generate all plots for custom metrics
    )
    print(f"Created {len(custom_figures['CustomMetrics'])} custom plots")


def example_metric_summary():
    """Example: Getting metric summaries and metadata."""
    print("\n=== Example: Metric Summary and Metadata ===")

    # Create circuit and scanner
    circuit = create_example_circuit()
    scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics])

    # Create visualizer
    visualizer = Visualizer(scanner=scanner, output_dir="qward/examples/img")

    # Get metric summary
    print("Metric Summary:")
    summary = visualizer.get_metric_summary()
    for metric_name, info in summary.items():
        print(f"  {metric_name}:")
        print(f"    Shape: {info['shape']}")
        print(f"    Columns: {len(info['columns'])}")
        print(f"    Has Data: {info['has_data']}")

    # Get available plots and their metadata
    print("\nAvailable Plots and Metadata:")
    available_plots = visualizer.get_available_plots()

    for metric_name, plot_names in available_plots.items():
        print(f"\n{metric_name} ({len(plot_names)} plots):")
        for plot_name in plot_names:
            metadata = visualizer.get_plot_metadata(metric_name, plot_name)
            print(f"  - {plot_name}:")
            print(f"    Description: {metadata.description}")
            print(f"    Type: {metadata.plot_type.value}")
            print(f"    Category: {metadata.category}")

    # Generate specific plots based on metadata
    print("\nGenerating plots by category...")

    # Generate all structure-related plots
    structure_plots = {}
    for metric_name, plot_names in available_plots.items():
        metric_structure_plots = []
        for plot_name in plot_names:
            metadata = visualizer.get_plot_metadata(metric_name, plot_name)
            if metadata.category == "structure":
                metric_structure_plots.append(plot_name)

        if metric_structure_plots:
            structure_plots[metric_name] = metric_structure_plots

    if structure_plots:
        structure_figures = visualizer.generate_plots(structure_plots, save=True, show=False)
        total_structure_plots = sum(len(plots) for plots in structure_figures.values())
        print(f"Created {total_structure_plots} structure-related plots")


def main():
    """Run all examples."""
    print("QWARD Visualization Examples")
    print("=" * 40)

    example_basic_visualizer()
    example_with_circuit_performance()
    example_custom_config()
    example_custom_data()
    example_individual_strategies()
    example_custom_strategy()
    example_metric_summary()

    print("\n" + "=" * 40)
    print("All examples completed!")
    print("Check qward/examples/img/ for generated plots")
    print("\nNew API Benefits Demonstrated:")
    print("- Type-safe constants (Metrics.QISKIT, Plots.Qiskit.CIRCUIT_STRUCTURE)")
    print("- Granular plot control with selections")
    print("- Rich metadata for each plot")
    print("- Memory-efficient defaults")
    print("- IDE autocompletion support")
    print("- Flexible plot generation strategies")


if __name__ == "__main__":
    main()
