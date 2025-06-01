"""
Example demonstrating the new unified Visualizer system for QWARD.
"""

from .utils import get_display, create_example_circuit
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics
from qward.visualization import Visualizer, PlotConfig
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

    # Visualize specific metric
    print("Creating QiskitMetrics visualizations with custom config...")
    figures = visualizer.visualize_metric("QiskitMetrics", save=True, show=False)
    print(f"Created {len(figures)} QiskitMetrics plots")


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

    custom_metrics_dict = {"QiskitMetrics": custom_qiskit_data}

    # Create visualizer with custom data
    visualizer = Visualizer(metrics_data=custom_metrics_dict, output_dir="qward/examples/img")

    # Show what's available
    visualizer.print_available_metrics()

    # Create visualizations
    print("Creating visualizations from custom data...")
    all_figures = visualizer.visualize_all(save=True, show=False)
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
    if "QiskitMetrics" in metrics_data:
        print("Creating QiskitMetrics visualizations using QiskitVisualizer strategy...")
        qiskit_strategy = QiskitVisualizer(
            metrics_dict={"QiskitMetrics": metrics_data["QiskitMetrics"]},
            output_dir="qward/examples/img",
        )
        qiskit_figures = qiskit_strategy.plot_all(save=True, show=False)
        print(f"Created {len(qiskit_figures)} QiskitMetrics plots")

    # Use ComplexityVisualizer strategy directly
    if "ComplexityMetrics" in metrics_data:
        print("Creating ComplexityMetrics visualizations using ComplexityVisualizer strategy...")
        complexity_strategy = ComplexityVisualizer(
            metrics_dict={"ComplexityMetrics": metrics_data["ComplexityMetrics"]},
            output_dir="qward/examples/img",
        )
        complexity_figures = complexity_strategy.plot_all(save=True, show=False)
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
        perf_figures = perf_strategy.plot_all(save=True, show=False)
        print(f"Created {len(perf_figures)} CircuitPerformance plots")


def example_custom_strategy():
    """Example: Creating and registering a custom visualization strategy."""
    print("\n=== Example: Custom Visualization Strategy ===")

    from qward.visualization import VisualizationStrategy
    import matplotlib.pyplot as plt
    import pandas as pd

    # Define a custom strategy
    class CustomMetricsStrategy(VisualizationStrategy):
        """Custom visualization strategy for demonstration."""

        def create_dashboard(self, save=True, show=True):
            """Create a custom dashboard."""
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("Custom Metrics Dashboard", fontsize=16)

            # Plot 1: Sample bar chart
            ax1.bar(["A", "B", "C"], [1, 2, 3], color=self.config.color_palette[:3])
            ax1.set_title("Sample Bar Chart")

            # Plot 2: Sample line plot
            ax2.plot([1, 2, 3, 4], [1, 4, 2, 3], color=self.config.color_palette[0])
            ax2.set_title("Sample Line Plot")

            # Plot 3: Sample scatter plot
            ax3.scatter([1, 2, 3], [3, 1, 2], color=self.config.color_palette[1])
            ax3.set_title("Sample Scatter Plot")

            # Plot 4: Text summary
            ax4.text(
                0.5, 0.5, "Custom Strategy\nDemonstration", ha="center", va="center", fontsize=14
            )
            ax4.set_title("Summary")
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)

            plt.tight_layout()

            if save:
                self.save_plot(fig, "custom_strategy_dashboard")
            if show:
                self.show_plot(fig)
            return fig

        def plot_all(self, save=True, show=True):
            """Generate all plots for this strategy."""
            return [self.create_dashboard(save, show)]

    # Create some dummy data for the custom strategy
    custom_data = pd.DataFrame({"metric_a": [1, 2, 3], "metric_b": [4, 5, 6]})

    # Create visualizer and register custom strategy
    visualizer = Visualizer(
        metrics_data={"CustomMetrics": custom_data}, output_dir="qward/examples/img"
    )

    # Register the custom strategy
    visualizer.register_strategy("CustomMetrics", CustomMetricsStrategy)

    print("Registered custom strategy!")
    print(f"Available strategies: {list(visualizer.list_registered_strategies().keys())}")

    # Use the custom strategy
    print("Creating visualizations with custom strategy...")
    custom_figures = visualizer.visualize_metric("CustomMetrics", save=True, show=False)
    print(f"Created {len(custom_figures)} custom plots")


def example_metric_summary():
    """Example: Getting metric summaries."""
    print("\n=== Example: Metric Summary ===")

    # Create circuit and scanner
    circuit = create_example_circuit()
    scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics])

    # Create visualizer
    visualizer = Visualizer(scanner=scanner, output_dir="qward/examples/img")

    # Get summary
    summary = visualizer.get_metric_summary()

    print("Metrics Summary:")
    for metric_name, info in summary.items():
        print(f"\n{metric_name}:")
        print(f"  Shape: {info['shape']}")
        print(f"  Has Data: {info['has_data']}")
        print(f"  Columns: {len(info['columns'])} total")
        if info["sample_values"]:
            print("  Sample values:")
            for key, value in list(info["sample_values"].items())[:3]:  # Show first 3
                print(f"    {key}: {value}")


if __name__ == "__main__":
    print("QWARD Visualizer Examples")
    print("=" * 50)

    # Run examples
    example_basic_visualizer()
    example_with_circuit_performance()
    example_custom_config()
    example_custom_data()
    example_individual_strategies()
    example_custom_strategy()
    example_metric_summary()

    print("\n" + "=" * 50)
    print(
        "All examples completed! Check the 'qward/examples/img/' and 'qward/examples/custom_plots/' directories for generated plots."
    )
