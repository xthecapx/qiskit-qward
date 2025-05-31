"""
Example demonstrating the new unified Visualizer system for QWARD.
"""

from qward.examples.utils import get_display, create_example_circuit
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformance
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
    visualizer = Visualizer(scanner=scanner)

    # Print available metrics
    visualizer.print_available_metrics()

    # Create dashboard for all available metrics
    print("\nCreating dashboard...")
    dashboards = visualizer.create_dashboard(save=True, show=False)

    print(f"Created {len(dashboards)} dashboards:")
    for metric_name, fig in dashboards.items():
        print(f"  - {metric_name}: {fig}")


def example_with_circuit_performance():
    """Example: Using all three visualizers including CircuitPerformance."""
    print("\n=== Example: All Visualizers with CircuitPerformance ===")

    # Create a circuit
    circuit = create_example_circuit()

    # Run circuit on simulator to get jobs for CircuitPerformance
    simulator = AerSimulator()
    jobs = [simulator.run(circuit, shots=500) for _ in range(3)]

    # Create CircuitPerformance with multiple jobs
    circuit_performance = CircuitPerformance(circuit=circuit, jobs=jobs)

    # Create scanner with all metrics
    scanner = Scanner(
        circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics, circuit_performance]
    )

    # Create visualizer
    visualizer = Visualizer(scanner=scanner)

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
    visualizer = Visualizer(scanner=scanner, config=custom_config, output_dir="custom_plots")

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
    visualizer = Visualizer(metrics_data=custom_metrics_dict)

    # Show what's available
    visualizer.print_available_metrics()

    # Create visualizations
    print("Creating visualizations from custom data...")
    all_figures = visualizer.visualize_all(save=True, show=False)
    print(f"Created visualizations for {len(all_figures)} metric types")


def example_individual_visualizers():
    """Example: Using individual metric visualizers directly."""
    print("\n=== Example: Individual Visualizers ===")

    from qward.visualization import (
        QiskitMetricsVisualizer,
        ComplexityMetricsVisualizer,
        CircuitPerformanceVisualizer,
    )

    # Create circuit and get metrics
    circuit = create_example_circuit()

    # Create scanner with all metrics including CircuitPerformance
    simulator = AerSimulator()
    jobs = [simulator.run(circuit, shots=500) for _ in range(2)]
    circuit_performance = CircuitPerformance(circuit=circuit, jobs=jobs)

    scanner = Scanner(
        circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics, circuit_performance]
    )
    metrics_data = scanner.calculate_metrics()

    # Use QiskitMetrics visualizer directly
    if "QiskitMetrics" in metrics_data:
        print("Creating QiskitMetrics visualizations...")
        qiskit_viz = QiskitMetricsVisualizer(
            metrics_dict={"QiskitMetrics": metrics_data["QiskitMetrics"]}
        )
        qiskit_figures = qiskit_viz.plot_all(save=True, show=False)
        print(f"Created {len(qiskit_figures)} QiskitMetrics plots")

    # Use ComplexityMetrics visualizer directly
    if "ComplexityMetrics" in metrics_data:
        print("Creating ComplexityMetrics visualizations...")
        complexity_viz = ComplexityMetricsVisualizer(
            metrics_dict={"ComplexityMetrics": metrics_data["ComplexityMetrics"]}
        )
        complexity_figures = complexity_viz.plot_all(save=True, show=False)
        print(f"Created {len(complexity_figures)} ComplexityMetrics plots")

    # Use CircuitPerformance visualizer directly
    circuit_perf_data = {
        k: v for k, v in metrics_data.items() if k.startswith("CircuitPerformance")
    }
    if circuit_perf_data:
        print("Creating CircuitPerformance visualizations...")
        perf_viz = CircuitPerformanceVisualizer(metrics_dict=circuit_perf_data)
        perf_figures = perf_viz.plot_all(save=True, show=False)
        print(f"Created {len(perf_figures)} CircuitPerformance plots")


def example_metric_summary():
    """Example: Getting metric summaries."""
    print("\n=== Example: Metric Summary ===")

    # Create circuit and scanner
    circuit = create_example_circuit()
    scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics])

    # Create visualizer
    visualizer = Visualizer(scanner=scanner)

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
    example_individual_visualizers()
    example_metric_summary()

    print("\n" + "=" * 50)
    print(
        "All examples completed! Check the 'img/' and 'custom_plots/' directories for generated plots."
    )
