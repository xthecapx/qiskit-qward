#!/usr/bin/env python3
"""
QWARD Quickstart Example

This example demonstrates the essential QWARD workflow:
1. Create a quantum circuit
2. Calculate metrics using Scanner
3. Visualize results with the Visualizer API

This is the recommended starting point for new users.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner, Visualizer
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics
from qward.visualization.constants import Metrics, Plots
from qward.visualization import PlotConfig


def create_bell_circuit() -> QuantumCircuit:
    """Create a Bell state circuit."""
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    return circuit


def fluent_api_example():
    """Example 0: New fluent API."""
    print("=== Example 0: Fluent API ===\n")

    circuit = create_bell_circuit()

    result = Scanner(circuit).scan()
    print(f"Metrics calculated: {list(result.keys())}")

    result.summary()
    result.visualize(save=True, show=False, output_dir="qward/examples/img/fluent")
    print("Plots saved to qward/examples/img/fluent/")


def fluent_performance_example():
    """Example 0b: Fluent API with CircuitPerformanceMetrics."""
    print("\n=== Example 0b: Fluent API + Performance ===\n")

    circuit = create_bell_circuit()
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=1024)

    def bell_success(outcome):
        return outcome.replace(" ", "") in ["00", "11"]

    result = (
        Scanner(circuit)
        .add(CircuitPerformanceMetrics, job=job, success_criteria=bell_success)
        .scan()
        .summary()
    )

    print(f"Metrics: {list(result.keys())}")


def basic_metrics_example():
    """Example 1: Basic metrics calculation."""
    print("=== Example 1: Basic Metrics ===\n")

    # Create circuit
    circuit = create_bell_circuit()
    print(f"Created circuit with {circuit.num_qubits} qubits")

    # Create scanner with metrics strategies
    scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics])

    # Calculate metrics
    metrics_dict = scanner.calculate_metrics()

    # Display summary
    scanner.display_summary(metrics_dict)


def visualization_example():
    """Example 2: Visualization with the Visualizer API."""
    print("\n=== Example 2: Visualization ===\n")

    # Create a more complex circuit
    circuit = QuantumCircuit(4, 4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)
    circuit.ry(0.5, 1)
    circuit.rz(0.3, 2)
    circuit.measure_all()

    print(f"Created circuit with {circuit.num_qubits} qubits and depth {circuit.depth()}")

    # Create scanner and visualizer
    scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics])
    visualizer = Visualizer(scanner=scanner, output_dir="qward/examples/img")

    # Generate a single specific plot
    print("\n1. Generating single plot...")
    visualizer.generate_plot(
        metric_name=Metrics.QISKIT,
        plot_name=Plots.Qiskit.CIRCUIT_STRUCTURE,
        save=True,
        show=False,
    )
    print("   ✓ Circuit structure plot saved")

    # Generate selected plots
    print("\n2. Generating selected plots...")
    selected_plots = visualizer.generate_plots(
        selections={
            Metrics.QISKIT: [
                Plots.Qiskit.CIRCUIT_STRUCTURE,
                Plots.Qiskit.GATE_DISTRIBUTION,
            ],
            Metrics.COMPLEXITY: [
                Plots.Complexity.GATE_BASED_METRICS,
                Plots.Complexity.COMPLEXITY_RADAR,
            ],
        },
        save=True,
        show=False,
    )

    for metric, plots in selected_plots.items():
        print(f"   ✓ {metric}: {len(plots)} plots generated")

    # Generate all plots for a metric
    print("\n3. Generating all ComplexityMetrics plots...")
    all_complexity = visualizer.generate_plots(
        selections={Metrics.COMPLEXITY: None},  # None = all plots
        save=True,
        show=False,
    )
    print(f"   ✓ Generated {len(all_complexity[Metrics.COMPLEXITY])} plots")

    # Create dashboard
    print("\n4. Creating dashboard...")
    dashboards = visualizer.create_dashboard(save=True, show=False)
    print(f"   ✓ Created {len(dashboards)} dashboards")


def circuit_performance_example():
    """Example 3: CircuitPerformance metrics with simulation."""
    print("\n=== Example 3: Circuit Performance ===\n")

    # Create circuit
    circuit = create_bell_circuit()

    # Run simulations
    simulator = AerSimulator()
    jobs = [simulator.run(circuit, shots=1024) for _ in range(3)]

    # Wait for completion
    for job in jobs:
        job.result()

    # Define success criteria for Bell state
    def bell_success(outcome):
        """Success: |00⟩ or |11⟩"""
        clean = outcome.replace(" ", "")
        return clean in ["00", "11"]

    # Create CircuitPerformance metrics
    circuit_performance = CircuitPerformanceMetrics(
        circuit=circuit,
        jobs=jobs,
        success_criteria=bell_success,
    )

    # Create scanner
    scanner = Scanner(circuit=circuit)
    scanner.add_strategy(circuit_performance)

    # Calculate and display metrics
    metrics_dict = scanner.calculate_metrics()
    scanner.display_summary(metrics_dict)

    # Visualize
    visualizer = Visualizer(scanner=scanner, output_dir="qward/examples/img/quickstart")

    # Generate performance plots
    visualizer.generate_plots(
        selections={Metrics.CIRCUIT_PERFORMANCE: None},
        save=True,
        show=False,
    )
    print("✓ Performance plots saved to qward/examples/img/quickstart/")

    # Show aggregate success rate
    aggregate = metrics_dict.get("CircuitPerformance.aggregate")
    if aggregate is not None and not aggregate.empty:
        mean_success = aggregate.iloc[0]["success_metrics.mean_success_rate"]
        print(f"\nMean success rate: {mean_success:.3f}")


def explore_available_plots():
    """Example 4: Exploring available plots and metadata."""
    print("\n=== Example 4: Exploring Available Plots ===\n")

    circuit = create_bell_circuit()
    scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics])
    visualizer = Visualizer(scanner=scanner)

    # Get available plots
    available = visualizer.get_available_plots()

    for metric_name, plot_names in available.items():
        print(f"{metric_name} ({len(plot_names)} plots):")
        for plot_name in plot_names:
            metadata = visualizer.get_plot_metadata(metric_name, plot_name)
            print(f"  - {plot_name}: {metadata.description}")
        print()


def custom_config_example():
    """Example 5: Custom plot configuration."""
    print("\n=== Example 5: Custom Configuration ===\n")

    circuit = create_bell_circuit()
    scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics])

    # Create custom config
    custom_config = PlotConfig(
        figsize=(12, 8),
        style="quantum",
        color_palette=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
        alpha=0.8,
        dpi=300,
    )

    # Create visualizer with custom config
    visualizer = Visualizer(
        scanner=scanner,
        config=custom_config,
        output_dir="qward/examples/img/custom",
    )

    # Generate plots with custom styling
    visualizer.generate_plot(
        metric_name=Metrics.COMPLEXITY,
        plot_name=Plots.Complexity.COMPLEXITY_RADAR,
        save=True,
        show=False,
    )
    print("✓ Custom-styled radar chart saved")


def main():
    """Run all quickstart examples."""
    print("🚀 QWARD Quickstart Examples")
    print("=" * 50)

    fluent_api_example()
    fluent_performance_example()
    basic_metrics_example()
    visualization_example()
    circuit_performance_example()
    explore_available_plots()
    custom_config_example()

    print("\n" + "=" * 50)
    print("✅ Quickstart complete!")
    print("\nKey QWARD features demonstrated:")
    print("  • Scanner for metrics calculation")
    print("  • Visualizer for plot generation")
    print("  • Type-safe constants (Metrics.*, Plots.*)")
    print("  • CircuitPerformance for execution analysis")
    print("  • Custom plot configuration")
    print("\n📁 Plots saved to: qward/examples/img/")


if __name__ == "__main__":
    main()
