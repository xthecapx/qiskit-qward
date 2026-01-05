#!/usr/bin/env python3
"""
QWARD Metric-Specific Visualizers Example

This example demonstrates visualization for each metric type:
1. StructuralMetrics
2. BehavioralMetrics
3. QuantumSpecificMetrics

Each metric type has its own visualization strategy with specific plots.
"""

from qiskit import QuantumCircuit
from qward import Scanner
from qward.metrics import (
    StructuralMetrics,
    BehavioralMetrics,
    QuantumSpecificMetrics,
)
from qward.visualization import Visualizer, PlotConfig
from qward.visualization.constants import Metrics, Plots


def create_demo_circuit() -> QuantumCircuit:
    """Create a circuit that demonstrates various quantum properties."""
    qc = QuantumCircuit(4, 4)

    # Superposition
    qc.h(0)
    qc.h(1)

    # Entanglement
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cz(2, 3)

    # Parameterized gates
    qc.ry(0.5, 2)
    qc.rz(0.3, 3)

    # Measurement
    qc.measure_all()

    return qc


# =============================================================================
# Structural Metrics Visualization
# =============================================================================


def structural_metrics_example():
    """Demonstrate StructuralMetrics visualization."""
    print("=== Structural Metrics Visualization ===\n")

    circuit = create_demo_circuit()
    print(f"Circuit: {circuit.num_qubits} qubits, depth {circuit.depth()}")

    # Create scanner with StructuralMetrics
    scanner = Scanner(circuit=circuit)
    scanner.add_strategy(StructuralMetrics(circuit))

    metrics = scanner.calculate_metrics()
    print(f"Calculated metrics: {list(metrics.keys())}")

    # Create visualizer
    viz = Visualizer(scanner=scanner, output_dir="qward/examples/img/structural")

    # Generate single plot
    viz.generate_plot(
        metric_name=Metrics.STRUCTURAL,
        plot_name=Plots.Structural.SUMMARY,
        save=True,
        show=False,
    )
    print("  ✓ Structural summary plot saved")

    # Generate all structural plots
    all_plots = viz.generate_plots(
        selections={Metrics.STRUCTURAL: None},
        save=True,
        show=False,
    )

    if Metrics.STRUCTURAL in all_plots:
        print(f"  ✓ Generated {len(all_plots[Metrics.STRUCTURAL])} structural plots")

    # Dashboard
    dashboards = viz.create_dashboard(save=True, show=False)
    print(f"  ✓ Created {len(dashboards)} dashboard(s)")

    return scanner, metrics


# =============================================================================
# Behavioral Metrics Visualization
# =============================================================================


def behavioral_metrics_example():
    """Demonstrate BehavioralMetrics visualization."""
    print("\n=== Behavioral Metrics Visualization ===\n")

    circuit = create_demo_circuit()

    # Create scanner with BehavioralMetrics
    scanner = Scanner(circuit=circuit)
    scanner.add_strategy(BehavioralMetrics(circuit))

    metrics = scanner.calculate_metrics()
    print(f"Calculated metrics: {list(metrics.keys())}")

    # Create visualizer
    viz = Visualizer(scanner=scanner, output_dir="qward/examples/img/behavioral")

    # Generate single plot
    viz.generate_plot(
        metric_name=Metrics.BEHAVIORAL,
        plot_name=Plots.Behavioral.SUMMARY,
        save=True,
        show=False,
    )
    print("  ✓ Behavioral summary plot saved")

    # Generate all behavioral plots
    all_plots = viz.generate_plots(
        selections={Metrics.BEHAVIORAL: None},
        save=True,
        show=False,
    )

    if Metrics.BEHAVIORAL in all_plots:
        print(f"  ✓ Generated {len(all_plots[Metrics.BEHAVIORAL])} behavioral plots")

    # Dashboard
    dashboards = viz.create_dashboard(save=True, show=False)
    print(f"  ✓ Created {len(dashboards)} dashboard(s)")

    return scanner, metrics


# =============================================================================
# Quantum-Specific Metrics Visualization
# =============================================================================


def quantum_specific_metrics_example():
    """Demonstrate QuantumSpecificMetrics visualization."""
    print("\n=== Quantum-Specific Metrics Visualization ===\n")

    circuit = create_demo_circuit()

    # Create scanner with QuantumSpecificMetrics
    scanner = Scanner(circuit=circuit)
    scanner.add_strategy(QuantumSpecificMetrics(circuit))

    metrics = scanner.calculate_metrics()
    print(f"Calculated metrics: {list(metrics.keys())}")

    # Create visualizer with quantum style
    config = PlotConfig(style="quantum")
    viz = Visualizer(
        scanner=scanner,
        config=config,
        output_dir="qward/examples/img/quantum_specific",
    )

    # Print available plots
    viz.print_available_metrics()

    # Generate dashboard
    dashboards = viz.create_dashboard(save=True, show=False)
    print(f"  ✓ Created {len(dashboards)} dashboard(s)")

    # Generate individual plots if available
    try:
        viz.generate_plot(
            metric_name="QuantumSpecificMetrics",
            plot_name="all_metrics_bar",
            save=True,
            show=False,
        )
        print("  ✓ All metrics bar plot saved")

        viz.generate_plot(
            metric_name="QuantumSpecificMetrics",
            plot_name="quantum_radar",
            save=True,
            show=False,
        )
        print("  ✓ Quantum radar plot saved")
    except Exception as e:
        print(f"  Note: Some plots not available: {e}")

    return scanner, metrics


# =============================================================================
# Combined Example
# =============================================================================


def combined_metrics_example():
    """Demonstrate all metric types together."""
    print("\n=== Combined Metrics Visualization ===\n")

    circuit = create_demo_circuit()

    # Create scanner with all metric types
    scanner = Scanner(
        circuit=circuit,
        strategies=[
            StructuralMetrics,
            BehavioralMetrics,
            QuantumSpecificMetrics,
        ],
    )

    metrics = scanner.calculate_metrics()
    print(f"Calculated {len(metrics)} metric types:")
    for name in metrics.keys():
        print(f"  - {name}")

    # Display summary
    scanner.display_summary(metrics)

    # Create visualizer
    viz = Visualizer(scanner=scanner, output_dir="qward/examples/img/combined")

    # Print all available metrics
    print("\nAvailable metrics and plots:")
    viz.print_available_metrics()

    # Create all dashboards
    dashboards = viz.create_dashboard(save=True, show=False)
    print(f"\n✓ Created {len(dashboards)} dashboards")

    # Generate specific plots from each metric type
    selections = {}

    if Metrics.STRUCTURAL in viz.get_available_metrics():
        selections[Metrics.STRUCTURAL] = [Plots.Structural.SUMMARY]

    if Metrics.BEHAVIORAL in viz.get_available_metrics():
        selections[Metrics.BEHAVIORAL] = [Plots.Behavioral.SUMMARY]

    if selections:
        selected_plots = viz.generate_plots(selections=selections, save=True, show=False)
        total = sum(len(p) for p in selected_plots.values())
        print(f"✓ Generated {total} selected plots")

    return scanner, metrics


def main():
    """Run all metric visualizer examples."""
    print("📊 QWARD Metric-Specific Visualizers")
    print("=" * 50)

    # Individual metric types
    structural_metrics_example()
    behavioral_metrics_example()
    quantum_specific_metrics_example()

    # Combined
    combined_metrics_example()

    print("\n" + "=" * 50)
    print("✅ Metric visualizer examples complete!")
    print("\nMetric types demonstrated:")
    print("  • StructuralMetrics - circuit structure analysis")
    print("  • BehavioralMetrics - gate behavior patterns")
    print("  • QuantumSpecificMetrics - quantum properties")
    print("\n📁 Plots saved to:")
    print("  • qward/examples/img/structural/")
    print("  • qward/examples/img/behavioral/")
    print("  • qward/examples/img/quantum_specific/")
    print("  • qward/examples/img/combined/")


if __name__ == "__main__":
    main()

