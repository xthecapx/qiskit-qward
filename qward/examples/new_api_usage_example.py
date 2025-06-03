#!/usr/bin/env python3
"""
Example demonstrating the new QWARD visualization API with constants.
This shows practical usage patterns for the refactored system.
"""

from qward import Scanner, Visualizer
from qward.metrics import QiskitMetrics, ComplexityMetrics
from qward.visualization.constants import Metrics, Plots
from qward.visualization import PlotConfig
from qiskit import QuantumCircuit


def main():
    """Demonstrate practical usage of the new visualization API."""

    print("=== QWARD New Visualization API Usage Example ===")

    # Create a more complex quantum circuit
    qc = QuantumCircuit(4, 4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.ry(0.5, 1)
    qc.rz(0.3, 2)
    qc.measure_all()

    print(f"Created circuit with {qc.num_qubits} qubits and {len(qc)} gates")

    # Create scanner and visualizer
    scanner = Scanner(circuit=qc, strategies=[QiskitMetrics, ComplexityMetrics])
    visualizer = Visualizer(scanner=scanner, output_dir="qward/examples/img")

    # Example 1: Generate a single specific plot
    print("\n1. Generating single plot...")
    single_plot = visualizer.generate_plot(
        metric_name=Metrics.QISKIT, plot_name=Plots.Qiskit.CIRCUIT_STRUCTURE, save=True, show=False
    )
    print("   ✓ Single circuit structure plot generated")

    # Example 2: Generate selected plots for analysis
    print("\n2. Generating selected plots for analysis...")
    selected_plots = visualizer.generate_plots(
        selections={
            Metrics.QISKIT: [
                Plots.Qiskit.CIRCUIT_STRUCTURE,
                Plots.Qiskit.GATE_DISTRIBUTION,
                Plots.Qiskit.CIRCUIT_SUMMARY,
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
        print(f"   ✅ {metric}: Generated {len(plots)} plots")

    # Example 3: Generate all plots for a specific metric
    print("\n3. Generating all ComplexityMetrics plots...")
    all_complexity_plots = visualizer.generate_plots(
        selections={Metrics.COMPLEXITY: None}, save=True, show=False  # None = all plots
    )

    complexity_plots = all_complexity_plots[Metrics.COMPLEXITY]
    print(f"   ✅ Generated all {len(complexity_plots)} ComplexityMetrics plots")

    # Example 4: Get plot information before generating
    print("\n4. Exploring available plots and metadata...")
    available_plots = visualizer.get_available_plots()

    for metric_name, plot_names in available_plots.items():
        print(f"   {metric_name} has {len(plot_names)} plots:")
        for plot_name in plot_names:
            metadata = visualizer.get_plot_metadata(metric_name, plot_name)
            print(f"     - {plot_name}: {metadata.description} ({metadata.plot_type.value})")

    # Example 5: Custom configuration with selected plots
    print("\n5. Using custom configuration...")
    custom_config = PlotConfig(
        figsize=(12, 8),
        style="quantum",
        color_palette=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
        alpha=0.8,
        dpi=300,
    )

    custom_visualizer = Visualizer(
        scanner=scanner, config=custom_config, output_dir="qward/examples/img/custom"
    )

    # Generate a radar chart with custom styling
    radar_fig = custom_visualizer.generate_plot(
        metric_name=Metrics.COMPLEXITY,
        plot_name=Plots.Complexity.COMPLEXITY_RADAR,
        save=True,
        show=False,
    )
    print(f"   ✅ Generated custom-styled radar chart")

    # Example 6: Dashboard creation (unchanged API)
    print("\n6. Creating comprehensive dashboards...")
    dashboards = visualizer.create_dashboard(save=True, show=False)
    print(f"   ✅ Created {len(dashboards)} dashboards")

    # Example 7: Practical workflow - circuit comparison
    print("\n7. Practical workflow - comparing circuit variants...")

    # Create a variant circuit
    qc_variant = QuantumCircuit(4, 4)
    qc_variant.h([0, 1, 2, 3])  # More parallel gates
    qc_variant.cx(0, 2)
    qc_variant.cx(1, 3)
    qc_variant.measure_all()

    # Analyze variant
    variant_scanner = Scanner(circuit=qc_variant, strategies=[ComplexityMetrics])
    variant_visualizer = Visualizer(
        scanner=variant_scanner, output_dir="qward/examples/img/variant"
    )

    # Generate comparison plots
    comparison_plots = {
        Metrics.COMPLEXITY: [
            Plots.Complexity.GATE_BASED_METRICS,
            Plots.Complexity.EFFICIENCY_METRICS,
        ]
    }

    variant_results = variant_visualizer.generate_plots(
        selections=comparison_plots, save=True, show=False
    )

    print(f"   ✅ Generated comparison plots for circuit variant")

    # Summary
    print("\n=== Summary ===")
    print("✅ New API provides:")
    print("   - Type-safe constants (Metrics.QISKIT, Plots.Qiskit.CIRCUIT_STRUCTURE)")
    print("   - Granular plot control (single plots, selected plots, all plots)")
    print("   - Rich metadata for each plot")
    print("   - Memory-efficient defaults (save=False, show=False)")
    print("   - Consistent API patterns")
    print("   - Easy circuit comparison workflows")
    print("   - Unchanged dashboard functionality")

    print(f"\n📁 All plots saved to: qward/examples/img/")


if __name__ == "__main__":
    main()
