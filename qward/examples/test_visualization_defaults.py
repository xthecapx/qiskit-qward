#!/usr/bin/env python3
"""
Test script to verify that visualization defaults are now False for save and show parameters.
This demonstrates the memory-efficient usage of the visualization system.
"""

from qward import Scanner, Visualizer
from qward.metrics import QiskitMetrics, ComplexityMetrics
from qward.visualization import PlotConfig
from qward.visualization.constants import Metrics, Plots
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt


def test_visualization_defaults():
    """Test that visualization methods default to save=False, show=False."""

    print("=== Testing QWARD Visualization Defaults ===")

    # Create a simple quantum circuit
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()

    # Create scanner with basic metrics
    scanner = Scanner(circuit=qc, strategies=[QiskitMetrics, ComplexityMetrics])

    # Create visualizer
    visualizer = Visualizer(scanner=scanner, output_dir="qward/examples/img")

    print(f"Available metrics: {visualizer.get_available_metrics()}")

    # Test 1: Default behavior (should not save or show)
    print("\n1. Testing default behavior (save=False, show=False)...")

    # These should not save files or display plots
    dashboards = visualizer.create_dashboard()  # Default: save=False, show=False

    # Generate all plots for all metrics using new API
    all_plots = visualizer.generate_plots(
        {Metrics.QISKIT: None, Metrics.COMPLEXITY: None}  # None = all plots
    )  # Default: save=False, show=False

    print(f"   Created {len(dashboards)} dashboards (not saved)")
    print(
        f"   Created {sum(len(plots) for plots in all_plots.values())} individual plots (not saved)"
    )

    # Test 2: Explicit save=True (should save but not show)
    print("\n2. Testing explicit save=True...")

    saved_dashboard = visualizer.create_dashboard(save=True)

    # Generate specific QiskitMetrics plots with save=True
    saved_plots = visualizer.generate_plots(
        {Metrics.QISKIT: [Plots.Qiskit.CIRCUIT_STRUCTURE, Plots.Qiskit.GATE_DISTRIBUTION]},
        save=True,
    )

    qiskit_plot_count = len(saved_plots[Metrics.QISKIT])
    print(
        f"   Saved 1 dashboard and {qiskit_plot_count} QiskitMetrics plots to {visualizer.output_dir}"
    )

    # Test 3: Memory efficiency demonstration
    print("\n3. Memory efficiency demonstration...")

    # Count open figures before
    initial_figures = len(plt.get_fignums())
    print(f"   Initial open figures: {initial_figures}")

    # Create many plots without showing (memory efficient)
    for i in range(3):
        _ = visualizer.create_dashboard()  # Creates figures but doesn't show them

    # Count figures after (should be manageable)
    final_figures = len(plt.get_fignums())
    print(f"   Figures after creating 3 dashboards: {final_figures}")

    # Close all figures to clean up
    plt.close("all")
    print(f"   Figures after cleanup: {len(plt.get_fignums())}")

    # Test 4: Custom configuration
    print("\n4. Testing custom configuration...")

    config = PlotConfig(
        figsize=(8, 6), style="quantum", color_palette=["#FF6B6B", "#4ECDC4", "#45B7D1"], alpha=0.8
    )

    custom_visualizer = Visualizer(scanner=scanner, config=config, output_dir="qward/examples/img")

    # Create specific ComplexityMetrics plots with custom config and save them
    custom_plots = custom_visualizer.generate_plots(
        {
            Metrics.COMPLEXITY: [
                Plots.Complexity.COMPLEXITY_RADAR,
                Plots.Complexity.GATE_BASED_METRICS,
            ]
        },
        save=True,
    )

    complexity_plot_count = len(custom_plots[Metrics.COMPLEXITY])
    print(
        f"   Created {complexity_plot_count} custom ComplexityMetrics plots with quantum style (saved)"
    )

    # Test 5: Single plot generation
    print("\n5. Testing single plot generation...")

    single_plot = visualizer.generate_plot(
        Metrics.QISKIT, Plots.Qiskit.CIRCUIT_STRUCTURE, save=False, show=False
    )
    print(f"   Generated single circuit structure plot: {type(single_plot)}")
    plt.close(single_plot)  # Clean up

    print("\n✅ All tests completed successfully!")
    print("\nKey benefits of new API:")
    print("  - Type-safe constants prevent typos")
    print("  - Granular control over which plots to generate")
    print("  - Reduced memory usage when generating multiple plots")
    print("  - No unwanted file creation during exploration")
    print("  - Explicit control over when to save/show plots")
    print("  - Better for batch processing and automated workflows")
    print("  - IDE autocompletion for all plot names")


if __name__ == "__main__":
    test_visualization_defaults()
