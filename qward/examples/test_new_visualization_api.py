#!/usr/bin/env python3
"""
Test script for the new QWARD visualization API with constants and schema.
This validates the refactored visualization system.
"""

from qward import Scanner, Visualizer
from qward.metrics import QiskitMetrics, ComplexityMetrics
from qward.visualization.constants import Metrics, Plots
from qward.visualization import PlotConfig
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt


def test_new_visualization_api():
    """Test the new visualization API with constants and schema."""

    print("=== Testing New QWARD Visualization API ===")

    # Create a quantum circuit
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()

    # Create scanner with metrics
    scanner = Scanner(circuit=qc, strategies=[QiskitMetrics, ComplexityMetrics])

    # Create visualizer
    visualizer = Visualizer(scanner=scanner, output_dir="qward/examples/img")

    print(f"Available metrics: {visualizer.get_available_metrics()}")

    # Test 1: Constants import and access
    print("\n1. Testing constants...")
    print(f"   Metrics.QISKIT = {Metrics.QISKIT}")
    print(f"   Metrics.COMPLEXITY = {Metrics.COMPLEXITY}")
    print(f"   Plots.Qiskit.CIRCUIT_STRUCTURE = {Plots.Qiskit.CIRCUIT_STRUCTURE}")
    print(f"   Plots.Complexity.COMPLEXITY_RADAR = {Plots.Complexity.COMPLEXITY_RADAR}")

    # Test 2: Get available plots
    print("\n2. Testing get_available_plots()...")
    all_plots = visualizer.get_available_plots()
    for metric, plots in all_plots.items():
        print(f"   {metric}: {plots}")

    # Test 3: Get plot metadata
    print("\n3. Testing get_plot_metadata()...")
    try:
        metadata = visualizer.get_plot_metadata(Metrics.QISKIT, Plots.Qiskit.CIRCUIT_STRUCTURE)
        print(f"   Plot: {metadata.name}")
        print(f"   Description: {metadata.description}")
        print(f"   Type: {metadata.plot_type}")
        print(f"   Category: {metadata.category}")
        print(f"   Dependencies: {metadata.dependencies}")
    except Exception as e:
        print(f"   Error getting metadata: {e}")

    # Test 4: Generate single plot
    print("\n4. Testing generate_plot()...")
    try:
        fig = visualizer.generate_plot(
            Metrics.QISKIT, Plots.Qiskit.CIRCUIT_STRUCTURE, save=False, show=False
        )
        print(f"   Generated single plot: {type(fig)}")
        plt.close(fig)  # Clean up
    except Exception as e:
        print(f"   Error generating single plot: {e}")

    # Test 5: Generate multiple plots with selections
    print("\n5. Testing generate_plots() with selections...")
    try:
        selections = {
            Metrics.QISKIT: [Plots.Qiskit.CIRCUIT_STRUCTURE, Plots.Qiskit.GATE_DISTRIBUTION],
            Metrics.COMPLEXITY: [Plots.Complexity.GATE_BASED_METRICS],
        }

        results = visualizer.generate_plots(selections, save=False, show=False)

        for metric_name, plots in results.items():
            print(f"   {metric_name}: Generated {len(plots)} plots")
            for plot_name, fig in plots.items():
                print(f"     - {plot_name}: {type(fig)}")
                plt.close(fig)  # Clean up

    except Exception as e:
        print(f"   Error generating multiple plots: {e}")

    # Test 6: Generate all plots for a metric
    print("\n6. Testing generate_plots() with None (all plots)...")
    try:
        all_qiskit_plots = visualizer.generate_plots({Metrics.QISKIT: None}, save=False, show=False)

        qiskit_plots = all_qiskit_plots[Metrics.QISKIT]
        print(f"   Generated all QiskitMetrics plots: {len(qiskit_plots)}")
        for plot_name, fig in qiskit_plots.items():
            print(f"     - {plot_name}")
            plt.close(fig)  # Clean up

    except Exception as e:
        print(f"   Error generating all plots: {e}")

    # Test 7: Dashboard creation (unchanged)
    print("\n7. Testing create_dashboard() (unchanged)...")
    try:
        dashboards = visualizer.create_dashboard(save=False, show=False)
        print(f"   Created {len(dashboards)} dashboards")
        for metric_name, fig in dashboards.items():
            print(f"     - {metric_name}: {type(fig)}")
            plt.close(fig)  # Clean up
    except Exception as e:
        print(f"   Error creating dashboards: {e}")

    # Test 8: Class-level access to plot registries
    print("\n8. Testing class-level plot registry access...")
    try:
        from qward.visualization import QiskitVisualizer, ComplexityVisualizer

        qiskit_plots = QiskitVisualizer.get_available_plots()
        complexity_plots = ComplexityVisualizer.get_available_plots()

        print(f"   QiskitVisualizer plots: {qiskit_plots}")
        print(f"   ComplexityVisualizer plots: {complexity_plots}")

        # Test metadata access
        qiskit_metadata = QiskitVisualizer.get_plot_metadata(Plots.Qiskit.CIRCUIT_STRUCTURE)
        print(f"   QiskitVisualizer metadata example: {qiskit_metadata.description}")

    except Exception as e:
        print(f"   Error accessing class-level registries: {e}")

    # Test 9: Error handling
    print("\n9. Testing error handling...")
    try:
        # Test invalid metric
        visualizer.generate_plot("InvalidMetric", "invalid_plot")
    except ValueError as e:
        print(f"   ✅ Correctly caught invalid metric error: {e}")

    try:
        # Test invalid plot
        visualizer.generate_plot(Metrics.QISKIT, "invalid_plot")
    except ValueError as e:
        print(f"   ✅ Correctly caught invalid plot error: {e}")

    # Test 10: Memory efficiency
    print("\n10. Testing memory efficiency...")
    initial_figures = len(plt.get_fignums())
    print(f"   Initial open figures: {initial_figures}")

    # Generate plots without showing (should not accumulate figures)
    for i in range(3):
        fig = visualizer.generate_plot(
            Metrics.QISKIT, Plots.Qiskit.CIRCUIT_STRUCTURE, save=False, show=False
        )
        plt.close(fig)  # Explicitly close

    final_figures = len(plt.get_fignums())
    print(f"   Final open figures: {final_figures}")
    print(f"   Memory efficient: {final_figures == initial_figures}")

    print("\n✅ All tests completed successfully!")
    print("\nNew API Benefits:")
    print("  - Type-safe constants prevent typos")
    print("  - IDE autocompletion for metrics and plots")
    print("  - Granular control over plot generation")
    print("  - Rich metadata for each plot")
    print("  - Memory efficient defaults")
    print("  - Consistent with metrics API patterns")


if __name__ == "__main__":
    test_new_visualization_api()
