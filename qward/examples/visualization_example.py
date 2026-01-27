#!/usr/bin/env python3
"""
QWARD Visualization Example

This comprehensive example demonstrates:
1. Unified Visualizer usage
2. Direct strategy usage for fine-grained control
3. Custom configurations
4. Memory-efficient patterns
5. Plot metadata and exploration
"""

import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qward import Scanner, Visualizer
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics
from qward.visualization import (
    PlotConfig,
    QiskitVisualizer,
    ComplexityVisualizer,
    CircuitPerformanceVisualizer,
    VisualizationStrategy,
    PlotMetadata,
    PlotType,
)
from qward.visualization.constants import Metrics, Plots


def create_sample_circuit() -> QuantumCircuit:
    """Create a sample quantum circuit for demonstrations."""
    circuit = QuantumCircuit(3, 3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.rz(0.5, 2)
    circuit.measure_all()
    return circuit


# =============================================================================
# Example 1: Unified Visualizer (Recommended)
# =============================================================================


def unified_visualizer_example():
    """Example: Using the unified Visualizer class (recommended approach)."""
    print("=== Example 1: Unified Visualizer ===\n")

    circuit = create_sample_circuit()
    scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics])

    # Create visualizer from scanner
    visualizer = Visualizer(scanner=scanner, output_dir="qward/examples/img")

    # Print available metrics
    visualizer.print_available_metrics()

    # Generate single plot
    print("\nGenerating single plot...")
    visualizer.generate_plot(
        metric_name=Metrics.QISKIT,
        plot_name=Plots.Qiskit.CIRCUIT_STRUCTURE,
        save=True,
        show=False,
    )
    print("  ✓ Circuit structure plot saved")

    # Generate selected plots
    print("\nGenerating selected plots...")
    selected = visualizer.generate_plots(
        selections={
            Metrics.QISKIT: [Plots.Qiskit.GATE_DISTRIBUTION, Plots.Qiskit.CIRCUIT_SUMMARY],
            Metrics.COMPLEXITY: [Plots.Complexity.COMPLEXITY_RADAR],
        },
        save=True,
        show=False,
    )
    for metric, plots in selected.items():
        print(f"  ✓ {metric}: {len(plots)} plots")

    # Create dashboard
    print("\nCreating dashboards...")
    dashboards = visualizer.create_dashboard(save=True, show=False)
    print(f"  ✓ Created {len(dashboards)} dashboards")


def with_circuit_performance_example():
    """Example: Visualizer with CircuitPerformance metrics."""
    print("\n=== Example 2: With CircuitPerformance ===\n")

    circuit = create_sample_circuit()

    # Run simulations
    simulator = AerSimulator()
    jobs = [simulator.run(circuit, shots=500) for _ in range(3)]

    # Create metrics
    circuit_performance = CircuitPerformanceMetrics(circuit=circuit, jobs=jobs)

    scanner = Scanner(
        circuit=circuit,
        strategies=[QiskitMetrics, ComplexityMetrics, circuit_performance],
    )

    visualizer = Visualizer(scanner=scanner, output_dir="qward/examples/img")

    # Print what's available
    visualizer.print_available_metrics()

    # Create comprehensive dashboard
    dashboards = visualizer.create_dashboard(save=True, show=False)
    print(f"Created {len(dashboards)} dashboards with CircuitPerformance")


# =============================================================================
# Example 2: Direct Strategy Usage
# =============================================================================


def direct_qiskit_strategy():
    """Example: Using QiskitVisualizer directly."""
    print("\n=== Example 3: Direct QiskitVisualizer ===\n")

    circuit = create_sample_circuit()
    scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics])
    metrics_data = scanner.calculate_metrics()

    # Use strategy directly
    strategy = QiskitVisualizer(
        metrics_dict={Metrics.QISKIT: metrics_data[Metrics.QISKIT]},
        output_dir="qward/examples/img/direct_qiskit",
    )

    # Generate individual plots
    strategy.generate_plot(Plots.Qiskit.CIRCUIT_STRUCTURE, save=True, show=False)
    strategy.generate_plot(Plots.Qiskit.GATE_DISTRIBUTION, save=True, show=False)
    print("  ✓ QiskitVisualizer plots saved")

    # Create dashboard
    strategy.create_dashboard(save=True, show=False)
    print("  ✓ QiskitVisualizer dashboard saved")

    # Show available plots
    print("\n  Available QiskitVisualizer plots:")
    for plot_name in strategy.get_available_plots():
        metadata = strategy.get_plot_metadata(plot_name)
        print(f"    - {plot_name}: {metadata.description}")


def direct_complexity_strategy():
    """Example: Using ComplexityVisualizer with custom config."""
    print("\n=== Example 4: Direct ComplexityVisualizer ===\n")

    circuit = create_sample_circuit()
    scanner = Scanner(circuit=circuit, strategies=[ComplexityMetrics])
    metrics_data = scanner.calculate_metrics()

    # Custom configuration
    custom_config = PlotConfig(
        figsize=(14, 10),
        style="quantum",
        color_palette=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
        alpha=0.8,
    )

    strategy = ComplexityVisualizer(
        metrics_dict={Metrics.COMPLEXITY: metrics_data[Metrics.COMPLEXITY]},
        output_dir="qward/examples/img/direct_complexity",
        config=custom_config,
    )

    # Generate specific plots
    strategy.generate_plot(Plots.Complexity.GATE_BASED_METRICS, save=True, show=False)
    strategy.generate_plot(Plots.Complexity.COMPLEXITY_RADAR, save=True, show=False)
    strategy.generate_plot(Plots.Complexity.EFFICIENCY_METRICS, save=True, show=False)
    print("  ✓ ComplexityVisualizer plots with custom config saved")


def direct_performance_strategy():
    """Example: Using CircuitPerformanceVisualizer directly."""
    print("\n=== Example 5: Direct CircuitPerformanceVisualizer ===\n")

    circuit = create_sample_circuit()
    simulator = AerSimulator()

    # Run jobs with different shot counts
    jobs = [
        simulator.run(circuit, shots=500),
        simulator.run(circuit, shots=1000),
        simulator.run(circuit, shots=1500),
    ]

    circuit_performance = CircuitPerformanceMetrics(circuit=circuit, jobs=jobs)
    scanner = Scanner(circuit=circuit, strategies=[circuit_performance])
    metrics_data = scanner.calculate_metrics()

    # Extract CircuitPerformance data
    perf_data = {k: v for k, v in metrics_data.items() if k.startswith("CircuitPerformance")}

    strategy = CircuitPerformanceVisualizer(
        metrics_dict=perf_data,
        output_dir="qward/examples/img/direct_performance",
    )

    # Generate plots
    strategy.generate_plot(Plots.CircuitPerformance.SUCCESS_ERROR_COMPARISON, save=True, show=False)
    strategy.generate_plot(Plots.CircuitPerformance.FIDELITY_COMPARISON, save=True, show=False)
    strategy.create_dashboard(save=True, show=False)
    print("  ✓ CircuitPerformanceVisualizer plots saved")


# =============================================================================
# Example 3: Custom Data and Strategy
# =============================================================================


def custom_data_example():
    """Example: Using custom metrics data."""
    print("\n=== Example 6: Custom Metrics Data ===\n")

    # Create custom data (normally from your own calculations)
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

    custom_metrics = {Metrics.QISKIT: custom_qiskit_data}

    # Create visualizer with custom data
    visualizer = Visualizer(metrics_data=custom_metrics, output_dir="qward/examples/img")
    visualizer.print_available_metrics()

    # Generate plots
    figures = visualizer.generate_plots(
        {Metrics.QISKIT: None},
        save=True,
        show=False,
    )
    print(f"Created {len(figures[Metrics.QISKIT])} plots from custom data")


def custom_strategy_example():
    """Example: Creating a custom visualization strategy."""
    print("\n=== Example 7: Custom Strategy ===\n")

    class CustomMetricsStrategy(VisualizationStrategy):
        """Custom visualization strategy example."""

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

    # Use custom strategy
    custom_data = pd.DataFrame([{"custom.metric": 42}])
    visualizer = Visualizer(
        metrics_data={"CustomMetrics": custom_data},
        output_dir="qward/examples/img",
    )
    visualizer.register_strategy("CustomMetrics", CustomMetricsStrategy)

    figures = visualizer.generate_plots({"CustomMetrics": None}, save=True, show=False)
    print(f"Created {len(figures['CustomMetrics'])} custom plots")


# =============================================================================
# Example 4: Memory Efficiency and Best Practices
# =============================================================================


def memory_efficiency_example():
    """Example: Memory-efficient visualization patterns."""
    print("\n=== Example 8: Memory Efficiency ===\n")

    circuit = create_sample_circuit()
    scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics])
    visualizer = Visualizer(scanner=scanner, output_dir="qward/examples/img")

    # Default behavior: save=False, show=False (memory efficient)
    print("Default behavior (save=False, show=False):")
    initial_figures = len(plt.get_fignums())
    print(f"  Initial open figures: {initial_figures}")

    # Create multiple plots without showing
    dashboards = visualizer.create_dashboard()  # Defaults: save=False, show=False
    all_plots = visualizer.generate_plots(
        selections={Metrics.QISKIT: None, Metrics.COMPLEXITY: None}
    )

    print(f"  Created {len(dashboards)} dashboards (not saved)")
    total_plots = sum(len(plots) for plots in all_plots.values())
    print(f"  Created {total_plots} individual plots (not saved)")

    # Clean up
    plt.close("all")
    print(f"  Figures after cleanup: {len(plt.get_fignums())}")

    # Explicit save when needed
    print("\nExplicit save when needed:")
    visualizer.create_dashboard(save=True)
    saved_plots = visualizer.generate_plots(
        selections={
            Metrics.QISKIT: [Plots.Qiskit.CIRCUIT_STRUCTURE, Plots.Qiskit.GATE_DISTRIBUTION]
        },
        save=True,
    )
    print(f"  Saved dashboard and {len(saved_plots[Metrics.QISKIT])} plots")


def plot_metadata_exploration():
    """Example: Exploring plot metadata."""
    print("\n=== Example 9: Plot Metadata Exploration ===\n")

    circuit = create_sample_circuit()
    scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics])
    visualizer = Visualizer(scanner=scanner)

    # Get metric summary
    print("Metric Summary:")
    summary = visualizer.get_metric_summary()
    for metric_name, info in summary.items():
        print(f"  {metric_name}:")
        print(f"    Shape: {info['shape']}")
        print(f"    Columns: {len(info['columns'])}")

    # Get available plots and metadata
    print("\nAvailable Plots:")
    available = visualizer.get_available_plots()

    for metric_name, plot_names in available.items():
        print(f"\n{metric_name} ({len(plot_names)} plots):")
        for plot_name in plot_names:
            metadata = visualizer.get_plot_metadata(metric_name, plot_name)
            print(f"  - {plot_name}:")
            print(f"      Type: {metadata.plot_type.value}")
            print(f"      Category: {metadata.category}")

    # Generate plots by category
    print("\nGenerating structure-related plots:")
    structure_plots = {}
    for metric_name, plot_names in available.items():
        metric_structure = [
            p
            for p in plot_names
            if visualizer.get_plot_metadata(metric_name, p).category == "structure"
        ]
        if metric_structure:
            structure_plots[metric_name] = metric_structure

    if structure_plots:
        figures = visualizer.generate_plots(structure_plots, save=True, show=False)
        total = sum(len(p) for p in figures.values())
        print(f"  Created {total} structure-related plots")


def main():
    """Run all visualization examples."""
    print("🎨 QWARD Visualization Examples")
    print("=" * 50)

    # Unified Visualizer (recommended)
    unified_visualizer_example()
    with_circuit_performance_example()

    # Direct strategy usage
    direct_qiskit_strategy()
    direct_complexity_strategy()
    direct_performance_strategy()

    # Custom data and strategies
    custom_data_example()
    custom_strategy_example()

    # Best practices
    memory_efficiency_example()
    plot_metadata_exploration()

    print("\n" + "=" * 50)
    print("✅ Visualization examples complete!")
    print("\nKey concepts demonstrated:")
    print("  • Unified Visualizer (recommended approach)")
    print("  • Direct strategy usage for fine control")
    print("  • Custom configurations and strategies")
    print("  • Memory-efficient patterns")
    print("  • Plot metadata exploration")
    print("\n📁 Plots saved to: qward/examples/img/")


if __name__ == "__main__":
    main()
