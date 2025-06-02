#!/usr/bin/env python3
"""
Example demonstrating direct usage of QWARD visualization strategies.

This example shows how to use individual visualization strategies directly,
without going through the unified Visualizer class. This approach is useful
when you need fine-grained control over specific visualizations or want to
integrate QWARD visualizations into your own custom workflows.
"""

import pandas as pd
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics
from qward.visualization import (
    CircuitPerformanceVisualizer,
    QiskitVisualizer,
    ComplexityVisualizer,
    PlotConfig,
)
from qward.visualization.constants import Metrics, Plots


def create_sample_circuit():
    """Create a sample quantum circuit for demonstration."""
    circuit = QuantumCircuit(3, 3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.rz(0.5, 2)
    circuit.measure_all()
    return circuit


def example_qiskit_strategy_direct():
    """Example: Using QiskitVisualizer visualization strategy directly."""
    print("\n=== Example: Direct QiskitVisualizer Strategy Usage ===")

    # Create circuit and calculate QiskitMetrics
    circuit = create_sample_circuit()
    scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics])
    metrics_data = scanner.calculate_metrics()

    # Use QiskitVisualizer strategy directly
    qiskit_strategy = QiskitVisualizer(
        metrics_dict={Metrics.QISKIT: metrics_data[Metrics.QISKIT]},
        output_dir="qward/examples/img/direct_qiskit",
    )

    print("Creating individual QiskitVisualizer plots using new API...")

    # Generate individual plots using new API with type-safe constants
    print("Generating individual QiskitMetrics plots...")
    qiskit_strategy.generate_plot(Plots.Qiskit.CIRCUIT_STRUCTURE, save=True, show=False)
    print(f"   ✓ Circuit structure plot saved")
    qiskit_strategy.generate_plot(Plots.Qiskit.GATE_DISTRIBUTION, save=True, show=False)
    print(f"   ✓ Gate distribution plot saved")
    qiskit_strategy.generate_plot(Plots.Qiskit.INSTRUCTION_METRICS, save=True, show=False)
    print(f"   ✓ Instruction metrics plot saved")
    qiskit_strategy.generate_plot(Plots.Qiskit.CIRCUIT_SUMMARY, save=True, show=False)
    print(f"   ✓ Circuit summary plot saved")

    # Create dashboard
    qiskit_strategy.create_dashboard(save=True, show=False)
    print("  ✅ QiskitVisualizer dashboard created")

    # Show available plots and metadata
    print("  Available QiskitVisualizer plots:")
    for plot_name in qiskit_strategy.get_available_plots():
        metadata = qiskit_strategy.get_plot_metadata(plot_name)
        print(f"    - {plot_name}: {metadata.description}")


def example_complexity_strategy_direct():
    """Example: Using ComplexityVisualizer visualization strategy directly."""
    print("\n=== Example: Direct ComplexityVisualizer Strategy Usage ===")

    # Create circuit and calculate ComplexityMetrics
    circuit = create_sample_circuit()
    scanner = Scanner(circuit=circuit, strategies=[ComplexityMetrics])
    metrics_data = scanner.calculate_metrics()

    # Use ComplexityVisualizer strategy directly with custom config
    custom_config = PlotConfig(
        figsize=(14, 10),
        style="quantum",
        color_palette=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"],
        alpha=0.8,
    )

    complexity_strategy = ComplexityVisualizer(
        metrics_dict={Metrics.COMPLEXITY: metrics_data[Metrics.COMPLEXITY]},
        output_dir="qward/examples/img/direct_complexity",
        config=custom_config,
    )

    print("Creating individual ComplexityVisualizer plots using new API...")

    # Create specific plots using new API
    complexity_strategy.generate_plot(Plots.Complexity.GATE_BASED_METRICS, save=True, show=False)
    print("  ✅ Gate-based metrics plot created")

    complexity_strategy.generate_plot(Plots.Complexity.COMPLEXITY_RADAR, save=True, show=False)
    print("  ✅ Complexity radar chart created")

    complexity_strategy.generate_plot(Plots.Complexity.EFFICIENCY_METRICS, save=True, show=False)
    print("  ✅ Efficiency metrics plot created")

    # Create dashboard
    complexity_strategy.create_dashboard(save=True, show=False)
    print("  ✅ ComplexityVisualizer dashboard created")

    # Show available plots and metadata
    print("  Available ComplexityVisualizer plots:")
    for plot_name in complexity_strategy.get_available_plots():
        metadata = complexity_strategy.get_plot_metadata(plot_name)
        print(f"    - {plot_name}: {metadata.description} ({metadata.plot_type.value})")


def example_circuit_performance_strategy_direct():
    """Example: Using CircuitPerformanceVisualizer visualization strategy directly."""
    print("\n=== Example: Direct CircuitPerformanceVisualizer Strategy Usage ===")

    # Create circuit and run multiple jobs
    circuit = create_sample_circuit()
    simulator = AerSimulator()

    # Run multiple jobs with different shot counts
    jobs = []
    shot_counts = [500, 1000, 1500]

    print("Running quantum jobs...")
    for shots in shot_counts:
        job = simulator.run(circuit, shots=shots)
        jobs.append(job)
        print(f"  Job with {shots} shots submitted")

    # Wait for jobs and create CircuitPerformanceMetrics metrics
    circuit_performance = CircuitPerformanceMetrics(circuit=circuit, jobs=jobs)
    scanner = Scanner(circuit=circuit, strategies=[circuit_performance])
    metrics_data = scanner.calculate_metrics()

    # Extract CircuitPerformance data
    circuit_perf_data = {
        k: v for k, v in metrics_data.items() if k.startswith("CircuitPerformance")
    }

    # Use CircuitPerformanceVisualizer strategy directly
    perf_strategy = CircuitPerformanceVisualizer(
        metrics_dict=circuit_perf_data, output_dir="qward/examples/img/direct_performance"
    )

    print("Creating individual CircuitPerformanceVisualizer plots using new API...")

    # Create specific plots using new API
    perf_strategy.generate_plot(
        Plots.CircuitPerformance.SUCCESS_ERROR_COMPARISON, save=True, show=False
    )
    print("  ✅ Success vs error comparison plot created")

    perf_strategy.generate_plot(Plots.CircuitPerformance.FIDELITY_COMPARISON, save=True, show=False)
    print("  ✅ Fidelity comparison plot created")

    perf_strategy.generate_plot(Plots.CircuitPerformance.SHOT_DISTRIBUTION, save=True, show=False)
    print("  ✅ Shot distribution plot created")

    perf_strategy.generate_plot(Plots.CircuitPerformance.AGGREGATE_SUMMARY, save=True, show=False)
    print("  ✅ Aggregate summary plot created")

    # Create dashboard
    perf_strategy.create_dashboard(save=True, show=False)
    print("  ✅ CircuitPerformanceVisualizer dashboard created")

    # Show available plots and metadata
    print("  Available CircuitPerformanceVisualizer plots:")
    for plot_name in perf_strategy.get_available_plots():
        metadata = perf_strategy.get_plot_metadata(plot_name)
        print(f"    - {plot_name}: {metadata.description} ({metadata.plot_type.value})")


def example_custom_strategy_workflow():
    """Example: Custom workflow using multiple strategies."""
    print("\n=== Example: Custom Multi-Strategy Workflow ===")

    # Create circuit and calculate all metrics
    circuit = create_sample_circuit()
    scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics])
    metrics_data = scanner.calculate_metrics()

    # Create custom configuration for all strategies
    custom_config = PlotConfig(
        figsize=(10, 8),
        style="minimal",
        dpi=200,
        save_format="pdf",  # Save as PDF instead of PNG
        color_palette=["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#592E83"],
        alpha=0.9,
    )

    print("Creating custom workflow with PDF outputs...")

    # Use QiskitVisualizer strategy with custom config
    qiskit_strategy = QiskitVisualizer(
        metrics_dict={Metrics.QISKIT: metrics_data[Metrics.QISKIT]},
        output_dir="qward/examples/img/custom_workflow",
        config=custom_config,
    )

    # Create only specific plots we want using new API
    selected_qiskit_plots = qiskit_strategy.generate_plots(
        [Plots.Qiskit.CIRCUIT_STRUCTURE, Plots.Qiskit.GATE_DISTRIBUTION], save=True, show=False
    )
    print(f"  ✅ {len(selected_qiskit_plots)} selected QiskitVisualizer plots created as PDF")

    # Use ComplexityVisualizer strategy with same config
    complexity_strategy = ComplexityVisualizer(
        metrics_dict={Metrics.COMPLEXITY: metrics_data[Metrics.COMPLEXITY]},
        output_dir="qward/examples/img/custom_workflow",
        config=custom_config,
    )

    # Create only the radar chart and efficiency metrics
    selected_complexity_plots = complexity_strategy.generate_plots(
        [Plots.Complexity.COMPLEXITY_RADAR, Plots.Complexity.EFFICIENCY_METRICS],
        save=True,
        show=False,
    )
    print(
        f"  ✅ {len(selected_complexity_plots)} selected ComplexityVisualizer plots created as PDF"
    )

    # Create combined dashboard-style figure
    print("  ✅ Custom workflow completed with granular plot selection")


def example_strategy_with_custom_data():
    """Example: Using strategies with custom data instead of Scanner."""
    print("\n=== Example: Strategy with Custom Data ===")

    # Create custom QiskitMetrics data
    custom_qiskit_data = pd.DataFrame(
        [
            {
                "basic_metrics.depth": 8,
                "basic_metrics.width": 4,
                "basic_metrics.size": 12,
                "basic_metrics.num_qubits": 4,
                "basic_metrics.num_clbits": 4,
                "basic_metrics.count_ops.h": 3,
                "basic_metrics.count_ops.cx": 4,
                "basic_metrics.count_ops.rz": 2,
                "basic_metrics.count_ops.measure": 4,
                "instruction_metrics.num_connected_components": 1,
                "instruction_metrics.num_nonlocal_gates": 4,
                "instruction_metrics.num_tensor_factors": 1,
                "instruction_metrics.num_unitary_factors": 1,
            }
        ]
    )

    # Create custom ComplexityMetrics data
    custom_complexity_data = pd.DataFrame(
        [
            {
                "gate_based_metrics.gate_count": 12,
                "gate_based_metrics.circuit_depth": 8,
                "gate_based_metrics.t_count": 0,
                "gate_based_metrics.cnot_count": 4,
                "gate_based_metrics.two_qubit_count": 4,
                "gate_based_metrics.multi_qubit_ratio": 0.33,
                "standardized_metrics.circuit_volume": 32,
                "standardized_metrics.gate_density": 0.375,
                "entanglement_metrics.entangling_gate_density": 0.33,
                "advanced_metrics.parallelism_factor": 1.5,
                "advanced_metrics.parallelism_efficiency": 0.75,
                "advanced_metrics.circuit_efficiency": 0.85,
                "derived_metrics.square_ratio": 0.5,
            }
        ]
    )

    print("Creating visualizations from custom data...")

    # Use QiskitVisualizer with custom data
    qiskit_strategy = QiskitVisualizer(
        metrics_dict={Metrics.QISKIT: custom_qiskit_data},
        output_dir="qward/examples/img/custom_data",
    )

    # Generate all QiskitVisualizer plots
    qiskit_plots = qiskit_strategy.generate_all_plots(save=True, show=False)
    print(f"  ✅ Created {len(qiskit_plots)} QiskitVisualizer plots from custom data")

    # Use ComplexityVisualizer with custom data
    complexity_strategy = ComplexityVisualizer(
        metrics_dict={Metrics.COMPLEXITY: custom_complexity_data},
        output_dir="qward/examples/img/custom_data",
    )

    # Generate all ComplexityVisualizer plots
    complexity_plots = complexity_strategy.generate_all_plots(save=True, show=False)
    print(f"  ✅ Created {len(complexity_plots)} ComplexityVisualizer plots from custom data")


def main():
    """Run all direct strategy examples."""
    print("QWARD Direct Visualization Strategy Examples")
    print("=" * 50)

    example_qiskit_strategy_direct()
    example_complexity_strategy_direct()
    example_circuit_performance_strategy_direct()
    example_custom_strategy_workflow()
    example_strategy_with_custom_data()

    print("\n" + "=" * 50)
    print("All direct strategy examples completed!")
    print("\nGenerated plots in:")
    print("- qward/examples/img/direct_qiskit/")
    print("- qward/examples/img/direct_complexity/")
    print("- qward/examples/img/direct_performance/")
    print("- qward/examples/img/custom_workflow/")
    print("- qward/examples/img/custom_data/")

    print("\nNew API Benefits Demonstrated:")
    print("- Type-safe constants prevent typos")
    print("- Granular plot control with generate_plots()")
    print("- Rich metadata for each plot")
    print("- Memory-efficient defaults")
    print("- IDE autocompletion support")
    print("- Flexible plot selection strategies")


if __name__ == "__main__":
    main()
