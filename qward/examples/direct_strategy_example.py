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
        metrics_dict={"QiskitMetrics": metrics_data["QiskitMetrics"]},
        output_dir="qward/examples/img/direct_qiskit",
    )

    print("Creating individual QiskitVisualizer plots...")

    # Create specific plots
    qiskit_strategy.plot_circuit_structure(save=True, show=False)
    print("  ✅ Circuit structure plot created")

    qiskit_strategy.plot_gate_distribution(save=True, show=False)
    print("  ✅ Gate distribution plot created")

    qiskit_strategy.plot_instruction_metrics(save=True, show=False)
    print("  ✅ Instruction metrics plot created")

    qiskit_strategy.plot_circuit_summary(save=True, show=False)
    print("  ✅ Circuit summary plot created")

    # Create dashboard
    qiskit_strategy.create_dashboard(save=True, show=False)
    print("  ✅ QiskitVisualizer dashboard created")


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
        metrics_dict={"ComplexityMetrics": metrics_data["ComplexityMetrics"]},
        output_dir="qward/examples/img/direct_complexity",
        config=custom_config,
    )

    print("Creating individual ComplexityVisualizer plots...")

    # Create specific plots
    complexity_strategy.plot_gate_based_metrics(save=True, show=False)
    print("  ✅ Gate-based metrics plot created")

    complexity_strategy.plot_complexity_radar(save=True, show=False)
    print("  ✅ Complexity radar chart created")

    complexity_strategy.plot_quantum_volume_analysis(save=True, show=False)
    print("  ✅ Quantum volume analysis plot created")

    complexity_strategy.plot_efficiency_metrics(save=True, show=False)
    print("  ✅ Efficiency metrics plot created")

    # Create dashboard
    complexity_strategy.create_dashboard(save=True, show=False)
    print("  ✅ ComplexityVisualizer dashboard created")


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

    print("Creating individual CircuitPerformanceVisualizer plots...")

    # Create specific plots
    perf_strategy.plot_success_error_comparison(save=True, show=False)
    print("  ✅ Success vs error comparison plot created")

    perf_strategy.plot_fidelity_comparison(save=True, show=False)
    print("  ✅ Fidelity comparison plot created")

    perf_strategy.plot_shot_distribution(save=True, show=False)
    print("  ✅ Shot distribution plot created")

    perf_strategy.plot_aggregate_summary(save=True, show=False)
    print("  ✅ Aggregate summary plot created")

    # Create dashboard
    perf_strategy.create_dashboard(save=True, show=False)
    print("  ✅ CircuitPerformanceVisualizer dashboard created")


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
        metrics_dict={"QiskitMetrics": metrics_data["QiskitMetrics"]},
        output_dir="qward/examples/img/custom_workflow",
        config=custom_config,
    )

    # Create only specific plots we want
    qiskit_strategy.plot_circuit_structure(save=True, show=False)
    qiskit_strategy.plot_gate_distribution(save=True, show=False)
    print("  ✅ Selected QiskitVisualizer plots created as PDF")

    # Use ComplexityVisualizer strategy with same config
    complexity_strategy = ComplexityVisualizer(
        metrics_dict={"ComplexityMetrics": metrics_data["ComplexityMetrics"]},
        output_dir="qward/examples/img/custom_workflow",
        config=custom_config,
    )

    # Create only the radar chart and efficiency metrics
    complexity_strategy.plot_complexity_radar(save=True, show=False)
    complexity_strategy.plot_efficiency_metrics(save=True, show=False)
    print("  ✅ Selected ComplexityVisualizer plots created as PDF")


def example_strategy_with_custom_data():
    """Example: Using strategies with completely custom data."""
    print("\n=== Example: Strategy with Custom Data ===")

    # Create custom metrics data (simulating external data source)
    custom_qiskit_data = pd.DataFrame(
        [
            {
                "basic_metrics.depth": 12,
                "basic_metrics.width": 6,
                "basic_metrics.size": 24,
                "basic_metrics.num_qubits": 6,
                "basic_metrics.num_clbits": 6,
                "basic_metrics.count_ops.h": 6,
                "basic_metrics.count_ops.cx": 10,
                "basic_metrics.count_ops.rz": 8,
                "instruction_metrics.num_connected_components": 1,
                "instruction_metrics.num_nonlocal_gates": 10,
                "instruction_metrics.num_tensor_factors": 1,
                "instruction_metrics.num_unitary_factors": 1,
            }
        ]
    )

    custom_complexity_data = pd.DataFrame(
        [
            {
                "gate_based_metrics.gate_count": 24,
                "gate_based_metrics.circuit_depth": 12,
                "gate_based_metrics.t_count": 0,
                "gate_based_metrics.cnot_count": 10,
                "gate_based_metrics.two_qubit_count": 10,
                "standardized_metrics.circuit_volume": 72,
                "standardized_metrics.gate_density": 0.33,
                "quantum_volume.standard_quantum_volume": 64,
                "quantum_volume.enhanced_quantum_volume": 68,
                "quantum_volume.effective_depth": 10,
                "advanced_metrics.parallelism_efficiency": 0.75,
                "advanced_metrics.circuit_efficiency": 0.8,
            }
        ]
    )

    print("Using strategies with custom external data...")

    # Use strategies with custom data
    qiskit_strategy = QiskitVisualizer(
        metrics_dict={"QiskitMetrics": custom_qiskit_data},
        output_dir="qward/examples/img/custom_data",
    )

    complexity_strategy = ComplexityVisualizer(
        metrics_dict={"ComplexityMetrics": custom_complexity_data},
        output_dir="qward/examples/img/custom_data",
    )

    # Create dashboards
    qiskit_strategy.create_dashboard(save=True, show=False)
    complexity_strategy.create_dashboard(save=True, show=False)

    print("  ✅ Dashboards created from custom data")


def main():
    """Run all direct strategy examples."""
    print("🎯 QWARD Direct Strategy Usage Examples")
    print("=" * 50)
    print("These examples show how to use visualization strategies directly,")
    print("without the unified Visualizer class, for maximum flexibility.")

    # Run all examples
    example_qiskit_strategy_direct()
    example_complexity_strategy_direct()
    example_circuit_performance_strategy_direct()
    example_custom_strategy_workflow()
    example_strategy_with_custom_data()

    print("\n" + "=" * 50)
    print("🎉 All direct strategy examples completed!")
    print("\nGenerated plots in:")
    print("- qward/examples/img/direct_qiskit/")
    print("- qward/examples/img/direct_complexity/")
    print("- qward/examples/img/direct_performance/")
    print("- qward/examples/img/custom_workflow/")
    print("- qward/examples/img/custom_data/")
    print("\n💡 Key benefits of direct strategy usage:")
    print("• Fine-grained control over individual plots")
    print("• Custom configurations per strategy")
    print("• Integration with external data sources")
    print("• Flexible output formats and locations")
    print("• Perfect for custom workflows and pipelines")


if __name__ == "__main__":
    main()
