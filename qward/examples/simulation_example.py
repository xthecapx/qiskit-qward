#!/usr/bin/env python3
"""
QWARD Simulation Example

This comprehensive example demonstrates:
1. Using QWARD with Aer simulator
2. Different noise models (depolarizing, Pauli, custom)
3. Multiple job analysis
4. CircuitPerformance metrics with visualization
5. Circuit comparison workflows
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    pauli_error,
    depolarizing_error,
)

from qward import Scanner, Visualizer
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics
from qward.metrics.defaults import get_default_strategies
from qward.visualization import PlotConfig, CircuitPerformanceVisualizer
from qward.visualization.constants import Metrics, Plots
from qward.examples.utils import create_example_circuit, get_display

display = get_display()


# =============================================================================
# Circuit Creation
# =============================================================================


def create_test_circuits():
    """Create various test circuits for analysis."""
    circuits = {}

    # Bell state
    bell = QuantumCircuit(2, 2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure_all()
    circuits["Bell"] = bell

    # GHZ state
    ghz = QuantumCircuit(3, 3)
    ghz.h(0)
    ghz.cx(0, 1)
    ghz.cx(0, 2)
    ghz.measure_all()
    circuits["GHZ"] = ghz

    # Variational circuit
    var = QuantumCircuit(4, 4)
    var.ry(0.5, 0)
    var.ry(0.3, 1)
    var.cx(0, 1)
    var.ry(0.7, 2)
    var.cx(1, 2)
    var.cx(2, 3)
    var.measure_all()
    circuits["Variational"] = var

    return circuits


def define_success_criteria():
    """Define success criteria for different circuit types."""

    def bell_success(outcome):
        clean = outcome.replace(" ", "")
        return clean in ["00", "11"]

    def ghz_success(outcome):
        clean = outcome.replace(" ", "")
        return clean in ["000", "111"]

    def variational_success(outcome):
        clean = outcome.replace(" ", "")
        return clean.count("1") % 2 == 0

    return {
        "Bell": bell_success,
        "GHZ": ghz_success,
        "Variational": variational_success,
    }


# =============================================================================
# Basic Examples
# =============================================================================


def basic_simulation_example():
    """Example 1: Basic simulation with default strategies."""
    print("=== Example 1: Basic Simulation ===\n")

    circuit = create_example_circuit()

    # Using default strategies
    scanner = Scanner(circuit=circuit)
    for strategy in get_default_strategies():
        scanner.add_strategy(strategy(circuit))

    metrics = scanner.calculate_metrics()

    print("Metrics calculated:")
    for name, df in metrics.items():
        print(f"  {name}: {df.shape}")

    return scanner


def strategies_in_constructor():
    """Example 2: Strategies via constructor."""
    print("\n=== Example 2: Strategies in Constructor ===\n")

    circuit = create_example_circuit()

    # Pass strategies directly
    scanner = Scanner(
        circuit=circuit,
        strategies=[QiskitMetrics, ComplexityMetrics],
    )

    metrics = scanner.calculate_metrics()
    scanner.display_summary(metrics)

    return scanner


# =============================================================================
# Noise Models
# =============================================================================


def create_noise_models():
    """Create different noise models for testing."""
    noise_models = {}

    # Depolarizing noise
    depol_model = NoiseModel()
    depol_error_1q = depolarizing_error(0.05, 1)
    depol_error_2q = depolarizing_error(0.10, 2)
    depol_model.add_all_qubit_quantum_error(depol_error_1q, ["u1", "u2", "u3"])
    depol_model.add_all_qubit_quantum_error(depol_error_2q, ["cx"])
    readout_10 = ReadoutError([[0.9, 0.1], [0.1, 0.9]])
    depol_model.add_all_qubit_readout_error(readout_10)
    noise_models["depolarizing"] = depol_model

    # Pauli noise
    pauli_model = NoiseModel()
    pauli_error_1q = pauli_error([("X", 0.05), ("Y", 0.05), ("Z", 0.05), ("I", 0.85)])
    pauli_error_2q = pauli_error([("XX", 0.05), ("YY", 0.05), ("ZZ", 0.05), ("II", 0.85)])
    pauli_model.add_all_qubit_quantum_error(pauli_error_1q, ["u1", "u2", "u3"])
    pauli_model.add_all_qubit_quantum_error(pauli_error_2q, ["cx"])
    readout_5 = ReadoutError([[0.95, 0.05], [0.05, 0.95]])
    pauli_model.add_all_qubit_readout_error(readout_5)
    noise_models["pauli"] = pauli_model

    return noise_models


def noise_model_example():
    """Example 3: Multiple noise models comparison."""
    print("\n=== Example 3: Noise Model Comparison ===\n")

    circuit = create_example_circuit()
    noise_models = create_noise_models()

    # Run with different noise models
    jobs = []

    # No noise
    ideal_sim = AerSimulator()
    jobs.append(ideal_sim.run(circuit, shots=1024))

    # Depolarizing noise
    depol_sim = AerSimulator(noise_model=noise_models["depolarizing"])
    jobs.append(depol_sim.run(circuit, shots=1024))

    # Pauli noise
    pauli_sim = AerSimulator(noise_model=noise_models["pauli"])
    jobs.append(pauli_sim.run(circuit, shots=1024))

    # Wait for completion
    for job in jobs:
        job.result()

    # Create CircuitPerformance metrics
    def success_criteria(outcome):
        return outcome.replace(" ", "") == "11"

    circuit_performance = CircuitPerformanceMetrics(
        circuit=circuit,
        success_criteria=success_criteria,
    )
    circuit_performance.add_job(jobs)

    # Create scanner
    scanner = Scanner(circuit=circuit)
    scanner.add_strategy(circuit_performance)

    metrics = scanner.calculate_metrics()

    # Display results
    print("Individual job metrics:")
    display(metrics["CircuitPerformance.individual_jobs"])

    print("\nAggregate metrics:")
    display(metrics["CircuitPerformance.aggregate"])

    return scanner, jobs


# =============================================================================
# Comprehensive Analysis
# =============================================================================


def analyze_circuit(circuit_name, circuit, jobs, success_criteria):
    """Perform comprehensive analysis of a circuit."""
    print(f"\n{'='*60}")
    print(f"🔬 ANALYZING {circuit_name.upper()} CIRCUIT")
    print(f"{'='*60}")

    # Create metrics
    circuit_performance = CircuitPerformanceMetrics(
        circuit=circuit,
        jobs=jobs,
        success_criteria=success_criteria,
    )

    scanner = Scanner(
        circuit=circuit,
        strategies=[circuit_performance, QiskitMetrics, ComplexityMetrics],
    )

    metrics = scanner.calculate_metrics()

    # Display summary
    print("\n📋 SCANNER SUMMARY")
    scanner.display_summary(metrics)

    # Metrics structure
    print(f"\n📈 METRICS STRUCTURE")
    for name, df in metrics.items():
        print(f"  {name}: {df.shape[0]} rows × {df.shape[1]} columns")

    return scanner, metrics


def visualization_workflow(scanner, circuit_name):
    """Create visualizations for a circuit."""
    print(f"\n🎨 VISUALIZATIONS FOR {circuit_name.upper()}")

    config = PlotConfig(
        figsize=(12, 8),
        style="quantum",
        color_palette=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
        dpi=150,
    )

    visualizer = Visualizer(
        scanner=scanner,
        config=config,
        output_dir=f"qward/examples/img/{circuit_name.lower()}",
    )

    # Available metrics
    available = visualizer.get_available_metrics()
    print(f"  Available: {available}")

    # Generate CircuitPerformance plots
    if Metrics.CIRCUIT_PERFORMANCE in available:
        visualizer.generate_plot(
            metric_name=Metrics.CIRCUIT_PERFORMANCE,
            plot_name=Plots.CircuitPerformance.SUCCESS_ERROR_COMPARISON,
            save=True,
            show=False,
        )
        print("  ✓ Success/error comparison saved")

    # Generate selected plots
    selections = {}
    if Metrics.QISKIT in available:
        selections[Metrics.QISKIT] = [
            Plots.Qiskit.CIRCUIT_STRUCTURE,
            Plots.Qiskit.GATE_DISTRIBUTION,
        ]
    if Metrics.COMPLEXITY in available:
        selections[Metrics.COMPLEXITY] = [
            Plots.Complexity.GATE_BASED_METRICS,
            Plots.Complexity.COMPLEXITY_RADAR,
        ]

    if selections:
        plots = visualizer.generate_plots(selections=selections, save=True, show=False)
        total = sum(len(p) for p in plots.values())
        print(f"  ✓ Generated {total} selected plots")

    # Dashboard
    dashboards = visualizer.create_dashboard(save=True, show=False)
    print(f"  ✓ Created {len(dashboards)} dashboards")

    return visualizer


def compare_circuits(results):
    """Compare metrics across circuits."""
    print(f"\n🔍 CIRCUIT COMPARISON")
    print("=" * 80)

    comparison = {}

    for circuit_name, (scanner, metrics) in results.items():
        comparison[circuit_name] = {}

        if "QiskitMetrics" in metrics:
            qm = metrics["QiskitMetrics"]
            if not qm.empty:
                comparison[circuit_name]["qubits"] = qm.iloc[0]["basic_metrics.num_qubits"]
                comparison[circuit_name]["depth"] = qm.iloc[0]["basic_metrics.depth"]

        if "ComplexityMetrics" in metrics:
            cm = metrics["ComplexityMetrics"]
            if not cm.empty:
                comparison[circuit_name]["gates"] = cm.iloc[0]["gate_based_metrics.gate_count"]
                comparison[circuit_name]["cnots"] = cm.iloc[0]["gate_based_metrics.cnot_count"]

        if "CircuitPerformance.aggregate" in metrics:
            perf = metrics["CircuitPerformance.aggregate"]
            if not perf.empty:
                comparison[circuit_name]["success"] = perf.iloc[0][
                    "success_metrics.mean_success_rate"
                ]
                comparison[circuit_name]["fidelity"] = perf.iloc[0][
                    "fidelity_metrics.mean_fidelity"
                ]

    # Print comparison table
    headers = ["Circuit", "Qubits", "Depth", "Gates", "CNOTs", "Success", "Fidelity"]
    print(
        f"{headers[0]:<12} {headers[1]:<7} {headers[2]:<6} {headers[3]:<6} {headers[4]:<6} {headers[5]:<8} {headers[6]:<8}"
    )
    print("-" * 60)

    for name, data in comparison.items():
        qubits = data.get("qubits", "N/A")
        depth = data.get("depth", "N/A")
        gates = data.get("gates", "N/A")
        cnots = data.get("cnots", "N/A")
        success = data.get("success", "N/A")
        fidelity = data.get("fidelity", "N/A")

        if isinstance(success, float):
            success = f"{success:.3f}"
        if isinstance(fidelity, float):
            fidelity = f"{fidelity:.3f}"

        print(f"{name:<12} {qubits:<7} {depth:<6} {gates:<6} {cnots:<6} {success:<8} {fidelity:<8}")


def comprehensive_workflow():
    """Run comprehensive analysis workflow."""
    print("\n=== Comprehensive Workflow ===\n")

    circuits = create_test_circuits()
    success_criteria = define_success_criteria()
    simulator = AerSimulator()

    results = {}

    for circuit_name, circuit in circuits.items():
        # Run simulations
        print(f"\nRunning {circuit_name} simulations...")
        jobs = [simulator.run(circuit, shots=1024) for _ in range(3)]
        for job in jobs:
            job.result()

        # Analyze
        scanner, metrics = analyze_circuit(
            circuit_name,
            circuit,
            jobs,
            success_criteria[circuit_name],
        )

        # Visualize
        visualization_workflow(scanner, circuit_name)

        results[circuit_name] = (scanner, metrics)

    # Compare
    compare_circuits(results)

    return results


def main():
    """Run all simulation examples."""
    print("🚀 QWARD Simulation Examples")
    print("=" * 60)

    # Basic examples
    basic_simulation_example()
    strategies_in_constructor()

    # Noise models
    noise_model_example()

    # Comprehensive workflow
    comprehensive_workflow()

    print("\n" + "=" * 60)
    print("✅ Simulation examples complete!")
    print("\nKey concepts demonstrated:")
    print("  • Aer simulator integration")
    print("  • Noise models (depolarizing, Pauli)")
    print("  • CircuitPerformance metrics")
    print("  • Multiple job analysis")
    print("  • Circuit comparison workflows")
    print("\n📁 Results saved to: qward/examples/img/")


if __name__ == "__main__":
    main()
