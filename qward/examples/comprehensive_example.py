#!/usr/bin/env python3
"""
Comprehensive QWARD Example - Updated for Latest API

This example demonstrates all the key features of the QWARD library:
- Circuit analysis with multiple metrics
- New visualization API with keyword-only arguments
- Scanner summary functionality
- Proper column separation between individual and aggregate metrics
- IBM Quantum Runtime integration patterns
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner, Visualizer
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics
from qward.visualization.constants import Metrics, Plots
from qward.visualization import PlotConfig


def create_test_circuits():
    """Create various test circuits for analysis."""
    circuits = {}

    # Bell state circuit
    bell = QuantumCircuit(2, 2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure_all()
    circuits["Bell"] = bell

    # GHZ state circuit
    ghz = QuantumCircuit(3, 3)
    ghz.h(0)
    ghz.cx(0, 1)
    ghz.cx(0, 2)
    ghz.measure_all()
    circuits["GHZ"] = ghz

    # Variational circuit
    variational = QuantumCircuit(4, 4)
    variational.ry(0.5, 0)
    variational.ry(0.3, 1)
    variational.cx(0, 1)
    variational.ry(0.7, 2)
    variational.cx(1, 2)
    variational.cx(2, 3)
    variational.measure_all()
    circuits["Variational"] = variational

    return circuits


def run_simulations(circuit, num_jobs=3, shots_per_job=1024):
    """Run multiple simulation jobs for a circuit."""
    simulator = AerSimulator()
    jobs = []

    print(f"Running {num_jobs} simulation jobs with {shots_per_job} shots each...")
    for i in range(num_jobs):
        job = simulator.run(circuit, shots=shots_per_job)
        jobs.append(job)
        print(f"  Job {i+1} submitted: {job.job_id()}")

    # Wait for completion
    for i, job in enumerate(jobs):
        result = job.result()
        print(f"  Job {i+1} completed")

    return jobs


def define_success_criteria():
    """Define success criteria for different circuit types."""

    def bell_success(outcome):
        """Bell state: |00⟩ or |11⟩"""
        clean = outcome.replace(" ", "")
        return clean in ["00", "11"]

    def ghz_success(outcome):
        """GHZ state: |000⟩ or |111⟩"""
        clean = outcome.replace(" ", "")
        return clean in ["000", "111"]

    def variational_success(outcome):
        """Variational: any outcome with even parity"""
        clean = outcome.replace(" ", "")
        return clean.count("1") % 2 == 0

    return {"Bell": bell_success, "GHZ": ghz_success, "Variational": variational_success}


def analyze_circuit(circuit_name, circuit, jobs, success_criteria):
    """Perform comprehensive analysis of a single circuit."""
    print(f"\n{'='*60}")
    print(f"🔬 ANALYZING {circuit_name.upper()} CIRCUIT")
    print(f"{'='*60}")

    # Create metrics strategies
    circuit_performance = CircuitPerformanceMetrics(
        circuit=circuit, jobs=jobs, success_criteria=success_criteria
    )

    # Create scanner with all strategies
    scanner = Scanner(
        circuit=circuit,
        strategies=[
            circuit_performance,
            QiskitMetrics,
            ComplexityMetrics,
        ],
    )

    # Calculate metrics
    print("📊 Calculating metrics...")
    metrics_dict = scanner.calculate_metrics()

    # Display professional summary
    print("\n📋 SCANNER SUMMARY")
    print("-" * 40)
    scanner.display_summary(metrics_dict)

    # Show detailed metrics structure
    print(f"\n📈 DETAILED METRICS STRUCTURE")
    print("-" * 40)
    for metric_name, df in metrics_dict.items():
        print(f"{metric_name}: {df.shape[0]} rows × {df.shape[1]} columns")
        if not df.empty:
            # Show a few key columns as examples
            key_columns = [
                col
                for col in df.columns
                if any(
                    keyword in col.lower()
                    for keyword in ["success", "fidelity", "depth", "gate_count"]
                )
            ][:3]
            if key_columns:
                print(f"  Sample columns: {', '.join(key_columns)}")

    return scanner, metrics_dict


def demonstrate_visualization_api(scanner, circuit_name):
    """Demonstrate the new visualization API features."""
    print(f"\n🎨 VISUALIZATION API DEMONSTRATION")
    print("-" * 40)

    # Create visualizer with custom config
    config = PlotConfig(
        figsize=(12, 8),
        style="quantum",
        color_palette=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
        alpha=0.8,
        dpi=150,
    )

    visualizer = Visualizer(
        scanner=scanner, config=config, output_dir=f"qward/examples/img/{circuit_name.lower()}"
    )

    # Show available metrics and plots
    print("Available metrics for visualization:")
    available_metrics = visualizer.get_available_metrics()
    for metric in available_metrics:
        plots = visualizer.get_available_plots(metric)
        print(f"  {metric}: {len(plots[metric])} plots available")

    # Example 1: Generate single specific plot
    print("\n1. Generating single plot...")
    if Metrics.CIRCUIT_PERFORMANCE in available_metrics:
        single_plot = visualizer.generate_plot(
            metric_name=Metrics.CIRCUIT_PERFORMANCE,
            plot_name=Plots.CircuitPerformance.SUCCESS_ERROR_COMPARISON,
            save=True,
            show=False,
        )
        print("   ✅ Success/error comparison plot generated")

    # Example 2: Generate selected plots
    print("\n2. Generating selected plots...")
    selections = {}
    if Metrics.QISKIT in available_metrics:
        selections[Metrics.QISKIT] = [
            Plots.Qiskit.CIRCUIT_STRUCTURE,
            Plots.Qiskit.GATE_DISTRIBUTION,
        ]
    if Metrics.COMPLEXITY in available_metrics:
        selections[Metrics.COMPLEXITY] = [
            Plots.Complexity.GATE_BASED_METRICS,
            Plots.Complexity.COMPLEXITY_RADAR,
        ]

    if selections:
        selected_plots = visualizer.generate_plots(selections=selections, save=True, show=False)
        total_plots = sum(len(plots) for plots in selected_plots.values())
        print(f"   ✅ Generated {total_plots} selected plots")

    # Example 3: Generate all plots for a metric
    print("\n3. Generating all plots for ComplexityMetrics...")
    if Metrics.COMPLEXITY in available_metrics:
        all_complexity = visualizer.generate_plots(
            selections={Metrics.COMPLEXITY: None}, save=True, show=False  # None = all plots
        )
        complexity_count = len(all_complexity[Metrics.COMPLEXITY])
        print(f"   ✅ Generated all {complexity_count} ComplexityMetrics plots")

    # Example 4: Create comprehensive dashboard
    print("\n4. Creating comprehensive dashboard...")
    dashboards = visualizer.create_dashboard(save=True, show=False)
    print(f"   ✅ Created {len(dashboards)} dashboards")

    print(f"\n📁 All visualizations saved to: qward/examples/img/{circuit_name.lower()}/")

    return visualizer


def compare_circuits(circuit_results):
    """Compare results across different circuits."""
    print(f"\n🔍 CIRCUIT COMPARISON")
    print("=" * 60)

    comparison_data = {}

    for circuit_name, (scanner, metrics_dict) in circuit_results.items():
        comparison_data[circuit_name] = {}

        # Extract key metrics for comparison
        if "QiskitMetrics" in metrics_dict:
            qiskit_df = metrics_dict["QiskitMetrics"]
            if not qiskit_df.empty:
                comparison_data[circuit_name]["qubits"] = qiskit_df.iloc[0][
                    "basic_metrics.num_qubits"
                ]
                comparison_data[circuit_name]["depth"] = qiskit_df.iloc[0]["basic_metrics.depth"]
                comparison_data[circuit_name]["size"] = qiskit_df.iloc[0]["basic_metrics.size"]

        if "ComplexityMetrics" in metrics_dict:
            complexity_df = metrics_dict["ComplexityMetrics"]
            if not complexity_df.empty:
                comparison_data[circuit_name]["gate_count"] = complexity_df.iloc[0][
                    "gate_based_metrics.gate_count"
                ]
                comparison_data[circuit_name]["cnot_count"] = complexity_df.iloc[0][
                    "gate_based_metrics.cnot_count"
                ]

        # Get aggregate performance metrics if available
        aggregate_key = "CircuitPerformance.aggregate"
        if aggregate_key in metrics_dict:
            perf_df = metrics_dict[aggregate_key]
            if not perf_df.empty:
                comparison_data[circuit_name]["mean_success_rate"] = perf_df.iloc[0][
                    "success_metrics.mean_success_rate"
                ]
                comparison_data[circuit_name]["mean_fidelity"] = perf_df.iloc[0][
                    "fidelity_metrics.mean_fidelity"
                ]

    # Display comparison table
    print("Circuit Comparison Summary:")
    print("-" * 80)
    headers = ["Circuit", "Qubits", "Depth", "Gates", "CNOTs", "Success Rate", "Fidelity"]
    print(
        f"{headers[0]:<12} {headers[1]:<7} {headers[2]:<6} {headers[3]:<6} {headers[4]:<6} {headers[5]:<12} {headers[6]:<8}"
    )
    print("-" * 80)

    for circuit_name, data in comparison_data.items():
        qubits = data.get("qubits", "N/A")
        depth = data.get("depth", "N/A")
        gates = data.get("gate_count", "N/A")
        cnots = data.get("cnot_count", "N/A")
        success = data.get("mean_success_rate", "N/A")
        fidelity = data.get("mean_fidelity", "N/A")

        # Format success rate and fidelity
        if isinstance(success, float):
            success = f"{success:.3f}"
        if isinstance(fidelity, float):
            fidelity = f"{fidelity:.3f}"

        print(
            f"{circuit_name:<12} {qubits:<7} {depth:<6} {gates:<6} {cnots:<6} {success:<12} {fidelity:<8}"
        )


def main():
    """Main demonstration function."""
    print("🚀 QWARD Comprehensive Example - Updated API")
    print("=" * 60)

    # Create test circuits
    circuits = create_test_circuits()
    success_criteria = define_success_criteria()

    # Analyze each circuit
    circuit_results = {}

    for circuit_name, circuit in circuits.items():
        # Run simulations
        jobs = run_simulations(circuit, num_jobs=3, shots_per_job=1024)

        # Analyze circuit
        scanner, metrics_dict = analyze_circuit(
            circuit_name, circuit, jobs, success_criteria[circuit_name]
        )

        # Demonstrate visualization
        visualizer = demonstrate_visualization_api(scanner, circuit_name)

        circuit_results[circuit_name] = (scanner, metrics_dict)

    # Compare all circuits
    compare_circuits(circuit_results)

    print(f"\n✅ ANALYSIS COMPLETE")
    print("=" * 60)
    print("Key Features Demonstrated:")
    print("  ✓ Multiple circuit types (Bell, GHZ, Variational)")
    print("  ✓ Comprehensive metrics (Performance, Qiskit, Complexity)")
    print("  ✓ New keyword-only visualization API")
    print("  ✓ Professional Scanner summaries")
    print("  ✓ Proper column separation (individual vs aggregate)")
    print("  ✓ Custom plot configurations")
    print("  ✓ Circuit comparison analysis")
    print("  ✓ Memory-efficient defaults (save=False, show=False)")
    print("\n📁 All results saved to: qward/examples/img/")


if __name__ == "__main__":
    main()
