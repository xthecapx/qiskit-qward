"""
Quick visualization example using QWARD CircuitPerformance metrics.

This script demonstrates the basic visualization capabilities with minimal setup.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner
from qward.metrics import CircuitPerformanceMetrics
from qward.visualization import CircuitPerformanceVisualizer
from qward.visualization.constants import Metrics, Plots


def main():
    """Create and visualize CircuitPerformance metrics."""
    # Create a Bell state circuit
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    # Run multiple simulations
    simulator = AerSimulator()
    jobs = []
    for _ in range(3):
        job = simulator.run(circuit, shots=1024)
        jobs.append(job)

    # Wait for completion
    for job in jobs:
        job.result()

    # Create scanner with CircuitPerformance strategy
    scanner = Scanner(circuit=circuit)

    def bell_state_success(outcome):
        """Success criteria for Bell state: |00⟩ or |11⟩."""
        clean = outcome.replace(" ", "")
        return clean in ["00", "11"]

    circuit_performance = CircuitPerformanceMetrics(
        circuit=circuit, success_criteria=bell_state_success
    )
    circuit_performance.add_job(jobs)
    scanner.add_strategy(circuit_performance)

    # Calculate metrics
    metrics_dict = scanner.calculate_metrics()

    metrics_dict["CircuitPerformance.aggregate"].plot()

    # Create visualizations using new API
    visualizer = CircuitPerformanceVisualizer(
        metrics_dict, output_dir="qward/examples/img/quickstart"
    )

    # Generate all plots using new API
    all_plots = visualizer.generate_all_plots(save=True, show=False)
    print(f"Generated {len(all_plots)} CircuitPerformance plots")

    # Generate specific performance plots using new API
    performance_plots = visualizer.generate_plots(
        {
            Metrics.CIRCUIT_PERFORMANCE: [
                Plots.CircuitPerformance.SUCCESS_ERROR_COMPARISON,
                Plots.CircuitPerformance.FIDELITY_COMPARISON,
            ]
        },
        save=True,
        show=False,
    )
    print(f"Generated {len(performance_plots)} specific plots")

    # Create dashboard
    dashboard = visualizer.create_dashboard(save=True, show=False)
    print("Dashboard created")

    print("Visualizations saved to qward/examples/img/quickstart/")

    # Print summary
    aggregate_data = metrics_dict.get("CircuitPerformance.aggregate")
    if aggregate_data is not None and not aggregate_data.empty:
        mean_success = aggregate_data.iloc[0]["success_metrics.mean_success_rate"]
        print(f"Mean success rate: {mean_success:.3f}")

    # Show available plots and metadata
    print("\nAvailable CircuitPerformance plots:")
    available_plots = visualizer.get_available_plots()
    for plot_name in available_plots:
        metadata = visualizer.get_plot_metadata(plot_name)
        print(f"  - {plot_name}: {metadata.description}")


if __name__ == "__main__":
    main()
