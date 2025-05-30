"""
Quick visualization example using QWARD CircuitPerformance metrics.

This script demonstrates the basic visualization capabilities with minimal setup.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner
from qward.metrics import CircuitPerformance
from qward.visualization import CircuitPerformanceVisualizer


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

    circuit_performance = CircuitPerformance(circuit=circuit, success_criteria=bell_state_success)
    circuit_performance.add_job(jobs)
    scanner.add_strategy(circuit_performance)

    # Calculate metrics
    metrics_dict = scanner.calculate_metrics()

    metrics_dict["CircuitPerformance.aggregate"].plot()

    # Create visualizations
    visualizer = CircuitPerformanceVisualizer(metrics_dict, output_dir="img/quickstart")
    visualizer.plot_all(save=True, show=False)

    print("Visualizations saved to img/quickstart/")

    # Print summary
    aggregate_data = metrics_dict.get("CircuitPerformance.aggregate")
    if aggregate_data is not None and not aggregate_data.empty:
        mean_success = aggregate_data.iloc[0]["mean_success_rate"]
        print(f"Mean success rate: {mean_success:.3f}")


if __name__ == "__main__":
    main()
