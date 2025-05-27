"""
Quick visualization example for QWARD.

This script demonstrates the basic visualization capabilities with minimal setup.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner
from qward.metrics import SuccessRate
from qward.visualization import SuccessRateVisualizer


def main():
    """Run a simple visualization example."""
    # 1. Create a Bell state circuit
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    print("Created Bell state circuit")

    # 2. Run the circuit on simulator
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=1024)
    job.result()  # Wait for completion

    print("Circuit executed successfully")

    # 3. Calculate metrics
    scanner = Scanner(circuit=circuit)

    # Define success criteria (we expect |00⟩ or |11⟩ for Bell state)
    def bell_state_success(outcome):
        return outcome in ["00", "11"]

    success_rate = SuccessRate(circuit=circuit, success_criteria=bell_state_success)
    success_rate.add_job(job)
    scanner.add_strategy(success_rate)

    metrics_dict = scanner.calculate_metrics()
    print("Metrics calculated")

    # 4. Create visualizations
    visualizer = SuccessRateVisualizer(metrics_dict, output_dir="img/quickstart")

    # Generate all plots
    print("\nGenerating visualizations...")
    figures = visualizer.plot_all(save=True, show=False)
    print(f"✅ Created {len(figures)} plots in img/quickstart/")

    # Show success rate summary
    aggregate_data = metrics_dict.get("SuccessRate.aggregate")
    if aggregate_data is not None:
        success_rate_value = aggregate_data["mean_success_rate"].iloc[0]
        print(f"\nBell state success rate: {success_rate_value:.2%}")

    print("\nVisualization complete! Check the img/quickstart/ directory for plots.")


if __name__ == "__main__":
    main()
