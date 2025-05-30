"""
Demo showing how to use QWARD's CircuitPerformance visualization capabilities.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner
from qward.metrics import CircuitPerformance
from qward.visualization import CircuitPerformanceVisualizer, PlotConfig


def demo_circuit_performance_visualization():
    """
    Demonstrate CircuitPerformance visualization with different configurations.
    """
    print("QWARD CircuitPerformance Visualization Demo")
    print("=" * 40)

    # Create a simple quantum circuit
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    print(f"Circuit: {circuit}")

    # Create multiple jobs with different shot counts
    simulator = AerSimulator()
    jobs = []

    # Job 1: 1024 shots
    job1 = simulator.run(circuit, shots=1024)
    jobs.append(job1)

    # Job 2: 2048 shots
    job2 = simulator.run(circuit, shots=2048)
    jobs.append(job2)

    # Job 3: 512 shots
    job3 = simulator.run(circuit, shots=512)
    jobs.append(job3)

    # Wait for jobs to complete
    for job in jobs:
        job.result()

    # Create scanner and add CircuitPerformance strategy
    scanner = Scanner(circuit=circuit)
    circuit_performance_strategy = CircuitPerformance(circuit=circuit)

    # Add all jobs
    circuit_performance_strategy.add_job(jobs)
    scanner.add_strategy(circuit_performance_strategy)

    # Calculate metrics
    print("\nCalculating CircuitPerformance metrics...")
    metrics_dict = scanner.calculate_metrics()

    print("\nCircuitPerformance metrics calculated successfully!")
    print(f"Individual jobs: {len(metrics_dict['CircuitPerformance.individual_jobs'])} jobs")
    print(f"Aggregate data: {metrics_dict['CircuitPerformance.aggregate'].shape}")

    # Demo 1: Default visualization
    print("\n" + "=" * 40)
    print("Demo 1: Default Visualization")
    print("=" * 40)

    visualizer1 = CircuitPerformanceVisualizer(metrics_dict, output_dir="img/demo_plots_default")
    figures1 = visualizer1.plot_all(save=True, show=False)
    print(f"✅ Created {len(figures1)} plots with default settings")

    # Demo 2: Custom configuration
    print("\n" + "=" * 40)
    print("Demo 2: Custom Configuration")
    print("=" * 40)

    custom_config = PlotConfig(
        figsize=(12, 8),
        style="quantum",
        dpi=150,
        save_format="svg",
        color_palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
    )

    visualizer2 = CircuitPerformanceVisualizer(
        metrics_dict, output_dir="img/demo_plots_custom", config=custom_config
    )
    figures2 = visualizer2.plot_all(save=True, show=False)
    print(f"✅ Created {len(figures2)} plots with custom configuration")

    # Demo 3: Individual plot methods
    print("\n" + "=" * 40)
    print("Demo 3: Individual Plot Methods")
    print("=" * 40)

    visualizer3 = CircuitPerformanceVisualizer(metrics_dict, output_dir="img/demo_plots_individual")

    # Create specific plots
    print("Creating success vs error comparison...")
    visualizer3.plot_success_error_comparison(save=True, show=False)

    print("Creating fidelity comparison...")
    visualizer3.plot_fidelity_comparison(save=True, show=False)

    print("Creating shot distribution...")
    visualizer3.plot_shot_distribution(save=True, show=False)

    print("Creating aggregate summary...")
    visualizer3.plot_aggregate_summary(save=True, show=False)

    print("✅ Created 4 individual plots")

    # Demo 4: Dashboard view
    print("\n" + "=" * 40)
    print("Demo 4: Dashboard View")
    print("=" * 40)

    visualizer4 = CircuitPerformanceVisualizer(metrics_dict, output_dir="img/demo_plots_dashboard")
    visualizer4.create_dashboard(save=True, show=False)
    print("✅ Created comprehensive dashboard")

    print("\n" + "=" * 40)
    print("Demo Complete!")
    print("=" * 40)
    print("Check the following directories for generated plots:")
    print("- img/demo_plots_default/")
    print("- img/demo_plots_custom/")
    print("- img/demo_plots_individual/")
    print("- img/demo_plots_dashboard/")


if __name__ == "__main__":
    demo_circuit_performance_visualization()
