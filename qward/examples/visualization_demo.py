"""
Demo showing how to use QWARD's CircuitPerformanceVisualizer visualization capabilities.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner
from qward.metrics import CircuitPerformanceMetrics
from qward.visualization import CircuitPerformanceVisualizer, PlotConfig


def demo_circuit_performance_visualization():
    """
    Demonstrate CircuitPerformanceVisualizer visualization with different configurations.
    """
    print("QWARD CircuitPerformanceVisualizer Visualization Demo")
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

    # Create scanner and add CircuitPerformanceMetrics strategy
    scanner = Scanner(circuit=circuit)
    circuit_performance_strategy = CircuitPerformanceMetrics(circuit=circuit)

    # Add all jobs
    circuit_performance_strategy.add_job(jobs)
    scanner.add_strategy(circuit_performance_strategy)

    # Calculate metrics
    print("\nCalculating CircuitPerformanceMetrics metrics...")
    metrics_dict = scanner.calculate_metrics()

    print("\nCircuitPerformanceMetrics metrics calculated successfully!")
    print(f"Individual jobs: {len(metrics_dict['CircuitPerformance.individual_jobs'])} jobs")
    print(f"Aggregate data: {metrics_dict['CircuitPerformance.aggregate'].shape}")

    # Demo 1: Default visualization using strategy directly
    print("\n" + "=" * 40)
    print("Demo 1: Default Visualization (Direct Strategy)")
    print("=" * 40)

    strategy1 = CircuitPerformanceVisualizer(
        metrics_dict, output_dir="qward/examples/img/demo_plots_default"
    )
    figures1 = strategy1.plot_all(save=True, show=False)
    print(f"✅ Created {len(figures1)} plots with default settings")

    # Demo 2: Custom configuration using strategy directly
    print("\n" + "=" * 40)
    print("Demo 2: Custom Configuration (Direct Strategy)")
    print("=" * 40)

    custom_config = PlotConfig(
        figsize=(12, 8),
        style="quantum",
        dpi=300,
        save_format="svg",
        color_palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
    )

    strategy2 = CircuitPerformanceVisualizer(
        metrics_dict, output_dir="qward/examples/img/demo_plots_custom", config=custom_config
    )
    figures2 = strategy2.plot_all(save=True, show=False)
    print(f"✅ Created {len(figures2)} plots with custom configuration")

    # Demo 3: Individual plot methods using strategy directly
    print("\n" + "=" * 40)
    print("Demo 3: Individual Plot Methods (Direct Strategy)")
    print("=" * 40)

    strategy3 = CircuitPerformanceVisualizer(
        metrics_dict, output_dir="qward/examples/img/demo_plots_individual"
    )

    # Create specific plots
    print("Creating success vs error comparison...")
    strategy3.plot_success_error_comparison(save=True, show=False)

    print("Creating fidelity comparison...")
    strategy3.plot_fidelity_comparison(save=True, show=False)

    print("Creating shot distribution...")
    strategy3.plot_shot_distribution(save=True, show=False)

    print("Creating aggregate summary...")
    strategy3.plot_aggregate_summary(save=True, show=False)

    print("✅ Created 4 individual plots")

    # Demo 4: Dashboard view using strategy directly
    print("\n" + "=" * 40)
    print("Demo 4: Dashboard View (Direct Strategy)")
    print("=" * 40)

    strategy4 = CircuitPerformanceVisualizer(
        metrics_dict, output_dir="qward/examples/img/demo_plots_dashboard"
    )
    strategy4.create_dashboard(save=True, show=False)
    print("✅ Created comprehensive dashboard")

    # Demo 5: Using the unified Visualizer (recommended approach)
    print("\n" + "=" * 40)
    print("Demo 5: Using Unified Visualizer (Recommended)")
    print("=" * 40)

    from qward.visualization import Visualizer

    # Create visualizer from scanner (this is the recommended approach)
    visualizer = Visualizer(scanner=scanner, output_dir="qward/examples/img/demo_plots_unified")

    # Show available metrics
    print("Available metrics:", visualizer.get_available_metrics())

    # Create all dashboards
    dashboards = visualizer.create_dashboard(save=True, show=False)
    print(f"✅ Created {len(dashboards)} unified dashboards")

    # Create all individual plots
    all_figures = visualizer.visualize_all(save=True, show=False)
    print(f"✅ Created visualizations for {len(all_figures)} metric types")

    print("\n" + "=" * 40)
    print("Demo Complete!")
    print("=" * 40)
    print("Check the following directories for generated plots:")
    print("- qward/examples/img/demo_plots_default/")
    print("- qward/examples/img/demo_plots_custom/")
    print("- qward/examples/img/demo_plots_individual/")
    print("- qward/examples/img/demo_plots_dashboard/")
    print("- qward/examples/img/demo_plots_unified/")
    print("\nRecommended approach: Use the unified Visualizer class (Demo 5)")
    print("Direct strategy usage (Demos 1-4) is useful for advanced customization.")


if __name__ == "__main__":
    demo_circuit_performance_visualization()
