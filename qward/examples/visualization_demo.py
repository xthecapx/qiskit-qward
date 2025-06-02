"""
Demo showing how to use QWARD's CircuitPerformanceVisualizer visualization capabilities.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner
from qward.metrics import CircuitPerformanceMetrics
from qward.visualization import CircuitPerformanceVisualizer, PlotConfig
from qward.visualization.constants import Metrics, Plots


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

    # Use new API to generate all plots
    figures1 = strategy1.generate_all_plots(save=True, show=False)
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

    # Use new API to generate all plots with custom config
    figures2 = strategy2.generate_all_plots(save=True, show=False)
    print(f"✅ Created {len(figures2)} plots with custom configuration")

    # Demo 3: Individual plot methods using strategy directly
    print("\n" + "=" * 40)
    print("Demo 3: Individual Plot Methods (Direct Strategy)")
    print("=" * 40)

    strategy3 = CircuitPerformanceVisualizer(
        metrics_dict, output_dir="qward/examples/img/demo_plots_individual"
    )

    # Create specific plots using new API
    print("Creating success vs error comparison...")
    strategy3.generate_plot(
        Plots.CircuitPerformance.SUCCESS_ERROR_COMPARISON, save=True, show=False
    )
    print("   ✓ Success vs error comparison plot created")
    strategy3.generate_plot(Plots.CircuitPerformance.FIDELITY_COMPARISON, save=True, show=False)
    print("   ✓ Fidelity comparison plot created")
    strategy3.generate_plot(Plots.CircuitPerformance.SHOT_DISTRIBUTION, save=True, show=False)
    print("   ✓ Shot distribution plot created")
    strategy3.generate_plot(Plots.CircuitPerformance.AGGREGATE_SUMMARY, save=True, show=False)
    print("   ✓ Aggregate summary plot created")

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

    # Create all individual plots using new API
    all_figures = visualizer.generate_plots(
        {Metrics.CIRCUIT_PERFORMANCE: None}, save=True, show=False  # None = all plots
    )
    print(f"✅ Created visualizations for {len(all_figures)} metric types")

    # Demo 6: Granular plot selection with new API
    print("\n" + "=" * 40)
    print("Demo 6: Granular Plot Selection (New API)")
    print("=" * 40)

    # Generate specific plots using new API
    selected_plots = strategy3.generate_plots(
        [
            Plots.CircuitPerformance.SUCCESS_ERROR_COMPARISON,
            Plots.CircuitPerformance.FIDELITY_COMPARISON,
        ],
        save=True,
        show=False,
    )

    # When calling generate_plots() on an individual visualizer, it returns a dict of plot_name -> figure
    print(f"✅ Created {len(selected_plots)} selected CircuitPerformance plots")

    # Show plot metadata
    print("\nAvailable CircuitPerformance plots and their metadata:")
    available_plots = visualizer.get_available_plots(Metrics.CIRCUIT_PERFORMANCE)
    for plot_name in available_plots[Metrics.CIRCUIT_PERFORMANCE]:
        metadata = visualizer.get_plot_metadata(Metrics.CIRCUIT_PERFORMANCE, plot_name)
        print(f"  - {plot_name}: {metadata.description} ({metadata.plot_type.value})")

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
    print("\nNew API Benefits:")
    print("- Type-safe constants prevent typos")
    print("- Granular control over plot selection")
    print("- Rich metadata for each plot")
    print("- IDE autocompletion support")


if __name__ == "__main__":
    demo_circuit_performance_visualization()
