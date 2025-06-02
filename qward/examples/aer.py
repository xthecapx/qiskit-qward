"""
Example script demonstrating the use of QWARD with Aer simulator.

This script shows how to use QWARD to analyze quantum circuits using the Aer simulator.
"""

from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics
from qward.examples.utils import create_example_circuit, get_display

from qiskit_aer import AerSimulator, AerJob
from qiskit import QuantumCircuit

display = get_display()


def example_default_strategies(circuit: QuantumCircuit):
    """
    Example 1: Using default strategies

    This example demonstrates how to use the default strategies provided by QWARD.
    The default strategies include QiskitMetrics and ComplexityMetrics.

    Args:
        circuit: The quantum circuit to analyze
    """
    print("\nExample 1: Using default strategies")

    # Create a scanner with the circuit
    scanner = Scanner(circuit=circuit)

    # Add default strategies
    from qward.metrics.defaults import get_default_strategies

    default_strategies = get_default_strategies()
    for strategy in default_strategies:
        scanner.add_strategy(strategy(circuit))

    # Calculate metrics
    metrics_dict = scanner.calculate_metrics()

    # Display each metric DataFrame
    for metric_name, df in metrics_dict.items():
        print(f"\n{metric_name} DataFrame:")
        display(df)

    return scanner


def example_qiskit_strategy(circuit: QuantumCircuit):
    """
    Example 2: Using QiskitMetrics strategy

    This example demonstrates how to use the QiskitMetrics strategy to analyze
    the basic properties of a quantum circuit.

    Args:
        circuit: The quantum circuit to analyze
    """
    print("\nExample 2: Using QiskitMetrics strategy")

    # Create a scanner with the circuit
    scanner = Scanner(circuit=circuit)

    # Add QiskitMetrics strategy
    scanner.add_strategy(QiskitMetrics(circuit))

    # Calculate metrics
    metrics_dict = scanner.calculate_metrics()

    # Display the QiskitMetrics DataFrame
    print("\nQiskitMetrics DataFrame:")
    display(metrics_dict["QiskitMetrics"])

    return scanner


def example_constructor_strategies(circuit: QuantumCircuit):
    """
    Example 3: Using strategies in constructor

    This example demonstrates how to pass strategies directly to the Scanner constructor.

    Args:
        circuit: The quantum circuit to analyze
    """
    print("\nExample 3: Using strategies in constructor")

    # Create a scanner with strategies in constructor
    scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics])

    # Calculate metrics
    metrics_dict = scanner.calculate_metrics()

    # Display each metric DataFrame
    for metric_name, df in metrics_dict.items():
        print(f"\n{metric_name} DataFrame:")
        display(df)

    return scanner


def example_complexity_metrics(circuit: QuantumCircuit):
    """
    Example 4: Using ComplexityMetrics strategy

    This example demonstrates how to use the ComplexityMetrics strategy to analyze
    the complexity of a quantum circuit.

    Args:
        circuit: The quantum circuit to analyze
    """
    print("\nExample 4: Using ComplexityMetrics strategy")

    # Create a scanner with the circuit
    scanner = Scanner(circuit=circuit)

    # Add ComplexityMetrics strategy
    scanner.add_strategy(ComplexityMetrics(circuit))

    # Calculate metrics
    metrics_dict = scanner.calculate_metrics()

    # Display the ComplexityMetrics DataFrame
    print("\nComplexityMetrics DataFrame:")
    display(metrics_dict["ComplexityMetrics"])

    return scanner


def example_multiple_strategies(circuit: QuantumCircuit):
    """
    Example 5: Using multiple strategies

    This example demonstrates how to use multiple strategies together.

    Args:
        circuit: The quantum circuit to analyze
    """
    print("\nExample 5: Using multiple strategies")

    # Create a scanner with the circuit
    scanner = Scanner(circuit=circuit)

    # Add multiple strategies
    scanner.add_strategy(QiskitMetrics(circuit))
    scanner.add_strategy(ComplexityMetrics(circuit))

    # Calculate metrics
    metrics_dict = scanner.calculate_metrics()

    # Display each metric DataFrame
    for metric_name, df in metrics_dict.items():
        print(f"\n{metric_name} DataFrame:")
        display(df)

    return scanner


def example_circuit_performance_metrics(circuit: QuantumCircuit):
    """
    Example 6: Using CircuitPerformance strategy

    This example demonstrates how to use the CircuitPerformance strategy to analyze
    the performance of a quantum circuit execution.

    Args:
        circuit: The quantum circuit to analyze
    """
    print("\nExample 6: Using CircuitPerformance strategy")

    # Create an Aer simulator
    simulator = AerSimulator()

    # Run the circuit
    job = simulator.run(circuit, shots=1024)

    # Get the result
    result = job.result()

    print("Job result:")
    print(result)

    # Get the counts from the job result
    counts = result.get_counts()

    print("counts:")
    print(counts)

    # Create a scanner with the circuit and job
    scanner = Scanner(circuit=circuit, job=job)

    # Add CircuitPerformance strategy
    circuit_performance_strategy = CircuitPerformanceMetrics(
        circuit=circuit, job=job, success_criteria=lambda x: x == "11"
    )
    scanner.add_strategy(circuit_performance_strategy)

    # Calculate metrics
    metrics_dict = scanner.calculate_metrics()

    # Display both individual and aggregate metrics
    print("\nCircuitPerformance individual jobs DataFrame:")
    display(metrics_dict["CircuitPerformance.individual_jobs"])

    print("\nCircuitPerformance aggregate metrics DataFrame:")
    if "CircuitPerformance.aggregate" in metrics_dict:
        display(metrics_dict["CircuitPerformance.aggregate"])
    else:
        print("Note: Aggregate metrics only available for multiple jobs.")

    return scanner, job


def example_all_strategies(circuit: QuantumCircuit, job: AerJob):
    """
    Example 7: Using all strategies together

    This example demonstrates how to use all strategies together by creating QiskitMetrics,
    ComplexityMetrics, and CircuitPerformance instances and adding them to the Scanner.

    Args:
        circuit: The quantum circuit to analyze
        job: The Aer job from the simulator
    """
    print("\nExample 7: Using all strategies together")

    # Get the result
    result = job.result()

    # Get the counts from the job result
    counts = result.get_counts()

    # Create a scanner with the circuit and job
    scanner = Scanner(circuit=circuit, job=job)

    # Add all strategies
    scanner.add_strategy(QiskitMetrics(circuit))
    scanner.add_strategy(ComplexityMetrics(circuit))
    scanner.add_strategy(
        CircuitPerformanceMetrics(circuit=circuit, job=job, success_criteria=lambda x: x == "11")
    )

    # Calculate metrics
    metrics_dict = scanner.calculate_metrics()

    # Display each metric DataFrame
    for metric_name, df in metrics_dict.items():
        # Skip CircuitPerformance metrics as we'll handle them separately
        if not metric_name.startswith("CircuitPerformance"):
            print(f"\n{metric_name} DataFrame:")
            display(df)

    # Display CircuitPerformance metrics separately
    if "CircuitPerformance.individual_jobs" in metrics_dict:
        print("\nCircuitPerformance Individual Jobs DataFrame:")
        display(metrics_dict["CircuitPerformance.individual_jobs"])

    if "CircuitPerformance.aggregate" in metrics_dict:
        print("\nCircuitPerformance Aggregate Metrics DataFrame:")
        display(metrics_dict["CircuitPerformance.aggregate"])

    return scanner


def example_multiple_jobs_success_rate(circuit: QuantumCircuit):
    """
    Example 8: Using CircuitPerformance with multiple Aer simulator jobs

    This example demonstrates how to use CircuitPerformance with multiple Aer simulator jobs
    to analyze the success rate across different runs. It includes comprehensive
    visualizations showing:
    - Individual job success vs error rates
    - Fidelity comparison across jobs
    - Shot distribution (successful vs failed)
    - Aggregate statistics summary

    Args:
        circuit: The quantum circuit to analyze
    """
    print("\nExample 8: Using CircuitPerformance with multiple Aer simulator jobs")

    # Import noise model components
    from qiskit_aer.noise import (
        NoiseModel,
        ReadoutError,
        pauli_error,
        depolarizing_error,
    )

    # Create an Aer simulator with default settings (no noise)
    simulator = AerSimulator()

    # Run the circuit multiple times with different noise models
    jobs = []

    # Run with default noise model (no noise)
    job1 = simulator.run(circuit, shots=1024)
    jobs.append(job1)

    # Create a noise model with depolarizing errors
    noise_model1 = NoiseModel()

    # Add depolarizing error to all single qubit gates
    depol_error = depolarizing_error(0.05, 1)  # 5% depolarizing error
    noise_model1.add_all_qubit_quantum_error(depol_error, ["u1", "u2", "u3"])

    # Add depolarizing error to all two qubit gates
    depol_error_2q = depolarizing_error(0.1, 2)  # 10% depolarizing error
    noise_model1.add_all_qubit_quantum_error(depol_error_2q, ["cx"])

    # Add readout error
    readout_error = ReadoutError([[0.9, 0.1], [0.1, 0.9]])  # 10% readout error
    noise_model1.add_all_qubit_readout_error(readout_error)

    # Create a simulator with the first noise model
    noisy_simulator1 = AerSimulator(noise_model=noise_model1)
    job2 = noisy_simulator1.run(circuit, shots=1024)
    jobs.append(job2)

    # Create a noise model with Pauli errors
    noise_model2 = NoiseModel()

    # Add Pauli error to all single qubit gates
    pauli_error_1q = pauli_error([("X", 0.05), ("Y", 0.05), ("Z", 0.05), ("I", 0.85)])
    noise_model2.add_all_qubit_quantum_error(pauli_error_1q, ["u1", "u2", "u3"])

    # Add Pauli error to all two qubit gates
    pauli_error_2q = pauli_error([("XX", 0.05), ("YY", 0.05), ("ZZ", 0.05), ("II", 0.85)])
    noise_model2.add_all_qubit_quantum_error(pauli_error_2q, ["cx"])

    # Add readout error
    readout_error = ReadoutError([[0.95, 0.05], [0.05, 0.95]])  # 5% readout error
    noise_model2.add_all_qubit_readout_error(readout_error)

    # Create a simulator with the second noise model
    noisy_simulator2 = AerSimulator(noise_model=noise_model2)
    job3 = noisy_simulator2.run(circuit, shots=1024)
    jobs.append(job3)

    # Wait for all jobs to complete
    for job in jobs:
        job.result()

    # Create a scanner with the circuit
    scanner = Scanner(circuit=circuit)

    # Add CircuitPerformance strategy with multiple jobs
    circuit_performance_strategy = CircuitPerformanceMetrics(
        circuit=circuit, success_criteria=lambda x: x == "11"
    )

    # Add jobs one by one to demonstrate the new functionality
    circuit_performance_strategy.add_job(jobs[0])  # Add first job
    circuit_performance_strategy.add_job(jobs[1:])  # Add remaining jobs as a list

    scanner.add_strategy(circuit_performance_strategy)

    # Calculate metrics
    metrics_dict = scanner.calculate_metrics()

    # Display the individual jobs metrics
    print("\nCircuitPerformance individual jobs DataFrame:")
    display(metrics_dict["CircuitPerformance.individual_jobs"])

    # Display the aggregate metrics
    print("\nCircuitPerformance aggregate metrics DataFrame:")
    display(metrics_dict["CircuitPerformance.aggregate"])

    # Create visualizations using QWARD's visualization module
    from qward.visualization import CircuitPerformanceVisualizer, PlotConfig
    from qward.visualization.constants import Metrics, Plots

    print("\nCreating visualizations using QWARD CircuitPerformanceVisualizer...")

    # Create a custom configuration for high-quality plots
    config = PlotConfig(figsize=(10, 6), style="quantum", dpi=300, save_format="png")

    # Extract CircuitPerformance data for visualization
    circuit_perf_data = {
        k: v for k, v in metrics_dict.items() if k.startswith("CircuitPerformance")
    }

    # Create visualizer with custom configuration
    visualizer = CircuitPerformanceVisualizer(
        metrics_dict=circuit_perf_data, output_dir="qward/examples/img", config=config
    )

    # Create comprehensive dashboard
    print("\nCreating CircuitPerformance visualization dashboard...")
    dashboard_fig = visualizer.create_dashboard(save=True, show=False)
    print("✅ Dashboard saved to qward/examples/img/")

    # Create all individual plots using new API
    print("Creating individual CircuitPerformance plots...")
    all_figures = visualizer.generate_all_plots(save=True, show=False)
    print(f"✅ Created {len(all_figures)} individual plots")

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

    # Show available plots and metadata
    print("\nAvailable CircuitPerformance plots:")
    available_plots = visualizer.get_available_plots()
    for plot_name in available_plots:
        metadata = visualizer.get_plot_metadata(plot_name)
        print(f"  - {plot_name}: {metadata.description} ({metadata.plot_type.value})")

    return scanner, jobs


def main():
    """
    Main function to run all examples.
    """
    print("QWARD Aer Examples")
    print("=" * 50)

    # Create a simple quantum circuit for testing
    circuit = create_example_circuit()

    print("Quantum Circuit:")
    display(circuit.draw(output="mpl"))

    # Run examples
    example_default_strategies(circuit)
    example_qiskit_strategy(circuit)
    example_constructor_strategies(circuit)
    example_complexity_metrics(circuit)
    example_multiple_strategies(circuit)
    scanner, job = example_circuit_performance_metrics(circuit)
    example_all_strategies(circuit, job)
    example_multiple_jobs_success_rate(circuit)

    print("\n" + "=" * 50)
    print("All examples completed successfully!")


if __name__ == "__main__":
    main()
