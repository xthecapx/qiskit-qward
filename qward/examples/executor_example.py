"""
Example usage of QuantumCircuitExecutor with different noise models.

This script demonstrates how to use the QuantumCircuitExecutor class to run
quantum circuits with various noise models and analyze the results using qward.
"""

from qiskit import QuantumCircuit
from qward.algorithms.executor import QuantumCircuitExecutor
from qiskit_aer.noise import NoiseModel, depolarizing_error


def all_zeros_success(outcome):
    """Success criteria: all measurement outcomes must be 0."""
    clean = outcome.replace(" ", "")
    return all(bit == "0" for bit in clean)


def basic_usage_example():
    """Basic usage example with Bell state circuit."""
    print("🔬 BASIC QUANTUM CIRCUIT EXECUTOR USAGE")
    print("=" * 60)

    # Create a simple Bell state circuit
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    print("Circuit:")
    print(circuit.draw())
    print()

    # Create executor (noise models are now specified per simulation)
    executor = QuantumCircuitExecutor(shots=1024)

    # Example 1: No noise (ideal simulation)
    print("\n📊 Example 1: Ideal simulation (no noise)")
    print("-" * 40)
    results_ideal = executor.simulate(
        circuit, success_criteria=all_zeros_success, show_results=True
    )

    return executor, circuit, results_ideal


def noise_model_examples(executor, circuit):
    """Examples with different noise models."""
    print("\n\n🌪️ NOISE MODEL EXAMPLES")
    print("=" * 60)

    # Example 2: Depolarizing noise model
    print("\n📊 Example 2: Depolarizing noise (5% error rate)")
    print("-" * 50)
    results_depol = executor.simulate(
        circuit,
        success_criteria=all_zeros_success,
        show_results=True,
        noise_model="depolarizing",
        noise_level=0.05,  # 5% error rate
    )

    # Example 3: Pauli noise model
    print("\n📊 Example 3: Pauli noise (10% error rate)")
    print("-" * 50)
    results_pauli = executor.simulate(
        circuit,
        success_criteria=all_zeros_success,
        show_results=True,
        noise_model="pauli",
        noise_level=0.1,  # 10% error rate
    )

    # Example 4: Mixed noise model
    print("\n📊 Example 4: Mixed noise (3% error rate)")
    print("-" * 50)
    results_mixed = executor.simulate(
        circuit,
        success_criteria=all_zeros_success,
        show_results=True,
        noise_model="mixed",
        noise_level=0.03,  # 3% error rate
    )

    return results_depol, results_pauli, results_mixed


def custom_noise_example(executor, circuit):
    """Example with custom noise model."""
    print("\n\n🛠️ CUSTOM NOISE MODEL EXAMPLE")
    print("=" * 60)

    # Example 5: Custom noise model
    print("\n📊 Example 5: Custom noise model")
    print("-" * 40)

    custom_noise = NoiseModel()
    custom_noise.add_all_qubit_quantum_error(depolarizing_error(0.02, 1), ["h", "x"])
    custom_noise.add_all_qubit_quantum_error(depolarizing_error(0.05, 2), ["cx"])

    results_custom = executor.simulate(
        circuit, success_criteria=all_zeros_success, show_results=True, noise_model=custom_noise
    )

    return results_custom


def compare_results(*results_list):
    """Compare results from different noise models."""
    print("\n\n📈 RESULTS COMPARISON")
    print("=" * 60)

    names = ["Ideal", "Depolarizing", "Pauli", "Mixed", "Custom"]

    print("Comparison of success rates:")
    for name, results in zip(names, results_list):
        success_rate = results.get("success_rate", "N/A")
        print(f"{name:12} | Success Rate: {success_rate}")

    print("\nqWard metrics summary:")
    for name, results in zip(names[:2], results_list[:2]):  # Show first two for brevity
        print(f"\n{name} qward metrics:")
        for metric_name, df in results["qward_metrics"].items():
            print(f"  - {metric_name}: {len(df)} rows")
            if "CircuitPerformance" in metric_name and len(df) > 0:
                # Try to extract success rate from CircuitPerformance metrics
                if "success_metrics.success_rate" in df.columns:
                    success_rate = df.iloc[0]["success_metrics.success_rate"]
                    print(f"    Success rate: {success_rate:.3f}")


def qbraid_example(executor, circuit):
    """Example with qBraid execution (if available)."""
    print("\n\n☁️ QBRAID EXECUTION EXAMPLE")
    print("=" * 60)

    try:
        print("Attempting qBraid execution...")
        qbraid_results = executor.run_qbraid(circuit, success_criteria=all_zeros_success)
        print(f"✅ qBraid execution completed: {qbraid_results['status']}")

        if qbraid_results["status"] == "completed":
            print(f"Results: {qbraid_results['counts']}")
        else:
            print(f"Status details: {qbraid_results}")

    except ImportError:
        print("❌ qBraid not available - install with: pip install qbraid")
    except Exception as e:
        print(f"❌ qBraid execution failed: {e}")


def main():
    """Run all examples."""
    print("🚀 QUANTUM CIRCUIT EXECUTOR EXAMPLES")
    print("=" * 80)

    # Basic usage
    executor, circuit, results_ideal = basic_usage_example()

    # Noise model examples
    results_depol, results_pauli, results_mixed = noise_model_examples(executor, circuit)

    # Custom noise example
    results_custom = custom_noise_example(executor, circuit)

    # Compare all results
    compare_results(results_ideal, results_depol, results_pauli, results_mixed, results_custom)

    # qBraid example
    qbraid_example(executor, circuit)

    print("\n🎉 ALL EXAMPLES COMPLETED!")
    print("Check the generated plots in qward/examples/img/ for detailed visualizations.")


if __name__ == "__main__":
    main()
