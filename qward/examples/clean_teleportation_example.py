"""
Clean teleportation example using the refactored v_tp.py and QuantumCircuitExecutor.

This example demonstrates the new clean separation between circuit generation
and execution, showing how to use the TeleportationCircuitGenerator with
the QuantumCircuitExecutor for comprehensive analysis.
"""

from qward.algorithms.v_tp import (
    VariationTeleportationProtocol,
    StandardTeleportationProtocol,
    TeleportationCircuitGenerator,
)
from qward.algorithms.executor import QuantumCircuitExecutor


def teleportation_success(outcome):
    """Success criteria: all auxiliary qubits should measure 0 after teleportation validation."""
    clean = outcome.replace(" ", "")
    return all(bit == "0" for bit in clean)


def basic_teleportation_example():
    """Basic example with variation teleportation protocol."""
    print("🌀 BASIC TELEPORTATION EXAMPLE")
    print("=" * 50)

    # Create teleportation circuit generator
    generator = TeleportationCircuitGenerator(
        payload_size=2, gates=["x", "h"], protocol_type="variation"  # Apply X and H gates
    )

    print("Generated circuit:")
    print(generator.circuit.draw())
    print()

    # Create executor
    executor = QuantumCircuitExecutor(shots=1024)

    # Simulate directly with executor
    print("🚀 Running simulation...")
    results = executor.simulate(
        generator.circuit, success_criteria=teleportation_success, show_results=True
    )

    return generator, executor, results


def protocol_comparison_example():
    """Compare different teleportation protocols."""
    print("\n\n🔬 PROTOCOL COMPARISON EXAMPLE")
    print("=" * 50)

    # Create executor
    executor = QuantumCircuitExecutor(shots=2048)

    protocols = {
        "Variation": VariationTeleportationProtocol(),
        "Standard": StandardTeleportationProtocol(),
    }

    results = {}

    for name, protocol in protocols.items():
        print(f"\n📊 Testing {name} Protocol")
        print("-" * 30)

        # Create generator with specific protocol
        generator = TeleportationCircuitGenerator(
            protocol=protocol, payload_size=1, gates=["x"]  # Simple X gate test
        )

        # Simulate
        result = executor.simulate(
            generator.circuit, success_criteria=teleportation_success, show_results=True
        )
        results[name] = result

        print(f"✅ {name} protocol completed\n")

    return results


def noise_analysis_example():
    """Analyze teleportation under different noise conditions."""
    print("\n\n🌪️ NOISE ANALYSIS EXAMPLE")
    print("=" * 50)

    # Create teleportation circuit
    generator = TeleportationCircuitGenerator(
        payload_size=1, gates=["h"], protocol_type="variation"  # Hadamard gate test
    )

    # Create executor
    executor = QuantumCircuitExecutor(shots=1024)

    # Test different noise levels
    noise_levels = [0.0, 0.02, 0.05, 0.1]

    for noise_level in noise_levels:
        print(f"\n🎯 Testing with {noise_level*100}% depolarizing noise")
        print("-" * 40)

        # Simulate with noise
        result = executor.simulate(
            generator.circuit,
            success_criteria=teleportation_success,
            show_results=True,
            noise_model="depolarizing" if noise_level > 0 else None,
            noise_level=noise_level,
        )

        print(f"✅ Noise level {noise_level*100}% completed\n")


def custom_gates_example():
    """Example with custom gate sequences."""
    print("\n\n🛠️ CUSTOM GATES EXAMPLE")
    print("=" * 50)

    # Create generator with multiple random gates
    generator = TeleportationCircuitGenerator(
        payload_size=3, gates=5, protocol_type="variation"  # Generate 5 random gates
    )

    print("Generated gates:", generator.input_gates)
    print("Circuit:")
    print(generator.circuit.draw())
    print()

    # Create executor
    executor = QuantumCircuitExecutor(shots=2048)

    # Simulate with mixed noise
    print("🚀 Running simulation with mixed noise...")
    results = executor.simulate(
        generator.circuit,
        success_criteria=teleportation_success,
        show_results=True,
        noise_model="mixed",
        noise_level=0.03,
    )

    return generator, results


def main():
    """Run all examples."""
    print("🚀 CLEAN TELEPORTATION EXAMPLES")
    print("=" * 80)

    # Basic example
    basic_teleportation_example()

    # Protocol comparison
    protocol_comparison_example()

    # Noise analysis
    noise_analysis_example()

    # Custom gates
    custom_gates_example()

    print("\n🎉 ALL EXAMPLES COMPLETED!")
    print("The v_tp.py file is now clean and modular!")
    print("- Circuit generation is separate from execution")
    print("- Use QuantumCircuitExecutor directly: executor.simulate(generator.circuit, ...)")
    print("- Easy integration with qward metrics and visualization")


if __name__ == "__main__":
    main()
