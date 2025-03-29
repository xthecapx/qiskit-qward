"""
Example demonstrating the use of Qiskit Qward teleportation validator.

This example creates a teleportation validator and runs a simulation.
"""

# Standard imports
# Third-party imports
from qiskit.exceptions import QiskitError

# Local imports
from qiskit_qward.validators.teleportation_validator import TeleportationValidator

# Comment out the experiments import as it might not exist yet
# from qiskit_qward.experiments.experiments import Experiments


def main():
    """
    Create a teleportation validator and run simulations.

    Demonstrates basic capabilities of the Qward framework.
    """
    # Create a teleportation validator
    validator = TeleportationValidator(
        payload_size=3,
        gates=["h", "x"],  # Apply Hadamard and X gates to the payload qubit
        use_barriers=True,
    )

    # Create experiments framework - commented out since it might not exist yet
    # experiments = Experiments()

    # Just show the validator features that are available
    print("Teleportation Circuit:")
    print(f"Circuit depth: {validator.depth()}")
    print(f"Operation count: {validator.count_ops()}")

    # Run simulation if available
    try:
        print("\nRunning simulation...")
        results = validator.run_simulation(
            show_histogram=False
        )  # Set show_histogram to False to avoid display issues in CI
        print(f"Simulation results available: {bool(results)}")
    except (QiskitError, ImportError, RuntimeError) as e:
        print(f"Simulation error: {e}")

    # Use Qiskit features directly
    validator.draw()  # Draw the circuit


if __name__ == "__main__":
    main()
