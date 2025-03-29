# from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
# from qiskit.circuit.library import XGate, HGate
from qiskit.primitives import PrimitiveResult
from qiskit.providers import Backend
from qiskit_aer import AerSimulator

# from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
# from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
# from qiskit.visualization import plot_histogram
# from qiskit.exceptions import QiskitError
import numpy as np
import os
from typing import Union, List
from dotenv import load_dotenv
from .base_validator import BaseValidator


class TeleportationValidator(BaseValidator):
    """
    Validator for quantum teleportation protocol.
    Implements the standard teleportation protocol with optional gate operations.
    """

    def __init__(
        self,
        payload_size: int = 3,
        gates: Union[List, int] = None,
        use_barriers: bool = True,
        save_statevector: bool = False,
    ):
        """
        Initialize the teleportation validator.

        Args:
            payload_size (int): Size of the payload qubit
            gates (Union[List, int]): List of gates to apply to the payload qubit
            use_barriers (bool): Whether to include barriers in the circuit
            save_statevector (bool): Whether to save the statevector
        """
        super().__init__(
            num_qubits=3, num_clbits=2, use_barriers=use_barriers, name="teleportation"
        )
        self.payload_size = payload_size
        self.gates = gates if isinstance(gates, list) else [gates] if gates is not None else []
        self.save_statevector = save_statevector

        # Create the circuit
        self._create_circuit()

    def _create_circuit(self):
        """Create the quantum teleportation circuit."""
        # Prepare the payload qubit (q[0])
        for gate in self.gates:
            if gate == "h":
                self.h(0)
            elif gate == "x":
                self.x(0)

        if self.use_barriers:
            self.barrier()

        # Create Bell pair between Alice (q[1]) and Bob (q[2])
        self.h(1)
        self.cx(1, 2)

        if self.use_barriers:
            self.barrier()

        # Alice's operations
        self.cx(0, 1)
        self.h(0)

        if self.use_barriers:
            self.barrier()

        # Alice's measurements
        self.measure(0, 0)
        self.measure(1, 1)

        if self.use_barriers:
            self.barrier()

        # Bob's conditional operations
        self.x(2).c_if(1, 1)
        self.z(2).c_if(0, 1)

    def _run(self, circuits, parameter_values=None, **run_options):
        """
        Run the teleportation circuit as a Qiskit primitive.

        Args:
            circuits: The circuits to run
            parameter_values: Optional parameter values
            **run_options: Additional run options

        Returns:
            PrimitiveResult: The results of the primitive
        """
        # For now, we'll just run the simulation
        simulator = AerSimulator()
        result = simulator.run(self).result()

        # Create a PrimitiveResult object
        return PrimitiveResult(
            data={
                "counts": result.get_counts(),
                "statevector": result.get_statevector() if self.save_statevector else None,
            }
        )

    def get_ibm_credentials(self):
        """
        Get IBM Quantum credentials from environment variables.

        Returns:
            tuple: (channel, token)
        """
        load_dotenv()
        channel = os.getenv("IBM_QUANTUM_CHANNEL", "ibm_quantum")
        token = os.getenv("IBM_QUANTUM_TOKEN")
        return channel, token

    def get_generic_backend(self) -> Backend:
        """
        Get a generic backend for simulation.

        Returns:
            Backend: The backend instance
        """
        return AerSimulator()

    def validate(self, expected_state: np.ndarray = None):
        """
        Validate the teleportation circuit.

        Args:
            expected_state (np.ndarray, optional): Expected final state vector

        Returns:
            dict: Validation results
        """
        # Run simulation
        results = self.run_simulation(show_histogram=True)

        # Calculate success metrics
        success_rate = results["results_metrics"]["success_rate"]
        circuit_metrics = results["circuit_metrics"]

        validation_results = {
            "success_rate": success_rate,
            "circuit_metrics": circuit_metrics,
            "is_valid": success_rate > 0.5,  # Basic validation threshold
        }

        if expected_state is not None:
            # Add state vector comparison if expected state is provided
            from qiskit_aer import AerSimulator

            simulator = AerSimulator()
            result = simulator.run(self, save_statevector=True).result()
            final_state = result.get_statevector()

            # Calculate fidelity
            fidelity = np.abs(np.vdot(expected_state, final_state)) ** 2
            validation_results["fidelity"] = fidelity
            validation_results["is_valid"] = validation_results["is_valid"] and fidelity > 0.9

        return validation_results
