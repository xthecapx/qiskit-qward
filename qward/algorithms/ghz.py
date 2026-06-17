"""
GHZ state preparation algorithm.

Prepares the maximally entangled GHZ state: (|0...0> + |1...1>) / sqrt(2)
using a Hadamard gate followed by a CNOT chain.
"""

from typing import List

from qiskit import QuantumCircuit


class GHZ:
    """GHZ state preparation circuit.

    Creates a maximally entangled state across n qubits.
    Circuit depth is O(n) — linear in qubit count.

    Args:
        num_qubits: Number of qubits (>= 2).
        use_barriers: Whether to add barriers for visualization.
    """

    def __init__(self, num_qubits: int, use_barriers: bool = True):
        if num_qubits < 2:
            raise ValueError(f"GHZ requires at least 2 qubits, got {num_qubits}")

        self._num_qubits = num_qubits
        self._use_barriers = use_barriers
        self._circuit = self._build_circuit()

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def expected_outcomes(self) -> List[str]:
        """Expected outcomes: |0...0> and |1...1> each with ~50% probability."""
        return ["0" * self._num_qubits, "1" * self._num_qubits]

    def _build_circuit(self) -> QuantumCircuit:
        n = self._num_qubits
        qc = QuantumCircuit(n, n)

        # Hadamard on first qubit
        qc.h(0)

        if self._use_barriers:
            qc.barrier()

        # CNOT chain: entangle all qubits
        for i in range(n - 1):
            qc.cx(i, i + 1)

        if self._use_barriers:
            qc.barrier()

        # Measure all
        qc.measure(range(n), range(n))

        return qc


class GHZCircuitGenerator:
    """Generator for GHZ circuits at various scales."""

    @staticmethod
    def generate(num_qubits: int, use_barriers: bool = True) -> QuantumCircuit:
        return GHZ(num_qubits, use_barriers).circuit

    @staticmethod
    def get_scaling_configs(max_qubits: int = 20) -> List[dict]:
        """Generate scaling configurations for GHZ experiments."""
        configs = []
        for n in range(2, max_qubits + 1):
            configs.append(
                {
                    "config_id": f"GHZ{n}",
                    "num_qubits": n,
                    "description": f"{n}-qubit GHZ state",
                }
            )
        return configs
