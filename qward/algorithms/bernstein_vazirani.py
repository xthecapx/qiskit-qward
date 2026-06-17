"""
Bernstein-Vazirani algorithm implementation.

Determines a hidden bitstring s from an oracle f(x) = s·x mod 2
using a single quantum query.
"""

from typing import List, Optional

from qiskit import QuantumCircuit


class BernsteinVazirani:
    """Bernstein-Vazirani algorithm for finding a hidden bitstring.

    The circuit has O(1) depth (independent of n) after the oracle,
    making it highly resilient to hardware noise at large qubit counts.

    Args:
        secret_string: The hidden bitstring to find (e.g., "10110").
        use_barriers: Whether to add barriers for visualization.
    """

    def __init__(self, secret_string: str, use_barriers: bool = True):
        if not all(c in "01" for c in secret_string):
            raise ValueError(f"Secret string must be binary, got: {secret_string}")
        if len(secret_string) < 1:
            raise ValueError("Secret string must have at least 1 bit")

        self._secret_string = secret_string
        self._num_qubits = len(secret_string)
        self._use_barriers = use_barriers
        self._circuit = self._build_circuit()

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def secret_string(self) -> str:
        return self._secret_string

    @property
    def expected_outcomes(self) -> List[str]:
        """Expected measurement outcome (Qiskit little-endian: reversed)."""
        return [self._secret_string[::-1]]

    def _build_circuit(self) -> QuantumCircuit:
        n = self._num_qubits
        # n query qubits + 1 ancilla
        qc = QuantumCircuit(n + 1, n)

        # Initialize ancilla to |->
        qc.x(n)
        qc.h(range(n + 1))

        if self._use_barriers:
            qc.barrier()

        # Oracle: CNOT from qubit i to ancilla for each '1' in secret
        for i, bit in enumerate(self._secret_string):
            if bit == "1":
                qc.cx(i, n)

        if self._use_barriers:
            qc.barrier()

        # Apply H to query qubits and measure
        qc.h(range(n))
        qc.measure(range(n), range(n))

        return qc


class BernsteinVaziraniCircuitGenerator:
    """Generator for BV circuits at various scales."""

    @staticmethod
    def generate(secret_string: str, use_barriers: bool = True) -> QuantumCircuit:
        return BernsteinVazirani(secret_string, use_barriers).circuit

    @staticmethod
    def get_scaling_configs(max_qubits: int = 14) -> List[dict]:
        """Generate scaling configurations for BV experiments."""
        configs = []
        for n in range(2, max_qubits + 1):
            # All-ones secret
            configs.append(
                {
                    "config_id": f"BV{n}-ONES",
                    "num_qubits": n,
                    "secret_string": "1" * n,
                    "description": f"{n} qubits, all-ones secret",
                }
            )
            # Alternating secret
            alt = "".join("1" if i % 2 == 0 else "0" for i in range(n))
            configs.append(
                {
                    "config_id": f"BV{n}-ALT",
                    "num_qubits": n,
                    "secret_string": alt,
                    "description": f"{n} qubits, alternating secret",
                }
            )
            # Single-bit secret (last bit)
            single = "0" * (n - 1) + "1"
            configs.append(
                {
                    "config_id": f"BV{n}-SINGLE",
                    "num_qubits": n,
                    "secret_string": single,
                    "description": f"{n} qubits, single-bit secret",
                }
            )
        return configs
