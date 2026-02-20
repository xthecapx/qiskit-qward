"""IQP (Instantaneous Quantum Polynomial) encoding.

U_IQP(x) = H^{otimes d} . D(x) . H^{otimes d}
where D(x) = exp(i sum_i x_i Z_i + i sum_{i<j} x_i x_j Z_i Z_j)

Decomposed as:
  1. H on all qubits
  2. Rz(2*x_i) on qubit i for each i
  3. RZZ(2*x_i*x_j) on qubits (i,j) for each pair i<j
  4. H on all qubits
"""

from typing import Optional

import numpy as np
from qiskit import QuantumCircuit

from qml_data_encoding.encodings.base import BaseEncoding


class IQPEncoding(BaseEncoding):
    """IQP encoding with diagonal ZZ interactions.

    Args:
        n_features: Number of features (= number of qubits).
        interaction: ``"full"`` for all-to-all or
            ``"nearest_neighbor"`` for linear chain.
    """

    def __init__(
        self,
        n_features: int,
        interaction: str = "full",
    ) -> None:
        if interaction not in ("full", "nearest_neighbor"):
            raise ValueError(
                f"interaction must be 'full' or 'nearest_neighbor', " f"got '{interaction}'"
            )
        super().__init__(n_features=n_features, n_qubits=n_features)
        self.interaction = interaction

    def _get_pairs(self) -> list:
        """Return qubit pairs for the ZZ interactions."""
        if self.interaction == "full":
            return [(i, j) for i in range(self.n_qubits) for j in range(i + 1, self.n_qubits)]
        else:  # nearest_neighbor
            return [(i, i + 1) for i in range(self.n_qubits - 1)]

    @staticmethod
    def _rzz(qc: QuantumCircuit, angle: float, i: int, j: int) -> None:
        """Decompose RZZ(angle) into CX + Rz + CX."""
        qc.cx(i, j)
        qc.rz(angle, j)
        qc.cx(i, j)

    def encode(
        self,
        x: np.ndarray,
        theta: Optional[np.ndarray] = None,
    ) -> QuantumCircuit:
        """Build IQP encoding circuit.

        Args:
            x: Feature vector of shape (n_features,).
            theta: Unused.

        Returns:
            QuantumCircuit implementing the IQP encoding.
        """
        x = np.asarray(x, dtype=float)
        qc = QuantumCircuit(self.n_qubits)

        # First Hadamard layer
        for i in range(self.n_qubits):
            qc.h(i)

        # Diagonal: single-qubit Rz(2*x_i)
        for i in range(self.n_qubits):
            qc.rz(2.0 * float(x[i]), i)

        # Diagonal: two-qubit RZZ(2*x_i*x_j)
        pairs = self._get_pairs()
        for i, j in pairs:
            self._rzz(qc, 2.0 * float(x[i]) * float(x[j]), i, j)

        # Second Hadamard layer
        for i in range(self.n_qubits):
            qc.h(i)

        return qc
