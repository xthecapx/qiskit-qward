"""Amplitude encoding: maps a normalized vector to quantum amplitudes.

|phi(x)> = sum_i x_tilde_i |i>  where x_tilde = x / ||x||_2
"""

from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

from qml_data_encoding.encodings.base import BaseEncoding

# Number of decomposition rounds needed to reduce StatePreparation
# to base gates (u, cx) that Aer can simulate.
_DECOMPOSE_DEPTH = 4


class AmplitudeEncoding(BaseEncoding):
    """Amplitude encoding using Qiskit ``StatePreparation``.

    Args:
        n_features: Number of features in the data vector.
    """

    def __init__(self, n_features: int) -> None:
        n_qubits = int(np.ceil(np.log2(max(n_features, 2))))
        super().__init__(n_features=n_features, n_qubits=n_qubits)

    def encode(
        self,
        x: np.ndarray,
        theta: Optional[np.ndarray] = None,
    ) -> QuantumCircuit:
        """Build amplitude encoding circuit.

        The input vector is L2-normalized automatically.  Zero-padding
        is applied when *n_features* is not a power of two.

        Args:
            x: Real-valued feature vector of shape (n_features,).
            theta: Unused.

        Returns:
            QuantumCircuit encoding the normalized amplitudes.

        Raises:
            ValueError: If *x* is the zero vector.
        """
        x = np.asarray(x, dtype=float)
        norm = np.linalg.norm(x)
        if norm == 0.0:
            raise ValueError("Cannot amplitude-encode the zero vector.")
        x_norm = x / norm

        # Pad to next power of 2
        dim = 2**self.n_qubits
        if len(x_norm) < dim:
            x_norm = np.concatenate([x_norm, np.zeros(dim - len(x_norm))])

        qc = QuantumCircuit(self.n_qubits)
        qc.append(
            StatePreparation(x_norm),
            list(range(self.n_qubits)),
        )

        # Decompose to base gates so Aer can simulate directly.
        for _ in range(_DECOMPOSE_DEPTH):
            qc = qc.decompose()

        return qc
