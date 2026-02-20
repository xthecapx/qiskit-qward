"""Basis encoding: maps binary vectors to computational basis states.

|phi(x)> = |b_1 b_2 ... b_d>  using  U(x) = tensor_i X^{b_i}
"""

from typing import Optional

import numpy as np
from qiskit import QuantumCircuit

from qml_data_encoding.encodings.base import BaseEncoding


class BasisEncoding(BaseEncoding):
    """Basis encoding for binary feature vectors.

    Args:
        n_features: Number of binary features (= number of qubits).
    """

    def __init__(self, n_features: int) -> None:
        super().__init__(n_features=n_features, n_qubits=n_features)

    def encode(
        self,
        x: np.ndarray,
        theta: Optional[np.ndarray] = None,
    ) -> QuantumCircuit:
        """Build basis encoding circuit for binary vector *x*.

        Args:
            x: Binary vector of shape (n_features,) with values in {0, 1}.
            theta: Unused (basis encoding has no trainable parameters).

        Returns:
            QuantumCircuit encoding the binary vector.

        Raises:
            ValueError: If *x* contains non-binary values.
        """
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n_features,):
            raise ValueError(f"Expected shape ({self.n_features},), got {x.shape}")
        if not np.all((x == 0) | (x == 1)):
            raise ValueError("Basis encoding requires binary input (0 or 1 per feature).")

        qc = QuantumCircuit(self.n_qubits)
        for i, b in enumerate(x):
            if b == 1.0:
                qc.x(i)
        return qc
