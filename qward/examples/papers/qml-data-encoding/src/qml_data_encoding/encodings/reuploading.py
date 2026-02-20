"""Data re-uploading encoding.

U(x, theta) = prod_{l=1}^{L} [W(theta_l) . S(x)]
where S(x) = tensor_i Ry(x_i)  and
      W(theta_l) = CX_chain . tensor_i Ry(theta_{l,i})
"""

from typing import Optional

import numpy as np
from qiskit import QuantumCircuit

from qml_data_encoding.encodings.base import BaseEncoding


class ReuploadingEncoding(BaseEncoding):
    """Data re-uploading encoding with trainable parameters.

    Args:
        n_features: Number of features (= number of qubits).
        n_layers: Number of re-uploading layers *L*.
    """

    def __init__(self, n_features: int, n_layers: int = 1) -> None:
        super().__init__(n_features=n_features, n_qubits=n_features)
        self.n_layers = n_layers

    @property
    def n_trainable_params(self) -> int:
        """Total number of trainable rotation parameters."""
        return self.n_layers * self.n_features

    def encode(
        self,
        x: np.ndarray,
        theta: Optional[np.ndarray] = None,
    ) -> QuantumCircuit:
        """Build re-uploading encoding circuit.

        Args:
            x: Feature vector of shape (n_features,).
            theta: Trainable parameter vector of shape
                (n_layers * n_features,).  Defaults to all zeros.

        Returns:
            QuantumCircuit with L layers of [S(x) . W(theta)].
        """
        x = np.asarray(x, dtype=float)
        if theta is None:
            theta = np.zeros(self.n_trainable_params)
        theta = np.asarray(theta, dtype=float)

        qc = QuantumCircuit(self.n_qubits)
        d = self.n_features

        for layer in range(self.n_layers):
            # Entangling layer: linear chain of CX gates
            for i in range(d - 1):
                qc.cx(i, i + 1)

            # S(x): data encoding via Ry
            for i in range(d):
                qc.ry(float(x[i]), i)

            # W(theta_l): trainable Ry rotations
            theta_l = theta[layer * d : (layer + 1) * d]
            for i in range(d):
                qc.ry(float(theta_l[i]), i)

        return qc

    def expectation_value(
        self,
        x: np.ndarray,
        theta: np.ndarray,
    ) -> float:
        """Compute <Z_0> expectation value for the encoded state.

        Used for Fourier spectrum analysis.

        Args:
            x: Feature vector.
            theta: Trainable parameters.

        Returns:
            Expectation value of Z on qubit 0.
        """
        sv = self._statevector(x, theta)
        n = len(sv)
        n_qubits = self.n_qubits
        # Z on qubit 0: +1 for |0...>, -1 for |1...>
        # In Qiskit qubit ordering, qubit 0 is the least-significant bit.
        expectation = 0.0
        for i in range(n):
            # Bit 0 of index i
            bit0 = (i >> 0) & 1
            sign = 1.0 - 2.0 * bit0
            expectation += sign * np.abs(sv[i]) ** 2
        return float(expectation)
