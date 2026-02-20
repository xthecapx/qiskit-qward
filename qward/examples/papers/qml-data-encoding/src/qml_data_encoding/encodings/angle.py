"""Angle (rotation) encoding: maps features to rotation angles.

For Ry:  |phi(x)> = tensor_i Ry(x_i)|0>
For Rx:  |phi(x)> = tensor_i Rx(x_i)|0>
For Rz:  |phi(x)> = tensor_i Rz(x_i)|0>
"""

from typing import Optional

import numpy as np
from qiskit import QuantumCircuit

from qml_data_encoding.encodings.base import BaseEncoding


class AngleEncoding(BaseEncoding):
    """Angle encoding using single-qubit rotations.

    Args:
        n_features: Number of features (= number of qubits).
        rotation_axis: One of "x", "y", "z".
    """

    def __init__(
        self,
        n_features: int,
        rotation_axis: str = "y",
    ) -> None:
        if rotation_axis not in ("x", "y", "z"):
            raise ValueError(f"rotation_axis must be 'x', 'y', or 'z', " f"got '{rotation_axis}'")
        super().__init__(n_features=n_features, n_qubits=n_features)
        self.rotation_axis = rotation_axis

    def encode(
        self,
        x: np.ndarray,
        theta: Optional[np.ndarray] = None,
    ) -> QuantumCircuit:
        """Build angle encoding circuit.

        Args:
            x: Feature vector of shape (n_features,).
            theta: Unused.

        Returns:
            QuantumCircuit with one rotation gate per qubit.
        """
        x = np.asarray(x, dtype=float)
        qc = QuantumCircuit(self.n_qubits)
        gate_fn = {"x": qc.rx, "y": qc.ry, "z": qc.rz}[self.rotation_axis]
        # Map feature i to qubit (n-1-i) so the statevector ordering
        # matches the standard tensor product convention:
        # kron(state_0, state_1, ..., state_{n-1}).
        for i in range(self.n_qubits):
            gate_fn(float(x[i]), self.n_qubits - 1 - i)
        return qc
