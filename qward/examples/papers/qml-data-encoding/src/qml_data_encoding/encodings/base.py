"""Abstract base class for quantum data encodings."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


class BaseEncoding(ABC):
    """Abstract base for all quantum data encoding methods.

    Subclasses must implement ``encode`` to build a QuantumCircuit
    from a classical data vector.  Common operations (kernel, kernel
    matrix, Meyer-Wallach) are implemented here using statevector
    simulation.
    """

    def __init__(self, n_features: int, n_qubits: int) -> None:
        self.n_features = n_features
        self.n_qubits = n_qubits
        self._sv_backend = AerSimulator(method="statevector")

    @abstractmethod
    def encode(
        self,
        x: np.ndarray,
        theta: Optional[np.ndarray] = None,
    ) -> QuantumCircuit:
        """Build encoding circuit for data vector *x*."""

    # ------------------------------------------------------------------
    # Statevector helper
    # ------------------------------------------------------------------

    def _statevector(
        self,
        x: np.ndarray,
        theta: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return the statevector produced by encoding *x*."""
        circuit = self.encode(x, theta)
        circuit.save_statevector()
        result = self._sv_backend.run(circuit).result()
        return result.get_statevector().data

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------

    def kernel(
        self,
        x: np.ndarray,
        x_prime: np.ndarray,
        theta: Optional[np.ndarray] = None,
    ) -> float:
        """Compute K(x, x') = |<phi(x)|phi(x')>|^2."""
        sv1 = self._statevector(x, theta)
        sv2 = self._statevector(x_prime, theta)
        return float(np.abs(np.dot(sv1.conj(), sv2)) ** 2)

    def kernel_matrix(
        self,
        X: np.ndarray,
        theta: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the kernel Gram matrix for dataset *X*."""
        n = len(X)
        svs = [self._statevector(X[i], theta) for i in range(n)]
        K = np.zeros((n, n))
        for i in range(n):
            K[i, i] = 1.0
            for j in range(i + 1, n):
                val = float(np.abs(np.dot(svs[i].conj(), svs[j])) ** 2)
                K[i, j] = val
                K[j, i] = val
        return K

    # ------------------------------------------------------------------
    # Meyer-Wallach entanglement measure
    # ------------------------------------------------------------------

    def meyer_wallach(
        self,
        x: np.ndarray,
        theta: Optional[np.ndarray] = None,
    ) -> float:
        """Compute the Meyer-Wallach entanglement measure.

        Q(|psi>) = (2/n) sum_k (1 - tr(rho_k^2))
        where rho_k is the reduced density matrix of qubit k.
        """
        from qml_data_encoding.metrics.entanglement import (
            meyer_wallach_from_statevector,
        )

        sv = self._statevector(x, theta)
        return meyer_wallach_from_statevector(sv, self.n_qubits)
