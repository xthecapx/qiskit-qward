"""
Pauli decomposition of Hermitian matrices.

Decomposes an arbitrary Hermitian matrix M into a weighted sum of Pauli
tensor products using the trace formula:

    c_P = Tr(M * P) / 2^q

where P ranges over all q-qubit Pauli strings {I, X, Y, Z}^{otimes q}.

Non-power-of-two matrices are embedded into the next power-of-two dimension
via penalty padding (see Phase 2 theoretical design, Section 3).

References:
    - Phase 2: phase2_theoretical_design.md, Section 2
    - Nielsen & Chuang, Chapter 4 (Pauli group)
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp


class PauliDecomposition:
    """Result of decomposing a Hermitian matrix into Pauli basis.

    Behaves like a dict mapping Pauli labels to real coefficients, while
    also exposing the underlying Qiskit SparsePauliOp for direct use in
    quantum primitives.

    Attributes:
        sparse_pauli_op: Qiskit SparsePauliOp representation.
        num_qubits: Number of qubits in the decomposition.
        original_dimension: Original matrix dimension before any padding.
        was_padded: Whether the matrix required padding to power of 2.
        padding_penalty: Penalty value used for padding (None if not padded).
    """

    def __init__(
        self,
        sparse_pauli_op: SparsePauliOp,
        num_qubits: int,
        original_dimension: int,
        was_padded: bool = False,
        padding_penalty: Optional[float] = None,
    ):
        self.sparse_pauli_op = sparse_pauli_op
        self.num_qubits = num_qubits
        self.original_dimension = original_dimension
        self.was_padded = was_padded
        self.padding_penalty = padding_penalty
        # Build internal dict for dict-like access
        self._dict: dict = {}
        labels = sparse_pauli_op.paulis.to_labels()
        coeffs = sparse_pauli_op.coeffs
        for label, coeff in zip(labels, coeffs):
            self._dict[label] = float(coeff.real)

    # ---- Dict-like interface ----

    def __getitem__(self, key: str) -> float:
        return self._dict[key]

    def __contains__(self, key: str) -> bool:
        return key in self._dict

    def __iter__(self):
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def items(self):
        """Return (label, coefficient) pairs."""
        return self._dict.items()

    def values(self):
        """Return coefficient values."""
        return self._dict.values()

    def keys(self):
        """Return Pauli string labels."""
        return self._dict.keys()

    def get(self, key: str, default=None):
        """Get coefficient for a Pauli label."""
        return self._dict.get(key, default)

    # ---- Quantum-specific interface ----

    @property
    def coefficients(self) -> np.ndarray:
        """Real-valued Pauli coefficients as numpy array."""
        return self.sparse_pauli_op.coeffs.real

    @property
    def labels(self) -> List[str]:
        """Pauli string labels (e.g., 'IZ', 'XX')."""
        return self.sparse_pauli_op.paulis.to_labels()

    @property
    def num_terms(self) -> int:
        """Number of non-zero Pauli terms."""
        return len(self.sparse_pauli_op)

    def to_matrix(self) -> np.ndarray:
        """Reconstruct the full matrix from the Pauli decomposition.

        Returns:
            The reconstructed Hermitian matrix as a numpy array.
        """
        return self.sparse_pauli_op.to_matrix()


def _validate_matrix(matrix: np.ndarray) -> None:
    """Validate that the input is a Hermitian matrix.

    Args:
        matrix: Input matrix to validate.

    Raises:
        TypeError: If input is not a numpy ndarray.
        ValueError: If matrix is not square or not Hermitian.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"Expected numpy ndarray, got {type(matrix).__name__}")
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D array, got {matrix.ndim}D array")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {matrix.shape}")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-10):
        raise ValueError("Matrix is not Hermitian (M != M^dagger)")


def _embed_matrix(
    matrix: np.ndarray,
    target_dim: int,
    penalty: Optional[float] = None,
) -> np.ndarray:
    """Embed an n x n matrix into a target_dim x target_dim matrix.

    Uses penalty padding as described in Phase 2, Section 3:
    the extra diagonal entries are set to a penalty value above
    the maximum eigenvalue.

    Args:
        matrix: Original n x n Hermitian matrix.
        target_dim: Target dimension (must be power of 2).
        penalty: Penalty value for padding. If None, computed as
            lambda_max + 2 * spectral_range (Eq. 5 from Phase 2).

    Returns:
        Padded target_dim x target_dim Hermitian matrix.
    """
    n = matrix.shape[0]
    if n == target_dim:
        return matrix.copy()

    if penalty is None:
        eigenvalues = np.linalg.eigvalsh(matrix)
        lam_min = eigenvalues[0]
        lam_max = eigenvalues[-1]
        spectral_range = lam_max - lam_min
        penalty = lam_max + 2 * spectral_range

    embedded = np.zeros((target_dim, target_dim), dtype=complex)
    embedded[:n, :n] = matrix
    for i in range(n, target_dim):
        embedded[i, i] = penalty

    return embedded


def pauli_decompose(
    matrix: np.ndarray,
    *,
    atol: float = 1e-12,
    penalty: Optional[float] = None,
) -> PauliDecomposition:
    """Decompose a Hermitian matrix into the Pauli string basis.

    Uses Qiskit's SparsePauliOp.from_operator for the decomposition,
    which implements the trace formula c_P = Tr(M * P) / 2^q.

    For non-power-of-two dimensions, the matrix is embedded into the
    next power of 2 using penalty padding.

    Args:
        matrix: Hermitian matrix (n x n numpy array).
        atol: Absolute tolerance for filtering near-zero coefficients.
        penalty: Penalty value for non-power-of-two embedding.
            If None, automatically computed.

    Returns:
        PauliDecomposition containing the SparsePauliOp and metadata.

    Raises:
        TypeError: If matrix is not a numpy ndarray.
        ValueError: If matrix is not square, not Hermitian, or too small.
    """
    _validate_matrix(matrix)

    n = matrix.shape[0]
    if n < 2:
        raise ValueError(f"Matrix dimension must be >= 2, got {n}")

    num_qubits = math.ceil(math.log2(n))
    target_dim = 2**num_qubits
    was_padded = n < target_dim

    if was_padded:
        work_matrix = _embed_matrix(matrix, target_dim, penalty=penalty)
        used_penalty = work_matrix[n, n].real
    else:
        work_matrix = matrix
        used_penalty = None

    operator = Operator(work_matrix)
    sparse_op = SparsePauliOp.from_operator(operator)

    # Filter near-zero terms for cleaner representation
    mask = np.abs(sparse_op.coeffs) > atol
    if mask.any():
        filtered_labels = [sparse_op.paulis[i].to_label() for i in range(len(sparse_op)) if mask[i]]
        filtered_coeffs = sparse_op.coeffs[mask]
        sparse_op = SparsePauliOp.from_list(list(zip(filtered_labels, filtered_coeffs)))

    return PauliDecomposition(
        sparse_pauli_op=sparse_op,
        num_qubits=num_qubits,
        original_dimension=n,
        was_padded=was_padded,
        padding_penalty=used_penalty,
    )
