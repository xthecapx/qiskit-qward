"""
Classical eigensolver baseline using NumPy.

Provides a classical reference implementation for validating VQE results.
Uses numpy.linalg.eigh for exact eigendecomposition.

This serves as the ground truth for comparing quantum eigensolver accuracy.
"""

from typing import List, Optional

import numpy as np

from .quantum_eigensolver import EigensolverBase, EigensolverResult


class ClassicalEigensolver(EigensolverBase):
    """Classical eigensolver using NumPy's exact eigendecomposition.

    This solver wraps numpy.linalg.eigh to provide a reference
    implementation with the same interface as QuantumEigensolver.
    It reports 0 iterations and no optimal_parameters since no
    variational optimization is involved.

    Args:
        matrix: Hermitian matrix (numpy ndarray).

    Raises:
        TypeError: If matrix is not a numpy ndarray.
        ValueError: If matrix is not Hermitian.
    """

    def __init__(self, matrix: np.ndarray):
        if not isinstance(matrix, np.ndarray):
            raise TypeError(f"Expected numpy ndarray, got {type(matrix).__name__}")
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Matrix must be square, got shape {matrix.shape}")
        if not np.allclose(matrix, matrix.conj().T, atol=1e-10):
            raise ValueError("Matrix is not Hermitian (M != M^dagger)")
        self.matrix = matrix

    def solve(self, **kwargs) -> EigensolverResult:
        """Find the minimum eigenvalue using NumPy.

        Returns:
            EigensolverResult with the minimum eigenvalue and
            corresponding eigenvector. iterations=0, no
            optimal_parameters.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.matrix)
        min_idx = np.argmin(eigenvalues)
        return EigensolverResult(
            eigenvalue=float(eigenvalues[min_idx]),
            eigenvector=eigenvectors[:, min_idx],
            optimal_parameters=None,
            iterations=0,
            cost_history=None,
            converged=True,
        )

    def solve_all(self) -> List[float]:
        """Find all eigenvalues using NumPy.

        Returns:
            Sorted list of all eigenvalues.
        """
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        return sorted(eigenvalues.tolist())
