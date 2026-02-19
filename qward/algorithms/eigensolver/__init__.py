"""
Quantum Eigensolver module for QWARD.

Provides VQE-based eigenvalue computation for small Hermitian matrices,
with classical validation and noise model support.

Example usage::

    import numpy as np
    from qward.algorithms.eigensolver import (
        QuantumEigensolver,
        ClassicalEigensolver,
        pauli_decompose,
    )

    matrix = np.array([[2, 1-1j], [1+1j, 3]], dtype=complex)

    # Quantum (VQE) eigensolver
    solver = QuantumEigensolver(matrix)
    result = solver.solve()
    print(result.eigenvalue)  # ~1.0

    # All eigenvalues via deflation
    eigenvalues = solver.solve_all()  # [1.0, 4.0]

    # Classical baseline
    classical = ClassicalEigensolver(matrix)
    classical_result = classical.solve()

    # Pauli decomposition
    decomposition = pauli_decompose(matrix)
    print(dict(decomposition.items()))
"""

from .pauli_decomposition import pauli_decompose, PauliDecomposition
from .quantum_eigensolver import (
    EigensolverResult,
    EigensolverBase,
    QuantumEigensolver,
)
from .classical_eigensolver import ClassicalEigensolver
from .ansatz import build_ansatz

__all__ = [
    "pauli_decompose",
    "PauliDecomposition",
    "EigensolverResult",
    "EigensolverBase",
    "QuantumEigensolver",
    "ClassicalEigensolver",
    "build_ansatz",
]
