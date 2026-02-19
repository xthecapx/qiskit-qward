"""
Quantum Eigensolver source modules.

This package contains the implementation of a VQE-based quantum eigensolver
for small Hermitian matrices (2x2, 3x3, 4x4) with classical validation.
"""

from .pauli_decomposition import pauli_decompose, PauliDecomposition
from .classical_baseline import ClassicalEigensolver
from .quantum_eigensolver import (
    EigensolverResult,
    EigensolverBase,
    QuantumEigensolver,
)
from .ansatz import build_ansatz

__all__ = [
    "pauli_decompose",
    "PauliDecomposition",
    "ClassicalEigensolver",
    "EigensolverResult",
    "EigensolverBase",
    "QuantumEigensolver",
    "build_ansatz",
]
