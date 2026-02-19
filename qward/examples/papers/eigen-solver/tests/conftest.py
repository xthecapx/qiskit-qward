"""
Shared fixtures for Quantum Eigensolver tests.

Specification:
  - phase1_problem_statement.md (corrected eigenvalues)
  - phase2_theoretical_design.md (Pauli decomposition, ansatz, embedding)
Author: test-engineer

This module provides reusable pytest fixtures for all eigensolver test files,
including test matrices with known eigenvalues, simulators, and helper utilities.

Eigenvalue corrections (from Phase 1):
  - M3 [[2, 1-1j], [1+1j, 3]]: eigenvalues are {1, 4} (char poly: l^2 - 5l + 4 = 0)
  - M4 [[2,1,0],[1,3,1],[0,1,2]]: eigenvalues are {1, 2, 4}
"""

import os
import sys

# Set matplotlib backend before any imports that might use it
os.environ["MPLBACKEND"] = "Agg"

# Make ``from eigen_solver.src.<module> import ...`` work.
# The filesystem uses ``eigen-solver/`` (hyphen) but Python imports need
# underscores.  We add the eigen-solver directory to sys.path so that the
# ``eigen_solver/`` sub-package (which contains ``src/``) is importable.
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Test matrix definitions (eigenvalues verified in Phase 1)
# ---------------------------------------------------------------------------


def _pauli_z():
    """M1: Pauli Z matrix (2x2, diagonal, trivial case).
    Eigenvalues: -1, 1
    Spectral range: 2.0
    Pauli decomposition: H = Z
    """
    return np.array([[1, 0], [0, -1]], dtype=complex)


def _pauli_x():
    """M2: Pauli X matrix (2x2, off-diagonal).
    Eigenvalues: -1, 1
    Spectral range: 2.0
    Pauli decomposition: H = X
    """
    return np.array([[0, 1], [1, 0]], dtype=complex)


def _general_hermitian_2x2():
    """M3: General 2x2 Hermitian matrix with complex off-diagonal entries.
    M = [[2, 1-1j], [1+1j, 3]]
    Eigenvalues: 1, 4  (char poly: lambda^2 - 5*lambda + 4 = 0)
    Spectral range: 3.0
    Pauli decomposition: H = 2.5*I + 1.0*X + 1.0*Y - 0.5*Z
    """
    return np.array([[2, 1 - 1j], [1 + 1j, 3]], dtype=complex)


def _symmetric_3x3():
    """M4: Symmetric real 3x3 matrix.
    M = [[2, 1, 0], [1, 3, 1], [0, 1, 2]]
    Eigenvalues: 1, 2, 4  (char poly: (2-lambda)(lambda-1)(lambda-4) = 0)
    Spectral range: 3.0
    Requires embedding into 4x4 with penalty for VQE.
    """
    return np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=complex)


def _symmetric_3x3_embedded(penalty=10):
    """M4 embedded: 3x3 symmetric matrix embedded in 4x4 with penalty.
    Eigenvalues of embedded matrix: {1, 2, 4, penalty}
    Pauli decomposition has 8 terms.
    """
    M = np.zeros((4, 4), dtype=complex)
    M[:3, :3] = _symmetric_3x3()
    M[3, 3] = penalty
    return M


def _heisenberg_xxx_4x4():
    """M5: 2-qubit Heisenberg XXX model (J=1), physically relevant 4x4 matrix.

    H = XX + YY + ZZ

    In the computational basis {|00>, |01>, |10>, |11>}:
    H = [[ 1,  0,  0,  0],
         [ 0, -1,  2,  0],
         [ 0,  2, -1,  0],
         [ 0,  0,  0,  1]]

    Eigenvalues: -3, 1, 1, 1  (singlet + triplet)
    Spectral range: 4.0
    Ground state: |psi-> = (|01> - |10>)/sqrt(2) with E = -3
    Pauli decomposition: H = XX + YY + ZZ (3 terms)
    """
    return np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 2, 0],
            [0, 2, -1, 0],
            [0, 0, 0, 1],
        ],
        dtype=complex,
    )


def _identity_2x2():
    """2x2 identity matrix (degenerate eigenvalues).
    Eigenvalues: 1, 1
    """
    return np.eye(2, dtype=complex)


# ---------------------------------------------------------------------------
# Fixtures: Test matrices
# ---------------------------------------------------------------------------


@pytest.fixture
def pauli_z_matrix():
    """M1: Pauli Z matrix with known eigenvalues [-1, 1]."""
    return _pauli_z()


@pytest.fixture
def pauli_x_matrix():
    """M2: Pauli X matrix with known eigenvalues [-1, 1]."""
    return _pauli_x()


@pytest.fixture
def general_hermitian_2x2():
    """M3: General 2x2 Hermitian matrix with eigenvalues [1, 4]."""
    return _general_hermitian_2x2()


@pytest.fixture
def symmetric_3x3():
    """M4: Symmetric real 3x3 matrix with eigenvalues [1, 2, 4]."""
    return _symmetric_3x3()


@pytest.fixture
def symmetric_3x3_embedded():
    """M4 embedded: 3x3 matrix embedded in 4x4 with penalty=10.
    Eigenvalues: [1, 2, 4, 10].
    """
    return _symmetric_3x3_embedded(penalty=10)


@pytest.fixture
def heisenberg_xxx_4x4():
    """M5: 2-qubit Heisenberg XXX model with eigenvalues [-3, 1, 1, 1]."""
    return _heisenberg_xxx_4x4()


@pytest.fixture
def identity_2x2():
    """2x2 identity matrix (degenerate eigenvalue test case)."""
    return _identity_2x2()


@pytest.fixture
def all_test_matrices():
    """Dictionary of all test matrices with their expected sorted eigenvalues.

    Eigenvalues verified against Phase 1 problem statement.

    Returns:
        dict: {name: (matrix, sorted_eigenvalues)}
    """
    return {
        "pauli_z": (_pauli_z(), [-1.0, 1.0]),
        "pauli_x": (_pauli_x(), [-1.0, 1.0]),
        "general_hermitian_2x2": (
            _general_hermitian_2x2(),
            [1.0, 4.0],
        ),
        "symmetric_3x3": (
            _symmetric_3x3(),
            [1.0, 2.0, 4.0],
        ),
        "heisenberg_xxx_4x4": (
            _heisenberg_xxx_4x4(),
            [-3.0, 1.0, 1.0, 1.0],
        ),
    }


@pytest.fixture
def two_by_two_matrices():
    """Subset of test matrices that require exactly 1 qubit (2x2)."""
    return {
        "pauli_z": (_pauli_z(), [-1.0, 1.0]),
        "pauli_x": (_pauli_x(), [-1.0, 1.0]),
        "general_hermitian_2x2": (
            _general_hermitian_2x2(),
            [1.0, 4.0],
        ),
    }


@pytest.fixture
def four_by_four_matrices():
    """Subset of test matrices that require exactly 2 qubits (4x4)."""
    return {
        "heisenberg_xxx_4x4": (
            _heisenberg_xxx_4x4(),
            [-3.0, 1.0, 1.0, 1.0],
        ),
    }


# Known Pauli decompositions from Phase 2 theoretical design
KNOWN_DECOMPOSITIONS = {
    "pauli_z": {"Z": 1.0},
    "pauli_x": {"X": 1.0},
    "general_hermitian_2x2": {"II": 2.5, "X": 1.0, "Y": 1.0, "Z": -0.5},
    "heisenberg_xxx_4x4": {"XX": 1.0, "YY": 1.0, "ZZ": 1.0},
}


# ---------------------------------------------------------------------------
# Fixtures: Tolerances from Phase 1 success criteria
# ---------------------------------------------------------------------------


@pytest.fixture
def ideal_tolerance():
    """Tolerance for ideal (noiseless) VQE: 1% of spectral range.

    From Phase 1: Energy error <= 0.01 (1% of spectral range) for ideal simulation.
    """
    return 0.01


@pytest.fixture
def noisy_tolerance():
    """Tolerance for noisy VQE: 5% of spectral range.

    From Phase 1: Energy error <= 0.05 (5% of spectral range) with noise.
    """
    return 0.05


@pytest.fixture
def relaxed_noisy_tolerance():
    """Relaxed tolerance for very noisy conditions: 10% of spectral range."""
    return 0.10


# ---------------------------------------------------------------------------
# Fixtures: Random Hermitian matrix generators
# ---------------------------------------------------------------------------


@pytest.fixture
def random_hermitian_2x2():
    """Generate a random 2x2 Hermitian matrix (seeded for reproducibility)."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    return (A + A.conj().T) / 2


@pytest.fixture
def random_hermitian_4x4():
    """Generate a random 4x4 Hermitian matrix (seeded for reproducibility)."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    return (A + A.conj().T) / 2


# ---------------------------------------------------------------------------
# Pytest markers registration
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")
    config.addinivalue_line("markers", "noisy: marks tests that use noise models")
