"""
Integration tests for the QWARD Eigensolver module.

Tests the public API of qward.algorithms.eigensolver to ensure
the library integration is correct. These tests verify the same
behavior as the sandbox tests but import from the library path.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Test matrix fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pauli_z():
    return np.array([[1, 0], [0, -1]], dtype=complex)


@pytest.fixture
def pauli_x():
    return np.array([[0, 1], [1, 0]], dtype=complex)


@pytest.fixture
def general_hermitian_2x2():
    return np.array([[2, 1 - 1j], [1 + 1j, 3]], dtype=complex)


@pytest.fixture
def symmetric_3x3():
    return np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=complex)


@pytest.fixture
def heisenberg_xxx():
    return np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 2, 0],
            [0, 2, -1, 0],
            [0, 0, 0, 1],
        ],
        dtype=complex,
    )


# ---------------------------------------------------------------------------
# Test imports from qward.algorithms
# ---------------------------------------------------------------------------


class TestLibraryImports:
    """Verify the public API is accessible from qward.algorithms."""

    def test_import_quantum_eigensolver(self):
        from qward.algorithms import QuantumEigensolver

        assert QuantumEigensolver is not None

    def test_import_classical_eigensolver(self):
        from qward.algorithms import ClassicalEigensolver

        assert ClassicalEigensolver is not None

    def test_import_eigensolver_result(self):
        from qward.algorithms import EigensolverResult

        assert EigensolverResult is not None

    def test_import_pauli_decompose(self):
        from qward.algorithms import pauli_decompose

        assert pauli_decompose is not None

    def test_import_from_submodule(self):
        from qward.algorithms.eigensolver import (
            QuantumEigensolver,
            ClassicalEigensolver,
            pauli_decompose,
            build_ansatz,
        )

        assert all([QuantumEigensolver, ClassicalEigensolver, pauli_decompose, build_ansatz])


# ---------------------------------------------------------------------------
# Test Pauli decomposition
# ---------------------------------------------------------------------------


class TestPauliDecompose:
    """Test pauli_decompose from library path."""

    def test_pauli_z_decomposition(self, pauli_z):
        from qward.algorithms import pauli_decompose

        decomp = pauli_decompose(pauli_z)
        assert abs(decomp.get("Z", 0.0) - 1.0) < 1e-10

    def test_general_hermitian_decomposition(self, general_hermitian_2x2):
        from qward.algorithms import pauli_decompose

        decomp = pauli_decompose(general_hermitian_2x2)
        assert abs(decomp.get("I", 0.0) - 2.5) < 1e-10
        assert abs(decomp.get("X", 0.0) - 1.0) < 1e-10
        assert abs(decomp.get("Y", 0.0) - 1.0) < 1e-10
        assert abs(decomp.get("Z", 0.0) - (-0.5)) < 1e-10

    def test_heisenberg_decomposition(self, heisenberg_xxx):
        from qward.algorithms import pauli_decompose

        decomp = pauli_decompose(heisenberg_xxx)
        assert abs(decomp.get("XX", 0.0) - 1.0) < 1e-10
        assert abs(decomp.get("YY", 0.0) - 1.0) < 1e-10
        assert abs(decomp.get("ZZ", 0.0) - 1.0) < 1e-10

    def test_non_power_of_two_embedding(self, symmetric_3x3):
        from qward.algorithms import pauli_decompose

        decomp = pauli_decompose(symmetric_3x3)
        assert decomp.was_padded is True
        assert decomp.original_dimension == 3
        assert decomp.num_qubits == 2

    def test_non_hermitian_raises(self):
        from qward.algorithms import pauli_decompose

        bad = np.array([[1, 2], [3, 4]], dtype=complex)
        with pytest.raises(ValueError):
            pauli_decompose(bad)


# ---------------------------------------------------------------------------
# Test Classical Eigensolver
# ---------------------------------------------------------------------------


class TestClassicalEigensolver:
    """Test ClassicalEigensolver from library path."""

    def test_solve_minimum(self, pauli_z):
        from qward.algorithms import ClassicalEigensolver

        solver = ClassicalEigensolver(pauli_z)
        result = solver.solve()
        assert abs(result.eigenvalue - (-1.0)) < 1e-10

    def test_solve_all(self, heisenberg_xxx):
        from qward.algorithms import ClassicalEigensolver

        solver = ClassicalEigensolver(heisenberg_xxx)
        eigenvalues = solver.solve_all()
        np.testing.assert_allclose(eigenvalues, [-3.0, 1.0, 1.0, 1.0], atol=1e-10)

    def test_result_attributes(self, pauli_z):
        from qward.algorithms import ClassicalEigensolver

        result = ClassicalEigensolver(pauli_z).solve()
        assert result.iterations == 0
        assert result.optimal_parameters is None
        assert result.converged is True
        assert result.eigenvector is not None


# ---------------------------------------------------------------------------
# Test Quantum Eigensolver (VQE)
# ---------------------------------------------------------------------------


class TestQuantumEigensolver:
    """Test QuantumEigensolver (VQE) from library path."""

    def test_vqe_pauli_z(self, pauli_z):
        from qward.algorithms import QuantumEigensolver

        solver = QuantumEigensolver(pauli_z)
        result = solver.solve()
        spectral_range = 2.0
        assert abs(result.eigenvalue - (-1.0)) < 0.01 * spectral_range

    def test_vqe_pauli_x(self, pauli_x):
        from qward.algorithms import QuantumEigensolver

        solver = QuantumEigensolver(pauli_x)
        result = solver.solve()
        spectral_range = 2.0
        assert abs(result.eigenvalue - (-1.0)) < 0.01 * spectral_range

    def test_vqe_general_hermitian(self, general_hermitian_2x2):
        from qward.algorithms import QuantumEigensolver

        solver = QuantumEigensolver(general_hermitian_2x2)
        result = solver.solve()
        spectral_range = 3.0
        assert abs(result.eigenvalue - 1.0) < 0.01 * spectral_range

    def test_vqe_heisenberg(self, heisenberg_xxx):
        from qward.algorithms import QuantumEigensolver

        solver = QuantumEigensolver(heisenberg_xxx)
        result = solver.solve()
        spectral_range = 4.0
        assert abs(result.eigenvalue - (-3.0)) < 0.01 * spectral_range

    def test_vqe_result_attributes(self, pauli_z):
        from qward.algorithms import QuantumEigensolver

        result = QuantumEigensolver(pauli_z).solve()
        assert result.optimal_parameters is not None
        assert result.iterations > 0
        assert result.cost_history is not None
        assert len(result.cost_history) > 0

    def test_vqe_solve_all_2x2(self, pauli_z):
        from qward.algorithms import QuantumEigensolver

        solver = QuantumEigensolver(pauli_z)
        eigenvalues = solver.solve_all()
        np.testing.assert_allclose(eigenvalues, [-1.0, 1.0], atol=0.05)

    def test_vqe_non_hermitian_raises(self):
        from qward.algorithms import QuantumEigensolver

        bad = np.array([[1, 2], [3, 4]], dtype=complex)
        with pytest.raises((ValueError, TypeError)):
            QuantumEigensolver(bad)

    def test_vqe_noisy_preset(self, pauli_z):
        from qward.algorithms import QuantumEigensolver

        solver = QuantumEigensolver(
            pauli_z,
            noise_preset="IBM-HERON-R2",
            shots=4096,
        )
        result = solver.solve()
        spectral_range = 2.0
        assert abs(result.eigenvalue - (-1.0)) < 0.05 * spectral_range
