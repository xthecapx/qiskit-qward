"""
Tests for classical eigenvalue baseline (NumPy validation).

Specification:
  - phase1_problem_statement.md (corrected eigenvalues, success criteria)
  - phase2_theoretical_design.md (embedding, Pauli decomposition verification)
Author: test-engineer

These tests validate that our test matrices have the expected eigenvalues
using NumPy's eigensolver. They MUST PASS before any quantum tests are run,
establishing the ground truth for VQE validation.

Eigenvalue corrections from Phase 1:
  - M3 [[2, 1-1j], [1+1j, 3]]: eigenvalues = {1, 4}
  - M4 [[2,1,0],[1,3,1],[0,1,2]]: eigenvalues = {1, 2, 4}
"""

import numpy as np
import pytest


class TestClassicalEigenvalues:
    """Validate classical eigenvalues for all test matrices using NumPy."""

    def test_pauli_z_eigenvalues(self, pauli_z_matrix):
        """M1: Pauli Z eigenvalues should be exactly -1 and 1."""
        eigenvalues = np.linalg.eigvalsh(pauli_z_matrix)
        expected = np.array([-1.0, 1.0])
        np.testing.assert_allclose(sorted(eigenvalues), expected, atol=1e-10)

    def test_pauli_x_eigenvalues(self, pauli_x_matrix):
        """M2: Pauli X eigenvalues should be exactly -1 and 1."""
        eigenvalues = np.linalg.eigvalsh(pauli_x_matrix)
        expected = np.array([-1.0, 1.0])
        np.testing.assert_allclose(sorted(eigenvalues), expected, atol=1e-10)

    def test_general_hermitian_2x2_eigenvalues(self, general_hermitian_2x2):
        """M3: Eigenvalues should be exactly 1 and 4.

        Characteristic polynomial: lambda^2 - 5*lambda + 4 = 0
        => (lambda - 1)(lambda - 4) = 0
        """
        eigenvalues = np.linalg.eigvalsh(general_hermitian_2x2)
        expected = np.array([1.0, 4.0])
        np.testing.assert_allclose(sorted(eigenvalues), expected, atol=1e-10)

    def test_symmetric_3x3_eigenvalues(self, symmetric_3x3):
        """M4: Eigenvalues should be exactly 1, 2, and 4.

        Characteristic polynomial: (2-lambda)(lambda-1)(lambda-4) = 0
        """
        eigenvalues = np.linalg.eigvalsh(symmetric_3x3)
        expected = np.array([1.0, 2.0, 4.0])
        np.testing.assert_allclose(sorted(eigenvalues), expected, atol=1e-10)

    def test_heisenberg_xxx_eigenvalues(self, heisenberg_xxx_4x4):
        """M5: Heisenberg XXX eigenvalues: -3, 1, 1, 1."""
        eigenvalues = np.linalg.eigvalsh(heisenberg_xxx_4x4)
        expected = np.array([-3.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(sorted(eigenvalues), expected, atol=1e-10)

    def test_identity_degenerate_eigenvalues(self, identity_2x2):
        """Identity 2x2 matrix has degenerate eigenvalues: 1, 1."""
        eigenvalues = np.linalg.eigvalsh(identity_2x2)
        expected = np.array([1.0, 1.0])
        np.testing.assert_allclose(sorted(eigenvalues), expected, atol=1e-10)

    def test_all_matrices_eigenvalues(self, all_test_matrices):
        """Validate classical eigenvalues for all test matrices at once."""
        for name, (matrix, expected_eigenvalues) in all_test_matrices.items():
            eigenvalues = np.linalg.eigvalsh(matrix)
            np.testing.assert_allclose(
                sorted(eigenvalues),
                sorted(expected_eigenvalues),
                atol=1e-4,
                err_msg=f"Eigenvalue mismatch for {name}",
            )


class TestClassicalEigenvectors:
    """Validate eigenvectors from classical solver."""

    def test_eigenvectors_orthonormal(self, all_test_matrices):
        """Eigenvectors from eigh should be orthonormal."""
        for name, (matrix, _) in all_test_matrices.items():
            _, eigvecs = np.linalg.eigh(matrix)
            # Check orthonormality: V^dagger V = I
            product = eigvecs.conj().T @ eigvecs
            np.testing.assert_allclose(
                product,
                np.eye(matrix.shape[0]),
                atol=1e-10,
                err_msg=f"Eigenvectors not orthonormal for {name}",
            )

    def test_eigendecomposition_reconstructs_matrix(self, all_test_matrices):
        """M = V diag(lambda) V^dagger should reconstruct the original matrix."""
        for name, (matrix, _) in all_test_matrices.items():
            eigenvalues, eigvecs = np.linalg.eigh(matrix)
            reconstructed = eigvecs @ np.diag(eigenvalues) @ eigvecs.conj().T
            np.testing.assert_allclose(
                reconstructed,
                matrix,
                atol=1e-10,
                err_msg=f"Eigendecomposition does not reconstruct matrix for {name}",
            )

    def test_pauli_z_ground_state(self, pauli_z_matrix):
        """Ground state of Pauli Z should be |1> (second basis state)."""
        eigenvalues, eigvecs = np.linalg.eigh(pauli_z_matrix)
        ground_idx = np.argmin(eigenvalues)
        ground_state = eigvecs[:, ground_idx]
        # |1> = [0, 1] (up to global phase)
        assert abs(abs(ground_state[1]) - 1.0) < 1e-10

    def test_pauli_x_ground_state(self, pauli_x_matrix):
        """Ground state of Pauli X should be |-> = (|0> - |1>)/sqrt(2)."""
        eigenvalues, eigvecs = np.linalg.eigh(pauli_x_matrix)
        ground_idx = np.argmin(eigenvalues)
        ground_state = eigvecs[:, ground_idx]
        # |-> = [1/sqrt(2), -1/sqrt(2)] (up to global phase)
        expected = np.array([1, -1]) / np.sqrt(2)
        # Allow global phase difference
        overlap = abs(np.dot(ground_state.conj(), expected))
        assert abs(overlap - 1.0) < 1e-10

    def test_heisenberg_ground_state_is_singlet(self, heisenberg_xxx_4x4):
        """Ground state of Heisenberg XXX should be singlet |psi-> = (|01> - |10>)/sqrt(2).

        From Phase 1: ground state energy = -3, state is singlet.
        """
        eigenvalues, eigvecs = np.linalg.eigh(heisenberg_xxx_4x4)
        ground_idx = np.argmin(eigenvalues)
        ground_state = eigvecs[:, ground_idx]

        # Singlet: |psi-> = (|01> - |10>)/sqrt(2) = [0, 1/sqrt(2), -1/sqrt(2), 0]
        expected = np.array([0, 1, -1, 0]) / np.sqrt(2)
        overlap = abs(np.dot(ground_state.conj(), expected))
        assert abs(overlap - 1.0) < 1e-10


class TestClassicalMatrixProperties:
    """Validate properties of test matrices."""

    def test_all_matrices_are_hermitian(self, all_test_matrices):
        """All test matrices must be Hermitian (M = M^dagger)."""
        for name, (matrix, _) in all_test_matrices.items():
            np.testing.assert_allclose(
                matrix,
                matrix.conj().T,
                atol=1e-10,
                err_msg=f"Matrix {name} is not Hermitian",
            )

    def test_all_matrices_have_real_eigenvalues(self, all_test_matrices):
        """All eigenvalues of Hermitian matrices must be real."""
        for name, (matrix, _) in all_test_matrices.items():
            eigenvalues = np.linalg.eigvals(matrix)
            np.testing.assert_allclose(
                eigenvalues.imag,
                0,
                atol=1e-10,
                err_msg=f"Matrix {name} has complex eigenvalues",
            )

    def test_trace_equals_sum_of_eigenvalues(self, all_test_matrices):
        """Trace of matrix should equal sum of eigenvalues."""
        for name, (matrix, expected_eigenvalues) in all_test_matrices.items():
            trace = np.trace(matrix)
            eigenvalue_sum = sum(expected_eigenvalues)
            np.testing.assert_allclose(
                trace.real,
                eigenvalue_sum,
                atol=1e-10,
                err_msg=f"Trace != sum(eigenvalues) for {name}",
            )

    def test_determinant_equals_product_of_eigenvalues(self, all_test_matrices):
        """Determinant should equal product of eigenvalues."""
        for name, (matrix, expected_eigenvalues) in all_test_matrices.items():
            det = np.linalg.det(matrix)
            eigenvalue_product = np.prod(expected_eigenvalues)
            np.testing.assert_allclose(
                det.real,
                eigenvalue_product,
                atol=1e-6,
                err_msg=f"det(M) != prod(eigenvalues) for {name}",
            )

    def test_spectral_range(self, all_test_matrices):
        """Verify spectral range matches Phase 1 expectations."""
        expected_ranges = {
            "pauli_z": 2.0,
            "pauli_x": 2.0,
            "general_hermitian_2x2": 3.0,
            "symmetric_3x3": 3.0,
            "heisenberg_xxx_4x4": 4.0,
        }
        for name, (matrix, expected_eigenvalues) in all_test_matrices.items():
            spectral_range = max(expected_eigenvalues) - min(expected_eigenvalues)
            assert abs(spectral_range - expected_ranges[name]) < 1e-10, (
                f"Spectral range mismatch for {name}: "
                f"got {spectral_range}, expected {expected_ranges[name]}"
            )

    @pytest.mark.parametrize("size", [2, 4, 8])
    def test_random_hermitian_matrix_properties(self, size):
        """Random Hermitian matrices should have real eigenvalues."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
        M = (A + A.conj().T) / 2

        # Must be Hermitian
        np.testing.assert_allclose(M, M.conj().T, atol=1e-10)

        # Must have real eigenvalues
        eigenvalues = np.linalg.eigvalsh(M)
        assert all(np.isreal(eigenvalues))

        # Trace = sum of eigenvalues
        np.testing.assert_allclose(np.trace(M).real, sum(eigenvalues), atol=1e-10)


class TestEmbedding3x3:
    """Test the 3x3 to 4x4 penalty embedding from Phase 2."""

    def test_embedded_matrix_is_hermitian(self, symmetric_3x3_embedded):
        """Embedded 4x4 matrix must be Hermitian."""
        M = symmetric_3x3_embedded
        np.testing.assert_allclose(M, M.conj().T, atol=1e-10)

    def test_embedded_matrix_eigenvalues(self, symmetric_3x3_embedded):
        """Embedded matrix should have eigenvalues {1, 2, 4, 10}."""
        eigenvalues = np.linalg.eigvalsh(symmetric_3x3_embedded)
        expected = np.array([1.0, 2.0, 4.0, 10.0])
        np.testing.assert_allclose(sorted(eigenvalues), expected, atol=1e-10)

    def test_embedded_preserves_original_eigenvalues(self, symmetric_3x3, symmetric_3x3_embedded):
        """First 3 eigenvalues of embedded matrix match original 3x3 eigenvalues."""
        original_eigenvalues = sorted(np.linalg.eigvalsh(symmetric_3x3))
        embedded_eigenvalues = sorted(np.linalg.eigvalsh(symmetric_3x3_embedded))

        # First 3 eigenvalues should match
        np.testing.assert_allclose(embedded_eigenvalues[:3], original_eigenvalues, atol=1e-10)

    def test_embedded_ground_state_matches_original(self, symmetric_3x3, symmetric_3x3_embedded):
        """Ground state of embedded matrix should correspond to original ground state."""
        _, orig_vecs = np.linalg.eigh(symmetric_3x3)
        _, emb_vecs = np.linalg.eigh(symmetric_3x3_embedded)

        # Ground state of embedded should be zero-padded version of original
        orig_ground = orig_vecs[:, 0]
        emb_ground = emb_vecs[:, 0]

        # First 3 components should match (up to global phase)
        overlap = abs(np.dot(orig_ground.conj(), emb_ground[:3]))
        assert abs(overlap - 1.0) < 1e-10, "Ground state mismatch between original and embedded"

    def test_penalty_exceeds_max_eigenvalue(self, symmetric_3x3):
        """Penalty value should exceed max eigenvalue of original matrix.

        From Phase 2, Eq. (5): p = lambda_max + 2 * spectral_range
        """
        eigenvalues = np.linalg.eigvalsh(symmetric_3x3)
        lambda_max = max(eigenvalues)
        spectral_range = max(eigenvalues) - min(eigenvalues)
        penalty = lambda_max + 2 * spectral_range  # = 4 + 2*3 = 10

        assert penalty > lambda_max
        assert abs(penalty - 10.0) < 1e-10


class TestClassicalEigensolverInterface:
    """Tests for the ClassicalEigensolver class interface.

    These tests define the API contract that the ClassicalEigensolver
    implementation must satisfy. They will FAIL until Phase 4 implementation.
    """

    def test_classical_eigensolver_instantiation(self, pauli_z_matrix):
        """ClassicalEigensolver should accept a numpy matrix."""
        from eigen_solver.src.classical_baseline import ClassicalEigensolver

        solver = ClassicalEigensolver(pauli_z_matrix)
        assert solver.matrix is not None

    def test_classical_eigensolver_solve_returns_result(self, pauli_z_matrix):
        """ClassicalEigensolver.solve() should return an EigensolverResult."""
        from eigen_solver.src.classical_baseline import ClassicalEigensolver
        from eigen_solver.src.quantum_eigensolver import EigensolverResult

        solver = ClassicalEigensolver(pauli_z_matrix)
        result = solver.solve()
        assert isinstance(result, EigensolverResult)

    def test_classical_eigensolver_finds_minimum_eigenvalue(self, pauli_z_matrix):
        """ClassicalEigensolver.solve() should find minimum eigenvalue."""
        from eigen_solver.src.classical_baseline import ClassicalEigensolver

        solver = ClassicalEigensolver(pauli_z_matrix)
        result = solver.solve()
        assert abs(result.eigenvalue - (-1.0)) < 1e-10

    def test_classical_eigensolver_solve_all(self, pauli_z_matrix):
        """ClassicalEigensolver.solve_all() should find all eigenvalues."""
        from eigen_solver.src.classical_baseline import ClassicalEigensolver

        solver = ClassicalEigensolver(pauli_z_matrix)
        eigenvalues = solver.solve_all()
        np.testing.assert_allclose(sorted(eigenvalues), [-1.0, 1.0], atol=1e-10)

    def test_classical_eigensolver_all_matrices(self, all_test_matrices):
        """ClassicalEigensolver should correctly solve all test matrices."""
        from eigen_solver.src.classical_baseline import ClassicalEigensolver

        for name, (matrix, expected) in all_test_matrices.items():
            solver = ClassicalEigensolver(matrix)
            eigenvalues = solver.solve_all()
            np.testing.assert_allclose(
                sorted(eigenvalues),
                sorted(expected),
                atol=1e-4,
                err_msg=f"ClassicalEigensolver failed for {name}",
            )

    def test_classical_eigensolver_result_has_eigenvector(self, pauli_z_matrix):
        """EigensolverResult should contain the ground state eigenvector."""
        from eigen_solver.src.classical_baseline import ClassicalEigensolver

        solver = ClassicalEigensolver(pauli_z_matrix)
        result = solver.solve()
        assert result.eigenvector is not None
        assert len(result.eigenvector) == 2

    def test_classical_eigensolver_result_iterations_zero(self, pauli_z_matrix):
        """ClassicalEigensolver should report 0 iterations (direct solver)."""
        from eigen_solver.src.classical_baseline import ClassicalEigensolver

        solver = ClassicalEigensolver(pauli_z_matrix)
        result = solver.solve()
        assert result.iterations == 0

    def test_classical_eigensolver_result_no_parameters(self, pauli_z_matrix):
        """ClassicalEigensolver should have no optimal_parameters."""
        from eigen_solver.src.classical_baseline import ClassicalEigensolver

        solver = ClassicalEigensolver(pauli_z_matrix)
        result = solver.solve()
        assert result.optimal_parameters is None
