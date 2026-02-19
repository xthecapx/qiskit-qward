"""
Tests for Pauli decomposition of Hermitian matrices.

Specification:
  - phase2_theoretical_design.md - Section 2 (Pauli Decomposition)
  - phase2_theoretical_design.md - Section 3 (Non-Power-of-Two Embedding)
Author: test-engineer

These tests verify that an arbitrary Hermitian matrix can be correctly
decomposed into a sum of Pauli tensor products with real coefficients:

    H = (1/2^q) * sum_P Tr(M * P) * P

where P ranges over all q-qubit Pauli strings {I, X, Y, Z}^q.

All tests will FAIL until Phase 4 implementation.
"""

import numpy as np
import pytest

# Pauli matrices for reference in tests
PAULI_I = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Known decompositions from Phase 2, Section 2.3
KNOWN_DECOMPOSITIONS = {
    "pauli_z": {"Z": 1.0},
    "pauli_x": {"X": 1.0},
    "general_hermitian_2x2": {"I": 2.5, "X": 1.0, "Y": 1.0, "Z": -0.5},
    "heisenberg_xxx_4x4": {"XX": 1.0, "YY": 1.0, "ZZ": 1.0},
}


def _reconstruct_from_pauli_dict(coefficients, num_qubits):
    """Reconstruct a matrix from a Pauli string coefficient dictionary.

    Args:
        coefficients: dict mapping Pauli string labels to real coefficients.
        num_qubits: number of qubits.

    Returns:
        np.ndarray: the reconstructed matrix.
    """
    pauli_map = {"I": PAULI_I, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}
    dim = 2**num_qubits
    result = np.zeros((dim, dim), dtype=complex)

    for label, coeff in coefficients.items():
        # Build tensor product for multi-qubit Pauli strings
        mat = np.array([[1.0]], dtype=complex)
        for char in label:
            mat = np.kron(mat, pauli_map[char])
        result += coeff * mat

    return result


class TestPauliDecomposition2x2:
    """Test Pauli decomposition for 2x2 matrices (1 qubit)."""

    def test_pauli_z_decomposition(self, pauli_z_matrix):
        """M1: Pauli Z should decompose to {Z: 1.0}."""
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        decomposition = pauli_decompose(pauli_z_matrix)
        reconstructed = _reconstruct_from_pauli_dict(decomposition, num_qubits=1)
        np.testing.assert_allclose(reconstructed, pauli_z_matrix, atol=1e-10)

    def test_pauli_x_decomposition(self, pauli_x_matrix):
        """M2: Pauli X should decompose to {X: 1.0}."""
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        decomposition = pauli_decompose(pauli_x_matrix)
        reconstructed = _reconstruct_from_pauli_dict(decomposition, num_qubits=1)
        np.testing.assert_allclose(reconstructed, pauli_x_matrix, atol=1e-10)

    def test_general_hermitian_2x2_decomposition(self, general_hermitian_2x2):
        """M3: Should decompose to 2.5*I + 1.0*X + 1.0*Y - 0.5*Z.

        From Phase 2, Section 2.3.
        """
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        decomposition = pauli_decompose(general_hermitian_2x2)
        reconstructed = _reconstruct_from_pauli_dict(decomposition, num_qubits=1)
        np.testing.assert_allclose(reconstructed, general_hermitian_2x2, atol=1e-10)

    def test_general_hermitian_2x2_known_coefficients(self, general_hermitian_2x2):
        """M3: Verify specific Pauli coefficients match Phase 2 prediction."""
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        decomposition = pauli_decompose(general_hermitian_2x2)
        expected = KNOWN_DECOMPOSITIONS["general_hermitian_2x2"]

        for pauli_label, expected_coeff in expected.items():
            actual_coeff = decomposition.get(pauli_label, 0.0)
            assert abs(actual_coeff - expected_coeff) < 1e-10, (
                f"Coefficient for {pauli_label}: expected {expected_coeff}, " f"got {actual_coeff}"
            )

    def test_identity_decomposition(self, identity_2x2):
        """Identity matrix should decompose to {I: 1.0}."""
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        decomposition = pauli_decompose(identity_2x2)
        reconstructed = _reconstruct_from_pauli_dict(decomposition, num_qubits=1)
        np.testing.assert_allclose(reconstructed, identity_2x2, atol=1e-10)

    def test_decomposition_coefficients_are_real(self, general_hermitian_2x2):
        """All Pauli coefficients for Hermitian matrices must be real.

        From Phase 2, Section 2.1: "If M is Hermitian, all c_P are real."
        """
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        decomposition = pauli_decompose(general_hermitian_2x2)
        for label, coeff in decomposition.items():
            assert np.isreal(coeff), f"Coefficient for {label} is not real: {coeff}"


class TestPauliDecomposition4x4:
    """Test Pauli decomposition for 4x4 matrices (2 qubits)."""

    def test_heisenberg_xxx_decomposition(self, heisenberg_xxx_4x4):
        """M5: Heisenberg XXX should decompose to XX + YY + ZZ (3 terms)."""
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        decomposition = pauli_decompose(heisenberg_xxx_4x4)
        reconstructed = _reconstruct_from_pauli_dict(decomposition, num_qubits=2)
        np.testing.assert_allclose(reconstructed, heisenberg_xxx_4x4, atol=1e-10)

    def test_heisenberg_xxx_known_coefficients(self, heisenberg_xxx_4x4):
        """M5: Verify Pauli coefficients are XX=1, YY=1, ZZ=1."""
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        decomposition = pauli_decompose(heisenberg_xxx_4x4)
        expected = KNOWN_DECOMPOSITIONS["heisenberg_xxx_4x4"]

        for pauli_label, expected_coeff in expected.items():
            actual_coeff = decomposition.get(pauli_label, 0.0)
            assert abs(actual_coeff - expected_coeff) < 1e-10, (
                f"Heisenberg coefficient for {pauli_label}: "
                f"expected {expected_coeff}, got {actual_coeff}"
            )

    def test_heisenberg_xxx_sparse_decomposition(self, heisenberg_xxx_4x4):
        """M5: Heisenberg should have only 3 non-zero Pauli terms."""
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        decomposition = pauli_decompose(heisenberg_xxx_4x4)
        nonzero = {k: v for k, v in decomposition.items() if abs(v) > 1e-10}
        assert len(nonzero) == 3, (
            f"Expected 3 non-zero Pauli terms for Heisenberg, got {len(nonzero)}: " f"{nonzero}"
        )

    def test_4x4_identity_decomposition(self):
        """4x4 identity should decompose to {II: 1.0}."""
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        identity_4x4 = np.eye(4, dtype=complex)
        decomposition = pauli_decompose(identity_4x4)
        reconstructed = _reconstruct_from_pauli_dict(decomposition, num_qubits=2)
        np.testing.assert_allclose(reconstructed, identity_4x4, atol=1e-10)

    def test_random_hermitian_4x4_decomposition(self, random_hermitian_4x4):
        """Random 4x4 Hermitian matrix should decompose and reconstruct."""
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        decomposition = pauli_decompose(random_hermitian_4x4)
        reconstructed = _reconstruct_from_pauli_dict(decomposition, num_qubits=2)
        np.testing.assert_allclose(reconstructed, random_hermitian_4x4, atol=1e-10)

    def test_4x4_decomposition_coefficients_are_real(self, heisenberg_xxx_4x4):
        """All Pauli coefficients for 4x4 Hermitian must be real."""
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        decomposition = pauli_decompose(heisenberg_xxx_4x4)
        for label, coeff in decomposition.items():
            assert np.isreal(coeff), f"Coefficient for {label} is not real: {coeff}"

    def test_embedded_3x3_decomposition(self, symmetric_3x3_embedded):
        """M4 embedded: 3x3 embedded in 4x4 should decompose correctly.

        From Phase 2, Section 2.3: 8 Pauli terms with penalty=10.
        """
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        decomposition = pauli_decompose(symmetric_3x3_embedded)
        reconstructed = _reconstruct_from_pauli_dict(decomposition, num_qubits=2)
        np.testing.assert_allclose(reconstructed, symmetric_3x3_embedded, atol=1e-10)


class TestPauliDecompositionProperties:
    """Test general properties of Pauli decomposition."""

    def test_decomposition_preserves_hermiticity(self, all_test_matrices):
        """Reconstructed matrix from Pauli decomposition must be Hermitian."""
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        for name, (matrix, _) in all_test_matrices.items():
            if matrix.shape[0] != matrix.shape[1]:
                continue
            n = matrix.shape[0]
            num_qubits = int(np.ceil(np.log2(n)))
            if 2**num_qubits != n:
                continue  # Skip non-power-of-2 (needs embedding)

            decomposition = pauli_decompose(matrix)
            reconstructed = _reconstruct_from_pauli_dict(decomposition, num_qubits)
            np.testing.assert_allclose(
                reconstructed,
                reconstructed.conj().T,
                atol=1e-10,
                err_msg=f"Reconstruction not Hermitian for {name}",
            )

    def test_decomposition_preserves_eigenvalues(self, all_test_matrices):
        """Eigenvalues of reconstructed matrix must match original."""
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        for name, (matrix, expected_eigenvalues) in all_test_matrices.items():
            n = matrix.shape[0]
            num_qubits = int(np.ceil(np.log2(n)))
            if 2**num_qubits != n:
                continue  # Skip non-power-of-2

            decomposition = pauli_decompose(matrix)
            reconstructed = _reconstruct_from_pauli_dict(decomposition, num_qubits)
            eigenvalues = np.linalg.eigvalsh(reconstructed)
            np.testing.assert_allclose(
                sorted(eigenvalues),
                sorted(expected_eigenvalues),
                atol=1e-4,
                err_msg=f"Eigenvalue mismatch after reconstruction for {name}",
            )

    def test_decomposition_preserves_trace(self, all_test_matrices):
        """Trace of reconstructed matrix must match original."""
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        for name, (matrix, _) in all_test_matrices.items():
            n = matrix.shape[0]
            num_qubits = int(np.ceil(np.log2(n)))
            if 2**num_qubits != n:
                continue

            decomposition = pauli_decompose(matrix)
            reconstructed = _reconstruct_from_pauli_dict(decomposition, num_qubits)
            np.testing.assert_allclose(
                np.trace(reconstructed),
                np.trace(matrix),
                atol=1e-10,
                err_msg=f"Trace mismatch for {name}",
            )

    def test_number_of_pauli_terms(self, all_test_matrices):
        """Number of non-zero Pauli terms should be at most 4^q."""
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        for name, (matrix, _) in all_test_matrices.items():
            n = matrix.shape[0]
            num_qubits = int(np.ceil(np.log2(n)))
            if 2**num_qubits != n:
                continue

            decomposition = pauli_decompose(matrix)
            max_terms = 4**num_qubits
            assert len(decomposition) <= max_terms, (
                f"Too many Pauli terms ({len(decomposition)} > {max_terms}) " f"for {name}"
            )


class TestPauliDecompositionEdgeCases:
    """Test edge cases and error handling for Pauli decomposition.

    From Phase 2, Section 10.2.
    """

    def test_zero_matrix(self):
        """Zero matrix should decompose to all-zero coefficients."""
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        zero = np.zeros((2, 2), dtype=complex)
        decomposition = pauli_decompose(zero)
        # All coefficients should be zero (or dict should be empty)
        for coeff in decomposition.values():
            assert abs(coeff) < 1e-10

    def test_non_hermitian_raises_error(self):
        """Non-Hermitian matrix should raise an error."""
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        non_hermitian = np.array([[1, 2], [3, 4]], dtype=complex)
        with pytest.raises((ValueError, TypeError)):
            pauli_decompose(non_hermitian)

    def test_non_square_raises_error(self):
        """Non-square matrix should raise an error."""
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        non_square = np.array([[1, 2, 3], [4, 5, 6]], dtype=complex)
        with pytest.raises((ValueError, TypeError)):
            pauli_decompose(non_square)

    def test_non_power_of_two_dimension(self, symmetric_3x3):
        """3x3 matrix (not power of 2) -- should either embed or raise.

        From Phase 2, Section 3: 3x3 matrices need penalty embedding into 4x4.
        The implementation should either:
        1. Automatically embed and decompose, or
        2. Raise an informative error about dimension requirements.
        """
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        try:
            decomposition = pauli_decompose(symmetric_3x3)
            # If it succeeds with auto-embedding, verify eigenvalues are preserved
            reconstructed = _reconstruct_from_pauli_dict(decomposition, num_qubits=2)
            original_eigenvalues = sorted(np.linalg.eigvalsh(symmetric_3x3))
            reconstructed_eigenvalues = sorted(np.linalg.eigvalsh(reconstructed))

            # Original eigenvalues should be a subset
            for ev in original_eigenvalues:
                assert any(
                    abs(ev - rev) < 1e-6 for rev in reconstructed_eigenvalues
                ), f"Original eigenvalue {ev} not found in reconstruction"
        except (ValueError, NotImplementedError):
            # Acceptable: implementation may require power-of-2 dimensions
            pass

    def test_negative_definite_matrix(self):
        """Negative-definite matrix: all eigenvalues negative.

        Edge case from Phase 2, Section 10.2.
        """
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        M = np.array([[-3, 0], [0, -1]], dtype=complex)
        decomposition = pauli_decompose(M)
        reconstructed = _reconstruct_from_pauli_dict(decomposition, num_qubits=1)
        np.testing.assert_allclose(reconstructed, M, atol=1e-10)

    def test_large_spectral_range(self):
        """Matrix with eigenvalues {0, 1000}: tests numerical stability.

        Edge case from Phase 2, Section 10.2.
        """
        from eigen_solver.src.pauli_decomposition import pauli_decompose

        M = np.array([[0, 0], [0, 1000]], dtype=complex)
        decomposition = pauli_decompose(M)
        reconstructed = _reconstruct_from_pauli_dict(decomposition, num_qubits=1)
        np.testing.assert_allclose(reconstructed, M, atol=1e-6)
