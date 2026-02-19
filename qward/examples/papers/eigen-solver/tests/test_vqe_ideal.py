"""
Tests for ideal (noiseless) VQE eigensolver.

Specification:
  - phase1_problem_statement.md (success criteria: 1% relative error)
  - phase2_theoretical_design.md (Sections 4-8: ansatz, measurement, proofs)
Author: test-engineer

These tests verify that the VQE-based quantum eigensolver finds correct
eigenvalues under ideal (statevector) simulation with no noise.

Success criteria (Phase 1, Section 4.1):
  - |lambda_VQE - lambda_exact| / spectral_range < 0.01 (1%)

Expected behaviors (Phase 2, Section 10.1):
  - M1 (Pauli Z): converges in few iterations, ground state |1>
  - M2 (Pauli X): creates superposition, ground state |->
  - M3 (General 2x2): correct energy E=1 despite complex coefficients
  - M4 (embedded 3x3): ignores penalty subspace, E=1
  - M5 (Heisenberg XXX): creates Bell-like state, E=-3

All tests will FAIL until Phase 4 implementation.
"""

import numpy as np
import pytest


class TestVQEGroundState2x2:
    """Test VQE finds ground state (minimum eigenvalue) for 2x2 matrices."""

    def test_vqe_ground_state_pauli_z(self, pauli_z_matrix, ideal_tolerance):
        """M1: VQE finds ground state energy -1 for Pauli Z (trivial).

        From Phase 2: "No optimization needed -- serves as a sanity check."
        """
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(pauli_z_matrix)
        result = solver.solve(shots=None)  # Statevector simulation

        classical_min = -1.0
        spectral_range = 2.0
        relative_error = abs(result.eigenvalue - classical_min) / spectral_range

        assert relative_error < ideal_tolerance, (
            f"VQE eigenvalue {result.eigenvalue} differs from classical "
            f"{classical_min} by {relative_error:.4f} of spectral range"
        )

    def test_vqe_ground_state_pauli_x(self, pauli_x_matrix, ideal_tolerance):
        """M2: VQE finds ground state energy -1 for Pauli X.

        From Phase 2: "Ground state is |-> = (|0> - |1>)/sqrt(2).
        Requires the ansatz to create a superposition state."
        """
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(pauli_x_matrix)
        result = solver.solve(shots=None)

        classical_min = -1.0
        spectral_range = 2.0
        relative_error = abs(result.eigenvalue - classical_min) / spectral_range

        assert relative_error < ideal_tolerance, (
            f"VQE eigenvalue {result.eigenvalue} differs from classical "
            f"{classical_min} by {relative_error:.4f} of spectral range"
        )

    def test_vqe_ground_state_general_hermitian(self, general_hermitian_2x2, ideal_tolerance):
        """M3: VQE finds ground state energy E=1 for general 2x2 Hermitian.

        From Phase 2: "Tests the ansatz's ability to represent states with
        complex amplitudes." Requires RY+RZ or full U3 parameterization.
        """
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(general_hermitian_2x2)
        result = solver.solve(shots=None)

        classical_min = 1.0  # Corrected from Phase 1
        spectral_range = 3.0
        relative_error = abs(result.eigenvalue - classical_min) / spectral_range

        assert relative_error < ideal_tolerance, (
            f"VQE eigenvalue {result.eigenvalue} differs from classical "
            f"{classical_min} by {relative_error:.4f} of spectral range"
        )


class TestVQEGroundState4x4:
    """Test VQE finds ground state for 4x4 matrices (2 qubits)."""

    def test_vqe_ground_state_heisenberg(self, heisenberg_xxx_4x4, ideal_tolerance):
        """M5: VQE finds ground state energy -3 for Heisenberg XXX.

        From Phase 2: "Ground state is the singlet |psi-> = (|01> - |10>)/sqrt(2)
        with energy -3. Tests the ansatz's ability to create entangled states."
        """
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(heisenberg_xxx_4x4)
        result = solver.solve(shots=None)

        classical_min = -3.0
        spectral_range = 4.0
        relative_error = abs(result.eigenvalue - classical_min) / spectral_range

        assert relative_error < ideal_tolerance, (
            f"VQE eigenvalue {result.eigenvalue} differs from classical "
            f"{classical_min} by {relative_error:.4f} of spectral range"
        )

    def test_vqe_ground_state_embedded_3x3(self, symmetric_3x3_embedded, ideal_tolerance):
        """M4 embedded: VQE finds ground state E=1, ignores penalty subspace.

        From Phase 2: "VQE must find the correct ground state while the
        optimizer avoids the penalized subspace."
        """
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(symmetric_3x3_embedded)
        result = solver.solve(shots=None)

        classical_min = 1.0  # Corrected: min eigenvalue of original 3x3
        spectral_range = 9.0  # 10 - 1 for embedded matrix
        relative_error = abs(result.eigenvalue - classical_min) / spectral_range

        assert relative_error < ideal_tolerance, (
            f"Embedded VQE eigenvalue {result.eigenvalue} differs from "
            f"classical {classical_min}"
        )
        # Extra check: must NOT find the penalty eigenvalue
        assert result.eigenvalue < 5.0, f"VQE converged to penalty state: E={result.eigenvalue}"


class TestVQEAllEigenvalues:
    """Test VQE finds all eigenvalues via deflation.

    From Phase 2, Section 7: Eigenvalue deflation protocol.
    """

    def test_vqe_all_eigenvalues_pauli_z(self, pauli_z_matrix):
        """VQE finds all eigenvalues [-1, 1] for Pauli Z via deflation."""
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(pauli_z_matrix)
        quantum_eigenvalues = solver.solve_all()

        classical_eigenvalues = [-1.0, 1.0]
        spectral_range = 2.0
        deflation_tolerance = 0.05  # 5% for deflation

        for q_ev, c_ev in zip(sorted(quantum_eigenvalues), sorted(classical_eigenvalues)):
            relative_error = abs(q_ev - c_ev) / spectral_range
            assert relative_error < deflation_tolerance, (
                f"Eigenvalue {q_ev} differs from {c_ev} "
                f"by {relative_error:.4f} of spectral range"
            )

    def test_vqe_all_eigenvalues_heisenberg(self, heisenberg_xxx_4x4):
        """VQE finds all eigenvalues [-3, 1, 1, 1] for Heisenberg.

        From Phase 2: "The 3-fold degeneracy at eigenvalue 1 tests
        deflation robustness."
        """
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(heisenberg_xxx_4x4)
        quantum_eigenvalues = solver.solve_all()

        classical_eigenvalues = [-3.0, 1.0, 1.0, 1.0]
        spectral_range = 4.0
        deflation_tolerance = 0.05

        for q_ev, c_ev in zip(sorted(quantum_eigenvalues), sorted(classical_eigenvalues)):
            relative_error = abs(q_ev - c_ev) / spectral_range
            assert relative_error < deflation_tolerance, (
                f"Eigenvalue {q_ev} differs from {c_ev} "
                f"by {relative_error:.4f} of spectral range"
            )

    def test_vqe_all_eigenvalues_general_hermitian(self, general_hermitian_2x2):
        """VQE finds all eigenvalues [1, 4] for M3."""
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(general_hermitian_2x2)
        quantum_eigenvalues = solver.solve_all()

        classical_eigenvalues = [1.0, 4.0]
        spectral_range = 3.0
        deflation_tolerance = 0.05

        for q_ev, c_ev in zip(sorted(quantum_eigenvalues), sorted(classical_eigenvalues)):
            relative_error = abs(q_ev - c_ev) / spectral_range
            assert relative_error < deflation_tolerance

    def test_vqe_correct_number_of_eigenvalues(self, all_test_matrices):
        """VQE solve_all should return correct number of eigenvalues."""
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        for name, (matrix, expected_eigenvalues) in all_test_matrices.items():
            solver = QuantumEigensolver(matrix)
            quantum_eigenvalues = solver.solve_all()
            assert len(quantum_eigenvalues) == len(expected_eigenvalues), (
                f"Expected {len(expected_eigenvalues)} eigenvalues for {name}, "
                f"got {len(quantum_eigenvalues)}"
            )


class TestVQEResult:
    """Test the structure and properties of VQE results."""

    def test_result_has_eigenvalue(self, pauli_z_matrix):
        """EigensolverResult must have an eigenvalue attribute."""
        from eigen_solver.src.quantum_eigensolver import (
            QuantumEigensolver,
            EigensolverResult,
        )

        solver = QuantumEigensolver(pauli_z_matrix)
        result = solver.solve(shots=None)
        assert isinstance(result, EigensolverResult)
        assert hasattr(result, "eigenvalue")
        assert isinstance(result.eigenvalue, (int, float, np.floating))

    def test_result_has_optimal_parameters(self, pauli_z_matrix):
        """EigensolverResult must have optimal_parameters from VQE."""
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(pauli_z_matrix)
        result = solver.solve(shots=None)
        assert result.optimal_parameters is not None

    def test_result_has_iteration_count(self, pauli_z_matrix):
        """EigensolverResult must report number of optimizer iterations."""
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(pauli_z_matrix)
        result = solver.solve(shots=None)
        assert hasattr(result, "iterations")
        assert result.iterations > 0

    def test_eigenvalue_is_upper_bound(self, all_test_matrices):
        """VQE eigenvalue should be >= true ground state (variational principle).

        From Phase 2, Section 8.1: For all normalized |psi>,
        <psi|H|psi> >= E_0 with equality iff |psi> is ground state.
        """
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        for name, (matrix, expected_eigenvalues) in all_test_matrices.items():
            n = matrix.shape[0]
            num_qubits = int(np.ceil(np.log2(n)))
            if 2**num_qubits != n:
                continue  # Skip non-power-of-2

            solver = QuantumEigensolver(matrix)
            result = solver.solve(shots=None)
            true_min = min(expected_eigenvalues)

            # Allow small numerical tolerance below
            assert result.eigenvalue >= true_min - 1e-6, (
                f"VQE result {result.eigenvalue} violates variational principle "
                f"(true min = {true_min}) for {name}"
            )


class TestVQEInstantiation:
    """Test QuantumEigensolver constructor and configuration."""

    def test_instantiation_with_matrix(self, pauli_z_matrix):
        """QuantumEigensolver should accept a numpy matrix."""
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(pauli_z_matrix)
        assert solver is not None

    def test_instantiation_with_custom_shots(self, pauli_z_matrix):
        """QuantumEigensolver should accept custom shot count."""
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(pauli_z_matrix, shots=8192)
        assert solver.shots == 8192

    def test_instantiation_creates_hamiltonian(self, pauli_z_matrix):
        """QuantumEigensolver should create a Hamiltonian from the matrix."""
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(pauli_z_matrix)
        assert solver.hamiltonian is not None

    def test_instantiation_creates_default_ansatz(self, pauli_z_matrix):
        """QuantumEigensolver should create a default ansatz if none given.

        From Phase 2, Section 4.5: Default is RY+RZ for 1-qubit,
        EfficientSU2 for 2-qubit.
        """
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(pauli_z_matrix)
        assert solver.ansatz is not None

    def test_non_hermitian_matrix_raises(self):
        """Passing a non-Hermitian matrix should raise an error.

        From Phase 2, Section 10.2: Hermiticity validation.
        """
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        non_hermitian = np.array([[1, 2], [3, 4]], dtype=complex)
        with pytest.raises((ValueError, TypeError)):
            QuantumEigensolver(non_hermitian)


class TestVQEWithAllMatrices:
    """Test VQE on all power-of-2 test matrices."""

    @pytest.mark.parametrize(
        "matrix_name",
        [
            "pauli_z",
            "pauli_x",
            "general_hermitian_2x2",
            "heisenberg_xxx_4x4",
        ],
    )
    def test_vqe_finds_ground_state(self, matrix_name, all_test_matrices, ideal_tolerance):
        """VQE finds ground state for each test matrix within 1% tolerance."""
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        matrix, expected_eigenvalues = all_test_matrices[matrix_name]
        solver = QuantumEigensolver(matrix)
        result = solver.solve(shots=None)

        classical_min = min(expected_eigenvalues)
        spectral_range = max(expected_eigenvalues) - min(expected_eigenvalues)

        if spectral_range > 0:
            relative_error = abs(result.eigenvalue - classical_min) / spectral_range
            assert relative_error < ideal_tolerance, (
                f"VQE error {relative_error:.4f} exceeds 1% tolerance " f"for {matrix_name}"
            )
        else:
            # Degenerate case: all eigenvalues are the same
            assert abs(result.eigenvalue - classical_min) < 0.01


class TestVQEEdgeCases:
    """Edge cases from Phase 2, Section 10.2."""

    def test_vqe_on_identity_matrix(self, identity_2x2):
        """Identity matrix: VQE should converge to E=1 immediately.

        All eigenvalues are 1, so any state is an eigenstate.
        """
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(identity_2x2)
        result = solver.solve(shots=None)
        assert abs(result.eigenvalue - 1.0) < 0.01

    def test_vqe_on_diagonal_matrix(self, pauli_z_matrix):
        """Already-diagonal matrix: ansatz should converge quickly.

        Ground state is a computational basis state |1>.
        """
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(pauli_z_matrix)
        result = solver.solve(shots=None)

        # Should converge quickly for diagonal matrix
        assert (
            result.iterations <= 50
        ), f"Diagonal matrix took too many iterations: {result.iterations}"
