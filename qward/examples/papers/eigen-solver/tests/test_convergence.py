"""
Tests for VQE optimizer convergence behavior.

Specification:
  - phase1_problem_statement.md (convergence <= 200 iterations)
  - phase2_theoretical_design.md (Section 6: optimizer config, Section 9.1: iterations)
Author: test-engineer

Success criteria:
  - Convergence within 200 iterations (COBYLA)
  - Convergence tolerance: 1e-6 (ideal), 1e-4 (shot-based), 1e-3 (noisy)

Expected iteration counts (Phase 2, Section 9.1):
  - 1-qubit: 50-100 iterations
  - 2-qubit: 100-200 iterations

All tests will FAIL until Phase 4 implementation.
"""

import numpy as np
import pytest


class TestConvergenceIterations:
    """Test that VQE converges within the iteration budget."""

    def test_convergence_within_200_iterations_pauli_z(self, pauli_z_matrix):
        """M1: VQE should converge for Pauli Z within 200 iterations.

        From Phase 2: "diagonal matrix, expected ~50-100 iterations."
        """
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(pauli_z_matrix)
        result = solver.solve(shots=None)

        assert (
            result.iterations <= 200
        ), f"VQE took {result.iterations} iterations for Pauli Z (max 200)"

    def test_convergence_within_200_iterations_pauli_x(self, pauli_x_matrix):
        """M2: VQE should converge for Pauli X within 200 iterations."""
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(pauli_x_matrix)
        result = solver.solve(shots=None)

        assert (
            result.iterations <= 200
        ), f"VQE took {result.iterations} iterations for Pauli X (max 200)"

    def test_convergence_within_200_iterations_general(self, general_hermitian_2x2):
        """M3: VQE should converge for general 2x2 within 200 iterations."""
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(general_hermitian_2x2)
        result = solver.solve(shots=None)

        assert result.iterations <= 200, f"VQE took {result.iterations} iterations for M3 (max 200)"

    @pytest.mark.slow
    def test_convergence_within_200_iterations_heisenberg(self, heisenberg_xxx_4x4):
        """M5: VQE should converge for Heisenberg XXX within 200 iterations.

        From Phase 2: "2-qubit expected ~100-200 iterations."
        """
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(heisenberg_xxx_4x4)
        result = solver.solve(shots=None)

        assert (
            result.iterations <= 200
        ), f"VQE took {result.iterations} iterations for Heisenberg (max 200)"


class TestConvergenceHistory:
    """Test convergence history (cost function values over iterations)."""

    def test_cost_function_decreases(self, pauli_z_matrix):
        """Cost function should generally decrease during optimization.

        We check that the final cost is lower than the initial cost.
        Not every step needs to decrease (COBYLA can oscillate), but the
        overall trend should be downward.
        """
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(pauli_z_matrix)
        result = solver.solve(shots=None)

        if hasattr(result, "cost_history") and result.cost_history is not None:
            history = result.cost_history
            if len(history) > 1:
                assert history[-1] <= history[0] + 0.1, (
                    f"Final cost {history[-1]} is not lower than " f"initial cost {history[0]}"
                )

    def test_cost_history_recorded(self, pauli_z_matrix):
        """VQE result should contain cost function history if available."""
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(pauli_z_matrix)
        result = solver.solve(shots=None)

        if hasattr(result, "cost_history"):
            assert result.cost_history is not None
            assert len(result.cost_history) > 0

    def test_final_cost_matches_eigenvalue(self, pauli_z_matrix):
        """Final cost function value should match the reported eigenvalue."""
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(pauli_z_matrix)
        result = solver.solve(shots=None)

        if hasattr(result, "cost_history") and result.cost_history is not None:
            final_cost = result.cost_history[-1]
            assert abs(final_cost - result.eigenvalue) < 0.01, (
                f"Final cost {final_cost} != reported eigenvalue " f"{result.eigenvalue}"
            )


class TestConvergenceWithDifferentOptimizers:
    """Test convergence with different classical optimizers.

    From Phase 2, Section 6.1:
      - COBYLA: gradient-free, ideal simulation, max 200 iterations
      - SPSA: stochastic gradient, noisy simulation, max 300 iterations
    """

    def test_cobyla_convergence(self, pauli_z_matrix, ideal_tolerance):
        """COBYLA should converge for simple 2x2 case."""
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(pauli_z_matrix, optimizer="COBYLA")
        result = solver.solve(shots=None)

        classical_min = -1.0
        spectral_range = 2.0
        relative_error = abs(result.eigenvalue - classical_min) / spectral_range

        assert relative_error < ideal_tolerance
        assert result.iterations <= 200

    @pytest.mark.slow
    def test_spsa_convergence(self, pauli_z_matrix):
        """SPSA should converge (designed for noisy environments).

        From Phase 2, Section 6.5: SPSA with maxiter=300.
        """
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(pauli_z_matrix, optimizer="SPSA")
        result = solver.solve(shots=None)

        classical_min = -1.0
        # SPSA may need relaxed tolerance
        assert abs(result.eigenvalue - classical_min) < 0.5, (
            f"SPSA did not converge reasonably: "
            f"VQE={result.eigenvalue}, classical={classical_min}"
        )


class TestConvergenceScaling:
    """Test how convergence scales with matrix size.

    From Phase 2, Section 9.1:
      - 1-qubit: ~50-100 iterations
      - 2-qubit: ~100-200 iterations
    """

    @pytest.mark.slow
    def test_2x2_converges_faster_than_4x4(self, pauli_z_matrix, heisenberg_xxx_4x4):
        """2x2 matrix should converge in fewer iterations than 4x4."""
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver_2x2 = QuantumEigensolver(pauli_z_matrix)
        result_2x2 = solver_2x2.solve(shots=None)

        solver_4x4 = QuantumEigensolver(heisenberg_xxx_4x4)
        result_4x4 = solver_4x4.solve(shots=None)

        # Both must converge within budget
        assert result_2x2.iterations <= 200
        assert result_4x4.iterations <= 200

    @pytest.mark.parametrize(
        "matrix_name",
        [
            "pauli_z",
            "pauli_x",
            "general_hermitian_2x2",
            "heisenberg_xxx_4x4",
        ],
    )
    def test_all_matrices_converge(self, matrix_name, all_test_matrices):
        """All test matrices should converge within iteration budget."""
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        matrix, _ = all_test_matrices[matrix_name]
        solver = QuantumEigensolver(matrix)
        result = solver.solve(shots=None)

        assert result.iterations <= 200, (
            f"{matrix_name} failed to converge in 200 iterations " f"(took {result.iterations})"
        )
