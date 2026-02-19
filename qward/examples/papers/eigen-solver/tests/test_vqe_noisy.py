"""
Tests for VQE eigensolver under noisy simulation.

Specification:
  - phase1_problem_statement.md (success criteria: 5% relative error)
  - phase2_theoretical_design.md (Section 5.3: shot budget, Section 9.1: resources)
Author: test-engineer

These tests verify that the VQE-based quantum eigensolver converges to
correct eigenvalues under realistic noise models.

Success criteria (Phase 1, Section 4.1):
  - |lambda_VQE - lambda_exact| / spectral_range < 0.05 (5%)

Statistical requirements (Phase 1, Section 4.4):
  - 8192 shots minimum for noisy simulation
  - 10 independent trials per configuration
  - Results within threshold for >= 8/10 trials

All tests will FAIL until Phase 4 implementation.
"""

import numpy as np
import pytest


@pytest.mark.noisy
class TestVQENoisyGroundState2x2:
    """Test VQE under noise for 2x2 matrices."""

    @pytest.mark.parametrize(
        "noise_preset",
        [
            "IBM-HERON-R2",
            "RIGETTI-ANKAA3",
        ],
    )
    def test_vqe_noisy_pauli_z(self, pauli_z_matrix, noise_preset, noisy_tolerance):
        """M1: VQE converges within 5% for Pauli Z under realistic noise."""
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(pauli_z_matrix, noise_preset=noise_preset)
        result = solver.solve(shots=8192)

        classical_min = -1.0
        spectral_range = 2.0
        relative_error = abs(result.eigenvalue - classical_min) / spectral_range

        assert relative_error < noisy_tolerance, (
            f"Noisy VQE eigenvalue {result.eigenvalue} differs from "
            f"classical {classical_min} by {relative_error:.4f} of spectral "
            f"range (tolerance: {noisy_tolerance}) with noise: {noise_preset}"
        )

    @pytest.mark.parametrize(
        "noise_preset",
        [
            "IBM-HERON-R2",
            "RIGETTI-ANKAA3",
        ],
    )
    def test_vqe_noisy_pauli_x(self, pauli_x_matrix, noise_preset, noisy_tolerance):
        """M2: VQE converges within 5% for Pauli X under realistic noise."""
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(pauli_x_matrix, noise_preset=noise_preset)
        result = solver.solve(shots=8192)

        classical_min = -1.0
        spectral_range = 2.0
        relative_error = abs(result.eigenvalue - classical_min) / spectral_range

        assert relative_error < noisy_tolerance, (
            f"Noisy VQE error {relative_error:.4f} exceeds tolerance "
            f"for Pauli X with {noise_preset}"
        )

    @pytest.mark.parametrize(
        "noise_preset",
        [
            "IBM-HERON-R2",
            "RIGETTI-ANKAA3",
        ],
    )
    def test_vqe_noisy_general_hermitian(
        self, general_hermitian_2x2, noise_preset, noisy_tolerance
    ):
        """M3: VQE converges for general 2x2 Hermitian under noise.

        Ground state energy = 1 (corrected from Phase 1).
        """
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(general_hermitian_2x2, noise_preset=noise_preset)
        result = solver.solve(shots=8192)

        classical_min = 1.0  # Corrected eigenvalue
        spectral_range = 3.0
        relative_error = abs(result.eigenvalue - classical_min) / spectral_range

        assert relative_error < noisy_tolerance, (
            f"Noisy VQE error {relative_error:.4f} exceeds tolerance "
            f"for general 2x2 Hermitian with {noise_preset}"
        )


@pytest.mark.noisy
class TestVQENoisyGroundState4x4:
    """Test VQE under noise for 4x4 matrices (2 qubits)."""

    @pytest.mark.parametrize(
        "noise_preset",
        [
            "IBM-HERON-R2",
            "RIGETTI-ANKAA3",
        ],
    )
    @pytest.mark.slow
    def test_vqe_noisy_heisenberg(self, heisenberg_xxx_4x4, noise_preset, relaxed_noisy_tolerance):
        """M5: VQE converges for Heisenberg XXX under noise (relaxed tolerance).

        4x4 matrices require deeper circuits (depth 10-14 from Phase 2),
        so noise impact is greater. Use 10% tolerance.
        """
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(heisenberg_xxx_4x4, noise_preset=noise_preset)
        result = solver.solve(shots=8192)

        classical_min = -3.0
        spectral_range = 4.0
        relative_error = abs(result.eigenvalue - classical_min) / spectral_range

        assert relative_error < relaxed_noisy_tolerance, (
            f"Noisy VQE error {relative_error:.4f} exceeds relaxed tolerance "
            f"for Heisenberg with {noise_preset}"
        )


@pytest.mark.noisy
class TestVQENoisyVariationalPrinciple:
    """Verify variational principle behavior under noise.

    From Phase 2, Section 8.1: E(theta) >= E_0 for noiseless.
    Under noise, systematic shifts toward spectral mean are expected
    (Phase 1, Section 5.6).
    """

    @pytest.mark.parametrize(
        "noise_preset",
        [
            "IBM-HERON-R2",
            "RIGETTI-ANKAA3",
        ],
    )
    def test_variational_bound_with_noise(self, pauli_z_matrix, noise_preset):
        """Noisy VQE should not stray too far from the true ground state.

        Due to noise, the eigenvalue may be shifted upward (toward spectral mean),
        which is consistent with the variational principle. We allow a margin
        for statistical fluctuations.
        """
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        true_min = -1.0
        solver = QuantumEigensolver(pauli_z_matrix, noise_preset=noise_preset)
        result = solver.solve(shots=8192)

        # Allow noise margin below ground state
        noise_margin = 0.2
        assert result.eigenvalue >= true_min - noise_margin, (
            f"Noisy VQE result {result.eigenvalue} is too far below "
            f"the true minimum {true_min} for {noise_preset}"
        )


@pytest.mark.noisy
class TestVQENoisyShots:
    """Test impact of shot count on noisy VQE accuracy.

    From Phase 2, Section 5.3: SE(E) <= sqrt(sum c_i^2 / N_shots).
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("shots", [1024, 4096, 8192])
    def test_accuracy_bounded_with_shots(self, pauli_z_matrix, shots):
        """Error should be bounded by shot noise + systematic noise.

        From Phase 2 shot budget: M1 needs 2500 shots for SE < 0.02.
        """
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        solver = QuantumEigensolver(pauli_z_matrix, noise_preset="IBM-HERON-R2")
        result = solver.solve(shots=shots)

        classical_min = -1.0
        error = abs(result.eigenvalue - classical_min)

        # Error bound: sampling noise + systematic noise
        max_error = 2.0 / np.sqrt(shots) + 0.15
        assert error < max_error, (
            f"Error {error:.4f} exceeds expected bound {max_error:.4f} " f"at {shots} shots"
        )


@pytest.mark.noisy
class TestVQENoisyReproducibility:
    """Test reproducibility of noisy VQE results.

    From Phase 1, Section 4.4: "10 independent trials per configuration.
    Report mean +/- std. Results within threshold for >= 8/10 trials."
    """

    @pytest.mark.slow
    def test_noisy_vqe_consistency(self, pauli_z_matrix):
        """Multiple noisy VQE runs should produce consistent results.

        Run VQE 3 times and check that the standard deviation of results
        is bounded.
        """
        from eigen_solver.src.quantum_eigensolver import QuantumEigensolver

        results = []
        for _ in range(3):
            solver = QuantumEigensolver(pauli_z_matrix, noise_preset="IBM-HERON-R2")
            result = solver.solve(shots=4096)
            results.append(result.eigenvalue)

        std_dev = np.std(results)
        assert std_dev < 0.5, (
            f"Noisy VQE results too variable: std={std_dev:.4f}, " f"values={results}"
        )
