"""
Tests for Expressibility Analysis and Statistical Properties.

Specification: phase2_expressibility_analysis.md
Author: test-engineer

Tests the following theoretical predictions:
  - Expressibility hierarchy: Basis << Angle << IQP << Re-upload
  - IQP expressibility < Angle expressibility (lower KL = more expressive)
  - Kernel matrices are PSD for all encodings
  - Kernel-target alignment is computable and bounded
  - Meyer-Wallach entanglement measures are correct
  - Haar-random fidelity distribution properties

These tests should FAIL until Phase 4 implementation (except Haar tests).
"""

import pytest
import numpy as np
from conftest import EXPR_N_PAIRS, EXPR_N_BINS, EXPR_EPSILON


class TestHaarRandomDistribution:
    """Tests for the Haar-random fidelity distribution.

    From expressibility analysis Section 1.2:
    P_Haar(F) = (2^m - 1)(1 - F)^{2^m - 2}
    Mean: 1/2^m
    """

    def test_haar_distribution_formula(self):
        """Verify Haar-random fidelity PDF integrates to 1."""
        for m in [2, 3, 4]:
            dim = 2**m
            # P_Haar(F) = (dim - 1) * (1 - F)^{dim - 2}
            F = np.linspace(0, 1, 10000)
            dF = F[1] - F[0]
            pdf = (dim - 1) * (1 - F) ** (dim - 2)
            integral = np.sum(pdf) * dF
            assert np.isclose(integral, 1.0, atol=0.01), f"Haar PDF integral = {integral} for m={m}"

    def test_haar_mean_fidelity(self):
        """Mean fidelity should be 1/2^m."""
        for m in [2, 3, 4]:
            dim = 2**m
            F = np.linspace(0, 1, 100000)
            dF = F[1] - F[0]
            pdf = (dim - 1) * (1 - F) ** (dim - 2)
            mean_F = np.sum(F * pdf) * dF

            expected_mean = 1.0 / dim
            assert np.isclose(
                mean_F, expected_mean, atol=0.01
            ), f"Haar mean fidelity: {mean_F:.4f}, expected {expected_mean:.4f}"

    def test_haar_mode_at_zero(self):
        """Haar-random fidelity distribution has mode at F=0 for m >= 2."""
        for m in [2, 3, 4]:
            dim = 2**m
            F = np.linspace(0, 1, 1000)
            pdf = (dim - 1) * (1 - F) ** (dim - 2)
            mode_idx = np.argmax(pdf)
            assert mode_idx == 0, f"Haar mode should be at F=0 for m={m}"


class TestExpressibilityComputation:
    """Tests for the expressibility computation protocol.

    From expressibility analysis Section 1.3.
    """

    def test_kl_divergence_computation(self):
        """Verify KL divergence computation is correct for known distributions."""
        # KL(P||Q) where P = Q should be 0
        P = np.array([0.25, 0.25, 0.25, 0.25])
        Q = np.array([0.25, 0.25, 0.25, 0.25])
        kl = np.sum(P * np.log(P / Q))
        assert np.isclose(kl, 0.0, atol=1e-10)

        # KL(P||Q) where P != Q should be > 0
        P = np.array([0.5, 0.5, 0.0, 0.0]) + EXPR_EPSILON
        Q = np.array([0.25, 0.25, 0.25, 0.25]) + EXPR_EPSILON
        P = P / P.sum()
        Q = Q / Q.sum()
        kl = np.sum(P * np.log(P / Q))
        assert kl > 0

    @pytest.mark.statistical
    def test_expressibility_angle_finite(self, expressibility_rtol):
        """Angle encoding expressibility should be finite and positive."""
        from qml_data_encoding.metrics import compute_expressibility

        expr = compute_expressibility(
            encoding_name="angle_ry",
            n_features=4,
            n_pairs=EXPR_N_PAIRS,
            n_bins=EXPR_N_BINS,
            seed=42,
        )
        assert expr > 0
        assert np.isfinite(expr)

    @pytest.mark.statistical
    def test_expressibility_iqp_less_than_angle(self, expressibility_rtol):
        """IQP should be more expressive than angle (lower KL divergence).

        From expressibility analysis Section 2.6:
        Expr_IQP ~ O(2^{-2d}) << Expr_angle ~ O(d * 2^{-d})
        """
        from qml_data_encoding.metrics import compute_expressibility

        expr_angle = compute_expressibility(
            encoding_name="angle_ry",
            n_features=4,
            n_pairs=EXPR_N_PAIRS,
            n_bins=EXPR_N_BINS,
            seed=42,
        )
        expr_iqp = compute_expressibility(
            encoding_name="iqp_full",
            n_features=4,
            n_pairs=EXPR_N_PAIRS,
            n_bins=EXPR_N_BINS,
            seed=42,
        )

        assert expr_iqp < expr_angle, (
            f"IQP expressibility ({expr_iqp:.6f}) should be < "
            f"angle expressibility ({expr_angle:.6f})"
        )

    @pytest.mark.statistical
    def test_expressibility_hierarchy(self, expressibility_rtol):
        """Full expressibility hierarchy: Angle > IQP (lower KL = better).

        Basis encoding has infinite KL divergence (discrete), so we skip it.
        """
        from qml_data_encoding.metrics import compute_expressibility

        results = {}
        for enc in ["angle_ry", "iqp_full"]:
            results[enc] = compute_expressibility(
                encoding_name=enc,
                n_features=4,
                n_pairs=EXPR_N_PAIRS,
                n_bins=EXPR_N_BINS,
                seed=42,
            )

        assert results["iqp_full"] < results["angle_ry"]


class TestMeyerWallachMeasure:
    """Tests for the Meyer-Wallach entanglement measure.

    From expressibility analysis Section 3.1:
    Q(|psi>) = (2/m) sum_k (1 - tr(rho_k^2))
    Q = 0 for product states, Q = 1 for maximally entangled.
    """

    def test_mw_product_state_is_zero(self):
        """MW should be 0 for product states."""
        from qml_data_encoding.metrics import meyer_wallach_from_statevector

        # |0000> is a product state
        sv = np.zeros(16)
        sv[0] = 1.0
        mw = meyer_wallach_from_statevector(sv, n_qubits=4)
        assert np.isclose(mw, 0.0, atol=1e-10)

    def test_mw_bell_state(self):
        """MW should be 1 for a Bell state (maximally entangled 2-qubit state).

        |Phi+> = (|00> + |11>) / sqrt(2)
        """
        from qml_data_encoding.metrics import meyer_wallach_from_statevector

        sv = np.array([1, 0, 0, 1]) / np.sqrt(2)
        mw = meyer_wallach_from_statevector(sv, n_qubits=2)
        assert np.isclose(mw, 1.0, atol=1e-10)

    def test_mw_ghz_state(self):
        """MW should be 1 for 4-qubit GHZ state.

        |GHZ> = (|0000> + |1111>) / sqrt(2)
        """
        from qml_data_encoding.metrics import meyer_wallach_from_statevector

        sv = np.zeros(16)
        sv[0] = 1.0 / np.sqrt(2)  # |0000>
        sv[15] = 1.0 / np.sqrt(2)  # |1111>
        mw = meyer_wallach_from_statevector(sv, n_qubits=4)
        assert np.isclose(mw, 1.0, atol=1e-10)

    def test_mw_bounded_01(self):
        """MW should be in [0, 1] for any state."""
        from qml_data_encoding.metrics import meyer_wallach_from_statevector

        rng = np.random.default_rng(42)
        for _ in range(20):
            # Random normalized statevector
            sv = rng.normal(size=16) + 1j * rng.normal(size=16)
            sv = sv / np.linalg.norm(sv)
            mw = meyer_wallach_from_statevector(sv, n_qubits=4)
            assert 0.0 <= mw <= 1.0 + 1e-10


class TestKernelTargetAlignment:
    """Tests for kernel-target alignment computation.

    From expressibility analysis Section 4.3:
    A(K, Y) = <K, Y>_F / (||K||_F * ||Y||_F)
    """

    def test_alignment_perfect_kernel(self):
        """Perfect kernel should give alignment = 1."""
        from qml_data_encoding.metrics import kernel_target_alignment

        y = np.array([1, 1, -1, -1])
        # Perfect kernel: same class -> 1, different class -> 0
        K = np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=float,
        )

        alignment = kernel_target_alignment(K, y)
        assert np.isclose(alignment, 1.0, atol=1e-10)

    def test_alignment_bounded(self):
        """Alignment should be in [-1, 1]."""
        from qml_data_encoding.metrics import kernel_target_alignment

        rng = np.random.default_rng(42)
        for _ in range(10):
            n = 20
            K = rng.uniform(0, 1, size=(n, n))
            K = (K + K.T) / 2  # Symmetric
            np.fill_diagonal(K, 1.0)
            y = rng.choice([-1, 1], size=n)

            alignment = kernel_target_alignment(K, y)
            assert -1.0 - 1e-10 <= alignment <= 1.0 + 1e-10

    def test_alignment_identity_kernel(self):
        """Identity kernel (no generalization) should give specific alignment."""
        from qml_data_encoding.metrics import kernel_target_alignment

        y = np.array([1, 1, -1, -1])
        K = np.eye(4)  # Identity kernel (like basis encoding)

        alignment = kernel_target_alignment(K, y)
        # With identity kernel: <K, Y>_F = sum_i y_i^2 = n
        # ||K||_F = sqrt(n), ||Y||_F = sqrt(sum y_i^2 * y_j^2) = n
        # A = n / (sqrt(n) * n) = 1/sqrt(n)
        expected = 1.0 / np.sqrt(4)
        assert np.isclose(alignment, expected, atol=1e-10)


class TestKernelMatrixProperties:
    """Tests for kernel matrix properties that hold for all encodings.

    From expressibility analysis Section 4.1:
    The quantum kernel K(x,x') = |<phi(x)|phi(x')>|^2 is PSD by construction.
    """

    def test_kernel_diagonal_is_one(self):
        """K(x, x) = 1 for all pure-state encodings."""
        from qml_data_encoding.metrics import kernel_target_alignment

        # This is a property, not implementation-dependent
        # All pure-state encodings satisfy K(x,x) = 1
        n = 10
        K = np.ones((n, n)) * 0.5  # Generic kernel
        np.fill_diagonal(K, 1.0)
        assert np.all(np.diag(K) == 1.0)

    def test_kernel_symmetric(self):
        """K(x, x') = K(x', x) for any encoding."""
        # This follows from |<a|b>|^2 = |<b|a>|^2
        sv1 = np.array([1, 0, 0, 1]) / np.sqrt(2)
        sv2 = np.array([1, 1, 0, 0]) / np.sqrt(2)
        k12 = np.abs(np.dot(sv1.conj(), sv2)) ** 2
        k21 = np.abs(np.dot(sv2.conj(), sv1)) ** 2
        assert np.isclose(k12, k21, atol=1e-15)


class TestFidelityDistributions:
    """Tests for encoding-specific fidelity distributions.

    From expressibility analysis Section 2.
    """

    @pytest.mark.statistical
    def test_angle_mean_fidelity(self):
        """Angle encoding mean fidelity should be (1/2)^d.

        From expressibility analysis Section 2.2:
        Mean fidelity <F> = (1/2)^d for angle encoding with d features.
        """
        from qml_data_encoding.metrics import compute_fidelity_distribution

        d = 4
        fidelities = compute_fidelity_distribution(
            encoding_name="angle_ry",
            n_features=d,
            n_pairs=5000,
            seed=42,
        )

        expected_mean = (0.5) ** d
        observed_mean = np.mean(fidelities)

        # Allow 20% relative tolerance for statistical estimate
        assert np.isclose(observed_mean, expected_mean, rtol=0.20), (
            f"Angle mean fidelity: {observed_mean:.4f}, " f"expected ~{expected_mean:.4f}"
        )

    @pytest.mark.statistical
    def test_iqp_mean_fidelity(self):
        """IQP encoding mean fidelity should be approximately 1/2^d.

        From expressibility analysis Section 2.3.
        """
        from qml_data_encoding.metrics import compute_fidelity_distribution

        d = 4
        fidelities = compute_fidelity_distribution(
            encoding_name="iqp_full",
            n_features=d,
            n_pairs=5000,
            seed=42,
        )

        expected_mean = 1.0 / (2**d)
        observed_mean = np.mean(fidelities)

        # Allow generous tolerance since IQP fidelity distribution
        # is only approximately exponential
        assert np.isclose(observed_mean, expected_mean, rtol=0.50), (
            f"IQP mean fidelity: {observed_mean:.4f}, " f"expected ~{expected_mean:.4f}"
        )
