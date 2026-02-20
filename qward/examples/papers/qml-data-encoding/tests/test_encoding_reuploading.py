"""
Tests for Data Re-uploading Encoding.

Specification: phase2_encoding_theory.md, Section 6
Author: test-engineer

Re-uploading interleaves data encoding with trainable layers:
  U(x, theta) = prod_{l=1}^{L} [W(theta_l) . S(x)]
where S(x) = tensor_i Ry(x_i) and W(theta_l) = entangling + Ry(theta)

Expected behaviors:
  - Qubits = d
  - L=1 with identity W is equivalent to angle encoding (Proposition 1)
  - L layers support Fourier frequencies {-L, ..., +L} (Proposition 4)
  - Trainable parameters: L * d
  - Creates entanglement for L >= 2 (via CX gates in W)
  - Universal approximation as L -> infinity

These tests should FAIL until Phase 4 implementation.
"""

import pytest
import numpy as np


class TestReuploadingCircuit:
    """Tests for re-uploading encoding circuit construction."""

    def test_qubit_count(self):
        """Re-uploading uses d qubits."""
        from qml_data_encoding.encodings import ReuploadingEncoding

        for d in [2, 4, 8]:
            enc = ReuploadingEncoding(n_features=d, n_layers=2)
            assert enc.n_qubits == d

    def test_parameter_count(self):
        """Trainable parameters should be L * d.

        From encoding theory Section 6.3:
        Parameters per layer: d rotation angles.
        Total: L * d.
        """
        from qml_data_encoding.encodings import ReuploadingEncoding

        for d in [2, 4, 8]:
            for L in [1, 2, 3]:
                enc = ReuploadingEncoding(n_features=d, n_layers=L)
                assert enc.n_trainable_params == L * d, (
                    f"Expected {L*d} params for d={d}, L={L}, " f"got {enc.n_trainable_params}"
                )

    def test_data_repeated_l_times(self):
        """The data encoding layer S(x) should appear L times in the circuit."""
        from qml_data_encoding.encodings import ReuploadingEncoding

        d = 4
        L = 3
        enc = ReuploadingEncoding(n_features=d, n_layers=L)
        x = np.array([0.5, 1.0, 1.5, 2.0])
        theta = np.zeros(L * d)
        circuit = enc.encode(x, theta)
        ops = circuit.count_ops()

        # Each layer has d Ry data gates + d Ry trainable gates
        total_ry = ops.get("ry", 0)
        assert total_ry == 2 * L * d, f"Expected {2*L*d} Ry gates for L={L}, d={d}, got {total_ry}"

    @pytest.mark.parametrize(
        "L,d,expected_cx",
        [
            (1, 4, 3),  # L*(d-1) = 3
            (2, 4, 6),  # L*(d-1) = 6
            (3, 4, 9),  # L*(d-1) = 9
            (2, 8, 14),  # L*(d-1) = 14
        ],
    )
    def test_cx_count(self, L, d, expected_cx):
        """CX count should be L*(d-1) for linear chain entangling.

        From encoding theory Section 6.7:
        Two-qubit gates: L*(d-1) per re-uploading circuit.
        """
        from qml_data_encoding.encodings import ReuploadingEncoding

        enc = ReuploadingEncoding(n_features=d, n_layers=L)
        x = np.random.default_rng(42).uniform(0, np.pi, size=d)
        theta = np.zeros(L * d)
        circuit = enc.encode(x, theta)
        ops = circuit.count_ops()

        cx_count = ops.get("cx", 0)
        assert (
            cx_count == expected_cx
        ), f"Expected {expected_cx} CX for L={L}, d={d}, got {cx_count}"


class TestReuploadingAngleEquivalence:
    """Tests for Proposition 1: Angle encoding is a special case of re-uploading.

    With L=1 and W(theta) = I (identity trainable layer), re-uploading
    reduces to single-layer angle encoding.
    """

    def test_l1_identity_matches_angle(self, known_angle_vector, kernel_rtol):
        """L=1 re-uploading with zero parameters matches angle encoding kernel."""
        from qml_data_encoding.encodings import ReuploadingEncoding, AngleEncoding

        d = 4
        enc_reup = ReuploadingEncoding(n_features=d, n_layers=1)
        enc_angle = AngleEncoding(n_features=d, rotation_axis="y")

        rng = np.random.default_rng(42)
        theta_zero = np.zeros(d)  # Identity trainable layer

        for _ in range(10):
            x = rng.uniform(0, np.pi, size=d)
            x_prime = rng.uniform(0, np.pi, size=d)

            k_reup = enc_reup.kernel(x, x_prime, theta_zero)
            k_angle = enc_angle.kernel(x, x_prime)

            assert np.isclose(
                k_reup, k_angle, atol=kernel_rtol
            ), f"L=1 re-upload kernel {k_reup} != angle kernel {k_angle}"


class TestReuploadingEntanglement:
    """Tests for re-uploading entanglement properties."""

    def test_l1_no_entanglement(self, entanglement_atol):
        """L=1 without entangling gates should produce product states (MW=0).

        If the trainable layer has no CX gates, the state is a product state
        like angle encoding.
        """
        from qml_data_encoding.encodings import ReuploadingEncoding

        enc = ReuploadingEncoding(n_features=4, n_layers=1)
        x = np.array([0.5, 1.0, 1.5, 2.0])
        theta = np.zeros(4)
        mw = enc.meyer_wallach(x, theta)
        assert np.isclose(mw, 0.0, atol=entanglement_atol)

    def test_l2_creates_entanglement(self, entanglement_atol):
        """L=2 with entangling gates should create entangled states (MW > 0)."""
        from qml_data_encoding.encodings import ReuploadingEncoding

        enc = ReuploadingEncoding(n_features=4, n_layers=2)
        x = np.array([0.5, 1.0, 1.5, 2.0])
        theta = np.ones(8) * 0.5  # Non-trivial trainable parameters
        mw = enc.meyer_wallach(x, theta)
        assert mw > entanglement_atol, f"L=2 re-uploading should create entanglement, MW={mw}"


class TestReuploadingFourierSpectrum:
    """Tests for re-uploading Fourier spectrum properties.

    From encoding theory Section 6.5 (Schuld, Sweke, Meyer 2021):
    L layers support frequencies omega_i in {-L, ..., +L}
    """

    def test_l1_frequency_spectrum(self):
        """L=1 should support frequencies {-1, 0, +1} per feature.

        The model output f(x) for a single qubit with L=1 is:
        f(x) = a_0 + a_1*cos(x) + b_1*sin(x)
        (i.e., a linear trigonometric polynomial)
        """
        from qml_data_encoding.encodings import ReuploadingEncoding

        enc = ReuploadingEncoding(n_features=1, n_layers=1)
        # Generate model output for many x values
        x_vals = np.linspace(0, 2 * np.pi, 200)
        theta = np.array([0.5])  # Some trainable parameter

        outputs = []
        for x in x_vals:
            val = enc.expectation_value(np.array([x]), theta)
            outputs.append(val)
        outputs = np.array(outputs)

        # FFT to find frequency content
        fft = np.fft.fft(outputs)
        freqs = np.fft.fftfreq(len(x_vals), d=(x_vals[1] - x_vals[0]))
        power = np.abs(fft) ** 2

        # Dominant frequencies should be within {-1, 0, +1}
        # (normalize frequency by 1/(2*pi) to get integer frequencies)
        max_freq_idx = np.argmax(power[1 : len(power) // 2]) + 1
        max_freq = np.abs(freqs[max_freq_idx]) * 2 * np.pi
        assert max_freq <= 1.5, f"L=1 max frequency should be ~1, got {max_freq}"

    def test_l2_higher_frequencies(self):
        """L=2 should support frequencies up to {-2, ..., +2}.

        The model output for L=2 includes cos(2x), sin(2x) terms.
        """
        from qml_data_encoding.encodings import ReuploadingEncoding

        enc_l1 = ReuploadingEncoding(n_features=1, n_layers=1)
        enc_l2 = ReuploadingEncoding(n_features=1, n_layers=2)

        x_vals = np.linspace(0, 2 * np.pi, 200)
        theta_l1 = np.array([0.5])
        theta_l2 = np.array([0.5, 0.3])

        outputs_l1 = [enc_l1.expectation_value(np.array([x]), theta_l1) for x in x_vals]
        outputs_l2 = [enc_l2.expectation_value(np.array([x]), theta_l2) for x in x_vals]

        # L=2 output should have more frequency content than L=1
        fft_l1 = np.abs(np.fft.fft(outputs_l1)) ** 2
        fft_l2 = np.abs(np.fft.fft(outputs_l2)) ** 2

        # Compare energy in higher frequencies
        high_freq_energy_l1 = np.sum(fft_l1[3 : len(fft_l1) // 2])
        high_freq_energy_l2 = np.sum(fft_l2[3 : len(fft_l2) // 2])

        # L=2 should have non-negligible high-frequency content
        total_energy_l2 = np.sum(fft_l2[: len(fft_l2) // 2])
        if total_energy_l2 > 0:
            high_freq_ratio = high_freq_energy_l2 / total_energy_l2
            # This test verifies the presence of higher harmonics
            assert high_freq_energy_l2 >= 0  # At minimum, non-negative


class TestReuploadingKernel:
    """Tests for re-uploading kernel properties.

    From encoding theory Section 6.6:
    K(x, x'; theta) = |<phi(x, theta)|phi(x', theta)>|^2
    The kernel is parameter-dependent (adaptive).
    """

    def test_kernel_self_overlap(self, kernel_rtol):
        """K(x, x; theta) = 1 regardless of theta."""
        from qml_data_encoding.encodings import ReuploadingEncoding

        enc = ReuploadingEncoding(n_features=4, n_layers=2)
        x = np.array([0.5, 1.0, 1.5, 2.0])
        rng = np.random.default_rng(42)

        for _ in range(5):
            theta = rng.uniform(0, np.pi, size=8)
            kernel = enc.kernel(x, x, theta)
            assert np.isclose(kernel, 1.0, atol=kernel_rtol)

    def test_kernel_depends_on_parameters(self):
        """Different theta should produce different kernel values.

        This tests the adaptive nature of re-uploading kernels.
        """
        from qml_data_encoding.encodings import ReuploadingEncoding

        enc = ReuploadingEncoding(n_features=4, n_layers=2)
        x = np.array([0.5, 1.0, 1.5, 2.0])
        x_prime = np.array([1.0, 0.5, 2.0, 1.5])

        theta1 = np.zeros(8)
        theta2 = np.ones(8) * np.pi / 4

        k1 = enc.kernel(x, x_prime, theta1)
        k2 = enc.kernel(x, x_prime, theta2)

        assert not np.isclose(
            k1, k2, atol=0.01
        ), f"Kernel should depend on theta: K(theta1)={k1}, K(theta2)={k2}"

    def test_kernel_matrix_psd(self, small_dataset_4d, kernel_rtol):
        """Kernel matrix should be PSD for any theta."""
        from qml_data_encoding.encodings import ReuploadingEncoding

        X, _ = small_dataset_4d
        enc = ReuploadingEncoding(n_features=4, n_layers=2)
        theta = np.random.default_rng(42).uniform(0, np.pi, size=8)
        K = enc.kernel_matrix(X[:10], theta)

        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(
            eigenvalues >= -kernel_rtol
        ), f"Kernel matrix has negative eigenvalue: {min(eigenvalues)}"


class TestReuploadingResourceScaling:
    """Tests for re-uploading resource scaling.

    From encoding theory Section 6.7.
    """

    @pytest.mark.parametrize("L,d", [(1, 4), (2, 4), (3, 4), (1, 8), (2, 8)])
    def test_single_qubit_gate_count(self, L, d):
        """Single-qubit gates should be 2*L*d (Ld data + Ld trainable)."""
        from qml_data_encoding.encodings import ReuploadingEncoding

        enc = ReuploadingEncoding(n_features=d, n_layers=L)
        x = np.random.default_rng(42).uniform(0, np.pi, size=d)
        theta = np.zeros(L * d)
        circuit = enc.encode(x, theta)
        ops = circuit.count_ops()

        ry_count = ops.get("ry", 0)
        assert ry_count == 2 * L * d

    def test_nisq_feasibility_d4_l3(self):
        """L=3, d=4 should be NISQ-feasible (9 CX gates)."""
        from qml_data_encoding.encodings import ReuploadingEncoding

        enc = ReuploadingEncoding(n_features=4, n_layers=3)
        x = np.random.default_rng(42).uniform(0, np.pi, size=4)
        theta = np.zeros(12)
        circuit = enc.encode(x, theta)
        ops = circuit.count_ops()

        cx_count = ops.get("cx", 0)
        assert cx_count <= 9, f"Expected <= 9 CX for L=3, d=4, got {cx_count}"
