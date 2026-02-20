"""
Tests for Angle (Rotation) Encoding with Ry gates.

Specification: phase2_encoding_theory.md, Section 4
Author: test-engineer

Angle encoding maps features to rotation angles:
  |phi(x)> = tensor_i Ry(x_i)|0>

Expected behaviors:
  - Qubits = d (one per feature)
  - Circuit depth O(1)
  - Zero two-qubit gates
  - Product state (MW entanglement = 0)
  - Kernel: K(x, x') = prod_i cos^2((x_i - x_i') / 2)
  - Kernel is factorizable (product of per-feature terms)
  - Rz-only encoding is trivial: K = 1 for all pairs

These tests should FAIL until Phase 4 implementation.
"""

import pytest
import numpy as np


class TestAngleEncodingCircuit:
    """Tests for angle encoding circuit construction."""

    def test_qubit_count_linear(self):
        """Angle encoding uses d qubits for d features."""
        from qml_data_encoding.encodings import AngleEncoding

        for d in [2, 4, 8, 16]:
            enc = AngleEncoding(n_features=d, rotation_axis="y")
            assert enc.n_qubits == d

    def test_circuit_depth_constant(self, known_angle_vector):
        """Circuit depth should be O(1) -- all Ry gates in parallel."""
        from qml_data_encoding.encodings import AngleEncoding

        enc = AngleEncoding(n_features=4, rotation_axis="y")
        circuit = enc.encode(known_angle_vector)
        assert circuit.depth() <= 1

    def test_zero_two_qubit_gates(self, known_angle_vector):
        """Angle encoding uses zero two-qubit gates."""
        from qml_data_encoding.encodings import AngleEncoding

        enc = AngleEncoding(n_features=4, rotation_axis="y")
        circuit = enc.encode(known_angle_vector)
        ops = circuit.count_ops()
        assert ops.get("cx", 0) == 0
        assert ops.get("cz", 0) == 0

    def test_exactly_d_single_qubit_gates(self, known_angle_vector):
        """Angle encoding uses exactly d single-qubit Ry gates."""
        from qml_data_encoding.encodings import AngleEncoding

        enc = AngleEncoding(n_features=4, rotation_axis="y")
        circuit = enc.encode(known_angle_vector)
        assert circuit.size() == 4

    def test_correct_statevector(self, known_angle_vector):
        """Verify statevector matches theoretical prediction.

        For Ry encoding:
        |phi(x)> = tensor_i (cos(xi/2)|0> + sin(xi/2)|1>)
        """
        from qml_data_encoding.encodings import AngleEncoding
        from qiskit_aer import AerSimulator

        enc = AngleEncoding(n_features=4, rotation_axis="y")
        circuit = enc.encode(known_angle_vector)
        circuit.save_statevector()

        backend = AerSimulator(method="statevector")
        result = backend.run(circuit).result()
        sv = result.get_statevector().data

        # Compute expected state: tensor product of single-qubit states
        x = known_angle_vector
        single_states = [np.array([np.cos(xi / 2), np.sin(xi / 2)]) for xi in x]
        expected_sv = single_states[0]
        for s in single_states[1:]:
            expected_sv = np.kron(expected_sv, s)

        # Compare (up to global phase)
        fidelity = np.abs(np.dot(sv.conj(), expected_sv)) ** 2
        assert np.isclose(fidelity, 1.0, atol=1e-10)

    def test_correct_state_zero_input(self, zero_vector_4d):
        """Encoding [0,0,0,0] should produce |0000>."""
        from qml_data_encoding.encodings import AngleEncoding
        from qiskit_aer import AerSimulator

        enc = AngleEncoding(n_features=4, rotation_axis="y")
        circuit = enc.encode(zero_vector_4d)
        circuit.save_statevector()

        backend = AerSimulator(method="statevector")
        sv = backend.run(circuit).result().get_statevector().data
        expected = np.zeros(16)
        expected[0] = 1.0
        np.testing.assert_allclose(np.abs(sv), np.abs(expected), atol=1e-10)

    def test_correct_state_pi_input(self):
        """Encoding [pi, pi, pi, pi] should produce |1111>."""
        from qml_data_encoding.encodings import AngleEncoding
        from qiskit_aer import AerSimulator

        enc = AngleEncoding(n_features=4, rotation_axis="y")
        x = np.array([np.pi, np.pi, np.pi, np.pi])
        circuit = enc.encode(x)
        circuit.save_statevector()

        backend = AerSimulator(method="statevector")
        sv = backend.run(circuit).result().get_statevector().data
        # |1111> is the last basis state (index 15)
        assert np.isclose(np.abs(sv[-1]) ** 2, 1.0, atol=1e-10)


class TestAngleEncodingKernel:
    """Tests for angle encoding kernel properties.

    From encoding theory Section 4.5 (Theorem):
    K(x, x') = prod_i cos^2((x_i - x_i') / 2)
    """

    def test_kernel_identical_vectors(self, identical_vectors_4d, kernel_rtol):
        """K(x, x) = 1 for any vector (self-overlap is perfect)."""
        from qml_data_encoding.encodings import AngleEncoding

        x, x_copy = identical_vectors_4d
        enc = AngleEncoding(n_features=4, rotation_axis="y")
        kernel = enc.kernel(x, x_copy)
        assert np.isclose(kernel, 1.0, atol=kernel_rtol)

    def test_kernel_known_pair(self, known_angle_vector_pair, kernel_rtol):
        """Verify kernel value for known vector pair."""
        from qml_data_encoding.encodings import AngleEncoding

        x, x_prime, expected_kernel = known_angle_vector_pair
        enc = AngleEncoding(n_features=4, rotation_axis="y")
        kernel = enc.kernel(x, x_prime)
        assert np.isclose(kernel, expected_kernel, atol=kernel_rtol)

    def test_kernel_is_factorizable(self, kernel_rtol):
        """Kernel should equal product of per-feature cosine terms.

        K(x, x') = prod_i cos^2((x_i - x_i') / 2)
        """
        from qml_data_encoding.encodings import AngleEncoding

        enc = AngleEncoding(n_features=4, rotation_axis="y")
        rng = np.random.default_rng(42)

        for _ in range(10):
            x = rng.uniform(0, np.pi, size=4)
            x_prime = rng.uniform(0, np.pi, size=4)

            kernel = enc.kernel(x, x_prime)
            expected = np.prod(np.cos((x - x_prime) / 2) ** 2)
            assert np.isclose(
                kernel, expected, atol=kernel_rtol
            ), f"Kernel {kernel} != expected {expected}"

    def test_kernel_symmetric(self, kernel_rtol):
        """K(x, x') = K(x', x)."""
        from qml_data_encoding.encodings import AngleEncoding

        enc = AngleEncoding(n_features=4, rotation_axis="y")
        rng = np.random.default_rng(42)

        for _ in range(10):
            x = rng.uniform(0, np.pi, size=4)
            x_prime = rng.uniform(0, np.pi, size=4)
            assert np.isclose(
                enc.kernel(x, x_prime),
                enc.kernel(x_prime, x),
                atol=kernel_rtol,
            )

    def test_kernel_bounded_01(self, kernel_rtol):
        """Kernel values should be in [0, 1]."""
        from qml_data_encoding.encodings import AngleEncoding

        enc = AngleEncoding(n_features=4, rotation_axis="y")
        rng = np.random.default_rng(42)

        for _ in range(50):
            x = rng.uniform(0, 2 * np.pi, size=4)
            x_prime = rng.uniform(0, 2 * np.pi, size=4)
            kernel = enc.kernel(x, x_prime)
            assert 0.0 - kernel_rtol <= kernel <= 1.0 + kernel_rtol

    def test_kernel_matrix_psd(self, small_dataset_4d, kernel_rtol):
        """Kernel matrix should be positive semi-definite."""
        from qml_data_encoding.encodings import AngleEncoding

        X, _ = small_dataset_4d
        enc = AngleEncoding(n_features=4, rotation_axis="y")
        K = enc.kernel_matrix(X[:15])

        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(
            eigenvalues >= -kernel_rtol
        ), f"Kernel matrix has negative eigenvalue: {min(eigenvalues)}"


class TestAngleEncodingEntanglement:
    """Tests for angle encoding entanglement properties.

    From expressibility analysis Section 3.2:
    Angle encoding produces product states -> MW = 0.
    """

    def test_product_state(self, known_angle_vector, entanglement_atol):
        """Angle encoded state should be a product state (MW = 0)."""
        from qml_data_encoding.encodings import AngleEncoding

        enc = AngleEncoding(n_features=4, rotation_axis="y")
        mw = enc.meyer_wallach(known_angle_vector)
        assert np.isclose(mw, 0.0, atol=entanglement_atol)

    def test_product_state_random_inputs(self, entanglement_atol):
        """MW = 0 for all random inputs (product structure is data-independent)."""
        from qml_data_encoding.encodings import AngleEncoding

        enc = AngleEncoding(n_features=4, rotation_axis="y")
        rng = np.random.default_rng(42)

        for _ in range(20):
            x = rng.uniform(0, np.pi, size=4)
            mw = enc.meyer_wallach(x)
            assert np.isclose(mw, 0.0, atol=entanglement_atol)


class TestAngleEncodingRzTrivial:
    """Tests that Rz-only angle encoding produces a trivial kernel.

    From encoding theory Section 4.6:
    K_Rz(x, x') = |prod_i e^{-i(xi'-xi)/2}|^2 = 1
    Rz encoding is USELESS for classification without Hadamard gates.
    """

    def test_rz_kernel_always_one(self, kernel_rtol):
        """Rz-only kernel should be identically 1 for all pairs."""
        from qml_data_encoding.encodings import AngleEncoding

        enc = AngleEncoding(n_features=4, rotation_axis="z")
        rng = np.random.default_rng(42)

        for _ in range(20):
            x = rng.uniform(0, 2 * np.pi, size=4)
            x_prime = rng.uniform(0, 2 * np.pi, size=4)
            kernel = enc.kernel(x, x_prime)
            assert np.isclose(
                kernel, 1.0, atol=kernel_rtol
            ), f"Rz kernel should be 1.0, got {kernel}"

    def test_rz_kernel_useless_for_classification(self, small_dataset_4d, kernel_rtol):
        """Rz kernel matrix should be all ones -- no class separation possible."""
        from qml_data_encoding.encodings import AngleEncoding

        X, _ = small_dataset_4d
        enc = AngleEncoding(n_features=4, rotation_axis="z")
        K = enc.kernel_matrix(X[:10])

        expected = np.ones((10, 10))
        np.testing.assert_allclose(K, expected, atol=kernel_rtol)


class TestAngleEncodingRxEquivalence:
    """Tests that Rx encoding has equivalent kernel to Ry.

    From encoding theory Section 4.6:
    K_Rx(x, x') = prod_i cos^2((x_i - x_i') / 2) (same as Ry)
    """

    def test_rx_kernel_equals_ry_kernel(self, kernel_rtol):
        """Rx and Ry kernels should produce identical values."""
        from qml_data_encoding.encodings import AngleEncoding

        enc_ry = AngleEncoding(n_features=4, rotation_axis="y")
        enc_rx = AngleEncoding(n_features=4, rotation_axis="x")
        rng = np.random.default_rng(42)

        for _ in range(20):
            x = rng.uniform(0, np.pi, size=4)
            x_prime = rng.uniform(0, np.pi, size=4)
            k_ry = enc_ry.kernel(x, x_prime)
            k_rx = enc_rx.kernel(x, x_prime)
            assert np.isclose(k_ry, k_rx, atol=kernel_rtol)


class TestAngleEncodingPeriodicity:
    """Tests for angle encoding periodicity behavior.

    From encoding theory Section 4.4:
    Rotations are 2*pi-periodic in statevector but measurement
    statistics have effective period of pi for distinguishing states.
    """

    def test_handles_negative_angles(self):
        """Encoding should handle negative input gracefully."""
        from qml_data_encoding.encodings import AngleEncoding

        enc = AngleEncoding(n_features=4, rotation_axis="y")
        x = np.array([-0.5, -1.0, -np.pi, -2 * np.pi])
        circuit = enc.encode(x)
        assert circuit is not None

    def test_handles_large_angles(self):
        """Encoding should handle angles > 2*pi gracefully."""
        from qml_data_encoding.encodings import AngleEncoding

        enc = AngleEncoding(n_features=4, rotation_axis="y")
        x = np.array([3 * np.pi, 4 * np.pi, 10.0, 100.0])
        circuit = enc.encode(x)
        assert circuit is not None

    def test_kernel_periodic(self, kernel_rtol):
        """Kernel should be 2*pi-periodic: K(x, x') = K(x + 2pi, x')."""
        from qml_data_encoding.encodings import AngleEncoding

        enc = AngleEncoding(n_features=4, rotation_axis="y")
        rng = np.random.default_rng(42)

        for _ in range(10):
            x = rng.uniform(0, np.pi, size=4)
            x_prime = rng.uniform(0, np.pi, size=4)
            k1 = enc.kernel(x, x_prime)
            k2 = enc.kernel(x + 2 * np.pi, x_prime)
            assert np.isclose(k1, k2, atol=kernel_rtol)
