"""
Tests for Amplitude Encoding.

Specification: phase2_encoding_theory.md, Section 3
Author: test-engineer

Amplitude encoding maps a normalized vector to quantum amplitudes:
  |phi(x)> = sum_i x_tilde_i |i>
where x_tilde = x / ||x||_2

Expected behaviors:
  - Qubits = ceil(log2(d)) -- logarithmic scaling
  - Circuit depth O(d) via Mottonen decomposition
  - Kernel = cos^2(theta) where theta is the angle between vectors
  - L2 normalization required: two vectors x and c*x map to same state
  - Magnitude information is lost
  - Can produce entangled states for generic inputs

These tests should FAIL until Phase 4 implementation.
"""

import pytest
import numpy as np


class TestAmplitudeEncodingCircuit:
    """Tests for amplitude encoding circuit construction."""

    def test_qubit_count_logarithmic(self):
        """Amplitude encoding uses ceil(log2(d)) qubits."""
        from qml_data_encoding.encodings import AmplitudeEncoding

        test_cases = [
            (2, 1),  # 2 features -> 1 qubit
            (4, 2),  # 4 features -> 2 qubits
            (8, 3),  # 8 features -> 3 qubits
            (16, 4),  # 16 features -> 4 qubits
            (3, 2),  # 3 features -> 2 qubits (padded to 4)
            (5, 3),  # 5 features -> 3 qubits (padded to 8)
        ]
        for d, expected_qubits in test_cases:
            enc = AmplitudeEncoding(n_features=d)
            assert (
                enc.n_qubits == expected_qubits
            ), f"For d={d}, expected {expected_qubits} qubits, got {enc.n_qubits}"

    def test_auto_normalization(self, known_amplitude_vector):
        """Amplitude encoding should auto-normalize input to unit L2 norm."""
        from qml_data_encoding.encodings import AmplitudeEncoding

        x, x_normalized = known_amplitude_vector
        enc = AmplitudeEncoding(n_features=4)
        # Should accept un-normalized input and normalize internally
        circuit = enc.encode(x)
        assert circuit is not None

    def test_rejects_zero_vector(self, zero_vector_4d):
        """Encoding should reject the zero vector (cannot normalize)."""
        from qml_data_encoding.encodings import AmplitudeEncoding

        enc = AmplitudeEncoding(n_features=4)
        with pytest.raises((ValueError, ZeroDivisionError)):
            enc.encode(zero_vector_4d)

    def test_correct_state_uniform(self):
        """Encoding [1,1,1,1] should produce |+> state on 2 qubits.

        [1,1,1,1] / ||[1,1,1,1]|| = [0.5, 0.5, 0.5, 0.5]
        State = 0.5(|00> + |01> + |10> + |11>)
        """
        from qml_data_encoding.encodings import AmplitudeEncoding
        from qiskit_aer import AerSimulator

        enc = AmplitudeEncoding(n_features=4)
        x = np.array([1.0, 1.0, 1.0, 1.0])
        circuit = enc.encode(x)
        circuit.measure_all()

        backend = AerSimulator()
        result = backend.run(circuit, shots=10000).result()
        counts = result.get_counts()

        # Each basis state should have ~25% probability
        for state in ["00", "01", "10", "11"]:
            count = counts.get(state, 0)
            assert 2000 < count < 3000, f"State {state} has count {count}, expected ~2500"

    def test_correct_state_single_amplitude(self):
        """Encoding [1,0,0,0] should produce |00>."""
        from qml_data_encoding.encodings import AmplitudeEncoding
        from qiskit_aer import AerSimulator

        enc = AmplitudeEncoding(n_features=4)
        x = np.array([1.0, 0.0, 0.0, 0.0])
        circuit = enc.encode(x)
        circuit.measure_all()

        backend = AerSimulator()
        result = backend.run(circuit, shots=100).result()
        counts = result.get_counts()
        assert counts.get("00", 0) == 100

    def test_correct_statevector(self, known_amplitude_vector):
        """Verify statevector matches normalized input amplitudes."""
        from qml_data_encoding.encodings import AmplitudeEncoding
        from qiskit_aer import AerSimulator

        x, x_normalized = known_amplitude_vector
        enc = AmplitudeEncoding(n_features=4)
        circuit = enc.encode(x)
        circuit.save_statevector()

        backend = AerSimulator(method="statevector")
        result = backend.run(circuit).result()
        sv = result.get_statevector().data

        # Statevector amplitudes should match normalized input
        np.testing.assert_allclose(np.abs(sv), np.abs(x_normalized), atol=1e-10)

    def test_padding_non_power_of_2(self):
        """Features not a power of 2 should be zero-padded."""
        from qml_data_encoding.encodings import AmplitudeEncoding

        enc = AmplitudeEncoding(n_features=3)
        x = np.array([1.0, 2.0, 3.0])
        circuit = enc.encode(x)
        # 3 features padded to 4 -> 2 qubits
        assert circuit.num_qubits == 2


class TestAmplitudeEncodingKernel:
    """Tests for amplitude encoding kernel properties.

    From encoding theory Section 3.4:
    K(x, x') = (x . x' / (||x|| ||x'||))^2 = cos^2(theta)
    """

    def test_kernel_identical_vectors(self, known_amplitude_vector):
        """K(x, x) = 1 for any non-zero vector."""
        from qml_data_encoding.encodings import AmplitudeEncoding

        x, _ = known_amplitude_vector
        enc = AmplitudeEncoding(n_features=4)
        kernel = enc.kernel(x, x)
        assert np.isclose(kernel, 1.0, atol=1e-10)

    def test_kernel_orthogonal_vectors(self):
        """K(x, x') = 0 for orthogonal vectors."""
        from qml_data_encoding.encodings import AmplitudeEncoding

        enc = AmplitudeEncoding(n_features=4)
        x = np.array([1.0, 0.0, 0.0, 0.0])
        x_prime = np.array([0.0, 1.0, 0.0, 0.0])
        kernel = enc.kernel(x, x_prime)
        assert np.isclose(kernel, 0.0, atol=1e-10)

    def test_kernel_equals_squared_cosine_similarity(self):
        """Kernel should match cos^2(theta) where theta is angle between vectors."""
        from qml_data_encoding.encodings import AmplitudeEncoding

        enc = AmplitudeEncoding(n_features=4)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        x_prime = np.array([4.0, 3.0, 2.0, 1.0])

        # Compute expected kernel
        cos_sim = np.dot(x, x_prime) / (np.linalg.norm(x) * np.linalg.norm(x_prime))
        expected_kernel = cos_sim**2

        kernel = enc.kernel(x, x_prime)
        assert np.isclose(kernel, expected_kernel, atol=1e-10)

    def test_kernel_scale_invariance(self):
        """K(x, c*x) = 1 for any c > 0 (magnitude information lost).

        From Proposition 3: Two data points x and c*x map to the same state.
        """
        from qml_data_encoding.encodings import AmplitudeEncoding

        enc = AmplitudeEncoding(n_features=4)
        x = np.array([1.0, 2.0, 3.0, 4.0])

        for c in [0.1, 0.5, 2.0, 10.0, 100.0]:
            kernel = enc.kernel(x, c * x)
            assert np.isclose(
                kernel, 1.0, atol=1e-10
            ), f"K(x, {c}*x) = {kernel}, expected 1.0 (magnitude lost)"

    def test_kernel_matrix_psd(self, unit_norm_dataset, kernel_rtol):
        """Kernel matrix should be positive semi-definite."""
        from qml_data_encoding.encodings import AmplitudeEncoding

        X, _ = unit_norm_dataset
        enc = AmplitudeEncoding(n_features=4)
        K = enc.kernel_matrix(X[:10])

        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(
            eigenvalues >= -kernel_rtol
        ), f"Kernel matrix has negative eigenvalue: {min(eigenvalues)}"

    def test_kernel_diagonal_is_one(self, unit_norm_dataset):
        """Diagonal of kernel matrix should be all ones."""
        from qml_data_encoding.encodings import AmplitudeEncoding

        X, _ = unit_norm_dataset
        enc = AmplitudeEncoding(n_features=4)
        K = enc.kernel_matrix(X[:10])

        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)


class TestAmplitudeEncodingNormalization:
    """Tests related to amplitude encoding's normalization requirement.

    From encoding theory Section 3.4:
    Amplitude encoding requires L2-normalized input.
    Magnitude information is destroyed.
    """

    def test_magnitude_information_lost(self):
        """Vectors differing only in magnitude should produce same state."""
        from qml_data_encoding.encodings import AmplitudeEncoding
        from qiskit_aer import AerSimulator

        enc = AmplitudeEncoding(n_features=4)
        x = np.array([1.0, 2.0, 3.0, 4.0])

        circuit_x = enc.encode(x)
        circuit_x.save_statevector()
        circuit_2x = enc.encode(2.0 * x)
        circuit_2x.save_statevector()

        backend = AerSimulator(method="statevector")
        sv_x = backend.run(circuit_x).result().get_statevector().data
        sv_2x = backend.run(circuit_2x).result().get_statevector().data

        # States should be identical (up to global phase)
        fidelity = np.abs(np.dot(sv_x.conj(), sv_2x)) ** 2
        assert np.isclose(fidelity, 1.0, atol=1e-10)

    def test_directional_information_preserved(self):
        """Vectors with different directions should produce different states."""
        from qml_data_encoding.encodings import AmplitudeEncoding
        from qiskit_aer import AerSimulator

        enc = AmplitudeEncoding(n_features=4)
        x1 = np.array([1.0, 0.0, 0.0, 0.0])
        x2 = np.array([0.0, 0.0, 0.0, 1.0])

        circuit_1 = enc.encode(x1)
        circuit_1.save_statevector()
        circuit_2 = enc.encode(x2)
        circuit_2.save_statevector()

        backend = AerSimulator(method="statevector")
        sv_1 = backend.run(circuit_1).result().get_statevector().data
        sv_2 = backend.run(circuit_2).result().get_statevector().data

        fidelity = np.abs(np.dot(sv_1.conj(), sv_2)) ** 2
        assert fidelity < 0.01, "Different directions should produce different states"


class TestAmplitudeEncodingGateCounts:
    """Tests for amplitude encoding resource requirements.

    From encoding theory Section 3.3:
    For m qubits (N = 2^m features):
    - CNOT gates: 2^{m+1} - 2m - 2
    - Single-qubit rotations: 2^{m+1} - 2
    """

    @pytest.mark.parametrize(
        "d,expected_qubits",
        [
            (4, 2),
            (8, 3),
            (16, 4),
        ],
    )
    def test_qubit_scaling(self, d, expected_qubits):
        """Verify logarithmic qubit scaling."""
        from qml_data_encoding.encodings import AmplitudeEncoding

        enc = AmplitudeEncoding(n_features=d)
        assert enc.n_qubits == expected_qubits

    def test_has_two_qubit_gates(self):
        """Amplitude encoding (d > 2) requires two-qubit gates."""
        from qml_data_encoding.encodings import AmplitudeEncoding

        enc = AmplitudeEncoding(n_features=4)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        circuit = enc.encode(x)
        ops = circuit.count_ops()
        total_2q = ops.get("cx", 0) + ops.get("cz", 0)
        assert total_2q > 0, "Amplitude encoding for d=4 needs 2-qubit gates"
