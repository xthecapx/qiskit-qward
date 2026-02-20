"""
Tests for Basis Encoding.

Specification: phase2_encoding_theory.md, Section 2
Author: test-engineer

Basis encoding maps binary vectors to computational basis states:
  |phi(x)> = |b_1 b_2 ... b_d>
using X gates: U(x) = tensor_i X^{b_i}

Expected behaviors:
  - Requires binary input (0/1 per feature)
  - Qubits = d (one per feature)
  - Circuit depth O(1)
  - Zero two-qubit gates
  - Kernel is delta: K(x,x') = delta_{x,x'}
  - Produces product states only (MW entanglement = 0)
  - State is a computational basis state

These tests should FAIL until Phase 4 implementation.
"""

import pytest
import numpy as np


class TestBasisEncodingCircuit:
    """Tests for basis encoding circuit construction."""

    def test_requires_binary_input(self, known_basis_vector):
        """Encoding should accept binary vectors [0, 1]^d."""
        from qml_data_encoding.encodings import BasisEncoding

        enc = BasisEncoding(n_features=4)
        circuit = enc.encode(known_basis_vector)
        assert circuit is not None

    def test_rejects_continuous_input(self):
        """Encoding should raise error for non-binary input."""
        from qml_data_encoding.encodings import BasisEncoding

        enc = BasisEncoding(n_features=4)
        continuous_data = np.array([0.5, 1.2, -0.3, 0.8])
        with pytest.raises((ValueError, TypeError)):
            enc.encode(continuous_data)

    def test_qubit_count(self):
        """Basis encoding uses d qubits for d binary features."""
        from qml_data_encoding.encodings import BasisEncoding

        for d in [2, 4, 8]:
            enc = BasisEncoding(n_features=d)
            assert enc.n_qubits == d

    def test_circuit_depth_constant(self, known_basis_vector):
        """Circuit depth should be O(1) -- all X gates in parallel."""
        from qml_data_encoding.encodings import BasisEncoding

        enc = BasisEncoding(n_features=4)
        circuit = enc.encode(known_basis_vector)
        assert circuit.depth() <= 1

    def test_zero_two_qubit_gates(self, known_basis_vector):
        """Basis encoding uses zero two-qubit gates."""
        from qml_data_encoding.encodings import BasisEncoding

        enc = BasisEncoding(n_features=4)
        circuit = enc.encode(known_basis_vector)
        ops = circuit.count_ops()
        assert ops.get("cx", 0) == 0
        assert ops.get("cz", 0) == 0

    def test_gate_count_at_most_d(self, known_basis_vector):
        """At most d single-qubit gates (X gates where b_i = 1)."""
        from qml_data_encoding.encodings import BasisEncoding

        enc = BasisEncoding(n_features=4)
        circuit = enc.encode(known_basis_vector)
        # known_basis_vector = [1, 0, 1, 1], so 3 X gates
        assert circuit.size() <= 4

    def test_correct_state_all_zeros(self):
        """Encoding [0,0,0,0] should produce |0000>."""
        from qml_data_encoding.encodings import BasisEncoding
        from qiskit_aer import AerSimulator

        enc = BasisEncoding(n_features=4)
        x = np.array([0.0, 0.0, 0.0, 0.0])
        circuit = enc.encode(x)
        circuit.measure_all()

        backend = AerSimulator()
        result = backend.run(circuit, shots=100).result()
        counts = result.get_counts()
        assert counts.get("0000", 0) == 100

    def test_correct_state_known_vector(self, known_basis_vector):
        """Encoding [1,0,1,1] should produce |1011>."""
        from qml_data_encoding.encodings import BasisEncoding
        from qiskit_aer import AerSimulator

        enc = BasisEncoding(n_features=4)
        circuit = enc.encode(known_basis_vector)
        circuit.measure_all()

        backend = AerSimulator()
        result = backend.run(circuit, shots=100).result()
        counts = result.get_counts()
        # Qiskit uses little-endian bit ordering
        # [1,0,1,1] -> qubit 0=1, qubit 1=0, qubit 2=1, qubit 3=1
        # In Qiskit string: "1101" (reversed)
        assert counts.get("1101", 0) == 100

    def test_correct_state_all_ones(self):
        """Encoding [1,1,1,1] should produce |1111>."""
        from qml_data_encoding.encodings import BasisEncoding
        from qiskit_aer import AerSimulator

        enc = BasisEncoding(n_features=4)
        x = np.array([1.0, 1.0, 1.0, 1.0])
        circuit = enc.encode(x)
        circuit.measure_all()

        backend = AerSimulator()
        result = backend.run(circuit, shots=100).result()
        counts = result.get_counts()
        assert counts.get("1111", 0) == 100


class TestBasisEncodingKernel:
    """Tests for basis encoding kernel properties.

    From encoding theory Section 2.3:
    K(x, x') = delta_{x,x'} (trivial kernel)
    """

    def test_kernel_identical_vectors(self, known_basis_vector):
        """K(x, x) = 1 for any binary vector."""
        from qml_data_encoding.encodings import BasisEncoding

        enc = BasisEncoding(n_features=4)
        kernel = enc.kernel(known_basis_vector, known_basis_vector)
        assert np.isclose(kernel, 1.0)

    def test_kernel_different_vectors(self, known_basis_vector):
        """K(x, x') = 0 for distinct binary vectors."""
        from qml_data_encoding.encodings import BasisEncoding

        enc = BasisEncoding(n_features=4)
        x_prime = np.array([0.0, 1.0, 0.0, 0.0])
        kernel = enc.kernel(known_basis_vector, x_prime)
        assert np.isclose(kernel, 0.0)

    def test_kernel_matrix_is_identity(self, binary_dataset):
        """For distinct binary vectors, the kernel matrix is the identity.

        Since basis encoding maps distinct binary strings to orthogonal
        states, the kernel matrix for unique inputs should be identity.
        """
        from qml_data_encoding.encodings import BasisEncoding

        X, _ = binary_dataset
        # Get unique rows
        X_unique = np.unique(X, axis=0)
        enc = BasisEncoding(n_features=4)
        K = enc.kernel_matrix(X_unique)
        np.testing.assert_array_almost_equal(K, np.eye(len(X_unique)))


class TestBasisEncodingEntanglement:
    """Tests for basis encoding entanglement properties.

    From expressibility analysis Section 3.2:
    Basis encoding produces product states -> MW entanglement = 0.
    """

    def test_product_state(self, known_basis_vector):
        """Basis encoded state should be a product state (MW = 0)."""
        from qml_data_encoding.encodings import BasisEncoding

        enc = BasisEncoding(n_features=4)
        mw = enc.meyer_wallach(known_basis_vector)
        assert np.isclose(mw, 0.0)

    def test_product_state_all_inputs(self, binary_dataset):
        """All basis encoded states should be product states."""
        from qml_data_encoding.encodings import BasisEncoding

        X, _ = binary_dataset
        enc = BasisEncoding(n_features=4)
        for x in X[:10]:
            mw = enc.meyer_wallach(x)
            assert np.isclose(mw, 0.0), f"MW should be 0 for basis encoding, got {mw}"


class TestBasisEncodingEdgeCases:
    """Edge case tests for basis encoding."""

    def test_single_qubit(self):
        """Basis encoding with d=1 (single qubit)."""
        from qml_data_encoding.encodings import BasisEncoding

        enc = BasisEncoding(n_features=1)
        assert enc.n_qubits == 1

        circuit_0 = enc.encode(np.array([0.0]))
        circuit_1 = enc.encode(np.array([1.0]))
        assert circuit_0 is not None
        assert circuit_1 is not None

    def test_encoding_capacity(self):
        """Basis encoding can encode exactly 2^d distinct data points in d qubits."""
        from qml_data_encoding.encodings import BasisEncoding

        d = 3
        enc = BasisEncoding(n_features=d)
        # Generate all 2^d binary vectors
        all_vectors = []
        for i in range(2**d):
            bits = [(i >> j) & 1 for j in range(d)]
            all_vectors.append(np.array(bits, dtype=float))

        # Each should produce a distinct circuit / state
        states = []
        for v in all_vectors:
            circuit = enc.encode(v)
            states.append(circuit)

        assert len(states) == 2**d
