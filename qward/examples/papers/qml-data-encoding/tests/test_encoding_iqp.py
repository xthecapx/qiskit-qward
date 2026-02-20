"""
Tests for IQP (Instantaneous Quantum Polynomial) Encoding.

Specification: phase2_encoding_theory.md, Section 5
Author: test-engineer

IQP encoding creates entangled states through pairwise ZZ interactions:
  U_IQP(x) = H^d * D(x) * H^d
  D(x) = exp(i sum_i x_i Z_i + i sum_{i<j} x_i*x_j Z_i Z_j)

Expected behaviors:
  - Qubits = d (one per feature)
  - Creates entanglement (MW > 0 for generic inputs)
  - Non-factorizable kernel (captures feature interactions)
  - Gate count: 2d Hadamards + d Rz + d(d-1)/2 RZZ gates
  - IQP is strictly more expressive than Angle encoding (Theorem, Section 8)
  - Kernel contains cross-terms x_i * x_j

These tests should FAIL until Phase 4 implementation.
"""

import pytest
import numpy as np


class TestIQPEncodingCircuit:
    """Tests for IQP encoding circuit construction."""

    def test_qubit_count_linear(self):
        """IQP encoding uses d qubits for d features."""
        from qml_data_encoding.encodings import IQPEncoding

        for d in [2, 4, 8]:
            enc = IQPEncoding(n_features=d)
            assert enc.n_qubits == d

    def test_has_hadamard_layers(self, known_iqp_vector_2d):
        """Circuit should have two layers of Hadamard gates."""
        from qml_data_encoding.encodings import IQPEncoding

        enc = IQPEncoding(n_features=2)
        circuit = enc.encode(known_iqp_vector_2d)
        ops = circuit.count_ops()
        assert ops.get("h", 0) == 4  # 2 H per qubit, 2 qubits

    def test_has_rz_gates(self, known_iqp_vector_2d):
        """Circuit should have d Rz gates for single-qubit diagonal."""
        from qml_data_encoding.encodings import IQPEncoding

        enc = IQPEncoding(n_features=2)
        circuit = enc.encode(known_iqp_vector_2d)
        ops = circuit.count_ops()
        assert ops.get("rz", 0) >= 2  # At least d Rz gates

    def test_has_rzz_gates(self, known_iqp_vector_2d):
        """Circuit should have d(d-1)/2 RZZ interactions."""
        from qml_data_encoding.encodings import IQPEncoding

        enc = IQPEncoding(n_features=2)
        circuit = enc.encode(known_iqp_vector_2d)
        ops = circuit.count_ops()
        # d=2: 1 RZZ gate (or its decomposition into 2 CX + 1 Rz)
        rzz_count = ops.get("rzz", 0)
        cx_count = ops.get("cx", 0)
        # Either native RZZ or decomposed (2 CX per RZZ)
        assert rzz_count >= 1 or cx_count >= 2

    @pytest.mark.parametrize(
        "d,expected_rzz",
        [
            (2, 1),
            (3, 3),
            (4, 6),
            (5, 10),
        ],
    )
    def test_rzz_count_quadratic(self, d, expected_rzz):
        """Number of RZZ interactions should be d*(d-1)/2."""
        from qml_data_encoding.encodings import IQPEncoding

        enc = IQPEncoding(n_features=d)
        x = np.random.default_rng(42).uniform(0, np.pi, size=d)
        circuit = enc.encode(x)
        ops = circuit.count_ops()

        # Count pairwise interactions (RZZ or CX pairs)
        rzz_count = ops.get("rzz", 0)
        if rzz_count > 0:
            assert rzz_count == expected_rzz
        else:
            # If decomposed: 2 CX per RZZ
            cx_count = ops.get("cx", 0)
            assert cx_count == 2 * expected_rzz

    def test_gate_counts_d4(self):
        """Verify gate counts for d=4 match Phase 2 resource table.

        From encoding_theory.md Section 7.1:
        IQP (full) d=4: 12 single-qubit gates, 12 two-qubit gates, 24 total
        Note: Hadamards counted as single-qubit gates.
        """
        from qml_data_encoding.encodings import IQPEncoding

        enc = IQPEncoding(n_features=4)
        x = np.array([0.5, 1.0, 1.5, 2.0])
        circuit = enc.encode(x)

        # The theoretical count is:
        # 2*4=8 H + 4 Rz = 12 single-qubit gates
        # 6 RZZ = 12 CX (two-qubit) gates
        # Total = 24
        # Allow some tolerance for different decompositions
        total = circuit.size()
        assert total >= 20, f"Expected ~24 gates for d=4 IQP, got {total}"


class TestIQPEncodingEntanglement:
    """Tests for IQP encoding entanglement creation.

    From expressibility analysis Section 3.2 and Proposition 2:
    IQP encoding produces entangled states for generic inputs.
    """

    def test_creates_entanglement_generic_input(self, entanglement_atol):
        """IQP should create entangled states for generic (non-zero) inputs."""
        from qml_data_encoding.encodings import IQPEncoding

        enc = IQPEncoding(n_features=2)
        x = np.array([1.0, 1.0])  # Generic input with x1*x2 != k*pi
        mw = enc.meyer_wallach(x)
        assert mw > entanglement_atol, f"IQP should produce entangled state for x=[1,1], MW={mw}"

    def test_creates_entanglement_4_qubits(self, entanglement_atol):
        """IQP with d=4 should produce entangled states."""
        from qml_data_encoding.encodings import IQPEncoding

        enc = IQPEncoding(n_features=4)
        x = np.array([0.5, 1.0, 1.5, 2.0])
        mw = enc.meyer_wallach(x)
        assert mw > entanglement_atol

    def test_entanglement_data_dependent(self):
        """Entanglement amount should depend on the data values.

        When x_i * x_j = k*pi for all pairs, entanglement is minimal.
        """
        from qml_data_encoding.encodings import IQPEncoding

        enc = IQPEncoding(n_features=2)
        # Two different inputs should generally produce different MW
        mw1 = enc.meyer_wallach(np.array([0.3, 0.7]))
        mw2 = enc.meyer_wallach(np.array([1.0, 2.0]))
        # They should not be identical in general
        assert not np.isclose(mw1, mw2, atol=1e-6)

    def test_entanglement_strictly_greater_than_angle(self, entanglement_atol):
        """IQP MW should be > 0 while Angle MW = 0 (Proposition 2 proof).

        This directly tests the theorem that IQP is strictly more expressive.
        """
        from qml_data_encoding.encodings import IQPEncoding, AngleEncoding

        x = np.array([1.0, 1.0, 0.5, 1.5])
        enc_iqp = IQPEncoding(n_features=4)
        enc_angle = AngleEncoding(n_features=4, rotation_axis="y")

        mw_iqp = enc_iqp.meyer_wallach(x)
        mw_angle = enc_angle.meyer_wallach(x)

        assert np.isclose(mw_angle, 0.0, atol=entanglement_atol)
        assert mw_iqp > entanglement_atol
        assert mw_iqp > mw_angle


class TestIQPEncodingKernel:
    """Tests for IQP encoding kernel properties.

    From encoding theory Section 5.5:
    K(x,x') = |1/2^d sum_s exp(i(f(x,s) - f(x',s)))|^2
    """

    def test_kernel_identical_vectors(self, kernel_rtol):
        """K(x, x) = 1 for any vector."""
        from qml_data_encoding.encodings import IQPEncoding

        enc = IQPEncoding(n_features=4)
        x = np.array([0.5, 1.0, 1.5, 2.0])
        kernel = enc.kernel(x, x)
        assert np.isclose(kernel, 1.0, atol=kernel_rtol)

    def test_kernel_bounded_01(self, kernel_rtol):
        """Kernel values should be in [0, 1]."""
        from qml_data_encoding.encodings import IQPEncoding

        enc = IQPEncoding(n_features=4)
        rng = np.random.default_rng(42)

        for _ in range(20):
            x = rng.uniform(0, np.pi, size=4)
            x_prime = rng.uniform(0, np.pi, size=4)
            kernel = enc.kernel(x, x_prime)
            assert -kernel_rtol <= kernel <= 1.0 + kernel_rtol

    def test_kernel_symmetric(self, kernel_rtol):
        """K(x, x') = K(x', x)."""
        from qml_data_encoding.encodings import IQPEncoding

        enc = IQPEncoding(n_features=4)
        rng = np.random.default_rng(42)

        for _ in range(10):
            x = rng.uniform(0, np.pi, size=4)
            x_prime = rng.uniform(0, np.pi, size=4)
            assert np.isclose(
                enc.kernel(x, x_prime),
                enc.kernel(x_prime, x),
                atol=kernel_rtol,
            )

    def test_kernel_not_factorizable(self, kernel_rtol):
        """IQP kernel should NOT equal the product of per-feature terms.

        This is the key distinction from angle encoding (Corollary, Section 8).
        """
        from qml_data_encoding.encodings import IQPEncoding

        enc = IQPEncoding(n_features=4)
        rng = np.random.default_rng(42)

        factorizable_count = 0
        n_tests = 20
        for _ in range(n_tests):
            x = rng.uniform(0.5, 2.5, size=4)
            x_prime = rng.uniform(0.5, 2.5, size=4)

            kernel = enc.kernel(x, x_prime)
            # If it were factorizable like angle encoding
            factorized = np.prod(np.cos((x - x_prime) / 2) ** 2)

            if np.isclose(kernel, factorized, atol=0.01):
                factorizable_count += 1

        # IQP kernel should differ from the factorized form in most cases
        assert (
            factorizable_count < n_tests // 2
        ), f"IQP kernel matched factorized form {factorizable_count}/{n_tests} times"

    def test_kernel_matrix_psd(self, small_dataset_4d, kernel_rtol):
        """Kernel matrix should be positive semi-definite."""
        from qml_data_encoding.encodings import IQPEncoding

        X, _ = small_dataset_4d
        enc = IQPEncoding(n_features=4)
        K = enc.kernel_matrix(X[:10])

        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(
            eigenvalues >= -kernel_rtol
        ), f"Kernel matrix has negative eigenvalue: {min(eigenvalues)}"


class TestIQPEncodingState:
    """Tests for IQP encoding state correctness."""

    def test_state_not_product_for_generic_input(self):
        """IQP state should not be a product state for x1*x2 != 0.

        Direct verification via Schmidt decomposition or reduced density matrix.
        """
        from qml_data_encoding.encodings import IQPEncoding
        from qiskit_aer import AerSimulator

        enc = IQPEncoding(n_features=2)
        x = np.array([1.0, 1.0])
        circuit = enc.encode(x)
        circuit.save_statevector()

        backend = AerSimulator(method="statevector")
        sv = backend.run(circuit).result().get_statevector().data

        # Reshape into 2x2 matrix for 2-qubit state
        psi_matrix = sv.reshape(2, 2)

        # Compute SVD; if product state, only 1 singular value is non-zero
        svd_vals = np.linalg.svd(psi_matrix, compute_uv=False)
        # For entangled state, at least 2 non-zero singular values
        n_significant = np.sum(svd_vals > 1e-10)
        assert (
            n_significant >= 2
        ), f"Expected entangled state (>=2 Schmidt coeffs), got {n_significant}"

    def test_correct_action_on_plus_state(self, known_iqp_vector_2d):
        """Verify IQP circuit: H -> D(x) -> H acting on |0>.

        Step 1: H|0> = |+>
        Step 2: D(x)|+> applies phases
        Step 3: Final H creates interference
        """
        from qml_data_encoding.encodings import IQPEncoding
        from qiskit_aer import AerSimulator

        enc = IQPEncoding(n_features=2)
        circuit = enc.encode(known_iqp_vector_2d)
        circuit.save_statevector()

        backend = AerSimulator(method="statevector")
        sv = backend.run(circuit).result().get_statevector().data

        # State should be a valid normalized vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)
        # State should have non-trivial superposition (not just one basis state)
        probs = np.abs(sv) ** 2
        nonzero_probs = probs[probs > 1e-10]
        assert len(nonzero_probs) > 1, "IQP should create superposition"


class TestIQPEncodingVariants:
    """Tests for IQP encoding variants (nearest-neighbor, k-local).

    From encoding theory Section 5.6.
    """

    def test_nn_iqp_fewer_gates(self):
        """Nearest-neighbor IQP should have d-1 RZZ gates (not d(d-1)/2)."""
        from qml_data_encoding.encodings import IQPEncoding

        d = 4
        enc_full = IQPEncoding(n_features=d, interaction="full")
        enc_nn = IQPEncoding(n_features=d, interaction="nearest_neighbor")

        x = np.array([0.5, 1.0, 1.5, 2.0])
        circuit_full = enc_full.encode(x)
        circuit_nn = enc_nn.encode(x)

        # NN should have fewer gates
        assert circuit_nn.size() < circuit_full.size()

    def test_nn_iqp_still_entangles(self, entanglement_atol):
        """NN-IQP should still create entanglement."""
        from qml_data_encoding.encodings import IQPEncoding

        enc = IQPEncoding(n_features=4, interaction="nearest_neighbor")
        x = np.array([0.5, 1.0, 1.5, 2.0])
        mw = enc.meyer_wallach(x)
        assert mw > entanglement_atol


class TestIQPEncodingNISQFeasibility:
    """Tests for NISQ feasibility constraints.

    From encoding_theory.md Section 5.7 and experimental_design.md Section 3.2:
    IQP full with d > 8 is infeasible on Heron (>56 CX gates).
    """

    @pytest.mark.parametrize(
        "d,expected_cx",
        [
            (2, 2),  # d(d-1) = 2
            (4, 12),  # d(d-1) = 12
            (6, 30),  # d(d-1) = 30
        ],
    )
    def test_cx_count_scaling(self, d, expected_cx):
        """CNOT count should be d*(d-1) for full IQP."""
        from qml_data_encoding.encodings import IQPEncoding

        enc = IQPEncoding(n_features=d, interaction="full")
        x = np.random.default_rng(42).uniform(0, np.pi, size=d)
        circuit = enc.encode(x)
        ops = circuit.count_ops()

        # RZZ decomposes into 2 CX each
        rzz_count = ops.get("rzz", 0)
        cx_count = ops.get("cx", 0)
        total_cx = cx_count + 2 * rzz_count
        assert (
            total_cx == expected_cx
        ), f"For d={d}, expected {expected_cx} CX equivalent, got {total_cx}"
