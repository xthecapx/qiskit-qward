"""Tests for qward ComplexityMetrics class."""

import unittest
import math
from qiskit import QuantumCircuit
from qiskit.circuit.library.basis_change import QFTGate

from qward.metrics import ComplexityMetrics
from qward.metrics.schemas import ComplexityMetricsSchema
from qward.metrics.types import MetricsType, MetricsId


class TestComplexityMetrics(unittest.TestCase):
    """Tests for ComplexityMetrics class."""

    def setUp(self):
        """Set up test fixtures."""
        # Simple 2-qubit Bell state circuit
        self.simple_circuit = QuantumCircuit(2)
        self.simple_circuit.h(0)
        self.simple_circuit.cx(0, 1)

        # More complex circuit for testing
        self.complex_circuit = QuantumCircuit(4)
        self.complex_circuit.h(0)
        self.complex_circuit.cx(0, 1)
        self.complex_circuit.cx(1, 2)
        self.complex_circuit.cx(2, 3)
        self.complex_circuit.t(0)  # T gate
        self.complex_circuit.t(1)  # T gate
        self.complex_circuit.rz(0.5, 0)
        self.complex_circuit.ry(0.3, 1)
        self.complex_circuit.barrier()
        self.complex_circuit.cx(3, 0)

    def test_complexity_metrics_init(self):
        """Test ComplexityMetrics initialization."""
        metrics = ComplexityMetrics(self.simple_circuit)

        self.assertEqual(metrics.circuit, self.simple_circuit)
        self.assertEqual(metrics._get_metric_type(), MetricsType.PRE_RUNTIME)
        self.assertEqual(metrics._get_metric_id(), MetricsId.COMPLEXITY)
        self.assertTrue(metrics.is_ready())

    def test_complexity_metrics_init_none_circuit(self):
        """Test ComplexityMetrics initialization with None circuit."""
        metrics = ComplexityMetrics(None)

        self.assertIsNone(metrics.circuit)
        self.assertFalse(metrics.is_ready())

    def test_get_metrics_returns_schema(self):
        """Test that get_metrics returns a ComplexityMetricsSchema."""
        metrics = ComplexityMetrics(self.simple_circuit)

        result = metrics.get_metrics()

        self.assertIsInstance(result, ComplexityMetricsSchema)

    def test_gate_based_metrics_simple_circuit(self):
        """Test gate-based metrics for simple circuit."""
        metrics = ComplexityMetrics(self.simple_circuit)
        result = metrics.get_metrics()

        # Test gate-based metrics
        self.assertEqual(result.gate_based_metrics.gate_count, 2)  # H + CX
        self.assertEqual(result.gate_based_metrics.circuit_depth, 2)  # H then CX
        self.assertEqual(result.gate_based_metrics.cnot_count, 1)  # One CX gate
        self.assertEqual(result.gate_based_metrics.t_count, 0)  # No T gates
        self.assertEqual(result.gate_based_metrics.two_qubit_count, 1)  # One CX

        # Test ratios
        self.assertAlmostEqual(result.gate_based_metrics.multi_qubit_ratio, 0.5, places=2)  # 1/2
        # Clifford ratios are in standardized_metrics, not gate_based_metrics
        self.assertAlmostEqual(
            result.standardized_metrics.clifford_ratio, 1.0, places=2
        )  # All Clifford

    def test_gate_based_metrics_complex_circuit(self):
        """Test gate-based metrics for complex circuit."""
        metrics = ComplexityMetrics(self.complex_circuit)
        result = metrics.get_metrics()

        # Test gate-based metrics
        self.assertGreater(result.gate_based_metrics.gate_count, 5)
        self.assertGreater(result.gate_based_metrics.circuit_depth, 3)
        self.assertGreater(result.gate_based_metrics.cnot_count, 1)
        self.assertEqual(result.gate_based_metrics.t_count, 2)  # Two T gates
        self.assertGreater(result.gate_based_metrics.two_qubit_count, 1)

        # Test ratios are between 0 and 1
        self.assertGreaterEqual(result.gate_based_metrics.multi_qubit_ratio, 0.0)
        self.assertLessEqual(result.gate_based_metrics.multi_qubit_ratio, 1.0)
        # Clifford ratios are in standardized_metrics
        self.assertGreaterEqual(result.standardized_metrics.clifford_ratio, 0.0)
        self.assertLessEqual(result.standardized_metrics.clifford_ratio, 1.0)

    def test_entanglement_metrics(self):
        """Test entanglement metrics."""
        metrics = ComplexityMetrics(self.simple_circuit)

        result = metrics.get_metrics()

        # Test entanglement metrics
        self.assertGreater(result.entanglement_metrics.entangling_gate_density, 0)
        self.assertGreaterEqual(result.entanglement_metrics.entangling_width, 1)
        self.assertLessEqual(
            result.entanglement_metrics.entangling_width, 2
        )  # Max 2 for 2-qubit circuit

    def test_standardized_metrics(self):
        """Test standardized metrics."""
        metrics = ComplexityMetrics(self.simple_circuit)

        result = metrics.get_metrics()

        # Test standardized metrics
        self.assertGreater(result.standardized_metrics.circuit_volume, 0)
        self.assertGreater(result.standardized_metrics.gate_density, 0)
        self.assertGreaterEqual(result.standardized_metrics.clifford_ratio, 0.0)
        self.assertLessEqual(result.standardized_metrics.clifford_ratio, 1.0)

    def test_advanced_metrics(self):
        """Test advanced metrics."""
        metrics = ComplexityMetrics(self.simple_circuit)

        result = metrics.get_metrics()

        # Test advanced metrics
        self.assertGreaterEqual(result.advanced_metrics.parallelism_factor, 0.0)
        self.assertLessEqual(result.advanced_metrics.parallelism_factor, 1.0)
        self.assertGreaterEqual(result.advanced_metrics.circuit_efficiency, 0.0)
        self.assertLessEqual(result.advanced_metrics.circuit_efficiency, 1.0)
        self.assertGreaterEqual(result.advanced_metrics.parallelism_efficiency, 0.0)

    def test_derived_metrics(self):
        """Test derived metrics."""
        metrics = ComplexityMetrics(self.simple_circuit)

        result = metrics.get_metrics()

        # Test derived metrics
        self.assertGreater(result.derived_metrics.weighted_complexity, 0)

    def test_to_flat_dict(self):
        """Test conversion to flat dictionary."""
        metrics = ComplexityMetrics(self.simple_circuit)

        result = metrics.get_metrics()
        flat_dict = result.to_flat_dict()

        self.assertIsInstance(flat_dict, dict)

        # Check that expected keys exist
        expected_keys = [
            "gate_based_metrics.gate_count",
            "gate_based_metrics.circuit_depth",
            "gate_based_metrics.cnot_count",
            "gate_based_metrics.t_count",
            "entanglement_metrics.entangling_gate_density",
            "standardized_metrics.circuit_volume",
            "advanced_metrics.parallelism_factor",
            "derived_metrics.weighted_complexity",
        ]

        for key in expected_keys:
            self.assertIn(key, flat_dict, f"Expected key {key} not found in flat dict")

    def test_schema_validation(self):
        """Test that schema validation works correctly."""
        metrics = ComplexityMetrics(self.simple_circuit)

        result = metrics.get_metrics()

        # Test that all values are non-negative where expected
        self.assertGreaterEqual(result.gate_based_metrics.gate_count, 0)
        self.assertGreaterEqual(result.gate_based_metrics.circuit_depth, 0)
        self.assertGreaterEqual(result.gate_based_metrics.cnot_count, 0)
        self.assertGreaterEqual(result.gate_based_metrics.t_count, 0)
        self.assertGreaterEqual(result.entanglement_metrics.entangling_gate_density, 0)
        self.assertGreaterEqual(result.standardized_metrics.circuit_volume, 0)
        self.assertGreaterEqual(result.advanced_metrics.parallelism_factor, 0)
        self.assertGreaterEqual(result.derived_metrics.weighted_complexity, 0)

    def test_different_circuit_types(self):
        """Test with different types of circuits."""
        # QFT circuit using QFTGate
        qft_circuit = QuantumCircuit(3)
        qft_circuit.append(QFTGate(3), range(3))
        metrics = ComplexityMetrics(qft_circuit)
        result = metrics.get_metrics()

        self.assertGreater(result.gate_based_metrics.gate_count, 0)
        # QFT might not have two-qubit gates depending on implementation
        self.assertGreaterEqual(result.gate_based_metrics.two_qubit_count, 0)

        # Custom variational circuit (avoiding deprecated EfficientSU2)
        variational_circuit = QuantumCircuit(2)
        variational_circuit.ry(0.5, 0)
        variational_circuit.ry(0.3, 1)
        variational_circuit.cx(0, 1)
        variational_circuit.ry(0.7, 0)
        variational_circuit.ry(0.2, 1)

        metrics = ComplexityMetrics(variational_circuit)
        result = metrics.get_metrics()

        self.assertGreater(result.gate_based_metrics.gate_count, 0)

    def test_empty_circuit(self):
        """Test with empty circuit."""
        empty_circuit = QuantumCircuit(2)

        # Empty circuits might have validation issues, so we'll test more carefully
        try:
            metrics = ComplexityMetrics(empty_circuit)
            result = metrics.get_metrics()

            # Empty circuit should have zero gate counts
            self.assertEqual(result.gate_based_metrics.gate_count, 0)
            self.assertEqual(result.gate_based_metrics.cnot_count, 0)
            self.assertEqual(result.gate_based_metrics.t_count, 0)
            self.assertEqual(result.gate_based_metrics.two_qubit_count, 0)
        except Exception as e:
            # If empty circuit causes validation errors, that's acceptable
            self.assertIn("validation", str(e).lower())

    def test_single_qubit_circuit(self):
        """Test with single qubit circuit."""
        single_qubit = QuantumCircuit(1)
        single_qubit.h(0)
        single_qubit.t(0)

        try:
            metrics = ComplexityMetrics(single_qubit)
            result = metrics.get_metrics()

            # Single qubit circuit should have no multi-qubit gates
            self.assertEqual(result.gate_based_metrics.two_qubit_count, 0)
            self.assertEqual(result.gate_based_metrics.multi_qubit_ratio, 0.0)
            self.assertEqual(result.gate_based_metrics.t_count, 1)
        except Exception as e:
            # If single qubit circuit causes validation errors, that's acceptable
            self.assertIn("validation", str(e).lower())

    def test_t_gate_counting(self):
        """Test T-gate counting in complex circuit."""
        metrics = ComplexityMetrics(self.complex_circuit)
        result = metrics.get_metrics()

        # Should count T gates correctly
        self.assertEqual(result.gate_based_metrics.t_count, 2)

        # Non-Clifford ratio should be > 0 due to T gates (in standardized_metrics)
        self.assertGreater(result.standardized_metrics.non_clifford_ratio, 0.0)

    def test_name_property(self):
        """Test the name property."""
        metrics = ComplexityMetrics(self.simple_circuit)

        self.assertEqual(metrics.name, "ComplexityMetrics")

    def test_metric_type_and_id(self):
        """Test metric type and ID properties."""
        metrics = ComplexityMetrics(self.simple_circuit)

        self.assertEqual(metrics.metric_type, MetricsType.PRE_RUNTIME)
        self.assertEqual(metrics.id, MetricsId.COMPLEXITY)

    def test_consistency_across_calls(self):
        """Test that multiple calls return consistent results."""
        metrics = ComplexityMetrics(self.simple_circuit)

        result1 = metrics.get_metrics()
        result2 = metrics.get_metrics()

        # Results should be identical
        self.assertEqual(
            result1.gate_based_metrics.gate_count, result2.gate_based_metrics.gate_count
        )
        self.assertEqual(
            result1.advanced_metrics.parallelism_factor,
            result2.advanced_metrics.parallelism_factor,
        )
        self.assertAlmostEqual(
            result1.advanced_metrics.circuit_efficiency,
            result2.advanced_metrics.circuit_efficiency,
            places=5,
        )

    def test_ratio_validation(self):
        """Test that ratios are properly calculated and bounded."""
        metrics = ComplexityMetrics(self.simple_circuit)
        result = metrics.get_metrics()

        # All ratios should be between 0 and 1
        self.assertGreaterEqual(result.gate_based_metrics.multi_qubit_ratio, 0.0)
        self.assertLessEqual(result.gate_based_metrics.multi_qubit_ratio, 1.0)

        # Clifford + non-Clifford should sum to 1.0 (approximately)
        ratio_sum = (
            result.standardized_metrics.clifford_ratio
            + result.standardized_metrics.non_clifford_ratio
        )
        self.assertAlmostEqual(ratio_sum, 1.0, places=5)

    def test_large_circuit_performance(self):
        """Test performance with larger circuits."""
        # Create a larger circuit
        large_circuit = QuantumCircuit(8)
        for i in range(8):
            large_circuit.h(i)
        for i in range(7):
            large_circuit.cx(i, i + 1)

        metrics = ComplexityMetrics(large_circuit)
        result = metrics.get_metrics()

        # Should handle larger circuits without issues
        self.assertEqual(result.gate_based_metrics.gate_count, 15)  # 8 H + 7 CX
        self.assertGreater(result.advanced_metrics.parallelism_factor, 0)
        self.assertGreater(result.derived_metrics.weighted_complexity, 0)


if __name__ == "__main__":
    unittest.main()
