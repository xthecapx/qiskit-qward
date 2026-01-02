"""Tests for qward BehavioralMetrics class."""

import unittest
from qiskit import QuantumCircuit
from qward.metrics.behavioral_metrics import BehavioralMetrics
from qward.schemas.behavioral_metrics_schema import BehavioralMetricsSchema


class TestBehavioralMetrics(unittest.TestCase):
    """Test cases for BehavioralMetrics class."""

    def setUp(self):
        """Set up test circuits."""
        # Simple Bell state circuit
        self.simple_circuit = QuantumCircuit(2, 2)
        self.simple_circuit.h(0)
        self.simple_circuit.cx(0, 1)
        self.simple_circuit.barrier()
        self.simple_circuit.measure_all()

        # More complex circuit for testing
        self.complex_circuit = QuantumCircuit(4, 2)

        self.complex_circuit.h(0)
        self.complex_circuit.x(2)
        self.complex_circuit.cx(0, 1)
        self.complex_circuit.cz(2, 3)
        self.complex_circuit.h(1)
        self.complex_circuit.x(3)
        self.complex_circuit.measure(0, 0)
        self.complex_circuit.reset(0)
        self.complex_circuit.h(0)
        self.complex_circuit.cx(1, 2)
        self.complex_circuit.swap(0, 3)
        self.complex_circuit.measure([2, 3], [0, 1])

    def test_initialization(self):
        """Test BehavioralMetrics initialization."""
        metrics = BehavioralMetrics(self.simple_circuit)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.circuit, self.simple_circuit)

    def test_basic_metrics_simple_circuit(self):
        """Test behavioral metrics calculation for simple circuit."""

        metrics = BehavioralMetrics(self.simple_circuit)
        result = metrics.get_metrics()
        self.assertIsInstance(result, BehavioralMetricsSchema)
        self.assertEqual(result.normalized_depth, 4)
        self.assertEqual(result.program_communication, 1)
        self.assertEqual(result.critical_depth, 1)
        self.assertEqual(result.measurement, 1 / 5)
        self.assertEqual(result.liveness, 1 / 2)
        self.assertAlmostEqual(result.parallelism, 0, places=9)

    def test_metrics_complex_circuit(self):
        """Test behavioral metrics for more complex circuit."""

        metrics = BehavioralMetrics(self.complex_circuit)
        result = metrics.get_metrics()
        self.assertIsInstance(result, BehavioralMetricsSchema)
        self.assertEqual(result.normalized_depth, 11)
        self.assertEqual(result.program_communication, 2 / 3)
        self.assertEqual(result.critical_depth, 0.5)
        self.assertEqual(result.measurement, 4 / 7)
        self.assertEqual(result.liveness, 17 / 28)
        self.assertAlmostEqual(result.parallelism, 2 / 21, places=9)

    def test_empty_circuit(self):
        """Test with empty circuit."""
        empty_circuit = QuantumCircuit(2)
        metrics = BehavioralMetrics(empty_circuit)
        result = metrics.get_metrics()
        self.assertEqual(result.normalized_depth, 0)
        self.assertEqual(result.program_communication, 0)
        self.assertEqual(result.critical_depth, 0)
        self.assertEqual(result.measurement, 0)
        self.assertEqual(result.liveness, 0)
        self.assertAlmostEqual(result.parallelism, 0, places=9)

    def test_single_qubit_circuit(self):
        """Test with single qubit circuit."""
        single_qubit = QuantumCircuit(1)
        single_qubit.h(0)
        single_qubit.rz(0.5, 0)
        metrics = BehavioralMetrics(single_qubit)
        result = metrics.get_metrics()
        self.assertEqual(result.normalized_depth, 2)
        self.assertEqual(result.program_communication, 0)
        self.assertEqual(result.critical_depth, 0)
        self.assertEqual(result.measurement, 0)
        self.assertEqual(result.liveness, 1)
        self.assertAlmostEqual(result.parallelism, 0, places=9)

    def test_consistency_across_calls(self):
        """Test that multiple calls return consistent results."""
        metrics = BehavioralMetrics(self.simple_circuit)
        result1 = metrics.get_metrics()
        result2 = metrics.get_metrics()
        self.assertEqual(result1.normalized_depth, result2.normalized_depth)
        self.assertEqual(result1.program_communication, result2.program_communication)
        self.assertEqual(result1.critical_depth, result2.critical_depth)
        self.assertEqual(result1.measurement, result2.measurement)
        self.assertEqual(result1.liveness, result2.liveness)
        self.assertEqual(result1.parallelism, result2.parallelism)


if __name__ == "__main__":
    unittest.main()
