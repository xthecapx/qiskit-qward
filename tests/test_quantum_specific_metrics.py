"""Tests for qward QuantumSpecificMetrics class."""

import unittest
from qiskit import QuantumCircuit

from qward.metrics.quantum_specific_metrics import QuantumSpecificMetrics, TORCH_AVAILABLE
from qward.schemas.quantum_specific_metrics_schema import QuantumSpecificMetricsSchema


class TestQuantumSpecificMetrics(unittest.TestCase):
    """Test cases for QuantumSpecificMetrics class."""

    def setUp(self):
        """Set up test circuits."""
        # Simple Bell state circuit
        self.simple_circuit = QuantumCircuit(2, 2)
        self.simple_circuit.h(0)
        self.simple_circuit.t(0)
        self.simple_circuit.cx(0, 1)

        # Circuito más complejo
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.s(1)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.h(2)
        self.complex_circuit = qc

    def test_initialization(self):
        """Test QuantumSpecificMetrics initialization."""
        metrics = QuantumSpecificMetrics(self.simple_circuit)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.circuit, self.simple_circuit)

    def test_basic_metrics_simple_circuit(self):
        """Test metrics calculation for simple Bell circuit."""
        metrics = QuantumSpecificMetrics(self.simple_circuit)
        result = metrics.get_metrics()
        self.assertIsInstance(result, QuantumSpecificMetricsSchema)

        self.assertAlmostEqual(result.spposq_ratio, 1 / 2, places=9)

        self.assertAlmostEqual(result.entanglement_ratio, 1 / 3, places=9)

        # Rango esperado para magic, coherence, sensitivity (only when PyTorch is available)
        if TORCH_AVAILABLE:
            self.assertGreaterEqual(result.magic, 0.7)
            self.assertLessEqual(result.magic, 1.1)

            self.assertGreaterEqual(result.coherence, 0.9)
            self.assertLessEqual(result.coherence, 1.1)

            self.assertGreaterEqual(result.sensitivity, 0.9)
            self.assertLessEqual(result.sensitivity, 1.1)
        else:
            # Without PyTorch, these metrics return 0.0
            self.assertEqual(result.magic, 0.0)
            self.assertEqual(result.coherence, 0.0)
            self.assertEqual(result.sensitivity, 0.0)

    def test_metrics_complex_circuit(self):
        """Test metrics for a more complex circuit."""
        metrics = QuantumSpecificMetrics(self.complex_circuit)
        result = metrics.get_metrics()
        self.assertIsInstance(result, QuantumSpecificMetricsSchema)

        self.assertAlmostEqual(result.spposq_ratio, 1 / 3, places=9)

        self.assertAlmostEqual(result.entanglement_ratio, 2 / 5, places=9)

        # These metrics use stochastic optimization, so values vary across runs.
        # We only verify they are non-negative and finite (valid outputs).
        import math

        if TORCH_AVAILABLE:
            # Values should be non-negative and finite
            self.assertGreaterEqual(result.magic, 0.0)
            self.assertTrue(math.isfinite(result.magic))

            self.assertGreaterEqual(result.coherence, 0.0)
            self.assertTrue(math.isfinite(result.coherence))

            self.assertGreaterEqual(result.sensitivity, 0.0)
            self.assertTrue(math.isfinite(result.sensitivity))
        else:
            # Without PyTorch, these metrics return 0.0
            self.assertEqual(result.magic, 0.0)
            self.assertEqual(result.coherence, 0.0)
            self.assertEqual(result.sensitivity, 0.0)

    def test_empty_circuit(self):
        """Test with empty circuit (no instructions)."""
        empty = QuantumCircuit(2)
        metrics = QuantumSpecificMetrics(empty)
        result = metrics.get_metrics()
        self.assertIsInstance(result, QuantumSpecificMetricsSchema)

        # Sin operaciones, spposq_ratio y entanglement_ratio deben ser 0
        self.assertAlmostEqual(result.spposq_ratio, 0.0, places=9)
        self.assertAlmostEqual(result.entanglement_ratio, 0.0, places=9)

        # Para identidad, magic/coherence/sensitivity ~ 0
        self.assertAlmostEqual(result.magic, 0.0, places=2)
        self.assertAlmostEqual(result.coherence, 0.0, places=2)
        self.assertAlmostEqual(result.sensitivity, 0.0, places=2)

    def test_consistency_across_calls(self):
        """Test that multiple calls return consistent results."""
        metrics = QuantumSpecificMetrics(self.simple_circuit)
        r1 = metrics.get_metrics()
        r2 = metrics.get_metrics()
        self.assertEqual(r1.spposq_ratio, r2.spposq_ratio)
        self.assertEqual(r1.entanglement_ratio, r2.entanglement_ratio)
        self.assertAlmostEqual(r1.magic, r2.magic, places=2)
        self.assertAlmostEqual(r1.coherence, r2.coherence, places=2)
        self.assertAlmostEqual(r1.sensitivity, r2.sensitivity, places=2)


if __name__ == "__main__":
    unittest.main()
