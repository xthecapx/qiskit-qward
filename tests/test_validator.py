"""Tests for qward Scanner class (legacy test file)."""

import unittest
from qiskit import QuantumCircuit
from qward.scanner import Scanner
from qward.metrics import QiskitMetrics


class TestScannerLegacy(unittest.TestCase):
    """Legacy tests for Scanner class."""

    def test_scanner_init(self):
        """Tests scanner initialization."""
        circuit = QuantumCircuit(2, 2)
        scanner = Scanner(circuit=circuit)

        self.assertEqual(scanner.circuit, circuit)
        self.assertIsNone(scanner.job)
        self.assertEqual(scanner.strategies, [])

    def test_scanner_with_strategies(self):
        """Test scanner with strategies."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)

        scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics])

        self.assertEqual(len(scanner.strategies), 1)
        self.assertIsInstance(scanner.strategies[0], QiskitMetrics)

        # Test calculation
        results = scanner.calculate_metrics()
        self.assertIsInstance(results, dict)
        self.assertIn("QiskitMetrics", results)

    def test_scanner_add_strategy(self):
        """Test adding strategies to scanner."""
        circuit = QuantumCircuit(2, 2)
        scanner = Scanner(circuit=circuit)

        # Add strategy class
        scanner.add_strategy(QiskitMetrics)
        self.assertEqual(len(scanner.strategies), 1)

        # Add strategy instance
        qiskit_metrics = QiskitMetrics(circuit)
        scanner.add_strategy(qiskit_metrics)
        self.assertEqual(len(scanner.strategies), 2)


if __name__ == "__main__":
    unittest.main()
