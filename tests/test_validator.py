"""Tests for qward validators."""

from unittest import TestCase

from qiskit import QuantumCircuit
from qward.scanner import Scanner


class TestScanner(TestCase):
    """Tests scanner class."""

    def test_scanner_init(self):
        """Tests scanner initialization."""
        circuit = QuantumCircuit(2, 2)
        scanner = Scanner(circuit=circuit)

        self.assertEqual(scanner.circuit, circuit)
        self.assertIsNone(scanner.job)
        self.assertIsNone(scanner.result)
        self.assertEqual(scanner.strategies, [])
