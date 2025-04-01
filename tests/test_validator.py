"""Tests for qiskit_qward validators."""

from unittest import TestCase

from qiskit_qward.scanning_quantum_circuit import ScanningQuantumCircuit


class TestScanningQuantumCircuit(TestCase):
    """Tests scanning quantum circuit."""

    def test_scanning_circuit_init(self):
        """Tests scanning quantum circuit initialization."""
        validator = ScanningQuantumCircuit(
            num_qubits=2, num_clbits=2, use_barriers=True, name="test_validator"
        )

        self.assertEqual(validator.num_qubits, 2)
        self.assertEqual(validator.num_clbits, 2)
        self.assertTrue(validator.use_barriers)
        self.assertEqual(validator.name, "test_validator")
