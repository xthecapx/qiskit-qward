"""Tests for qward QiskitMetrics class."""

import unittest
from qiskit import QuantumCircuit
from qiskit.circuit.library.basis_change import QFTGate

from qward.metrics.qiskit_metrics import QiskitMetrics
from qward.metrics.schemas import QiskitMetricsSchema
from qward.metrics.types import MetricsType, MetricsId


class TestQiskitMetrics(unittest.TestCase):
    """Test cases for QiskitMetrics class."""

    def setUp(self):
        """Set up test circuits."""
        # Simple Bell state circuit
        self.simple_circuit = QuantumCircuit(2, 2)
        self.simple_circuit.h(0)
        self.simple_circuit.cx(0, 1)
        self.simple_circuit.barrier()
        self.simple_circuit.measure_all()

        # More complex circuit for testing
        self.complex_circuit = QuantumCircuit(4)
        self.complex_circuit.h(0)
        self.complex_circuit.cx(0, 1)
        self.complex_circuit.cx(1, 2)
        self.complex_circuit.cx(2, 3)
        self.complex_circuit.rz(0.5, 0)
        self.complex_circuit.ry(0.3, 1)
        self.complex_circuit.barrier()
        self.complex_circuit.cx(3, 0)

    def test_initialization(self):
        """Test QiskitMetrics initialization."""
        metrics = QiskitMetrics(self.simple_circuit)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.circuit, self.simple_circuit)

    def test_basic_metrics_simple_circuit(self):
        """Test basic metrics calculation for simple circuit."""
        metrics = QiskitMetrics(self.simple_circuit)
        result = metrics.get_metrics()

        # Test basic metrics
        self.assertEqual(result.basic_metrics.num_qubits, 2)
        # measure_all() adds classical bits equal to qubits, but width includes both
        self.assertGreaterEqual(result.basic_metrics.num_clbits, 2)
        self.assertGreaterEqual(result.basic_metrics.depth, 3)
        self.assertGreaterEqual(result.basic_metrics.size, 4)
        # Width includes both qubits and classical bits
        self.assertGreaterEqual(result.basic_metrics.width, 4)

    def test_instruction_metrics_simple_circuit(self):
        """Test instruction metrics for simple circuit."""
        metrics = QiskitMetrics(self.simple_circuit)
        result = metrics.get_metrics()

        # Test instruction metrics
        self.assertGreaterEqual(result.instruction_metrics.num_connected_components, 1)
        self.assertGreaterEqual(result.instruction_metrics.num_nonlocal_gates, 0)
        self.assertGreaterEqual(result.instruction_metrics.num_tensor_factors, 1)
        self.assertGreaterEqual(result.instruction_metrics.num_unitary_factors, 1)
        self.assertIsInstance(result.instruction_metrics.instructions, dict)

    def test_instruction_metrics_complex_circuit(self):
        """Test instruction metrics for more complex circuit."""
        complex_circuit = QuantumCircuit(3)
        complex_circuit.h(0)
        complex_circuit.cx(0, 1)
        complex_circuit.cx(1, 2)
        complex_circuit.rz(0.5, 2)

        metrics = QiskitMetrics(complex_circuit)
        result = metrics.get_metrics()

        self.assertGreaterEqual(result.instruction_metrics.num_nonlocal_gates, 2)
        self.assertIn("cx", result.instruction_metrics.instructions)
        self.assertIn("h", result.instruction_metrics.instructions)

    def test_empty_circuit(self):
        """Test with empty circuit."""
        empty_circuit = QuantumCircuit(2)

        metrics = QiskitMetrics(empty_circuit)
        result = metrics.get_metrics()

        self.assertEqual(result.basic_metrics.num_qubits, 2)
        self.assertEqual(result.basic_metrics.size, 0)
        self.assertEqual(result.basic_metrics.depth, 0)

    def test_different_circuit_types(self):
        """Test with different types of circuits."""
        # QFT circuit using QFTGate
        qft_circuit = QuantumCircuit(3)
        qft_circuit.append(QFTGate(3), range(3))
        metrics = QiskitMetrics(qft_circuit)
        result = metrics.get_metrics()
        self.assertEqual(result.basic_metrics.num_qubits, 3)
        self.assertGreater(result.basic_metrics.size, 0)

        # Simple variational circuit
        variational_circuit = QuantumCircuit(2)
        variational_circuit.ry(0.5, 0)
        variational_circuit.ry(0.3, 1)
        variational_circuit.cx(0, 1)
        variational_circuit.ry(0.7, 0)
        variational_circuit.ry(0.2, 1)

        metrics = QiskitMetrics(variational_circuit)
        result = metrics.get_metrics()
        self.assertEqual(result.basic_metrics.num_qubits, 2)
        self.assertGreater(result.basic_metrics.size, 0)

    def test_single_qubit_circuit(self):
        """Test with single qubit circuit."""
        single_qubit = QuantumCircuit(1)
        single_qubit.h(0)
        single_qubit.rz(0.5, 0)

        metrics = QiskitMetrics(single_qubit)
        result = metrics.get_metrics()

        self.assertEqual(result.basic_metrics.num_qubits, 1)
        self.assertEqual(result.basic_metrics.size, 2)
        # No multi-qubit gates in single qubit circuit
        self.assertEqual(result.instruction_metrics.num_nonlocal_gates, 0)

    def test_circuit_with_barriers(self):
        """Test circuit with barriers."""
        circuit_with_barriers = QuantumCircuit(2)
        circuit_with_barriers.h(0)
        circuit_with_barriers.barrier()
        circuit_with_barriers.cx(0, 1)
        circuit_with_barriers.barrier()

        metrics = QiskitMetrics(circuit_with_barriers)
        result = metrics.get_metrics()

        self.assertEqual(result.basic_metrics.num_qubits, 2)
        # Size might not include barriers depending on implementation
        self.assertGreaterEqual(result.basic_metrics.size, 2)  # At least H and CX
        self.assertIn("barrier", result.instruction_metrics.instructions)

    def test_to_flat_dict(self):
        """Test conversion to flat dictionary."""
        metrics = QiskitMetrics(self.simple_circuit)

        result = metrics.get_metrics()
        flat_dict = result.to_flat_dict()

        self.assertIsInstance(flat_dict, dict)

        # Check that expected keys exist
        expected_keys = [
            "basic_metrics.depth",
            "basic_metrics.width",
            "basic_metrics.size",
            "basic_metrics.num_qubits",
            "basic_metrics.num_clbits",
            "instruction_metrics.num_connected_components",
            "instruction_metrics.num_nonlocal_gates",
        ]

        for key in expected_keys:
            self.assertIn(key, flat_dict, f"Expected key {key} not found in flat dict")

    def test_schema_validation(self):
        """Test that the returned schema is valid."""
        metrics = QiskitMetrics(self.simple_circuit)
        result = metrics.get_metrics()

        # Should not raise validation errors
        self.assertIsNotNone(result.basic_metrics)
        self.assertIsNotNone(result.instruction_metrics)
        self.assertIsNotNone(result.scheduling_metrics)

        # Test basic validation
        self.assertGreaterEqual(result.basic_metrics.depth, 0)
        self.assertGreaterEqual(result.basic_metrics.size, 0)
        self.assertGreaterEqual(result.basic_metrics.num_qubits, 0)

    def test_consistency_across_calls(self):
        """Test that multiple calls return consistent results."""
        metrics = QiskitMetrics(self.simple_circuit)

        result1 = metrics.get_metrics()
        result2 = metrics.get_metrics()

        # Basic metrics should be identical
        self.assertEqual(result1.basic_metrics.depth, result2.basic_metrics.depth)
        self.assertEqual(result1.basic_metrics.size, result2.basic_metrics.size)
        self.assertEqual(result1.basic_metrics.num_qubits, result2.basic_metrics.num_qubits)

        # Instruction metrics should be identical
        self.assertEqual(
            result1.instruction_metrics.num_nonlocal_gates,
            result2.instruction_metrics.num_nonlocal_gates,
        )


if __name__ == "__main__":
    unittest.main()
