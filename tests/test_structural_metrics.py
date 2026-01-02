"""Tests for qward StructuralMetrics class."""

import unittest
from qiskit import QuantumCircuit
from qward.metrics.structural_metrics import StructuralMetrics
from qward.schemas.structural_metrics_schema import StructuralMetricsSchema
from math import log2


class TestStructuralMetrics(unittest.TestCase):
    """Test cases for StructuralMetrics class."""

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
        """Test StructuralMetrics initialization."""
        metrics = StructuralMetrics(self.simple_circuit)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.circuit, self.simple_circuit)

    def test_basic_metrics_simple_circuit(self):
        """Test structural metrics calculation for simple circuit."""
        metrics = StructuralMetrics(self.simple_circuit)
        result = metrics.get_metrics()
        self.assertIsInstance(result, StructuralMetricsSchema)
        # Verifica todos los campos del schema para el circuito simple
        self.assertEqual(result.phi1_total_loc, 6)
        self.assertEqual(result.phi2_gate_loc, 2)
        self.assertEqual(result.phi3_measure_loc, 2)
        self.assertEqual(result.phi4_quantum_total_loc, 4)
        self.assertEqual(result.phi5_num_qubits, 2)
        self.assertEqual(result.phi6_num_gate_types, 2)
        self.assertEqual(result.unique_operators, 3)
        self.assertEqual(result.unique_operands, 4)
        self.assertEqual(result.total_operators, 4)
        self.assertEqual(result.total_operands, 7)
        program_len = result.total_operators + result.total_operands
        self.assertEqual(result.program_length, program_len)
        miu = result.unique_operators + result.unique_operands
        self.assertEqual(result.vocabulary, miu)
        estimated_len = result.unique_operators * log2(
            result.unique_operators
        ) + result.unique_operands * log2(result.unique_operands)
        self.assertAlmostEqual(result.estimated_length, estimated_len, places=9)
        vol = program_len * log2(miu)
        self.assertAlmostEqual(result.volume, vol, places=9)
        diff = (result.unique_operators / 2) * (result.total_operands / result.unique_operands)
        self.assertEqual(result.difficulty, diff)
        self.assertEqual(result.effort, diff * vol)
        self.assertEqual(result.width, 2)
        self.assertEqual(result.depth, 3)
        self.assertEqual(result.max_dens, 2)
        self.assertAlmostEqual(result.avg_dens, 6 / 5)
        self.assertEqual(result.size, 6)

    def test_metrics_complex_circuit(self):
        """Test structural metrics calculation for complex circuit."""
        metrics = StructuralMetrics(self.complex_circuit)
        result = metrics.get_metrics()
        self.assertIsInstance(result, StructuralMetricsSchema)
        # Verifica todos los campos del schema para el circuito complejo
        self.assertEqual(result.phi1_total_loc, 8)  # The barrier sentences are included
        self.assertEqual(result.phi2_gate_loc, 7)
        self.assertEqual(result.phi3_measure_loc, 0)
        self.assertEqual(result.phi4_quantum_total_loc, 7)
        self.assertEqual(result.phi5_num_qubits, 4)
        self.assertEqual(result.phi6_num_gate_types, 4)
        self.assertEqual(result.unique_operators, 4)
        self.assertEqual(result.unique_operands, 4)
        self.assertEqual(result.total_operators, 7)
        self.assertEqual(result.total_operands, 11)
        program_len = result.total_operators + result.total_operands
        self.assertEqual(result.program_length, program_len)
        miu = result.unique_operators + result.unique_operands
        self.assertEqual(result.vocabulary, miu)
        estimated_len = result.unique_operators * log2(
            result.unique_operators
        ) + result.unique_operands * log2(result.unique_operands)
        self.assertAlmostEqual(result.estimated_length, estimated_len, places=9)
        vol = program_len * log2(miu)
        self.assertAlmostEqual(result.volume, vol, places=9)
        diff = (result.unique_operators / 2) * (result.total_operands / result.unique_operands)
        self.assertEqual(result.difficulty, diff)
        self.assertEqual(result.effort, diff * vol)
        self.assertEqual(result.width, 4)
        self.assertEqual(result.depth, 5)
        self.assertEqual(result.max_dens, 2)
        self.assertAlmostEqual(result.avg_dens, 8 / 6)
        self.assertEqual(result.size, 8)

    def test_empty_circuit(self):
        """Test with empty circuit."""
        empty_circuit = QuantumCircuit(2)
        metrics = StructuralMetrics(empty_circuit)
        result = metrics.get_metrics()
        self.assertEqual(result.phi5_num_qubits, 2)
        self.assertEqual(result.width, 2)
        self.assertEqual(result.size, result.total_operators)
        self.assertEqual(result.program_length, result.total_operators + result.total_operands)
        self.assertEqual(result.vocabulary, result.unique_operators + result.unique_operands)

    def test_consistency_across_calls(self):
        """Test that multiple calls return consistent results."""
        metrics = StructuralMetrics(self.simple_circuit)
        result1 = metrics.get_metrics()
        result2 = metrics.get_metrics()
        self.assertEqual(result1.size, result2.size)
        self.assertEqual(result1.width, result2.width)
        self.assertEqual(result1.depth, result2.depth)
        self.assertEqual(result1.program_length, result2.program_length)


if __name__ == "__main__":
    unittest.main()
