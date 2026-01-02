"""Tests for qward ElementMetrics class."""

import unittest
from qiskit import QuantumCircuit
from qiskit.circuit.library import SGate, TGate
from qward.metrics.element_metrics import ElementMetrics
from qward.schemas.element_metrics_schema import ElementMetricsSchema


class TestElementMetrics(unittest.TestCase):
    """Test cases for ElementMetrics class."""

    def setUp(self):
        """Set up test circuits."""
        # Simple circuit
        self.simple_circuit = QuantumCircuit(5)
        self.simple_circuit.cx(0, 1)
        self.simple_circuit.cx(0, 3)
        self.simple_circuit.cx(2, 3)
        self.simple_circuit.ccx(1, 3, 4)
        self.simple_circuit.cx(0, 4)
        self.simple_circuit.cx(0, 1)
        self.simple_circuit.cx(1, 3)

        # Complex circuit as solicitado
        n_count = 3  # qubits de conteo
        n_target = 2  # qubits del eigenvector
        qc = QuantumCircuit(n_count + n_target, n_count)
        for q in range(n_count):
            qc.h(q)
        unitary_circ = QuantumCircuit(n_target)
        unitary_circ.cz(0, 1)
        unitary_circ.name = "U"
        unitary_gate = unitary_circ.to_gate()
        unitary2_gate = unitary_circ.power(2).to_gate()
        unitary4_gate = unitary_circ.power(4).to_gate()

        qc.append(unitary4_gate.control(), [0] + [3, 4])
        qc.h(0)
        qc.append(unitary2_gate.control(), [1] + [3, 4])
        qc.append(unitary_gate.control(), [2] + [3, 4])

        csgate_0 = SGate().control(1)
        qc.append(csgate_0, [0, 1])
        qc.h(1)

        csgate = SGate().control(1)
        ctgate = TGate().control(1)

        qc.append(ctgate, [0, 2])
        qc.append(csgate, [1, 2])
        qc.h(2)

        qc.measure(range(n_count), range(n_count))
        self.complex_circuit = qc

    def test_initialization(self):
        """Test ElementMetrics initialization."""
        metrics = ElementMetrics(self.simple_circuit)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.circuit, self.simple_circuit)

    def test_basic_metrics_simple_circuit(self):
        """Test element metrics calculation for simple circuit."""
        metrics = ElementMetrics(self.simple_circuit)
        result = metrics.get_metrics()
        self.assertIsInstance(result, ElementMetricsSchema)
        # Verifica todos los campos del schema para el circuito simple
        self.assertEqual(result.no_p_x, 0)
        self.assertEqual(result.no_p_y, 0)
        self.assertEqual(result.no_p_z, 0)
        self.assertEqual(result.t_no_p, 0)
        self.assertEqual(result.no_h, 0)
        self.assertAlmostEqual(result.percent_sppos_q, 0, places=9)
        self.assertEqual(result.no_other_sg, 0)
        self.assertEqual(result.t_no_csqg, 0)
        self.assertEqual(result.t_no_sqg, 0)
        self.assertEqual(result.no_c_or, 0)
        self.assertEqual(result.no_c_any_g, 7)
        self.assertEqual(result.no_swap, 0)
        self.assertEqual(result.no_cnot, 6)
        self.assertAlmostEqual(result.percent_q_in_cnot, 1, places=9)
        self.assertAlmostEqual(result.avg_cnot, 1.2, places=9)
        self.assertAlmostEqual(result.max_cnot, 3, places=9)
        self.assertAlmostEqual(result.no_toff, 1, places=9)
        self.assertAlmostEqual(result.percent_q_in_toff, 0.6, places=9)
        self.assertAlmostEqual(result.avg_toff, 0.2, places=9)
        self.assertAlmostEqual(result.max_toff, 1, places=9)
        self.assertEqual(result.no_gates, 7)
        self.assertAlmostEqual(result.no_c_gates, 7, places=9)
        self.assertAlmostEqual(result.percent_single_gates, 0, places=9)
        self.assertEqual(result.no_or, 0)
        self.assertAlmostEqual(result.percent_q_in_or, 0, places=9)
        self.assertAlmostEqual(result.percent_q_in_c_or, 0, places=9)
        self.assertAlmostEqual(result.avg_or_d, 0, places=9)
        self.assertEqual(result.max_or_d, 0)
        self.assertEqual(result.no_qm, 0)
        self.assertAlmostEqual(result.percent_qm, 0, places=9)
        self.assertAlmostEqual(result.percent_anc, 0, places=9)

    def test_metrics_complex_circuit(self):
        """Test element metrics for complex circuit."""
        metrics = ElementMetrics(self.complex_circuit)
        result = metrics.get_metrics()
        self.assertIsInstance(result, ElementMetricsSchema)
        # Verifica todos los campos del schema para el circuito complejo
        self.assertEqual(result.no_p_x, 0)
        self.assertEqual(result.no_p_y, 0)
        self.assertEqual(result.no_p_z, 0)
        self.assertEqual(result.t_no_p, 0)
        self.assertEqual(result.no_h, 6)
        self.assertAlmostEqual(result.percent_sppos_q, 0.6, places=9)
        self.assertEqual(result.no_other_sg, 0)
        self.assertEqual(result.t_no_csqg, 3)
        self.assertEqual(result.t_no_sqg, 9)
        self.assertEqual(result.no_c_or, 3)
        self.assertEqual(result.no_c_any_g, 6)
        self.assertEqual(result.no_swap, 0)
        self.assertEqual(result.no_cnot, 0)
        self.assertAlmostEqual(result.percent_q_in_cnot, 0, places=9)
        self.assertAlmostEqual(result.avg_cnot, 0, places=9)
        self.assertAlmostEqual(result.max_cnot, 0, places=9)
        self.assertAlmostEqual(result.no_toff, 0, places=9)
        self.assertAlmostEqual(result.percent_q_in_toff, 0, places=9)
        self.assertAlmostEqual(result.avg_toff, 0, places=9)
        self.assertAlmostEqual(result.max_toff, 0, places=9)
        self.assertEqual(result.no_gates, 12)
        self.assertAlmostEqual(result.no_c_gates, 6, places=9)
        self.assertAlmostEqual(result.percent_single_gates, 0.75, places=9)
        self.assertEqual(result.no_or, 3)
        self.assertAlmostEqual(result.percent_q_in_or, 0.4, places=9)
        self.assertAlmostEqual(result.percent_q_in_c_or, 1, places=9)
        self.assertAlmostEqual(result.avg_or_d, 2, places=9)
        self.assertEqual(result.max_or_d, 2)
        self.assertEqual(result.no_qm, 3)
        self.assertAlmostEqual(result.percent_qm, 0.6, places=9)
        self.assertAlmostEqual(result.percent_anc, 0, places=9)

    def test_empty_circuit(self):
        """Test with empty circuit."""
        empty_circuit = QuantumCircuit(2)
        metrics = ElementMetrics(empty_circuit)
        result = metrics.get_metrics()
        self.assertEqual(result.no_gates, 0)
        self.assertEqual(result.no_qm, 0)
        self.assertEqual(result.no_h, 0)

    def test_single_qubit_circuit(self):
        """Test with single qubit circuit."""
        single_qubit = QuantumCircuit(1)
        single_qubit.h(0)
        single_qubit.rz(0.5, 0)
        metrics = ElementMetrics(single_qubit)
        result = metrics.get_metrics()
        self.assertEqual(result.no_gates, 2)
        self.assertEqual(result.no_h, 1)
        self.assertEqual(result.no_qm, 0)

    def test_consistency_across_calls(self):
        """Test that multiple calls return consistent results."""
        metrics = ElementMetrics(self.simple_circuit)
        result1 = metrics.get_metrics()
        result2 = metrics.get_metrics()
        self.assertEqual(result1.no_gates, result2.no_gates)
        self.assertEqual(result1.no_h, result2.no_h)
        self.assertEqual(result1.no_qm, result2.no_qm)


if __name__ == "__main__":
    unittest.main()
