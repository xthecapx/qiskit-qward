"""Tests for qward Scanner class."""

import unittest
from unittest.mock import Mock, patch
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import pandas as pd

from qward.scanner import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics


class TestScanner(unittest.TestCase):
    """Tests for Scanner class."""

    def setUp(self):
        """Set up test fixtures."""
        self.circuit = QuantumCircuit(2, 2)
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        self.circuit.measure_all()

        # Create a mock job for testing
        self.simulator = AerSimulator()
        self.job = self.simulator.run(self.circuit, shots=100)

    def test_scanner_init_with_circuit_only(self):
        """Test scanner initialization with circuit only."""
        scanner = Scanner(circuit=self.circuit)

        self.assertEqual(scanner.circuit, self.circuit)
        self.assertIsNone(scanner.job)
        self.assertEqual(scanner.strategies, [])

    def test_scanner_init_with_circuit_and_job(self):
        """Test scanner initialization with circuit and job."""
        scanner = Scanner(circuit=self.circuit, job=self.job)

        self.assertEqual(scanner.circuit, self.circuit)
        self.assertEqual(scanner.job, self.job)
        self.assertEqual(scanner.strategies, [])

    def test_scanner_init_with_strategies_classes(self):
        """Test scanner initialization with strategy classes."""
        scanner = Scanner(circuit=self.circuit, strategies=[QiskitMetrics, ComplexityMetrics])

        self.assertEqual(len(scanner.strategies), 2)
        self.assertIsInstance(scanner.strategies[0], QiskitMetrics)
        self.assertIsInstance(scanner.strategies[1], ComplexityMetrics)

    def test_scanner_init_with_strategy_instances(self):
        """Test scanner initialization with strategy instances."""
        qiskit_metrics = QiskitMetrics(self.circuit)
        complexity_metrics = ComplexityMetrics(self.circuit)

        scanner = Scanner(circuit=self.circuit, strategies=[qiskit_metrics, complexity_metrics])

        self.assertEqual(len(scanner.strategies), 2)
        self.assertEqual(scanner.strategies[0], qiskit_metrics)
        self.assertEqual(scanner.strategies[1], complexity_metrics)

    def test_add_strategy_instance(self):
        """Test adding strategy instance."""
        scanner = Scanner(circuit=self.circuit)
        qiskit_metrics = QiskitMetrics(self.circuit)

        scanner.add_strategy(qiskit_metrics)

        self.assertEqual(len(scanner.strategies), 1)
        self.assertEqual(scanner.strategies[0], qiskit_metrics)

    def test_add_strategy_class(self):
        """Test adding strategy class."""
        scanner = Scanner(circuit=self.circuit)

        scanner.add_strategy(QiskitMetrics(self.circuit))

        self.assertEqual(len(scanner.strategies), 1)
        self.assertIsInstance(scanner.strategies[0], QiskitMetrics)

    def test_calculate_metrics_qiskit_only(self):
        """Test calculating metrics with QiskitMetrics only."""
        scanner = Scanner(circuit=self.circuit)
        scanner.add_strategy(QiskitMetrics(self.circuit))

        results = scanner.calculate_metrics()

        # Verify return type and structure
        self.assertIsInstance(results, dict)
        self.assertIn("QiskitMetrics", results)
        self.assertIsInstance(results["QiskitMetrics"], pd.DataFrame)

        # Verify DataFrame has expected columns
        df = results["QiskitMetrics"]
        self.assertGreater(len(df.columns), 0)
        self.assertEqual(len(df), 1)  # Should have one row

        # Verify some expected columns exist
        expected_columns = [
            "basic_metrics.depth",
            "basic_metrics.width",
            "basic_metrics.size",
            "basic_metrics.num_qubits",
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Expected column {col} not found")

    def test_calculate_metrics_complexity_only(self):
        """Test calculating metrics with ComplexityMetrics only."""
        scanner = Scanner(circuit=self.circuit)
        scanner.add_strategy(ComplexityMetrics(self.circuit))

        results = scanner.calculate_metrics()

        # Verify return type and structure
        self.assertIsInstance(results, dict)
        self.assertIn("ComplexityMetrics", results)
        self.assertIsInstance(results["ComplexityMetrics"], pd.DataFrame)

        # Verify DataFrame has expected columns
        df = results["ComplexityMetrics"]
        self.assertGreater(len(df.columns), 0)
        self.assertEqual(len(df), 1)

        # Verify some expected complexity columns exist
        expected_columns = [
            "gate_based_metrics.gate_count",
            "gate_based_metrics.circuit_depth",
            "advanced_metrics.parallelism_factor",
            "derived_metrics.weighted_complexity",
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Expected column {col} not found")

    def test_calculate_metrics_circuit_performance(self):
        """Test calculating metrics with CircuitPerformanceMetrics."""

        def success_criteria(result: str) -> bool:
            return result.replace(" ", "") in ["00", "11"]

        scanner = Scanner(circuit=self.circuit, job=self.job)
        scanner.add_strategy(
            CircuitPerformanceMetrics(
                circuit=self.circuit, job=self.job, success_criteria=success_criteria
            )
        )

        results = scanner.calculate_metrics()

        # Verify return type and structure
        self.assertIsInstance(results, dict)

        # CircuitPerformanceMetrics should create individual_jobs results
        self.assertIn("CircuitPerformance.individual_jobs", results)
        self.assertIsInstance(results["CircuitPerformance.individual_jobs"], pd.DataFrame)

        # Verify DataFrame has expected columns (schema-based format)
        df = results["CircuitPerformance.individual_jobs"]
        expected_columns = [
            "success_metrics.success_rate",
            "success_metrics.error_rate",
            "success_metrics.total_shots",
            "fidelity_metrics.fidelity",
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Expected column {col} not found")

    def test_calculate_metrics_multiple_strategies(self):
        """Test calculating metrics with multiple strategies."""
        scanner = Scanner(circuit=self.circuit)
        scanner.add_strategy(QiskitMetrics(self.circuit))
        scanner.add_strategy(ComplexityMetrics(self.circuit))

        results = scanner.calculate_metrics()

        # Verify return type and structure
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 2)
        self.assertIn("QiskitMetrics", results)
        self.assertIn("ComplexityMetrics", results)

        # Verify both are DataFrames
        self.assertIsInstance(results["QiskitMetrics"], pd.DataFrame)
        self.assertIsInstance(results["ComplexityMetrics"], pd.DataFrame)

    def test_set_circuit(self):
        """Test setting a new circuit."""
        scanner = Scanner(circuit=self.circuit)
        new_circuit = QuantumCircuit(3)

        scanner.set_circuit(new_circuit)

        self.assertEqual(scanner.circuit, new_circuit)

    def test_set_job(self):
        """Test setting a new job."""
        scanner = Scanner(circuit=self.circuit)

        scanner.set_job(self.job)

        self.assertEqual(scanner.job, self.job)

    def test_empty_strategies_calculate_metrics(self):
        """Test calculating metrics with no strategies."""
        scanner = Scanner(circuit=self.circuit)

        results = scanner.calculate_metrics()

        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 0)

    def test_strategy_error_handling(self):
        """Test error handling when strategy fails."""
        scanner = Scanner(circuit=self.circuit)

        # Create a mock strategy that will fail
        mock_strategy = Mock()
        mock_strategy.name = "MockStrategy"
        mock_strategy.get_metrics.side_effect = Exception("Test error")
        mock_strategy.is_ready.return_value = True
        mock_strategy.__class__.__name__ = "MockStrategy"

        scanner.strategies.append(mock_strategy)

        # Should raise the exception since Scanner doesn't handle errors gracefully
        with self.assertRaises(Exception):
            scanner.calculate_metrics()


if __name__ == "__main__":
    unittest.main()
