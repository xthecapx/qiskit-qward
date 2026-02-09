"""Tests for qward Scanner class."""

import tempfile
import unittest
from unittest.mock import Mock, patch
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import pandas as pd

from qward.scanner import Scanner
from qward.metrics import (
    CircuitPerformanceMetrics,
    ComplexityMetrics,
    MetricCalculator,
    QiskitMetrics,
)


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

        scanner.add_strategy(QiskitMetrics)

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
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Expected column {col} not found")

    def test_calculate_metrics_circuit_performance_with_dsr(self):
        """Test scanner includes DSR fields for single-job circuit performance."""
        counts = self.job.result().get_counts()
        expected_outcomes = [max(counts, key=counts.get)]

        scanner = Scanner(circuit=self.circuit, job=self.job)
        scanner.add_strategy(
            CircuitPerformanceMetrics(
                circuit=self.circuit,
                job=self.job,
                expected_outcomes=expected_outcomes,
            )
        )

        results = scanner.calculate_metrics()
        self.assertIn("CircuitPerformance.individual_jobs", results)
        df = results["CircuitPerformance.individual_jobs"]

        self.assertIn("dsr_metrics.dsr_michelson", df.columns)
        self.assertIn("dsr_metrics.expected_outcomes", df.columns)

    def test_calculate_metrics_circuit_performance_multi_job_dsr_split(self):
        """Test scanner splits DSR individual and aggregate fields correctly."""
        job2 = self.simulator.run(self.circuit, shots=100)
        counts = self.job.result().get_counts()
        expected_outcomes = [max(counts, key=counts.get)]

        scanner = Scanner(circuit=self.circuit)
        scanner.add_strategy(
            CircuitPerformanceMetrics(
                circuit=self.circuit,
                jobs=[self.job, job2],
                expected_outcomes=expected_outcomes,
            )
        )

        results = scanner.calculate_metrics()
        self.assertIn("CircuitPerformance.individual_jobs", results)
        self.assertIn("CircuitPerformance.aggregate", results)

        individual_df = results["CircuitPerformance.individual_jobs"]
        aggregate_df = results["CircuitPerformance.aggregate"]

        self.assertIn("dsr_metrics.dsr_michelson", individual_df.columns)
        self.assertIn("dsr_metrics.mean_dsr_michelson", aggregate_df.columns)
        self.assertIn("dsr_metrics.peak_mismatch_rate", aggregate_df.columns)
        self.assertIn("dsr_metrics.total_jobs", aggregate_df.columns)

        self.assertNotIn("dsr_metrics.peak_mismatch_rate", individual_df.columns)
        self.assertNotIn("dsr_metrics.total_jobs", individual_df.columns)

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


class TestScannerFluentAPI(unittest.TestCase):
    """Tests for fluent Scanner API: add(), scan(), and ScanResult."""

    def setUp(self):
        self.circuit = QuantumCircuit(2, 2)
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        self.circuit.measure_all()
        self.simulator = AerSimulator()
        self.job = self.simulator.run(self.circuit, shots=100)

    def test_add_strategy_accepts_class(self):
        """add_strategy() should accept class and auto-instantiate."""
        scanner = Scanner(circuit=self.circuit)
        scanner.add_strategy(QiskitMetrics)
        self.assertEqual(len(scanner.strategies), 1)
        self.assertIsInstance(scanner.strategies[0], QiskitMetrics)

    def test_add_returns_self(self):
        """add() should return scanner for chaining."""
        scanner = Scanner(circuit=self.circuit)
        result = scanner.add(QiskitMetrics)
        self.assertIs(result, scanner)

    def test_add_chaining(self):
        """add() should support chaining multiple strategies."""
        scanner = Scanner(circuit=self.circuit).add(QiskitMetrics).add(ComplexityMetrics)
        self.assertEqual(len(scanner.strategies), 2)

    def test_add_with_kwargs(self):
        """add() should forward kwargs to constructor for strategy classes."""
        scanner = Scanner(circuit=self.circuit)
        scanner.add(CircuitPerformanceMetrics, job=self.job)
        self.assertEqual(len(scanner.strategies), 1)
        self.assertIsInstance(scanner.strategies[0], CircuitPerformanceMetrics)

    def test_scan_returns_scan_result(self):
        """scan() should return ScanResult."""
        from qward.scanner import ScanResult

        result = Scanner(circuit=self.circuit, strategies=[QiskitMetrics]).scan()
        self.assertIsInstance(result, ScanResult)

    def test_scan_auto_enrolls_all_pre_runtime(self):
        """scan() should auto-enroll all pre-runtime strategies when none registered."""
        result = Scanner(circuit=self.circuit).scan()
        self.assertEqual(len(result.keys()), 6)

    def test_scan_respects_explicit_strategies(self):
        """scan() should not auto-enroll when strategies are explicitly registered."""
        result = Scanner(circuit=self.circuit, strategies=[QiskitMetrics]).scan()
        self.assertIn("QiskitMetrics", result.keys())
        self.assertEqual(len(list(result.keys())), 1)

    def test_scan_result_dict_access(self):
        """ScanResult should support dict-style access."""
        result = Scanner(circuit=self.circuit, strategies=[QiskitMetrics]).scan()
        self.assertIn("QiskitMetrics", result)
        df = result["QiskitMetrics"]
        self.assertIsInstance(df, pd.DataFrame)

    def test_scan_result_to_dict(self):
        """ScanResult.to_dict() should return plain dict."""
        result = Scanner(circuit=self.circuit, strategies=[QiskitMetrics]).scan()
        as_dict = result.to_dict()
        self.assertIsInstance(as_dict, dict)

    def test_scan_result_summary_returns_self(self):
        """summary() should return self for fluent chaining."""
        result = Scanner(circuit=self.circuit, strategies=[QiskitMetrics]).scan()
        returned = result.summary()
        self.assertIs(returned, result)

    def test_scan_result_visualize_returns_self(self):
        """visualize() should return self for fluent chaining."""
        result = Scanner(circuit=self.circuit, strategies=[QiskitMetrics]).scan()
        with tempfile.TemporaryDirectory() as tmpdir:
            returned = result.visualize(save=True, show=False, output_dir=tmpdir)
            self.assertIs(returned, result)

    def test_get_all_pre_runtime_strategies(self):
        """get_all_pre_runtime_strategies() should return 6 strategy classes."""
        from qward.metrics.defaults import get_all_pre_runtime_strategies

        strategies = get_all_pre_runtime_strategies()
        self.assertEqual(len(strategies), 6)
        for strategy in strategies:
            self.assertTrue(issubclass(strategy, MetricCalculator))


if __name__ == "__main__":
    unittest.main()
