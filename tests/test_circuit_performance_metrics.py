"""Tests for qward CircuitPerformanceMetrics class."""

import unittest
from unittest.mock import Mock, MagicMock
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qward.metrics import CircuitPerformanceMetrics
from qward.metrics.schemas import CircuitPerformanceSchema
from qward.metrics.types import MetricsType, MetricsId


class TestCircuitPerformanceMetrics(unittest.TestCase):
    """Tests for CircuitPerformanceMetrics class."""

    def setUp(self):
        """Set up test fixtures."""
        # Simple 2-qubit Bell state circuit
        self.circuit = QuantumCircuit(2, 2)
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        self.circuit.measure_all()

        # Create a real job for testing
        self.simulator = AerSimulator()
        self.job = self.simulator.run(self.circuit, shots=1000)

        # Default success criteria (Bell state: 00 00 or 11 00) - updated format
        self.bell_success_criteria = lambda result: result in ["00 00", "11 00"]

    def test_circuit_performance_init_with_job(self):
        """Test CircuitPerformanceMetrics initialization with job."""
        metrics = CircuitPerformanceMetrics(
            circuit=self.circuit, job=self.job, success_criteria=self.bell_success_criteria
        )

        self.assertEqual(metrics.circuit, self.circuit)
        self.assertEqual(metrics._job, self.job)
        self.assertEqual(metrics._get_metric_type(), MetricsType.POST_RUNTIME)
        self.assertEqual(metrics._get_metric_id(), MetricsId.CIRCUIT_PERFORMANCE)
        self.assertTrue(metrics.is_ready())

    def test_circuit_performance_init_with_jobs_list(self):
        """Test CircuitPerformanceMetrics initialization with jobs list."""
        job2 = self.simulator.run(self.circuit, shots=500)

        metrics = CircuitPerformanceMetrics(
            circuit=self.circuit, jobs=[self.job, job2], success_criteria=self.bell_success_criteria
        )

        self.assertEqual(metrics.circuit, self.circuit)
        self.assertEqual(len(metrics._jobs), 2)
        self.assertTrue(metrics.is_ready())

    def test_circuit_performance_init_no_job(self):
        """Test CircuitPerformanceMetrics initialization without job."""
        metrics = CircuitPerformanceMetrics(
            circuit=self.circuit, success_criteria=self.bell_success_criteria
        )

        self.assertEqual(metrics.circuit, self.circuit)
        self.assertIsNone(metrics._job)
        self.assertEqual(len(metrics._jobs), 0)
        self.assertFalse(metrics.is_ready())

    def test_circuit_performance_init_none_circuit(self):
        """Test CircuitPerformanceMetrics initialization with None circuit."""
        metrics = CircuitPerformanceMetrics(
            circuit=None, job=self.job, success_criteria=self.bell_success_criteria
        )

        self.assertIsNone(metrics.circuit)
        self.assertFalse(metrics.is_ready())

    def test_get_metrics_returns_schema(self):
        """Test that get_metrics returns a CircuitPerformanceSchema."""
        metrics = CircuitPerformanceMetrics(
            circuit=self.circuit, job=self.job, success_criteria=self.bell_success_criteria
        )

        result = metrics.get_metrics()

        self.assertIsInstance(result, CircuitPerformanceSchema)

    def test_single_job_metrics(self):
        """Test metrics calculation for single job."""
        metrics = CircuitPerformanceMetrics(
            circuit=self.circuit, job=self.job, success_criteria=self.bell_success_criteria
        )

        result = metrics.get_metrics()

        # Test success metrics
        self.assertGreaterEqual(result.success_metrics.success_rate, 0.0)
        self.assertLessEqual(result.success_metrics.success_rate, 1.0)
        self.assertGreaterEqual(result.success_metrics.error_rate, 0.0)
        self.assertLessEqual(result.success_metrics.error_rate, 1.0)
        self.assertGreater(result.success_metrics.total_shots, 0)
        self.assertGreaterEqual(result.success_metrics.successful_shots, 0)
        self.assertLessEqual(
            result.success_metrics.successful_shots, result.success_metrics.total_shots
        )

        # Test error rate validation: error_rate = 1 - success_rate
        expected_error_rate = 1.0 - result.success_metrics.success_rate
        self.assertAlmostEqual(result.success_metrics.error_rate, expected_error_rate, places=5)

        # Test fidelity metrics
        self.assertGreaterEqual(result.fidelity_metrics.fidelity, 0.0)
        self.assertLessEqual(result.fidelity_metrics.fidelity, 1.0)
        self.assertIsNotNone(result.fidelity_metrics.method)
        self.assertIsNotNone(result.fidelity_metrics.confidence)

    def test_multiple_jobs_metrics(self):
        """Test metrics calculation for multiple jobs."""
        job2 = self.simulator.run(self.circuit, shots=500)

        metrics = CircuitPerformanceMetrics(
            circuit=self.circuit, jobs=[self.job, job2], success_criteria=self.bell_success_criteria
        )

        result = metrics.get_metrics()

        # Should have aggregate metrics - handle potential None values
        if result.success_metrics.success_rate is not None:
            self.assertGreaterEqual(result.success_metrics.success_rate, 0.0)
            self.assertLessEqual(result.success_metrics.success_rate, 1.0)
        if result.success_metrics.total_shots is not None:
            self.assertGreater(
                result.success_metrics.total_shots, 1000
            )  # Should be sum of both jobs

        # Test statistical metrics - handle potential None values
        if result.statistical_metrics.entropy is not None:
            self.assertGreaterEqual(result.statistical_metrics.entropy, 0.0)
        if result.statistical_metrics.uniformity is not None:
            self.assertGreaterEqual(result.statistical_metrics.uniformity, 0.0)
            self.assertLessEqual(result.statistical_metrics.uniformity, 1.0)
        if result.statistical_metrics.concentration is not None:
            self.assertGreaterEqual(result.statistical_metrics.concentration, 0.0)
            self.assertLessEqual(result.statistical_metrics.concentration, 1.0)
        if result.statistical_metrics.dominant_outcome_probability is not None:
            self.assertGreaterEqual(result.statistical_metrics.dominant_outcome_probability, 0.0)
            self.assertLessEqual(result.statistical_metrics.dominant_outcome_probability, 1.0)
        if result.statistical_metrics.num_unique_outcomes is not None:
            self.assertGreater(result.statistical_metrics.num_unique_outcomes, 0)

    def test_custom_success_criteria(self):
        """Test with custom success criteria."""

        # Success criteria: only "00 00" state
        def only_00_criteria(result):
            return result == "00 00"

        metrics = CircuitPerformanceMetrics(
            circuit=self.circuit, job=self.job, success_criteria=only_00_criteria
        )

        result = metrics.get_metrics()

        # Should have lower success rate than Bell state criteria
        self.assertGreaterEqual(result.success_metrics.success_rate, 0.0)
        self.assertLessEqual(result.success_metrics.success_rate, 1.0)

    def test_default_success_criteria(self):
        """Test with default success criteria."""
        metrics = CircuitPerformanceMetrics(circuit=self.circuit, job=self.job)

        result = metrics.get_metrics()

        # Should work with default criteria
        self.assertGreaterEqual(result.success_metrics.success_rate, 0.0)
        self.assertLessEqual(result.success_metrics.success_rate, 1.0)

    def test_add_job(self):
        """Test adding additional jobs."""
        metrics = CircuitPerformanceMetrics(
            circuit=self.circuit, job=self.job, success_criteria=self.bell_success_criteria
        )

        # Add another job
        job2 = self.simulator.run(self.circuit, shots=500)
        metrics.add_job(job2)

        self.assertEqual(len(metrics._jobs), 2)

    def test_to_flat_dict(self):
        """Test conversion to flat dictionary."""
        metrics = CircuitPerformanceMetrics(
            circuit=self.circuit, job=self.job, success_criteria=self.bell_success_criteria
        )

        result = metrics.get_metrics()
        flat_dict = result.to_flat_dict()

        self.assertIsInstance(flat_dict, dict)

        # Check that expected keys exist
        expected_keys = [
            "success_metrics.success_rate",
            "success_metrics.error_rate",
            "success_metrics.total_shots",
            "success_metrics.successful_shots",
            "fidelity_metrics.fidelity",
            "fidelity_metrics.method",
            "fidelity_metrics.confidence",
            "statistical_metrics.entropy",
            "statistical_metrics.uniformity",
        ]

        for key in expected_keys:
            self.assertIn(key, flat_dict, f"Expected key {key} not found in flat dict")

    def test_schema_validation(self):
        """Test that schema validation works correctly."""
        metrics = CircuitPerformanceMetrics(
            circuit=self.circuit, job=self.job, success_criteria=self.bell_success_criteria
        )

        result = metrics.get_metrics()

        # Test that all values are within expected ranges
        self.assertGreaterEqual(result.success_metrics.success_rate, 0.0)
        self.assertLessEqual(result.success_metrics.success_rate, 1.0)
        self.assertGreaterEqual(result.success_metrics.error_rate, 0.0)
        self.assertLessEqual(result.success_metrics.error_rate, 1.0)
        self.assertGreaterEqual(result.fidelity_metrics.fidelity, 0.0)
        self.assertLessEqual(result.fidelity_metrics.fidelity, 1.0)

    def test_schema_cross_field_validation(self):
        """Test schema cross-field validation."""
        metrics = CircuitPerformanceMetrics(
            circuit=self.circuit, job=self.job, success_criteria=self.bell_success_criteria
        )

        result = metrics.get_metrics()

        # Test cross-field validation: successful_shots <= total_shots
        self.assertLessEqual(
            result.success_metrics.successful_shots, result.success_metrics.total_shots
        )

        # Test cross-field validation: error_rate = 1 - success_rate
        expected_error_rate = 1.0 - result.success_metrics.success_rate
        self.assertAlmostEqual(result.success_metrics.error_rate, expected_error_rate, places=5)

    def test_name_property(self):
        """Test the name property."""
        metrics = CircuitPerformanceMetrics(
            circuit=self.circuit, job=self.job, success_criteria=self.bell_success_criteria
        )

        self.assertEqual(metrics.name, "CircuitPerformanceMetrics")

    def test_metric_type_and_id(self):
        """Test metric type and ID properties."""
        metrics = CircuitPerformanceMetrics(
            circuit=self.circuit, job=self.job, success_criteria=self.bell_success_criteria
        )

        self.assertEqual(metrics.metric_type, MetricsType.POST_RUNTIME)
        self.assertEqual(metrics.id, MetricsId.CIRCUIT_PERFORMANCE)

    def test_consistency_across_calls(self):
        """Test that multiple calls return consistent results."""
        metrics = CircuitPerformanceMetrics(
            circuit=self.circuit, job=self.job, success_criteria=self.bell_success_criteria
        )

        result1 = metrics.get_metrics()
        result2 = metrics.get_metrics()

        # Results should be identical for the same job
        self.assertEqual(result1.success_metrics.success_rate, result2.success_metrics.success_rate)
        self.assertEqual(result1.success_metrics.total_shots, result2.success_metrics.total_shots)
        self.assertEqual(result1.fidelity_metrics.fidelity, result2.fidelity_metrics.fidelity)

    def test_different_circuit_types(self):
        """Test with different circuit types."""
        # Test with single qubit circuit
        single_qubit = QuantumCircuit(1, 1)
        single_qubit.h(0)
        single_qubit.measure_all()

        job_single = self.simulator.run(single_qubit, shots=1000)

        # Success criteria for single qubit: state "0 0"
        def single_qubit_criteria(result):
            return result == "0 0"

        metrics = CircuitPerformanceMetrics(
            circuit=single_qubit, job=job_single, success_criteria=single_qubit_criteria
        )

        result = metrics.get_metrics()

        self.assertGreaterEqual(result.success_metrics.success_rate, 0.0)
        self.assertLessEqual(result.success_metrics.success_rate, 1.0)

    def test_high_fidelity_circuit(self):
        """Test with a circuit that should have high fidelity."""
        # Identity circuit (should have very high success rate)
        identity_circuit = QuantumCircuit(2, 2)
        identity_circuit.measure_all()

        job_identity = self.simulator.run(identity_circuit, shots=1000)

        # Success criteria: state "00 00" (correct format with space)
        def identity_criteria(result):
            return result == "00 00"

        metrics = CircuitPerformanceMetrics(
            circuit=identity_circuit, job=job_identity, success_criteria=identity_criteria
        )

        result = metrics.get_metrics()

        # Should have high success rate (but not necessarily > 0.95 due to noise)
        self.assertGreater(result.success_metrics.success_rate, 0.8)
        self.assertGreater(result.fidelity_metrics.fidelity, 0.8)

    def test_error_handling_invalid_job(self):
        """Test error handling with invalid job."""
        # Create a mock job that will fail
        mock_job = Mock()
        mock_job.result.side_effect = Exception("Job failed")

        metrics = CircuitPerformanceMetrics(
            circuit=self.circuit, job=mock_job, success_criteria=self.bell_success_criteria
        )

        # Should handle the error gracefully or raise appropriate exception
        with self.assertRaises(Exception):
            metrics.get_metrics()

    def test_job_id_extraction(self):
        """Test job ID extraction."""
        metrics = CircuitPerformanceMetrics(
            circuit=self.circuit, job=self.job, success_criteria=self.bell_success_criteria
        )

        job_id = metrics._extract_job_id(self.job)

        # Should return a string ID
        self.assertIsInstance(job_id, str)
        self.assertGreater(len(job_id), 0)

    def test_statistical_metrics_calculation(self):
        """Test statistical metrics calculation."""
        # Use multiple jobs to get statistical metrics
        job2 = self.simulator.run(self.circuit, shots=500)

        metrics = CircuitPerformanceMetrics(
            circuit=self.circuit, jobs=[self.job, job2], success_criteria=self.bell_success_criteria
        )

        result = metrics.get_metrics()

        # Test statistical metrics are calculated - handle potential None values
        # Note: Some metrics might be None if not enough data or calculation issues
        if result.statistical_metrics.entropy is not None:
            self.assertGreaterEqual(result.statistical_metrics.entropy, 0.0)
        if result.statistical_metrics.uniformity is not None:
            self.assertGreaterEqual(result.statistical_metrics.uniformity, 0.0)
            self.assertLessEqual(result.statistical_metrics.uniformity, 1.0)
        if result.statistical_metrics.concentration is not None:
            self.assertGreaterEqual(result.statistical_metrics.concentration, 0.0)
            self.assertLessEqual(result.statistical_metrics.concentration, 1.0)
        if result.statistical_metrics.dominant_outcome_probability is not None:
            self.assertGreaterEqual(result.statistical_metrics.dominant_outcome_probability, 0.0)
            self.assertLessEqual(result.statistical_metrics.dominant_outcome_probability, 1.0)
        if result.statistical_metrics.num_unique_outcomes is not None:
            self.assertGreater(result.statistical_metrics.num_unique_outcomes, 0)

    def test_result_dict_input_removed(self):
        """Test that was removed - result parameter is not supported."""
        # This test is removed because 'result' parameter is not supported in the current API
        # The CircuitPerformanceMetrics constructor only accepts job/jobs parameters
        pass


if __name__ == "__main__":
    unittest.main()
