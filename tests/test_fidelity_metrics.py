"""Tests for FidelityMetrics class."""

import unittest

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qward.metrics.fidelity_metrics import FidelityMetrics
from qward.metrics.types import MetricsId, MetricsType
from qward.schemas.fidelity_schema import FidelitySchema


class TestFidelityMetricsInit(unittest.TestCase):
    """Test FidelityMetrics initialization."""

    def setUp(self):
        self.circuit = QuantumCircuit(2, 2)
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        self.circuit.measure_all()

        self.counts = {"00": 500, "11": 500}

    def test_init_with_counts(self):
        fm = FidelityMetrics(self.circuit, counts=self.counts)
        self.assertTrue(fm.is_ready())

    def test_init_with_job(self):
        sim = AerSimulator()
        job = sim.run(self.circuit, shots=1000)
        fm = FidelityMetrics(self.circuit, job=job)
        self.assertTrue(fm.is_ready())

    def test_init_no_counts_no_job(self):
        fm = FidelityMetrics(self.circuit)
        self.assertFalse(fm.is_ready())

    def test_init_both_counts_and_job_raises(self):
        sim = AerSimulator()
        job = sim.run(self.circuit, shots=100)
        with self.assertRaises(ValueError):
            FidelityMetrics(self.circuit, counts=self.counts, job=job)

    def test_target_state_sets_both(self):
        fm = FidelityMetrics(self.circuit, counts=self.counts, target_state="00")
        self.assertEqual(fm._expected_outcomes, ["00"])
        self.assertEqual(fm._target_histogram, {"00": 1.0})

    def test_metric_type(self):
        fm = FidelityMetrics(self.circuit, counts=self.counts)
        self.assertEqual(fm.metric_type, MetricsType.POST_RUNTIME)

    def test_metric_id(self):
        fm = FidelityMetrics(self.circuit, counts=self.counts)
        self.assertEqual(fm.id, MetricsId.FIDELITY)


class TestFidelityMetricsFromCounts(unittest.TestCase):
    """Test FidelityMetrics with raw counts (credential-free path)."""

    def setUp(self):
        self.circuit = QuantumCircuit(2, 2)
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        self.circuit.measure_all()

    def test_counts_with_target_state(self):
        counts = {"00": 900, "01": 30, "10": 20, "11": 50}
        fm = FidelityMetrics(self.circuit, counts=counts, target_state="00")
        schema = fm.get_metrics()

        self.assertIsInstance(schema, FidelitySchema)
        self.assertEqual(schema.shots, 1000)
        self.assertEqual(schema.unique_outcomes, 4)
        self.assertIsNotNone(schema.dsr)
        self.assertIsNotNone(schema.success_rate)
        self.assertIsNotNone(schema.hellinger_fidelity)
        self.assertIsNotNone(schema.tvd)
        self.assertIsNotNone(schema.tvd_fidelity)
        self.assertFalse(schema.peak_mismatch)

    def test_counts_with_expected_outcomes_only(self):
        counts = {"000": 800, "111": 200}
        fm = FidelityMetrics(self.circuit, counts=counts, expected_outcomes=["000"])
        schema = fm.get_metrics()

        self.assertIsNotNone(schema.dsr)
        self.assertIsNotNone(schema.success_rate)
        self.assertIsNone(schema.hellinger_fidelity)
        self.assertIsNone(schema.tvd)

    def test_counts_with_target_histogram_only(self):
        counts = {"00": 600, "11": 400}
        ideal = {"00": 0.5, "11": 0.5}
        fm = FidelityMetrics(self.circuit, counts=counts, target_histogram=ideal)
        schema = fm.get_metrics()

        self.assertIsNone(schema.dsr)
        self.assertIsNone(schema.success_rate)
        self.assertIsNotNone(schema.hellinger_fidelity)
        self.assertIsNotNone(schema.tvd)

    def test_dsr_perfect_result(self):
        counts = {"101": 1000}
        fm = FidelityMetrics(self.circuit, counts=counts, target_state="101")
        schema = fm.get_metrics()

        self.assertAlmostEqual(schema.dsr, 1.0)
        self.assertAlmostEqual(schema.success_rate, 1.0)
        self.assertAlmostEqual(schema.hellinger_fidelity, 1.0)
        self.assertAlmostEqual(schema.tvd, 0.0)
        self.assertAlmostEqual(schema.tvd_fidelity, 1.0)

    def test_dsr_zero_when_competitor_dominates(self):
        counts = {"00": 900, "11": 100}
        fm = FidelityMetrics(self.circuit, counts=counts, expected_outcomes=["11"])
        schema = fm.get_metrics()

        self.assertEqual(schema.dsr, 0.0)
        self.assertTrue(schema.peak_mismatch)

    def test_success_rate_calculation(self):
        counts = {"00": 700, "01": 100, "10": 100, "11": 100}
        fm = FidelityMetrics(self.circuit, counts=counts, expected_outcomes=["00", "11"])
        schema = fm.get_metrics()

        self.assertAlmostEqual(schema.success_rate, 0.8)

    def test_hellinger_fidelity_perfect(self):
        counts = {"00": 500, "11": 500}
        ideal = {"00": 0.5, "11": 0.5}
        fm = FidelityMetrics(self.circuit, counts=counts, target_histogram=ideal)
        schema = fm.get_metrics()

        self.assertAlmostEqual(schema.hellinger_fidelity, 1.0, places=4)
        self.assertAlmostEqual(schema.tvd, 0.0, places=4)

    def test_tvd_maximally_different(self):
        counts = {"00": 1000}
        ideal = {"11": 1.0}
        fm = FidelityMetrics(self.circuit, counts=counts, target_histogram=ideal)
        schema = fm.get_metrics()

        self.assertAlmostEqual(schema.tvd, 1.0)
        self.assertAlmostEqual(schema.tvd_fidelity, 0.0)
        self.assertAlmostEqual(schema.hellinger_fidelity, 0.0)

    def test_empty_counts(self):
        fm = FidelityMetrics(self.circuit, counts={})
        schema = fm.get_metrics()

        self.assertIsNone(schema.shots)
        self.assertIsNone(schema.dsr)

    def test_schema_validation_ranges(self):
        counts = {"00": 600, "01": 200, "10": 100, "11": 100}
        fm = FidelityMetrics(self.circuit, counts=counts, target_state="00")
        schema = fm.get_metrics()

        self.assertGreaterEqual(schema.dsr, 0.0)
        self.assertLessEqual(schema.dsr, 1.0)
        self.assertGreaterEqual(schema.success_rate, 0.0)
        self.assertLessEqual(schema.success_rate, 1.0)
        self.assertGreaterEqual(schema.hellinger_fidelity, 0.0)
        self.assertLessEqual(schema.hellinger_fidelity, 1.0)
        self.assertGreaterEqual(schema.tvd, 0.0)
        self.assertLessEqual(schema.tvd, 1.0)

    def test_to_flat_dict(self):
        counts = {"00": 900, "11": 100}
        fm = FidelityMetrics(self.circuit, counts=counts, target_state="00")
        schema = fm.get_metrics()
        flat = schema.to_flat_dict()

        self.assertIsInstance(flat, dict)
        self.assertIn("dsr", flat)
        self.assertIn("shots", flat)
        self.assertIn("hellinger_fidelity", flat)


class TestFidelityMetricsFromJob(unittest.TestCase):
    """Test FidelityMetrics with AerSimulator job."""

    def setUp(self):
        self.circuit = QuantumCircuit(2)
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        self.circuit.measure_all()

        self.sim = AerSimulator()
        self.job = self.sim.run(self.circuit, shots=4096)

    def test_job_extraction(self):
        fm = FidelityMetrics(self.circuit, job=self.job, expected_outcomes=["00", "11"])
        schema = fm.get_metrics()

        self.assertEqual(schema.shots, 4096)
        self.assertIsNotNone(schema.dsr)
        self.assertIsNotNone(schema.success_rate)
        self.assertGreater(schema.success_rate, 0.9)

    def test_job_with_target_histogram(self):
        ideal = {"00": 0.5, "11": 0.5}
        fm = FidelityMetrics(self.circuit, job=self.job, target_histogram=ideal)
        schema = fm.get_metrics()

        self.assertIsNotNone(schema.hellinger_fidelity)
        self.assertGreater(schema.hellinger_fidelity, 0.9)

    def test_job_with_target_state(self):
        fm = FidelityMetrics(self.circuit, job=self.job, target_state="00")
        schema = fm.get_metrics()

        self.assertIsNotNone(schema.dsr)
        self.assertIsNotNone(schema.hellinger_fidelity)


class TestFidelityMetricsSuccessCriteria(unittest.TestCase):
    """Test custom success criteria."""

    def setUp(self):
        self.circuit = QuantumCircuit(2, 2)
        self.circuit.x(0)
        self.circuit.x(1)
        self.circuit.measure_all()

    def test_custom_success_criteria(self):
        counts = {"11": 950, "00": 50}
        criteria = lambda s: s.replace(" ", "") == "11"
        fm = FidelityMetrics(self.circuit, counts=counts, success_criteria=criteria)
        schema = fm.get_metrics()

        self.assertAlmostEqual(schema.success_rate, 0.95)

    def test_expected_outcomes_takes_priority(self):
        counts = {"11": 950, "00": 50}
        criteria = lambda s: s == "00"
        fm = FidelityMetrics(
            self.circuit,
            counts=counts,
            expected_outcomes=["11"],
            success_criteria=criteria,
        )
        schema = fm.get_metrics()

        self.assertAlmostEqual(schema.success_rate, 0.95)


class TestFidelityMetricsMultiJob(unittest.TestCase):
    """Test FidelityMetrics multi-job support."""

    def setUp(self):
        self.circuit = QuantumCircuit(2)
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        self.circuit.measure_all()
        self.sim = AerSimulator()

    def test_init_with_jobs_list(self):
        job1 = self.sim.run(self.circuit, shots=1024)
        job2 = self.sim.run(self.circuit, shots=1024)
        fm = FidelityMetrics(self.circuit, jobs=[job1, job2], expected_outcomes=["00", "11"])

        self.assertTrue(fm.is_ready())
        self.assertEqual(len(fm._jobs), 2)

    def test_single_job_wraps_in_list(self):
        job = self.sim.run(self.circuit, shots=1024)
        fm = FidelityMetrics(self.circuit, job=job, expected_outcomes=["00", "11"])

        self.assertEqual(len(fm._jobs), 1)
        self.assertIs(fm._jobs[0], job)

    def test_job_and_jobs_no_duplicate(self):
        job1 = self.sim.run(self.circuit, shots=1024)
        job2 = self.sim.run(self.circuit, shots=1024)
        fm = FidelityMetrics(self.circuit, job=job1, jobs=[job1, job2], expected_outcomes=["00"])

        self.assertEqual(len(fm._jobs), 2)

    def test_add_job_single(self):
        job1 = self.sim.run(self.circuit, shots=1024)
        job2 = self.sim.run(self.circuit, shots=1024)
        fm = FidelityMetrics(self.circuit, job=job1, expected_outcomes=["00", "11"])
        fm.add_job(job2)

        self.assertEqual(len(fm._jobs), 2)

    def test_add_job_list(self):
        job1 = self.sim.run(self.circuit, shots=1024)
        job2 = self.sim.run(self.circuit, shots=1024)
        job3 = self.sim.run(self.circuit, shots=1024)
        fm = FidelityMetrics(self.circuit, job=job1, expected_outcomes=["00", "11"])
        fm.add_job([job2, job3])

        self.assertEqual(len(fm._jobs), 3)

    def test_add_job_no_duplicate(self):
        job1 = self.sim.run(self.circuit, shots=1024)
        fm = FidelityMetrics(self.circuit, job=job1, expected_outcomes=["00", "11"])
        fm.add_job(job1)

        self.assertEqual(len(fm._jobs), 1)

    def test_get_metrics_all_multi_job(self):
        job1 = self.sim.run(self.circuit, shots=4096)
        job2 = self.sim.run(self.circuit, shots=4096)
        job3 = self.sim.run(self.circuit, shots=4096)
        fm = FidelityMetrics(
            self.circuit,
            jobs=[job1, job2, job3],
            expected_outcomes=["00", "11"],
            target_state="00",
        )
        schemas = fm.get_metrics_all()

        self.assertEqual(len(schemas), 3)
        for s in schemas:
            self.assertIsInstance(s, FidelitySchema)
            self.assertIsNotNone(s.dsr)
            self.assertIsNotNone(s.success_rate)
            self.assertIsNotNone(s.hellinger_fidelity)
            self.assertGreater(s.shots, 0)

    def test_get_metrics_all_single_job(self):
        job = self.sim.run(self.circuit, shots=1024)
        fm = FidelityMetrics(self.circuit, job=job, expected_outcomes=["00", "11"])
        schemas = fm.get_metrics_all()

        self.assertEqual(len(schemas), 1)

    def test_get_metrics_all_counts_mode(self):
        counts = {"00": 500, "11": 500}
        fm = FidelityMetrics(self.circuit, counts=counts, target_state="00")
        schemas = fm.get_metrics_all()

        self.assertEqual(len(schemas), 1)
        self.assertIsNotNone(schemas[0].dsr)

    def test_get_metrics_backward_compat(self):
        """get_metrics() still returns single schema from first job."""
        job1 = self.sim.run(self.circuit, shots=4096)
        job2 = self.sim.run(self.circuit, shots=4096)
        fm = FidelityMetrics(
            self.circuit,
            jobs=[job1, job2],
            expected_outcomes=["00", "11"],
        )
        schema = fm.get_metrics()

        self.assertIsInstance(schema, FidelitySchema)
        self.assertIsNotNone(schema.dsr)

    def test_scanner_multi_job_dataframe(self):
        """Scanner creates multi-row DataFrame for multi-job FidelityMetrics."""
        from qward.scanner import Scanner

        job1 = self.sim.run(self.circuit, shots=4096)
        job2 = self.sim.run(self.circuit, shots=4096)
        fm = FidelityMetrics(
            self.circuit,
            jobs=[job1, job2],
            expected_outcomes=["00", "11"],
            target_state="00",
        )
        scanner = Scanner(circuit=self.circuit)
        scanner.add_strategy(fm)
        results = scanner.calculate_metrics()

        self.assertIn("FidelityMetrics", results)
        df = results["FidelityMetrics"]
        self.assertEqual(len(df), 2)
        self.assertIn("dsr", df.columns)
        self.assertIn("success_rate", df.columns)

    def test_counts_and_jobs_raises(self):
        job = self.sim.run(self.circuit, shots=1024)
        with self.assertRaises(ValueError):
            FidelityMetrics(
                self.circuit,
                counts={"00": 100},
                jobs=[job],
                expected_outcomes=["00"],
            )


if __name__ == "__main__":
    unittest.main()
