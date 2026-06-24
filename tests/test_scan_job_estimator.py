"""Tests for scan_job / scan_batch Estimator detection and routing."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from qiskit import QuantumCircuit

from qward.scan._ibm import (
    detect_primitive_type_from_job,
    extract_estimator_from_job,
)


def _make_estimator_job(evs, stds=None):
    """Build a mock job that mimics IBM Runtime Estimator output."""
    data = MagicMock()
    data.evs = np.array(evs)
    if stds is not None:
        data.stds = np.array(stds)
    else:
        del data.stds

    pub_result = MagicMock()
    pub_result.data = data

    result = MagicMock()
    result.__getitem__ = lambda self, idx: pub_result
    result.__len__ = lambda self: 1

    job = MagicMock()
    job.result.return_value = result
    return job


def _make_sampler_job(counts):
    """Build a mock job that mimics IBM Runtime Sampler output."""
    bit_array = MagicMock()
    bit_array.get_counts.return_value = counts

    data = MagicMock(spec=[])
    data.c = bit_array

    pub_result = MagicMock()
    pub_result.data = data

    result = MagicMock()
    result.__getitem__ = lambda self, idx: pub_result
    result.__len__ = lambda self: 1

    job = MagicMock()
    job.result.return_value = result
    return job


class TestDetectPrimitiveType(unittest.TestCase):
    """Test detect_primitive_type_from_job."""

    def test_estimator_detected(self):
        job = _make_estimator_job([0.95, -0.02, 1.0])
        self.assertEqual(detect_primitive_type_from_job(job), "estimator")

    def test_sampler_detected(self):
        job = _make_sampler_job({"00": 500, "11": 500})
        self.assertEqual(detect_primitive_type_from_job(job), "sampler")

    def test_broken_job_defaults_sampler(self):
        job = MagicMock()
        job.result.side_effect = RuntimeError("job failed")
        self.assertEqual(detect_primitive_type_from_job(job), "sampler")


class TestExtractEstimatorFromJob(unittest.TestCase):
    """Test extract_estimator_from_job."""

    def test_extracts_evs_and_stds(self):
        job = _make_estimator_job([0.9, 0.8, -0.5], [0.01, 0.02, 0.03])
        evs, stds = extract_estimator_from_job(job)

        np.testing.assert_array_almost_equal(evs, [0.9, 0.8, -0.5])
        np.testing.assert_array_almost_equal(stds, [0.01, 0.02, 0.03])

    def test_extracts_evs_without_stds(self):
        job = _make_estimator_job([1.0, 0.0])
        evs, stds = extract_estimator_from_job(job)

        np.testing.assert_array_almost_equal(evs, [1.0, 0.0])
        self.assertIsNone(stds)

    def test_scalar_evs_becomes_1d(self):
        job = _make_estimator_job(0.95)
        evs, _ = extract_estimator_from_job(job)
        self.assertEqual(evs.shape, (1,))


class TestScanJobEstimatorRouting(unittest.TestCase):
    """Test scan_job routes Estimator results through scan_post correctly."""

    def _mock_scan_job_call(self, job, **kwargs):
        """Simulate scan_job logic without IBM service."""
        circuit = QuantumCircuit(4)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(2, 3)

        from qward.scan._core import scan_post, scan_pre

        results = scan_pre(circuit)

        ptype = detect_primitive_type_from_job(job)
        if ptype == "estimator":
            evs, stds = extract_estimator_from_job(job)
            post = scan_post(
                circuit,
                expectation_values=evs,
                standard_deviations=stds,
                ideal_expectation_values=kwargs.get("ideal_expectation_values"),
                observable_labels=kwargs.get("observable_labels"),
            )
            results.update(post)
        else:
            from qward.scan._ibm import extract_counts_from_job

            counts = extract_counts_from_job(job)
            if counts:
                post = scan_post(circuit, counts, target_state=kwargs.get("target_state"))
                results.update(post)

        return results, ptype

    def test_estimator_job_produces_estimator_schema(self):
        job = _make_estimator_job(
            [0.95, 0.90, 0.85, 0.80, 0.05, -0.75],
            [0.01, 0.02, 0.01, 0.03, 0.02, 0.01],
        )
        ideal = np.array([1.0, 1.0, 1.0, 1.0, 0.0, -1.0])
        labels = ["ZZZZ", "XXXX", "ZZII", "IIZZ", "XZXZ", "YYYY"]

        results, ptype = self._mock_scan_job_call(
            job, ideal_expectation_values=ideal, observable_labels=labels
        )

        self.assertEqual(ptype, "estimator")
        self.assertIn("FidelityMetrics", results)
        df = results["FidelityMetrics"]
        self.assertIn("mean_observable_fidelity", df.columns)
        self.assertIn("mean_snr", df.columns)
        self.assertIn("depolarization_factor", df.columns)

    def test_sampler_job_produces_fidelity_schema(self):
        job = _make_sampler_job({"0000": 450, "1111": 450, "0001": 50, "1110": 50})

        results, ptype = self._mock_scan_job_call(job, target_state="0000")

        self.assertEqual(ptype, "sampler")
        self.assertIn("FidelityMetrics", results)
        df = results["FidelityMetrics"]
        self.assertIn("dsr", df.columns)
        self.assertIn("success_rate", df.columns)

    def test_estimator_without_ideal_still_works(self):
        job = _make_estimator_job([0.9, -0.5], [0.01, 0.02])

        results, ptype = self._mock_scan_job_call(job)

        self.assertEqual(ptype, "estimator")
        df = results["FidelityMetrics"]
        self.assertIn("mean_success_probability", df.columns)
        self.assertNotIn("depolarization_factor", df.columns)


if __name__ == "__main__":
    unittest.main()
