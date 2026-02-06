"""Tests for Differential Success Rate (DSR)."""

import unittest

from qward.metrics.differential_success_rate import (
    compute_dsr,
    compute_dsr_percent,
    compute_dsr_with_flags,
)


class TestDifferentialSuccessRate(unittest.TestCase):
    """Unit tests for DSR computations."""

    def test_example_from_doc(self):
        counts = {"01": 40, "00": 20, "10": 20, "11": 20}
        score = compute_dsr(counts, {"01"})
        self.assertAlmostEqual(score, 1 / 3, places=9)
        self.assertAlmostEqual(compute_dsr_percent(counts, {"01"}), 33.333333, places=5)

    def test_uniform_distribution(self):
        counts = {"00": 25, "01": 25, "10": 25, "11": 25}
        score = compute_dsr(counts, {"01"})
        self.assertAlmostEqual(score, 0.0, places=9)

    def test_near_uniform_distribution(self):
        counts = {"00": 25, "01": 26, "10": 25, "11": 24}
        score = compute_dsr(counts, {"01"})
        self.assertAlmostEqual(score, 0.01960784313725492, places=9)

    def test_strong_expected_peak(self):
        counts = {"00": 1, "01": 97, "10": 1, "11": 1}
        score = compute_dsr(counts, {"01"})
        self.assertAlmostEqual(score, 0.9795918367346939, places=9)

    def test_wrong_peak_dominates(self):
        counts = {"01": 10, "00": 30, "10": 30, "11": 30}
        score = compute_dsr(counts, {"01"})
        self.assertAlmostEqual(score, 0.0, places=9)

    def test_multiple_expected_outcomes(self):
        counts = {"00": 40, "11": 30, "01": 20, "10": 10}
        score = compute_dsr(counts, {"00", "11"})
        self.assertAlmostEqual(score, 0.2727272727, places=9)

    def test_multiple_expected_outcomes_baseline(self):
        counts = {"01": 40, "00": 20, "10": 20, "11": 20}
        score = compute_dsr(counts, {"01", "10"})
        self.assertAlmostEqual(score, 0.2, places=9)

    def test_multiple_expected_outcomes_near_uniform(self):
        counts = {"00": 25, "01": 26, "10": 25, "11": 24}
        score = compute_dsr(counts, {"01", "10"})
        self.assertAlmostEqual(score, 0.00990099009900991, places=9)

    def test_multiple_expected_outcomes_strong(self):
        counts = {"00": 1, "01": 97, "10": 1, "11": 1}
        score = compute_dsr(counts, {"01", "10"})
        self.assertAlmostEqual(score, 0.96, places=9)

    def test_multiple_expected_outcomes_strong(self):
        counts = {"00": 40, "11": 40, "01": 10, "10": 10}
        score = compute_dsr(counts, {"00", "11"})
        self.assertAlmostEqual(score, 0.6, places=9)

    def test_perfect(self):
        counts = {"01": 100}
        score = compute_dsr(counts, {"01"})
        self.assertAlmostEqual(score, 1.0, places=9)

    def test_peak_mismatch_flag(self):
        counts = {"00": 60, "01": 40, "10": 0, "11": 0}
        score, peak_mismatch = compute_dsr_with_flags(counts, {"01"})
        self.assertAlmostEqual(score, 0.0, places=9)
        self.assertTrue(peak_mismatch)

    def test_peak_mismatch_flag_with_tie(self):
        counts = {"00": 50, "01": 50, "10": 0, "11": 0}
        score, peak_mismatch = compute_dsr_with_flags(counts, {"01"})
        self.assertAlmostEqual(score, 0.0, places=9)
        self.assertFalse(peak_mismatch)

    def test_peak_mismatch_flag_all_max_wrong(self):
        counts = {"00": 50, "10": 50, "01": 0, "11": 0}
        score, peak_mismatch = compute_dsr_with_flags(counts, {"01", "11"})
        self.assertAlmostEqual(score, 0.0, places=9)
        self.assertTrue(peak_mismatch)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            compute_dsr({}, {"01"})
        with self.assertRaises(ValueError):
            compute_dsr({"01": 10}, [])
        with self.assertRaises(ValueError):
            compute_dsr({"01": -1}, {"01"})


if __name__ == "__main__":
    unittest.main()
