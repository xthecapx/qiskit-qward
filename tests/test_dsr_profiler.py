"""Tests for the DSR evaluation profile (DSRProfiler and pure compute functions)."""

import math
import unittest

from qward.metrics.differential_success_rate import (
    DSRProfiler,
    compute_chance_baseline,
    compute_chance_corrected_success,
    compute_coarse_hellinger_distance,
    compute_coarse_hellinger_fidelity,
    compute_coarse_tvd,
    compute_coarse_tvd_similarity,
    compute_dsr,
    compute_dsr_profile,
    compute_dsr_with_flags,
    compute_success_rate,
)
from qward.schemas.dsr_profile_schema import DSRProfileSchema


class TestSuccessRate(unittest.TestCase):
    def test_basic(self):
        counts = {"01": 40, "00": 20, "10": 20, "11": 20}
        self.assertAlmostEqual(compute_success_rate(counts, {"01"}), 0.4)

    def test_multi_target(self):
        counts = {"00": 40, "11": 30, "01": 20, "10": 10}
        self.assertAlmostEqual(compute_success_rate(counts, {"00", "11"}), 0.7)

    def test_missing_target_counts_zero(self):
        """A target that is never observed counts as zero, not an error."""
        counts = {"00": 100}
        self.assertAlmostEqual(compute_success_rate(counts, {"01"}), 0.0)


class TestChanceBaseline(unittest.TestCase):
    def test_single_target(self):
        self.assertAlmostEqual(compute_chance_baseline(1, 2), 0.25)

    def test_multi_target(self):
        self.assertAlmostEqual(compute_chance_baseline(3, 5), 3 / 32)

    def test_full_space(self):
        self.assertAlmostEqual(compute_chance_baseline(4, 2), 1.0)


class TestChanceCorrectedSuccess(unittest.TestCase):
    def test_perfect(self):
        counts = {"01": 100}
        score = compute_chance_corrected_success(counts, {"01"}, num_measured_qubits=2)
        self.assertAlmostEqual(score, 1.0)

    def test_exactly_at_chance_is_zero(self):
        # m=2, K=1 -> b=0.25; success rate exactly 0.25 -> score 0
        counts = {"01": 25, "00": 25, "10": 25, "11": 25}
        score = compute_chance_corrected_success(counts, {"01"}, num_measured_qubits=2)
        self.assertAlmostEqual(score, 0.0)

    def test_below_chance_clips_to_zero(self):
        counts = {"01": 5, "00": 50, "10": 40, "11": 5}
        score = compute_chance_corrected_success(counts, {"01"}, num_measured_qubits=2)
        self.assertEqual(score, 0.0)

    def test_worked_example(self):
        # p_E = 0.40, K=3, m=5 -> b = 3/32 = 0.09375
        counts = {}
        # Build counts totalling 100 shots with 40 on three expected outcomes combined
        counts = {"00001": 20, "00010": 20, "00100": 60 - 40}
        # simpler: directly construct via known counts summing to p_E=0.4 over 100 shots
        counts = {"00001": 13, "00010": 13, "00100": 14, "11111": 60}
        expected = {"00001", "00010", "00100"}
        p_e = sum(counts[o] for o in expected) / sum(counts.values())
        self.assertAlmostEqual(p_e, 0.40)
        score = compute_chance_corrected_success(counts, expected, num_measured_qubits=5)
        b = 3 / 32
        expected_score = (0.40 - b) / (1 - b)
        self.assertAlmostEqual(score, expected_score, places=6)

    def test_infers_num_measured_qubits_from_counts(self):
        counts = {"01": 40, "00": 20, "10": 20, "11": 20}
        score = compute_chance_corrected_success(counts, {"01"})
        # m inferred as 2 -> b = 0.25
        expected_score = (0.4 - 0.25) / (1 - 0.25)
        self.assertAlmostEqual(score, expected_score, places=6)

    def test_inconsistent_bitstring_lengths_require_explicit_m(self):
        counts = {"01": 40, "1": 60}
        with self.assertRaises(ValueError):
            compute_chance_corrected_success(counts, {"01"})
        # Providing m explicitly resolves it.
        score = compute_chance_corrected_success(counts, {"01"}, num_measured_qubits=2)
        self.assertIsInstance(score, float)

    def test_k_equals_full_space_returns_one(self):
        # K = 2**m: every outcome is "expected", so success is guaranteed.
        counts = {"00": 50, "01": 50}
        score = compute_chance_corrected_success(
            counts, {"00", "01"}, num_measured_qubits=1
        )
        self.assertAlmostEqual(score, 1.0)


class TestCoarseMetricsK1Collapse(unittest.TestCase):
    """For K=1, coarse TVD similarity and coarse Hellinger fidelity must
    collapse exactly to the raw success rate (see math.md)."""

    def test_collapse_various_success_levels(self):
        for counts in [
            {"01": 100},
            {"01": 40, "00": 20, "10": 20, "11": 20},
            {"01": 3, "00": 97},
            {"01": 25, "00": 25, "10": 25, "11": 25},
        ]:
            success = compute_success_rate(counts, {"01"})
            tvd_sim = compute_coarse_tvd_similarity(counts, {"01"})
            hf = compute_coarse_hellinger_fidelity(counts, {"01"})
            with self.subTest(counts=counts):
                self.assertAlmostEqual(tvd_sim, success, places=9)
                self.assertAlmostEqual(hf, success, places=9)

    def test_coarse_tvd_is_one_minus_success_at_k1(self):
        counts = {"01": 30, "00": 70}
        tvd = compute_coarse_tvd(counts, {"01"})
        success = compute_success_rate(counts, {"01"})
        self.assertAlmostEqual(tvd, 1.0 - success, places=9)


class TestCoarseMetricsMultiTarget(unittest.TestCase):
    def test_uniform_weights_perfectly_balanced(self):
        # K=2, all mass split evenly between the two expected outcomes.
        counts = {"00": 50, "11": 50}
        tvd = compute_coarse_tvd(counts, {"00", "11"})
        hf = compute_coarse_hellinger_fidelity(counts, {"00", "11"})
        self.assertAlmostEqual(tvd, 0.0, places=9)
        self.assertAlmostEqual(hf, 1.0, places=9)

    def test_uniform_weights_imbalanced_within_e_penalizes_similarity(self):
        # Same total success (1.0) but all mass on one of two expected
        # outcomes: coarse similarity must be strictly less than success.
        counts_balanced = {"00": 50, "11": 50}
        counts_imbalanced = {"00": 100, "11": 0}
        success_balanced = compute_success_rate(counts_balanced, {"00", "11"})
        success_imbalanced = compute_success_rate(counts_imbalanced, {"00", "11"})
        self.assertAlmostEqual(success_balanced, success_imbalanced)

        sim_balanced = compute_coarse_tvd_similarity(counts_balanced, {"00", "11"})
        sim_imbalanced = compute_coarse_tvd_similarity(counts_imbalanced, {"00", "11"})
        self.assertGreater(sim_balanced, sim_imbalanced)
        self.assertAlmostEqual(sim_balanced, 1.0, places=9)
        self.assertLess(sim_imbalanced, 1.0)

    def test_non_uniform_expected_weights(self):
        # Non-uniform reference distribution (e.g. QFT period-detection peaks).
        counts = {"00": 75, "11": 25}
        weights = {"00": 0.75, "11": 0.25}
        tvd = compute_coarse_tvd(counts, {"00", "11"}, expected_weights=weights)
        hf = compute_coarse_hellinger_fidelity(counts, {"00", "11"}, expected_weights=weights)
        self.assertAlmostEqual(tvd, 0.0, places=9)
        self.assertAlmostEqual(hf, 1.0, places=9)

    def test_missing_target_treated_as_zero_count(self):
        # One expected outcome never observed at all.
        counts = {"00": 100}
        tvd = compute_coarse_tvd(counts, {"00", "11"})
        # obs = {"00": 1.0, "11": 0.0, other: 0.0}; ideal = {"00": 0.5, "11": 0.5, other: 0}
        self.assertAlmostEqual(tvd, 0.5, places=9)

    def test_no_other_bin_when_all_mass_in_e(self):
        # All counts fall inside E: the "other" bin is present but zero.
        counts = {"00": 60, "11": 40}
        observed_other = 1.0 - compute_success_rate(counts, {"00", "11"})
        self.assertAlmostEqual(observed_other, 0.0)
        tvd = compute_coarse_tvd(counts, {"00", "11"})
        self.assertGreater(tvd, 0.0)  # imbalance between 60/40 vs uniform 50/50
        self.assertLess(tvd, 0.3)

    def test_invalid_weights_wrong_keys(self):
        counts = {"00": 50, "11": 50}
        with self.assertRaises(ValueError):
            compute_coarse_tvd(counts, {"00", "11"}, expected_weights={"00": 1.0})

    def test_invalid_weights_dont_sum_to_one(self):
        counts = {"00": 50, "11": 50}
        with self.assertRaises(ValueError):
            compute_coarse_tvd(
                counts, {"00", "11"}, expected_weights={"00": 0.5, "11": 0.6}
            )

    def test_invalid_weights_negative(self):
        counts = {"00": 50, "11": 50}
        with self.assertRaises(ValueError):
            compute_coarse_tvd(
                counts, {"00", "11"}, expected_weights={"00": 1.5, "11": -0.5}
            )


class TestCoarseHellingerDistanceMetric(unittest.TestCase):
    def test_zero_when_identical(self):
        counts = {"01": 100}
        self.assertAlmostEqual(compute_coarse_hellinger_distance(counts, {"01"}), 0.0, places=9)

    def test_positive_when_different(self):
        counts = {"01": 50, "00": 50}
        d = compute_coarse_hellinger_distance(counts, {"01"})
        self.assertGreater(d, 0.0)
        self.assertLessEqual(d, 1.0)


class TestMalformedInput(unittest.TestCase):
    def test_empty_counts(self):
        with self.assertRaises(ValueError):
            compute_success_rate({}, {"01"})

    def test_empty_expected_outcomes(self):
        with self.assertRaises(ValueError):
            compute_success_rate({"01": 10}, [])

    def test_negative_counts(self):
        with self.assertRaises(ValueError):
            compute_success_rate({"01": -1}, {"01"})

    def test_zero_total_counts(self):
        with self.assertRaises(ValueError):
            compute_coarse_tvd({"01": 0, "00": 0}, {"01"})


class TestDSRProfiler(unittest.TestCase):
    def test_profile_returns_schema(self):
        counts = {"01": 40, "00": 20, "10": 20, "11": 20}
        profile = DSRProfiler(counts, {"01"}, num_measured_qubits=2).profile()
        self.assertIsInstance(profile, DSRProfileSchema)
        self.assertEqual(profile.shots, 100)
        self.assertEqual(profile.num_measured_qubits, 2)
        self.assertAlmostEqual(profile.success_rate, 0.4)
        self.assertAlmostEqual(profile.chance_baseline, 0.25)
        self.assertAlmostEqual(profile.coarse_tvd_similarity, 0.4)
        self.assertAlmostEqual(profile.coarse_hellinger_fidelity, 0.4)

    def test_profile_includes_michelson_by_default(self):
        counts = {"01": 40, "00": 20, "10": 20, "11": 20}
        profile = DSRProfiler(counts, {"01"}).profile()
        self.assertIsNotNone(profile.dsr_michelson)
        self.assertIsNotNone(profile.peak_mismatch)

    def test_profile_can_exclude_michelson(self):
        counts = {"01": 40, "00": 20, "10": 20, "11": 20}
        profile = DSRProfiler(counts, {"01"}, include_michelson=False).profile()
        self.assertIsNone(profile.dsr_michelson)
        self.assertIsNone(profile.peak_mismatch)

    def test_default_weights_are_uniform(self):
        profiler = DSRProfiler({"00": 50, "11": 50}, {"00", "11"})
        self.assertAlmostEqual(profiler.expected_weights["00"], 0.5)
        self.assertAlmostEqual(profiler.expected_weights["11"], 0.5)

    def test_convenience_function_matches_class(self):
        counts = {"01": 40, "00": 20, "10": 20, "11": 20}
        via_class = DSRProfiler(counts, {"01"}, num_measured_qubits=2).profile()
        via_function = compute_dsr_profile(counts, {"01"}, num_measured_qubits=2)
        self.assertEqual(via_class.model_dump(), via_function.model_dump())

    def test_mismatched_expected_outcome_length_raises(self):
        counts = {"01": 40, "00": 20, "10": 20, "11": 20}
        with self.assertRaises(ValueError):
            DSRProfiler(counts, {"010"}, num_measured_qubits=2)

    def test_inconsistent_bitstring_lengths_without_explicit_m_raises(self):
        counts = {"01": 40, "1": 60}
        with self.assertRaises(ValueError):
            DSRProfiler(counts, {"01"})


class TestLegacyMichelsonRegression(unittest.TestCase):
    """Regression coverage: the optional Michelson peak-contrast layer must
    keep its historical behavior unchanged."""

    def test_example_from_doc(self):
        counts = {"01": 40, "00": 20, "10": 20, "11": 20}
        score = compute_dsr(counts, {"01"})
        self.assertAlmostEqual(score, 1 / 3, places=9)

    def test_uniform_distribution(self):
        counts = {"00": 25, "01": 25, "10": 25, "11": 25}
        score = compute_dsr(counts, {"01"})
        self.assertAlmostEqual(score, 0.0, places=9)

    def test_perfect(self):
        counts = {"01": 100}
        score = compute_dsr(counts, {"01"})
        self.assertAlmostEqual(score, 1.0, places=9)

    def test_peak_mismatch_flag(self):
        counts = {"00": 60, "01": 40, "10": 0, "11": 0}
        score, peak_mismatch = compute_dsr_with_flags(counts, {"01"})
        self.assertAlmostEqual(score, 0.0, places=9)
        self.assertTrue(peak_mismatch)


if __name__ == "__main__":
    unittest.main()
