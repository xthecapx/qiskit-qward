"""Behavioral tests for candidate screening and its separation from correctness."""

import itertools
import unittest

import numpy as np
from pydantic import ValidationError
from scipy.stats import binomtest

from qward import compute_output_screen
from qward.metrics import compute_dsr_profile, compute_dsr_with_flags
from qward.schemas.output_screening_schema import OutputScreeningSchema


class TestOutputScreening(unittest.TestCase):
    """Exercise finite-shot evidence, ties, sparse histograms, and input contracts."""

    def test_separation_mass_and_no_implicit_decision(self):
        counts = {"00": 12, "01": 4, "10": 3, "11": 1}
        result = compute_output_screen(counts)
        self.assertIsInstance(result, OutputScreeningSchema)
        self.assertEqual(result.shots, 20)
        self.assertEqual(result.leading_outcome, "00")
        self.assertEqual(result.leading_outcomes, ["00"])
        self.assertTrue(result.has_unique_peak)
        self.assertEqual((result.top_count, result.runner_up_count), (12, 4))
        self.assertEqual(result.top_probability, 0.6)
        self.assertEqual(result.runner_up_probability, 0.2)
        self.assertEqual(result.top_two_contrast, 0.5)
        self.assertEqual(result.pair_probability, 0.8)
        self.assertEqual(result.absolute_gap, 0.4)
        self.assertIsNone(result.significance_level)
        self.assertIsNone(result.rank_verified)
        self.assertNotIn("rank_verified", result.to_flat_dict())
        self.assertEqual(counts, {"00": 12, "01": 4, "10": 3, "11": 1})

    def test_identical_contrast_different_evidence(self):
        small = compute_output_screen({"00": 12, "01": 4}, significance_level=0.05)
        large = compute_output_screen({"00": 120, "01": 40}, significance_level=0.05)
        self.assertEqual(small.top_two_contrast, large.top_two_contrast)
        self.assertAlmostEqual(small.rank_pvalue, 0.076812744140625)
        self.assertFalse(small.rank_verified)
        self.assertTrue(large.rank_verified)

    def test_ties_do_not_choose_a_candidate(self):
        items = [("00", 50), ("01", 50), ("11", 50), ("10", 0)]
        for order in itertools.permutations(items):
            with self.subTest(order=order):
                result = compute_output_screen(dict(order), significance_level=0.05)
                self.assertIsNone(result.leading_outcome)
                self.assertEqual(result.leading_outcomes, ["00", "01", "11"])
                self.assertFalse(result.has_unique_peak)
                self.assertEqual(result.top_two_contrast, 0.0)
                self.assertEqual(result.rank_pvalue, 1.0)
                self.assertFalse(result.rank_verified)

    def test_top_counts_independent_of_insertion_order(self):
        items = [("00", 4), ("01", 4), ("10", 12), ("11", 0)]
        for order in itertools.permutations(items):
            result = compute_output_screen(dict(order))
            self.assertEqual(result.leading_outcome, "10")
            self.assertEqual((result.top_count, result.runner_up_count), (12, 4))

    def test_single_observed_state_requires_evidence(self):
        for shots, verified in [(1, False), (5, False), (6, True)]:
            result = compute_output_screen({"1" * 100: shots}, significance_level=0.05)
            self.assertEqual(result.runner_up_count, 0)
            self.assertEqual(result.top_two_contrast, 1.0)
            self.assertAlmostEqual(result.rank_pvalue, min(1, 2 ** (1 - shots)))
            self.assertEqual(result.rank_verified, verified)

    def test_zero_bins_do_not_change_results(self):
        self.assertEqual(
            compute_output_screen({"01": 6}),
            compute_output_screen({"00": 0, "01": 6, "10": 0}),
        )

    def test_exact_two_sided_values(self):
        # Includes ties, tiny samples, sparse tails, and a large balanced sample.
        for top, second in [(2, 1), (5, 0), (12, 4), (15, 5), (101, 100), (600, 400)]:
            with self.subTest(top=top, second=second):
                result = compute_output_screen({"0": top, "1": second})
                reference = binomtest(top, top + second, p=0.5, alternative="two-sided")
                self.assertAlmostEqual(result.rank_pvalue, reference.pvalue, places=14)

    def test_decision_uses_unrounded_pvalue_and_explicit_level(self):
        result = compute_output_screen({"0": 5}, significance_level=0.0625)
        self.assertTrue(result.rank_verified)
        self.assertEqual(result.significance_level, 0.0625)
        result = compute_output_screen({"0": 5}, significance_level=0.0624999)
        self.assertFalse(result.rank_verified)

    def test_numpy_integer_counts(self):
        result = compute_output_screen({"0": np.int64(12), "1": np.int32(4)})
        self.assertEqual(result.shots, 16)

    def test_rejects_invalid_counts(self):
        cases = [
            {},
            {"0": 0},
            {"0": -1},
            {"0": 1.0},
            {"0": 0.5},
            {"0": True},
            {"0": float("nan")},
            {"0": float("inf")},
            {"0": "12"},
            {"0": 2 + 0j},
            {"": 10},
            {0: 10},
        ]
        for counts in cases:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                compute_output_screen(counts)

    def test_rejects_invalid_significance_level(self):
        for level in [0, 1, -0.1, 2, True, "0.05", float("nan"), float("inf"), 0.05j]:
            with self.subTest(level=level), self.assertRaises(ValueError):
                compute_output_screen({"0": 12, "1": 4}, significance_level=level)

    def test_schema_serialization_and_constraints(self):
        result = compute_output_screen({"0": 12, "1": 4}, significance_level=0.05)
        self.assertEqual(
            OutputScreeningSchema.model_validate_json(result.model_dump_json()), result
        )
        for field, value in [
            ("shots", 0),
            ("rank_pvalue", 1.1),
            ("top_two_contrast", -0.1),
            ("significance_level", 0),
            ("leading_outcomes", []),
        ]:
            with self.subTest(field=field), self.assertRaises(ValidationError):
                OutputScreeningSchema(**(result.model_dump() | {field: value}))


class TestScreeningAndApplicationVerification(unittest.TestCase):
    """Preserve the distinction between a reliable mode and a useful answer."""

    def test_wrong_dominant_peak_still_requires_verification(self):
        counts = {"00": 800, "01": 100, "10": 80, "11": 44}
        screen = compute_output_screen(counts, significance_level=0.05)
        expected = {"01"}
        self.assertTrue(screen.rank_verified)
        self.assertNotIn(screen.leading_outcome, expected)
        profile = compute_dsr_profile(counts, expected)
        self.assertEqual(profile.dsr_michelson, 0)
        self.assertTrue(profile.peak_mismatch)
        self.assertGreater(profile.success_rate, 0)

    def test_two_good_peaks_are_not_a_failed_algorithm(self):
        counts = {"00": 478, "11": 446, "01": 24, "10": 24}
        screen = compute_output_screen(counts, significance_level=0.05)
        profile = compute_dsr_profile(counts, {"00", "11"})
        self.assertLess(screen.top_two_contrast, 0.04)
        self.assertFalse(screen.rank_verified)
        self.assertGreater(profile.dsr_michelson, 0.9)
        self.assertEqual(profile.target_coverage, 1)
        self.assertEqual(profile.min_target_count, 446)

    def test_missing_target_is_visible_despite_positive_dsr(self):
        for counts in [{"00": 15, "01": 5}, {"00": 15, "01": 5, "11": 0}]:
            profile = compute_dsr_profile(counts, {"00", "11"})
            self.assertEqual(profile.dsr_michelson, 0.2)
            self.assertEqual(profile.target_coverage, 0.5)
            self.assertEqual(profile.min_target_count, 0)

    def test_target_diagnostics_without_michelson(self):
        profile = compute_dsr_profile({"00": 100}, {"01", "11"}, include_michelson=False)
        self.assertEqual(profile.target_coverage, 0)
        self.assertEqual(profile.min_target_count, 0)
        self.assertIsNone(profile.dsr_michelson)

    def test_legacy_mismatch_allows_ties_screen_exposes_them(self):
        counts = {"00": 50, "01": 50}
        self.assertEqual(compute_dsr_with_flags(counts, {"01"}), (0, False))
        screen = compute_output_screen(counts)
        self.assertFalse(screen.has_unique_peak)
        self.assertIsNone(screen.leading_outcome)


if __name__ == "__main__":
    unittest.main()
