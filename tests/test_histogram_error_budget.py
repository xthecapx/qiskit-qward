"""Exact error-budget checks without executing circuits or sampling shots."""

import math

import pytest
from scipy.stats import binom

from qward.metrics.histogram_assessment import binomial_prediction_check


def test_two_counts_require_a_background_prediction():
    compatible = binomial_prediction_check(2, 1000, 0.002)
    assert compatible.range_status == "within_range"
    assert compatible.predicted_count_lower == 0
    assert compatible.predicted_count_upper == 5
    assert compatible.upper_tail_pvalue == pytest.approx(0.5942650918886205)
    assert compatible.residual_shot_z == 0
    rare_background = binomial_prediction_check(2, 1000, 0.000001)
    assert rare_background.exceeds_upper_budget
    assert rare_background.upper_tail_pvalue < 1e-6


@pytest.mark.parametrize("probability", [0, 0.002, 0.1, 0.5, 0.9, 1])
def test_exact_envelopes_control_null_tail_probability(probability):
    shots, alpha = 100, 0.01
    check = binomial_prediction_check(0, shots, probability, alpha=alpha)
    support = range(shots + 1)
    outside = sum(
        binom.pmf(k, shots, probability)
        for k in support
        if k < check.predicted_count_lower or k > check.predicted_count_upper
    )
    exceeded = sum(
        binom.pmf(k, shots, probability) for k in support if k > check.upper_count_budget
    )
    assert outside <= alpha + 1e-12
    assert exceeded <= alpha + 1e-12


def test_zero_variance_prediction_is_tested_exactly_without_infinite_z():
    ideal = binomial_prediction_check(0, 1000, 0)
    violated = binomial_prediction_check(1, 1000, 0)
    assert ideal.residual_shot_z is None
    assert ideal.range_status == "within_range"
    assert violated.residual_shot_z is None
    assert violated.upper_tail_pvalue == 0
    assert violated.exceeds_upper_budget
    assert '"residual_shot_z":null' in violated.model_dump_json()
    all_events = binomial_prediction_check(1000, 1000, 1)
    assert all_events.range_status == "within_range"
    assert all_events.upper_count_budget == 1000


def test_sigma_uses_prediction_variance_and_retains_residual_direction():
    high = binomial_prediction_check(120, 1000, 0.1)
    low = binomial_prediction_check(80, 1000, 0.1)
    assert high.residual_shot_z == pytest.approx(0.02 / math.sqrt(0.09 / 1000))
    assert low.residual_shot_z == pytest.approx(-0.02 / math.sqrt(0.09 / 1000))
    assert high.observed_shot_standard_error != high.predicted_shot_standard_error
    # An observed zero count still has positive uncertainty under this prediction.
    zero = binomial_prediction_check(0, 1000, 0.1)
    assert zero.observed_shot_standard_error == 0
    assert zero.predicted_shot_standard_error > 0
    assert zero.range_status == "below_range"


@pytest.mark.parametrize(
    "args",
    [
        (1, 0, 0.1),
        (-1, 100, 0.1),
        (101, 100, 0.1),
        (1.5, 100, 0.1),
        (True, 100, 0.1),
        (1, True, 0.1),
        (1, 100, float("nan")),
        (1, 100, 1.01),
    ],
)
def test_invalid_counts_and_probabilities_are_rejected(args):
    with pytest.raises(ValueError):
        binomial_prediction_check(*args)
