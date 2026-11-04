"""Finite-shot histogram assessment against analytic task references."""

import math
from numbers import Integral
from typing import Dict, Literal, Mapping, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from scipy.stats import beta, binom

from qward.metrics.base_metric import MetricCalculator
from qward.metrics.circuit_error_prediction import circuit_hash, validate_distribution
from qward.metrics.output_screening import compute_output_screen
from qward.metrics.types import MetricsId, MetricsType
from qward.schemas.error_prediction_schema import ErrorPredictionSchema
from qward.schemas.histogram_assessment_schema import (
    BinomialPredictionCheckSchema,
    HistogramAssessmentSchema,
)


def binomial_prediction_check(
    count: int, shots: int, predicted_probability: float, *, alpha: float = 0.05
) -> BinomialPredictionCheckSchema:
    """Compare a prespecified event count with an independent fixed prediction.

    Counts are binomial only for independent shots with a stationary probability.
    The central interval allocates alpha/2 to each tail. The separate upper
    budget allocates alpha to its upper tail. Equality with a bound is accepted.
    The z-score uses variance under the prediction and is descriptive; decisions
    use exact count bounds, including rare events and probabilities zero or one.
    No supplied probability is assumed to include calibration/model uncertainty.
    """
    if (
        isinstance(count, bool)
        or isinstance(shots, bool)
        or not isinstance(count, Integral)
        or not isinstance(shots, Integral)
        or shots < 1
        or not 0 <= count <= shots
        or not 0 <= predicted_probability <= 1
        or not 0 < alpha < 1
    ):
        raise ValueError("Invalid event count, shots, predicted probability or alpha")
    observed = count / shots
    expected_se = math.sqrt(predicted_probability * (1 - predicted_probability) / shots)
    lower = int(binom.ppf(alpha / 2, shots, predicted_probability))
    upper = int(binom.isf(alpha / 2, shots, predicted_probability))
    budget = int(binom.isf(alpha, shots, predicted_probability))
    return BinomialPredictionCheckSchema(
        shots=shots,
        count=count,
        alpha=alpha,
        predicted_probability=predicted_probability,
        observed_probability=observed,
        observed_shot_standard_error=math.sqrt(observed * (1 - observed) / shots),
        predicted_shot_standard_error=expected_se,
        residual_shot_z=(observed - predicted_probability) / expected_se if expected_se else None,
        predicted_count_lower=lower,
        predicted_count_upper=upper,
        upper_count_budget=budget,
        upper_tail_pvalue=float(binom.sf(count - 1, shots, predicted_probability)),
        lower_tail_pvalue=float(binom.cdf(count, shots, predicted_probability)),
        range_status=(
            "below_range" if count < lower else "above_range" if count > upper else "within_range"
        ),
        exceeds_upper_budget=count > budget,
    )


def binomial_interval(successes: int, shots: int, alpha: float) -> Tuple[float, float]:
    """Two-sided Clopper–Pearson interval, including 0/N and N/N boundaries."""
    if shots < 1 or not 0 <= successes <= shots or not 0 < alpha < 1:
        raise ValueError("Invalid successes, shots or alpha")
    lower = float(beta.ppf(alpha / 2, successes, shots - successes + 1)) if successes else 0.0
    upper = (
        float(beta.ppf(1 - alpha / 2, successes + 1, shots - successes))
        if successes < shots
        else 1.0
    )
    return lower, upper


def minimum_binomial_shots(
    null_probability: float,
    alternative_probability: float,
    *,
    alpha: float = 0.05,
    power: float = 0.9,
    maximum: int = 1_000_000,
) -> Optional[int]:
    """First fixed sample size with required one-sided exact binomial power.

    Returns None if the alternative does not exceed the null or no size up to
    ``maximum`` attains power. No quantum simulation or experiment is performed.
    """
    if not 0 <= null_probability <= 1 or not 0 <= alternative_probability <= 1:
        raise ValueError("Probabilities must be in [0, 1]")
    if not 0 < alpha < 1 or not 0 < power < 1 or maximum < 1:
        raise ValueError("Invalid alpha, power or maximum")
    if alternative_probability <= null_probability:
        return None
    for start in range(1, maximum + 1, 4096):
        sizes = np.arange(start, min(start + 4096, maximum + 1))
        critical = binom.isf(alpha, sizes, null_probability) + 1
        powers = binom.sf(critical - 1, sizes, alternative_probability)
        selected = np.flatnonzero(powers >= power)
        if len(selected):
            return int(sizes[selected[0]])
    return None


ModelStatus = Literal["consistent", "mismatch", "insufficient_calibration", "not_applicable"]


def distribution_status(
    probabilities: Optional[Mapping[str, float]],
    counts: Mapping[str, int],
    shots: int,
    alpha: float,
    *,
    target_count: Optional[int] = None,
    target_probability: Optional[float] = None,
) -> ModelStatus:
    """Bonferroni-simultaneous bin intervals, covering unobserved bins too.

    Without a materialized distribution (over 12 bits), a deterministic
    reference falls back to the exact interval for its target bin only.
    """
    if probabilities is not None:
        for key, prob in probabilities.items():
            lo, hi = binomial_interval(counts.get(key, 0), shots, alpha / len(probabilities))
            if not lo <= prob <= hi:
                return "mismatch"
        return "consistent"
    if target_count is None or target_probability is None:
        return "insufficient_calibration"
    lo, hi = binomial_interval(target_count, shots, alpha)
    return "consistent" if lo <= target_probability <= hi else "mismatch"


def normalized_counts(counts: Mapping[str, int], width: int) -> Dict[str, int]:
    """Validate complete raw output-register counts, merging register whitespace."""
    clean: Dict[str, int] = {}
    for label, count in counts.items():
        if not isinstance(label, str) or isinstance(count, bool) or not isinstance(count, Integral):
            raise ValueError("Raw counts must map binary strings to integer shot counts")
        key = label.replace(" ", "")
        if len(key) != width or set(key) - {"0", "1"} or count < 0:
            raise ValueError("Invalid output-register counts or width")
        clean[key] = clean.get(key, 0) + int(count)
    if not clean or sum(clean.values()) < 1:
        raise ValueError("Counts must contain at least one shot")
    return clean


class HistogramAssessment(MetricCalculator):
    """Opt-in POST_RUNTIME assessment, with application truth provided explicitly.

    ``alpha`` is a per-assessment error allocation. Campaign code must allocate
    it over tests/intervals across executions. ``consistent`` means failure to
    detect mismatch; it does not certify the prediction model. Point references
    use exact binomial inference. Wide sparse references need intervals only
    on the labels with positive ideal probability.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        counts: Mapping[str, int],
        reference: Mapping[str, float],
        prediction: Optional[ErrorPredictionSchema] = None,
        alpha: float = 0.05,
    ):
        super().__init__(circuit)
        if not 0 < alpha < 1:
            raise ValueError("Invalid alpha")
        self._counts = normalized_counts(counts, circuit.num_clbits)
        self._reference = validate_distribution(reference, circuit.num_clbits)
        self._prediction = prediction
        if prediction is not None and prediction.output_width != circuit.num_clbits:
            raise ValueError("Prediction output width does not match the histogram")
        if prediction is not None and prediction.circuit_hash != circuit_hash(circuit):
            raise ValueError("Prediction belongs to a different ISA circuit")
        self._alpha = alpha

    def _get_metric_type(self) -> MetricsType:
        return MetricsType.POST_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        return MetricsId.HISTOGRAM_ASSESSMENT

    def is_ready(self) -> bool:
        return bool(self._counts)

    def get_metrics(self) -> HistogramAssessmentSchema:
        counts, reference, alpha = self._counts, self._reference, self._alpha
        shots, width = sum(counts.values()), self.circuit.num_clbits
        rank = compute_output_screen(counts, significance_level=alpha)
        observed = {key: count / shots for key, count in counts.items()}
        tvd = 0.5 * sum(
            abs(observed.get(key, 0) - reference.get(key, 0))
            for key in observed.keys() | reference.keys()
        )
        success = interval = None
        if len(reference) == 1:
            target = next(iter(reference))
            success = observed.get(target, 0)
            interval = binomial_interval(counts.get(target, 0), shots, alpha)
            bounds = (1 - interval[1], 1 - interval[0])
            method = "exact_clopper_pearson_deterministic_target"
        elif width > 12:
            # TVD(P, R) = sum(max(R[x] - P[x], 0)) on the support of R.
            # Simultaneous intervals there cover the full TVD, even if all
            # observed shots fall outside that support.
            lower, upper = 0.0, 0.0
            for key, p in reference.items():
                lo, hi = binomial_interval(counts.get(key, 0), shots, alpha / len(reference))
                lower += max(0.0, p - hi)
                upper += max(0.0, p - lo)
            bounds = (lower, min(1.0, upper))
            method = "tvd_deficit_bounds_from_simultaneous_reference_support_intervals"
        else:
            lower, upper = 0.0, 0.0
            for x in range(2**width):
                key = f"{x:0{width}b}"
                lo, hi = binomial_interval(counts.get(key, 0), shots, alpha / 2**width)
                p = reference.get(key, 0)
                lower += max(lo - p, p - hi, 0)
                upper += max(abs(lo - p), abs(hi - p))
            bounds = (max(0, lower / 2), min(1, upper / 2))
            method = "conservative_tvd_bounds_from_simultaneous_bonferroni_binomial_intervals"
        prediction = self._prediction
        residual = None
        error_budget = None
        m2_budget = None
        m2_status: ModelStatus = "not_applicable"
        model: ModelStatus = "not_applicable"
        if prediction is not None:
            target_count = (
                counts.get(next(iter(prediction.reference_probabilities)), 0)
                if len(prediction.reference_probabilities) == 1
                else None
            )
            model = distribution_status(
                prediction.mixture_probabilities,
                counts,
                shots,
                alpha,
                target_count=target_count,
                target_probability=(
                    prediction.effective_visibility
                    + (1 - prediction.effective_visibility) * math.ldexp(1.0, -width)
                    if prediction.effective_visibility is not None
                    else None
                ),
            )
            m2_status = distribution_status(
                prediction.m2_probabilities,
                counts,
                shots,
                alpha,
                target_count=target_count,
                target_probability=prediction.m2_success,
            )
            if prediction.m1_success is not None:
                model_target = next(iter(prediction.reference_probabilities))
                residual = observed.get(model_target, 0) - prediction.m1_success
                if len(reference) == 1 and prediction.reference_probabilities == reference:
                    error_budget = binomial_prediction_check(
                        shots - counts.get(model_target, 0),
                        shots,
                        1 - prediction.m1_success,
                        alpha=alpha,
                    )
            if (
                prediction.m2_success is not None
                and len(reference) == 1
                and prediction.reference_probabilities == reference
            ):
                target = next(iter(reference))
                m2_budget = binomial_prediction_check(
                    shots - counts.get(target, 0), shots, 1 - prediction.m2_success, alpha=alpha
                )
        p1, p2 = rank.top_probability, rank.runner_up_probability
        return HistogramAssessmentSchema(
            shots=shots,
            alpha=alpha,
            tvd=tvd,
            tvd_interval=bounds,
            success_rate=success,
            success_interval=interval,
            model_status=model,
            rank_status="verified" if rank.rank_verified else "unresolved",
            normalized_entropy=max(
                0, min(1, -sum(p * math.log2(p) for p in observed.values() if p) / width)
            ),
            gap_standard_error=math.sqrt(max(0, (p1 + p2 - (p1 - p2) ** 2) / shots)),
            rank_pvalue=rank.rank_pvalue,
            prediction_residual=residual,
            prediction_snapshot_hash=prediction.snapshot_hash if prediction else None,
            interval_method=method,
            error_budget=error_budget,
            m2_model_status=m2_status,
            m2_error_budget=m2_budget,
        )
