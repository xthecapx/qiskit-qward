"""
Differential Success Rate (DSR) and companion evaluation measures.

Histogram-free, task-level evaluation of quantum job outcomes against an
analytically known expected-outcome set ``E`` (size ``K = |E|``), computed
only from measurement counts -- never from a full ideal histogram over all
``2**m`` outcomes.

DSR itself is the clipped Michelson contrast returned by :func:`compute_dsr`
and :func:`compute_dsr_michelson`. It compares the mean expected peak with
the strongest competing peak and is a derived ordinal measure of censored
peak dominance.

For convenience, :class:`DSRProfiler` and :class:`DSRProfileSchema` report
DSR alongside companion measures. Those companions are not alternative DSR
definitions:

- ``success_rate`` and ``chance_corrected_success`` are unary measures.
- ``coarse_tvd`` and ``coarse_hellinger_distance`` are metrics on the
  coarse ``K + 1``-bin simplex.
- ``coarse_tvd_similarity`` / ``coarse_hellinger_fidelity``: agreement, on a
  coarse ``K + 1``-bin distribution (one bin per element of ``E`` plus a
  single "other" bin), between the observed counts and a task-reference
  distribution over ``E`` (uniform ``1/K`` by default, or explicit
  ``expected_weights``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Iterable, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # Only for static type checking; the real import is deferred to
    # DSRProfiler.profile() / compute_dsr_profile() to avoid importing
    # qward.schemas (pydantic) at module load time.
    from qward.schemas.dsr_profile_schema import DSRProfileSchema

# Sentinel key for the aggregated "everything not in E" bin. Chosen to be
# very unlikely to collide with a real bitstring.
_OTHER_BIN = "__other__"


# ---------------------------------------------------------------------------
# DSR and companion measures: shared validation helpers
# ---------------------------------------------------------------------------


def _normalize_expected(expected_outcomes: Iterable[str]) -> Set[str]:
    expected_set = set(expected_outcomes)
    if not expected_set:
        raise ValueError("expected_outcomes must not be empty")
    return expected_set


def _validate_counts(counts: Mapping[str, int]) -> float:
    if not counts:
        raise ValueError("counts must not be empty")
    total = 0
    for outcome, count in counts.items():
        if count < 0:
            raise ValueError(f"counts[{outcome!r}] must be non-negative")
        total += count
    if total <= 0:
        raise ValueError("counts must sum to a positive value")
    return float(total)


def _infer_num_measured_qubits(counts: Mapping[str, int]) -> Optional[int]:
    """Infer the number of measured qubits from bitstring lengths.

    Returns None if the observed bitstrings have inconsistent lengths (the
    caller must then supply ``num_measured_qubits`` explicitly).
    """
    lengths = {len(key) for key in counts}
    if len(lengths) == 1:
        return lengths.pop()
    return None


def _default_expected_weights(expected_set: Set[str]) -> Dict[str, float]:
    k = len(expected_set)
    return {outcome: 1.0 / k for outcome in expected_set}


def _validate_expected_weights(
    expected_weights: Optional[Mapping[str, float]],
    expected_set: Set[str],
) -> Dict[str, float]:
    """Validate (or default to uniform) the task-reference weights over ``E``.

    ``expected_weights`` is the ideal probability of each expected outcome
    *given that the algorithm succeeded* (e.g. non-uniform QFT
    period-detection peaks). Defaults to uniform ``1/K`` when omitted.
    """
    if expected_weights is None:
        return _default_expected_weights(expected_set)

    weights = dict(expected_weights)
    if set(weights.keys()) != expected_set:
        raise ValueError(
            "expected_weights keys must exactly match expected_outcomes "
            f"(got {sorted(weights.keys())}, expected {sorted(expected_set)})"
        )
    for outcome, weight in weights.items():
        if weight < 0:
            raise ValueError(f"expected_weights[{outcome!r}] must be non-negative")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"expected_weights must sum to 1.0 (got {total})")
    return weights


def _clip_zero_one(value: float) -> float:
    if value <= 0:
        return 0.0
    if value >= 1:
        return 1.0
    return value


def _coarse_observed_probs(
    counts: Mapping[str, int], expected_set: Set[str], total: float
) -> Dict[str, float]:
    """Coarse (K + 1)-bin observed distribution: one bin per outcome in
    ``E`` (missing/unobserved outcomes count as zero) plus one "other" bin
    aggregating every count not in ``E``."""
    probs = {outcome: counts.get(outcome, 0) / total for outcome in expected_set}
    probs[_OTHER_BIN] = max(0.0, 1.0 - sum(probs.values()))
    return probs


def _coarse_ideal_probs(expected_weights: Mapping[str, float]) -> Dict[str, float]:
    ideal = dict(expected_weights)
    ideal[_OTHER_BIN] = 0.0
    return ideal


def _total_variation_distance(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def _hellinger_bhattacharyya(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    keys = set(p) | set(q)
    return sum(math.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in keys)


# ---------------------------------------------------------------------------
# Companion measures: pure compute functions
# ---------------------------------------------------------------------------


def compute_success_rate(counts: Mapping[str, int], expected_outcomes: Iterable[str]) -> float:
    """Raw success rate: fraction of shots landing in ``E``."""
    expected_set = _normalize_expected(expected_outcomes)
    total = _validate_counts(counts)
    return sum(counts.get(outcome, 0) for outcome in expected_set) / total


def compute_chance_baseline(num_expected_outcomes: int, num_measured_qubits: int) -> float:
    """Random-guessing baseline ``b = K / 2**m``."""
    if num_expected_outcomes < 0:
        raise ValueError("num_expected_outcomes must be non-negative")
    if num_measured_qubits < 0:
        raise ValueError("num_measured_qubits must be non-negative")
    return num_expected_outcomes / (2**num_measured_qubits)


def compute_chance_corrected_success(
    counts: Mapping[str, int],
    expected_outcomes: Iterable[str],
    *,
    num_measured_qubits: Optional[int] = None,
) -> float:
    """Chance-corrected success: ``clip((p_E - b) / (1 - b), 0, 1)``.

    ``num_measured_qubits`` is inferred from the bitstring lengths in
    ``counts`` when not supplied explicitly (requires consistent lengths).
    """
    expected_set = _normalize_expected(expected_outcomes)
    total = _validate_counts(counts)
    m = num_measured_qubits
    if m is None:
        m = _infer_num_measured_qubits(counts)
        if m is None:
            raise ValueError(
                "num_measured_qubits must be provided explicitly when observed "
                "bitstrings have inconsistent lengths"
            )

    p_e = sum(counts.get(outcome, 0) for outcome in expected_set) / total
    b = compute_chance_baseline(len(expected_set), m)
    denom = 1.0 - b
    if denom <= 1e-12:
        # K >= 2**m: every possible outcome is "expected", so success is
        # trivially guaranteed and there is no random-chance distinction.
        return 1.0
    return _clip_zero_one((p_e - b) / denom)


def compute_coarse_tvd(
    counts: Mapping[str, int],
    expected_outcomes: Iterable[str],
    *,
    expected_weights: Optional[Mapping[str, float]] = None,
) -> float:
    """Coarse total-variation distance (a genuine metric) between the
    observed and task-reference distributions over the ``K + 1`` bins
    ``{E} u {other}``."""
    expected_set = _normalize_expected(expected_outcomes)
    total = _validate_counts(counts)
    weights = _validate_expected_weights(expected_weights, expected_set)
    observed = _coarse_observed_probs(counts, expected_set, total)
    ideal = _coarse_ideal_probs(weights)
    return _total_variation_distance(observed, ideal)


def compute_coarse_tvd_similarity(
    counts: Mapping[str, int],
    expected_outcomes: Iterable[str],
    *,
    expected_weights: Optional[Mapping[str, float]] = None,
) -> float:
    """``1 - coarse_tvd``: a "higher is better" score (not a metric)."""
    return 1.0 - compute_coarse_tvd(counts, expected_outcomes, expected_weights=expected_weights)


def compute_coarse_hellinger_distance(
    counts: Mapping[str, int],
    expected_outcomes: Iterable[str],
    *,
    expected_weights: Optional[Mapping[str, float]] = None,
) -> float:
    """Coarse Hellinger distance (a genuine metric): ``sqrt(1 - BC)``."""
    expected_set = _normalize_expected(expected_outcomes)
    total = _validate_counts(counts)
    weights = _validate_expected_weights(expected_weights, expected_set)
    observed = _coarse_observed_probs(counts, expected_set, total)
    ideal = _coarse_ideal_probs(weights)
    bc = _hellinger_bhattacharyya(observed, ideal)
    return math.sqrt(max(0.0, 1.0 - bc))


def compute_coarse_hellinger_fidelity(
    counts: Mapping[str, int],
    expected_outcomes: Iterable[str],
    *,
    expected_weights: Optional[Mapping[str, float]] = None,
) -> float:
    """Coarse Hellinger fidelity (a "higher is better" score, not a
    metric): ``BC**2``."""
    expected_set = _normalize_expected(expected_outcomes)
    total = _validate_counts(counts)
    weights = _validate_expected_weights(expected_weights, expected_set)
    observed = _coarse_observed_probs(counts, expected_set, total)
    ideal = _coarse_ideal_probs(weights)
    bc = _hellinger_bhattacharyya(observed, ideal)
    return min(1.0, bc**2)


# ---------------------------------------------------------------------------
# DSR and companion measures: compatibility facade
# ---------------------------------------------------------------------------


class DSRProfiler:
    """Report DSR and companion measures from measurement counts.

    The API calls the result a profile for compatibility. DSR is only the
    Michelson-contrast field; success rate, chance-corrected success, and the
    coarse distance/similarity fields are companion measures. All are
    computed from measurement counts and an analytically known expected-
    outcome set ``E`` (plus, optionally, non-uniform reference weights).

    Example:
        >>> profiler = DSRProfiler({"01": 40, "00": 20, "10": 20, "11": 20}, {"01"})
        >>> profile = profiler.profile()
        >>> profile.success_rate
        0.4
    """

    def __init__(
        self,
        counts: Mapping[str, int],
        expected_outcomes: Iterable[str],
        *,
        num_measured_qubits: Optional[int] = None,
        expected_weights: Optional[Mapping[str, float]] = None,
        include_michelson: bool = True,
    ) -> None:
        self._counts: Dict[str, int] = dict(counts)
        self._expected_set = _normalize_expected(expected_outcomes)
        self._total = _validate_counts(self._counts)

        m = num_measured_qubits
        if m is None:
            m = _infer_num_measured_qubits(self._counts)
            if m is None:
                raise ValueError(
                    "num_measured_qubits must be provided explicitly when observed "
                    "bitstrings have inconsistent lengths"
                )
        for outcome in self._expected_set:
            if len(outcome) != m:
                raise ValueError(
                    f"expected outcome {outcome!r} has length {len(outcome)}, "
                    f"but num_measured_qubits is {m}"
                )
        self._num_measured_qubits = m
        self._expected_weights = _validate_expected_weights(expected_weights, self._expected_set)
        self._include_michelson = include_michelson

    @property
    def num_measured_qubits(self) -> int:
        return self._num_measured_qubits

    @property
    def expected_weights(self) -> Dict[str, float]:
        return dict(self._expected_weights)

    def success_rate(self) -> float:
        return sum(self._counts.get(o, 0) for o in self._expected_set) / self._total

    def chance_baseline(self) -> float:
        return compute_chance_baseline(len(self._expected_set), self._num_measured_qubits)

    def chance_corrected_success(self) -> float:
        return compute_chance_corrected_success(
            self._counts, self._expected_set, num_measured_qubits=self._num_measured_qubits
        )

    def _coarse_probs(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        observed = _coarse_observed_probs(self._counts, self._expected_set, self._total)
        ideal = _coarse_ideal_probs(self._expected_weights)
        return observed, ideal

    def coarse_tvd(self) -> float:
        observed, ideal = self._coarse_probs()
        return _total_variation_distance(observed, ideal)

    def coarse_hellinger_distance(self) -> float:
        observed, ideal = self._coarse_probs()
        bc = _hellinger_bhattacharyya(observed, ideal)
        return math.sqrt(max(0.0, 1.0 - bc))

    def coarse_hellinger_fidelity(self) -> float:
        observed, ideal = self._coarse_probs()
        bc = _hellinger_bhattacharyya(observed, ideal)
        return min(1.0, bc**2)

    def profile(self) -> DSRProfileSchema:
        """Return DSR and its companion measures as a validated schema."""
        from qward.schemas.dsr_profile_schema import DSRProfileSchema

        coarse_tvd = self.coarse_tvd()
        coarse_hd = self.coarse_hellinger_distance()

        result: Dict[str, Any] = {
            "shots": int(self._total),
            "num_measured_qubits": self._num_measured_qubits,
            "expected_outcomes": sorted(self._expected_set),
            "expected_weights": dict(self._expected_weights),
            "num_expected_outcomes": len(self._expected_set),
            "success_rate": round(self.success_rate(), 6),
            "chance_baseline": round(self.chance_baseline(), 6),
            "chance_corrected_success": round(self.chance_corrected_success(), 6),
            "coarse_tvd": round(coarse_tvd, 6),
            "coarse_tvd_similarity": round(1.0 - coarse_tvd, 6),
            "coarse_hellinger_distance": round(coarse_hd, 6),
            "coarse_hellinger_fidelity": round(self.coarse_hellinger_fidelity(), 6),
        }

        if self._include_michelson:
            dsr_val, peak_mismatch = compute_dsr_with_flags(self._counts, self._expected_set)
            result["dsr_michelson"] = round(dsr_val, 6)
            result["peak_mismatch"] = peak_mismatch

        return DSRProfileSchema(**result)


def compute_dsr_profile(
    counts: Mapping[str, int],
    expected_outcomes: Iterable[str],
    *,
    num_measured_qubits: Optional[int] = None,
    expected_weights: Optional[Mapping[str, float]] = None,
    include_michelson: bool = True,
) -> DSRProfileSchema:
    """Convenience wrapper: build a :class:`DSRProfiler` and return its profile."""
    return DSRProfiler(
        counts,
        expected_outcomes,
        num_measured_qubits=num_measured_qubits,
        expected_weights=expected_weights,
        include_michelson=include_michelson,
    ).profile()


# ---------------------------------------------------------------------------
# Differential Success Rate: clipped Michelson contrast
# ---------------------------------------------------------------------------
#
# Compares the expected outcomes against the strongest *competing* peak
# rather than the random-chance baseline used by the companion CCS measure.
# This Michelson formulation is the definition of DSR; the other measures
# returned by DSRProfiler are comparisons reported alongside it.


def compute_dsr(counts: Mapping[str, int], expected_outcomes: Iterable[str]) -> float:
    """Compute DSR using its clipped Michelson-contrast definition."""
    return compute_dsr_michelson(counts, expected_outcomes)


def compute_dsr_michelson(counts: Mapping[str, int], expected_outcomes: Iterable[str]) -> float:
    """DSR using Michelson contrast: (a-b)/(a+b)."""
    p_exp_bar, p_comp = _extract_peaks(counts, expected_outcomes)
    return _clip_zero_one(_contrast_michelson(p_exp_bar, p_comp))


def compute_dsr_ratio(counts: Mapping[str, int], expected_outcomes: Iterable[str]) -> float:
    """
    Experimental ratio-based contrast; not the DSR definition used by QWARD.

    The value is clipped to [0, 1] after normalization:
    score = (a/b) / (1 + a/b) = a / (a + b).
    """
    p_exp_bar, p_comp = _extract_peaks(counts, expected_outcomes)
    return _clip_zero_one(_contrast_ratio(p_exp_bar, p_comp))


def compute_dsr_log_ratio(counts: Mapping[str, int], expected_outcomes: Iterable[str]) -> float:
    """
    Experimental log-ratio contrast; not the DSR definition used by QWARD.

    Maps log(a/b) to [0, 1] with a sigmoid.
    """
    p_exp_bar, p_comp = _extract_peaks(counts, expected_outcomes)
    return _clip_zero_one(_contrast_log_ratio(p_exp_bar, p_comp))


def compute_dsr_normalized_margin(
    counts: Mapping[str, int], expected_outcomes: Iterable[str]
) -> float:
    """
    Experimental normalized-margin contrast; not the QWARD DSR definition.

    Computes (a-b)/max(a,b).
    """
    p_exp_bar, p_comp = _extract_peaks(counts, expected_outcomes)
    return _clip_zero_one(_contrast_normalized_margin(p_exp_bar, p_comp))


def compute_dsr_with_flags(
    counts: Mapping[str, int], expected_outcomes: Iterable[str]
) -> Tuple[float, bool]:
    """
    Compute Michelson-contrast DSR and return a peak-mismatch flag.

    Returns:
        (dsr_score, peak_mismatch)
    """
    expected_set = _normalize_expected(expected_outcomes)
    total = _validate_counts(counts)

    p_exp = sum(counts.get(bitstring, 0) for bitstring in expected_set) / total
    p_exp_bar = p_exp / len(expected_set)

    p_comp = 0.0
    max_count = -1
    peak_outcomes = set()
    for outcome, count in counts.items():
        if count > max_count:
            max_count = count
            peak_outcomes = {outcome}
        elif count == max_count:
            peak_outcomes.add(outcome)
        if outcome not in expected_set:
            p_comp = max(p_comp, count / total)

    dsr = _clip_zero_one(_contrast_michelson(p_exp_bar, p_comp))
    peak_mismatch = expected_set.isdisjoint(peak_outcomes)
    return dsr, peak_mismatch


def compute_dsr_percent(counts: Mapping[str, int], expected_outcomes: Iterable[str]) -> float:
    """
    Compute Michelson-contrast DSR as a percentage in [0, 100].
    """
    return 100.0 * compute_dsr(counts, expected_outcomes)


def _contrast_michelson(p_exp_bar: float, p_comp: float) -> float:
    denom = p_exp_bar + p_comp
    if denom <= 0:
        return 0.0
    return (p_exp_bar - p_comp) / denom


def _contrast_ratio(p_exp_bar: float, p_comp: float) -> float:
    denom = p_exp_bar + p_comp
    if denom <= 0:
        return 0.0
    return p_exp_bar / denom


def _contrast_log_ratio(p_exp_bar: float, p_comp: float) -> float:
    # Use a small epsilon to avoid division by zero.
    eps = 1e-12
    ratio = (p_exp_bar + eps) / (p_comp + eps)
    return 1.0 / (1.0 + math.exp(-math.log(ratio)))


def _contrast_normalized_margin(p_exp_bar: float, p_comp: float) -> float:
    denom = max(p_exp_bar, p_comp)
    if denom <= 0:
        return 0.0
    return (p_exp_bar - p_comp) / denom


def _extract_peaks(
    counts: Mapping[str, int], expected_outcomes: Iterable[str]
) -> Tuple[float, float]:
    expected_set = _normalize_expected(expected_outcomes)
    total = _validate_counts(counts)

    p_exp = sum(counts.get(bitstring, 0) for bitstring in expected_set) / total
    p_exp_bar = p_exp / len(expected_set)

    p_comp = 0.0
    for outcome, count in counts.items():
        if outcome not in expected_set:
            p_comp = max(p_comp, count / total)

    return p_exp_bar, p_comp


if __name__ == "__main__":
    # Quick manual sanity checks
    example_counts = {"01": 40, "00": 20, "10": 20, "11": 20}
    print("Success rate:", compute_success_rate(example_counts, {"01"}))
    print(
        "Chance-corrected success:",
        compute_chance_corrected_success(example_counts, {"01"}, num_measured_qubits=2),
    )
    print("Coarse TVD similarity:", compute_coarse_tvd_similarity(example_counts, {"01"}))
    print("Coarse Hellinger fidelity:", compute_coarse_hellinger_fidelity(example_counts, {"01"}))
    print("DSR (Michelson, optional layer):", compute_dsr(example_counts, {"01"}))
