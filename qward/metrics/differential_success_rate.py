"""
Differential Success Rate (DSR).

Histogram-only metric that quantifies how well the expected outcome(s) stand out
from the strongest competing peak in the observed distribution.
"""

from __future__ import annotations

from typing import Mapping, Iterable, Set, Tuple


def compute_dsr(counts: Mapping[str, int], expected_outcomes: Iterable[str]) -> float:
    """
    Compute the Differential Success Rate (DSR) from a histogram.

    Args:
        counts: Mapping from bitstring to counts.
        expected_outcomes: Iterable of expected bitstrings.

    Returns:
        DSR score in [0, 1] using the chosen default implementation.
    """
    return compute_dsr_michelson(counts, expected_outcomes)


def compute_dsr_michelson(counts: Mapping[str, int], expected_outcomes: Iterable[str]) -> float:
    """
    DSR using Michelson contrast: (a-b)/(a+b).
    """
    p_exp_bar, p_comp = _extract_peaks(counts, expected_outcomes)
    return _clip_zero_one(_contrast_michelson(p_exp_bar, p_comp))


def compute_dsr_ratio(counts: Mapping[str, int], expected_outcomes: Iterable[str]) -> float:
    """
    DSR using ratio a/b, clipped to [0, 1] after normalization:
    score = (a/b) / (1 + a/b) = a / (a + b).
    """
    p_exp_bar, p_comp = _extract_peaks(counts, expected_outcomes)
    return _clip_zero_one(_contrast_ratio(p_exp_bar, p_comp))


def compute_dsr_log_ratio(counts: Mapping[str, int], expected_outcomes: Iterable[str]) -> float:
    """
    DSR using log-ratio log(a/b), mapped to [0, 1] with a sigmoid.
    """
    p_exp_bar, p_comp = _extract_peaks(counts, expected_outcomes)
    return _clip_zero_one(_contrast_log_ratio(p_exp_bar, p_comp))


def compute_dsr_normalized_margin(
    counts: Mapping[str, int], expected_outcomes: Iterable[str]
) -> float:
    """
    DSR using normalized margin: (a-b)/max(a,b).
    """
    p_exp_bar, p_comp = _extract_peaks(counts, expected_outcomes)
    return _clip_zero_one(_contrast_normalized_margin(p_exp_bar, p_comp))


def compute_dsr_with_flags(
    counts: Mapping[str, int], expected_outcomes: Iterable[str]
) -> Tuple[float, bool]:
    """
    Compute DSR and return a peak-mismatch flag.

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
    Compute DSR as a percentage in [0, 100].
    """
    return 100.0 * compute_dsr(counts, expected_outcomes)


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
    # Sigmoid to map to (0,1)
    import math

    return 1.0 / (1.0 + math.exp(-math.log(ratio)))


def _contrast_normalized_margin(p_exp_bar: float, p_comp: float) -> float:
    denom = max(p_exp_bar, p_comp)
    if denom <= 0:
        return 0.0
    return (p_exp_bar - p_comp) / denom


def _clip_zero_one(value: float) -> float:
    if value <= 0:
        return 0.0
    if value >= 1:
        return 1.0
    return value


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
    print("DSR:", compute_dsr(example_counts, {"01"}))
    print("DSR%:", compute_dsr_percent(example_counts, {"01"}))
