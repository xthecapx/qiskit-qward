"""Histogram-only screening for applications seeking one outstanding candidate."""

import math
from numbers import Integral, Real
from typing import Mapping, Optional

from scipy.stats import binom

from qward.schemas.output_screening_schema import OutputScreeningSchema


def compute_output_screen(
    counts: Mapping[str, int], *, significance_level: Optional[float] = None
) -> OutputScreeningSchema:
    """Report observed peak separation and an exact two-sided rank test.

    Args:
        counts: Complete measured counts on the application's output register.
            Labels are opaque nonempty strings. Counts must be nonnegative
            integers, not probabilities, mitigated weights, or filtered top-k
            counts. Omitted unobserved outcomes have count zero. The caller
            must align/marginalize output registers before calling this function.
        significance_level: Optional per-histogram level strictly between zero
            and one. If omitted, ``rank_verified`` is None; the p-value is still
            returned. No contrast cutoff is applied.

    Returns:
        OutputScreeningSchema: Counts, contrast, mass, ties, and rank evidence.
        A tie gives no unique candidate, zero contrast, and p-value one. With
        only one observed outcome the runner-up count is zero. P-values retain
        floating-point precision (extreme tails can underflow to zero).

    Raises:
        ValueError: If counts are empty, invalid, sum to zero, or the supplied
            significance level is invalid.

    Notes:
        For top counts n1 >= n2, the p-value is
        min(1, 2 * P[Binomial(n1+n2, 1/2) >= n1]). This is the conservative
        nonrandomized form of the selected-winner test in Hung and Fithian,
        "Rank Verification for Exponential Families", section 1.2,
        https://arxiv.org/abs/1610.03944. The TWO-sided comparison accounts
        for selecting the observed ranks under a fixed-shot IID multinomial
        model. It does not control repeated looks or tests across many jobs.

        A verified rank does not imply a correct answer. A failed rank test
        does not establish uniformity or mean the output should be discarded.
        Several valid answers can yield low contrast. Use application-level
        verification after screening. No expected outcomes or ideal simulation
        enter this function. The count scan is linear in observed bins; only
        tied leading labels are sorted for deterministic serialization.
    """
    if significance_level is not None:
        if (
            isinstance(significance_level, bool)
            or not isinstance(significance_level, Real)
            or not math.isfinite(significance_level)
            or not 0 < significance_level < 1
        ):
            raise ValueError("significance_level must be finite and strictly between 0 and 1")

    if not counts:
        raise ValueError("counts must not be empty")
    total = top = second = 0
    leaders: list[str] = []
    for outcome, count in counts.items():
        if not isinstance(outcome, str) or not outcome:
            raise ValueError("outcome labels must be nonempty strings")
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError(f"counts[{outcome!r}] must be a nonnegative integer shot count")
        count = int(count)
        total += count
        if count > top:
            second, top = top, count
            leaders = [outcome]
        elif count == top and count > 0:
            second = top
            leaders.append(outcome)
        elif count > second:
            second = count
    if total == 0:
        raise ValueError("counts must sum to a positive value")

    pair_total = top + second
    unique = len(leaders) == 1
    pvalue = min(1.0, float(2 * binom.sf(top - 1, pair_total, 0.5)))
    return OutputScreeningSchema(
        shots=total,
        leading_outcome=leaders[0] if unique else None,
        leading_outcomes=sorted(leaders),
        has_unique_peak=unique,
        top_count=top,
        runner_up_count=second,
        top_probability=top / total,
        runner_up_probability=second / total,
        top_two_contrast=(top - second) / pair_total,
        absolute_gap=(top - second) / total,
        pair_probability=pair_total / total,
        rank_pvalue=pvalue,
        significance_level=significance_level,
        rank_verified=(
            (unique and pvalue <= significance_level) if significance_level is not None else None
        ),
    )
