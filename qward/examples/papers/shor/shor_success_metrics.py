"""Success metrics for Shor experiments.

Primary success: continued fractions recover true order r.
Secondary: nontrivial factor from factors_from_order (can succeed when
recovered order is a divisor of r, e.g. r=2 from the 1/2 peak at m=3).
Uninformative: phase 0 (k=0) occurs with probability 1/r ideally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from qward.algorithms.shor import (
    analyze_counts,
    bitstring_to_phase,
    factors_from_order,
    phase_to_order,
)


@dataclass
class ShorEvaluation:
    success_rate: float
    success_count: int
    factoring_success_rate: float
    factoring_success_count: int
    uninformative_rate: float
    uninformative_count: int
    random_chance: float
    advantage_ratio: float
    true_order: int
    notes: str = ""


def evaluate_counts(
    counts: Dict[str, int],
    *,
    a: int,
    N: int,
    num_control: int,
    true_order: int,
    random_chance: float,
) -> ShorEvaluation:
    shots = sum(counts.values()) or 1
    success = 0
    factoring = 0
    uninformative = 0
    for bitstring, count in counts.items():
        phase = bitstring_to_phase(bitstring, num_control)
        if phase == 0:
            uninformative += count
        r, _ = phase_to_order(phase, N)
        if r == true_order:
            success += count
        info = factors_from_order(a, r, N)
        if info.get("nontrivial"):
            factoring += count
    s_rate = success / shots
    f_rate = factoring / shots
    u_rate = uninformative / shots
    adv = s_rate / random_chance if random_chance > 0 else 0.0
    note = ""
    if num_control == 3 and true_order == 4:
        note = (
            "m=3 nuance: ideal peak at phase 1/2 yields r=2 via CF, "
            "failing primary success but often succeeding at factoring."
        )
    return ShorEvaluation(
        success_rate=s_rate,
        success_count=success,
        factoring_success_rate=f_rate,
        factoring_success_count=factoring,
        uninformative_rate=u_rate,
        uninformative_count=uninformative,
        random_chance=random_chance,
        advantage_ratio=adv,
        true_order=true_order,
        notes=note,
    )


def evaluation_to_dict(ev: ShorEvaluation) -> Dict[str, Any]:
    return {
        "success_rate": ev.success_rate,
        "success_count": ev.success_count,
        "factoring_success_rate": ev.factoring_success_rate,
        "factoring_success_count": ev.factoring_success_count,
        "uninformative_rate": ev.uninformative_rate,
        "uninformative_count": ev.uninformative_count,
        "random_chance": ev.random_chance,
        "advantage_ratio": ev.advantage_ratio,
        "true_order": ev.true_order,
        "notes": ev.notes,
        "threshold_30": ev.success_rate >= 0.30,
        "threshold_50": ev.success_rate >= 0.50,
        "quantum_advantage": ev.advantage_ratio > 2.0,
    }
