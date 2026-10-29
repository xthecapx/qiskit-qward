"""Replay one saved BV run: predict from archived calibration, then inspect counts.

No IBM connection, circuit execution, sampling, or file modification is performed.
The checks are exploratory diagnostics conditional on fixed probabilities.
"""

import argparse
import json
from pathlib import Path

from qward.examples.papers.bv.bv_calibration_analysis import load_rows
from qward.examples.papers.bv.bv_calibration_ibm import BVCalibrationIBMExperiment
from qward.metrics import CircuitErrorPrediction, HistogramAssessment
from qward.metrics.histogram_assessment import binomial_prediction_check

BASELINE = (
    Path(__file__).resolve().parent
    / "data/simulator/calibration_prediction/baseline_20261022_smoke_v2"
)


def show_saved_run(baseline: Path, profile: str, layers: int, run_index: int):
    """Demonstrate the metric APIs on one identifiable, integrity-checked record."""
    config_id = f"BVCP2-low_predicted_error-ALT-L{layers:03d}"
    matches = [
        row
        for row in load_rows(baseline / "raw")
        if row["config_id"] == config_id
        and row["noise_id"] == profile
        and row["run_index"] == run_index
    ]
    if len(matches) != 1:
        raise ValueError("Expected one completed matching run in the supplied baseline")
    row = matches[0]
    journal, prepared, circuits = BVCalibrationIBMExperiment._load(Path(row["source"]))
    result = next(r for r in journal["individual_results"] if r["job_id"] == row["job_id"])
    isa = circuits["isa.qpy"]

    # BEFORE execution: these inputs were archived before the original job ran.
    prediction = CircuitErrorPrediction(
        isa,
        snapshot=prepared["calibration_snapshot"],
        reference=prepared["injected_reference"],
    ).get_metrics()
    # Archives frozen before M2 existed lack its fields; every frozen field must still match.
    replayed = prediction.model_dump()
    frozen = prepared["prediction"]
    if any(
        replayed[key] != value
        for key, value in frozen.items()
        if key not in {"schema_version", "assumptions"}
    ):
        raise ValueError("Replayed prediction differs from the frozen prediction")

    # AFTER execution: use the same ISA circuit and prediction, plus saved counts.
    # Reuse the archive's derived screening allocation for this exploratory view.
    alpha = row["uncertainty_screening_alpha"]
    assessment = HistogramAssessment(
        isa,
        counts=result["counts"],
        reference=prepared["nominal_reference"],
        prediction=prediction,
        alpha=alpha,
    ).get_metrics()
    budget = assessment.error_budget
    if budget is None or prediction.mixture_tvd is None:
        raise ValueError("This demo requires a complete prediction for a primary BV run")
    target = next(iter(prepared["nominal_reference"]))
    signal = binomial_prediction_check(
        row["target_count"], assessment.shots, 2.0**-isa.num_clbits, alpha=alpha
    )
    mixture = binomial_prediction_check(
        budget.count, assessment.shots, prediction.mixture_tvd, alpha=alpha
    )
    difference_pp = 100 * (assessment.tvd - budget.predicted_probability)
    z_text = f"{budget.residual_shot_z:+.2f}" if budget.residual_shot_z is not None else "undefined"
    tail_text = (
        f"{signal.upper_tail_pvalue:.3g}"
        if signal.upper_tail_pvalue > 0
        else "below floating-point range (underflow)"
    )
    print(f"Saved simulator run: {config_id}, {profile}, repetition {run_index}")
    print(f"Source: {row['source']}")
    print(f"Correct answer: {target}; shots: {assessment.shots}")
    print(f"Counts: {json.dumps(result['counts'], sort_keys=True)}")
    print(f"Calibration coverage: {prediction.calibration_coverage:.0%}")
    print(f"Predicted error, product: {budget.predicted_probability:.2%}")
    print(f"Predicted error, mixture: {prediction.mixture_tvd:.2%}")
    print(f"Observed error: {assessment.tvd:.2%} ({budget.count} wrong answers)")
    print(f"Observed - product prediction: {difference_pp:+.2f} percentage points")
    print(f"Difference in predicted shot standard deviations: {z_text}")
    print(f"Per-check alpha: {alpha:.8g}; retrospective diagnostics, not a new joint test")
    print(
        f"Product count range: {budget.predicted_count_lower}-{budget.predicted_count_upper}; "
        f"upper budget: {budget.upper_count_budget}; exceeded: {budget.exceeds_upper_budget}"
    )
    print(
        f"Mixture count range: {mixture.predicted_count_lower}-{mixture.predicted_count_upper}; "
        f"upper budget: {mixture.upper_count_budget}; exceeded: {mixture.exceeds_upper_budget}"
    )
    m2 = assessment.m2_error_budget
    if m2 is not None:
        print(
            f"Predicted error, M2 depolarizing: {m2.predicted_probability:.2%}; "
            f"observed - M2: {100 * (assessment.tvd - m2.predicted_probability):+.2f} pp"
        )
        print(
            f"M2 count range: {m2.predicted_count_lower}-{m2.predicted_count_upper}; "
            f"range check: {m2.range_status}; full M2 histogram check: "
            f"{assessment.m2_model_status}"
        )
    print(f"Product range check: {budget.range_status}")
    print(f"Full mixture histogram check: {assessment.model_status}")
    print(
        f"BV target above uniform background: {signal.exceeds_upper_budget} "
        f"(assumed background={signal.predicted_probability:.2%}, "
        f"exact upper-tail p={tail_text})"
    )
    print(f"Winner rank: {assessment.rank_status}; not a correctness test")
    print("Budgets include shot variation only; prediction uncertainty is not calibrated.")
    print("No bins removed, no jobs submitted, no new shots generated.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--profile", choices=["IDEAL", "NOMINAL", "STRESS"], default="NOMINAL")
    parser.add_argument("--layers", type=int, choices=[0, 2, 8, 32], default=32)
    parser.add_argument("--run-index", type=int, default=0)
    args = parser.parse_args()
    show_saved_run(args.baseline, args.profile, args.layers, args.run_index)


if __name__ == "__main__":
    main()
