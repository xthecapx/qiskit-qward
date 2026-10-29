"""Offline analysis of saved BV calibration campaign JSON records.

Exports tracked results, error comparisons and conditional uncertainty checks.
Controls remain separate from primary BV results. No backend is contacted.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from qward.metrics.circuit_error_prediction import depolarizing_prediction
from qward.metrics.histogram_assessment import distribution_status, binomial_prediction_check

MODELS = ("m1_success", "m1_mixture", "m2_depolarizing")
# Per primary run: central range + upper budget for each model, plus target test.
CLAIMS_PER_RUN = 2 * len(MODELS) + 1


def classify_outcome(signal_resolved: bool, range_status: Optional[str]) -> str:
    """Combine task signal (histogram) with prediction consistency (calibration).

    The histogram decides whether this run's answer is usable; the prediction
    decides whether that outcome was expected. Range status covers shot noise
    only, so on hardware it needs a calibrated model-discrepancy allowance.
    """
    if range_status is None:
        return "prediction_unavailable"
    if signal_resolved:
        return {
            "within_range": "success",
            "above_range": "degraded_success",
            "below_range": "success_better_than_predicted",
        }[range_status]
    return {
        "within_range": "expected_failure",
        "above_range": "anomalous_failure",
        "below_range": "unresolved_better_than_predicted",
    }[range_status]


def compare_histogram_error(
    assessment: Dict,
    prediction: Dict,
    counts: Dict,
    *,
    primary: bool,
    screening_alpha: Optional[float] = None,
) -> Dict:
    """Compare errors to the same ideal BV reference, preserving residual direction.

    Predicted error uses the pre-run score; observed error is empirical TVD.
    Existing intervals cover sampling uncertainty, not calibration/model error.
    Injected-fault controls have different nominal and injected references, so
    these comparisons are restricted to the primary, unperturbed BV sweep.
    """
    observed = assessment["tvd"]
    lower, upper = assessment["tvd_interval"]
    result = {
        "observed_histogram_error": observed,
        "histogram_error_interval_lower": lower,
        "histogram_error_interval_upper": upper,
        "histogram_error_interval_alpha": assessment["alpha"],
    }
    if not primary:
        return result
    alpha = assessment["alpha"] if screening_alpha is None else screening_alpha
    result["uncertainty_screening_alpha"] = alpha
    shots = sum(counts.values())
    target = next(iter(prediction["reference_probabilities"]))
    target_count = sum(value for key, value in counts.items() if key.replace(" ", "") == target)
    result["target_bitstring"] = target
    result["target_count"] = target_count
    m2_success = prediction.get("m2_success")
    for model, expected in (
        (
            "m1_success",
            1 - prediction["m1_success"] if prediction["m1_success"] is not None else None,
        ),
        ("m1_mixture", prediction["mixture_tvd"]),
        ("m2_depolarizing", 1 - m2_success if m2_success is not None else None),
    ):
        result[f"predicted_histogram_error_{model}"] = expected
        result[f"histogram_error_residual_{model}"] = (
            observed - expected if expected is not None else None
        )
        result[f"histogram_error_absolute_residual_{model}"] = (
            abs(observed - expected) if expected is not None else None
        )
        result[f"{model}_error_in_observed_interval"] = (
            lower <= expected <= upper if expected is not None else None
        )
        if expected is not None:
            check = binomial_prediction_check(shots - target_count, shots, expected, alpha=alpha)
            result.update(
                {f"{model}_budget_{key}": value for key, value in check.model_dump().items()}
            )
    signal = binomial_prediction_check(target_count, shots, 2.0 ** -len(target), alpha=alpha)
    result["uniform_background_target_probability"] = signal.predicted_probability
    result["target_signal_shot_z"] = signal.residual_shot_z
    result["target_signal_pvalue"] = signal.upper_tail_pvalue
    result["target_signal_resolved_under_uniform_null"] = signal.exceeds_upper_budget
    result["uncertainty_review_action"] = (
        "investigate_prediction"
        if result.get("m1_success_budget_range_status") in ("above_range", "below_range")
        else (
            "prediction_unavailable"
            if prediction["m1_success"] is None
            else (
                "retain_for_task_analysis"
                if signal.exceeds_upper_budget
                else "unresolved_task_signal_at_current_shots"
            )
        )
    )
    result["outcome_class"] = classify_outcome(
        signal.exceeds_upper_budget, result.get("m2_depolarizing_budget_range_status")
    )
    m2_probs = prediction.get("m2_probabilities")
    if m2_probs is not None:
        observed_probs = {}
        for key, value in counts.items():
            label = key.replace(" ", "")
            observed_probs[label] = observed_probs.get(label, 0) + value / shots
        result["m2_prediction_histogram_tvd"] = 0.5 * sum(
            abs(observed_probs.get(key, 0) - m2_probs.get(key, 0))
            for key in observed_probs.keys() | m2_probs.keys()
        )
    else:
        result["m2_prediction_histogram_tvd"] = None
    # Same simultaneous-bin test and alpha as the record's M1 mixture model_status.
    clean = {}
    for key, value in counts.items():
        clean[key.replace(" ", "")] = clean.get(key.replace(" ", ""), 0) + value
    result["m2_model_status"] = distribution_status(
        m2_probs,
        clean,
        shots,
        assessment["alpha"],
        target_count=target_count,
        target_probability=m2_success,
    )
    model_probs = prediction["mixture_probabilities"]
    if model_probs is not None:
        observed_probs = {}
        for key, value in counts.items():
            label = key.replace(" ", "")
            observed_probs[label] = observed_probs.get(label, 0) + value / shots
        result["mixture_prediction_histogram_tvd"] = 0.5 * sum(
            abs(observed_probs.get(key, 0) - model_probs.get(key, 0))
            for key in observed_probs.keys() | model_probs.keys()
        )
    else:
        result["mixture_prediction_histogram_tvd"] = None
    return result


def with_m2(prediction: Dict, snapshot: Optional[Dict]) -> Dict:
    """Add M2 to records frozen before it existed, from their archived snapshot.

    M2 has no fitted parameters: it reads the same pre-run calibration, so the
    recomputed value is what a frozen M2 prediction would have contained.
    """
    if "m2_success" in prediction:
        return {**prediction, "m2_source": "frozen_prediction"}
    if snapshot is None:
        return {**prediction, "m2_success": None, "m2_source": "unavailable"}
    m2 = depolarizing_prediction(
        snapshot, prediction["reference_probabilities"], prediction["output_width"]
    )
    return {
        **prediction,
        "m2_polarization": m2["polarization"],
        "m2_success": m2["success"],
        "m2_tvd": m2["tvd"],
        "m2_probabilities": m2["probabilities"],
        "m2_source": "recomputed_from_archived_snapshot",
    }


def load_rows(directory: Path) -> List[Dict]:
    """Read existing rich experiment results; ignore prepared/failed jobs."""
    rows = []
    records = []
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("algorithm") != "BV-CALIBRATION-PREDICTION":
            continue
        records.append((path, record))
    primary_count = sum(
        result.get("status") == "completed" and bool(result.get("counts"))
        for _, record in records
        if record["config"]["role"] == "primary"
        for result in record.get("individual_results", [])
    )
    # Legacy assessments are separate from these derived screening claims.
    screening_alpha = 0.05 / (CLAIMS_PER_RUN * max(1, primary_count))
    for path, record in records:
        config = record["config"]
        for result in record.get("individual_results", []):
            if result.get("status") != "completed" or not result.get("counts"):
                continue
            assessment = result["histogram_assessment"]
            prediction = with_m2(result["prediction"], record.get("calibration_snapshot"))
            row = {
                "source": str(path),
                "config_id": config["config_id"],
                "experiment_id": result["experiment_id"],
                "job_id": result["job_id"],
                "run_index": result["run_index"],
                "campaign_id": record["campaign_id"],
                "window": record["window"],
                "backend_name": record["backend_name"],
                "execution_type": record.get("execution_type", "unknown"),
                "noise_id": record["noise_id"],
                "family": config["family"],
                "role": config["role"],
                "pattern": config["pattern"],
                "output_width": config["num_qubits"],
                "forward_layers": config["forward_layers"],
                "layout_label": config["layout_label"],
                "arm": record.get("arm", "baseline"),
                "batch_id": record.get("batch_id"),
                "shots": assessment["shots"],
                "success_rate": assessment["success_rate"],
                "tvd": assessment["tvd"],
                "model_status": assessment["model_status"],
                "rank_status": assessment["rank_status"],
                "calibration_coverage": prediction["calibration_coverage"],
                "snapshot_hash": prediction["snapshot_hash"],
                "circuit_hash": prediction["circuit_hash"],
                "m1_success": prediction["m1_success"],
                "m2_success": prediction.get("m2_success"),
                "m2_polarization": prediction.get("m2_polarization"),
                "m2_source": prediction.get("m2_source"),
                "m1_mixture_success": None,
                **prediction["features"],
                **compare_histogram_error(
                    assessment,
                    prediction,
                    result["counts"],
                    primary=config["role"] == "primary",
                    screening_alpha=screening_alpha,
                ),
            }
            # Drift diagnostic: same circuit re-predicted with calibration at run time.
            entry = next(
                (
                    e
                    for e in record.get("submission_journal", [])
                    if e.get("job_id") == result["job_id"]
                ),
                {},
            )
            late = entry.get("execution_time_prediction")
            row["execution_calibration_changed"] = entry.get("execution_calibration_changed")
            for name, key in (("m1", "m1_success"), ("m2", "m2_success")):
                early = prediction.get(key)
                row[f"execution_time_predicted_error_{name}"] = (
                    1 - late[key] if late and late.get(key) is not None else None
                )
                row[f"calibration_drift_{name}"] = (
                    late[key] - early
                    if late and late.get(key) is not None and early is not None
                    else None
                )
            if config["role"] == "primary" and prediction["effective_visibility"] is not None:
                visibility = prediction["effective_visibility"]
                row["m1_mixture_success"] = (
                    visibility + (1 - visibility) / 2 ** config["num_qubits"]
                )
            rows.append(row)
    keys = [(r["campaign_id"], r["job_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate jobs found; remove copied records before analysis")
    return rows


def export(directory: Path, output: Path) -> Dict:
    """Save tracking CSV and summary beside, but outside, raw experiment records."""
    rows = load_rows(directory)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "jobs.csv").open("w", newline="") as stream:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    error_fields = [
        "config_id",
        "noise_id",
        "pattern",
        "forward_layers",
        "run_index",
        "shots",
        "target_bitstring",
        "target_count",
        "predicted_histogram_error_m1_success",
        "predicted_histogram_error_m1_mixture",
        "predicted_histogram_error_m2_depolarizing",
        "m2_source",
        "execution_calibration_changed",
        "execution_time_predicted_error_m1",
        "execution_time_predicted_error_m2",
        "calibration_drift_m1",
        "calibration_drift_m2",
        "output_width",
        "layout_label",
        "arm",
        "batch_id",
        "observed_histogram_error",
        "histogram_error_interval_lower",
        "histogram_error_interval_upper",
        "histogram_error_interval_alpha",
        "uncertainty_review_action",
        "outcome_class",
        "uncertainty_screening_alpha",
        "uniform_background_target_probability",
        "target_signal_shot_z",
        "target_signal_pvalue",
        "target_signal_resolved_under_uniform_null",
        "histogram_error_residual_m1_success",
        "histogram_error_residual_m1_mixture",
        "histogram_error_absolute_residual_m1_success",
        "histogram_error_residual_m2_depolarizing",
        "histogram_error_absolute_residual_m1_mixture",
        "histogram_error_absolute_residual_m2_depolarizing",
        "m1_success_error_in_observed_interval",
        "m1_mixture_error_in_observed_interval",
        "m2_depolarizing_error_in_observed_interval",
        "mixture_prediction_histogram_tvd",
        "m2_prediction_histogram_tvd",
        "model_status",
        "m2_model_status",
        "experiment_id",
        "job_id",
        "source",
    ]
    error_fields.extend(sorted({key for row in rows for key in row if "_budget_" in key}))
    with (output / "error_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=error_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(row for row in rows if row["role"] == "primary")
    report = {
        "schema_version": "1.0",
        "executions": len(rows),
        "primary_executions": sum(r["role"] == "primary" for r in rows),
        "diagnostic_executions": sum(r["role"] != "primary" for r in rows),
        "shots": sum(r["shots"] for r in rows),
        "note": "Error intervals and conditional count budgets quantify shot uncertainty; prediction uncertainty is not calibrated.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export(args.data_dir, args.output_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
