"""Reproducible summary and scientific plots of a saved simulator baseline."""

import argparse
import csv
from collections import Counter
import importlib.metadata
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

from qward.examples.papers.bv.bv_calibration_analysis import MODELS, export, load_rows
from qward.examples.papers.bv.bv_calibration_ibm import BVCalibrationIBMExperiment
from qward.metrics.circuit_error_prediction import record_hash

WIDTH_MARKERS = {2: "o", 4: "s", 6: "^", 8: "D"}


def generate_report(root: Path):
    """Analyze persisted raw counts only; never rerun a simulator or fit to the holdout."""
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Baseline must finish before generating its comparison report")
    rows = load_rows(root / "raw")
    if (
        len(rows) != manifest["planned_executions"]
        or sum(r["shots"] for r in rows) != manifest["planned_shots"]
    ):
        raise ValueError("Saved observations do not match the baseline execution/shot budget")
    output = root / "analysis"
    export(root / "raw", output)
    primary = [r for r in rows if r["role"] == "primary"]
    profiles = [p["name"] for p in manifest["profiles"]]
    colors = {"IDEAL": "#62758a", "NOMINAL": "#007f86", "STRESS": "#bd4b31"}
    summaries, curve = [], []
    for profile in profiles:
        subset = [r for r in primary if r["noise_id"] == profile]
        summaries.append(
            {
                "profile": profile,
                "primary_executions": len(subset),
                "mixture_model_mismatches": sum(r["model_status"] == "mismatch" for r in subset),
                "mean_observed_histogram_error": float(
                    np.mean([r["observed_histogram_error"] for r in subset])
                ),
                "m1_success_error_mae": float(
                    np.mean([r["histogram_error_absolute_residual_m1_success"] for r in subset])
                ),
                "m1_mixture_error_mae": float(
                    np.mean([r["histogram_error_absolute_residual_m1_mixture"] for r in subset])
                ),
                "m2_depolarizing_error_mae": float(
                    np.mean(
                        [r["histogram_error_absolute_residual_m2_depolarizing"] for r in subset]
                    )
                ),
                "m2_depolarizing_error_mean_residual": float(
                    np.mean([r["histogram_error_residual_m2_depolarizing"] for r in subset])
                ),
                "m2_model_mismatches": sum(r.get("m2_model_status") == "mismatch" for r in subset),
                "m2_depolarizing_error_range_statuses": dict(
                    Counter(r["m2_depolarizing_budget_range_status"] for r in subset)
                ),
                "outcome_classes": dict(Counter(r["outcome_class"] for r in subset)),
                "m1_success_error_mean_residual": float(
                    np.mean([r["histogram_error_residual_m1_success"] for r in subset])
                ),
                "m1_mixture_error_mean_residual": float(
                    np.mean([r["histogram_error_residual_m1_mixture"] for r in subset])
                ),
                "mean_prediction_to_histogram_tvd": float(
                    np.mean([r["mixture_prediction_histogram_tvd"] for r in subset])
                ),
                "m1_success_error_range_statuses": dict(
                    Counter(r["m1_success_budget_range_status"] for r in subset)
                ),
                "m1_success_upper_budget_exceeded": sum(
                    r["m1_success_budget_exceeds_upper_budget"] for r in subset
                ),
                "m1_mixture_error_range_statuses": dict(
                    Counter(r["m1_mixture_budget_range_status"] for r in subset)
                ),
                "m1_mixture_upper_budget_exceeded": sum(
                    r["m1_mixture_budget_exceeds_upper_budget"] for r in subset
                ),
                "target_signal_resolved_under_uniform_null": sum(
                    r["target_signal_resolved_under_uniform_null"] for r in subset
                ),
            }
        )
        for width in sorted({r["output_width"] for r in subset}):
            for layers in sorted({r["forward_layers"] for r in subset}):
                cell = [
                    r
                    for r in subset
                    if r["forward_layers"] == layers and r["output_width"] == width
                ]
                curve.append(
                    {
                        "profile": profile,
                        "output_width": width,
                        "forward_layers": layers,
                        "executions": len(cell),
                        "observed_histogram_error": float(
                            np.mean([r["observed_histogram_error"] for r in cell])
                        ),
                        "predicted_histogram_error_m1_success": float(
                            np.mean([r["predicted_histogram_error_m1_success"] for r in cell])
                        ),
                        "predicted_histogram_error_m1_mixture": float(
                            np.mean([r["predicted_histogram_error_m1_mixture"] for r in cell])
                        ),
                        "predicted_histogram_error_m2_depolarizing": float(
                            np.mean([r["predicted_histogram_error_m2_depolarizing"] for r in cell])
                        ),
                        "histogram_error_residual_m1_success": float(
                            np.mean([r["histogram_error_residual_m1_success"] for r in cell])
                        ),
                        "histogram_error_residual_m1_mixture": float(
                            np.mean([r["histogram_error_residual_m1_mixture"] for r in cell])
                        ),
                        "histogram_error_residual_m2_depolarizing": float(
                            np.mean([r["histogram_error_residual_m2_depolarizing"] for r in cell])
                        ),
                    }
                )
    by_width = []
    for profile in profiles:
        for width in sorted({r["output_width"] for r in primary}):
            cell = [r for r in primary if r["noise_id"] == profile and r["output_width"] == width]
            if not cell:
                continue
            by_width.append(
                {
                    "profile": profile,
                    "output_width": width,
                    "executions": len(cell),
                    **{
                        f"{model}_mae": float(
                            np.mean([r[f"histogram_error_absolute_residual_{model}"] for r in cell])
                        )
                        for model in MODELS
                    },
                    **{
                        f"{model}_within_range": sum(
                            r[f"{model}_budget_range_status"] == "within_range" for r in cell
                        )
                        for model in MODELS
                    },
                }
            )
    with (output / "error_comparison_by_width.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(by_width[0]))
        writer.writeheader()
        writer.writerows(by_width)
    with (output / "error_comparison_by_depth.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    methods = Counter()
    wrong_peaks, uniform = [], []
    verified_records = 0
    for path in (root / "raw").glob("*.json"):
        record = json.loads(path.read_text())
        _, prepared, _ = BVCalibrationIBMExperiment._load(path)
        context = prepared["execution_context"]
        model_path = Path(context["noise_model_path"])
        if not model_path.exists():
            model_path = root / model_path.name
        if record_hash(json.loads(model_path.read_text())) != context["noise_model_hash"]:
            raise ValueError(f"Archived noise model failed verification: {path.name}")
        verified_records += 1
        methods.update({record["simulator_method"]: len(record["individual_results"])})
        if record["config"]["family"] == "uniform":
            uniform.extend(r["histogram_assessment"] for r in record["individual_results"])
        if record["config"]["family"] == "bv_target_deviation" and record["config_id"].endswith(
            "A3"
        ):
            wrong_peaks.extend(r["histogram_assessment"] for r in record["individual_results"])
    metrics = {
        "schema_version": "1.0",
        "executions": len(rows),
        "shots": sum(r["shots"] for r in rows),
        "profiles": summaries,
        "curves": curve,
        "by_width": by_width,
        "simulator_methods_by_execution": dict(methods),
        "wrong_peak_success_rates": sorted({r["success_rate"] for r in wrong_peaks}),
        "wrong_peak_rank_statuses": dict(Counter(r["rank_status"] for r in wrong_peaks)),
        "uniform_observed_tvd_range": (
            [min(r["tvd"] for r in uniform), max(r["tvd"] for r in uniform)] if uniform else None
        ),
        "qiskit_aer_version_at_analysis": importlib.metadata.version("qiskit-aer"),
        "verified_frozen_records_and_noise_models": verified_records,
        "interpretation": "Controlled synthetic-noise baseline; not hardware validation. Model-status tests refer to the full global-uniform mixture, not M1-success.",
        "uncertainty_screening": {
            "scope": "Retrospective checks conditional on fixed probabilities; shot variation only",
            "prediction_uncertainty_calibrated": False,
            "alpha_per_claim": primary[0]["uncertainty_screening_alpha"],
            "multiplicity": "Bonferroni 0.05/(7 * primary runs), separate from legacy claims",
            "automatic_discard": False,
        },
    }
    (output / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
        sizes = {r["shots"] for r in primary}
        alphas = {r["uncertainty_screening_alpha"] for r in primary}
        for ax, model, title in zip(
            axes,
            MODELS,
            ("M1 product-as-success", "M1 uniform mixture", "M2 depolarizing"),
        ):
            if len(sizes) == len(alphas) == 1:
                shots, alpha = next(iter(sizes)), next(iter(alphas))
                probabilities = np.linspace(0, 1, 501)
                ax.fill_between(
                    probabilities,
                    binom.ppf(alpha / 2, shots, probabilities) / shots,
                    binom.isf(alpha / 2, shots, probabilities) / shots,
                    color="#dce3ed",
                    label="Central predicted count range",
                )
                ax.plot(
                    probabilities,
                    binom.isf(alpha, shots, probabilities) / shots,
                    "--",
                    color="#66758c",
                    linewidth=1,
                    label="One-sided upper budget",
                )
            for profile in profiles:
                subset = [r for r in primary if r["noise_id"] == profile]
                ax.scatter(
                    [r[f"predicted_histogram_error_{model}"] for r in subset],
                    [r["observed_histogram_error"] for r in subset],
                    s=22,
                    alpha=0.65,
                    color=colors[profile],
                    label=profile,
                )
            ax.set(
                xlabel="Predicted error probability",
                ylabel="Observed error fraction",
                xlim=(-0.02, 1),
                ylim=(-0.02, 1),
                title=title,
            )
            ax.grid(alpha=0.15)
            ax.legend(frameon=False, fontsize=7)
        fig.suptitle("BV error budgets | conditional shot uncertainty; model uncertainty excluded")
        fig.savefig(output / "error_budget_screening.png", dpi=180)
        fig.savefig(output / "error_budget_screening.svg")
        plt.close(fig)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
        for ax, model, title in zip(
            axes,
            MODELS,
            (
                "M1: error = 1 - Q",
                "M1 mixture: error = (1 - Q)(1 - 1/D)",
                "M2: error = 1 - [λ·A + (1 - λ)/D]",
            ),
        ):
            for profile in profiles:
                subset = [r for r in primary if r["noise_id"] == profile]
                for width in sorted({r["output_width"] for r in subset}):
                    part = [r for r in subset if r["output_width"] == width]
                    observed = np.array([r["observed_histogram_error"] for r in part])
                    lower = np.array([r["histogram_error_interval_lower"] for r in part])
                    upper = np.array([r["histogram_error_interval_upper"] for r in part])
                    ax.errorbar(
                        [r[f"predicted_histogram_error_{model}"] for r in part],
                        observed,
                        yerr=[np.maximum(0, observed - lower), np.maximum(0, upper - observed)],
                        fmt=WIDTH_MARKERS.get(width, "o"),
                        markersize=3.5,
                        linewidth=0.6,
                        capsize=0,
                        color=colors[profile],
                        alpha=0.6,
                        label=f"{profile}, {width} bits",
                    )
            ax.plot([0, 1], [0, 1], ":", color="#88919e", label="Exact scalar agreement")
            ax.set(
                xlabel="Predicted histogram error BEFORE execution",
                ylabel="Observed histogram error AFTER execution",
                xlim=(-0.02, 1),
                ylim=(-0.02, 1),
                title=title,
            )
            ax.grid(alpha=0.15)
            ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.suptitle(
            "BV error comparison | each point is one execution; bars show shot uncertainty",
            fontsize=12,
        )
        fig.savefig(output / "expected_vs_observed_error.png", dpi=180)
        fig.savefig(output / "expected_vs_observed_error.svg")
        plt.close(fig)
    lines = [
        "# BV noisy-simulator baseline",
        "",
        f"Completed {len(rows)} executions ({metrics['shots']:,} shots): {len(manifest['config_ids'])} configurations × {len(profiles)} profiles × {manifest['num_runs']} seeds, with {manifest['shots_per_execution']:,} shots per execution.",
        "",
        "## Predicted error versus final histogram error",
        "",
        "For a primary BV circuit, the ideal distribution is a point mass at the known answer t. Consequently, the empirical histogram error is exactly `E_observed = TVD(counts/N, ideal) = 1 - counts[t]/N`. No ideal state simulation is needed. This is the observed error estimate; the stored interval represents its finite-shot uncertainty.",
        "",
        "Let Q be the calibration product saved before execution. We compare two explicit hypotheses against that same observed error:",
        "",
        "- **M1-success:** `E_predicted = 1 - Q`. This treats Q as a prediction of target success.",
        "- **M1-mixture:** `E_predicted = (1 - Q)(1 - 1/D)`, with D=2^m. This treats Q as visibility in a global uniform-output mixture. At two measured bits, this is `0.75*(1-Q)`.",
        "- **Observed − predicted:** the signed difference between histogram error and predicted error. The table expresses this in **percentage points (pp)**: positive means more error than predicted, negative means less. This difference is not a sigma score; uncertainty-standardized scores are reported separately below.",
        "",
        "These quantities are available for every primary BV run in [error_comparison.csv](error_comparison.csv), together with counts, error intervals, interval inclusion, and identifiers linking to the frozen prediction. Injected-fault controls are excluded from this scalar comparison because their hardware reference differs from the nominal task reference.",
        "",
        "- **M2-depolarizing:** each native gate with average infidelity r and dimension d scrambles the register with probability `p = r·d/(d−1)`; polarization `λ = Π(1−p)`; readout applies independent bit flips A. Then `P(t) = λ·Π(1−r_ro) + (1−λ)/D` and `E_predicted = 1 − P(t)`. Like M1 it aggregates per-instruction calibration only, but it can never predict more error than a scrambled output (`1 − 1/D`). Older records predate M2; their M2 value is recomputed from the archived pre-run snapshot (`m2_source` column). M2 has no fitted parameters.",
        "",
        "| Noise | Bits | L | Observed error | M1 product | Obs − M1 (pp) | M1 mixture | Obs − mix (pp) | M2 depolarizing | Obs − M2 (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in curve:
        lines.append(
            f"| {c['profile']} | {c['output_width']} | {c['forward_layers']} | {c['observed_histogram_error']:.2%} | {c['predicted_histogram_error_m1_success']:.2%} | {100*c['histogram_error_residual_m1_success']:+.2f} | {c['predicted_histogram_error_m1_mixture']:.2%} | {100*c['histogram_error_residual_m1_mixture']:+.2f} | {c['predicted_histogram_error_m2_depolarizing']:.2%} | {100*c['histogram_error_residual_m2_depolarizing']:+.2f} |"
        )
    example = next(
        (
            r
            for r in primary
            if r["noise_id"] == "NOMINAL"
            and r["pattern"] == "ALT"
            and r["forward_layers"] == 32
            and r["output_width"] == 2
            and r["layout_label"] == "low_predicted_error"
            and r["run_index"] == 0
        ),
        primary[0],
    )
    stress_l8 = next(
        (
            c
            for c in curve
            if c["profile"] == "STRESS" and c["forward_layers"] == 8 and c["output_width"] == 2
        ),
        None,
    )
    if stress_l8 is not None:
        lines.extend(
            [
                "",
                f"**Reading STRESS, L=8:** the mixture model predicted {stress_l8['predicted_histogram_error_m1_mixture']:.2%} error; the observed error was {stress_l8['observed_histogram_error']:.2%}. Their difference is **{100*stress_l8['histogram_error_residual_m1_mixture']:+.2f} percentage points**. The mixture prediction underestimated the wrong-answer fraction by that amount. This is not a sigma score or a relative percent change. The product prediction for the same row was {stress_l8['predicted_histogram_error_m1_success']:.2%}, giving {100*stress_l8['histogram_error_residual_m1_success']:+.2f} percentage points; that model overestimated error.",
            ]
        )
    product_z = example["m1_success_budget_residual_shot_z"]
    mixture_z = example["m1_mixture_budget_residual_shot_z"]
    product_z_text = f"{product_z:+.2f}" if product_z is not None else "undefined (zero variance)"
    mixture_z_text = f"{mixture_z:+.2f}" if mixture_z is not None else "undefined (zero variance)"
    lines.extend(
        [
            "",
            "Each row averages three secrets × three seeds × the layouts in scope. `L` is the number of forward padding layers before their inverse; larger L increases noise exposure while preserving the ideal BV answer. `STRESS` uses the strongest simulated noise profile. See [error_comparison_by_depth.csv](error_comparison_by_depth.csv). A row average does not have a single per-run sigma value.",
            "",
            "### One actual saved run",
            "",
            f"Configuration `{example['config_id']}`, noise `{example['noise_id']}`, repetition {example['run_index']}: the expected answer `{example['target_bitstring']}` occurred **{example['target_count']}/{example['shots']}** times.",
            "",
            f"- Before execution: Q={1-example['predicted_histogram_error_m1_success']:.6f}, so the product hypothesis predicted **{example['predicted_histogram_error_m1_success']:.2%} error**.",
            f"- After execution: `1 - {example['target_count']}/{example['shots']}` gives **{example['observed_histogram_error']:.2%} histogram error**.",
            f"- Difference: **{100*example['histogram_error_residual_m1_success']:+.2f} percentage points**. The mixture hypothesis predicted **{example['predicted_histogram_error_m1_mixture']:.2%}**, with residual **{100*example['histogram_error_residual_m1_mixture']:+.2f} pp**.",
            f"- The stored {(1-example['histogram_error_interval_alpha'])*100:.4f}% sampling interval is **[{example['histogram_error_interval_lower']:.2%}, {example['histogram_error_interval_upper']:.2%}]**. Product estimate inside: `{example['m1_success_error_in_observed_interval']}`; mixture estimate inside: `{example['m1_mixture_error_in_observed_interval']}`. These are interval-inclusion diagnostics, not new jointly calibrated hypothesis tests, and they do not include calibration or model uncertainty.",
            "",
            "![Predicted versus observed histogram error](expected_vs_observed_error.png)",
            "",
            "Points above the diagonal have more observed error than predicted; points below have less. Vertical bars are the stored per-run sampling intervals. Equal scalar error does not imply equal histogram shape: `mixture_prediction_histogram_tvd` separately measures TVD(observed histogram, predicted histogram). The existing `model_status` tests that full mixture distribution, rather than only whether the two error scalars agree.",
            "",
            "",
            "## Error budget and uncertainty screening",
            "",
            "The new screening compares the observed **number of wrong BV answers** with an exact binomial count range calculated from each frozen prediction and the shot count. This is a predictive range for counts, whereas the earlier Clopper–Pearson interval estimates an unknown probability from the observed counts. Neither includes model uncertainty.",
            "",
            "For each primary run, error_comparison.csv now includes `*_budget_residual_shot_z`, central predicted count bounds, a separate one-sided `*_budget_upper_count_budget`, exact tail probabilities, and range/budget flags. The signed z-score is `(observed_error - predicted_error)/sqrt(predicted_error*(1-predicted_error)/shots)`. It uses the predicted shot variance, not the plug-in observed variance used in the earlier illustrative calculation. Decisions use exact bounds; a zero-variance prediction has a null z-score and is still tested exactly.",
            "",
            f"These retrospective diagnostics use alpha={example['uncertainty_screening_alpha']:.8g} per claim: 0.05 divided by seven claims per primary run and {len(primary)} primary runs. Seven claims means a central range and an upper budget for each of the three models, plus one prespecified-target test against uniform background. The adjustment applies to these new checks, not a combined guarantee with legacy tests or future repeated looks. The analysis policy was selected after inspecting this baseline, so these results are exploratory rather than a confirmatory validation.",
            "",
            f"For the example above, **{example['shots']-example['target_count']} wrong answers** were observed. The product prediction's central range is **{example['m1_success_budget_predicted_count_lower']}–{example['m1_success_budget_predicted_count_upper']}**, with upper budget **{example['m1_success_budget_upper_count_budget']}**; its signed residual is **{product_z_text} predicted shot standard deviations**. The mixture range is **{example['m1_mixture_budget_predicted_count_lower']}–{example['m1_mixture_budget_predicted_count_upper']}**, upper budget **{example['m1_mixture_budget_upper_count_budget']}**, and residual **{mixture_z_text}**. Which prediction is used changes the conclusion.",
            "",
            "| Noise | Runs | Above product upper budget | Below product central range | Above mixture upper budget | BV target resolved vs uniform |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                f"| {s['profile']} | {s['primary_executions']} | {s['m1_success_upper_budget_exceeded']} | {s['m1_success_error_range_statuses'].get('below_range', 0)} | {s['m1_mixture_upper_budget_exceeded']} | {s['target_signal_resolved_under_uniform_null']} |"
                for s in summaries
            ],
            "",
            "![Predicted count ranges and observed errors](error_budget_screening.png)",
            "",
            "Passing an upper budget does not establish useful signal: an overly pessimistic prediction can pass every run. The separate target test compares the known BV answer with a uniform-background probability of 1/D using an exact upper binomial tail. Uniform background is an explicit hypothesis, not a calibrated per-bin hardware-noise model. An unresolved target remains inconclusive at these shots, not proven absent. Diagnostic controls are excluded from this BV-specific screen.",
            "",
            "`uncertainty_review_action` is an exploratory review suggestion based on the product prediction and target test. Unexpectedly low or high error requests prediction review; independently inspect the upper-budget flag for unexpectedly poor runs. Every raw histogram is retained. Legacy 80% success / 0.10 TVD decisions have been removed from current code and exports. Historical raw records retain their original fields for provenance; the current report does not use them.",
            "",
            "Before this becomes a hardware acceptance rule, estimate prediction discrepancy from separate calibration windows, freeze the resulting predictive bound, and check coverage and false dismissals on new windows. Three simulator seeds under one stationary noise model per profile do not calibrate drift or hardware prediction uncertainty. A per-bin noise-only screen additionally requires a validated background probability for that bin.",
            "",
            "## Outcome classes: combining histogram signal with prediction consistency",
            "",
            "The histogram decides whether this run's answer is usable (target resolved above the 1/D background). The M2 prediction decides whether that outcome was expected (observed wrong-answer count inside M2's central count range). `outcome_class` in error_comparison.csv combines the two:",
            "",
            "| | Target resolved | Target unresolved |",
            "|---|---|---|",
            "| Within M2 range | `success` | `expected_failure` |",
            "| Above M2 range (more error) | `degraded_success` | `anomalous_failure` |",
            "| Below M2 range (less error) | `success_better_than_predicted` | `unresolved_better_than_predicted` |",
            "",
            "| Noise | "
            + " | ".join(sorted({k for s in summaries for k in s["outcome_classes"]}))
            + " |",
            "|---|" + "---:|" * len({k for s in summaries for k in s["outcome_classes"]}),
            *[
                f"| {s['profile']} | "
                + " | ".join(
                    str(s["outcome_classes"].get(k, 0))
                    for k in sorted({k for t in summaries for k in t["outcome_classes"]})
                )
                + " |"
                for s in summaries
            ],
            "",
            "Range status here includes shot noise only. In this simulator the noise is generated from the same calibration the predictor reads, so the classes test the aggregation model, not hardware predictivity. On hardware, widen the range by a model-discrepancy term estimated on separate calibration windows before using these classes as decisions.",
            "",
            "## Design and provenance",
            "",
            f"Scope `{manifest.get('scope', 'smoke')}`: {len(manifest['config_ids'])} configurations; measured widths {sorted({r['output_width'] for r in rows})}; layouts {sorted({r['layout_label'] for r in rows})}; {len(primary)} primary BV executions and {len(rows) - len(primary)} diagnostic-control executions. Primary BV patterns are ONES/ALT/SINGLE × L=0/2/8/32. The `contrasting_predicted_error` layout has twice the error rates. Both circuit constructions and result tracking are shared with the IBM runner.",
            "",
            "| Profile | Average 1Q infidelity | Average 2Q infidelity | Symmetric readout flip |",
            "|---|---:|---:|---:|",
        ]
    )
    for p in manifest["profiles"]:
        lines.append(
            f"| {p['name']} | {p['one_qubit_error']:.3%} | {p['two_qubit_error']:.3%} | {p['readout_error']:.3%} |"
        )
    lines.extend(
        [
            "",
            "The noise model is built from the same synthetic Target used by the predictor. Aer converts average gate infidelity to local depolarization strength; virtual RZ has no noise, readout is symmetric and independent, and thermal/idle noise is disabled. Routing and native decomposition are included. GenericBackendV2 is a local synthetic device, not a live IBM calibration snapshot.",
            "",
            f"Methods: {dict(methods)}. Stabilizer is used for BV and Clifford controls; matrix-product-state handles the π/6 rotation diagnostic. No dense statevector or density-matrix method is used. RNG seed base: {manifest['seed']}; exact per-execution seeds are in raw records. Repetitions share stationary calibration and must not be described as hardware drift windows.",
            "",
            f"Every prediction and calibration snapshot was archived before its local simulation. All {verified_records} prepared-record/QPY/noise-model hashes have been verified. See the top-level manifest, raw JSON records, QPY files, and noise-model JSONs. The analysis reads those saved counts without rerunning shots or fitting a new prediction.",
            "",
            "## Prediction accuracy on primary BV only",
            "",
            "| Profile | BV runs | M1 product MAE | M1 mixture MAE | M2 depolarizing MAE | M1 mixture histogram mismatches | M2 histogram mismatches |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for s in summaries:
        lines.append(
            f"| {s['profile']} | {s['primary_executions']} | {s['m1_success_error_mae']:.2%} | {s['m1_mixture_error_mae']:.2%} | {s['m2_depolarizing_error_mae']:.2%} | {s['mixture_model_mismatches']}/{s['primary_executions']} | {s['m2_model_mismatches']}/{s['primary_executions']} |"
        )
    lines.extend(
        [
            "",
            "| Profile | Bits | Runs | M1 product MAE | M1 mixture MAE | M2 MAE | Runs within M1 / M2 count range |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *[
                f"| {w['profile']} | {w['output_width']} | {w['executions']} | {w['m1_success_mae']:.2%} | {w['m1_mixture_mae']:.2%} | {w['m2_depolarizing_mae']:.2%} | {w['m1_success_within_range']} / {w['m2_depolarizing_within_range']} |"
                for w in by_width
            ],
            "",
            "MAE is the mean absolute observed-minus-predicted error, expressed in probability units (1% here means one percentage point). Full histogram mismatches also examine how probability is distributed among wrong answers; that is a different check from comparing total error.",
            "",
            "## Diagnostic checks",
            "",
            f"- Deliberately wrong BV peaks: observed target success rates {metrics['wrong_peak_success_rates']}; rank statuses {metrics['wrong_peak_rank_statuses']}. A well-resolved winner can be the wrong answer.",
            f"- Uniform-output controls: observed TVD range {metrics['uniform_observed_tvd_range']}. These circuits are intended to produce uniform outputs; a BV target-signal rule does not apply to them.",
            "- GHZ Z/X controls and SPAM/fault treatments remain in jobs.csv and raw records because they test readout, reference alignment, and failure modes of histogram screening.",
            "",
            "## Scope and remaining validation",
            "",
            "The experiment uses the existing BV builder, controlled padding, the M1 product and uniform-mixture predictions, the parameter-free M2 depolarizing aggregation, exact count budgets, and archived simulator/IBM results. No learned predictor or additional idle/mirror campaign is included. Raw counts and frozen predictions are preserved.",
            "",
            "The count budgets are conditional on each fixed prediction and account for shot variation only. Before automatic hardware triage, prediction uncertainty needs calibration and validation across independent time windows. Per-bin noise screening also needs a validated background model. This baseline does not justify automatic histogram deletion.",
            "",
            "Artifacts: [execution manifest](../manifest.json), [job table](jobs.csv), [error comparison](error_comparison.csv), [metrics](baseline_metrics.json), [error-budget SVG](error_budget_screening.svg).",
        ]
    )
    (output / "baseline_report.md").write_text("\n".join(lines) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_dir", type=Path)
    args = parser.parse_args()
    metrics = generate_report(args.baseline_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
