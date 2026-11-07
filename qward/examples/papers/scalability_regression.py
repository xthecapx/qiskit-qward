"""Offline, campaign-adjusted decay and held-out generalization analysis for thesis H2.

Run with ``python -m qward.examples.papers.scalability_regression --help``.
Only committed exported tables are read; this module never contacts a QPU.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import warnings

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from qward.examples.papers import decay_groups, decay_relationships

PAPERS = Path(__file__).resolve().parent
FAMILIES = {
    "fourier": ("QFT", "PHASE-ESTIMATION"),
    "oracle": ("GROVER", "BERNSTEIN-VAZIRANI"),
    "entanglement": ("VTP", "GHZ"),
}
PRIMARY_ALGORITHMS = tuple(algorithm for pair in FAMILIES.values() for algorithm in pair)
EXTRA_ALGORITHMS = ("SHOR", "RANDOM-VOLUMETRIC")
METRICS = (
    "a.num_qubits",
    "a.depth",
    "a.size",
    "a.ComplexityMetrics.gate_based_metrics.two_qubit_count",
    "a.StructuralMetrics.volume",
    "a.BehavioralMetrics.program_communication",
    "a.quantum_specific.entanglement_ratio",
    "b.depth",
    "b.two_qubit_depth",
    "b.two_qubit_gates",
)
MODELS = ("ols", "binomial_logit")
METRIC_LABELS = dict(
    zip(
        METRICS,
        (
            "Logical qubits",
            "Logical depth",
            "Logical operations",
            "Logical two-qubit gates",
            "Logical structural volume",
            "Logical program communication",
            "Logical entanglement ratio",
            "ISA depth",
            "ISA two-qubit depth",
            "ISA two-qubit gates",
        ),
    )
)
ALGORITHM_LABELS = {
    "QFT": "QFT",
    "PHASE-ESTIMATION": "PE",
    "GROVER": "Grover",
    "BERNSTEIN-VAZIRANI": "BV",
    "VTP": "vTP",
    "GHZ": "GHZ",
    "SHOR": "Shor",
    "RANDOM-VOLUMETRIC": "Random volumetric",
}
IDENTITY = {
    "meta.algorithm": "algorithm",
    "meta.backend_name": "backend",
    "meta.campaign_id": "campaign",
    "meta.window": "window",
    "meta.config_id": "config",
    "meta.job_id": "job_id",
    "meta.run_id": "run_id",
    "meta.shots": "shots",
    "f.fidelity.success_rate": "success",
    "f.fidelity.chance_baseline": "chance",
}


def load_data(tables: Path, metrics=METRICS, backend="ibm_fez", campaigns=None):
    """Validate manifest integrity and outcomes, retaining only pre-runtime predictors."""
    metrics = tuple(metrics)
    if len(metrics) != len(set(metrics)):
        raise ValueError("Duplicate predictors are not allowed")
    if not metrics or any(
        not name.startswith(("a.", "b."))
        or name.endswith(("compute_seconds", ".seed", ".cached", ".available"))
        for name in metrics
    ):
        raise ValueError(
            "Predictors must be circuit metrics from groups a/b, not targets or timing"
        )
    manifest_path = tables / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0":
        raise ValueError("Unsupported dataset manifest schema")
    names = [part["file"] for part in manifest["parts"]]
    if (
        not names
        or len(names) != len(set(names))
        or any(Path(name).name != name or not name.endswith(".csv") for name in names)
    ):
        raise ValueError("Invalid or duplicate CSV part names")
    frames, inputs = [], []
    for part in manifest["parts"]:
        path = tables / part["file"]
        if path.stat().st_size != part["bytes"]:
            raise ValueError(f"Dataset size mismatch: {path.name}")
        header = pd.read_csv(path, nrows=0).columns
        missing = set(IDENTITY) | set(metrics) | {"f.counts_json"}
        missing -= set(header)
        if missing:
            raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")
        target_cols = [col for col in decay_relationships.EXPORTED_TARGETS if col in header]
        frame = pd.read_csv(
            path, usecols=list(IDENTITY) + list(metrics) + ["f.counts_json", *target_cols]
        )
        if len(frame) != part["rows"]:
            raise ValueError(f"Dataset row count mismatch: {path.name}")
        frames.append(frame)
        inputs.append({**part, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    data = pd.concat(frames, ignore_index=True).rename(
        columns={**IDENTITY, **decay_relationships.EXPORTED_TARGETS}
    )
    for name in decay_relationships.EXPORTED_TARGETS.values():
        if name in data:
            data[name] = pd.to_numeric(data[name], errors="raise")
            if not data[name].dropna().between(0, 1).all():
                raise ValueError(f"Expected a bounded quality score for {name}")
    if len(data) != manifest["rows"] or data["job_id"].duplicated().any():
        raise ValueError("Manifest row count mismatch or duplicate job IDs")
    if data[list(IDENTITY.values())].isna().any().any():
        raise ValueError("Missing job identity, success, shot count, or chance baseline")
    for column in ["shots", "success", "chance", *metrics]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if (
        not np.isfinite(data[["shots", "success", "chance"]]).all().all()
        or (data["shots"] <= 0).any()
        or not np.allclose(data["shots"], np.rint(data["shots"]), atol=1e-7, rtol=0)
        or not data["success"].between(0, 1).all()
        or not data["chance"].between(0, 1).all()
    ):
        raise ValueError(
            "Success rates must represent integer accepted shots out of positive shots"
        )
    data["cluster"] = data["algorithm"].astype(str) + ":" + data["config"].astype(str)
    data["period"] = data["campaign"].astype(str) + " / window " + data["window"].astype(str)
    algorithms = PRIMARY_ALGORITHMS + EXTRA_ALGORITHMS
    selected = data["backend"].eq(backend) & data["algorithm"].isin(algorithms)
    if campaigns:
        selected &= data["campaign"].isin(campaigns)
    selected_data = data.loc[selected].copy()
    if selected_data.empty:
        raise ValueError("No analysis jobs match the selected backend/campaigns")
    # Exported target schemas round rates to six decimals. Recover exact accepted
    # counts from histograms and frozen success sets, never round rates into counts.
    reference_path = tables / "references.csv"
    references = pd.read_csv(reference_path, usecols=["run_id", "accepted_json", "ideal_json"])
    if references["run_id"].duplicated().any():
        raise ValueError("Duplicate reference run IDs")
    accepted_labels = references.set_index("run_id")["accepted_json"].map(json.loads).to_dict()
    ideals = references.set_index("run_id")["ideal_json"].map(json.loads).to_dict()
    accepted_counts, ideal_success = [], []
    for _, row in selected_data.iterrows():
        labels = accepted_labels.get(row["run_id"])
        counts = json.loads(row["f.counts_json"])
        if (
            not isinstance(labels, list)
            or len(labels) != len(set(labels))
            or not isinstance(counts, dict)
            or any(not isinstance(value, int) or value < 0 for value in counts.values())
            or sum(counts.values()) != row["shots"]
        ):
            raise ValueError(f"Invalid counts or frozen success labels: {row['job_id']}")
        accepted_counts.append(sum(counts.get(label, 0) for label in labels))
        ideal = ideals.get(row["run_id"])
        if (
            not isinstance(ideal, dict)
            or not ideal
            or any(not np.isfinite(value) or value < 0 for value in ideal.values())
            or not np.isclose(sum(ideal.values()), 1, atol=1e-6)
        ):
            raise ValueError(f"Invalid frozen ideal distribution: {row['job_id']}")
        ideal_success.append(sum(ideal.get(label, 0) for label in labels))
    selected_data["accepted"] = accepted_counts
    exact_success = selected_data["accepted"] / selected_data["shots"]
    if not np.allclose(exact_success, selected_data["success"], atol=5.1e-7, rtol=0):
        raise ValueError("Exported success rate disagrees with integer accepted shots")
    rounding_correction = float((exact_success - selected_data["success"]).abs().max())
    selected_data["success"] = exact_success
    selected_data["ideal_success"] = ideal_success
    selected_data["error"] = 1 - exact_success
    selected_data = selected_data.drop(columns="f.counts_json")
    provenance = {
        "backend": backend,
        "input_jobs": len(data),
        "selected_jobs": len(selected_data),
        "configuration_count": selected_data["cluster"].nunique(),
        "excluded_jobs": len(data) - len(selected_data),
        "campaign_jobs": selected_data.groupby("campaign").size().to_dict(),
        "period_jobs": selected_data.groupby("period").size().to_dict(),
        "algorithm_jobs": selected_data.groupby("algorithm").size().to_dict(),
        "exported_target_jobs": {
            name: int(selected_data[name].count()) if name in selected_data else 0
            for name in decay_relationships.EXPORTED_TARGETS.values()
        },
        "algorithm_configs": selected_data.groupby("algorithm")["cluster"].nunique().to_dict(),
        "metrics": list(metrics),
        "metric_coverage": {
            metric: {
                "available_jobs": int(np.isfinite(selected_data[metric]).sum()),
                "missing_jobs": int((~np.isfinite(selected_data[metric])).sum()),
            }
            for metric in metrics
        },
        "inputs": inputs,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "references_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "maximum_export_rate_rounding_correction": rounding_correction,
        "versions": {
            "python": platform.python_version(),
            "statsmodels": statsmodels.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    return selected_data, provenance


@dataclass
class RegressionFit:
    """Training-only scaling and category levels travel with a fitted regression."""

    result: object
    metric: str
    model: str
    center: float
    scale: float
    periods: tuple
    algorithms: tuple
    clusters: int

    def design(self, data):
        """Reject unknown period/algorithm levels rather than silently assigning a baseline."""
        if not set(data["period"]).issubset(self.periods):
            raise ValueError("Prediction contains an unseen collection period")
        x = (data[self.metric] - self.center) / self.scale
        design = pd.DataFrame({"intercept": 1.0, "x": x}, index=data.index)
        for index, period in enumerate(self.periods[1:], 1):
            design[f"period_{index}"] = data["period"].eq(period).astype(float)
        if self.algorithms:
            if not set(data["algorithm"]).issubset(self.algorithms):
                raise ValueError("Unknown algorithm in descriptive interaction model")
            other = data["algorithm"].eq(self.algorithms[1]).astype(float)
            design["algorithm_difference"] = other
            design["slope_difference"] = x * other
        return design


def fit_regression(data, metric, model, algorithms=()):
    """Fit OLS or grouped-binomial GLM with campaign effects and config-clustered SEs."""
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    if data[metric].nunique() < 3:
        raise ValueError("Fewer than three distinct predictor values")
    scale = float(data[metric].std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Constant or non-finite predictor")
    fit = RegressionFit(
        None,
        metric,
        model,
        float(data[metric].mean()),
        scale,
        tuple(data.groupby("period").size().sort_values(ascending=False, kind="stable").index),
        tuple(algorithms),
        data["cluster"].nunique(),
    )
    design = fit.design(data)
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError(
            "Rank-deficient design: metric, campaign, or algorithm effects are confounded"
        )
    if fit.clusters < max(6, design.shape[1] + 2):
        raise ValueError("Too few independent configuration clusters for this design")
    covariance = {"groups": data["cluster"], "use_correction": True}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PerfectSeparationWarning)
        if model == "ols":
            result = sm.OLS(data["success"], design, missing="raise").fit(
                cov_type="cluster", cov_kwds=covariance, use_t=True
            )
        else:
            endog = np.column_stack((data["accepted"], data["shots"] - data["accepted"]))
            result = sm.GLM(endog, design, family=sm.families.Binomial(), missing="raise").fit(
                cov_type="cluster", cov_kwds=covariance, use_t=True, maxiter=200
            )
            if not result.converged:
                raise ValueError("Binomial regression did not converge")
        if any(issubclass(item.category, PerfectSeparationWarning) for item in caught):
            raise ValueError("Perfect separation: finite binomial slope inference is unavailable")
    if not np.isfinite(result.params).all() or not np.isfinite(result.cov_params()).all().all():
        raise ValueError("Non-finite coefficients or covariance")
    if (np.diag(result.cov_params()) < 0).any():
        raise ValueError("Invalid negative variance in clustered covariance")
    fit.result = result
    return fit


def larger_configuration_split(data, metric, fraction=0.2):
    """Hold out the largest configurations, keeping all campaigns/repetitions together."""
    if not 0 < fraction < 1:
        raise ValueError("Holdout fraction must be strictly between zero and one")
    sizes = data.groupby("cluster")[metric].median()
    cutoff = sizes.quantile(1 - fraction)
    held_out = set(sizes.index[sizes > cutoff])
    test = data[data["cluster"].isin(held_out)].copy()
    train = data[~data["cluster"].isin(held_out)].copy()
    if train.empty or test.empty:
        raise ValueError("Tied/constant sizes cannot form a larger-configuration holdout")
    return train, test


def prediction_scores(observed, predicted):
    """Unclipped rate-scale scores; undefined R-squared is explicitly missing."""
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    residual = observed - predicted
    total = np.sum((observed - observed.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1 - np.sum(residual**2) / total) if total > 1e-12 else np.nan,
    }


def evaluate_transfer(train, test, metric, model, context):
    """Predict untouched outcomes; preprocessing and baseline use only training data."""
    fit = fit_regression(train, metric, model)
    supported = test["period"].isin(fit.periods)
    unseen = int((~supported).sum())
    test = test.loc[supported].copy()
    if test.empty:
        raise ValueError("No target jobs have a collection period represented in training")
    prediction = np.asarray(fit.result.predict(fit.design(test)))
    baseline = test["period"].map(train.groupby("period")["success"].mean()).to_numpy()
    in_range = test[metric].between(train[metric].min(), train[metric].max()).to_numpy()
    scores = prediction_scores(test["success"], prediction)
    base_scores = prediction_scores(test["success"], baseline)
    summary = {
        **context,
        "metric": metric,
        "model": model,
        "train_jobs": len(train),
        "test_jobs": len(test),
        "train_configs": train["cluster"].nunique(),
        "test_configs": test["cluster"].nunique(),
        "unseen_period_jobs": unseen,
        "in_range_jobs": int(in_range.sum()),
        "extrapolation_jobs": int((~in_range).sum()),
        "train_metric_min": train[metric].min(),
        "train_metric_max": train[metric].max(),
        "out_of_bounds_predictions": int(((prediction < 0) | (prediction > 1)).sum()),
        **scores,
        "baseline_rmse": base_scores["rmse"],
        "rmse_skill": (
            1 - scores["rmse"] / base_scores["rmse"] if base_scores["rmse"] > 1e-12 else np.nan
        ),
        "in_range_rmse": (
            prediction_scores(test.loc[in_range, "success"], prediction[in_range])["rmse"]
            if in_range.any()
            else np.nan
        ),
        "in_range_r2": (
            prediction_scores(test.loc[in_range, "success"], prediction[in_range])["r2"]
            if in_range.any()
            else np.nan
        ),
        "extrapolation_rmse": (
            prediction_scores(test.loc[~in_range, "success"], prediction[~in_range])["rmse"]
            if (~in_range).any()
            else np.nan
        ),
    }
    rows = test[["job_id", "algorithm", "config", "campaign", "period", "success", "chance"]].copy()
    rows["metric_value"] = test[metric]
    rows["predicted"] = prediction
    rows["baseline_predicted"] = baseline
    rows["within_training_range"] = in_range
    for key, value in {**context, "metric": metric, "model": model}.items():
        rows[key] = value
    return summary, rows


def describe_pair(data, metric, model, family, algorithms):
    """Estimate a slope contrast; failure to reject is never labeled equivalence."""
    fit = fit_regression(data, metric, model, algorithms)
    contrast = np.zeros(len(fit.result.params))
    contrast[list(fit.result.params.index).index("slope_difference")] = 1
    test = fit.result.t_test(contrast)
    interval = np.asarray(test.conf_int()).ravel()
    difference = float(np.asarray(test.effect).item())
    if not np.isfinite(test.pvalue).all() or not np.isfinite(interval).all():
        raise ValueError("Slope contrast has no finite uncertainty estimate")
    diagnostic = {
        "family": family,
        "metric": metric,
        "model": model,
        "algorithm_1": algorithms[0],
        "algorithm_2": algorithms[1],
        "jobs": len(data),
        "configs": fit.clusters,
        "campaigns": data["campaign"].nunique(),
        "periods": len(fit.periods),
        "missing_metric_jobs": 0,
        "metric_scale": fit.scale,
        "slope_1": fit.result.params["x"] / fit.scale,
        "slope_2": (fit.result.params["x"] + difference) / fit.scale,
        "slope_difference": difference / fit.scale,
        "difference_ci_low": interval[0] / fit.scale,
        "difference_ci_high": interval[1] / fit.scale,
        "slope_difference_p": float(np.asarray(test.pvalue).item()),
        "link": "identity" if model == "ols" else "logit",
        "reference_period": fit.periods[0],
        "residual_rmse": prediction_scores(data["success"], fit.result.predict(fit.design(data)))[
            "rmse"
        ],
    }
    return fit, diagnostic


def plot_pair(fit, data, family, output):
    """Export a scientific plot with campaign-adjusted mean curves and robust mean CIs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    for algorithm, color in zip(fit.algorithms, ("#2463a0", "#c45c20")):
        subset = data[data["algorithm"].eq(algorithm)]
        grid = pd.DataFrame(
            {
                fit.metric: np.linspace(subset[fit.metric].min(), subset[fit.metric].max(), 100),
                "period": fit.periods[0],
                "algorithm": algorithm,
            }
        )
        predicted = fit.result.get_prediction(fit.design(grid)).summary_frame()
        axis.scatter(subset[fit.metric], subset["success"], color=color, alpha=0.35, s=15)
        axis.plot(
            grid[fit.metric], predicted["mean"], color=color, label=ALGORITHM_LABELS[algorithm]
        )
        axis.fill_between(
            grid[fit.metric],
            predicted["mean_ci_lower"],
            predicted["mean_ci_upper"],
            color=color,
            alpha=0.15,
        )
    model_label = "linear regression" if fit.model == "ols" else "binomial logistic regression"
    axis.set(
        xlabel=METRIC_LABELS.get(fit.metric, fit.metric),
        ylabel="Success rate",
        title=f"{family.capitalize()} family · {model_label}",
    )
    axis.axhline(0, color="grey", linewidth=0.5)
    axis.axhline(1, color="grey", linewidth=0.5)
    axis.legend()
    figure.text(
        0.01,
        -0.035,
        f"Raw jobs; curves at {fit.periods[0]}. Shading: 95% mean CI.",
        fontsize=8,
    )
    filename = f"{family}-{fit.model}-{fit.metric.replace('.', '_')}.png"
    figure.savefig(output / filename, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return filename


def analyze(data, metrics, output, holdout=0.2, plots=True):
    """Run prespecified univariate comparisons and both directions of transfer per family."""
    comparisons, evaluations, predictions, skipped, figures = [], [], [], [], []
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = output / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for family, algorithms in FAMILIES.items():
        pair = data[data["algorithm"].isin(algorithms)]
        for metric in metrics:
            clean = pair.loc[np.isfinite(pair[metric])].copy()
            for model in MODELS:
                context = {"family": family, "metric": metric, "model": model}
                try:
                    fit, comparison = describe_pair(clean, metric, model, family, algorithms)
                    comparison["missing_metric_jobs"] = len(pair) - len(clean)
                    comparisons.append(comparison)
                    summary_name = f"{family}-{model}-{metric.replace('.', '_')}.txt"
                    (summary_dir / summary_name).write_text(
                        fit.result.summary().as_text(), encoding="utf-8"
                    )
                    if plots:
                        figures.append(plot_pair(fit, clean, family, figure_dir))
                except (ValueError, np.linalg.LinAlgError) as error:
                    skipped.append(
                        {**context, "protocol": "slope_comparison", "reason": str(error)}
                    )
                for source in algorithms:
                    source_data = clean[clean["algorithm"].eq(source)]
                    other = next(algorithm for algorithm in algorithms if algorithm != source)
                    for protocol in ("unseen_algorithm", "larger_configurations"):
                        evaluation_context = {
                            "family": family,
                            "protocol": protocol,
                            "source": source,
                            "target": other if protocol == "unseen_algorithm" else source,
                        }
                        try:
                            if protocol == "unseen_algorithm":
                                train = source_data
                                test = clean[clean["algorithm"].eq(other)]
                            else:
                                train, test = larger_configuration_split(
                                    source_data, metric, holdout
                                )
                            summary, rows = evaluate_transfer(
                                train, test, metric, model, evaluation_context
                            )
                            evaluations.append(summary)
                            predictions.append(rows)
                        except (ValueError, np.linalg.LinAlgError) as error:
                            skipped.append({**context, **evaluation_context, "reason": str(error)})
    comparisons = pd.DataFrame(comparisons)
    if not comparisons.empty:
        comparisons["slope_difference_q_bh"] = multipletests(
            comparisons["slope_difference_p"], method="fdr_bh"
        )[1]
    comparisons.to_csv(output / "slope_comparisons.csv", index=False)
    evaluations = pd.DataFrame(evaluations)
    evaluations.to_csv(output / "heldout_scores.csv", index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(
            output / "heldout_predictions.csv", index=False
        )
    pd.DataFrame(skipped).to_csv(output / "skipped_models.csv", index=False)
    return comparisons, evaluations, figures, skipped


def evaluate_normalized_transfer(train, test, metric, context):
    """Fit OLS to ideal/chance-adjusted success; never treat it as binomial counts."""
    normalized = []
    for frame in (train, test):
        frame = frame.copy()
        gap = frame["ideal_success"] - frame["chance"]
        if not np.isfinite(gap).all() or (gap <= 1e-8).any():
            raise ValueError("Ideal success must exceed chance for normalized analysis")
        frame["success"] = (frame["success"] - frame["chance"]) / gap
        normalized.append(frame)
    summary, rows = evaluate_transfer(*normalized, metric, "ols", context)
    # Values below zero are meaningful (below chance); do not clip them.
    summary["normalized_outside_unit_interval"] = summary.pop("out_of_bounds_predictions")
    original = test.loc[rows.index]
    rows["observed_raw_success"] = original["success"]
    rows["ideal_success"] = original["ideal_success"]
    gap = original["ideal_success"] - original["chance"]
    rows["predicted_raw_success"] = original["chance"] + gap * rows["predicted"]
    summary.update(
        {
            f"raw_{key}": value
            for key, value in prediction_scores(
                original["success"], rows["predicted_raw_success"]
            ).items()
        }
    )
    summary["out_of_bounds_predictions"] = int((~rows["predicted_raw_success"].between(0, 1)).sum())
    return summary, rows


def compare_control_predictions(predictions):
    """Compare family and random-control models on identical held-out target jobs."""
    comparisons = []
    controls = predictions[predictions["protocol"].eq("hardware_control")]
    candidates = predictions[
        predictions["protocol"].isin(["family_transfer", "composition_transfer"])
    ]
    keys = ["target", "metric", "model", "outcome"]
    for values, family in candidates.groupby(["source", *keys]):
        source, *levels = values
        control = controls
        for key, value in zip(keys, levels):
            control = control[control[key].eq(value)]
        shared = family.merge(
            control, on="job_id", suffixes=("_family", "_control"), validate="one_to_one"
        )
        for scope in ("shared_jobs", "within_both_training_ranges"):
            paired = shared
            if scope == "within_both_training_ranges":
                paired = shared[
                    shared["within_training_range_family"] & shared["within_training_range_control"]
                ]
            if paired.empty:
                continue
            if not np.allclose(paired["success_family"], paired["success_control"]):
                raise ValueError("Control comparison has inconsistent target outcomes")
            family_rmse = prediction_scores(paired["success_family"], paired["predicted_family"])[
                "rmse"
            ]
            control_rmse = prediction_scores(paired["success_family"], paired["predicted_control"])[
                "rmse"
            ]
            comparisons.append(
                {
                    "source": source,
                    **dict(zip(keys, levels)),
                    "scope": scope,
                    "jobs": len(paired),
                    "configs": paired["config_family"].nunique(),
                    "family_rmse": family_rmse,
                    "control_rmse": control_rmse,
                    "control_minus_family_rmse": control_rmse - family_rmse,
                }
            )
    return pd.DataFrame(comparisons)


def analyze_extensions(data, metrics, output, holdout=0.2):
    """Keep composition tests and generic hardware controls separate from thesis pairs."""
    experiments = (
        [("composition_transfer", source, "SHOR") for source in ("QFT", "PHASE-ESTIMATION")]
        + [
            ("hardware_control", "RANDOM-VOLUMETRIC", target)
            for target in (*PRIMARY_ALGORITHMS, "SHOR")
        ]
        + [
            ("family_transfer", source, target)
            for pair in FAMILIES.values()
            for source, target in (pair, pair[::-1])
        ]
        + [("larger_configurations", algorithm, algorithm) for algorithm in EXTRA_ALGORITHMS]
    )
    evaluations, predictions, skipped = [], [], []
    for protocol, source, target in experiments:
        for metric in metrics:
            for outcome, model in (
                ("raw_success", "ols"),
                ("raw_success", "binomial_logit"),
                ("ideal_chance_normalized", "ols"),
            ):
                context = {
                    "protocol": protocol,
                    "source": source,
                    "target": target,
                    "outcome": outcome,
                }
                source_data = data[data["algorithm"].eq(source)]
                target_data = data[data["algorithm"].eq(target)]
                train = source_data.loc[np.isfinite(source_data[metric])].copy()
                test = target_data.loc[np.isfinite(target_data[metric])].copy()
                context["source_missing_metric_jobs"] = len(source_data) - len(train)
                context["target_missing_metric_jobs"] = len(target_data) - len(test)
                try:
                    if protocol == "larger_configurations":
                        train, test = larger_configuration_split(train, metric, holdout)
                    if outcome == "ideal_chance_normalized":
                        # Exclude only undefined normalizations, with explicit cohort counts.
                        train_valid = (train["ideal_success"] - train["chance"]).gt(1e-8)
                        test_valid = (test["ideal_success"] - test["chance"]).gt(1e-8)
                        context["source_undefined_normalization_jobs"] = int((~train_valid).sum())
                        context["target_undefined_normalization_jobs"] = int((~test_valid).sum())
                        summary, rows = evaluate_normalized_transfer(
                            train.loc[train_valid], test.loc[test_valid], metric, context
                        )
                    else:
                        summary, rows = evaluate_transfer(train, test, metric, model, context)
                        rows["ideal_success"] = test.loc[rows.index, "ideal_success"]
                    evaluations.append(summary)
                    predictions.append(rows)
                except (ValueError, np.linalg.LinAlgError) as error:
                    skipped.append(
                        {**context, "metric": metric, "model": model, "reason": str(error)}
                    )
    scores = pd.DataFrame(evaluations)
    scores.to_csv(output / "extension_scores.csv", index=False)
    pd.DataFrame(skipped).to_csv(output / "extension_skipped_models.csv", index=False)
    paired = pd.DataFrame()
    if predictions:
        rows = pd.concat(predictions, ignore_index=True)
        rows.to_csv(output / "extension_predictions.csv", index=False)
        paired = compare_control_predictions(rows)
    paired.to_csv(output / "control_comparisons.csv", index=False)
    return scores, paired, skipped


def write_report(output, provenance, comparisons, scores, figures, skipped, r2_threshold):
    """Keep exploratory results separate from confirmatory thesis claims."""
    counts = pd.DataFrame(provenance["algorithm_jobs"].items(), columns=["algorithm", "jobs"])
    lines = [
        "# Scalability generalization: exploratory regression analysis",
        "",
        "For the relationship between success decay in related algorithms, see the "
        "[matched decay profiles](decay/report.md).",
        "",
        f"Backend: **{provenance['backend']}**. Selected jobs: **{provenance['selected_jobs']}**; "
        f"configurations: **{provenance['configuration_count']}**; "
        f"campaigns: **{len(provenance['campaign_jobs'])}**.",
        f"Collection periods (campaign × window): **{len(provenance['period_jobs'])}**.",
        "",
        "```text",
        counts.to_string(index=False),
        "```",
        "",
        "## Method",
        "",
        "Success rate is accepted shots / shots. Error rate is its complement, 1 − success; "
        "declining success means increasing error. Each model has one pre-runtime circuit metric, "
        "an intercept, and campaign/window fixed effects. OLS uses an identity link; binomial GLM uses "
        "accepted/rejected shot counts and a logit link. Standard errors cluster by "
        "algorithm/configuration across every campaign and repetition.",
        "Exported rates are rounded to six decimals. Exact success is reconstructed from "
        "the stored histograms and frozen accepted labels, after checking the rounded rates "
        "agree; binomial counts are not obtained by rounding an approximate rate.",
        "",
        "The descriptive model adds algorithm intercept and slope differences. Its slope contrast "
        "is algorithm 2 minus algorithm 1; coefficients are per original metric unit. BH q-values "
        "adjust all reported slope tests across families, metrics, and model types. A large "
        "p/q-value is NOT evidence of equivalent decay or generalization.",
        "",
        "Unseen-algorithm tests train on one member of a family and predict the other without "
        "using target outcomes. Larger-configuration tests hold out the upper tail of configuration "
        "median metric values, preserving all repetitions and campaigns of a configuration. "
        "Scaling and per-period mean baselines are learned from training data only. Unseen "
        "collection periods cannot be predicted by fixed effects and are explicitly counted as unsupported. "
        "Metric range overlap is reported separately from extrapolation. OLS predictions are not clipped.",
        "",
        "## Interpretation limits",
        "",
        "- This is exploratory metric screening, not a locked confirmatory test. Selecting the best "
        "metric using these holdouts requires a fresh campaign or nested validation for confirmation.",
        "- The thesis pairs QFT/PE, Grover/BV, and vTP/GHZ. An oracle-based pairing does not imply "
        "BV contains Grover, nor does GHZ contain teleportation. These tests assess paradigm-level "
        "transfer, not arbitrary composition of primitives. Shor composition tests and random-volumetric "
        "controls are reported separately below; all eight algorithms enter the analysis.",
        "- Success criteria and random-chance floors differ across algorithms/configurations. "
        "Raw-success transfer alone cannot establish a universal physical noise-decay law. "
        "Chance baselines are included in held-out predictions for sensitivity analysis, never as predictors.",
        "- Campaign/window effects reduce observed period confounding; they do not remove within-period "
        "calibration, layout, or structural differences. One backend cannot establish cross-provider transfer.",
        "- Missing metrics are excluded per feature, not imputed. Rank deficiency, separation, "
        "insufficient independent configurations, and convergence failures are logged in skipped_models.csv.",
        "- Confidence bands describe fitted mean uncertainty, not prediction intervals for future jobs.",
        "",
        "## Held-out performance",
        "",
        f"The thesis R² ≥ {r2_threshold:.2f} criterion is displayed per prespecified candidate; "
        "MAE, RMSE, and improvement over the training-only campaign baseline must also be assessed. "
        "Negative R² values and out-of-bounds OLS predictions are retained.",
        "",
    ]
    if not scores.empty:
        summary = scores.groupby(["protocol", "model"]).agg(
            tests=("r2", "size"),
            median_r2=("r2", "median"),
            median_mae=("mae", "median"),
            median_rmse=("rmse", "median"),
        )
        lines += ["```text", summary.to_string(), "```", ""]
        passing = scores[scores["r2"].ge(r2_threshold)]
        lines += [
            f"Candidates reaching the R² criterion: {len(passing)} / {len(scores)}. "
            "These are candidate-level results, not acceptance of H2.",
            "",
        ]
        primary = (
            scores[scores["protocol"].eq("unseen_algorithm") & scores["model"].eq("binomial_logit")]
            .sort_values("r2", ascending=False)
            .drop_duplicates(["family", "source", "target"])
        )
        lines += [
            "Best observed binomial transfer scores among the explored metrics "
            "(post hoc screening; use a fresh holdout to confirm):",
            "",
            "```text",
            primary[["family", "source", "target", "metric", "r2", "mae", "rmse"]].to_string(
                index=False
            ),
            "```",
            "",
        ]
    lines += [
        "## Artifacts",
        "",
        "- `provenance.json`: hashed inputs, selected campaigns, software versions, and settings.",
        "- `slope_comparisons.csv`: slopes, cluster-robust contrast intervals, p/q-values, missingness.",
        "- `summaries/`: full statsmodels descriptive fit summaries and coefficient tables. "
        "Their residual normality diagnostics are descriptive; they do not account for clustering.",
        "- `heldout_scores.csv`: all algorithm-transfer and larger-configuration evaluations.",
        "- `heldout_predictions.csv`: per-job held-out predictions, chance floors, baselines, support.",
        f"- `skipped_models.csv`: {len(skipped)} explicitly unsupported fits/evaluations.",
        "",
        "## Decay curves",
        "",
    ]
    for filename in figures:
        lines += [f"![{filename}](figures/{filename})", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_extension_report(output, scores, paired, skipped):
    """Explain the extension's endpoints, controls, and period support beside its results."""
    lines = [
        "",
        "## Shor composition and random-volumetric controls",
        "",
        "QFT → Shor and PE → Shor hold all Shor outcomes out of training. Random-volumetric "
        "models predict each of the seven algorithmic workloads as a generic hardware control. "
        "Shor and random volumetric also receive separate larger-configuration holdouts.",
        "",
        "Raw success uses OLS and binomial GLM. A separate OLS sensitivity analysis uses "
        "(success − chance) / (ideal success − chance), with both reference probabilities taken "
        "from frozen circuit definitions, never measured target outcomes. Zero means chance and "
        "one means ideal task success. Undefined denominators are excluded and counted; values "
        "are never clipped. This response is not binomial. Normalized scores and raw-success "
        "scores have different units; raw_* columns score the normalized model after converting "
        "its predictions back to each target's raw probability scale.",
        "",
        "Current Shor references have ideal order-recovery success 0.5, versus 1.0 for the "
        "mirror circuits. Shor success means order recovery, not completed factorization. "
        "The exported Shor configurations all use N=15: width scaling does not demonstrate "
        "generalization to larger integers. The Shor runner requests dynamical decoupling and "
        "gate twirling; algorithm-specific execution settings are not removed by period effects. "
        "Mirror circuits have inverse correlations and a fixed seed in these configurations, "
        "so this control is not a universal noise benchmark.",
        "",
        "Collection periods absent from the source remain unsupported, including the Shor-only "
        "growth campaign for QFT/PE/RV transfer. All jobs remain in the cohort; unsupported "
        "prediction counts are explicit. A failure on raw success alone cannot distinguish "
        "structural transfer failure from task-definition and execution-setting differences.",
        "",
        "`control_comparisons.csv` compares family/composition models with RV models using "
        "the same target job IDs, metric, outcome, and model. Positive control_minus_family_rmse "
        "favors the family model. Shared jobs and jobs within both training ranges are reported "
        "separately. These are descriptive error differences, not significance tests or evidence "
        "that circuit-family membership causes an improvement.",
        "",
    ]
    if not scores.empty:
        composition = scores[scores["protocol"].eq("composition_transfer")]
        best = composition.sort_values("r2", ascending=False).drop_duplicates(
            ["source", "outcome", "model"]
        )
        lines += [
            "Best observed Shor-transfer candidates (exploratory selection):",
            "",
            "```text",
            best[
                [
                    "source",
                    "outcome",
                    "model",
                    "metric",
                    "r2",
                    "mae",
                    "test_jobs",
                    "unseen_period_jobs",
                ]
            ].to_string(index=False),
            "```",
            "",
        ]
    lines += [
        f"Extension evaluations: {len(scores)}; paired control comparisons: {len(paired)}; "
        f"unsupported cases: {len(skipped)}.",
        "",
        "Tables: `extension_scores.csv`, `extension_predictions.csv`, "
        "`extension_skipped_models.csv`, and `control_comparisons.csv`. "
        "The original six-algorithm tables and slope tests retain their original scope.",
        "",
    ]
    report = output / "report.md"
    content = report.read_text(encoding="utf-8")
    section = "\n".join(lines)
    marker = "## Decay curves"
    content = content.replace(marker, section + "\n" + marker, 1)
    report.write_text(content, encoding="utf-8")


def main():
    """Analyze existing exports with reproducible, explicit cohort settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=PAPERS / "dataset_out")
    parser.add_argument("--out", type=Path, default=Path("report/scalability_regression"))
    parser.add_argument("--backend", default="ibm_fez")
    parser.add_argument(
        "--campaign", action="append", help="Repeat to select campaigns; default: all"
    )
    parser.add_argument("--metrics", nargs="+", default=list(METRICS))
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--r2-threshold", type=float, default=0.8)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--group-tolerance",
        type=float,
        default=5.0,
        help="Maximum median gap in fitted change, in target points (default 5)",
    )
    parser.add_argument(
        "--decay-only",
        action="store_true",
        help="Compare observed decay profiles on shared resource levels",
    )
    args = parser.parse_args()
    if not np.isfinite(args.group_tolerance) or args.group_tolerance <= 0:
        parser.error("--group-tolerance must be positive and finite")
    if not 0 < args.holdout_fraction < 1:
        parser.error("--holdout-fraction must be strictly between zero and one")
    data, provenance = load_data(args.tables, args.metrics, args.backend, args.campaign)
    args.out.mkdir(parents=True, exist_ok=True)
    decay_dir = args.out / "decay"
    decay_relationships.analyze_decay(
        data, args.metrics, FAMILIES, decay_dir, not args.no_plots, args.group_tolerance
    )
    decay_provenance = {
        **provenance,
        "minimum_shared_levels": 3,
        "group_tolerance_points": args.group_tolerance,
        "group_source_sha256": hashlib.sha256(Path(decay_groups.__file__).read_bytes()).hexdigest(),
        "decay_source_sha256": hashlib.sha256(
            Path(decay_relationships.__file__).read_bytes()
        ).hexdigest(),
    }
    (decay_dir / "provenance.json").write_text(
        json.dumps(decay_provenance, indent=2), encoding="utf-8"
    )
    print(f"Decay relationships: {(decay_dir / 'report.md').resolve()}")
    if args.decay_only:
        return
    provenance.update(
        {"holdout_fraction": args.holdout_fraction, "r2_threshold": args.r2_threshold}
    )
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    comparisons, scores, figures, skipped = analyze(
        data, args.metrics, args.out, args.holdout_fraction, not args.no_plots
    )
    write_report(args.out, provenance, comparisons, scores, figures, skipped, args.r2_threshold)
    extension_scores, paired, extension_skipped = analyze_extensions(
        data, args.metrics, args.out, args.holdout_fraction
    )
    write_extension_report(args.out, extension_scores, paired, extension_skipped)
    print(f"Analyzed {len(data)} jobs across {data['campaign'].nunique()} campaigns.")
    print(
        f"Slope comparisons: {len(comparisons)}; held-out evaluations: {len(scores)}; "
        f"unsupported cases: {len(skipped)}."
    )
    print(
        f"Extension evaluations: {len(extension_scores)}; paired control comparisons: "
        f"{len(paired)}; unsupported extension cases: {len(extension_skipped)}."
    )
    print(f"Report: {(args.out / 'report.md').resolve()}")


if __name__ == "__main__":
    main()
