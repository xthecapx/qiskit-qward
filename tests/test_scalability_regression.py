"""Offline regression safeguards: counts, repeated circuits, and untouched holdouts."""

import json

import numpy as np
import pandas as pd
import pytest

from qward.examples.papers import scalability_regression as analysis
from qward.examples.papers import decay_relationships
from qward.examples.papers.decay_groups import group_by_decay


def test_complete_linkage_ties_chains_and_unknown_gaps():
    # A–B and B–C tie. Lexical order picks A–B, and A–C prevents chaining.
    gaps = {("A", "B"): 1.0, ("B", "C"): 1.0, ("A", "C"): 9.0}
    expected = [("A", "B"), ("C",), ("D",)]
    assert group_by_decay(["D", "C", "B", "A"], gaps, 5.0) == expected
    assert group_by_decay(["B", "A", "D", "C"], dict(reversed(list(gaps.items()))), 5.0) == expected
    assert group_by_decay(["A", "B", "C"], {("A", "B"): 1.0, ("B", "C"): 2.0}, 5.0) == [
        ("A", "B"),
        ("C",),
    ]


def test_agreeing_profiles_can_have_different_local_decay_and_slopes():
    levels = pd.DataFrame(
        {"first": [1.0, 0.9, 0.7, 0.4], "second": [0.6, 0.45, 0.35, 0.30]}, index=[2, 3, 4, 5]
    )
    result = decay_relationships.paired_decay(levels)
    assert result["profile_spearman"] == pytest.approx(1)
    assert result["step_rate_spearman"] == pytest.approx(-1)
    assert result["slope_1"] < result["slope_2"] < 0
    assert result["both_decline"] == 3


def test_decay_matches_period_and_level_and_weights_configurations_equally(tmp_path):
    rows = []
    for algorithm, widths in [
        ("QFT", [3, 4, 5, 6]),
        ("PHASE-ESTIMATION", [3, 4, 5, 7]),
        ("RANDOM-VOLUMETRIC", [3, 4, 5, 8]),
    ]:
        for x in widths:
            for config, repeats, success in [("a", 8, 0.1), ("b", 1, 0.9)]:
                rows.extend(
                    [
                        dict(
                            algorithm=algorithm,
                            config=config,
                            cluster=f"{algorithm}:{x}:{config}",
                            backend="chip",
                            period="full",
                            width=x,
                            success=success,
                            ideal_success=1,
                            chance=0.1,
                        )
                    ]
                    * repeats
                )
    data = pd.DataFrame(rows)
    lone = data[data.algorithm.eq("QFT")].assign(period="pilot")
    decay_relationships.analyze_decay(
        pd.concat([data, lone]), ["width"], analysis.FAMILIES, tmp_path, plots=False
    )
    matched = pd.read_csv(tmp_path / "matched_levels.csv")
    pair = matched[(matched.role == "fourier") & (matched.outcome == "success")]
    assert pair.metric_value.tolist() == [3, 4, 5]
    assert pair.period.eq("full").all()
    assert np.allclose(pair.mean_1, 0.5)
    assert pair.configs_1.eq(2).all()
    controls = pd.read_csv(tmp_path / "matched_controls.csv")
    assert controls.shared_values.eq("3,4,5").all() and len(controls) == 2


@pytest.fixture
def cohort():
    """Two algorithms with intentionally different slopes and three collection periods."""
    random = np.random.default_rng(413)
    rows = []
    for algorithm, slope in [("QFT", -0.12), ("PHASE-ESTIMATION", -0.32)]:
        for campaign, offset in [("early", 0), ("middle", 0.2), ("late", -0.3)]:
            for config in range(20):
                for repetition in range(2):
                    probability = 1 / (1 + np.exp(-(2 + offset + slope * config)))
                    accepted = random.binomial(1024, probability)
                    rows.append(
                        {
                            "algorithm": algorithm,
                            "campaign": campaign,
                            "period": campaign,
                            "config": str(config),
                            "cluster": f"{algorithm}:{config}",
                            "job_id": f"{algorithm}:{campaign}:{config}:{repetition}",
                            "shots": 1024,
                            "accepted": accepted,
                            "success": accepted / 1024,
                            "chance": 0.1,
                            "ideal_success": 1.0,
                            "a.depth": float(config),
                        }
                    )
    return pd.DataFrame(rows)


def exported_tables(tmp_path, success=0.5):
    """A minimal on-disk export, with the same integrity contract as the real dataset."""
    frame = pd.DataFrame(
        {
            "meta.algorithm": ["QFT", "PHASE-ESTIMATION"],
            "meta.backend_name": ["ibm_fez"] * 2,
            "meta.campaign_id": ["campaign"] * 2,
            "meta.window": [1, 1],
            "meta.config_id": ["a", "b"],
            "meta.job_id": ["job-a", "job-b"],
            "meta.run_id": ["run-a", "run-b"],
            "meta.shots": [10, 10],
            "f.fidelity.success_rate": [success, 0.6],
            "f.fidelity.dsr": [0.3, None],
            "f.fidelity.chance_baseline": [0.25, 0.25],
            "f.counts_json": [json.dumps({"00": 5, "01": 5}), json.dumps({"00": 6, "01": 4})],
            "a.depth": [2, 4],
        }
    )
    path = tmp_path / "dataset-001.csv"
    frame.to_csv(path, index=False)
    pd.DataFrame(
        {
            "run_id": ["run-a", "run-b"],
            "accepted_json": [json.dumps(["00"])] * 2,
            "ideal_json": [json.dumps({"00": 1.0})] * 2,
        }
    ).to_csv(tmp_path / "references.csv", index=False)
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "rows": 2,
                "parts": [{"file": path.name, "rows": 2, "bytes": path.stat().st_size}],
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_export_count_integrity_and_pre_runtime_only(tmp_path):
    tables = exported_tables(tmp_path)
    data, provenance = analysis.load_data(tables, ["a.depth"])
    assert data["accepted"].tolist() == [5, 6]
    assert data["error"].tolist() == pytest.approx([0.5, 0.4])
    assert provenance["selected_jobs"] == 2
    assert provenance["exported_target_jobs"]["dsr"] == 1
    assert provenance["exported_target_jobs"]["hellinger_fidelity"] == 0
    assert data.loc[0, "dsr"] == pytest.approx(0.3)
    assert len(provenance["inputs"][0]["sha256"]) == 64
    with pytest.raises(ValueError, match="Predictors must be circuit metrics"):
        analysis.load_data(tables, ["f.fidelity.success_rate"])
    with pytest.raises(ValueError, match="Predictors must be circuit metrics"):
        analysis.load_data(tables, ["a.quantum_specific.compute_seconds"])


def test_noninteger_accepted_shots_are_rejected(tmp_path):
    tables = exported_tables(tmp_path, success=0.55)
    with pytest.raises(ValueError, match="integer accepted shots"):
        analysis.load_data(tables, ["a.depth"])


def test_rounded_export_rate_is_reconstructed_from_exact_histogram(tmp_path):
    tables = exported_tables(tmp_path)
    path = tables / "dataset-001.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "meta.shots"] = 1024
    frame.loc[0, "f.fidelity.success_rate"] = 0.756836
    frame.loc[0, "f.counts_json"] = json.dumps({"00": 775, "01": 249})
    frame.to_csv(path, index=False)
    manifest_path = tables / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["parts"][0]["bytes"] = path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    data, provenance = analysis.load_data(tables, ["a.depth"])
    assert data.iloc[0]["accepted"] == 775
    assert data.iloc[0]["success"] == 775 / 1024
    assert 0 < provenance["maximum_export_rate_rounding_correction"] < 5.1e-7


def test_manifest_mismatch_is_not_silently_used(tmp_path):
    tables = exported_tables(tmp_path)
    with (tables / "dataset-001.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="size mismatch"):
        analysis.load_data(tables, ["a.depth"])


def test_size_split_keeps_every_campaign_and_repetition_together(cohort):
    source = cohort[cohort["algorithm"].eq("QFT")]
    train, test = analysis.larger_configuration_split(source, "a.depth")
    assert set(train["cluster"]).isdisjoint(test["cluster"])
    assert train["a.depth"].max() < test["a.depth"].min()
    assert train["config"].nunique() == 16
    assert test["config"].nunique() == 4
    assert test.groupby("cluster").size().eq(6).all()


@pytest.mark.parametrize("model", analysis.MODELS)
def test_transfer_predictions_ignore_target_outcomes(cohort, model):
    train = cohort[cohort["algorithm"].eq("QFT")]
    test = cohort[cohort["algorithm"].eq("PHASE-ESTIMATION")].copy()
    summary, first = analysis.evaluate_transfer(train, test, "a.depth", model, {})
    test["success"] = 1 - test["success"]
    test["accepted"] = test["shots"] - test["accepted"]
    _, changed = analysis.evaluate_transfer(train, test, "a.depth", model, {})
    np.testing.assert_allclose(first["predicted"], changed["predicted"])
    np.testing.assert_allclose(first["baseline_predicted"], changed["baseline_predicted"])
    fit = analysis.fit_regression(train, "a.depth", model)
    assert fit.center == train["a.depth"].mean()
    assert (
        fit.clusters == 20
    )  # 120 jobs are 20 circuit configurations, not 120 independent circuits.
    assert summary["train_configs"] == summary["test_configs"] == 20


@pytest.mark.parametrize("model", analysis.MODELS)
def test_interaction_detects_different_slopes_on_common_scale(cohort, model):
    fit, result = analysis.describe_pair(
        cohort, "a.depth", model, "fourier", analysis.FAMILIES["fourier"]
    )
    assert result["slope_2"] < result["slope_1"] < 0
    assert result["difference_ci_high"] < 0
    assert result["slope_difference_p"] < 0.05
    assert fit.clusters == 40
    assert result["campaigns"] == 3


def test_extrapolation_and_unseen_campaigns_are_explicit(cohort):
    train = cohort[cohort["algorithm"].eq("QFT")]
    test = cohort[cohort["algorithm"].eq("PHASE-ESTIMATION")].copy()
    test["a.depth"] += 30
    test.loc[test.index[:2], "campaign"] = "never-seen"
    test.loc[test.index[:2], "period"] = "never-seen"
    result, rows = analysis.evaluate_transfer(train, test, "a.depth", "binomial_logit", {})
    assert result["unseen_period_jobs"] == 2
    assert result["extrapolation_jobs"] == len(test) - 2
    assert result["in_range_jobs"] == 0
    assert not rows["within_training_range"].any()
    assert rows["predicted"].between(0, 1).all()
    fit = analysis.fit_regression(train, "a.depth", "ols")
    with pytest.raises(ValueError, match="unseen collection period"):
        fit.design(test)


def test_campaign_confounding_and_constant_targets_are_not_hidden(cohort):
    source = cohort[cohort["algorithm"].eq("QFT")].copy()
    source["a.depth"] = source["campaign"].map({"early": 1, "middle": 2, "late": 3})
    with pytest.raises(ValueError, match="Rank-deficient"):
        analysis.fit_regression(source, "a.depth", "ols")
    assert np.isnan(analysis.prediction_scores([1, 1], [0.5, 0.6])["r2"])


def test_windows_within_one_campaign_remain_separate_controls(cohort):
    data = cohort.copy()
    data["campaign"] = "shared-campaign"
    fit, result = analysis.describe_pair(
        data, "a.depth", "ols", "fourier", analysis.FAMILIES["fourier"]
    )
    assert len(fit.periods) == 3
    assert result["campaigns"] == 1
    assert result["periods"] == 3
    assert {"period_1", "period_2"}.issubset(fit.result.params.index)


def test_analysis_exports_failures_and_does_not_claim_equivalence(cohort, tmp_path):
    comparisons, scores, figures, skipped = analysis.analyze(
        cohort, ["a.depth"], tmp_path, plots=False
    )
    assert len(comparisons) == 2
    assert len(scores) == 8
    assert comparisons["slope_difference_q_bh"].between(0, 1).all()
    assert figures == []
    assert skipped  # The fixture deliberately has no oracle/entanglement families.
    assert (tmp_path / "heldout_predictions.csv").exists()
    assert (tmp_path / "skipped_models.csv").exists()


def test_export_includes_shor_and_random_volumetric(tmp_path):
    tables = exported_tables(tmp_path)
    path = tables / "dataset-001.csv"
    frame = pd.read_csv(path)
    frame["meta.algorithm"] = ["SHOR", "RANDOM-VOLUMETRIC"]
    frame.to_csv(path, index=False)
    manifest_path = tables / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["parts"][0]["bytes"] = path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    data, provenance = analysis.load_data(tables, ["a.depth"])
    assert set(data["algorithm"]) == set(analysis.EXTRA_ALGORITHMS)
    assert data["ideal_success"].eq(1).all()
    assert provenance["excluded_jobs"] == 0


def test_normalized_transfer_uses_ideal_references_without_target_leakage(cohort):
    train = cohort[cohort["algorithm"].eq("QFT")]
    test = cohort[cohort["algorithm"].eq("PHASE-ESTIMATION")].copy()
    test["ideal_success"] = 0.5
    test["success"] *= 0.5
    _, first = analysis.evaluate_normalized_transfer(train, test, "a.depth", {})
    expected = test["chance"] + 0.4 * first["predicted"]
    np.testing.assert_allclose(first["predicted_raw_success"], expected)
    test["success"] = 0  # Below chance is retained, not clipped to zero.
    _, changed = analysis.evaluate_normalized_transfer(train, test, "a.depth", {})
    np.testing.assert_allclose(first["predicted_raw_success"], changed["predicted_raw_success"])
    assert changed["success"].eq(-0.25).all()
    test["ideal_success"] = test["chance"]
    with pytest.raises(ValueError, match="must exceed chance"):
        analysis.evaluate_normalized_transfer(train, test, "a.depth", {})


def test_control_comparison_matches_jobs_and_training_ranges():
    rows = []
    for protocol, source, jobs, predictions in [
        ("composition_transfer", "QFT", ["shared", "family-only"], [0.5, 0.9]),
        ("hardware_control", "RANDOM-VOLUMETRIC", ["shared", "control-only"], [0.8, 0.1]),
    ]:
        for job, prediction in zip(jobs, predictions):
            rows.append(
                {
                    "protocol": protocol,
                    "source": source,
                    "target": "SHOR",
                    "job_id": job,
                    "config": job,
                    "metric": "a.depth",
                    "model": "ols",
                    "outcome": "raw_success",
                    "success": 0.5,
                    "predicted": prediction,
                    "within_training_range": source == "QFT",
                }
            )
    paired = analysis.compare_control_predictions(pd.DataFrame(rows))
    assert len(paired) == 1  # No jobs are within both ranges.
    assert paired.iloc[0]["jobs"] == 1
    assert paired.iloc[0]["control_minus_family_rmse"] == pytest.approx(0.3)


def test_extensions_hold_out_shor_and_include_extra_size_tests(cohort, tmp_path):
    extras = []
    for algorithm in analysis.EXTRA_ALGORITHMS:
        extra = cohort[cohort["algorithm"].eq("QFT")].copy()
        extra["algorithm"] = algorithm
        extra["cluster"] = algorithm + ":" + extra["config"]
        extra["job_id"] = algorithm + ":" + extra["job_id"]
        extras.append(extra)
    data = pd.concat([cohort, *extras], ignore_index=True)
    scores, paired, _ = analysis.analyze_extensions(data, ["a.depth"], tmp_path)
    composition = scores[scores["protocol"].eq("composition_transfer")]
    assert set(composition["source"]) == {"QFT", "PHASE-ESTIMATION"}
    assert composition["target"].eq("SHOR").all()
    size = scores[scores["protocol"].eq("larger_configurations")]
    assert set(size["target"]) == set(analysis.EXTRA_ALGORITHMS)
    assert not paired.empty
    assert set(scores["outcome"]) == {"raw_success", "ideal_chance_normalized"}
    assert (tmp_path / "extension_predictions.csv").exists()
