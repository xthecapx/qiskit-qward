"""Reproduce an exploratory contrast/rank-screening study; never edits source data.

Run from the repository root with .venv/bin/python and a writable MPLCONFIGDIR.
All policies are illustrative sensitivity checks, not fitted or recommended cutoffs.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
import platform
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/qward-matplotlib")
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import binom, binomtest

from qward.metrics.differential_success_rate import compute_dsr_michelson

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / "DSR_result.csv"
SEED = 20260909
REPEATS = 30
BUDGETS = (64, 128, 256, 512)
POLICIES = (
    "inspect_all",
    "unique_mode",
    "contrast_0.2",
    "contrast_0.5",
    "contrast_0.8",
    "rank_0.05",
    "rank_0.01",
    "contrast_0.2_and_rank_0.05",
    "gap_0.05",
)


@lru_cache(maxsize=None)
def rank_p(n1: int, n2: int) -> float:
    """Hung--Fithian multinomial winner verification: TWO-sided exact test."""
    return min(1.0, float(2 * binom.sf(n1 - 1, n1 + n2, 0.5)))


def screen(counts: np.ndarray) -> dict:
    """No expected outcomes or target count enter the screening calculation."""
    total = int(counts.sum())
    assert total > 0
    if len(counts) > 1:
        pair = np.partition(counts, -2)[-2:]
        n2, n1 = sorted(map(int, pair))
    else:
        n1, n2 = int(counts[0]), 0
    contrast = (n1 - n2) / (n1 + n2)
    return {
        "n1": n1,
        "n2": n2,
        "contrast": contrast,
        "pair_mass": (n1 + n2) / total,
        "top_mass": n1 / total,
        "gap": (n1 - n2) / total,
        "rank_p": rank_p(n1, n2),
        "unique": n1 > n2,
    }


def policies(s: dict) -> dict:
    return {
        "inspect_all": True,
        "unique_mode": s["unique"],
        "contrast_0.2": s["contrast"] >= 0.2,
        "contrast_0.5": s["contrast"] >= 0.5,
        "contrast_0.8": s["contrast"] >= 0.8,
        "rank_0.05": s["rank_p"] <= 0.05,
        "rank_0.01": s["rank_p"] <= 0.01,
        "contrast_0.2_and_rank_0.05": s["contrast"] >= 0.2 and s["rank_p"] <= 0.05,
        "gap_0.05": s["gap"] >= 0.05,
    }


def target_metrics(hist: dict, expected: set) -> dict:
    total = sum(hist.values())
    target_counts = [hist.get(x, 0) for x in expected]
    mass = sum(target_counts)
    competitor = max((n for x, n in hist.items() if x not in expected), default=0)
    mean = mass / len(expected)

    def contrast(a: float, b: float) -> float:
        return (a - b) / (a + b) if a + b else 0.0

    signed = contrast(mean, competitor)
    return {
        "target_mass": mass / total,
        "wrong_max_count": competitor,
        "target_mean_signed": signed,
        "target_dsr": max(0.0, signed),
        "target_sum_signed": contrast(mass, competitor),
        "target_min_signed": contrast(min(target_counts), competitor),
        "target_coverage": sum(n > 0 for n in target_counts) / len(expected),
        "target_min_count": min(target_counts),
        "target_max_count": max(target_counts),
    }


def load_data() -> tuple[pd.DataFrame, list]:
    raw = pd.read_csv(SOURCE, dtype={"expected_outcomes": str})
    rows, parsed = [], []
    for row_id, row in raw.iterrows():
        hist = json.loads(row.histogram)
        expected = set(row.expected_outcomes.split(","))
        widths = {len(x) for x in expected}
        assert len(widths) == 1
        width = widths.pop()
        assert hist and all(isinstance(n, int) and n >= 0 for n in hist.values())
        assert sum(hist.values()) == row.shots
        assert all(set(x) <= {"0", "1"} and len(x) >= width for x in hist)
        aligned = any(len(x) != width for x in hist)
        # Follow each dataset's extraction convention, not a universal bit order.
        # QFT: enrich_hellinger keeps the left register. Teleportation:
        # build_csv_from_json._compute_teleportation_fidelity keeps the right payload.
        marginal = Counter()
        for state, count in hist.items():
            output_state = state[-width:] if row.algorithm == "TELEPORTATION" else state[:width]
            marginal[output_state] += count
        hist = dict(marginal)
        labels = np.array(sorted(hist))
        counts = np.array([hist[x] for x in labels], dtype=np.int64)
        s = screen(counts)
        targets = target_metrics(hist, expected)
        native = compute_dsr_michelson(hist, expected)
        assert np.isclose(native, targets["target_dsr"], atol=1e-12)
        tied = labels[counts == s["n1"]]
        # Expected correctness under uniform random tie-breaking, no all-zero preference.
        candidate_correct = sum(x in expected for x in tied) / len(tied)
        cohort = (
            "primary"
            if row.algorithm != "TELEPORTATION"
            else (
                "teleportation_simulator"
                if row.execution_type == "SIMULATION"
                else "teleportation_hardware"
            )
        )
        record = {
            key: row[key]
            for key in (
                "algorithm",
                "config_id",
                "backend_name",
                "execution_type",
                "shots",
                "num_qubits",
                "optimization_level",
                "transpiled_depth",
                "result_id",
                "source_file",
            )
        }
        record.update(
            row_id=row_id,
            cohort=cohort,
            output_width=width,
            k=len(expected),
            target_group="singleton" if len(expected) == 1 else "multiple",
            register_aligned=aligned,
            candidate_correct=candidate_correct,
            candidate=str(tied[0]) if len(tied) == 1 else "TIED",
            stored_dsr=float(row.dsr_michelson),
            **s,
            **targets,
        )
        record.update({f"pass_{name}": flag for name, flag in policies(s).items()})
        rows.append(record)
        parsed.append((labels, counts, expected))
    return pd.DataFrame(rows), parsed


def summarize(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    records = []
    for key, block in frame.groupby(groups, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        for policy in POLICIES:
            selected = block[f"pass_{policy}"].astype(bool)
            n, n_selected = len(block), int(selected.sum())
            good = float(block.loc[selected, "candidate_correct"].sum())
            all_good = float(block.candidate_correct.sum())
            records.append(
                dict(zip(groups, key))
                | {
                    "policy": policy,
                    "jobs": n,
                    "selected": n_selected,
                    "correct_selected": good,
                    "wrong_selected": n_selected - good,
                    "correct_deferred": all_good - good,
                    "selection_rate": n_selected / n,
                    "correct_fraction_selected": good / n_selected if n_selected else np.nan,
                    "correct_candidate_retention": good / all_good if all_good else np.nan,
                }
            )
    return pd.DataFrame(records)


def split_study(frame: pd.DataFrame, parsed: list) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    records = []
    primary = frame[(frame.cohort == "primary") & (frame.shots == 1024)]
    for i, row in enumerate(primary.itertuples()):
        labels, counts, expected = parsed[row.row_id]
        assert counts.sum() == 1024
        for budget in BUDGETS:
            for repeat in range(REPEATS):
                discovery = rng.multivariate_hypergeometric(counts, budget)
                remaining = counts - discovery
                s = screen(discovery)
                candidate_idx = int(rng.choice(np.flatnonzero(discovery == s["n1"])))
                candidate = labels[candidate_idx]
                hold_count = remaining[candidate_idx]
                hold_other = np.max(np.delete(remaining, candidate_idx), initial=0)
                records.append(
                    {
                        "row_id": row.row_id,
                        "algorithm": row.algorithm,
                        "config_id": row.config_id,
                        "backend_name": row.backend_name,
                        "target_group": row.target_group,
                        "budget": budget,
                        "repeat": repeat,
                        "candidate_correct": candidate in expected,
                        "holdout_unique_leader": hold_count > hold_other,
                        "holdout_candidate_mass": hold_count / remaining.sum(),
                        **s,
                        **{f"pass_{name}": flag for name, flag in policies(s).items()},
                    }
                )
        if (i + 1) % 200 == 0:
            print(f"Split study: {i + 1}/{len(primary)} jobs", flush=True)
    return pd.DataFrame(records)


def summarize_splits(splits: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (group, budget), block in splits.groupby(["target_group", "budget"]):
        for policy in POLICIES:
            selected = block[f"pass_{policy}"].astype(bool)
            chosen = block[selected]
            records.append(
                {
                    "target_group": group,
                    "budget": budget,
                    "policy": policy,
                    "jobs": block.row_id.nunique(),
                    "repeats": REPEATS,
                    "mean_selected_jobs": selected.sum() / REPEATS,
                    "selection_rate": selected.mean(),
                    "correct_fraction_selected": chosen.candidate_correct.mean(),
                    "holdout_unique_leader_fraction": chosen.holdout_unique_leader.mean(),
                    "mean_correct_selected_jobs": chosen.candidate_correct.sum() / REPEATS,
                    "mean_wrong_selected_jobs": (~chosen.candidate_correct).sum() / REPEATS,
                }
            )
    return pd.DataFrame(records)


def controls() -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    records = []
    for width in (2, 8, 16, 28):
        for shots in (64, 256, 1024):
            for repeat in range(1000):
                draws = rng.integers(0, 2**width, size=shots)
                _, counts = np.unique(draws, return_counts=True)
                s = screen(counts)
                records.append(
                    {
                        "scenario": "uniform",
                        "width": width,
                        "shots": shots,
                        "repeat": repeat,
                        **s,
                        **{f"pass_{name}": flag for name, flag in policies(s).items()},
                    }
                )
    result = pd.DataFrame(records)
    result.groupby(["scenario", "width", "shots"]).agg(
        **{name: (f"pass_{name}", "mean") for name in POLICIES},
        mean_contrast=("contrast", "mean"),
        mean_gap=("gap", "mean"),
    ).reset_index().to_csv(OUT / "uniform_controls.csv", index=False)
    return result


def checks(frame: pd.DataFrame) -> dict:
    c = frame.contrast.to_numpy()
    ratio = frame.n1 / (frame.n1 + frame.n2)
    margin = (frame.n1 - frame.n2) / frame.n1
    assert np.allclose(ratio, (1 + c) / 2)
    assert np.allclose(margin, 2 * c / (1 + c))
    assert np.allclose(frame.gap, frame.contrast * frame.pair_mass)
    for threshold in (0.2, 0.5, 0.8):
        # Equal rational boundaries can round differently after the transformation.
        assert np.array_equal(c >= threshold - 1e-12, ratio >= (1 + threshold) / 2 - 1e-12)
        assert np.array_equal(
            c >= threshold - 1e-12, margin >= 2 * threshold / (1 + threshold) - 1e-12
        )
    for n1, n2 in ((1, 0), (5, 0), (6, 0), (2, 1), (15, 5), (150, 50), (500, 500)):
        assert np.isclose(rank_p(n1, n2), binomtest(n1, n1 + n2).pvalue)
    assert screen(np.array([500, 500]))["contrast"] == 0
    assert rank_p(5, 0) > 0.05 and rank_p(6, 0) < 0.05
    assert screen(np.array([12, 4]))["contrast"] == screen(np.array([120, 40]))["contrast"]
    assert rank_p(12, 4) > 0.05 and rank_p(120, 40) < 0.05
    singleton_correct = frame[(frame.k == 1) & (frame.candidate_correct == 1)]
    assert np.allclose(singleton_correct.contrast, singleton_correct.target_dsr)
    error = (frame.target_dsr - frame.stored_dsr).abs()
    assert error.max() <= 1e-6, "Recomputed DSR must reproduce the source's rounded values"
    return {
        "algebra_and_rank_test_checks": "passed",
        "stored_dsr_max_absolute_error": float(error.max()),
        "stored_dsr_mismatches_above_1e-6": int((error > 1e-6).sum()),
    }


def figures(frame: pd.DataFrame, splits: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"singleton": "#1b9e77", "multiple": "#d95f02"}
    primary = frame[frame.cohort == "primary"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for group, block in primary.groupby("target_group"):
        axes[0].scatter(
            block.target_dsr,
            block.contrast,
            s=15,
            alpha=0.45,
            label=f"{group} target ({len(block)})",
            c=colors[group],
        )
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.7)
    axes[0].set(
        xlabel="Current target-aware DSR",
        ylabel="Label-free top-two contrast",
        title="Target-aware DSR versus observed contrast",
    )
    axes[0].legend(fontsize=8)
    for correct, color, label in (
        (True, "#1b9e77", "Correct leader"),
        (False, "#7570b3", "Incorrect leader"),
    ):
        block = primary[primary.unique & (primary.candidate_correct == int(correct))]
        axes[1].scatter(block.contrast, block.top_mass, s=17, alpha=0.55, c=color, label=label)
    axes[1].set(
        xlabel="Label-free top-two contrast",
        ylabel="Observed leading probability",
        title="Peak separation does not certify correctness",
    )
    axes[1].legend(fontsize=8)
    fig.savefig(OUT / "contrast_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    selected_policies = ("contrast_0.2", "contrast_0.5", "rank_0.05")
    policy_labels = {
        "contrast_0.2": "Contrast ≥ 0.2",
        "contrast_0.5": "Contrast ≥ 0.5",
        "rank_0.05": "Rank test, p ≤ 0.05",
    }
    for ax, group in zip(axes, ("singleton", "multiple")):
        for policy in selected_policies:
            block = splits[(splits.target_group == group) & (splits.policy == policy)]
            ax.plot(block.budget, block.selection_rate, "o-", label=policy_labels[policy])
        ax.set(
            xlabel="Discovery shots (remainder held out)",
            ylabel="Fraction of jobs selected",
            title=f"{group.capitalize()} target tasks",
            ylim=(0, 1),
        )
        ax.legend(fontsize=8)
    fig.savefig(OUT / "shot_budget_sensitivity.png", dpi=180)
    plt.close(fig)


def main() -> None:
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    frame, parsed = load_data()
    validation = checks(frame)
    frame.to_csv(OUT / "job_metrics.csv", index=False)
    summary = summarize(frame, ["cohort", "target_group"])
    summary.to_csv(OUT / "screening_summary.csv", index=False)
    summarize(frame, ["cohort", "algorithm", "target_group"]).to_csv(
        OUT / "screening_by_algorithm.csv", index=False
    )
    summarize(frame, ["cohort", "backend_name", "target_group"]).to_csv(
        OUT / "screening_by_backend.csv", index=False
    )
    summarize(frame, ["cohort", "shots", "target_group"]).to_csv(
        OUT / "screening_by_shots.csv", index=False
    )
    by_config = summarize(frame, ["cohort", "algorithm", "config_id", "target_group"])
    by_config.to_csv(OUT / "screening_by_config.csv", index=False)
    # A sensitivity summary giving each circuit configuration equal weight.
    by_config.groupby(["cohort", "target_group", "policy"]).agg(
        configurations=("config_id", "size"),
        mean_config_selection_rate=("selection_rate", "mean"),
        mean_config_correct_selected_fraction=("correct_fraction_selected", "mean"),
        configs_with_selected_jobs=("correct_fraction_selected", "count"),
    ).reset_index().to_csv(OUT / "screening_config_macro.csv", index=False)
    example_ids = (4, 486, 655)
    frame[frame.row_id.isin(example_ids)].to_csv(OUT / "hardware_examples.csv", index=False)
    multi = frame[(frame.cohort == "primary") & (frame.k > 1)]
    diagnostics = {
        "multi_target_jobs": len(multi),
        "mean_zero_sum_positive": int(
            ((multi.target_dsr == 0) & (multi.target_sum_signed > 0)).sum()
        ),
        "mean_positive_min_nonpositive": int(
            ((multi.target_dsr > 0) & (multi.target_min_signed <= 0)).sum()
        ),
        "mean_positive_missing_target": int(
            ((multi.target_dsr > 0) & (multi.target_coverage < 1)).sum()
        ),
        "dsr_at_least_0.8_top2_below_0.2": int(
            ((multi.target_dsr >= 0.8) & (multi.contrast < 0.2)).sum()
        ),
    }
    (OUT / "target_aggregation_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    print("Data audit and full-shot screening complete", validation, flush=True)
    split_records = split_study(frame, parsed)
    split_summary = summarize_splits(split_records)
    split_summary.to_csv(OUT / "split_summary.csv", index=False)
    # Average repetitions within each real job; do not present them as independent QPU jobs.
    numeric = [
        "candidate_correct",
        "holdout_unique_leader",
        "holdout_candidate_mass",
        "contrast",
        "gap",
        "rank_p",
    ] + [f"pass_{p}" for p in POLICIES]
    split_records.groupby(
        ["row_id", "algorithm", "config_id", "backend_name", "target_group", "budget"]
    )[numeric].mean().reset_index().to_csv(OUT / "split_per_job.csv", index=False)
    # Ratios require joint selection/correctness, not products of their separate means.
    for policy in POLICIES:
        split_records[f"correct_selected_{policy}"] = (
            split_records[f"pass_{policy}"] & split_records.candidate_correct
        )
        split_records[f"stable_selected_{policy}"] = (
            split_records[f"pass_{policy}"] & split_records.holdout_unique_leader
        )
    joint = [c for c in split_records if c.startswith(("correct_selected_", "stable_selected_"))]
    split_records.groupby(["row_id", "budget"])[joint].mean().reset_index().to_csv(
        OUT / "split_per_job_joint.csv", index=False
    )
    controls()
    figures(frame, split_summary)
    manifest = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": source_hash,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": SEED,
        "split_repeats": REPEATS,
        "split_budgets": BUDGETS,
        "split_eligible_jobs": int(((frame.cohort == "primary") & (frame.shots == 1024)).sum()),
        "uniform_repeats_per_cell": 1000,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "rows": len(frame),
        "cohorts": frame.cohort.value_counts().to_dict(),
        "register_alignment_rows": int(frame.register_aligned.sum()),
        "policies": POLICIES,
        "validation": validation,
        "source_unchanged": hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash,
    }
    assert manifest["source_unchanged"]
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary[summary.cohort == "primary"].to_string(index=False), flush=True)
    print("Study complete. Outputs:", OUT, flush=True)


if __name__ == "__main__":
    main()
