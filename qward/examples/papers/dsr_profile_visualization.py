"""
DSR evaluation profile visualizations.

Reads the unified ``DSR_result.csv`` (built by ``build_csv_from_json.py``
after running ``enrich_dsr_profile.py``) and produces four profile-focused
figures:

  1. ``2_{algorithm}_dsr_profile_comparison[_{provider}].png`` -- the four
     profile components (success rate, chance-corrected success, coarse TVD
     similarity, coarse Hellinger fidelity) grouped by qubit count, per
     algorithm and provider.
  2. ``2_multi_answer_divergence.png`` -- for K > 1 configurations, shows how
     coarse TVD similarity / Hellinger fidelity can diverge from raw success
     rate when success mass is unevenly split across the expected outcomes.
  3. ``2_full_vs_coarse_comparison.png`` -- for K = 1 configurations, verifies
     that the coarse profile components collapse exactly onto the
     full-distribution Hellinger fidelity / TVD fidelity (a sanity check of
     the "K=1 redundancy" claim, not new information).
  4. ``2_success_uncertainty_summary.png`` -- mean success rate and
     chance-corrected success per (algorithm, qubit count), with bootstrap
     95% CIs across repeated runs.

Usage:
  PYTHONPATH=. uv run python qward/examples/papers/dsr_profile_visualization.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from qward.utils.styles import (
    COLORBREWER_PALETTE,
    TITLE_SIZE,
    LABEL_SIZE,
    TICK_SIZE,
    LEGEND_SIZE,
    apply_axes_defaults,
)

PAPERS_DIR = Path(__file__).resolve().parent
PLOTS_DIR = PAPERS_DIR / "plots"
CSV_PATH = PAPERS_DIR / "DSR_result.csv"

PROFILE_METRICS: List[Tuple[str, str]] = [
    ("success_rate", "Success Rate"),
    ("chance_corrected_success", "Chance-Corrected Success"),
    ("coarse_tvd_similarity", "Coarse TVD Similarity"),
    ("coarse_hellinger_fidelity", "Coarse Hellinger Fidelity"),
]

PROFILE_COLORS: Dict[str, str] = {
    "success_rate": COLORBREWER_PALETTE[1],
    "chance_corrected_success": COLORBREWER_PALETTE[2],
    "coarse_tvd_similarity": COLORBREWER_PALETTE[3],
    "coarse_hellinger_fidelity": COLORBREWER_PALETTE[4],
}


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": TICK_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "figure.titlesize": TITLE_SIZE,
            "axes.linewidth": 1.5,
            "axes.grid": True,
            "grid.alpha": 0.7,
            "grid.linestyle": "--",
            "lines.linewidth": 3,
            "lines.markersize": 12,
        }
    )


def _load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["success_rate", "chance_corrected_success"])
    return df


def _backend_group(name: str) -> str:
    n = str(name).lower()
    if "ankaa" in n or "forte" in n or "rigetti" in n:
        return "Rigetti (AWS)"
    if "ibm" in n:
        return "IBM"
    return "other"


# ---------------------------------------------------------------------------
# (1) Four-component profile by workload / backend / size
# ---------------------------------------------------------------------------


def plot_profile_by_workload(df: pd.DataFrame) -> None:
    """Boxplot of the four profile components grouped by qubit count, for
    each algorithm and provider."""
    _apply_plot_style()

    for algorithm in ("GROVER", "QFT"):
        sub_algo = df[df["algorithm"] == algorithm].copy()
        if sub_algo.empty:
            print(f"  {algorithm}: no rows, skipping profile-by-workload plot")
            continue
        sub_algo["backend_group"] = sub_algo["backend_name"].apply(_backend_group)

        for provider, provider_label in (("IBM", "ibm"), ("Rigetti (AWS)", "aws")):
            sub = sub_algo[sub_algo["backend_group"] == provider]
            # IBM data carries multiple optimization levels; keep opt=3 for parity
            # with the rest of the paper's IBM figures.
            if "optimization_level" in sub.columns and provider == "IBM":
                opt = pd.to_numeric(sub["optimization_level"], errors="coerce")
                sub = sub[opt.isna() | (opt == 3)]
            if sub.empty:
                print(f"  {algorithm} ({provider_label}): no rows, skipping")
                continue

            qubits = sorted(sub["num_qubits"].dropna().unique().astype(int))
            if not qubits:
                continue

            fig, ax = plt.subplots(figsize=(15, 6))
            n_metrics = len(PROFILE_METRICS)
            width = 0.2

            q_min, q_max = min(qubits), max(qubits)
            full_range = list(range(q_min, q_max + 1))
            for idx, q in enumerate(full_range):
                if idx % 2 == 0:
                    ax.axvspan(q - 0.5, q + 0.5, color="#f0f0f0", zorder=0)

            for i, (key, label) in enumerate(PROFILE_METRICS):
                box_data, positions = [], []
                for q in qubits:
                    vals = sub.loc[sub["num_qubits"] == q, key].dropna().values
                    if len(vals):
                        box_data.append(vals)
                        positions.append(q + (i - n_metrics / 2 + 0.5) * width)
                if box_data:
                    ax.boxplot(
                        box_data,
                        positions=positions,
                        widths=width * 0.9,
                        patch_artist=True,
                        showfliers=False,
                        boxprops=dict(facecolor=PROFILE_COLORS[key], alpha=0.7),
                        medianprops=dict(color="black", linewidth=3),
                        whiskerprops=dict(linewidth=2),
                        capprops=dict(linewidth=2),
                    )

            ax.set_xlabel("Number of Qubits", fontsize=LABEL_SIZE, fontweight="bold")
            ax.set_ylabel("Score", fontsize=LABEL_SIZE, fontweight="bold")
            apply_axes_defaults(ax)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks(full_range)
            ax.set_xticklabels([str(q) for q in full_range])
            ax.set_xlim(q_min - 0.5, q_max + 0.5)
            ax.set_title(f"{algorithm} DSR Profile ({provider})", fontweight="bold")

            legend_elements = [
                Patch(facecolor=PROFILE_COLORS[key], alpha=0.7, label=label)
                for key, label in PROFILE_METRICS
            ]
            ax.legend(handles=legend_elements, fontsize=LEGEND_SIZE, loc="lower left")

            plt.tight_layout()
            PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            filename = f"2_{algorithm.lower()}_dsr_profile_comparison_{provider_label}.png"
            fig.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"  Saved: {PLOTS_DIR / filename}  ({len(sub)} rows, qubits {q_min}-{q_max})")


# ---------------------------------------------------------------------------
# (2) Multi-answer divergence: success rate vs. coarse similarity for K > 1
# ---------------------------------------------------------------------------


def plot_multi_answer_divergence(df: pd.DataFrame) -> None:
    """For K > 1 configurations, show mean success rate vs. coarse TVD
    similarity / Hellinger fidelity per config, revealing cases where success
    mass is unevenly split across the expected outcomes."""
    _apply_plot_style()

    sub = df.copy()
    sub["num_expected"] = (
        sub["expected_outcomes"]
        .fillna("")
        .apply(lambda s: len([o for o in str(s).split(",") if o]))
    )
    multi = sub[sub["num_expected"] > 1]
    if multi.empty:
        print("  No K>1 configurations found, skipping multi-answer divergence plot")
        return

    grouped = (
        multi.groupby(["algorithm", "config_id"])[
            ["success_rate", "coarse_tvd_similarity", "coarse_hellinger_fidelity", "num_expected"]
        ]
        .mean()
        .reset_index()
        .sort_values(["algorithm", "config_id"])
    )

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(grouped))
    width = 0.25

    ax.bar(
        x - width,
        grouped["success_rate"],
        width,
        label="Success Rate",
        color=COLORBREWER_PALETTE[1],
        alpha=0.85,
    )
    ax.bar(
        x,
        grouped["coarse_tvd_similarity"],
        width,
        label="Coarse TVD Similarity",
        color=COLORBREWER_PALETTE[2],
        alpha=0.85,
    )
    ax.bar(
        x + width,
        grouped["coarse_hellinger_fidelity"],
        width,
        label="Coarse Hellinger Fidelity",
        color=COLORBREWER_PALETTE[3],
        alpha=0.85,
    )

    labels = [f"{row.config_id} (K={int(row.num_expected)})" for row in grouped.itertuples()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=TICK_SIZE - 4)
    ax.set_ylabel("Score", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Multi-Answer Divergence: Coarse Similarity vs. Raw Success Rate (K > 1)",
        fontweight="bold",
    )
    apply_axes_defaults(ax)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper right")

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = "2_multi_answer_divergence.png"
    fig.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {PLOTS_DIR / filename}  ({len(grouped)} K>1 configs)")


# ---------------------------------------------------------------------------
# (3) Full-distribution HF/TVD vs. coarse profile equivalents (K = 1 only)
# ---------------------------------------------------------------------------


def plot_full_vs_coarse_comparison(df: pd.DataFrame) -> None:
    """For K=1 configurations, scatter full-distribution Hellinger fidelity /
    TVD fidelity against their coarse-profile equivalents. Points should lie
    exactly on the y=x line -- this is a sanity check of the documented K=1
    redundancy, not new evidence."""
    _apply_plot_style()

    sub = df.dropna(
        subset=[
            "hellinger_fidelity",
            "tvd_fidelity",
            "coarse_hellinger_fidelity",
            "coarse_tvd_similarity",
        ]
    ).copy()
    sub["num_expected"] = (
        sub["expected_outcomes"]
        .fillna("")
        .apply(lambda s: len([o for o in str(s).split(",") if o]))
    )
    sub = sub[sub["num_expected"] == 1]
    if sub.empty:
        print("  No K=1 rows with full HF/TVD available, skipping full-vs-coarse plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, (full_col, coarse_col, label) in zip(
        axes,
        [
            ("hellinger_fidelity", "coarse_hellinger_fidelity", "Hellinger Fidelity"),
            ("tvd_fidelity", "coarse_tvd_similarity", "TVD Fidelity / Similarity"),
        ],
    ):
        colors = [
            COLORBREWER_PALETTE[1] if a == "GROVER" else COLORBREWER_PALETTE[2]
            for a in sub["algorithm"]
        ]
        ax.scatter(sub[full_col], sub[coarse_col], c=colors, alpha=0.6, s=40, edgecolors="none")
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.5, label="y = x")
        ax.set_xlabel(f"Full-distribution {label}", fontsize=LABEL_SIZE, fontweight="bold")
        ax.set_ylabel(f"Coarse {label}", fontsize=LABEL_SIZE, fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        apply_axes_defaults(ax)
        max_abs_diff = float(np.max(np.abs(sub[full_col] - sub[coarse_col])))
        ax.set_title(f"max |Δ| = {max_abs_diff:.2e}", fontsize=LABEL_SIZE)

    legend_elements = [
        Patch(facecolor=COLORBREWER_PALETTE[1], alpha=0.7, label="GROVER"),
        Patch(facecolor=COLORBREWER_PALETTE[2], alpha=0.7, label="QFT"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=2,
        fontsize=LEGEND_SIZE,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle(
        "Full-Distribution vs. Coarse-Profile Metrics: Exact Only When the Ideal Is a\n"
        "Delta on E (QFT round-trip); Grover's Finite-Iteration Ideal Deviates from a Delta",
        fontweight="bold",
        fontsize=LABEL_SIZE - 4,
        y=1.12,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.82))
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = "2_full_vs_coarse_comparison.png"
    fig.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {PLOTS_DIR / filename}  ({len(sub)} K=1 rows)")


# ---------------------------------------------------------------------------
# (4) Uncertainty-aware summary of success / chance-corrected success
# ---------------------------------------------------------------------------


def _bootstrap_mean_ci(
    values: np.ndarray, n_boot: int = 5000, seed: int = 42
) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 3:
        return mean, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.array(
        [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, float(lo), float(hi)


def plot_success_uncertainty_summary(df: pd.DataFrame) -> None:
    """Mean success rate and chance-corrected success per (algorithm, qubit
    count), with bootstrap 95% CIs computed across repeated runs (captures
    run-to-run variability; not a per-shot binomial CI)."""
    _apply_plot_style()

    for algorithm in ("GROVER", "QFT"):
        sub = df[df["algorithm"] == algorithm].copy()
        if sub.empty:
            continue
        qubits = sorted(sub["num_qubits"].dropna().unique().astype(int))
        if not qubits:
            continue

        rows = []
        for q in qubits:
            grp = sub[sub["num_qubits"] == q]
            s_mean, s_lo, s_hi = _bootstrap_mean_ci(grp["success_rate"].values)
            c_mean, c_lo, c_hi = _bootstrap_mean_ci(grp["chance_corrected_success"].values)
            rows.append((q, s_mean, s_lo, s_hi, c_mean, c_lo, c_hi, len(grp)))

        fig, ax = plt.subplots(figsize=(13, 6))
        qs = [r[0] for r in rows]
        s_means = [r[1] for r in rows]
        s_err = [
            [m - lo if not np.isnan(lo) else 0 for m, lo in zip(s_means, [r[2] for r in rows])],
            [hi - m if not np.isnan(hi) else 0 for m, hi in zip(s_means, [r[3] for r in rows])],
        ]
        c_means = [r[4] for r in rows]
        c_err = [
            [m - lo if not np.isnan(lo) else 0 for m, lo in zip(c_means, [r[5] for r in rows])],
            [hi - m if not np.isnan(hi) else 0 for m, hi in zip(c_means, [r[6] for r in rows])],
        ]

        offset = 0.15
        ax.errorbar(
            [q - offset for q in qs],
            s_means,
            yerr=s_err,
            fmt="o",
            color=COLORBREWER_PALETTE[1],
            label="Success Rate",
            capsize=4,
            markersize=9,
        )
        ax.errorbar(
            [q + offset for q in qs],
            c_means,
            yerr=c_err,
            fmt="s",
            color=COLORBREWER_PALETTE[2],
            label="Chance-Corrected Success",
            capsize=4,
            markersize=9,
        )
        ax.axhline(0.0, color="#999999", linestyle=":", linewidth=1)
        ax.set_xlabel("Number of Qubits", fontsize=LABEL_SIZE, fontweight="bold")
        ax.set_ylabel("Score (mean, 95% bootstrap CI)", fontsize=LABEL_SIZE, fontweight="bold")
        ax.set_xticks(qs)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{algorithm}: Success vs. Chance-Corrected Success", fontweight="bold")
        apply_axes_defaults(ax)
        ax.legend(fontsize=LEGEND_SIZE, loc="upper right")

        plt.tight_layout()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"2_{algorithm.lower()}_success_uncertainty_summary.png"
        fig.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {PLOTS_DIR / filename}  ({sum(r[7] for r in rows)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    df = _load_data()
    print(f"Loaded {len(df)} rows from {CSV_PATH}")

    print("\n(1) Profile by workload/backend/size:")
    plot_profile_by_workload(df)

    print("\n(2) Multi-answer divergence:")
    plot_multi_answer_divergence(df)

    print("\n(3) Full vs. coarse comparison (K=1):")
    plot_full_vs_coarse_comparison(df)

    print("\n(4) Success / chance-corrected success uncertainty summary:")
    plot_success_uncertainty_summary(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
