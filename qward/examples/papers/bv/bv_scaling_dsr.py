#!/usr/bin/env python3
"""
Bernstein-Vazirani DSR scaling study (paper figures).

Demonstrates the scalability advantage of DSR over full-distribution fidelity
metrics (Hellinger fidelity / TVD fidelity): past the classical statevector
"wall" (~30 qubits on a laptop) the ideal reference distribution needed by
HF/TVDF cannot be built, so those metrics become *infeasible* -- yet DSR is
still computed directly from the measured counts of a real 30-qubit IBM job.

Success semantics for the multi-PUB IBM job (ONES / ALT / SINGLE):
  * **Per-state**: each pattern is enriched independently via QWARD
    ``FidelityMetrics`` (K=1 expected outcome). The paper reports these
    separately because the three targets discriminate very differently
    (including the all-zero competing peak rate).
  * **Job-level OR**: the algorithm succeeds if *any* pattern recovers its
    secret (``job_success_or``); the probabilistic OR rate is
    ``1 - Π(1 - p_i)`` across per-state success rates.

Enrich the raw JSONs first (optional but recommended)::

    PYTHONPATH=. uv run python qward/examples/papers/enrich_dsr_profile.py \\
        --dataset bv-ibm --force

Then run this script::

    uv run python qward/examples/papers/bv/bv_scaling_dsr.py --phase all
    uv run python qward/examples/papers/bv/bv_scaling_dsr.py --phase analyze
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qward.metrics import FidelityMetrics
from qward.utils.styles import (
    COLORBREWER_PALETTE,
    TITLE_SIZE,
    LABEL_SIZE,
    TICK_SIZE,
    LEGEND_SIZE,
    apply_axes_defaults,
)
from qward.examples.papers.bv.bv_scaling_utils import (
    PATTERN_NAMES,
    secrets_for,
    expected_outcome_from_secret,
    build_bv_circuit,
    circuit_metrics,
    predicted_statevector_bytes,
    available_memory_bytes,
    full_distribution_fidelities,
    synthetic_noisy_counts,
    run_wall_sweep,
)

BV_DIR = Path(__file__).resolve().parent
DATA_DIR = BV_DIR / "data"
RAW_DIR = DATA_DIR / "qpu" / "raw"
PAPERS_DIR = BV_DIR.parent
PLOTS_DIR = PAPERS_DIR / "plots"

WALL_CSV = DATA_DIR / "bv_scaling_wall_sweep.csv"
DSR_CSV = DATA_DIR / "bv_scaling_dsr_ibm.csv"  # per expected-outcome / pattern
JOB_OR_CSV = DATA_DIR / "bv_scaling_dsr_job_or.csv"  # job-level OR across patterns
CROSS_CSV = DATA_DIR / "bv_scaling_hf_vs_dsr.csv"

TARGET_N = 29  # 30 total qubits, the IBM hardware run
CONTROL_NS = [4, 8, 12, 16, 20]  # below the wall -> HF/TVDF computable
CONTROL_FLIP_P = 0.02
PATTERN_COLORS = {
    "ONES": COLORBREWER_PALETTE[1],
    "ALT": COLORBREWER_PALETTE[2],
    "SINGLE": COLORBREWER_PALETTE[3],
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


# --------------------------------------------------------------------------- #
# Phase: sweep
# --------------------------------------------------------------------------- #
def phase_sweep(args) -> pd.DataFrame:
    print("=" * 70)
    print("Phase SWEEP: statevector-time growth until the simulation wall")
    print("=" * 70)
    rows, wall_n = run_wall_sweep(
        min_n=2,
        max_n=args.max_n,
        attempt_timeout_s=args.timeout,
        pattern="ALT",
        verbose=True,
    )

    # Circuit metrics beyond the wall (no simulation needed) for the growth panel.
    metrics_only = []
    for n in range(wall_n + 1, args.metrics_max_n + 1):
        secret = secrets_for(n)["ALT"]
        qc = build_bv_circuit(secret, use_barriers=False)
        m = circuit_metrics(qc)
        metrics_only.append(
            {
                "n_secret": n,
                "secret_pattern": "ALT",
                **m,
                "predicted_bytes": predicted_statevector_bytes(m["num_qubits_total"]),
                "predicted_gib": predicted_statevector_bytes(m["num_qubits_total"]) / 1024**3,
                "simulated": False,
                "elapsed_s": None,
                "status": "not_attempted (beyond wall)",
                "top_bitstring": None,
                "top_probability": None,
                "matches_expected": None,
            }
        )

    df = pd.concat([pd.DataFrame(rows), pd.DataFrame(metrics_only)], ignore_index=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(WALL_CSV, index=False)
    print(f"\nSimulation wall at n_secret={wall_n} ({wall_n + 1} total qubits).")
    print(f"Saved {WALL_CSV}")
    return df


def _wall_n_from_sweep() -> int:
    if not WALL_CSV.exists():
        return 25
    df = pd.read_csv(WALL_CSV)
    ok = df[df["simulated"] == True]  # noqa: E712
    return int(ok["n_secret"].max()) if not ok.empty else 25


# --------------------------------------------------------------------------- #
# Phase: analyze
# --------------------------------------------------------------------------- #
def _load_ibm_runs(min_n: int = 2) -> List[Dict]:
    """Load per-pattern IBM runs from the raw JSON store."""
    runs = []
    if not RAW_DIR.exists():
        return runs
    for path in sorted(RAW_DIR.glob("BV*_IBM_*.json")):
        try:
            payload = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        cfg = payload.get("config", {})
        for ir in payload.get("individual_results", []):
            counts = ir.get("counts")
            if not counts:
                continue
            secret = ir.get("secret_string") or cfg.get("secret_string")
            expected = ir.get("expected_outcome") or (
                expected_outcome_from_secret(secret) if secret else None
            )
            if not expected:
                continue
            n = int(ir.get("num_qubits") or cfg.get("num_qubits") or len(expected))
            if n < min_n:
                continue
            pat = payload.get("config_id", cfg.get("config_id", "")).split("-")[-1]
            runs.append(
                {
                    "source_path": str(path),
                    "job_id": ir.get("job_id"),
                    "backend_name": ir.get("backend_name") or payload.get("backend_name"),
                    "optimization_level": ir.get("optimization_level"),
                    "n_secret": n,
                    "num_qubits_total": n + 1,
                    "pattern_name": pat,
                    "secret": secret,
                    "expected_output": expected,
                    "counts": {str(k).replace(" ", ""): int(v) for k, v in counts.items()},
                }
            )
    return runs


def _qward_dsr_profile(counts: Dict[str, int], expected: str, secret: str) -> Dict:
    """Compute the full DSR profile via QWARD FidelityMetrics (no ideal histogram)."""
    circuit = build_bv_circuit(secret, use_barriers=False)
    fm = FidelityMetrics(
        circuit,
        counts=counts,
        expected_outcomes=[expected],
        include_dsr_profile=True,
        include_michelson=True,
    )
    schema = fm.get_metrics()
    if hasattr(schema, "model_dump"):
        data = schema.model_dump()
    elif hasattr(schema, "dict"):
        data = schema.dict()
    else:
        data = dict(schema)
    # FidelitySchema names Michelson DSR as ``dsr``; keep both keys for callers.
    if data.get("dsr_michelson") is None and data.get("dsr") is not None:
        data["dsr_michelson"] = data["dsr"]
    return data


def _counts_of_all_zeros(counts: Dict[str, int], n: int) -> int:
    """Shots that landed on the all-zero bitstring (important competing peak)."""
    return int(counts.get("0" * n, 0))


def phase_analyze(args) -> Dict[str, pd.DataFrame]:
    print("=" * 70)
    print("Phase ANALYZE: QWARD DSR profile (per-state + job-level OR)")
    print("=" * 70)
    wall_n = _wall_n_from_sweep()
    print(f"Using simulation wall n_secret={wall_n} (from sweep).")

    # --- Per expected-outcome DSR via QWARD FidelityMetrics ---
    ibm_runs = _load_ibm_runs()
    dsr_rows = []
    for r in ibm_runs:
        expected = r["expected_output"]
        counts = r["counts"]
        n = r["n_secret"]
        prof = _qward_dsr_profile(counts, expected, r["secret"])
        top = max(counts, key=counts.get) if counts else None
        zeros = _counts_of_all_zeros(counts, n)
        shots = sum(counts.values())
        dsr_rows.append(
            {
                "job_id": r["job_id"],
                "backend_name": r["backend_name"],
                "optimization_level": r["optimization_level"],
                "n_secret": n,
                "num_qubits_total": r["num_qubits_total"],
                "pattern_name": r["pattern_name"],
                "secret": r["secret"],
                "expected_output": expected,
                "top_outcome": top,
                "top_matches_expected": top == expected,
                "shots": shots,
                "all_zero_count": zeros,
                "all_zero_rate": zeros / shots if shots else 0.0,
                "success_rate": prof.get("success_rate"),
                "chance_baseline": prof.get("chance_baseline"),
                "chance_corrected_success": prof.get("chance_corrected_success"),
                "coarse_tvd": prof.get("coarse_tvd"),
                "coarse_tvd_similarity": prof.get("coarse_tvd_similarity"),
                "coarse_hellinger_distance": prof.get("coarse_hellinger_distance"),
                "coarse_hellinger_fidelity": prof.get("coarse_hellinger_fidelity"),
                "dsr_michelson": prof.get("dsr_michelson"),
                "peak_mismatch": prof.get("peak_mismatch"),
                "source_path": r["source_path"],
            }
        )

    dsr_df = pd.DataFrame(dsr_rows).sort_values(["job_id", "n_secret", "pattern_name"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dsr_df.to_csv(DSR_CSV, index=False)
    print(f"\nPer-state (per expected outcome) DSR via QWARD -> {DSR_CSV}")
    if not dsr_df.empty:
        print(
            dsr_df[
                [
                    "job_id",
                    "pattern_name",
                    "success_rate",
                    "all_zero_rate",
                    "dsr_michelson",
                    "top_matches_expected",
                ]
            ].to_string(index=False)
        )

    # --- Job-level OR: algorithm succeeds if ANY pattern recovers its secret ---
    job_rows = []
    if not dsr_df.empty:
        for job_id, grp in dsr_df.groupby("job_id", dropna=False):
            rates = grp["success_rate"].astype(float).tolist()
            # Probabilistic OR of independent pattern successes: 1 - Π(1 - p_i)
            fail_prod = 1.0
            for p in rates:
                fail_prod *= 1.0 - float(p)
            job_or_rate = 1.0 - fail_prod
            any_top_match = bool(grp["top_matches_expected"].any())
            job_rows.append(
                {
                    "job_id": job_id,
                    "backend_name": grp["backend_name"].iloc[0],
                    "n_secret": int(grp["n_secret"].iloc[0]),
                    "num_qubits_total": int(grp["num_qubits_total"].iloc[0]),
                    "num_patterns": len(grp),
                    "patterns": ",".join(grp["pattern_name"].tolist()),
                    # Per-state rates kept for the paper (discrimination by target).
                    "success_rate_ONES": _pattern_value(grp, "ONES", "success_rate"),
                    "success_rate_ALT": _pattern_value(grp, "ALT", "success_rate"),
                    "success_rate_SINGLE": _pattern_value(grp, "SINGLE", "success_rate"),
                    "all_zero_rate_ONES": _pattern_value(grp, "ONES", "all_zero_rate"),
                    "all_zero_rate_ALT": _pattern_value(grp, "ALT", "all_zero_rate"),
                    "all_zero_rate_SINGLE": _pattern_value(grp, "SINGLE", "all_zero_rate"),
                    "dsr_michelson_ONES": _pattern_value(grp, "ONES", "dsr_michelson"),
                    "dsr_michelson_ALT": _pattern_value(grp, "ALT", "dsr_michelson"),
                    "dsr_michelson_SINGLE": _pattern_value(grp, "SINGLE", "dsr_michelson"),
                    # Job-level OR aggregates.
                    "job_success_or": any_top_match,  # True if any pattern top-matches
                    "job_success_rate_or": job_or_rate,  # 1 - Π(1 - p_i)
                    "mean_per_state_success": float(grp["success_rate"].mean()),
                }
            )

    job_df = pd.DataFrame(job_rows)
    job_df.to_csv(JOB_OR_CSV, index=False)
    print(f"\nJob-level OR success -> {JOB_OR_CSV}")
    if not job_df.empty:
        print(
            job_df[
                [
                    "job_id",
                    "job_success_or",
                    "job_success_rate_or",
                    "success_rate_ONES",
                    "success_rate_ALT",
                    "success_rate_SINGLE",
                ]
            ].to_string(index=False)
        )

    # --- HF/TVDF feasibility crossover (still uses QWARD DSR for the DSR columns) ---
    cross_rows = []
    for n in CONTROL_NS:
        secret = secrets_for(n)["ALT"]
        expected = expected_outcome_from_secret(secret)
        counts = synthetic_noisy_counts(expected, 4096, CONTROL_FLIP_P, seed=100 + n)
        fd = full_distribution_fidelities(counts, secret, timeout_s=args.timeout)
        prof = _qward_dsr_profile(counts, expected, secret)
        cross_rows.append(
            _cross_row("below wall (control)", n, "ALT", fd, prof, None, "ideal+synthetic")
        )

    for r in ibm_runs:
        if r["n_secret"] <= wall_n:
            continue
        fd = full_distribution_fidelities(
            r["counts"], r["secret"], timeout_s=args.timeout, allow_attempt=args.force_attempt
        )
        prof = _qward_dsr_profile(r["counts"], r["expected_output"], r["secret"])
        cross_rows.append(
            _cross_row(
                "beyond wall (IBM hardware)",
                r["n_secret"],
                r["pattern_name"],
                fd,
                prof,
                r["job_id"],
                r["backend_name"],
            )
        )

    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_csv(CROSS_CSV, index=False)
    n_infeasible = int((cross_df["hf_status"] != "computed").sum()) if not cross_df.empty else 0
    print(
        f"\nSaved {CROSS_CSV}. HF/TVDF infeasible for {n_infeasible}/{len(cross_df)} rows; "
        f"DSR (QWARD) computed for all {len(cross_df)}."
    )
    return {"dsr": dsr_df, "job_or": job_df, "cross": cross_df}


def _pattern_value(grp: pd.DataFrame, pattern: str, column: str):
    sub = grp[grp["pattern_name"] == pattern]
    if sub.empty:
        return None
    return sub.iloc[0][column]


def _cross_row(regime, n, pattern, fd, prof, job_id, backend) -> Dict:
    return {
        "regime": regime,
        "n_secret": n,
        "num_qubits_total": n + 1,
        "pattern_name": pattern,
        "predicted_gib": round(fd["predicted_statevector_gib"], 4),
        "num_ideal_amplitudes": fd["num_ideal_amplitudes"],
        "hf_status": fd["status"],
        "hellinger_fidelity": fd["hellinger_fidelity"],
        "tvd_fidelity": fd["tvd_fidelity"],
        "dsr_michelson": prof.get("dsr_michelson"),
        "coarse_hellinger_fidelity": prof.get("coarse_hellinger_fidelity"),
        "success_rate": prof.get("success_rate"),
        "job_id": job_id,
        "backend_name": backend,
    }


# --------------------------------------------------------------------------- #
# Phase: plots
# --------------------------------------------------------------------------- #
def _plot_simulation_wall(wall_df: pd.DataFrame, wall_n: int) -> None:
    _apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    sim = wall_df[wall_df["simulated"] == True]  # noqa: E712
    timeouts = wall_df[wall_df["status"] == "timeout"]

    ax = axes[0]
    if not sim.empty:
        ax.plot(
            sim["num_qubits_total"], sim["elapsed_s"], "o-",
            color=COLORBREWER_PALETTE["IBM"], label="Statevector time",
        )
    if not timeouts.empty:
        y = float(sim["elapsed_s"].max()) * 1.5 if not sim.empty else 200.0
        ax.scatter(
            timeouts["num_qubits_total"], [y] * len(timeouts), marker="X", s=400,
            color=COLORBREWER_PALETTE[4], linewidths=3, zorder=5, label="Timeout (wall)",
        )
    ax.axvline(wall_n + 1, color="gray", linestyle="--", label=f"Wall = {wall_n + 1} qubits")
    ax.axvline(TARGET_N + 1, color=COLORBREWER_PALETTE[3], linestyle=":", label=f"IBM run = {TARGET_N + 1} qubits")
    ax.set_yscale("log")
    ax.set_xlabel("Total qubits")
    ax.set_ylabel("Ideal statevector time (s, log)")
    ax.set_title("HF/TVDF need this simulation; DSR does not")
    apply_axes_defaults(ax)
    ax.legend(fontsize=LEGEND_SIZE)

    ax = axes[1]
    order = wall_df.sort_values("num_qubits_total")
    ax.plot(
        order["num_qubits_total"], order["predicted_gib"], "s-",
        color=COLORBREWER_PALETTE[5], label="Ideal statevector memory",
    )
    budget_gib = available_memory_bytes() * 0.6 / 1024**3
    ax.axhline(budget_gib, color=COLORBREWER_PALETTE[4], linestyle="-.",
               label=f"Laptop budget ~{budget_gib:.0f} GiB")
    ax.axvline(wall_n + 1, color="gray", linestyle="--")
    ax.axvline(TARGET_N + 1, color=COLORBREWER_PALETTE[3], linestyle=":")
    ax.set_yscale("log")
    ax.set_xlabel("Total qubits")
    ax.set_ylabel("Predicted memory (GiB, log)")
    ax.set_title("Ideal-distribution memory explodes exponentially")
    apply_axes_defaults(ax)
    ax.legend(fontsize=LEGEND_SIZE)

    fig.tight_layout()
    out = PLOTS_DIR / "3_bv_simulation_wall.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_hf_vs_dsr(cross_df: pd.DataFrame, wall_n: int) -> None:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(16, 10))

    computed = cross_df[cross_df["hf_status"] == "computed"].sort_values("num_qubits_total")
    infeasible = cross_df[cross_df["hf_status"] != "computed"]

    # Shade the region where HF/TVDF cannot be computed.
    x_max = float(cross_df["num_qubits_total"].max()) + 1
    ax.axvspan(wall_n + 1.5, x_max, color=COLORBREWER_PALETTE[4], alpha=0.10,
               label="HF/TVDF infeasible")

    if not computed.empty:
        ax.plot(
            computed["num_qubits_total"], computed["hellinger_fidelity"], "s-",
            color=COLORBREWER_PALETTE[5], label="Hellinger fidelity (computed)",
        )
        ax.plot(
            computed["num_qubits_total"], computed["tvd_fidelity"], "^-",
            color=COLORBREWER_PALETTE[6], label="TVD fidelity (computed)",
        )
    if not infeasible.empty:
        ax.scatter(
            infeasible["num_qubits_total"], [0.5] * len(infeasible), marker="X", s=400,
            color=COLORBREWER_PALETTE[4], linewidths=3, zorder=5,
            label="HF/TVDF unobtainable",
        )

    # DSR from the SAME counts, computable at every size. Line follows the
    # below-wall controls; the real IBM patterns are shown as stars at 30 qubits.
    controls = cross_df[cross_df["regime"].str.startswith("below")].sort_values("num_qubits_total")
    if not controls.empty:
        ax.plot(
            controls["num_qubits_total"], controls["dsr_michelson"], "D-",
            color=COLORBREWER_PALETTE[3], label="DSR Michelson (counts only)",
        )
    beyond = cross_df[cross_df["regime"].str.startswith("beyond")]
    if not beyond.empty:
        ax.scatter(
            beyond["num_qubits_total"], beyond["dsr_michelson"], marker="*", s=700,
            color=COLORBREWER_PALETTE["IBM"], edgecolors="black", linewidths=1.5, zorder=6,
            label="DSR on real IBM 30-qubit job",
        )
        for _, row in beyond.iterrows():
            ax.annotate(
                row["pattern_name"], (row["num_qubits_total"], row["dsr_michelson"]),
                textcoords="offset points", xytext=(12, 0), fontsize=LEGEND_SIZE - 4,
            )

    ax.axvline(wall_n + 1, color="gray", linestyle="--", label=f"Simulation wall")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Total qubits")
    ax.set_ylabel("Metric value")
    ax.set_title("DSR scales past the classical simulation wall; HF/TVDF do not")
    apply_axes_defaults(ax)
    ax.legend(fontsize=LEGEND_SIZE, loc="center left")

    fig.tight_layout()
    out = PLOTS_DIR / "3_bv_hf_vs_dsr_scalability.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_ibm_dsr_profile(dsr_df: pd.DataFrame, job_df: Optional[pd.DataFrame] = None) -> None:
    target = dsr_df[dsr_df["n_secret"] == TARGET_N]
    if target.empty:
        print(f"No IBM data at n_secret={TARGET_N}; skipping DSR-profile plot.")
        return
    _apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    # Left: per-state DSR profile components (discrimination by expected outcome).
    ax = axes[0]
    components = [
        ("success_rate", "Success rate"),
        ("chance_corrected_success", "Chance-corrected"),
        ("coarse_tvd_similarity", "Coarse TVD sim."),
        ("coarse_hellinger_fidelity", "Coarse HF"),
        ("dsr_michelson", "DSR Michelson"),
    ]
    patterns = [p for p in PATTERN_NAMES if p in set(target["pattern_name"])]
    x = np.arange(len(components))
    width = 0.8 / max(len(patterns), 1)

    for i, pat in enumerate(patterns):
        row = target[target["pattern_name"] == pat].iloc[0]
        vals = [row[c] for c, _ in components]
        ax.bar(
            x + i * width, vals, width, label=pat,
            color=PATTERN_COLORS.get(pat, COLORBREWER_PALETTE[i + 1]), edgecolor="black",
        )

    ax.set_xticks(x + width * (len(patterns) - 1) / 2)
    ax.set_xticklabels([lbl for _, lbl in components], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("DSR profile value")
    backend = target["backend_name"].iloc[0]
    ax.set_title(f"Per-state DSR profile ({TARGET_N + 1} qubits, {backend})")
    apply_axes_defaults(ax)
    ax.legend(fontsize=LEGEND_SIZE, title="Expected outcome")

    # Right: per-state success vs all-zero rate + job-level OR.
    ax = axes[1]
    x2 = np.arange(len(patterns))
    success_vals = [
        float(target[target["pattern_name"] == p]["success_rate"].iloc[0]) for p in patterns
    ]
    zero_vals = [
        float(target[target["pattern_name"] == p]["all_zero_rate"].iloc[0])
        if "all_zero_rate" in target.columns
        else 0.0
        for p in patterns
    ]
    ax.bar(
        x2 - 0.2, success_vals, 0.4, label="Per-state success rate",
        color=COLORBREWER_PALETTE["IBM"], edgecolor="black",
    )
    ax.bar(
        x2 + 0.2, zero_vals, 0.4, label="All-zero rate (competing)",
        color=COLORBREWER_PALETTE[4], edgecolor="black",
    )
    if job_df is not None and not job_df.empty:
        job_or = float(job_df["job_success_rate_or"].iloc[0])
        ax.axhline(
            job_or, color=COLORBREWER_PALETTE[3], linestyle="--", linewidth=3,
            label=f"Job-level OR success = {job_or:.3f}",
        )
        if bool(job_df["job_success_or"].iloc[0]):
            ax.text(
                0.02, 0.95, "Job OR: recovered ≥1 pattern",
                transform=ax.transAxes, fontsize=LEGEND_SIZE, va="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )
    ax.set_xticks(x2)
    ax.set_xticklabels(patterns)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Per-state success vs all-zero + job OR")
    apply_axes_defaults(ax)
    ax.legend(fontsize=LEGEND_SIZE)

    fig.tight_layout()
    out = PLOTS_DIR / "3_bv_ibm_dsr_profile_n29.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def phase_plots(args) -> None:
    print("=" * 70)
    print("Phase PLOTS: paper-styled figures")
    print("=" * 70)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    wall_n = _wall_n_from_sweep()

    if WALL_CSV.exists():
        _plot_simulation_wall(pd.read_csv(WALL_CSV), wall_n)
    else:
        print(f"Missing {WALL_CSV}; run --phase sweep first (skipping wall plot).")

    if CROSS_CSV.exists():
        _plot_hf_vs_dsr(pd.read_csv(CROSS_CSV), wall_n)
    else:
        print(f"Missing {CROSS_CSV}; run --phase analyze first (skipping crossover plot).")

    job_df = pd.read_csv(JOB_OR_CSV) if JOB_OR_CSV.exists() else None
    if DSR_CSV.exists():
        _plot_ibm_dsr_profile(pd.read_csv(DSR_CSV), job_df)
    else:
        print(f"Missing {DSR_CSV}; run --phase analyze first (skipping profile plot).")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Bernstein-Vazirani DSR scaling study.")
    parser.add_argument(
        "--phase", choices=["sweep", "analyze", "plots", "all"], default="all",
        help="Which phase to run (default: all).",
    )
    parser.add_argument("--max-n", type=int, default=34, help="Max n_secret for the wall sweep.")
    parser.add_argument(
        "--metrics-max-n", type=int, default=40,
        help="Max n_secret for metrics-only (beyond-wall) enumeration.",
    )
    parser.add_argument(
        "--timeout", type=float, default=180.0,
        help="Hard per-statevector timeout in seconds (Docker-safe).",
    )
    parser.add_argument(
        "--force-attempt", action="store_true",
        help="Actually attempt the 30-qubit statevector (watch it die) instead of the memory pre-check.",
    )
    args = parser.parse_args(argv)

    if args.phase in ("sweep", "all"):
        phase_sweep(args)
    if args.phase in ("analyze", "all"):
        phase_analyze(args)
    if args.phase in ("plots", "all"):
        phase_plots(args)


if __name__ == "__main__":
    main()
