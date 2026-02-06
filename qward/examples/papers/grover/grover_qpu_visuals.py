#!/usr/bin/env python3
"""
Grover QPU Metrics Visualization

Loads QPU result JSONs from data/qpu/raw/, aggregates by config_id (latest run),
and produces plots to understand when results stop being useful.

Output: visuals/ folder with PNGs and README.md.

Usage:
    python grover_qpu_visuals.py
    python grover_qpu_visuals.py --data-dir data/qpu/raw --out-dir visuals
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


# Default paths (relative to script location)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "qpu" / "raw"
DEFAULT_OUT_DIR = SCRIPT_DIR / "visuals"


def load_qpu_results(data_dir: Path) -> List[Dict[str, Any]]:
    """Load all JSON files from data_dir. Returns list of full result dicts."""
    if not data_dir.exists():
        return []
    results = []
    for path in sorted(data_dir.glob("*.json")):
        try:
            with open(path, "r") as f:
                results.append(json.load(f))
        except (json.JSONDecodeError, IOError):
            continue
    return results


def latest_per_config(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """One result per config_id: keep the one with latest saved_at."""
    by_config: Dict[str, Dict[str, Any]] = {}
    for r in results:
        cid = r.get("config_id") or r.get("config", {}).get("config_id")
        if not cid:
            continue
        saved = r.get("saved_at", "")
        if cid not in by_config or saved > by_config[cid].get("saved_at", ""):
            by_config[cid] = r
    return by_config


def extract_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract plottable metrics from one result."""
    config = result.get("config", {})
    summary = result.get("batch_summary", {})
    ind = result.get("individual_results", [])
    # Transpiled depth: use max across optimization levels (worst case on hardware)
    transpiled_depths = [r.get("transpiled_depth") for r in ind if r.get("transpiled_depth") is not None]
    circ_depths = [r.get("circuit_depth") for r in ind if r.get("circuit_depth") is not None]
    success_rates = [r.get("success_rate") for r in ind if r.get("success_rate") is not None]
    advantage_ratios = [r.get("advantage_ratio") for r in ind if r.get("advantage_ratio") is not None]
    return {
        "config_id": config.get("config_id"),
        "num_qubits": config.get("num_qubits"),
        "mean_success_rate": summary.get("mean_success_rate"),
        "std_success_rate": summary.get("std_success_rate"),
        "min_success_rate": summary.get("min_success_rate"),
        "max_success_rate": summary.get("max_success_rate"),
        "mean_advantage_ratio": summary.get("mean_quantum_advantage_ratio"),
        "quantum_advantage_demonstrated": summary.get("quantum_advantage_demonstrated"),
        "theoretical_success": config.get("theoretical_success"),
        "random_chance": config.get("classical_random_prob"),
        "transpiled_depth_max": max(transpiled_depths) if transpiled_depths else None,
        "transpiled_depth_mean": (sum(transpiled_depths) / len(transpiled_depths)) if transpiled_depths else None,
        "circuit_depth": circ_depths[0] if circ_depths else None,
        "success_rates_per_run": success_rates,
        "advantage_ratios_per_run": advantage_ratios,
        "backend_name": result.get("backend_name"),
    }


def plot_success_vs_qubits(rows: List[Dict], out_dir: Path) -> None:
    """Success rate vs num_qubits with theoretical and random reference lines."""
    fig, ax = plt.subplots(figsize=(8, 5))
    # Filter rows with valid num_qubits and mean_success_rate
    valid = [r for r in rows if r.get("num_qubits") is not None and r.get("mean_success_rate") is not None]
    if not valid:
        return
    qubits = [r["num_qubits"] for r in valid]
    mean_sr = [r["mean_success_rate"] for r in valid]
    std_sr = [r["std_success_rate"] if r.get("std_success_rate") is not None else 0 for r in valid]
    theoretical = [r.get("theoretical_success") for r in valid]
    random_chance = [r.get("random_chance") for r in valid]
    config_ids = [r.get("config_id") for r in valid]

    # Sort by num_qubits for clean x-axis
    order = sorted(range(len(qubits)), key=lambda i: (qubits[i], config_ids[i]))
    qubits = [qubits[i] for i in order]
    mean_sr = [mean_sr[i] for i in order]
    std_sr = [std_sr[i] for i in order]
    theoretical = [theoretical[i] for i in order]
    random_chance = [random_chance[i] for i in order]
    config_ids = [config_ids[i] for i in order]

    x = range(len(qubits))
    std_sr = [float(s) if s is not None else 0.0 for s in std_sr]
    ax.errorbar(x, mean_sr, yerr=std_sr, fmt="o-", capsize=4, label="QPU mean success rate", color="C0")
    if any(t is not None for t in theoretical):
        ax.plot(x, [t if t is not None else 0 for t in theoretical], "s--", alpha=0.8, label="Theoretical (no noise)", color="C1")
    if any(rc is not None for rc in random_chance):
        ax.plot(x, [rc if rc is not None else 0 for rc in random_chance], "^--", alpha=0.8, label="Random guess", color="C2")
    ax.axhline(0.30, color="gray", linestyle=":", alpha=0.7, label="30% threshold")
    ax.axhline(0.50, color="gray", linestyle="-.", alpha=0.7, label="50% threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{q}\n{c}" for q, c in zip(qubits, config_ids)], fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Config (num_qubits)")
    ax.set_title("Grover QPU: Success rate vs circuit size")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "grover_success_rate_vs_qubits.png", dpi=150)
    plt.close()


def plot_success_vs_transpiled_depth(rows: List[Dict], out_dir: Path) -> None:
    """When transpiled depth grows, success typically drops."""
    fig, ax = plt.subplots(figsize=(8, 5))
    depth = [r["transpiled_depth_max"] or r["transpiled_depth_mean"] for r in rows]
    mean_sr = [r["mean_success_rate"] for r in rows]
    config_ids = [r["config_id"] for r in rows]
    # Filter valid
    valid = [(d, s, c) for d, s, c in zip(depth, mean_sr, config_ids) if d is not None and s is not None]
    if not valid:
        return
    depth, mean_sr, config_ids = zip(*sorted(valid, key=lambda x: x[0]))
    ax.plot(depth, mean_sr, "o-", color="C0")
    for i, cid in enumerate(config_ids):
        ax.annotate(cid, (depth[i], mean_sr[i]), fontsize=7, alpha=0.8)
    ax.axhline(0.30, color="gray", linestyle=":", alpha=0.7)
    ax.axhline(0.50, color="gray", linestyle="-.", alpha=0.7)
    ax.set_xlabel("Transpiled circuit depth (max over opt levels)")
    ax.set_ylabel("Mean success rate")
    ax.set_title("Grover QPU: Success rate vs transpiled depth (noise impact)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "grover_success_rate_vs_transpiled_depth.png", dpi=150)
    plt.close()


def plot_advantage_ratio_vs_qubits(rows: List[Dict], out_dir: Path) -> None:
    """Quantum advantage ratio (success_rate / random_chance) vs qubits."""
    fig, ax = plt.subplots(figsize=(8, 5))
    qubits = [r["num_qubits"] for r in rows if r["num_qubits"] is not None]
    adv = [r["mean_advantage_ratio"] for r in rows if r["num_qubits"] is not None]
    config_ids = [r["config_id"] for r in rows if r["num_qubits"] is not None]
    order = sorted(range(len(qubits)), key=lambda i: (qubits[i], config_ids[i]))
    qubits = [qubits[i] for i in order]
    adv = [adv[i] if adv[i] is not None else 0 for i in order]
    config_ids = [config_ids[i] for i in order]
    x = range(len(qubits))
    colors = ["C2" if a and a > 2 else "C3" for a in adv]
    ax.bar(x, adv, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(2.0, color="black", linestyle="--", label="Quantum advantage (2× random)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{q}\n{c}" for q, c in zip(qubits, config_ids)], fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Advantage ratio (success / random chance)")
    ax.set_xlabel("Config (num_qubits)")
    ax.set_title("Grover QPU: Quantum advantage ratio by config")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "grover_advantage_ratio_vs_qubits.png", dpi=150)
    plt.close()


def plot_threshold_pass(rows: List[Dict], out_dir: Path) -> None:
    """Which configs pass 30%, 50%, 70%, 90% success rate (from individual runs)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    config_ids = [r["config_id"] for r in rows]
    qubits = [r["num_qubits"] or 0 for r in rows]
    order = sorted(range(len(config_ids)), key=lambda i: (qubits[i], config_ids[i]))
    config_ids = [config_ids[i] for i in order]
    # For each config, get max success rate across runs (best opt level)
    max_sr = []
    for i in order:
        r = rows[i]
        rates = r.get("success_rates_per_run") or [r.get("mean_success_rate")]
        max_sr.append(max([x for x in rates if x is not None], default=0))
    x = range(len(config_ids))
    width = 0.2
    t30 = [1 if s >= 0.30 else 0 for s in max_sr]
    t50 = [1 if s >= 0.50 else 0 for s in max_sr]
    t70 = [1 if s >= 0.70 else 0 for s in max_sr]
    t90 = [1 if s >= 0.90 else 0 for s in max_sr]
    ax.bar([i - 1.5 * width for i in x], t30, width, label="≥30%", color="C0", alpha=0.8)
    ax.bar([i - 0.5 * width for i in x], t50, width, label="≥50%", color="C1", alpha=0.8)
    ax.bar([i + 0.5 * width for i in x], t70, width, label="≥70%", color="C2", alpha=0.8)
    ax.bar([i + 1.5 * width for i in x], t90, width, label="≥90%", color="C3", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(config_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Pass (1) / Fail (0)")
    ax.set_xlabel("Config")
    ax.set_title("Grover QPU: Threshold pass by config (best run)")
    ax.legend(loc="upper right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Fail", "Pass"])
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "grover_threshold_pass_by_config.png", dpi=150)
    plt.close()


def plot_success_by_opt_level(rows: List[Dict], out_dir: Path) -> None:
    """Per-config: success rate by optimization level (0,1,2,3)."""
    from collections import defaultdict
    data: Dict[str, Dict[int, float]] = defaultdict(dict)
    for r in rows:
        cid = r.get("config_id")
        if not cid:
            continue
        for ind in (r.get("individual_results") or []):
            opt = ind.get("optimization_level")
            sr = ind.get("success_rate")
            if opt is not None and sr is not None:
                data[cid][opt] = sr
    if not data:
        return
    # Sort configs by num_qubits then id (use rows for ordering)
    id_to_qubits = {r.get("config_id"): (r.get("num_qubits") or 0, r.get("config_id") or "") for r in rows}
    config_ids = sorted(data.keys(), key=lambda c: id_to_qubits.get(c, (0, c)))
    opts = sorted(set(opt for per in data.values() for opt in per))
    x = range(len(config_ids))
    width = 0.8 / max(len(opts), 1)
    fig, ax = plt.subplots(figsize=(max(10, len(config_ids) * 0.5), 5))
    for i, opt in enumerate(opts):
        vals = [data[c].get(opt) for c in config_ids]
        vals = [v if v is not None else 0 for v in vals]
        off = (i - len(opts) / 2 + 0.5) * width
        ax.bar([xi + off for xi in x], vals, width=width * 0.9, label=f"Opt {opt}")
    ax.axhline(0.30, color="gray", linestyle=":", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(config_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Config")
    ax.set_title("Grover QPU: Success rate by optimization level")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "grover_success_by_optimization_level.png", dpi=150)
    plt.close()


def run_visualizations(data_dir: Path, out_dir: Path) -> None:
    """Load data, build metrics, generate all plots."""
    results = load_qpu_results(data_dir)
    if not results:
        print(f"No JSON files found in {data_dir}")
        return
    by_config = latest_per_config(results)
    rows = []
    for cid, result in sorted(by_config.items()):
        m = extract_metrics(result)
        # Need to attach individual_results for some plots
        m["individual_results"] = result.get("individual_results", [])
        rows.append(m)
    # Sort by num_qubits then config_id
    rows.sort(key=lambda r: (r["num_qubits"] or 0, r["config_id"] or ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_success_vs_qubits(rows, out_dir)
    plot_success_vs_transpiled_depth(rows, out_dir)
    plot_advantage_ratio_vs_qubits(rows, out_dir)
    plot_threshold_pass(rows, out_dir)
    plot_success_by_opt_level(rows, out_dir)
    print(f"Grover: {len(rows)} configs, plots saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Grover QPU metrics visualization")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="QPU raw JSON directory")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for plots")
    args = parser.parse_args()
    run_visualizations(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
