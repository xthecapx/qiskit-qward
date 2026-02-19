#!/usr/bin/env python3
"""
QFT QPU Metrics Visualization

Loads QPU result JSONs from provider folders under data/ (primary layout:
qpu/raw for IBM and qpu/aws for AWS), aggregates by config_id (latest run),
and produces plots to understand when results stop being useful.

Output: visuals/ folder with PNGs and README.md.

Usage:
    python qft_qpu_visuals.py
    python qft_qpu_visuals.py --provider aws
    python qft_qpu_visuals.py --data-dir data/qpu/raw --out-dir visuals
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR / "data"
DEFAULT_OUT_DIR = SCRIPT_DIR / "visuals"


def collect_raw_dirs(data_root: Path, provider: Optional[str] = None) -> List[Path]:
    """Collect result directories from data root.

    Args:
        data_root: Root directory containing provider subdirectories.
        provider: Optional provider filter (e.g. ``qpu`` or ``aws``).

    Returns:
        List of existing result directories. Supports:
        - Preferred layout: ``data/qpu/raw`` (IBM), ``data/qpu/aws`` (AWS)
        - Legacy layout: ``data/aws/raw``
    """
    if not data_root.exists():
        return []

    candidates: List[Path] = []
    if provider == "qpu":
        candidates.append(data_root / "qpu" / "raw")
    elif provider == "aws":
        candidates.extend([data_root / "qpu" / "aws", data_root / "aws" / "raw"])
    elif provider:
        candidates.extend([data_root / "qpu" / provider, data_root / provider / "raw"])
    else:
        candidates.extend([data_root / "qpu" / "raw", data_root / "qpu" / "aws"])
        for provider_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
            raw_dir = provider_dir / "raw"
            if raw_dir.exists():
                candidates.append(raw_dir)

    unique_existing: List[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path.exists() and path not in seen:
            seen.add(path)
            unique_existing.append(path)
    return unique_existing


def load_qpu_results(data_dirs: List[Path]) -> List[Dict[str, Any]]:
    """Load all JSON files from one or more raw data directories."""
    results = []
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        for path in sorted(data_dir.glob("*.json")):
            try:
                with open(path, "r") as f:
                    results.append(json.load(f))
            except (json.JSONDecodeError, IOError):
                continue
    return results


def latest_per_config(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """One result per config_id: keep latest saved_at."""
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
    """Extract plottable metrics from one QFT result."""
    config = result.get("config", {})
    summary = result.get("batch_summary", {})
    ind = result.get("individual_results", [])
    transpiled_depths = [
        r.get("transpiled_depth") for r in ind if r.get("transpiled_depth") is not None
    ]
    success_rates = [r.get("success_rate") for r in ind if r.get("success_rate") is not None]
    advantage_ratios = [
        r.get("advantage_ratio") for r in ind if r.get("advantage_ratio") is not None
    ]
    # QFT random chance: 1/search_space for roundtrip, or num_peaks/search_space for period_detection
    search_space = config.get("search_space") or (2 ** (config.get("num_qubits") or 0))
    period = config.get("period")
    if config.get("test_mode") == "period_detection" and period:
        num_peaks = search_space // period
        random_chance_val = num_peaks / search_space if search_space else None
    else:
        random_chance_val = 1.0 / search_space if search_space else None
    return {
        "config_id": config.get("config_id"),
        "num_qubits": config.get("num_qubits"),
        "test_mode": config.get("test_mode"),
        "mean_success_rate": summary.get("mean_success_rate"),
        "std_success_rate": summary.get("std_success_rate"),
        "min_success_rate": summary.get("min_success_rate"),
        "max_success_rate": summary.get("max_success_rate"),
        "mean_advantage_ratio": summary.get("mean_quantum_advantage_ratio"),
        "quantum_advantage_demonstrated": summary.get("quantum_advantage_demonstrated"),
        "search_space": search_space,
        "random_chance": random_chance_val,
        "transpiled_depth_max": max(transpiled_depths) if transpiled_depths else None,
        "transpiled_depth_mean": (
            (sum(transpiled_depths) / len(transpiled_depths)) if transpiled_depths else None
        ),
        "circuit_depth": ind[0].get("circuit_depth") if ind else None,
        "success_rates_per_run": success_rates,
        "advantage_ratios_per_run": advantage_ratios,
        "backend_name": result.get("backend_name"),
    }


def plot_success_vs_qubits(rows: List[Dict], out_dir: Path) -> None:
    """Success rate vs num_qubits with random chance reference."""
    fig, ax = plt.subplots(figsize=(10, 5))
    qubits = [r["num_qubits"] for r in rows if r["num_qubits"] is not None]
    mean_sr = [r["mean_success_rate"] for r in rows if r["num_qubits"] is not None]
    std_sr = [r["std_success_rate"] or 0 for r in rows if r["num_qubits"] is not None]
    random_chance = [r["random_chance"] for r in rows if r["num_qubits"] is not None]
    config_ids = [r["config_id"] for r in rows if r["num_qubits"] is not None]
    order = sorted(range(len(qubits)), key=lambda i: (qubits[i], config_ids[i]))
    qubits = [qubits[i] for i in order]
    mean_sr = [mean_sr[i] for i in order]
    std_sr = [std_sr[i] for i in order]
    random_chance = [random_chance[i] for i in order]
    config_ids = [config_ids[i] for i in order]
    x = range(len(qubits))
    ax.errorbar(
        x, mean_sr, yerr=std_sr, fmt="o-", capsize=4, label="QPU mean success rate", color="C0"
    )
    ax.plot(x, random_chance, "^--", alpha=0.8, label="Random guess (1/2^n)", color="C2")
    ax.axhline(0.30, color="gray", linestyle=":", alpha=0.7, label="30% threshold")
    ax.axhline(0.50, color="gray", linestyle="-.", alpha=0.7, label="50% threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{q}\n{c}" for q, c in zip(qubits, config_ids)], fontsize=8, rotation=45, ha="right"
    )
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Config (num_qubits)")
    ax.set_title("QFT QPU: Success rate vs circuit size")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "qft_success_rate_vs_qubits.png", dpi=150)
    plt.close()


def plot_success_vs_transpiled_depth(rows: List[Dict], out_dir: Path) -> None:
    """Success rate vs transpiled depth (noise impact)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    depth = [r["transpiled_depth_max"] or r["transpiled_depth_mean"] for r in rows]
    mean_sr = [r["mean_success_rate"] for r in rows]
    config_ids = [r["config_id"] for r in rows]
    valid = [
        (d, s, c) for d, s, c in zip(depth, mean_sr, config_ids) if d is not None and s is not None
    ]
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
    ax.set_title("QFT QPU: Success rate vs transpiled depth (noise impact)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "qft_success_rate_vs_transpiled_depth.png", dpi=150)
    plt.close()


def plot_advantage_ratio_vs_qubits(rows: List[Dict], out_dir: Path) -> None:
    """Quantum advantage ratio vs qubits."""
    fig, ax = plt.subplots(figsize=(10, 5))
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
    ax.set_xticklabels(
        [f"{q}\n{c}" for q, c in zip(qubits, config_ids)], fontsize=8, rotation=45, ha="right"
    )
    ax.set_ylabel("Advantage ratio (success / random chance)")
    ax.set_xlabel("Config (num_qubits)")
    ax.set_title("QFT QPU: Quantum advantage ratio by config")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "qft_advantage_ratio_vs_qubits.png", dpi=150)
    plt.close()


def plot_threshold_pass(rows: List[Dict], out_dir: Path) -> None:
    """Which configs pass 30%, 50%, 70%, 90% (best run)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    config_ids = [r["config_id"] for r in rows]
    qubits = [r["num_qubits"] or 0 for r in rows]
    order = sorted(range(len(config_ids)), key=lambda i: (qubits[i], config_ids[i]))
    config_ids = [config_ids[i] for i in order]
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
    ax.set_title("QFT QPU: Threshold pass by config (best run)")
    ax.legend(loc="upper right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Fail", "Pass"])
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "qft_threshold_pass_by_config.png", dpi=150)
    plt.close()


def plot_success_by_opt_level(rows: List[Dict], out_dir: Path) -> None:
    """Success rate by optimization level per config."""
    from collections import defaultdict

    data: Dict[str, Dict[int, float]] = defaultdict(dict)
    for r in rows:
        cid = r.get("config_id")
        if not cid:
            continue
        for ind in r.get("individual_results") or []:
            opt = ind.get("optimization_level")
            sr = ind.get("success_rate")
            if opt is not None and sr is not None:
                data[cid][opt] = sr
    if not data:
        return
    id_to_qubits = {
        r.get("config_id"): (r.get("num_qubits") or 0, r.get("config_id") or "") for r in rows
    }
    config_ids = sorted(data.keys(), key=lambda c: id_to_qubits.get(c, (0, c)))
    opts = sorted(set(opt for per in data.values() for opt in per))
    x = range(len(config_ids))
    width = 0.8 / max(len(opts), 1)
    fig, ax = plt.subplots(figsize=(max(12, len(config_ids) * 0.5), 5))
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
    ax.set_title("QFT QPU: Success rate by optimization level")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "qft_success_by_optimization_level.png", dpi=150)
    plt.close()


def run_visualizations(data_dirs: List[Path], out_dir: Path) -> None:
    """Load data from provider raw dirs, build metrics, generate plots."""
    results = load_qpu_results(data_dirs)
    if not results:
        joined = ", ".join(str(d) for d in data_dirs) if data_dirs else "<none>"
        print(f"No JSON files found in: {joined}")
        return
    by_config = latest_per_config(results)
    rows = []
    for cid, result in sorted(by_config.items()):
        m = extract_metrics(result)
        m["individual_results"] = result.get("individual_results", [])
        rows.append(m)
    rows.sort(key=lambda r: (r["num_qubits"] or 0, r["config_id"] or ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_success_vs_qubits(rows, out_dir)
    plot_success_vs_transpiled_depth(rows, out_dir)
    plot_advantage_ratio_vs_qubits(rows, out_dir)
    plot_threshold_pass(rows, out_dir)
    plot_success_by_opt_level(rows, out_dir)
    print(f"QFT: {len(rows)} configs, plots saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="QFT QPU metrics visualization")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Raw JSON directory (legacy override; takes precedence)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Data root containing provider folders (default: data/)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Provider filter under data root (e.g. qpu, aws)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for plots"
    )
    args = parser.parse_args()
    if args.data_dir is not None:
        data_dirs = [args.data_dir]
    else:
        data_dirs = collect_raw_dirs(args.data_root, args.provider)
    run_visualizations(data_dirs, args.out_dir)


if __name__ == "__main__":
    main()
