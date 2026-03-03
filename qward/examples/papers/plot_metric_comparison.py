"""
Compare DSR (Michelson), Hellinger Fidelity, and TVD Fidelity across qubit counts.

Generates two plots:
  - grover_metric_comparison.png  (Grover algorithm)
  - qft_metric_comparison.png     (QFT algorithm)

Each plot shows boxplots for the three metrics grouped by number of qubits,
revealing how each metric tracks degradation as problem size grows.

Usage:
  PYTHONPATH=. uv run python qward/examples/papers/plot_metric_comparison.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from qward.utils.styles import (
    COLORBREWER_PALETTE,
    TITLE_SIZE,
    LABEL_SIZE,
    TICK_SIZE,
    LEGEND_SIZE,
    FIG_SIZE,
    apply_axes_defaults,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PAPERS_DIR = Path(__file__).resolve().parent
PLOTS_DIR = PAPERS_DIR / "plots"

DATASETS = {
    "GROVER": [
        PAPERS_DIR / "grover" / "data" / "qpu" / "raw",
        PAPERS_DIR / "grover" / "data" / "qpu" / "aws",
    ],
    "QFT": [
        PAPERS_DIR / "qft" / "data" / "qpu" / "raw",
        PAPERS_DIR / "qft" / "data" / "qpu" / "aws",
    ],
}

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

METRICS = [
    ("dsr_michelson", "DSR (Michelson)"),
    ("hellinger_fidelity", "Hellinger Fidelity"),
    ("tvd_fidelity", "TVD Fidelity"),
]

METRIC_COLORS = {
    "dsr_michelson": COLORBREWER_PALETTE[1],       # Teal
    "hellinger_fidelity": COLORBREWER_PALETTE[2],   # Orange
    "tvd_fidelity": COLORBREWER_PALETTE[3],         # Purple
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_results(directories: List[Path]) -> List[Dict]:
    """Load individual results from all JSON files in the given directories."""
    results = []
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            config = payload.get("config", {})
            for result in payload.get("individual_results", []):
                counts = result.get("counts")
                if not counts:
                    continue
                # Only include results that have all three metrics
                if (
                    "dsr_michelson" not in result
                    or "hellinger_fidelity" not in result
                    or "tvd_fidelity" not in result
                ):
                    continue

                results.append({
                    "num_qubits": result.get("num_qubits", config.get("num_qubits")),
                    "optimization_level": result.get("optimization_level"),
                    "dsr_michelson": result["dsr_michelson"],
                    "hellinger_fidelity": result["hellinger_fidelity"],
                    "tvd_fidelity": result["tvd_fidelity"],
                })
    return results


def _group_by_qubits(
    results: List[Dict],
    optimization_levels: Tuple[int, ...] = (2, 3),
) -> Tuple[List[int], Dict[str, Dict[int, List[float]]]]:
    """Group metric values by qubit count.

    For IBM results (which have optimization_level), keeps only results
    whose level is in *optimization_levels*.  AWS results (no
    optimization_level) are always included.

    Returns:
        (sorted_qubits, metric_data) where metric_data maps
        metric_key -> {num_qubits: [values]}
    """
    metric_data: Dict[str, Dict[int, List[float]]] = {
        key: defaultdict(list) for key, _ in METRICS
    }
    all_qubits: set = set()

    for r in results:
        nq = r.get("num_qubits")
        if nq is None:
            continue
        nq = int(nq)

        # Filter by optimization level for IBM data
        opt = r.get("optimization_level")
        if opt is not None and int(opt) not in optimization_levels:
            continue

        all_qubits.add(nq)
        for key, _ in METRICS:
            val = r.get(key)
            if val is not None:
                metric_data[key][nq].append(float(val))

    return sorted(all_qubits), metric_data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _apply_plot_style():
    plt.rcParams.update({
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
    })


def _render_metric_comparison(
    ax,
    qubits: List[int],
    metric_data: Dict[str, Dict[int, List[float]]],
    algorithm: str,
) -> None:
    """Render the three-metric comparison boxplot onto *ax*."""
    if not qubits:
        return

    n_metrics = len(METRICS)
    width = 0.25

    q_min = min(qubits)
    q_max = max(qubits)
    full_range = list(range(q_min, q_max + 1))

    # Alternating background shading
    for idx, q in enumerate(full_range):
        if idx % 2 == 0:
            ax.axvspan(q - 0.5, q + 0.5, color="#f0f0f0", zorder=0)

    # Dashed separators
    for idx in range(len(full_range) - 1):
        mid = (full_range[idx] + full_range[idx + 1]) / 2
        ax.axvline(mid, color="#cccccc", linestyle="--", linewidth=1, zorder=1)

    # Draw boxplots per metric
    for i, (key, label) in enumerate(METRICS):
        box_data = []
        positions = []

        for q in qubits:
            values = metric_data[key].get(q, [])
            if values:
                box_data.append(values)
                positions.append(q + (i - n_metrics / 2 + 0.5) * width)

        if box_data:
            color = METRIC_COLORS[key]
            ax.boxplot(
                box_data,
                positions=positions,
                widths=width * 0.9,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=color, alpha=0.7),
                medianprops=dict(color="black", linewidth=3),
                whiskerprops=dict(linewidth=2),
                capprops=dict(linewidth=2),
            )

    # Styling
    ax.set_xlabel("Number of Qubits", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Score", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_title(algorithm, fontsize=TITLE_SIZE, fontweight="bold")
    apply_axes_defaults(ax)
    ax.set_ylim(-0.05, 1.05)

    ax.set_xticks(full_range)
    ax.set_xticklabels([str(q) for q in full_range])
    ax.set_xlim(q_min - 0.5, q_max + 0.5)

    # Legend
    legend_elements = [
        Patch(facecolor=METRIC_COLORS[key], alpha=0.7, label=label)
        for key, label in METRICS
    ]
    ax.legend(handles=legend_elements, fontsize=LEGEND_SIZE, loc="upper right")


def _plot_algorithm(algorithm: str, directories: List[Path]) -> None:
    """Generate the metric comparison plot for a single algorithm."""
    _apply_plot_style()

    results = _load_results(directories)
    if not results:
        print(f"  {algorithm}: no results found, skipping")
        return

    qubits, metric_data = _group_by_qubits(results)
    if not qubits:
        print(f"  {algorithm}: no qubit data, skipping")
        return

    total = sum(len(v) for d in metric_data.values() for v in d.values())
    print(f"  {algorithm}: {len(results)} results, qubits {min(qubits)}-{max(qubits)}, "
          f"{total} data points")

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    _render_metric_comparison(ax, qubits, metric_data, algorithm)

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{algorithm.lower()}_metric_comparison.png"
    fig.savefig(
        PLOTS_DIR / filename, dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)
    print(f"  Saved: {PLOTS_DIR / filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Generating metric comparison plots...")
    for algorithm, directories in DATASETS.items():
        _plot_algorithm(algorithm, directories)
    print("Done.")


if __name__ == "__main__":
    main()
