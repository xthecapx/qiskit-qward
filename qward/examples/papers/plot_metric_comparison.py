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
import numpy as np
from matplotlib.patches import Patch, Rectangle

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
    "BV": [
        PAPERS_DIR / "bv" / "data" / "qpu" / "raw",
    ],
}

# Ladder-only metric comparison (all three metrics required).
BV_METRIC_MAX_QUBITS = 14

# Dense statevector / ideal-histogram wall on the experiment machine
# (Apple M3 Pro, 18 GB): BV uses n_secret + 1 qubits → wall at 26 total qubits
# (n_secret = 25). Beyond-wall IBM runs use n = 29, 30, 31.
BV_STATEVECTOR_WALL_TOTAL_QUBITS = 26
BV_STATEVECTOR_WALL_N_SECRET = BV_STATEVECTOR_WALL_TOTAL_QUBITS - 1  # 25


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

METRICS = [
    ("dsr_michelson", "DSR (Michelson)"),
    ("hellinger_fidelity", "Hellinger Fidelity"),
    ("tvd_fidelity", "TVD Fidelity"),
]

# BV wall figure: include raw success rate alongside DSR / HF / TVDF.
WALL_METRICS = [
    ("dsr_michelson", "DSR (Michelson)"),
    ("success_rate", "Success Rate"),
    ("hellinger_fidelity", "Hellinger Fidelity"),
    ("tvd_fidelity", "TVD Fidelity"),
]

METRIC_COLORS = {
    "dsr_michelson": COLORBREWER_PALETTE[1],  # Teal
    "success_rate": COLORBREWER_PALETTE[4],  # Pink
    "hellinger_fidelity": COLORBREWER_PALETTE[2],  # Orange
    "tvd_fidelity": COLORBREWER_PALETTE[3],  # Purple
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_results(
    directories: List[Path],
    provider: str = "all",
    algorithm: str = "",
) -> List[Dict]:
    """Load individual results from all JSON files in the given directories.

    Args:
        directories: List of directories to scan for JSON files.
        provider: ``"all"`` (default), ``"ibm"``, or ``"aws"``.
        algorithm: Dataset key used for algorithm-specific filters (e.g. BV).
    """
    results = []
    for directory in directories:
        if not directory.exists():
            continue

        # Filter directories by provider
        dir_name = directory.name.lower()
        if provider == "ibm" and dir_name not in ("raw",):
            continue
        if provider == "aws" and dir_name not in ("aws",):
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
                # Only include results that have all three metrics with real values
                # (wall-size BV stores hellinger_fidelity=null / tvd_fidelity=null)
                hf = result.get("hellinger_fidelity")
                tvd_f = result.get("tvd_fidelity")
                dsr = result.get("dsr_michelson")
                if dsr is None or hf is None or tvd_f is None:
                    continue

                nq = result.get("num_qubits", config.get("num_qubits"))
                if algorithm == "BV" and nq is not None and int(nq) > BV_METRIC_MAX_QUBITS:
                    continue

                results.append(
                    {
                        "num_qubits": nq,
                        "optimization_level": result.get("optimization_level"),
                        "dsr_michelson": dsr,
                        "hellinger_fidelity": hf,
                        "tvd_fidelity": tvd_f,
                    }
                )
    return results


def _group_by_qubits(
    results: List[Dict],
    optimization_levels: Tuple[int, ...] = (3,),
) -> Tuple[List[int], Dict[str, Dict[int, List[float]]]]:
    """Group metric values by qubit count.

    For IBM results (which have optimization_level), keeps only results
    whose level is in *optimization_levels*.  AWS results (no
    optimization_level) are always included.

    Returns:
        (sorted_qubits, metric_data) where metric_data maps
        metric_key -> {num_qubits: [values]}
    """
    metric_data: Dict[str, Dict[int, List[float]]] = {key: defaultdict(list) for key, _ in METRICS}
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


def _render_metric_comparison(
    ax,
    qubits: List[int],
    metric_data: Dict[str, Dict[int, List[float]]],
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
    apply_axes_defaults(ax)
    ax.set_ylim(-0.05, 1.05)

    ax.set_xticks(full_range)
    ax.set_xticklabels([str(q) for q in full_range])
    ax.set_xlim(q_min - 0.5, q_max + 0.5)

    # Legend
    legend_elements = [
        Patch(facecolor=METRIC_COLORS[key], alpha=0.7, label=label) for key, label in METRICS
    ]
    ax.legend(handles=legend_elements, fontsize=LEGEND_SIZE, loc="upper right")


def _plot_algorithm(
    algorithm: str,
    directories: List[Path],
    provider: str = "all",
) -> None:
    """Generate the metric comparison plot for a single algorithm."""
    _apply_plot_style()

    results = _load_results(directories, provider=provider, algorithm=algorithm)
    if not results:
        print(f"  {algorithm} ({provider}): no results found, skipping")
        return

    qubits, metric_data = _group_by_qubits(results)
    if not qubits:
        print(f"  {algorithm} ({provider}): no qubit data, skipping")
        return

    total = sum(len(v) for d in metric_data.values() for v in d.values())
    print(
        f"  {algorithm} ({provider}): {len(results)} results, "
        f"qubits {min(qubits)}-{max(qubits)}, {total} data points"
    )

    fig, ax = plt.subplots(figsize=(15, 6))
    _render_metric_comparison(ax, qubits, metric_data)

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if provider == "all" else f"_{provider}"
    filename = f"1_{algorithm.lower()}_metric_comparison{suffix}.png"
    fig.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {PLOTS_DIR / filename}")


def _load_bv_wall_results(directories: List[Path]) -> List[Dict]:
    """Load BV IBM results for the wall figure (DSR required; HF/TVDF optional).

    Prefers the richest file per ``config_id`` (multi-run ALT campaign) and keeps
    only ALT patterns used in the paper figures.
    """
    by_config: Dict[str, tuple] = {}
    for directory in directories:
        if not directory.exists() or directory.name.lower() != "raw":
            continue
        for path in sorted(directory.glob("BV*-ALT*.json")):
            try:
                payload = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            cid = payload.get("config_id") or path.name.split("_IBM_")[0]
            if "-ALT" not in str(cid):
                continue
            irs = [r for r in payload.get("individual_results", []) if r.get("counts")]
            if not irs:
                continue
            prev = by_config.get(cid)
            if prev is None or len(irs) > prev[0]:
                by_config[cid] = (len(irs), path, payload, irs)

    results: List[Dict] = []
    for _n, _path, payload, irs in by_config.values():
        config = payload.get("config", {})
        for result in irs:
            dsr = result.get("dsr_michelson")
            if dsr is None:
                continue
            nq = result.get("num_qubits", config.get("num_qubits"))
            opt = result.get("optimization_level")
            if opt is not None and int(opt) != 3:
                continue
            success = result.get("success_rate")
            if success is None and result.get("expected_outcome") and result.get("counts"):
                shots = sum(result["counts"].values())
                exp = result["expected_outcome"]
                success = (
                    result["counts"].get(exp, 0) / shots if shots else 0.0
                )
            results.append(
                {
                    "num_qubits": int(nq),
                    "dsr_michelson": float(dsr),
                    "success_rate": float(success) if success is not None else None,
                    "hellinger_fidelity": (
                        float(result["hellinger_fidelity"])
                        if result.get("hellinger_fidelity") is not None
                        else None
                    ),
                    "tvd_fidelity": (
                        float(result["tvd_fidelity"])
                        if result.get("tvd_fidelity") is not None
                        else None
                    ),
                }
            )
    return results


def _render_bv_wall_comparison(
    ax,
    qubits: List[int],
    metric_data: Dict[str, Dict[int, List[float]]],
) -> None:
    """BV metric comparison with a statevector wall marker.

    HF/TVDF are drawn only where values exist (pre-wall). DSR and success rate
    continue past the wall. A vertical line marks the dense-simulation limit
    (26 total qubits).
    """
    if not qubits:
        return

    # Compress the gap between the ladder (≤14) and wall sizes (29–31) so the
    # figure stays readable while preserving real qubit labels on the ticks.
    ladder = [q for q in qubits if q <= BV_METRIC_MAX_QUBITS]
    wall = [q for q in qubits if q > BV_STATEVECTOR_WALL_N_SECRET]
    display_qubits = ladder + wall
    # Map real qubit count → display x position (unit spacing, one gap before wall)
    gap = 1.5
    xpos: Dict[int, float] = {}
    x = 0.0
    for i, q in enumerate(display_qubits):
        if i > 0 and display_qubits[i - 1] <= BV_METRIC_MAX_QUBITS < q:
            x += gap
        xpos[q] = x
        x += 1.0

    n_metrics = len(WALL_METRICS)
    width = 0.18
    xs = [xpos[q] for q in display_qubits]
    x_min, x_max = min(xs) - 0.5, max(xs) + 0.5

    for idx, q in enumerate(display_qubits):
        left = xpos[q] - 0.5
        right = xpos[q] + 0.5
        if idx % 2 == 0:
            ax.axvspan(left, right, color="#f0f0f0", zorder=0)

    # Wall marker only (label described in the paper caption/text).
    wall_x = (
        xpos[ladder[-1]] + xpos[wall[0]]
    ) / 2.0 if ladder and wall else float(BV_STATEVECTOR_WALL_N_SECRET)
    ax.axvline(wall_x, color="#444444", linestyle="--", linewidth=2.0, zorder=3)

    for i, (key, _label) in enumerate(WALL_METRICS):
        box_data = []
        positions = []
        for q in display_qubits:
            values = metric_data.get(key, {}).get(q, [])
            if values:
                box_data.append(values)
                positions.append(xpos[q] + (i - n_metrics / 2 + 0.5) * width)
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
            # Collapsed all-zero boxes hide facecolor; draw a short stub so the
            # metric color stays visible (same approach as combined DSR plots).
            half_w = width * 0.45
            stub_h = 0.10
            for vals, pos in zip(box_data, positions):
                arr = np.asarray(vals, dtype=float)
                iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
                if iqr < 0.02:
                    med = float(np.median(arr))
                    y0 = 0.0 if med <= stub_h / 2 else med - stub_h / 2
                    ax.add_patch(
                        Rectangle(
                            (pos - half_w, y0),
                            2 * half_w,
                            stub_h,
                            facecolor=color,
                            edgecolor="black",
                            linewidth=2.0,
                            alpha=0.95,
                            zorder=5,
                            clip_on=False,
                        )
                    )

    ax.set_xlabel("Number of Qubits", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("DSR", fontsize=LABEL_SIZE, fontweight="bold")
    apply_axes_defaults(ax)
    ax.set_ylim(-0.12, 1.22)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(q) for q in display_qubits])
    ax.set_xlim(x_min, x_max)
    # No legend: metric colors are described in the paper caption.


def _plot_bv_wall(directories: List[Path]) -> None:
    """Generate BV IBM wall figure: DSR past the HF/TVDF simulation limit."""
    _apply_plot_style()
    results = _load_bv_wall_results(directories)
    if not results:
        print("  BV wall (ibm): no results found, skipping")
        return

    # Group without dropping DSR/success-only rows past the wall
    metric_data: Dict[str, Dict[int, List[float]]] = {
        key: defaultdict(list) for key, _ in WALL_METRICS
    }
    all_qubits: set = set()
    for r in results:
        nq = int(r["num_qubits"])
        all_qubits.add(nq)
        for key, _ in WALL_METRICS:
            val = r.get(key)
            if val is not None:
                metric_data[key][nq].append(float(val))

    qubits = sorted(all_qubits)
    print(
        f"  BV wall (ibm): {len(results)} results, "
        f"qubits {min(qubits)}-{max(qubits)}; "
        f"HF groups={sorted(metric_data['hellinger_fidelity'])}; "
        f"DSR groups={sorted(metric_data['dsr_michelson'])}; "
        f"SR groups={sorted(metric_data['success_rate'])}"
    )

    fig, ax = plt.subplots(figsize=(16, 8.5))
    _render_bv_wall_comparison(ax, qubits, metric_data)
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "1_bv_metric_comparison_wall_ibm.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Generating metric comparison plots...")
    for algorithm, directories in DATASETS.items():
        _plot_algorithm(algorithm, directories, provider="all")
        _plot_algorithm(algorithm, directories, provider="ibm")
        _plot_algorithm(algorithm, directories, provider="aws")
    _plot_bv_wall(DATASETS["BV"])
    print("Done.")


if __name__ == "__main__":
    main()
