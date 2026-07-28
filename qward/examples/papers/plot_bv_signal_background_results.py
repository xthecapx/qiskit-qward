"""Generate paper figures for the BV signal-plus-background IBM campaign."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from qward.examples.papers.bv.bv_signal_background import (
    build_dynamic_signal_background_circuit,
    make_spec,
)
from qward.utils.styles import (
    COLORBREWER_PALETTE,
    FIG_SIZE,
    LABEL_SIZE,
    TICK_SIZE,
    apply_axes_defaults,
)

PAPERS_DIR = Path(__file__).resolve().parent
DATA_DIR = PAPERS_DIR / "bv" / "data" / "qpu" / "signal_background" / "raw"
PLOTS_DIR = PAPERS_DIR / "plots"
TOTAL_QUBITS = (27, 28, 29)

CIRCUIT_FIGURE = PLOTS_DIR / "1_bv_signal_background_circuit.png"
DSR_HISTOGRAM_FIGURE = PLOTS_DIR / "1_bv_signal_background_dsr_histogram_ibm.png"

TARGET_COLOR = COLORBREWER_PALETTE[1]
COMPETITOR_COLOR = COLORBREWER_PALETTE[4]
BACKGROUND_COLOR = "#bdbdbd"
TOP_OUTCOMES = 40


def _load_latest_batch(total_qubits: int) -> dict[str, Any]:
    """Load the latest completed batch for one campaign configuration."""
    paths = sorted(DATA_DIR.glob(f"BVSB{total_qubits}_IBM_*.json"))
    if not paths:
        raise FileNotFoundError(f"No BVSB{total_qubits} result found in {DATA_DIR}")

    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise ValueError(f"Batch {paths[-1].name} is not completed")
    if len(payload.get("individual_results", [])) != 10:
        raise ValueError(f"Batch {paths[-1].name} does not contain 10 runs")
    return payload


def load_campaign_batches() -> list[dict[str, Any]]:
    """Load and validate the three campaign batches."""
    batches = [_load_latest_batch(total_qubits) for total_qubits in TOTAL_QUBITS]
    backends = {batch.get("backend_name") for batch in batches}
    if backends != {"ibm_marrakesh"}:
        raise ValueError(f"Expected only ibm_marrakesh results, found {sorted(backends)}")
    return batches


def generate_circuit_figure() -> Path:
    """Render a readable representative instance of the dynamic circuit."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    circuit = build_dynamic_signal_background_circuit(make_spec(6))
    figure = circuit.draw(
        output="mpl",
        fold=30,
        idle_wires=False,
        cregbundle=True,
        style="iqp",
    )
    figure.savefig(CIRCUIT_FIGURE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return CIRCUIT_FIGURE


def _apply_plot_style() -> None:
    """Apply the typography and line styles used by the paper figures."""
    plt.rcParams.update(
        {
            "font.size": TICK_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "axes.linewidth": 1.5,
            "axes.grid": True,
            "grid.alpha": 0.7,
            "grid.linestyle": "--",
        }
    )


def _select_positive_dsr_run(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the single run with the highest (positive) DSR across all batches."""
    best: dict[str, Any] | None = None
    for batch in batches:
        for result in batch["individual_results"]:
            dsr = float(result.get("dsr_michelson") or 0.0)
            if best is None or dsr > float(best.get("dsr_michelson") or 0.0):
                best = result
    if best is None or float(best.get("dsr_michelson") or 0.0) <= 0.0:
        raise ValueError("No run with positive DSR was found in the campaign")
    return best


def generate_dsr_histogram_figure(batches: list[dict[str, Any]]) -> Path:
    """Plot the measured outcomes of the only job that scored DSR > 0."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    _apply_plot_style()

    run = _select_positive_dsr_run(batches)
    counts: dict[str, int] = {str(k): int(v) for k, v in run["counts"].items()}
    expected = {str(state) for state in run["expected_outcomes"]}

    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    top = ranked[:TOP_OUTCOMES]
    positions = np.arange(1, len(top) + 1)
    heights = [count for _, count in top]
    strongest_competitor = max(
        ((state, count) for state, count in counts.items() if state not in expected),
        key=lambda item: item[1],
    )[0]
    colors = [
        (
            TARGET_COLOR
            if state in expected
            else COMPETITOR_COLOR if state == strongest_competitor else BACKGROUND_COLOR
        )
        for state, _ in top
    ]

    figure, axis = plt.subplots(figsize=FIG_SIZE)
    axis.bar(positions, heights, color=colors, edgecolor="black", linewidth=0.5, zorder=3)

    axis.set_xticks([])
    axis.text(
        len(top) + 1.2,
        0.45,
        r"$\cdots$",
        ha="center",
        va="center",
        fontsize=LABEL_SIZE,
        color="#666666",
    )
    axis.set_xlabel(
        "Top-40 outcome frequency",
        fontsize=LABEL_SIZE,
        fontweight="bold",
    )
    axis.set_ylabel("Measured shots", fontsize=LABEL_SIZE, fontweight="bold")
    axis.set_xlim(0.3, len(top) + 2.2)
    axis.set_ylim(0, max(heights) + 2)
    apply_axes_defaults(axis)
    figure.tight_layout()
    figure.savefig(DSR_HISTOGRAM_FIGURE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return DSR_HISTOGRAM_FIGURE


def main() -> None:
    """Generate the circuit and DSR histogram figures."""
    batches = load_campaign_batches()
    circuit_path = generate_circuit_figure()
    histogram_path = generate_dsr_histogram_figure(batches)
    print(f"Saved circuit figure to {circuit_path}")
    print(f"Saved DSR histogram figure to {histogram_path}")


if __name__ == "__main__":
    main()
