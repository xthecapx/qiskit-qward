"""Render proposed dashboard figures from the validated experiment export.

Run from the repository root with ``uv run python scripts/preview_ibm_dashboard.py``.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
from matplotlib.lines import Line2D  # pylint: disable=wrong-import-position
from matplotlib.patches import Patch  # pylint: disable=wrong-import-position
from matplotlib.ticker import PercentFormatter  # pylint: disable=wrong-import-position
import pandas as pd  # pylint: disable=wrong-import-position

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "qward/examples/papers"
OUTPUT = ROOT / "docs/figures/ibm_quantum_dashboard"
COLORS = ["#2563eb", "#0d9488", "#d97706", "#9333ea", "#e11d48", "#475569", "#0891b2", "#a16207"]
METRICS = [
    ("f.fidelity.dsr", "DSR", "Expected-peak contrast"),
    ("f.fidelity.success_rate", "Success rate", "Shots satisfying the task criteria"),
    (
        "f.fidelity.hellinger_fidelity",
        "Hellinger fidelity",
        "Agreement with the ideal distribution",
    ),
]


def decorate(figure, axes, title, subtitle):
    """Apply consistent presentation styling to the three preview figures."""
    figure.set_facecolor("#f8fafc")
    figure.suptitle(title, x=0.055, y=0.985, ha="left", fontsize=22, weight="bold", color="#0f172a")
    figure.text(0.055, 0.89, subtitle, fontsize=11, color="#475569")
    for axis in axes:
        axis.set_facecolor("white")
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["bottom", "left"]].set_color("#cbd5e1")
        axis.tick_params(colors="#475569")
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)


def save(figure, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / name, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)


def boundary_preview(window):
    """Render the proposed landing view for the measured QFT SR width sweep."""
    selected = window[
        (window["meta.algorithm"] == "QFT") & window["meta.config_id"].str.fullmatch(r"SR\d+")
    ]
    success = "f.fidelity.success_rate"
    dsr = "f.fidelity.dsr"
    target = 0.50
    widths = selected.groupby("a.num_qubits")
    meets = widths[success].min() >= target
    below = widths[success].max() < target
    # The headline describes a single crossing in this complete example sweep.
    last, first = meets[meets].index.max(), below[below].index.min()
    if (
        first != last + 1
        or not meets.loc[:last].all()
        or not below.loc[first:].all()
        or not widths.size().eq(2).all()
    ):
        raise ValueError("QFT SR preview requires one crossing supported by both repetitions")
    figure, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    decorate(
        figure,
        axes,
        f"QFT met the target through {last} qubits. Both jobs fell below at {first}.",
        "IBM Fez   /   Window 1   /   QFT · SR family   /   Raw counts   /   2 jobs per width · 1,024 shots each",
    )
    for axis, metric, title, color in zip(
        axes,
        (success, dsr),
        ("Did enough shots succeed?", "Was the expected peak still visible?"),
        ("#146755", "#3759a3"),
    ):
        means = widths[metric].mean()
        axis.axvspan(last, first, color="#dce5ed", alpha=0.55)
        axis.plot(means.index, means, color=color, lw=1.5)
        axis.scatter(selected["a.num_qubits"], selected[metric], color=color, s=30, zorder=3)
        axis.set(xlabel="Logical qubits", xticks=means.index, ylim=(0, 1.06))
        axis.set_title(title, loc="left", fontsize=14, weight="bold", pad=11)
    axes[0].axhline(target, color="#9c5100", linestyle="--", lw=1.5)
    axes[0].text(2.05, target + 0.025, "Example target · 50%", color="#874609", fontsize=10)
    axes[0].set_ylabel("Shots satisfying the task")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_ylabel("DSR · expected-peak contrast")
    figure.text(
        0.065,
        0.14,
        f"At {first} qubits: {widths[success].mean().loc[first]:.1%} mean success"
        f"   ·   {widths[dsr].mean().loc[first]:.2f} mean DSR"
        f"   ·   executed depth {int(widths['b.depth'].mean().loc[first])}",
        fontsize=12,
        weight="bold",
        color="#152940",
    )
    figure.text(
        0.065,
        0.085,
        "Each dot is a job. Shading marks the two tested widths on either side of the task target.",
        color="#475569",
    )
    figure.text(
        0.065,
        0.035,
        "Inspect a run histogram     /     Change the task target     /     Explore plots and filters",
        color="#28447a",
    )
    figure.subplots_adjust(left=0.065, right=0.98, bottom=0.27, top=0.76, wspace=0.25)
    save(figure, "00_algorithm_boundary.png")


def width_preview(window):
    selected = window[window["meta.algorithm"] == "BERNSTEIN-VAZIRANI"].copy()
    selected["family"] = selected["meta.config_id"].str.split("-").str[-1]
    figure, axes = plt.subplots(1, 3, figsize=(17, 6))
    decorate(
        figure,
        axes,
        "QWARD / DSR screening across circuit width",
        "IBM Fez · window 1 · Bernstein–Vazirani · 1,024 shots per job · two jobs per configuration",
    )
    for axis, (metric, title, description) in zip(axes, METRICS):
        for family, color in zip(("SINGLE", "ALT", "ONES"), COLORS):
            rows = selected[selected["family"] == family]
            mean = rows.groupby("a.num_qubits")[metric].mean()
            axis.scatter(rows["a.num_qubits"], rows[metric], color=color, alpha=0.30, s=24)
            axis.plot(
                mean.index,
                mean.values,
                color=color,
                marker="o",
                markersize=4,
                linewidth=2,
                label=family,
            )
        axis.set(xlabel="Logical qubits, including the BV ancilla", ylim=(-0.03, 1.04))
        axis.set_ylabel("Metric value")
        axis.set_title(title, loc="left", pad=32, fontsize=15, weight="bold")
        axis.text(0, 1.02, description, transform=axis.transAxes, fontsize=9, color="#64748b")
        axis.set_xticks([3, 5, 7, 9, 11, 13, 15])
        axis.legend(frameon=False, loc="lower left", fontsize=9)
    figure.text(
        0.055,
        0.035,
        "Dots show individual jobs; lines join configuration means within each BV family. BV14 uses 15 logical qubits.",
        fontsize=10,
        color="#475569",
    )
    figure.subplots_adjust(left=0.055, right=0.98, top=0.78, bottom=0.17, wspace=0.27)
    save(figure, "01_metrics_vs_qubits.png")


def depth_preview(window):
    figure, axes = plt.subplots(1, 3, figsize=(17, 6))
    names = sorted(window["meta.algorithm"].unique())
    decorate(
        figure,
        axes,
        "QWARD / Screening and quality across executed depth",
        "IBM Fez · window 1 · all eight algorithms · each point is a two-job configuration mean",
    )
    means = window.groupby(["meta.algorithm", "meta.config_id"]).mean(numeric_only=True)
    for axis, (metric, title, description) in zip(axes, METRICS):
        for name, color in zip(names, COLORS):
            group = means.loc[name]
            axis.scatter(
                group["b.depth"],
                group[metric],
                color=color,
                s=28,
                alpha=0.75,
                edgecolor="white",
                linewidth=0.3,
            )
        axis.set(
            xlabel="Executed ISA depth · logarithmic scale",
            ylabel="Metric value",
            ylim=(-0.03, 1.04),
            xscale="log",
        )
        axis.set_title(title, loc="left", pad=32, fontsize=15, weight="bold")
        axis.text(0, 1.02, description, transform=axis.transAxes, fontsize=9, color="#64748b")
    labels = {
        "BERNSTEIN-VAZIRANI": "BV",
        "PHASE-ESTIMATION": "Phase estimation",
        "RANDOM-VOLUMETRIC": "Random volumetric",
    }
    handles = [
        Line2D([], [], marker="o", linestyle="", color=c, label=labels.get(n, n))
        for n, c in zip(names, COLORS)
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.51, 0.005),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    figure.subplots_adjust(left=0.055, right=0.98, top=0.78, bottom=0.25, wspace=0.27)
    save(figure, "02_metrics_vs_depth.png")


def histogram_preview(window):
    figure, axes = plt.subplots(1, 3, figsize=(18, 8))
    decorate(
        figure,
        axes,
        "QWARD / Inspect the measured histogram",
        "IBM Fez · window 1 · first repetition of each selected configuration · probabilities use all 1,024 shots",
    )
    for axis, config in zip(axes, ("GHZ10", "BV14-ONES", "VTP-S2-TRIPLE")):
        row = window[window["meta.config_id"] == config].sort_values("meta.run_index").iloc[0]
        path = Path(row["meta.record_path"])
        if not path.exists():
            algorithm_dir = {"BERNSTEIN-VAZIRANI": "bv", "GHZ": "ghz", "VTP": "vtp"}[
                row["meta.algorithm"]
            ]
            path = PAPERS / algorithm_dir / "data/qpu/v2" / path.name
        record = json.loads(path.read_text(encoding="utf-8"))
        prepared = json.loads(
            (path.parent / record["artifact_directory"] / "prepared.json").read_text(
                encoding="utf-8"
            )
        )
        counts = json.loads(row["f.counts_json"])
        reference = prepared["reference"]
        expected = set(prepared["payload"]["success_set"])
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        competing = [key for key, _ in ranked if key not in reference and key not in expected][:5]
        labels = sorted(set(reference) | expected) + competing
        shown = set(labels)
        observed = [counts.get(k, 0) / row["meta.shots"] for k in labels]
        ideal = [reference.get(k, 0) for k in labels]
        remainder = sum(v for k, v in counts.items() if k not in shown) / row["meta.shots"]
        if remainder > 0:
            labels.append("Other outcomes")
            observed.append(remainder)
            ideal.append(sum(v for k, v in reference.items() if k not in shown))
        positions = list(range(len(labels)))
        colors = ["#2563eb" if k in expected else "#94a3b8" for k in labels]
        axis.barh(positions, observed, color=colors, height=0.58)
        axis.scatter(ideal, positions, marker="D", s=40, color="#d97706", zorder=4)
        axis.set_yticks(positions, labels, fontfamily="monospace", fontsize=8)
        axis.invert_yaxis()
        axis.set(xlim=(0, 1.04), xlabel="Probability")
        axis.set_title(config, loc="left", pad=32, fontsize=15, weight="bold")
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.text(
            0,
            -0.20,
            f"DSR  {row['f.fidelity.dsr']:.3f}     Success  {row['f.fidelity.success_rate']:.1%}\nHellinger  {row['f.fidelity.hellinger_fidelity']:.3f}     TVD  {row['f.fidelity.tvd']:.3f}",
            transform=axis.transAxes,
            fontsize=10,
            linespacing=1.8,
            color="#0f172a",
        )
        exact = "Exact reference" if row["g.reference_exact"] else "Sampled ideal reference"
        axis.text(0, 1.025, exact, transform=axis.transAxes, fontsize=9, color="#64748b")
    figure.legend(
        handles=[
            Patch(color="#2563eb", label="Observed / accepted outcome"),
            Patch(color="#94a3b8", label="Observed / other outcome"),
            Line2D(
                [],
                [],
                marker="D",
                linestyle="",
                color="#d97706",
                label="Ideal reference probability",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=3,
        frameon=False,
    )
    figure.text(
        0.10,
        0.014,
        "Other outcomes aggregates omitted labels. DSR compares against individual competing peaks.",
        fontsize=9,
        color="#475569",
    )
    figure.subplots_adjust(left=0.10, right=0.98, top=0.80, bottom=0.28, wspace=0.75)
    save(figure, "03_histogram_explorer.png")


def main():
    frame = pd.read_csv(PAPERS / "dataset_out/dataset.csv")
    window = frame[
        (frame["meta.backend_name"] == "ibm_fez")
        & (frame["meta.campaign_id"] == "fez-dataset-v2")
        & (frame["meta.window"] == 1)
    ]
    if len(window) != 344 or not window["meta.job_id"].is_unique:
        raise ValueError("Expected the validated 344-job Fez window 1 export")
    boundary_preview(window)
    width_preview(window)
    depth_preview(window)
    histogram_preview(window)
    print(f"Saved four preview images to {OUTPUT}")


if __name__ == "__main__":
    main()
