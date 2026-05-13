"""Generate slide assets for the coin_toss qWard example.

Produces:
  - circuit.png          : 2-qubit fair coin_toss + CX circuit diagram
  - hist_ideal.png       : ideal simulator histogram
  - hist_noisy.png       : IBM Heron R1 noise-model histogram
  - metrics.json         : pre-runtime + post-runtime metrics

Output directory: docs/slides/proposal/img/coin_toss/

Run with:
    uv run qward/examples/papers/coin_toss/generate_slide_assets.py
"""

from __future__ import annotations

import json
import math
from math import sqrt
from pathlib import Path
from typing import Dict, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from qward.scanner import Scanner
from qward.metrics import (
    QiskitMetrics,
    ComplexityMetrics,
    StructuralMetrics,
)
from qward.algorithms import NoiseModelGenerator, get_preset_noise_config

REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = REPO_ROOT / "docs" / "slides" / "proposal" / "img" / "coin_toss"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHOTS = 1024
LABELS = ["00", "01", "10", "11"]


def build_circuit() -> QuantumCircuit:
    """Entangled 2-qubit coin toss — one Ry(pi/2) rotation + CX (Bell state).

    |00> -Ry(pi/2) on q0-> (|00> + |10>)/sqrt(2) -CX(0,1)-> (|00> + |11>)/sqrt(2).
    Outcomes: 50% |00>, 50% |11> — two correlated coins always matching.
    """
    qc = QuantumCircuit(2, 2, name="CoinToss_Bell")
    qc.ry(math.pi / 2, 0)
    qc.cx(0, 1)
    qc.barrier()
    qc.measure([0, 1], [0, 1])
    return qc


def build_scaling_circuit(n_cnots: int, n_qubits: int = 4) -> QuantumCircuit:
    """Generic multi-qubit circuit with N CNOTs cycling through neighbour pairs.

    Shows how pre-runtime metrics scale with circuit size.
    """
    qc = QuantumCircuit(n_qubits, n_qubits, name=f"Scale_{n_cnots}CX")
    for q in range(n_qubits):
        qc.h(q)
    pairs = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    for k in range(n_cnots):
        ctrl, tgt = pairs[k % len(pairs)]
        qc.cx(ctrl, tgt)
        # sprinkle rotations for depth/complexity
        if k % 7 == 6:
            qc.ry(math.pi / 3, (k * 3) % n_qubits)
    qc.barrier()
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def save_circuit_png(qc: QuantumCircuit, path: Path) -> None:
    fig = qc.draw(output="mpl", style={"backgroundcolor": "#ffffff"})
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_simulator(qc: QuantumCircuit, noise_model: NoiseModel | None, shots: int) -> Dict[str, int]:
    sim = AerSimulator(noise_model=noise_model) if noise_model else AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    return dict(result.get_counts())


def plot_histogram(counts: Mapping[str, int], title: str, path: Path, color: str) -> None:
    values = [counts.get(lab, 0) for lab in LABELS]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.bar(LABELS, values, color=color, edgecolor="#1a1a1a", linewidth=0.6)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Counts", fontsize=9)
    ax.set_xlabel("Measured state", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def counts_to_probs(counts: Mapping[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: v / total for k, v in counts.items()}


def hellinger_fidelity(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    keys = set(p) | set(q)
    bc = sum(sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in keys)
    return bc**2


def tvd(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def main() -> None:
    qc = build_circuit()

    print(f"Saving circuit to {OUT_DIR}/circuit.png")
    save_circuit_png(qc, OUT_DIR / "circuit.png")

    print("Running ideal simulator...")
    counts_ideal = run_simulator(qc, noise_model=None, shots=SHOTS)

    print("Running noisy simulator (COMB-HIGH — aggressive combined noise)...")
    noise_cfg = get_preset_noise_config("COMB-HIGH")
    noise_model = NoiseModelGenerator.create_from_config(noise_cfg)
    counts_noisy = run_simulator(qc, noise_model=noise_model, shots=SHOTS)

    print("Plotting histograms...")
    plot_histogram(counts_ideal, "Ideal simulator", OUT_DIR / "hist_ideal.png", "#16A085")
    plot_histogram(counts_noisy, "Noisy (COMB-HIGH)", OUT_DIR / "hist_noisy.png", "#F39C12")

    print("Running qWard Scanner...")
    scanner = Scanner(circuit=qc)
    scanner.add(QiskitMetrics)
    scanner.add(ComplexityMetrics)
    scanner.add(StructuralMetrics)
    metric_dfs = scanner.calculate_metrics()

    pre_runtime: Dict[str, Dict[str, object]] = {}
    for name, df in metric_dfs.items():
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        pre_runtime[name] = {
            k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v))
            for k, v in row.items()
        }

    # Post-runtime: expected distribution = Bell state (|00> and |11> only)
    expected = {"00": 0.5, "01": 0.0, "10": 0.0, "11": 0.5}
    p_ideal = counts_to_probs(counts_ideal)
    p_noisy = counts_to_probs(counts_noisy)

    # Success = outcome in Bell support {|00>, |11>}
    def success_rate(counts):
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return (counts.get("00", 0) + counts.get("11", 0)) / total

    post_runtime = {
        "shots": SHOTS,
        "expected_distribution": expected,
        "success_criterion": "outcome in {|00>, |11>}",
        "ideal": {
            "counts": counts_ideal,
            "probs": p_ideal,
            "success_rate": success_rate(counts_ideal),
            "hellinger_fidelity": hellinger_fidelity(p_ideal, expected),
            "tvd": tvd(p_ideal, expected),
        },
        "noisy": {
            "counts": counts_noisy,
            "probs": p_noisy,
            "success_rate": success_rate(counts_noisy),
            "hellinger_fidelity": hellinger_fidelity(p_noisy, expected),
            "tvd": tvd(p_noisy, expected),
        },
    }

    out = {
        "circuit_name": "CoinToss2 + CX",
        "num_qubits": 2,
        "shots": SHOTS,
        "pre_runtime": pre_runtime,
        "post_runtime": post_runtime,
    }

    out_path = OUT_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote metrics to {out_path}")

    # --- Scaling circuits: 10 CNOT and 100 CNOT variants (pre-runtime only) ---
    scaling_metrics = {}
    for n in (10, 100):
        print(f"\nScaling circuit — {n} CNOTs...")
        sqc = build_scaling_circuit(n)
        save_circuit_png(sqc, OUT_DIR / f"circuit_{n}cx.png")

        scanner_s = Scanner(circuit=sqc)
        scanner_s.add(QiskitMetrics)
        scanner_s.add(ComplexityMetrics)
        scanner_s.add(StructuralMetrics)
        dfs = scanner_s.calculate_metrics()
        entry = {}
        for name, df in dfs.items():
            if df.empty:
                continue
            row = df.iloc[0].to_dict()
            entry[name] = {
                k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v))
                for k, v in row.items()
            }
        scaling_metrics[f"n{n}"] = entry
        print(
            f"  depth={entry.get('QiskitMetrics', {}).get('basic_metrics.depth', '?')}, "
            f"size={entry.get('QiskitMetrics', {}).get('basic_metrics.size', '?')}, "
            f"cnot={entry.get('ComplexityMetrics', {}).get('gate_based_metrics.cnot_count', '?')}"
        )

    scaling_path = OUT_DIR / "scaling_metrics.json"
    with open(scaling_path, "w") as f:
        json.dump(scaling_metrics, f, indent=2, default=str)
    print(f"Wrote scaling metrics to {scaling_path}")

    print("\nSummary:")
    print(f"  Bell circuit: depth=3, size=4")
    print(
        f"  Ideal   HF = {post_runtime['ideal']['hellinger_fidelity']:.4f}, "
        f"TVD = {post_runtime['ideal']['tvd']:.4f}"
    )
    print(
        f"  Noisy   HF = {post_runtime['noisy']['hellinger_fidelity']:.4f}, "
        f"TVD = {post_runtime['noisy']['tvd']:.4f}"
    )


if __name__ == "__main__":
    main()
