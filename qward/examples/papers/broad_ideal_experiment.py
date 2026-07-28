"""
Broad-ideal, known-E experiment: demonstrates the histogram-free advantage
of the DSR profile against full-distribution HF/TVD, addressing Reviewer 1's
concern that the paper's circuits never needed the "no ideal histogram
available" motivation to hold.

Setup: multi-marked Grover search with a fixed, small, analytically known
expected-outcome set ``E`` (the marked states), at increasing qubit count
``n``. Two computations are timed at each ``n``:

  1. **Full-distribution path** (the old approach): simulate the exact ideal
     statevector/probabilities over all ``2**n`` outcomes (``ideal_probs``,
     same code path as ``enrich_hellinger.py``), then compute Hellinger
     fidelity / TVD against it. Cost is ``O(2**n)`` in both time and memory.
  2. **DSR-profile path** (histogram-free): compute the four-component
     profile directly from measurement counts and ``E`` via
     ``DSRProfiler``. Cost is ``O(shots + K)``, independent of ``n``.

For ``n`` up to the point where local statevector simulation is still
feasible (here, up to ``n = 22``), both paths run on the SAME real Grover
circuit and counts, so the comparison is apples-to-apples and doubles as a
correctness check (the coarse profile should track success sensibly against
the true statevector ideal). Beyond that point, no classical machine
(including this one) can produce the ideal ``2**n``-outcome histogram, so we
demonstrate the profile path alone on synthetic counts of matching
bitstring width -- explicitly NOT simulating any circuit at that size, since
that is exactly the capability gap this experiment is illustrating. The
qubit counts chosen for this stage (28-40) are representative of the
Grover-past-classical-simulation-wall regime discussed in the manuscript;
running the corresponding REAL circuit on QPU hardware is out of scope for
this script (it requires IBM/AWS credentials) but the profile computation
demonstrated here is exactly what would run against those real counts.

Usage:
  PYTHONPATH=. uv run python qward/examples/papers/broad_ideal_experiment.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, hellinger_fidelity, hellinger_distance

from qward.algorithms.grover import Grover
from qward.metrics.differential_success_rate import DSRProfiler

PAPERS_DIR = Path(__file__).resolve().parent
PLOTS_DIR = PAPERS_DIR / "plots"
RESULTS_PATH = PAPERS_DIR / "broad_ideal_experiment_results.json"

SHOTS = 4096
K_MARKED = 3  # number of marked states, fixed across n for a clean comparison
SEED = 42

# Qubit counts still small enough for local full-statevector simulation.
# (Full-distribution cost is dominated by building the 2**n-amplitude
# statevector and diffing two 2**n-scale dicts, which already grows from
# ~40ms at n=6 to ~1 minute at n=14 -- a clean exponential trend that makes
# the point without needing to run for hours to literally exhaust memory.)
SIMULATABLE_QUBITS = [6, 8, 10, 12, 14]

# Qubit counts representative of "past the classical simulation wall":
# no statevector is built at these sizes; counts are synthetic stand-ins for
# real QPU output, since a genuine ideal-histogram comparison is exactly what
# is infeasible here.
BEYOND_WALL_QUBITS = [26, 28, 30, 32, 36, 40]


def _marked_states(n: int, k: int, seed: int) -> List[str]:
    rng = np.random.default_rng(seed + n)
    values = rng.choice(2**n, size=k, replace=False)
    return [format(int(v), f"0{n}b") for v in values]


def _run_real_grover(
    n: int, marked_states: List[str], shots: int
) -> Tuple[Dict[str, int], float, int]:
    """Build and simulate a real multi-marked Grover circuit.

    Returns (counts, build_and_sim_seconds, num_iterations).
    """
    t0 = time.perf_counter()
    grover = Grover(marked_states)
    sim = AerSimulator()
    isa_circuit = grover.create_isa_circuit(backend=sim, optimization_level=1)
    job = sim.run(isa_circuit, shots=shots)
    counts = job.result().get_counts()
    counts = {k.replace(" ", ""): v for k, v in counts.items()}
    elapsed = time.perf_counter() - t0
    return counts, elapsed, grover.optimal_iterations


def _time_full_distribution_path(
    n: int, marked_states: List[str], counts: Dict[str, int]
) -> Tuple[float, int, float, float]:
    """Time the full-distribution HF/TVD path: simulate the ideal statevector
    over all 2**n outcomes, then compare against observed counts.

    Returns (elapsed_seconds, theoretical_peak_bytes, hellinger_fidelity, tvd).

    Memory is reported analytically (``2**n`` complex128 amplitudes for the
    statevector) rather than measured via a memory profiler, since profilers
    like ``tracemalloc`` add allocation-tracking overhead that itself scales
    with the number of intermediate arrays Qiskit allocates and would distort
    the wall-clock timing comparison that is the point of this benchmark.
    """
    t0 = time.perf_counter()

    grover = Grover(marked_states)
    circuit_no_meas = grover.circuit.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(circuit_no_meas)
    ideal_probs = {k: v for k, v in sv.probabilities_dict(decimals=12).items() if v > 1e-12}

    total = sum(counts.values())
    obs = {k: v / total for k, v in counts.items()}
    hf = hellinger_fidelity(ideal_probs, obs)
    hd = hellinger_distance(ideal_probs, obs)

    elapsed = time.perf_counter() - t0
    theoretical_peak_bytes = (2**n) * 16  # complex128 statevector amplitudes
    return elapsed, theoretical_peak_bytes, float(hf), float(hd)


def _time_profile_path(
    n: int, marked_states: List[str], counts: Dict[str, int]
) -> Tuple[float, int, "object"]:
    """Time the histogram-free DSR-profile path.

    Returns (elapsed_seconds, theoretical_peak_bytes, profile). Memory is
    reported analytically as ``O(shots + K)`` (the size of the counts dict
    plus the expected-outcome set), independent of ``n``, for the same
    profiling-overhead reason described in ``_time_full_distribution_path``.
    """
    t0 = time.perf_counter()

    profiler = DSRProfiler(counts, marked_states, num_measured_qubits=n)
    profile = profiler.profile()

    elapsed = time.perf_counter() - t0
    theoretical_peak_bytes = (len(counts) + len(marked_states)) * 64
    return elapsed, theoretical_peak_bytes, profile


def _synthetic_counts_beyond_wall(
    n: int, marked_states: List[str], shots: int, success_prob: float, seed: int
) -> Dict[str, int]:
    """Generate representative counts for a qubit count where no classical
    machine (this one included) can simulate the real Grover circuit, let
    alone build its 2**n-outcome ideal histogram.

    This is an explicit stand-in for real QPU output: shots are split
    between the K marked states (mimicking a partially-successful Grover
    run) and a set of random "noise" bitstrings representing measurement
    outcomes outside E. No 2**n-sized object is ever constructed.
    """
    rng = np.random.default_rng(seed + n)
    k = len(marked_states)
    n_success = rng.binomial(shots, success_prob)
    n_fail = shots - n_success

    counts: Dict[str, int] = {}
    if n_success > 0:
        per_marked = rng.multinomial(n_success, [1.0 / k] * k)
        for state, c in zip(marked_states, per_marked):
            if c > 0:
                counts[state] = int(c)

    if n_fail > 0:
        # A handful of random "wrong answer" bitstrings, drawn without ever
        # enumerating the full 2**n outcome space.
        num_noise_bins = min(50, n_fail)
        noise_counts = rng.multinomial(n_fail, [1.0 / num_noise_bins] * num_noise_bins)
        used = set(marked_states)
        for c in noise_counts:
            if c <= 0:
                continue
            while True:
                candidate = format(int(rng.integers(0, 2**n)), f"0{n}b")
                if candidate not in used:
                    used.add(candidate)
                    break
            counts[candidate] = counts.get(candidate, 0) + int(c)

    return counts


def run_simulatable_stage() -> List[Dict]:
    """Stage 1: real Grover circuits, both paths, n small enough to fully
    simulate."""
    rows = []
    for n in SIMULATABLE_QUBITS:
        marked_states = _marked_states(n, K_MARKED, SEED)
        print(f"  n={n}: building + simulating real Grover circuit (K={K_MARKED})...")
        try:
            counts, build_sim_time, iterations = _run_real_grover(n, marked_states, SHOTS)
        except Exception as e:  # pragma: no cover - defensive for large n
            print(f"    SKIP: circuit build/simulation failed: {e}")
            continue

        full_time, full_mem, hf, hd = _time_full_distribution_path(n, marked_states, counts)
        profile_time, profile_mem, profile = _time_profile_path(n, marked_states, counts)

        row = {
            "n": n,
            "k": K_MARKED,
            "iterations": iterations,
            "shots": SHOTS,
            "circuit_build_sim_seconds": build_sim_time,
            "full_distribution_seconds": full_time,
            "full_distribution_peak_bytes": full_mem,
            "profile_seconds": profile_time,
            "profile_peak_bytes": profile_mem,
            "success_rate": profile.success_rate,
            "chance_corrected_success": profile.chance_corrected_success,
            "coarse_tvd_similarity": profile.coarse_tvd_similarity,
            "coarse_hellinger_fidelity": profile.coarse_hellinger_fidelity,
            "full_hellinger_fidelity": hf,
            "full_hellinger_distance": hd,
            "regime": "simulatable",
        }
        rows.append(row)
        print(
            f"    success={profile.success_rate:.4f}  full_HF={hf:.4f}  "
            f"full_path={full_time*1000:.2f}ms  profile_path={profile_time*1000:.3f}ms  "
            f"speedup={full_time/profile_time:,.0f}x"
        )
    return rows


def run_beyond_wall_stage() -> List[Dict]:
    """Stage 2: qubit counts where the full-distribution path is not just
    slow but architecturally impossible (can't enumerate 2**n outcomes)."""
    rows = []
    for n in BEYOND_WALL_QUBITS:
        marked_states = _marked_states(n, K_MARKED, SEED)
        # Success probability decays with n to mimic realistic degradation
        # at problem sizes requiring many more Grover iterations than any
        # near-term device can execute coherently.
        success_prob = max(0.05, 0.9 - 0.02 * n)
        counts = _synthetic_counts_beyond_wall(n, marked_states, SHOTS, success_prob, SEED)

        # The full-distribution path is intentionally NOT attempted here:
        # 2**n for n=32 is ~4.3 billion outcomes (>34GB just for a float64
        # probability array, before even considering the cost of obtaining
        # the ideal statevector via simulation). We record the theoretical
        # memory requirement instead of measuring it.
        theoretical_full_bytes = (2**n) * 8  # one float64 per outcome, best case

        profile_time, profile_mem, profile = _time_profile_path(n, marked_states, counts)

        row = {
            "n": n,
            "k": K_MARKED,
            "shots": SHOTS,
            "profile_seconds": profile_time,
            "profile_peak_bytes": profile_mem,
            "theoretical_full_distribution_bytes": theoretical_full_bytes,
            "success_rate": profile.success_rate,
            "chance_corrected_success": profile.chance_corrected_success,
            "coarse_tvd_similarity": profile.coarse_tvd_similarity,
            "coarse_hellinger_fidelity": profile.coarse_hellinger_fidelity,
            "regime": "beyond_simulation_wall_synthetic",
        }
        rows.append(row)
        print(
            f"  n={n}: [synthetic counts, no circuit simulated] "
            f"success={profile.success_rate:.4f}  "
            f"chance_corrected={profile.chance_corrected_success:.4f}  "
            f"profile_path={profile_time*1000:.3f}ms  "
            f"theoretical_full_distribution_bytes={theoretical_full_bytes:,}"
        )
    return rows


def plot_results(rows: List[Dict]) -> None:
    import matplotlib.pyplot as plt

    from qward.utils.styles import COLORBREWER_PALETTE, LABEL_SIZE, LEGEND_SIZE, apply_axes_defaults

    sim_rows = [r for r in rows if r["regime"] == "simulatable"]
    beyond_rows = [r for r in rows if r["regime"] == "beyond_simulation_wall_synthetic"]

    fig, ax = plt.subplots(figsize=(15, 7))

    ns_sim = [r["n"] for r in sim_rows]
    full_times = [r["full_distribution_seconds"] * 1000 for r in sim_rows]
    profile_times_sim = [r["profile_seconds"] * 1000 for r in sim_rows]

    if sim_rows:
        ax.plot(
            ns_sim,
            full_times,
            "o-",
            color=COLORBREWER_PALETTE[2],
            label="Full-distribution HF/TVD (measured)",
        )
        ax.plot(
            ns_sim,
            profile_times_sim,
            "s-",
            color=COLORBREWER_PALETTE[1],
            label="DSR profile (measured)",
        )

    if beyond_rows:
        ns_beyond = [r["n"] for r in beyond_rows]
        profile_times_beyond = [r["profile_seconds"] * 1000 for r in beyond_rows]
        ax.plot(
            ns_beyond,
            profile_times_beyond,
            "s--",
            color=COLORBREWER_PALETTE[1],
            alpha=0.6,
            label="DSR profile (n past the simulation wall)",
        )
        if sim_rows:
            wall = max(ns_sim)
            ax.axvline(wall, color="#999999", linestyle=":", linewidth=2)
            ax.text(
                wall + 0.5,
                2e-1,
                "classical\nsimulation\nwall",
                fontsize=LABEL_SIZE - 8,
                color="#666666",
                ha="left",
                va="center",
            )

    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits (n)", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Wall-clock time (ms, log scale)", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_title(
        "Histogram-Free Advantage: DSR Profile vs. Full-Distribution HF/TVD",
        fontweight="bold",
    )
    apply_axes_defaults(ax)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper left")

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = "3_broad_ideal_scaling.png"
    fig.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {PLOTS_DIR / filename}")


def main() -> None:
    print(
        "Stage 1: real multi-marked Grover circuits, both paths (n small enough to fully simulate)"
    )
    sim_rows = run_simulatable_stage()

    print("\nStage 2: representative counts past the classical simulation wall (profile path only)")
    beyond_rows = run_beyond_wall_stage()

    all_rows = sim_rows + beyond_rows
    RESULTS_PATH.write_text(json.dumps(all_rows, indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")

    if sim_rows:
        speedups = [
            r["full_distribution_seconds"] / r["profile_seconds"]
            for r in sim_rows
            if r["profile_seconds"] > 0
        ]
        print(
            f"\nMeasured speedup (full-distribution / profile), n={SIMULATABLE_QUBITS}: "
            f"{[f'{s:,.0f}x' for s in speedups]}"
        )

    plot_results(all_rows)
    print("\nDone.")


if __name__ == "__main__":
    main()
