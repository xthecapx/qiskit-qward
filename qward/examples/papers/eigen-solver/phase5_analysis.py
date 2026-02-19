"""
Phase 5: Execution & Analysis -- Quantum Eigensolver for Small Hermitian Matrices.

This script executes the full Phase 5 analysis pipeline:
  1. Ideal VQE on all 5 test matrices (statevector)
  2. Noisy VQE on all 5 test matrices (IBM Heron R1-R3, Rigetti Ankaa-3)
  3. Quantum vs. classical comparison table
  4. Convergence visualization (energy vs. iteration)
  5. Noise impact characterization
  6. Statistical significance analysis (n=10 trials per configuration)

Output:
  - results/  : CSV tables and JSON summary
  - img/      : PNG plots (convergence, noise impact, comparison)

Author: Quantum Data Scientist
Date: 2026-02-19
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# Matplotlib backend must be set before import
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy import stats

# Add the eigen-solver directory to sys.path for imports
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

# Add the QWARD root to path
_QWARD_ROOT = os.path.abspath(os.path.join(_PROJECT_DIR, "..", "..", "..", ".."))
if _QWARD_ROOT not in sys.path:
    sys.path.insert(0, _QWARD_ROOT)

from eigen_solver.src.quantum_eigensolver import QuantumEigensolver, EigensolverResult
from eigen_solver.src.classical_baseline import ClassicalEigensolver
from eigen_solver.src.pauli_decomposition import pauli_decompose

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(_PROJECT_DIR, "results")
IMG_DIR = os.path.join(_PROJECT_DIR, "img")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

N_TRIALS = 10  # Number of independent trials for statistical analysis
NOISE_PRESETS = ["IBM-HERON-R1", "IBM-HERON-R2", "IBM-HERON-R3", "RIGETTI-ANKAA3"]
SHOTS_NOISY = 8192
SHOTS_IDEAL_SHOTBASED = 4096

# ---------------------------------------------------------------------------
# Test Matrix Definitions (from Phase 1, verified eigenvalues)
# ---------------------------------------------------------------------------

TEST_MATRICES = {
    "M1_PauliZ": {
        "matrix": np.array([[1, 0], [0, -1]], dtype=complex),
        "eigenvalues": [-1.0, 1.0],
        "spectral_range": 2.0,
        "description": "Pauli Z (2x2, diagonal, trivial)",
        "pauli_terms": "Z",
    },
    "M2_PauliX": {
        "matrix": np.array([[0, 1], [1, 0]], dtype=complex),
        "eigenvalues": [-1.0, 1.0],
        "spectral_range": 2.0,
        "description": "Pauli X (2x2, off-diagonal)",
        "pauli_terms": "X",
    },
    "M3_General2x2": {
        "matrix": np.array([[2, 1 - 1j], [1 + 1j, 3]], dtype=complex),
        "eigenvalues": [1.0, 4.0],
        "spectral_range": 3.0,
        "description": "General Hermitian (2x2, complex off-diagonal)",
        "pauli_terms": "2.5*I + 1.0*X + 1.0*Y - 0.5*Z",
    },
    "M4_Symmetric3x3": {
        "matrix": np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=complex),
        "eigenvalues": [1.0, 2.0, 4.0],
        "spectral_range": 3.0,
        "description": "Symmetric real (3x3, embedded in 4x4)",
        "pauli_terms": "8 terms (embedded)",
    },
    "M5_HeisenbergXXX": {
        "matrix": np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 2, 0],
                [0, 2, -1, 0],
                [0, 0, 0, 1],
            ],
            dtype=complex,
        ),
        "eigenvalues": [-3.0, 1.0, 1.0, 1.0],
        "spectral_range": 4.0,
        "description": "Heisenberg XXX (4x4, entangled ground state)",
        "pauli_terms": "XX + YY + ZZ",
    },
}

# Absolute thresholds from Phase 1, Section 4.3
IDEAL_THRESHOLDS = {
    "M1_PauliZ": 0.020,
    "M2_PauliX": 0.020,
    "M3_General2x2": 0.030,
    "M4_Symmetric3x3": 0.030,
    "M5_HeisenbergXXX": 0.040,
}

NOISY_THRESHOLDS = {
    "M1_PauliZ": 0.100,
    "M2_PauliX": 0.100,
    "M3_General2x2": 0.150,
    "M4_Symmetric3x3": 0.150,
    "M5_HeisenbergXXX": 0.200,
}


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def normalized_error(vqe_value, exact_value, spectral_range):
    """Compute normalized error relative to spectral range."""
    if spectral_range == 0:
        return 0.0
    return abs(vqe_value - exact_value) / spectral_range


def compute_trial_stats(values):
    """Compute descriptive statistics from a list of trial values."""
    arr = np.array(values)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    ci_95 = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem) if n > 1 else (mean, mean)
    median = np.median(arr)
    return {
        "n": n,
        "mean": float(mean),
        "std": float(std),
        "sem": float(sem),
        "median": float(median),
        "ci_95_lower": float(ci_95[0]),
        "ci_95_upper": float(ci_95[1]),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


# ---------------------------------------------------------------------------
# Phase 5.1: Ideal VQE Execution
# ---------------------------------------------------------------------------


def run_ideal_vqe(verbose=True):
    """Run ideal (statevector) VQE on all test matrices.

    Returns:
        dict: {matrix_name: EigensolverResult}
    """
    if verbose:
        print("=" * 70)
        print("PHASE 5.1: IDEAL VQE (Statevector Simulation)")
        print("=" * 70)

    results = {}
    for name, spec in TEST_MATRICES.items():
        matrix = spec["matrix"]
        expected_min = min(spec["eigenvalues"])
        spectral_range = spec["spectral_range"]

        if verbose:
            print(f"\n--- {name}: {spec['description']} ---")
            print(f"  Expected min eigenvalue: {expected_min}")

        t0 = time.time()
        solver = QuantumEigensolver(matrix, optimizer="COBYLA", maxiter=200)
        result = solver.solve(shots=None)
        elapsed = time.time() - t0

        err = normalized_error(result.eigenvalue, expected_min, spectral_range)
        threshold = IDEAL_THRESHOLDS[name]
        passed = err < (threshold / spectral_range + 0.001)

        if verbose:
            print(f"  VQE eigenvalue:     {result.eigenvalue:.6f}")
            print(f"  Normalized error:   {err:.6f} (threshold: {threshold/spectral_range:.4f})")
            print(f"  Absolute error:     {abs(result.eigenvalue - expected_min):.6f}")
            print(f"  Iterations:         {result.iterations}")
            print(f"  Converged:          {result.converged}")
            print(f"  Wall time:          {elapsed:.2f}s")
            print(f"  PASS: {'YES' if passed else 'NO'}")

        results[name] = {
            "result": result,
            "elapsed": elapsed,
            "normalized_error": err,
            "absolute_error": abs(result.eigenvalue - expected_min),
            "passed": passed,
        }

    return results


# ---------------------------------------------------------------------------
# Phase 5.1b: All Eigenvalues via Deflation (Ideal)
# ---------------------------------------------------------------------------


def run_ideal_deflation(verbose=True):
    """Run ideal VQE deflation to find all eigenvalues.

    Returns:
        dict: {matrix_name: {"quantum": [...], "classical": [...]}}
    """
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 5.1b: ALL EIGENVALUES VIA DEFLATION (Ideal)")
        print("=" * 70)

    results = {}
    for name, spec in TEST_MATRICES.items():
        matrix = spec["matrix"]
        expected = sorted(spec["eigenvalues"])

        if verbose:
            print(f"\n--- {name} ---")
            print(f"  Expected eigenvalues: {expected}")

        t0 = time.time()
        solver = QuantumEigensolver(matrix, optimizer="COBYLA", maxiter=200)
        quantum_eigenvalues = solver.solve_all()
        elapsed = time.time() - t0

        classical_solver = ClassicalEigensolver(matrix)
        classical_eigenvalues = classical_solver.solve_all()

        if verbose:
            print(f"  Quantum eigenvalues:  {[f'{v:.4f}' for v in quantum_eigenvalues]}")
            print(f"  Classical eigenvalues: {[f'{v:.4f}' for v in classical_eigenvalues]}")
            print(f"  Wall time: {elapsed:.2f}s")

        results[name] = {
            "quantum": quantum_eigenvalues,
            "classical": classical_eigenvalues,
            "expected": expected,
            "elapsed": elapsed,
        }

    return results


# ---------------------------------------------------------------------------
# Phase 5.2: Noisy VQE Execution
# ---------------------------------------------------------------------------


def run_noisy_vqe(verbose=True):
    """Run noisy VQE on all test matrices with each noise preset.

    Returns:
        dict: {matrix_name: {preset: EigensolverResult}}
    """
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 5.2: NOISY VQE (Shot-Based with Noise Models)")
        print("=" * 70)

    results = {}
    for name, spec in TEST_MATRICES.items():
        matrix = spec["matrix"]
        expected_min = min(spec["eigenvalues"])
        spectral_range = spec["spectral_range"]

        if verbose:
            print(f"\n--- {name}: {spec['description']} ---")

        results[name] = {}
        for preset in NOISE_PRESETS:
            if verbose:
                print(f"  Noise preset: {preset}")

            t0 = time.time()
            solver = QuantumEigensolver(
                matrix,
                optimizer="COBYLA",
                noise_preset=preset,
                shots=SHOTS_NOISY,
                maxiter=200,
                num_restarts=3,
            )
            result = solver.solve()
            elapsed = time.time() - t0

            err = normalized_error(result.eigenvalue, expected_min, spectral_range)
            threshold = NOISY_THRESHOLDS[name]

            if verbose:
                print(f"    VQE eigenvalue: {result.eigenvalue:.6f}")
                print(f"    Normalized error: {err:.6f}")
                print(f"    Absolute error: {abs(result.eigenvalue - expected_min):.6f}")
                print(f"    Iterations: {result.iterations}")
                print(f"    Wall time: {elapsed:.2f}s")

            results[name][preset] = {
                "result": result,
                "elapsed": elapsed,
                "normalized_error": err,
                "absolute_error": abs(result.eigenvalue - expected_min),
            }

    return results


# ---------------------------------------------------------------------------
# Phase 5.3: Statistical Significance Analysis (n=10 trials)
# ---------------------------------------------------------------------------


def run_statistical_analysis(verbose=True):
    """Run n=10 independent trials for each matrix under each noise preset.

    Returns:
        dict: {matrix_name: {preset: trial_stats}}
    """
    if verbose:
        print("\n" + "=" * 70)
        print(f"PHASE 5.3: STATISTICAL ANALYSIS (n={N_TRIALS} trials)")
        print("=" * 70)

    all_stats = {}
    for name, spec in TEST_MATRICES.items():
        matrix = spec["matrix"]
        expected_min = min(spec["eigenvalues"])
        spectral_range = spec["spectral_range"]

        if verbose:
            print(f"\n--- {name} ---")

        all_stats[name] = {}

        # Ideal statevector trials -- use different random seeds per trial
        # to get independent samples. The solver's solve() uses a fixed seed
        # (rng=42), so we call _run_single_vqe directly with varying initial
        # parameters.
        if verbose:
            print(f"  Running {N_TRIALS} ideal trials...")
        ideal_eigenvalues = []
        ideal_errors = []
        ideal_iterations = []
        solver = QuantumEigensolver(
            matrix,
            optimizer="COBYLA",
            maxiter=200,
            num_restarts=1,
        )
        num_params = solver.ansatz.num_parameters
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(1000 + trial)
            initial_params = rng.uniform(0, np.pi, size=num_params)
            result = solver._run_single_vqe(solver.hamiltonian, None, initial_params)
            ideal_eigenvalues.append(result.eigenvalue)
            ideal_errors.append(normalized_error(result.eigenvalue, expected_min, spectral_range))
            ideal_iterations.append(result.iterations)

        ideal_stats = compute_trial_stats(ideal_eigenvalues)
        ideal_stats["error_stats"] = compute_trial_stats(ideal_errors)
        ideal_stats["iteration_stats"] = compute_trial_stats(ideal_iterations)

        threshold = IDEAL_THRESHOLDS[name] / spectral_range
        n_pass = sum(1 for e in ideal_errors if e < threshold + 0.001)
        ideal_stats["n_pass"] = n_pass
        ideal_stats["pass_rate"] = n_pass / N_TRIALS

        if verbose:
            print(
                f"    Ideal: mean={ideal_stats['mean']:.6f} "
                f"+/- {ideal_stats['std']:.6f}, "
                f"pass rate: {n_pass}/{N_TRIALS}"
            )

        all_stats[name]["ideal"] = ideal_stats

        # Noisy trials for each preset -- use different random seeds per trial
        for preset in NOISE_PRESETS:
            if verbose:
                print(f"  Running {N_TRIALS} noisy trials ({preset})...")
            noisy_eigenvalues = []
            noisy_errors = []
            noisy_iterations = []
            solver_noisy = QuantumEigensolver(
                matrix,
                optimizer="COBYLA",
                noise_preset=preset,
                shots=SHOTS_NOISY,
                maxiter=200,
                num_restarts=1,
            )
            num_params_n = solver_noisy.ansatz.num_parameters
            for trial in range(N_TRIALS):
                rng = np.random.default_rng(2000 + trial * 100)
                initial_params = rng.uniform(0, np.pi, size=num_params_n)
                result = solver_noisy._run_single_vqe(
                    solver_noisy.hamiltonian, SHOTS_NOISY, initial_params
                )
                noisy_eigenvalues.append(result.eigenvalue)
                noisy_errors.append(
                    normalized_error(result.eigenvalue, expected_min, spectral_range)
                )
                noisy_iterations.append(result.iterations)

            noisy_stats = compute_trial_stats(noisy_eigenvalues)
            noisy_stats["error_stats"] = compute_trial_stats(noisy_errors)
            noisy_stats["iteration_stats"] = compute_trial_stats(noisy_iterations)

            threshold = NOISY_THRESHOLDS[name] / spectral_range
            n_pass = sum(1 for e in noisy_errors if e < threshold + 0.001)
            noisy_stats["n_pass"] = n_pass
            noisy_stats["pass_rate"] = n_pass / N_TRIALS
            noisy_stats["eigenvalues"] = noisy_eigenvalues
            noisy_stats["errors"] = noisy_errors

            if verbose:
                print(
                    f"    {preset}: mean={noisy_stats['mean']:.6f} "
                    f"+/- {noisy_stats['std']:.6f}, "
                    f"pass rate: {n_pass}/{N_TRIALS}"
                )

            all_stats[name][preset] = noisy_stats

    return all_stats


# ---------------------------------------------------------------------------
# Phase 5.4: Generate Comparison Table
# ---------------------------------------------------------------------------


def generate_comparison_table(ideal_results, noisy_results, deflation_results):
    """Generate and save the quantum vs. classical comparison table."""
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON TABLE")
    print("=" * 70)

    rows = []
    for name, spec in TEST_MATRICES.items():
        expected_min = min(spec["eigenvalues"])
        spectral_range = spec["spectral_range"]

        # Ideal VQE result
        ideal = ideal_results[name]
        row = {
            "Matrix": name,
            "Description": spec["description"],
            "Qubits": int(np.ceil(np.log2(max(spec["matrix"].shape[0], 2)))),
            "Classical_Min_Eigenvalue": expected_min,
            "VQE_Ideal_Eigenvalue": ideal["result"].eigenvalue,
            "VQE_Ideal_Abs_Error": ideal["absolute_error"],
            "VQE_Ideal_Norm_Error": ideal["normalized_error"],
            "VQE_Ideal_Iterations": ideal["result"].iterations,
            "VQE_Ideal_Time_s": ideal["elapsed"],
            "VQE_Ideal_Pass": ideal["passed"],
        }

        # Noisy results for each preset
        for preset in NOISE_PRESETS:
            noisy = noisy_results[name][preset]
            row[f"VQE_{preset}_Eigenvalue"] = noisy["result"].eigenvalue
            row[f"VQE_{preset}_Abs_Error"] = noisy["absolute_error"]
            row[f"VQE_{preset}_Norm_Error"] = noisy["normalized_error"]

        # Deflation results
        defl = deflation_results[name]
        row["Deflation_Quantum"] = str([f"{v:.4f}" for v in defl["quantum"]])
        row["Deflation_Classical"] = str([f"{v:.4f}" for v in defl["classical"]])

        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved comparison table to {csv_path}")

    # Print a nicely formatted summary
    print("\n  QUANTUM vs. CLASSICAL COMPARISON SUMMARY:")
    print("  " + "-" * 66)
    fmt = "  {:<20s} {:>10s} {:>10s} {:>10s} {:>10s}"
    print(fmt.format("Matrix", "Classical", "VQE Ideal", "Abs Error", "Norm Err"))
    print("  " + "-" * 66)
    for _, row in df.iterrows():
        print(
            fmt.format(
                row["Matrix"],
                f"{row['Classical_Min_Eigenvalue']:.4f}",
                f"{row['VQE_Ideal_Eigenvalue']:.4f}",
                f"{row['VQE_Ideal_Abs_Error']:.6f}",
                f"{row['VQE_Ideal_Norm_Error']:.6f}",
            )
        )
    print("  " + "-" * 66)

    return df


# ---------------------------------------------------------------------------
# Phase 5.5: Convergence Visualization
# ---------------------------------------------------------------------------


def plot_convergence(ideal_results):
    """Plot VQE convergence curves (energy vs. iteration) for all matrices."""
    print("\n" + "=" * 70)
    print("GENERATING CONVERGENCE PLOTS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (name, spec) in enumerate(TEST_MATRICES.items()):
        ax = axes[idx]
        result = ideal_results[name]["result"]
        expected_min = min(spec["eigenvalues"])

        if result.cost_history is not None and len(result.cost_history) > 0:
            iterations = range(1, len(result.cost_history) + 1)
            ax.plot(
                iterations,
                result.cost_history,
                color=colors[idx],
                linewidth=1.5,
                label="VQE energy",
            )
            ax.axhline(
                y=expected_min,
                color="red",
                linestyle="--",
                linewidth=1.0,
                label=f"Exact = {expected_min}",
            )
            ax.set_xlabel("Function Evaluations")
            ax.set_ylabel("Energy")
            ax.set_title(f"{name}\n{spec['description']}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                "No cost history available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{name}")

    # Remove unused subplot
    axes[-1].set_visible(False)

    fig.suptitle(
        "VQE Convergence: Energy vs. Function Evaluations (Ideal Simulation)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = os.path.join(IMG_DIR, "convergence_ideal.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved convergence plot to {path}")


# ---------------------------------------------------------------------------
# Phase 5.6: Noise Impact Characterization
# ---------------------------------------------------------------------------


def plot_noise_impact(noisy_results, stat_results):
    """Plot noise impact across presets for all matrices."""
    print("\n" + "=" * 70)
    print("GENERATING NOISE IMPACT PLOTS")
    print("=" * 70)

    # --- Plot 1: Normalized error by noise preset (bar chart) ---
    fig, ax = plt.subplots(figsize=(14, 7))

    matrix_names = list(TEST_MATRICES.keys())
    presets = NOISE_PRESETS
    n_matrices = len(matrix_names)
    n_presets = len(presets)
    bar_width = 0.18
    x = np.arange(n_matrices)

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    for j, preset in enumerate(presets):
        errors = []
        for name in matrix_names:
            errors.append(noisy_results[name][preset]["normalized_error"])
        ax.bar(
            x + j * bar_width,
            errors,
            bar_width,
            label=preset,
            color=colors[j],
            alpha=0.85,
        )

    # Add threshold lines
    for i, name in enumerate(matrix_names):
        spectral_range = TEST_MATRICES[name]["spectral_range"]
        threshold = NOISY_THRESHOLDS[name] / spectral_range
        ax.plot(
            [i - 0.1, i + n_presets * bar_width],
            [threshold, threshold],
            "k--",
            linewidth=0.8,
            alpha=0.5,
        )

    ax.set_xlabel("Test Matrix", fontsize=12)
    ax.set_ylabel("Normalized Error", fontsize=12)
    ax.set_title(
        "Noise Impact: Normalized Error by Hardware Preset",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x + bar_width * (n_presets - 1) / 2)
    ax.set_xticklabels(matrix_names, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    path = os.path.join(IMG_DIR, "noise_impact_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved noise impact bar chart to {path}")

    # --- Plot 2: Statistical spread (box plot from n=10 trials) ---
    if stat_results:
        fig, axes = plt.subplots(1, len(matrix_names), figsize=(20, 6), sharey=True)
        if len(matrix_names) == 1:
            axes = [axes]

        for i, name in enumerate(matrix_names):
            ax = axes[i]
            data = []
            labels = []

            for preset in presets:
                if preset in stat_results[name]:
                    s = stat_results[name][preset]
                    if "errors" in s:
                        data.append(s["errors"])
                        labels.append(preset.replace("IBM-HERON-", "H").replace("RIGETTI-", "R-"))

            if data:
                bp = ax.boxplot(data, labels=labels, patch_artist=True)
                for patch, color in zip(bp["boxes"], colors[: len(data)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.5)

            spectral_range = TEST_MATRICES[name]["spectral_range"]
            threshold = NOISY_THRESHOLDS[name] / spectral_range
            ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1, label="5% threshold")
            ax.set_title(name, fontsize=10)
            ax.tick_params(axis="x", rotation=45)
            if i == 0:
                ax.set_ylabel("Normalized Error")
            ax.grid(True, axis="y", alpha=0.3)

        fig.suptitle(
            f"Error Distribution Across Noise Presets (n={N_TRIALS} trials)",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        path = os.path.join(IMG_DIR, "noise_impact_boxplots.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved noise impact box plots to {path}")


# ---------------------------------------------------------------------------
# Phase 5.7: Eigenvalue Deflation Visualization
# ---------------------------------------------------------------------------


def plot_deflation_comparison(deflation_results):
    """Plot quantum vs. classical eigenvalues side by side."""
    print("\n" + "=" * 70)
    print("GENERATING DEFLATION COMPARISON PLOT")
    print("=" * 70)

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    for idx, (name, spec) in enumerate(TEST_MATRICES.items()):
        ax = axes[idx]
        defl = deflation_results[name]
        q_eigs = defl["quantum"]
        c_eigs = defl["classical"]

        x_q = np.zeros(len(q_eigs))
        x_c = np.ones(len(c_eigs))

        ax.scatter(x_q, q_eigs, marker="o", s=80, c="#1f77b4", zorder=5, label="VQE")
        ax.scatter(x_c, c_eigs, marker="s", s=80, c="#d62728", zorder=5, label="Classical")

        # Draw connecting lines
        for qe, ce in zip(sorted(q_eigs), sorted(c_eigs)):
            ax.plot([0, 1], [qe, ce], "k-", alpha=0.2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["VQE", "Classical"], fontsize=9)
        ax.set_ylabel("Eigenvalue")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "All Eigenvalues: VQE Deflation vs. Classical (Ideal)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    path = os.path.join(IMG_DIR, "deflation_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved deflation comparison to {path}")


# ---------------------------------------------------------------------------
# Phase 5.8: Statistical Summary Table
# ---------------------------------------------------------------------------


def save_statistical_summary(stat_results):
    """Save statistical analysis results to CSV and JSON."""
    print("\n" + "=" * 70)
    print("SAVING STATISTICAL SUMMARY")
    print("=" * 70)

    rows = []
    for name, presets in stat_results.items():
        expected_min = min(TEST_MATRICES[name]["eigenvalues"])
        spectral_range = TEST_MATRICES[name]["spectral_range"]

        for preset_name, s in presets.items():
            row = {
                "Matrix": name,
                "Condition": preset_name,
                "Expected_Min": expected_min,
                "Spectral_Range": spectral_range,
                "Mean_Eigenvalue": s["mean"],
                "Std_Eigenvalue": s["std"],
                "SEM": s["sem"],
                "CI_95_Lower": s["ci_95_lower"],
                "CI_95_Upper": s["ci_95_upper"],
                "Median_Eigenvalue": s["median"],
                "Min_Eigenvalue": s["min"],
                "Max_Eigenvalue": s["max"],
                "N_Trials": s["n"],
            }
            if "error_stats" in s:
                row["Mean_Norm_Error"] = s["error_stats"]["mean"]
                row["Std_Norm_Error"] = s["error_stats"]["std"]
            if "n_pass" in s:
                row["N_Pass"] = s["n_pass"]
                row["Pass_Rate"] = s["pass_rate"]
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "statistical_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved statistical summary CSV to {csv_path}")

    # Also save as JSON (excluding numpy arrays)
    json_safe = {}
    for name, presets in stat_results.items():
        json_safe[name] = {}
        for pname, s in presets.items():
            safe = {k: v for k, v in s.items() if k not in ("eigenvalues", "errors")}
            json_safe[name][pname] = safe

    json_path = os.path.join(RESULTS_DIR, "statistical_summary.json")
    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2)
    print(f"  Saved statistical summary JSON to {json_path}")

    # Print pass rate summary
    print("\n  STATISTICAL PASS RATE SUMMARY:")
    print("  " + "-" * 60)
    fmt = "  {:<20s} {:<18s} {:>5s}/{:<5s} {:>8s}"
    print(fmt.format("Matrix", "Condition", "Pass", "Total", "Rate"))
    print("  " + "-" * 60)
    for _, row in df.iterrows():
        if "N_Pass" in row and not pd.isna(row.get("N_Pass")):
            print(
                fmt.format(
                    row["Matrix"],
                    row["Condition"],
                    str(int(row["N_Pass"])),
                    str(int(row["N_Trials"])),
                    f"{row['Pass_Rate']:.0%}",
                )
            )
    print("  " + "-" * 60)

    return df


# ---------------------------------------------------------------------------
# Phase 5.9: Summary Heat Map
# ---------------------------------------------------------------------------


def plot_summary_heatmap(stat_results):
    """Plot a summary heatmap of pass rates across matrices and presets."""
    print("\n  Generating summary heatmap...")

    matrix_names = list(TEST_MATRICES.keys())
    conditions = ["ideal"] + NOISE_PRESETS
    pass_rates = np.zeros((len(matrix_names), len(conditions)))

    for i, name in enumerate(matrix_names):
        for j, cond in enumerate(conditions):
            if cond in stat_results[name]:
                s = stat_results[name][cond]
                pass_rates[i, j] = s.get("pass_rate", 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pass_rates, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix_names)))
    ax.set_yticklabels(matrix_names)

    # Annotate cells
    for i in range(len(matrix_names)):
        for j in range(len(conditions)):
            val = pass_rates[i, j]
            color = "black" if 0.3 < val < 0.8 else "white"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", color=color, fontsize=11)

    fig.colorbar(im, label="Pass Rate")
    ax.set_title(
        f"VQE Pass Rate Summary (n={N_TRIALS} trials, threshold: 1% ideal / 5% noisy)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    path = os.path.join(IMG_DIR, "pass_rate_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved pass rate heatmap to {path}")


# ---------------------------------------------------------------------------
# Phase 5.10: Pauli Decomposition Verification Table
# ---------------------------------------------------------------------------


def verify_pauli_decompositions(verbose=True):
    """Verify Pauli decompositions and save report."""
    if verbose:
        print("\n" + "=" * 70)
        print("PAULI DECOMPOSITION VERIFICATION")
        print("=" * 70)

    rows = []
    for name, spec in TEST_MATRICES.items():
        matrix = spec["matrix"]
        decomp = pauli_decompose(matrix)
        reconstructed = decomp.to_matrix()

        # Compare original (possibly padded) matrix
        n = matrix.shape[0]
        if decomp.was_padded:
            # Compare only the original block
            original_block = reconstructed[:n, :n]
            reconstruction_error = np.max(np.abs(original_block - matrix))
        else:
            reconstruction_error = np.max(np.abs(reconstructed - matrix))

        row = {
            "Matrix": name,
            "Num_Qubits": decomp.num_qubits,
            "Num_Pauli_Terms": decomp.num_terms,
            "Was_Padded": decomp.was_padded,
            "Padding_Penalty": decomp.padding_penalty,
            "Reconstruction_Error": float(reconstruction_error),
            "Labels": ", ".join(decomp.labels),
        }
        rows.append(row)

        if verbose:
            print(f"\n  {name}:")
            print(f"    Qubits: {decomp.num_qubits}")
            print(f"    Pauli terms: {decomp.num_terms}")
            print(f"    Labels: {decomp.labels}")
            coeffs_str = ", ".join(
                f"{l}:{c:.4f}" for l, c in zip(decomp.labels, decomp.coefficients)
            )
            print(f"    Coefficients: {coeffs_str}")
            print(f"    Padded: {decomp.was_padded}")
            print(f"    Reconstruction error: {reconstruction_error:.2e}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "pauli_decompositions.csv")
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"\n  Saved Pauli decomposition table to {csv_path}")

    return df


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main():
    """Run the full Phase 5 analysis pipeline."""
    print("\n" + "#" * 70)
    print("#  PHASE 5: EXECUTION & ANALYSIS")
    print("#  Quantum Eigensolver for Small Hermitian Matrices")
    print("#" * 70)
    total_start = time.time()

    # Step 0: Verify Pauli decompositions
    pauli_df = verify_pauli_decompositions()

    # Step 1: Ideal VQE
    ideal_results = run_ideal_vqe()

    # Step 2: Ideal deflation (all eigenvalues)
    deflation_results = run_ideal_deflation()

    # Step 3: Noisy VQE (single run per preset)
    noisy_results = run_noisy_vqe()

    # Step 4: Statistical analysis (n=10 trials)
    stat_results = run_statistical_analysis()

    # Step 5: Generate comparison table
    comparison_df = generate_comparison_table(ideal_results, noisy_results, deflation_results)

    # Step 6: Convergence plots
    plot_convergence(ideal_results)

    # Step 7: Noise impact plots
    plot_noise_impact(noisy_results, stat_results)

    # Step 8: Deflation comparison plot
    plot_deflation_comparison(deflation_results)

    # Step 9: Save statistical summary
    stat_df = save_statistical_summary(stat_results)

    # Step 10: Summary heatmap
    plot_summary_heatmap(stat_results)

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print("PHASE 5 COMPLETE")
    print("=" * 70)
    print(f"  Total wall time: {total_elapsed:.1f}s")
    print(f"  Results directory: {RESULTS_DIR}")
    print(f"  Images directory: {IMG_DIR}")
    print(f"\n  Output files:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"    results/{f}")
    for f in sorted(os.listdir(IMG_DIR)):
        print(f"    img/{f}")

    return {
        "ideal": ideal_results,
        "deflation": deflation_results,
        "noisy": noisy_results,
        "stats": stat_results,
    }


if __name__ == "__main__":
    main()
