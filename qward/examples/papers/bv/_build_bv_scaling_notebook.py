#!/usr/bin/env python3
"""One-shot builder: writes Quantum_Benchmark_26/bv_scaling_dsr.ipynb (+ plan copy)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from textwrap import dedent

NB_PATH = Path("/Users/cristianmarquezbarrios/Documents/code/Quantum_Benchmark_26/bv_scaling_dsr.ipynb")
PLAN_SRC = Path(__file__).resolve().parent / "bv_scaling_dsr_plan.md"
PLAN_DST = Path("/Users/cristianmarquezbarrios/Documents/code/Quantum_Benchmark_26/bv_scaling_dsr_plan.md")


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).lstrip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).lstrip("\n").splitlines(keepends=True),
    }


def build_cells() -> list[dict]:
    cells: list[dict] = []

    cells.append(
        md(
            """
            # Bernstein–Vazirani Scaling Experiment — DSR-Only Validation Past the Simulation Wall

            See [`bv_scaling_dsr_plan.md`](./bv_scaling_dsr_plan.md) for the full written plan and rationale.

            **Goal.** Grow the Bernstein–Vazirani (BV) circuit until the *ideal* histogram needed for
            full-distribution Hellinger Fidelity (HF) / TVD Fidelity becomes classically infeasible
            on this laptop (the "simulation wall", ~**30 total qubits**), while the **DSR profile**
            remains computable at any size because the expected outcome is the analytic secret
            (`secret[::-1]`, little-endian), not a simulated ideal distribution.

            **Technique.** Same as [`hair_colours_scaling_dsr.ipynb`](./hair_colours_scaling_dsr.ipynb):
            find the wall → run IBM beyond it → compute DSR only. This notebook also aggregates
            **existing** IBM BV runs (qward campaign + local `results/`) for multi-run validation.

            **Phases (v1 core):**
            1. Correctness validation (analytic secret vs. small-`n` statevector marginal)
            2. Growth & timing sweep (find the simulation wall)
            3. Backend feasibility (transpile candidates beyond the wall)
            4. IBM execution (manual `SUBMIT_JOB` flag)
            5. Multi-run DSR analysis (new jobs + existing saved runs)

            **Convention.** BV uses `n` secret bits + 1 ancilla = `n+1` total qubits; only `n` bits
            are measured. Default target: `TARGET_N = 29` → **30 total qubits**.
            """
        )
    )

    cells.append(
        code(
            r"""
            import json
            import math
            import os
            import sys
            import time
            from datetime import datetime, timezone
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector

            try:
                import psutil

                HAVE_PSUTIL = True
            except ImportError:
                HAVE_PSUTIL = False

            # Prefer the local qiskit-qward checkout (has BernsteinVazirani + DSRProfiler).
            QWARD_CHECKOUT = Path(
                os.environ.get(
                    "QWARD_CHECKOUT",
                    "/Users/cristianmarquezbarrios/Documents/code/qiskit-qward",
                )
            ).resolve()
            if QWARD_CHECKOUT.exists() and str(QWARD_CHECKOUT) not in sys.path:
                sys.path.insert(0, str(QWARD_CHECKOUT))

            RESULTS_DIR = Path("results")
            RESULTS_DIR.mkdir(exist_ok=True)
            PLOTS_DIR = RESULTS_DIR / "plots"
            PLOTS_DIR.mkdir(parents=True, exist_ok=True)

            QWARD_BV_RAW_DIR = Path(
                os.environ.get(
                    "QWARD_BV_RAW_DIR",
                    str(QWARD_CHECKOUT / "qward" / "examples" / "papers" / "bv" / "data" / "qpu" / "raw"),
                )
            )


            def ts() -> str:
                # UTC timestamp so re-runs never overwrite past results.
                return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


            print(f"psutil available: {HAVE_PSUTIL}")
            print(f"QWARD_CHECKOUT: {QWARD_CHECKOUT} (exists={QWARD_CHECKOUT.exists()})")
            print(f"QWARD_BV_RAW_DIR: {QWARD_BV_RAW_DIR} (exists={QWARD_BV_RAW_DIR.exists()})")
            """
        )
    )

    cells.append(
        md(
            """
            ## Circuit builder, analytic expected outcome, and DSR helpers

            Prefer `qward.algorithms.BernsteinVazirani` and
            `qward.metrics.differential_success_rate.compute_dsr_profile`. If the installed
            package is too old, fall back to an inline BV builder and Michelson DSR.
            """
        )
    )

    cells.append(
        code(
            r"""
            HAVE_QWARD_BV = False
            HAVE_QWARD_DSR = False

            try:
                from qward.algorithms import BernsteinVazirani

                HAVE_QWARD_BV = True
            except ImportError:
                BernsteinVazirani = None  # type: ignore

            try:
                from qward.metrics.differential_success_rate import (
                    compute_dsr_michelson,
                    compute_dsr_profile,
                )

                HAVE_QWARD_DSR = True
            except ImportError:
                compute_dsr_michelson = None  # type: ignore
                compute_dsr_profile = None  # type: ignore

            print(f"qward BernsteinVazirani: {HAVE_QWARD_BV}")
            print(f"qward DSR profile:       {HAVE_QWARD_DSR}")


            def expected_outcome_from_secret(secret_string: str) -> str:
                # Qiskit little-endian measurement of the query register.
                return secret_string[::-1]


            def secrets_for(n: int) -> dict[str, str]:
                # Fixed secret patterns to sweep (ONES / ALT / SINGLE).
                return {
                    "ONES": "1" * n,
                    "ALT": "".join("1" if i % 2 == 0 else "0" for i in range(n)),
                    "SINGLE": "0" * (n - 1) + "1",
                }


            def build_bv_circuit(secret_string: str, use_barriers: bool = True) -> QuantumCircuit:
                # Build BV circuit; prefer qward, else inline equivalent.
                if HAVE_QWARD_BV:
                    return BernsteinVazirani(secret_string, use_barriers=use_barriers).circuit

                n = len(secret_string)
                qc = QuantumCircuit(n + 1, n)
                qc.x(n)
                qc.h(range(n + 1))
                if use_barriers:
                    qc.barrier()
                for i, bit in enumerate(secret_string):
                    if bit == "1":
                        qc.cx(i, n)
                if use_barriers:
                    qc.barrier()
                qc.h(range(n))
                qc.measure(range(n), range(n))
                return qc


            def _clip01(x: float) -> float:
                return max(0.0, min(1.0, x))


            def compute_dsr_michelson_inline(counts: dict[str, int], expected_outcomes: set[str]) -> float:
                total = sum(counts.values())
                if total <= 0 or not expected_outcomes:
                    return 0.0
                p_exp = sum(counts.get(b, 0) for b in expected_outcomes) / total
                p_exp_bar = p_exp / len(expected_outcomes)
                p_comp = 0.0
                for outcome, count in counts.items():
                    if outcome not in expected_outcomes:
                        p_comp = max(p_comp, count / total)
                denom = p_exp_bar + p_comp
                if denom <= 0:
                    return 0.0
                return _clip01((p_exp_bar - p_comp) / denom)


            def dsr_profile_from_counts(counts: dict[str, int], expected: str) -> dict:
                # Return a flat DSR profile dict (library or inline fallback).
                counts = {str(k).replace(" ", ""): int(v) for k, v in counts.items()}
                expected = expected.replace(" ", "")
                total = sum(counts.values())
                success = counts.get(expected, 0) / total if total else 0.0
                m = len(expected)
                b = 1.0 / (2**m) if m > 0 else 1.0
                chance_corrected = _clip01((success - b) / (1.0 - b)) if (1.0 - b) > 1e-12 else 1.0
                michelson = (
                    float(compute_dsr_michelson(counts, {expected}))
                    if HAVE_QWARD_DSR
                    else compute_dsr_michelson_inline(counts, {expected})
                )

                if HAVE_QWARD_DSR:
                    profile = compute_dsr_profile(counts, {expected}, include_michelson=True)
                    flat = profile.to_flat_dict()
                    flat.setdefault("dsr_michelson", michelson)
                    flat["success_rate"] = flat.get("success_rate", success)
                    return flat

                return {
                    "shots": total,
                    "num_measured_qubits": m,
                    "expected_outcomes": [expected],
                    "num_expected_outcomes": 1,
                    "success_rate": success,
                    "chance_baseline": b,
                    "chance_corrected_success": chance_corrected,
                    "coarse_tvd_similarity": success,
                    "coarse_hellinger_fidelity": success,
                    "dsr_michelson": michelson,
                    "peak_mismatch": (max(counts, key=counts.get) != expected) if counts else True,
                }


            _secret = "1011"
            _qc = build_bv_circuit(_secret)
            print(_qc.draw(fold=-1))
            print("expected:", expected_outcome_from_secret(_secret))
            _example = {"1101": 900, "0000": 40, "1111": 40, "1010": 20}
            print("DSR profile smoke:", dsr_profile_from_counts(_example, expected_outcome_from_secret(_secret)))
            """
        )
    )

    cells.append(
        md(
            """
            ## Phase 1 — Correctness validation (small `n`)

            For each `n = 2..12` and each secret pattern, confirm that the measured-register
            marginal puts ~100% probability on `secret[::-1]`.
            """
        )
    )

    cells.append(
        code(
            r"""
            def circuit_without_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
                return circuit.remove_final_measurements(inplace=False)


            def top_measured_outcome_from_statevector(
                circuit: QuantumCircuit, n_measured: int
            ) -> tuple[str, float]:
                # Marginalize over the ancilla; return (bitstring, probability) on the query register.
                bare = circuit_without_measurements(circuit)
                sv = Statevector.from_instruction(bare)
                probs = sv.probabilities(qargs=list(range(n_measured)))
                top_idx = int(np.argmax(probs))
                bitstring = format(top_idx, f"0{n_measured}b")
                return bitstring, float(probs[top_idx])


            validation_rows = []
            for n in range(2, 13):
                for pattern_name, secret in secrets_for(n).items():
                    qc = build_bv_circuit(secret)
                    sim_bitstring, sim_prob = top_measured_outcome_from_statevector(qc, n)
                    predicted = expected_outcome_from_secret(secret)
                    match = (sim_bitstring == predicted) and math.isclose(sim_prob, 1.0, abs_tol=1e-9)
                    validation_rows.append(
                        {
                            "n_secret": n,
                            "total_qubits": n + 1,
                            "pattern_name": pattern_name,
                            "secret": secret,
                            "predicted": predicted,
                            "simulated": sim_bitstring,
                            "sim_probability": sim_prob,
                            "match": match,
                        }
                    )
                    assert match, (
                        f"Mismatch at n={n}, pattern={pattern_name}: "
                        f"predicted={predicted}, simulated={sim_bitstring}, p={sim_prob}"
                    )

            validation_df = pd.DataFrame(validation_rows)
            print(f"All {len(validation_df)} classical-vs-quantum checks passed.")
            validation_df.tail(9)
            """
        )
    )

    cells.append(
        md(
            """
            ## Phase 2 — Growth & timing sweep (find the simulation wall)

            Record circuit metrics for every `n`, and statevector time where memory allows.
            Memory prediction: `16 * 2**(n+1) * safety` vs available RAM.

            Each statevector attempt runs in a **child process** with a hard
            `ATTEMPT_TIMEOUT_S` deadline. If Docker is thrashing / swapping, the
            attempt is killed and treated as the wall (no manual Kernel interrupt
            required for that step). Soft budget `TIME_BUDGET_S` still stops the
            sweep when a successful attempt already took too long.

            Wall confirmed on this machine: **27 total qubits** (`n_secret=26`
            timed out; last OK `n_secret=25` ~31s). `FORCE_NS` is empty;
            `CONFIRMED_WALL_N_SECRET = 25` locks that result. Past the wall,
            full HF/TVD need an ideal histogram we cannot build; DSR / coarse
            profile only need the analytic secret + hardware counts.
            """
        )
    )

    cells.append(
        code(
            r"""
            import multiprocessing as mp

            MIN_N_SECRET = 2
            MAX_N_SECRET = 32
            TIME_BUDGET_S = 600  # stop sweep if a successful attempt already took this long
            ATTEMPT_TIMEOUT_S = 180  # hard kill per statevector attempt (Docker-safe)
            BYTES_PER_AMPLITUDE = 16
            MEMORY_SAFETY_FACTOR = 4
            MEMORY_USE_FRACTION = 0.6
            ASSUMED_AVAILABLE_MEMORY_GB = 8
            TIMING_PATTERN_NAME = "ALT"
            # Wall confirmed on this laptop/Docker run:
            #   n=25 (26 qubits) OK ~31s; n=26 (27 qubits) TIMEOUT 180s.
            # Do not force further statevector attempts.
            FORCE_NS: set[int] = set()
            CONFIRMED_WALL_N_SECRET = 25  # last successful; 27 total qubits is the wall


            def available_memory_bytes() -> int:
                if HAVE_PSUTIL:
                    return psutil.virtual_memory().available
                return int(ASSUMED_AVAILABLE_MEMORY_GB * (1024**3))


            def predicted_statevector_bytes(total_qubits: int) -> int:
                return BYTES_PER_AMPLITUDE * (2**total_qubits) * MEMORY_SAFETY_FACTOR


            def circuit_metrics(circuit: QuantumCircuit) -> dict:
                ops = circuit.count_ops()
                return {
                    "num_qubits_total": circuit.num_qubits,
                    "num_qubits_measured": circuit.num_clbits,
                    "depth": circuit.depth(),
                    "cx_count": int(ops.get("cx", 0)),
                    "total_gates": int(sum(ops.values())),
                }


            def _bv_statevector_top_worker(secret: str):
                # Self-contained worker (fork-safe): rebuild circuit, no notebook globals.
                from qiskit import QuantumCircuit as _QC
                from qiskit.quantum_info import Statevector as _SV
                import numpy as _np

                n = len(secret)
                qc = _QC(n + 1, n)
                qc.x(n)
                qc.h(range(n + 1))
                for i, bit in enumerate(secret):
                    if bit == "1":
                        qc.cx(i, n)
                qc.h(range(n))
                bare = qc.remove_final_measurements(inplace=False)
                probs = _SV.from_instruction(bare).probabilities(qargs=list(range(n)))
                top_idx = int(_np.argmax(probs))
                return format(top_idx, f"0{n}b"), float(probs[top_idx])


            def run_with_timeout(fn, args, timeout_s: float):
                # Run fn(*args) in a child process; terminate on timeout.
                ctx = mp.get_context("fork")
                queue = ctx.Queue()

                def _target():
                    try:
                        queue.put(("ok", fn(*args)))
                    except Exception as exc:  # noqa: BLE001
                        queue.put(("err", f"{type(exc).__name__}: {exc}"))

                proc = ctx.Process(target=_target)
                proc.start()
                proc.join(timeout_s)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(10)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(5)
                    return None, "timeout"
                if queue.empty():
                    return None, "no_result"
                status, payload = queue.get()
                if status == "ok":
                    return payload, None
                return None, payload


            sweep_rows = []
            last_elapsed_s = None
            consecutive_skips = 0

            print(
                f"Phase 2 config: ATTEMPT_TIMEOUT_S={ATTEMPT_TIMEOUT_S}s, "
                f"TIME_BUDGET_S={TIME_BUDGET_S}s, MAX_N_SECRET={MAX_N_SECRET}"
            )

            for n in range(MIN_N_SECRET, MAX_N_SECRET + 1):
                secret = secrets_for(n)[TIMING_PATTERN_NAME]
                qc = build_bv_circuit(secret)
                metrics = circuit_metrics(qc)
                total_qubits = metrics["num_qubits_total"]

                row = {
                    "n_secret": n,
                    "secret_pattern": TIMING_PATTERN_NAME,
                    **metrics,
                    "predicted_bytes": predicted_statevector_bytes(total_qubits),
                    "simulated": False,
                    "elapsed_s": None,
                    "status": None,
                    "top_bitstring": None,
                    "top_probability": None,
                    "matches_expected": None,
                }

                predicted_bytes = row["predicted_bytes"]
                available_bytes = available_memory_bytes()
                forced = n in FORCE_NS

                if not forced and predicted_bytes > available_bytes * MEMORY_USE_FRACTION:
                    row["status"] = "skipped_memory_prediction"
                    print(
                        f"n={n:3d} total_qubits={total_qubits:3d}  SKIPPED "
                        f"(predicted {predicted_bytes / 1024**3:.1f} GiB > "
                        f"{MEMORY_USE_FRACTION:.0%} of {available_bytes / 1024**3:.1f} GiB available)",
                        flush=True,
                    )
                    sweep_rows.append(row)
                    consecutive_skips += 1
                    if consecutive_skips >= 2:
                        print("Two consecutive memory-predicted skips -- stopping the sweep here.", flush=True)
                        break
                    continue

                if not forced and last_elapsed_s is not None and last_elapsed_s * 2 > TIME_BUDGET_S:
                    row["status"] = "skipped_time_prediction"
                    print(f"n={n:3d}  SKIPPED (projected time > {TIME_BUDGET_S}s budget)", flush=True)
                    sweep_rows.append(row)
                    break

                if forced and predicted_bytes > available_bytes * MEMORY_USE_FRACTION:
                    print(
                        f"n={n:3d} total_qubits={total_qubits:3d}  FORCED past the memory guard "
                        f"(predicted {predicted_bytes / 1024**3:.1f} GiB vs "
                        f"{available_bytes / 1024**3:.1f} GiB available) -- attempting anyway.",
                        flush=True,
                    )

                print(
                    f"n={n:3d} total_qubits={total_qubits:3d}  attempting statevector "
                    f"(hard timeout {ATTEMPT_TIMEOUT_S}s)...",
                    flush=True,
                )
                start = time.perf_counter()
                result, err = run_with_timeout(
                    _bv_statevector_top_worker, (secret,), ATTEMPT_TIMEOUT_S
                )
                elapsed = time.perf_counter() - start

                if err == "timeout":
                    row.update({"elapsed_s": elapsed, "status": "timeout"})
                    print(
                        f"n={n:3d}  TIMEOUT after {elapsed:.1f}s "
                        f"(limit {ATTEMPT_TIMEOUT_S}s) -- treating as wall.",
                        flush=True,
                    )
                    sweep_rows.append(row)
                    break

                if err is not None:
                    row.update({"elapsed_s": elapsed, "status": f"error: {err}"})
                    print(f"n={n:3d}  ERROR after {elapsed:.1f}s: {err}", flush=True)
                    sweep_rows.append(row)
                    break

                bitstring, prob = result
                expected = expected_outcome_from_secret(secret)
                row.update(
                    {
                        "simulated": True,
                        "elapsed_s": elapsed,
                        "status": "ok",
                        "top_bitstring": bitstring,
                        "top_probability": prob,
                        "matches_expected": bitstring == expected,
                    }
                )
                last_elapsed_s = elapsed
                consecutive_skips = 0
                print(
                    f"n={n:3d} total_qubits={total_qubits:3d}  OK  "
                    f"elapsed={elapsed:.3f}s  match={row['matches_expected']}  p={prob:.6f}",
                    flush=True,
                )
                sweep_rows.append(row)
                if elapsed > TIME_BUDGET_S:
                    print(f"Elapsed exceeded TIME_BUDGET_S={TIME_BUDGET_S}; stopping sweep.", flush=True)
                    break

            sweep_df = pd.DataFrame(sweep_rows)
            sim_ok = sweep_df[sweep_df["simulated"] == True]  # noqa: E712
            wall_n = MIN_N_SECRET - 1 if sim_ok.empty else int(sim_ok["n_secret"].max())
            # Prefer the hardware-confirmed wall from this campaign when present.
            if CONFIRMED_WALL_N_SECRET is not None:
                wall_n = int(CONFIRMED_WALL_N_SECRET)
                print(
                    f"Using CONFIRMED_WALL_N_SECRET={wall_n} "
                    f"(27 total qubits is the timeout wall).",
                    flush=True,
                )
            print(
                f"\nEmpirical simulation wall (last successful n_secret) = {wall_n} "
                f"(total qubits = {wall_n + 1})",
                flush=True,
            )
            sweep_df
            """
        )
    )

    cells.append(
        md(
            """
            ### Circuit metrics beyond the simulation wall (no simulation needed)

            Builds `QuantumCircuit` objects only (no statevector). Each `n` prints a
            heartbeat; the whole cell also has `METRICS_CELL_TIMEOUT_S` so a stuck
            Docker session fails closed instead of hanging forever.
            """
        )
    )

    cells.append(
        code(
            r"""
            MAX_N_METRICS_ONLY = 40  # secret bits; total qubits = n+1
            METRICS_CELL_TIMEOUT_S = 60  # fail the cell if metrics enumeration exceeds this

            metrics_only_rows = []
            metrics_t0 = time.perf_counter()
            print(
                f"Enumerating circuit metrics for n_secret={wall_n + 1}..{MAX_N_METRICS_ONLY} "
                f"(timeout {METRICS_CELL_TIMEOUT_S}s)...",
                flush=True,
            )

            for n in range(wall_n + 1, MAX_N_METRICS_ONLY + 1):
                if time.perf_counter() - metrics_t0 > METRICS_CELL_TIMEOUT_S:
                    print(
                        f"METRICS_CELL_TIMEOUT_S={METRICS_CELL_TIMEOUT_S}s exceeded at n={n}; "
                        f"stopping metrics-only enumeration.",
                        flush=True,
                    )
                    break
                t_n = time.perf_counter()
                secret = secrets_for(n)[TIMING_PATTERN_NAME]
                qc = build_bv_circuit(secret)
                metrics = circuit_metrics(qc)
                metrics_only_rows.append(
                    {
                        "n_secret": n,
                        "secret_pattern": TIMING_PATTERN_NAME,
                        **metrics,
                        "predicted_bytes": predicted_statevector_bytes(metrics["num_qubits_total"]),
                        "simulated": False,
                        "elapsed_s": None,
                        "status": "not_attempted (beyond wall)",
                        "top_bitstring": None,
                        "top_probability": None,
                        "matches_expected": None,
                    }
                )
                print(
                    f"  n={n:3d} total_qubits={metrics['num_qubits_total']:3d}  "
                    f"depth={metrics['depth']}  cx={metrics['cx_count']}  "
                    f"({time.perf_counter() - t_n:.3f}s)",
                    flush=True,
                )

            full_sweep_df = pd.concat(
                [sweep_df, pd.DataFrame(metrics_only_rows)], ignore_index=True
            )
            sweep_csv_path = RESULTS_DIR / f"bv_timing_sweep_{ts()}.csv"
            full_sweep_df.to_csv(sweep_csv_path, index=False)
            print(
                f"Empirical simulation wall at n_secret = {wall_n} "
                f"({wall_n + 1} total qubits, {int(2 ** (wall_n + 1)):,} amplitudes).",
                flush=True,
            )
            print(f"Saved to {sweep_csv_path}", flush=True)
            full_sweep_df.tail(12)
            """
        )
    )

    cells.append(
        code(
            r"""
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

            sim_rows = full_sweep_df[full_sweep_df["simulated"] == True]  # noqa: E712
            if not sim_rows.empty:
                axes[0].plot(sim_rows["n_secret"], sim_rows["elapsed_s"], "o-", color="tab:red", label="statevector time")
            timeout_rows = full_sweep_df[full_sweep_df["status"] == "timeout"]
            if not timeout_rows.empty:
                y = (
                    float(sim_rows["elapsed_s"].max()) * 1.5
                    if not sim_rows.empty
                    else float(ATTEMPT_TIMEOUT_S)
                )
                axes[0].scatter(
                    timeout_rows["n_secret"],
                    [y] * len(timeout_rows),
                    marker="x",
                    s=120,
                    color="black",
                    linewidths=2.5,
                    label=f"timeout ({ATTEMPT_TIMEOUT_S}s)",
                    zorder=5,
                )
            forced_ok = full_sweep_df[
                (full_sweep_df["simulated"] == True)  # noqa: E712
                & (full_sweep_df["n_secret"].isin(FORCE_NS))
            ]
            if not forced_ok.empty:
                axes[0].scatter(
                    forced_ok["n_secret"],
                    forced_ok["elapsed_s"],
                    marker="*",
                    s=180,
                    color="tab:orange",
                    label="forced past memory guard",
                    zorder=6,
                )
            axes[0].axvline(wall_n, color="gray", linestyle="--", label=f"wall at n_secret={wall_n}")
            axes[0].axvline(29, color="tab:purple", linestyle=":", label="target n_secret=29 (30 qubits)")
            axes[0].set_yscale("log")
            axes[0].set_xlabel("n_secret (total qubits = n+1)")
            axes[0].set_ylabel("Ideal statevector simulation time (s, log)")
            axes[0].set_title("Simulation-time growth (HF/TVD need this; DSR does not)")
            axes[0].legend(fontsize=8)

            axes[1].plot(full_sweep_df["n_secret"], full_sweep_df["cx_count"], "o-", color="tab:blue", label="CX count")
            axes[1].plot(full_sweep_df["n_secret"], full_sweep_df["depth"], "s-", color="tab:green", label="Depth")
            axes[1].axvline(wall_n, color="gray", linestyle="--")
            axes[1].axvline(29, color="tab:purple", linestyle=":")
            axes[1].set_xlabel("n_secret (total qubits = n+1)")
            axes[1].set_ylabel("Gate count")
            axes[1].set_title("Circuit metrics keep growing mildly past the wall")
            axes[1].legend()

            plt.tight_layout()
            wall_plot_path = PLOTS_DIR / f"bv_simulation_wall_{ts()}.png"
            plt.savefig(wall_plot_path, dpi=150)
            print(f"Saved to {wall_plot_path}")
            plt.show()
            """
        )
    )

    cells.append(
        md(
            """
            ## Phase 3 — Backend feasibility check (transpile candidates beyond the wall)

            Transpile candidates beyond the wall at opt-level 3 on the least-busy IBM backend.
            """
        )
    )

    cells.append(
        code(
            r"""
            from dotenv import load_dotenv
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            from qiskit_ibm_runtime import QiskitRuntimeService

            load_dotenv()

            service = QiskitRuntimeService(
                channel=os.getenv("IBM_QUANTUM_CHANNEL"),
                token=os.getenv("IBM_QUANTUM_TOKEN"),
                instance=os.getenv("IBM_QUANTUM_INSTANCE"),
            )

            CANDIDATE_OFFSETS = [1, 2, 3, 5]
            candidate_ns = sorted(set([wall_n + o for o in CANDIDATE_OFFSETS] + [29]))
            candidate_ns = [n for n in candidate_ns if n > wall_n]
            print(f"Candidate n_secret beyond wall (n={wall_n}): {candidate_ns}")

            backend = service.least_busy(operational=True, simulator=False)
            print(f"Least-busy backend: {backend.name} ({backend.num_qubits} qubits)")

            feasibility_rows = []
            for n in candidate_ns:
                total_qubits_needed = n + 1
                if total_qubits_needed > backend.num_qubits:
                    feasibility_rows.append(
                        {
                            "n_secret": n,
                            "num_qubits_total": total_qubits_needed,
                            "status": "exceeds_backend_qubits",
                        }
                    )
                    continue

                secret = secrets_for(n)[TIMING_PATTERN_NAME]
                qc = build_bv_circuit(secret, use_barriers=False)
                pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
                transpiled = pm.run(qc)
                ops = transpiled.count_ops()
                twoq = sum(v for k, v in ops.items() if k in ("cx", "cz", "ecr", "rzx"))
                feasibility_rows.append(
                    {
                        "n_secret": n,
                        "num_qubits_total": total_qubits_needed,
                        "logical_cx": int(qc.count_ops().get("cx", 0)),
                        "logical_depth": qc.depth(),
                        "transpiled_depth": transpiled.depth(),
                        "transpiled_2q_gates": int(twoq),
                        "transpiled_total_gates": int(sum(ops.values())),
                        "status": "ok",
                    }
                )
                print(f"n={n:3d}  transpiled depth={transpiled.depth():5d}  2q-gates={twoq:5d}")

            feasibility_df = pd.DataFrame(feasibility_rows)
            feasibility_csv_path = RESULTS_DIR / f"bv_transpile_feasibility_{ts()}.csv"
            feasibility_df.to_csv(feasibility_csv_path, index=False)
            print(f"Saved to {feasibility_csv_path}")
            feasibility_df
            """
        )
    )

    cells.append(
        md(
            """
            ### Pick the target `n_secret`

            Default is **29** (30 total qubits) when the backend can host it.
            """
        )
    )

    cells.append(
        code(
            r"""
            # EDIT ME after inspecting feasibility_df above.
            # Default: 29 secret bits = 30 total qubits (beyond the 27-qubit wall).
            # Phase 4 still submits ONE job with 3 pattern PUBs at this size.
            ok_ns = feasibility_df.loc[feasibility_df["status"] == "ok", "n_secret"].tolist()
            TARGET_N = 29 if 29 in ok_ns else (max(ok_ns) if ok_ns else candidate_ns[0])

            print(f"TARGET_N = {TARGET_N}  (total qubits = {TARGET_N + 1})")
            print(
                f"Phase 4 will submit 1 IBM job with 3 PUBs "
                f"(ONES/ALT/SINGLE) at this size when SUBMIT_JOB=True."
            )
            print(feasibility_df[feasibility_df["n_secret"] == TARGET_N])
            """
        )
    )


    cells.append(
        md(
            """
            ## Phase 4 — Submit a new IBM job (optional wait)

            Flip `SUBMIT_JOB = True` to submit **one** new IBM `SamplerV2` job
            with **three PUBs** (`ONES`, `ALT`, `SINGLE`) at `TARGET_N`
            (default 29 → 30 qubits, past the 27-qubit wall).

            After submit:
            - `WAIT_FOR_JOB = True` polls IBM like qward (`poll_interval=10`,
              `JOB_TIMEOUT_S=600`). If still not done, it **stops waiting** and
              keeps the `job_id` for Phase 5.
            - Every submission is appended to `results/bv_ibm_job_registry.json`
              so Phase 5 can load **all** past executions for multi-run DSR.

            To submit another run later: set `SUBMIT_JOB = True` again and re-run
            **only this cell** (do not clear the registry).
            """
        )
    )

    cells.append(
        code(
            r"""
            from qiskit_ibm_runtime import SamplerV2

            SUBMIT_JOB = False  # True = submit a NEW job this run
            WAIT_FOR_JOB = True  # poll after submit; stop when JOB_TIMEOUT_S hits
            JOB_TIMEOUT_S = 600  # same default as qward IBM experiments
            POLL_INTERVAL_S = 10
            SHOTS = 4096
            REGISTRY_PATH = RESULTS_DIR / "bv_ibm_job_registry.json"

            if "TARGET_N" not in dir():
                TARGET_N = 29
            if "wall_n" not in dir():
                wall_n = 25

            try:
                backend
                service
            except NameError:
                from dotenv import load_dotenv
                from qiskit_ibm_runtime import QiskitRuntimeService

                load_dotenv()
                service = QiskitRuntimeService(
                    channel=os.getenv("IBM_QUANTUM_CHANNEL"),
                    token=os.getenv("IBM_QUANTUM_TOKEN"),
                    instance=os.getenv("IBM_QUANTUM_INSTANCE"),
                )
                backend = service.least_busy(operational=True, simulator=False)
                print(f"Reconnected. Backend: {backend.name}")

            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


            def _load_registry() -> dict:
                if REGISTRY_PATH.exists():
                    return json.loads(REGISTRY_PATH.read_text())
                return {"jobs": []}


            def _save_registry(registry: dict) -> None:
                REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(REGISTRY_PATH, "w") as f:
                    json.dump(registry, f, indent=2)


            def wait_for_ibm_job(job, timeout_s: float, poll_interval_s: float) -> str:
                # Mirror qward executor/ibm_experiment_base polling.
                print(f"Waiting for job {job.job_id()} (timeout={timeout_s}s, poll={poll_interval_s}s)...")
                t0 = time.time()
                last_status = None
                while time.time() - t0 < timeout_s:
                    status = str(job.status())
                    elapsed = int(time.time() - t0)
                    if status != last_status:
                        print(f"  [{elapsed}s] status={status}", flush=True)
                        last_status = status
                    status_u = status.upper()
                    if any(s in status_u for s in ("DONE", "COMPLETED")):
                        print(f"  Job finished: {status}")
                        return status
                    if any(s in status_u for s in ("CANCELLED", "CANCELED", "ERROR", "FAILED")):
                        print(f"  Job ended unsuccessfully: {status}")
                        return status
                    time.sleep(poll_interval_s)
                status = str(job.status())
                print(
                    f"  STOPPED WAITING after {timeout_s}s (last status={status}). "
                    "Keep the job_id and re-run Phase 5 later.",
                    flush=True,
                )
                return status


            target_secrets = secrets_for(TARGET_N)
            pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

            prepared_circuits = {}
            expected_outputs = {}
            pattern_order = list(target_secrets.keys())

            for pattern_name in pattern_order:
                secret = target_secrets[pattern_name]
                qc = build_bv_circuit(secret, use_barriers=False)
                transpiled = pm.run(qc)
                prepared_circuits[pattern_name] = transpiled
                expected_outputs[pattern_name] = expected_outcome_from_secret(secret)
                print(
                    f"{pattern_name:8s} secret={secret}  expected={expected_outputs[pattern_name]}  "
                    f"transpiled_depth={transpiled.depth()}"
                )

            job_metadata = {
                "n_secret": TARGET_N,
                "num_qubits_total": TARGET_N + 1,
                "backend_name": backend.name,
                "shots": SHOTS,
                "pattern_order": pattern_order,
                "secrets": target_secrets,
                "expected_outputs": expected_outputs,
                "wall_n_from_phase2": int(wall_n),
                "submitted_at": None,
                "job_id": None,
                "final_status": None,
            }

            if SUBMIT_JOB:
                sampler = SamplerV2(mode=backend)
                job = sampler.run([prepared_circuits[p] for p in pattern_order], shots=SHOTS)
                job_metadata["submitted_at"] = datetime.now(timezone.utc).isoformat()
                job_metadata["job_id"] = job.job_id()
                print("=" * 60)
                print(f"SUBMITTED job_id = {job.job_id()}")
                print(f"backend         = {backend.name}")
                print("To submit ANOTHER run later: set SUBMIT_JOB=True and re-run this cell.")
                print("=" * 60)

                if WAIT_FOR_JOB:
                    job_metadata["final_status"] = wait_for_ibm_job(
                        job, JOB_TIMEOUT_S, POLL_INTERVAL_S
                    )
                else:
                    print("WAIT_FOR_JOB=False -- not polling; use Phase 5 with this job_id.")
            else:
                print(
                    "SUBMIT_JOB is False -- not submitting a new job. "
                    "Flip to True and re-run this cell to queue another IBM execution."
                )

            metadata_path = RESULTS_DIR / f"bv_ibm_job_n{TARGET_N}_{ts()}.json"
            with open(metadata_path, "w") as f:
                json.dump(job_metadata, f, indent=2)
            print(f"Saved job metadata to {metadata_path}")

            if job_metadata["job_id"]:
                latest_path = RESULTS_DIR / f"bv_ibm_job_n{TARGET_N}_latest.json"
                with open(latest_path, "w") as f:
                    json.dump(job_metadata, f, indent=2)
                print(f"Also saved resume pointer to {latest_path}")

                registry = _load_registry()
                entry = {
                    "job_id": job_metadata["job_id"],
                    "n_secret": TARGET_N,
                    "backend_name": backend.name,
                    "shots": SHOTS,
                    "submitted_at": job_metadata["submitted_at"],
                    "metadata_path": str(metadata_path),
                    "final_status": job_metadata.get("final_status"),
                }
                # de-dupe by job_id
                registry["jobs"] = [
                    j for j in registry.get("jobs", []) if j.get("job_id") != entry["job_id"]
                ]
                registry["jobs"].append(entry)
                _save_registry(registry)
                print(
                    f"Registry now has {len(registry['jobs'])} job(s) at {REGISTRY_PATH}"
                )
            """
        )
    )

    cells.append(
        md(
            """
            ## Phase 5 — Wait / retrieve jobs + multi-run DSR

            This cell:
            1. Collects job ids from `NEW_JOB_IDS` (paste), the in-session submit,
               **and** `results/bv_ibm_job_registry.json` / all `bv_ibm_job_n*.json`
               (so every past IBM execution is included for multi-run stats).
            2. Polls each unfinished job up to `JOB_TIMEOUT_S` (then stops waiting).
            3. Retrieves DONE jobs and merges with existing qward campaign runs.
            4. Downstream cells compute DSR across **all** loaded executions.

            To add another IBM run: go back to Phase 4, `SUBMIT_JOB=True`, re-run
            that cell, then re-run Phase 5.
            """
        )
    )

    cells.append(
        code(
            r"""
            # =============================================================================
            # EDIT ME:
            #   NEW_JOB_IDS = ["dxxx..."]          # paste ids to force-include
            #   AUTO_LOAD_SAVED_JOB_IDS = True     # load ALL registry / metadata jobs
            #   WAIT_FOR_JOBS = True               # poll unfinished jobs, then stop
            # =============================================================================
            NEW_JOB_IDS: list[str] = [
                # "paste-ibm-job-id-here",
            ]
            AUTO_LOAD_SAVED_JOB_IDS = True
            WAIT_FOR_JOBS = True
            JOB_TIMEOUT_S = 600
            POLL_INTERVAL_S = 10
            RESUME_METADATA_PATH: str | None = None  # optional explicit metadata JSON
            REGISTRY_PATH = RESULTS_DIR / "bv_ibm_job_registry.json"

            if "TARGET_N" not in dir():
                TARGET_N = 29
            if "wall_n" not in dir():
                wall_n = 25

            try:
                if job_metadata.get("job_id"):
                    NEW_JOB_IDS = list(dict.fromkeys(NEW_JOB_IDS + [job_metadata["job_id"]]))
            except NameError:
                job_metadata = None

            NEW_JOB_IDS = [jid.strip() for jid in NEW_JOB_IDS if jid and not jid.startswith("paste")]

            BARE_COUNTS_EXPECTED: dict[str, dict] = {}
            FILTER_N_SECRET: set[int] | None = None


            def _normalize_counts(counts: dict) -> dict[str, int]:
                return {str(k).replace(" ", ""): int(v) for k, v in counts.items()}


            def _infer_pattern(secret: str) -> str:
                mapping = secrets_for(len(secret))
                for name, s in mapping.items():
                    if s == secret:
                        return name
                return "CUSTOM"


            def _default_phase4_metadata(n_secret: int, wall: int) -> dict:
                secrets = secrets_for(n_secret)
                return {
                    "n_secret": n_secret,
                    "num_qubits_total": n_secret + 1,
                    "backend_name": None,
                    "shots": 4096,
                    "pattern_order": list(secrets.keys()),
                    "secrets": secrets,
                    "expected_outputs": {
                        name: expected_outcome_from_secret(secret)
                        for name, secret in secrets.items()
                    },
                    "wall_n_from_phase2": wall,
                    "job_id": None,
                }


            def discover_saved_job_ids() -> list[str]:
                ids: list[str] = []
                if REGISTRY_PATH.exists():
                    reg = json.loads(REGISTRY_PATH.read_text())
                    for entry in reg.get("jobs", []):
                        jid = entry.get("job_id")
                        if jid:
                            ids.append(jid)
                for path in sorted(RESULTS_DIR.glob("bv_ibm_job_n*.json")):
                    if path.name.endswith("_latest.json"):
                        continue
                    try:
                        meta = json.loads(path.read_text())
                    except Exception:
                        continue
                    jid = meta.get("job_id")
                    if jid:
                        ids.append(jid)
                # preserve order, unique
                return list(dict.fromkeys(ids))


            def _load_metadata_for_job(job_id: str) -> dict:
                if RESUME_METADATA_PATH:
                    path = Path(RESUME_METADATA_PATH)
                    if path.exists():
                        meta = json.loads(path.read_text())
                        print(f"Loaded metadata from RESUME_METADATA_PATH={path}")
                        return meta
                    print(f"RESUME_METADATA_PATH not found: {path}")

                if job_metadata and job_metadata.get("job_id") == job_id:
                    return job_metadata

                if REGISTRY_PATH.exists():
                    reg = json.loads(REGISTRY_PATH.read_text())
                    for entry in reg.get("jobs", []):
                        if entry.get("job_id") == job_id and entry.get("metadata_path"):
                            mpath = Path(entry["metadata_path"])
                            if mpath.exists():
                                print(f"Loaded metadata for {job_id} from registry -> {mpath}")
                                return json.loads(mpath.read_text())

                candidates = sorted(
                    RESULTS_DIR.glob("bv_ibm_job_n*.json"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for path in candidates:
                    try:
                        meta = json.loads(path.read_text())
                    except Exception:
                        continue
                    if meta.get("job_id") == job_id:
                        print(f"Loaded metadata for {job_id} from {path}")
                        return meta

                latest = RESULTS_DIR / f"bv_ibm_job_n{TARGET_N}_latest.json"
                if latest.exists():
                    meta = json.loads(latest.read_text())
                    print(f"Using pattern metadata from {latest} for job {job_id}")
                    meta = dict(meta)
                    meta["job_id"] = job_id
                    return meta

                print(
                    f"No saved metadata for {job_id}; reconstructing defaults "
                    f"for TARGET_N={TARGET_N} (ONES/ALT/SINGLE)."
                )
                meta = _default_phase4_metadata(TARGET_N, wall_n)
                meta["job_id"] = job_id
                return meta


            def ensure_ibm_service():
                global service
                try:
                    return service
                except NameError:
                    from dotenv import load_dotenv
                    from qiskit_ibm_runtime import QiskitRuntimeService

                    load_dotenv()
                    service = QiskitRuntimeService(
                        channel=os.getenv("IBM_QUANTUM_CHANNEL"),
                        token=os.getenv("IBM_QUANTUM_TOKEN"),
                        instance=os.getenv("IBM_QUANTUM_INSTANCE"),
                    )
                    print("Reconnected QiskitRuntimeService for job retrieval.")
                    return service


            def wait_for_ibm_job_id(svc, job_id: str, timeout_s: float, poll_interval_s: float) -> str:
                job = svc.job(job_id)
                print(f"Waiting for job {job_id} (timeout={timeout_s}s, poll={poll_interval_s}s)...")
                t0 = time.time()
                last_status = None
                while time.time() - t0 < timeout_s:
                    status = str(job.status())
                    elapsed = int(time.time() - t0)
                    if status != last_status:
                        print(f"  [{elapsed}s] status={status}", flush=True)
                        last_status = status
                    status_u = status.upper()
                    if any(s in status_u for s in ("DONE", "COMPLETED", "CANCELLED", "CANCELED", "ERROR", "FAILED")):
                        return status
                    time.sleep(poll_interval_s)
                status = str(job.status())
                print(
                    f"  STOPPED WAITING after {timeout_s}s (last status={status}).",
                    flush=True,
                )
                return status


            def load_qward_campaign_runs(raw_dir: Path) -> list[dict]:
                rows = []
                if not raw_dir.exists():
                    return rows
                for path in sorted(raw_dir.glob("*.json")):
                    try:
                        payload = json.loads(path.read_text())
                    except Exception as exc:  # noqa: BLE001
                        print(f"skip {path.name}: {exc}")
                        continue
                    if payload.get("algorithm") not in (None, "BERNSTEIN-VAZIRANI", "BV"):
                        if not path.name.startswith("BV"):
                            continue
                    for ir in payload.get("individual_results", []):
                        counts = ir.get("counts")
                        if not counts:
                            continue
                        secret = ir.get("secret_string") or payload.get("config", {}).get("secret_string")
                        expected = ir.get("expected_outcome") or (
                            expected_outcome_from_secret(secret) if secret else None
                        )
                        if not expected:
                            continue
                        if secret is None:
                            secret = expected[::-1]
                        n = len(expected)
                        rows.append(
                            {
                                "source": "qward_campaign",
                                "source_path": str(path),
                                "job_id": ir.get("job_id"),
                                "backend_name": ir.get("backend_name") or payload.get("backend_name"),
                                "optimization_level": ir.get("optimization_level"),
                                "n_secret": n,
                                "num_qubits_total": n + 1,
                                "pattern_name": _infer_pattern(secret),
                                "secret": secret,
                                "expected_output": expected,
                                "shots": int(ir.get("shots") or sum(counts.values())),
                                "counts": _normalize_counts(counts),
                                "role": "positive_control" if n < 29 else "wall_or_beyond",
                            }
                        )
                return rows


            def load_bare_counts_runs(results_dir: Path) -> list[dict]:
                rows = []
                for path in sorted(results_dir.glob("*Bernstein_Vazirani*.json")):
                    meta = BARE_COUNTS_EXPECTED.get(path.name)
                    if not meta:
                        print(f"skip bare counts {path.name}: add to BARE_COUNTS_EXPECTED to include")
                        continue
                    counts = _normalize_counts(json.loads(path.read_text()))
                    if "secret" in meta:
                        secret = meta["secret"]
                        expected = expected_outcome_from_secret(secret)
                    else:
                        expected = meta["expected"]
                        secret = expected[::-1]
                    n = len(expected)
                    rows.append(
                        {
                            "source": "benchmark_results_bare",
                            "source_path": str(path),
                            "job_id": None,
                            "backend_name": meta.get("backend_name", "unknown"),
                            "optimization_level": meta.get("optimization_level"),
                            "n_secret": n,
                            "num_qubits_total": n + 1,
                            "pattern_name": _infer_pattern(secret),
                            "secret": secret,
                            "expected_output": expected,
                            "shots": sum(counts.values()),
                            "counts": counts,
                            "role": "positive_control" if n < 29 else "wall_or_beyond",
                        }
                    )
                return rows


            def load_phase4_job_runs(job_ids: list[str]) -> list[dict]:
                rows = []
                if not job_ids:
                    print("No IBM job ids to retrieve.")
                    return rows

                svc = ensure_ibm_service()
                for job_id in job_ids:
                    metadata = _load_metadata_for_job(job_id)
                    if WAIT_FOR_JOBS:
                        status = wait_for_ibm_job_id(svc, job_id, JOB_TIMEOUT_S, POLL_INTERVAL_S)
                    else:
                        status = str(svc.job(job_id).status())
                        print(f"Job {job_id} status: {status}")

                    status_u = str(status).upper()
                    if not any(s in status_u for s in ("DONE", "COMPLETED")):
                        print(
                            f"  Skipping retrieve for {job_id} (status={status}). "
                            "Re-run Phase 5 later, or increase JOB_TIMEOUT_S."
                        )
                        continue

                    job = svc.job(job_id)
                    result = job.result()
                    pattern_order = metadata.get("pattern_order") or list(
                        secrets_for(metadata.get("n_secret", TARGET_N)).keys()
                    )
                    expected_outputs = metadata.get("expected_outputs") or {}
                    secrets = metadata.get("secrets") or {}
                    n = int(metadata.get("n_secret", TARGET_N))

                    if len(result) != len(pattern_order):
                        print(
                            f"  WARNING: job has {len(result)} pubs but metadata "
                            f"expects {len(pattern_order)} patterns -- pairing by index."
                        )

                    for idx, pattern_name in enumerate(pattern_order):
                        if idx >= len(result):
                            break
                        pub = result[idx]
                        bit_array = None
                        for attr in ("c", "meas", "cr"):
                            if hasattr(pub.data, attr):
                                bit_array = getattr(pub.data, attr)
                                break
                        if bit_array is None:
                            for attr in [a for a in dir(pub.data) if not a.startswith("_")]:
                                obj = getattr(pub.data, attr)
                                if hasattr(obj, "get_counts"):
                                    bit_array = obj
                                    break
                        if bit_array is None:
                            print(f"  could not extract counts for pub {idx} ({pattern_name})")
                            continue
                        counts = _normalize_counts(bit_array.get_counts())
                        secret = secrets.get(pattern_name) or secrets_for(n)[pattern_name]
                        expected = expected_outputs.get(pattern_name) or expected_outcome_from_secret(
                            secret
                        )
                        try:
                            backend_name = metadata.get("backend_name") or job.backend().name
                        except Exception:
                            backend_name = metadata.get("backend_name") or "unknown"
                        rows.append(
                            {
                                "source": "phase4_ibm_job",
                                "source_path": job_id,
                                "job_id": job_id,
                                "backend_name": backend_name,
                                "optimization_level": 3,
                                "n_secret": n,
                                "num_qubits_total": n + 1,
                                "pattern_name": pattern_name,
                                "secret": secret,
                                "expected_output": expected,
                                "shots": sum(counts.values()),
                                "counts": counts,
                                "role": "wall_or_beyond" if n >= 26 else "positive_control",
                            }
                        )
                        print(
                            f"  retrieved {pattern_name}: shots={sum(counts.values())} "
                            f"expected={expected} top={max(counts, key=counts.get)}",
                            flush=True,
                        )
                return rows


            all_job_ids = list(NEW_JOB_IDS)
            if AUTO_LOAD_SAVED_JOB_IDS:
                discovered = discover_saved_job_ids()
                all_job_ids = list(dict.fromkeys(all_job_ids + discovered))
                print(f"Discovered saved job ids: {discovered or '(none yet)'}")

            print(f"IBM job ids to wait/retrieve: {all_job_ids or '(none)'}")

            run_rows = []
            run_rows.extend(load_qward_campaign_runs(QWARD_BV_RAW_DIR))
            run_rows.extend(load_bare_counts_runs(RESULTS_DIR))
            run_rows.extend(load_phase4_job_runs(all_job_ids))

            if FILTER_N_SECRET is not None:
                run_rows = [r for r in run_rows if r["n_secret"] in FILTER_N_SECRET]

            # Expose for save cell
            NEW_JOB_IDS = all_job_ids

            print(
                f"Loaded {len(run_rows)} run×pattern rows. "
                f"Sources: {sorted({r['source'] for r in run_rows}) or 'none'}"
            )
            n_ibm = len({r['job_id'] for r in run_rows if r.get('source') == 'phase4_ibm_job'})
            print(f"Distinct IBM Phase-4 jobs in this analysis: {n_ibm}")
            """
        )
    )

    cells.append(
        code(
            r"""
            analysis_rows = []
            for run in run_rows:
                counts = run["counts"]
                expected = run["expected_output"]
                profile = dsr_profile_from_counts(counts, expected)
                total = sum(counts.values())
                top = max(counts, key=counts.get) if counts else None
                analysis_rows.append(
                    {
                        "source": run["source"],
                        "job_id": run["job_id"],
                        "backend_name": run["backend_name"],
                        "optimization_level": run["optimization_level"],
                        "n_secret": run["n_secret"],
                        "num_qubits_total": run["num_qubits_total"],
                        "pattern_name": run["pattern_name"],
                        "secret": run["secret"],
                        "expected_output": expected,
                        "top_outcome": top,
                        "top_matches_expected": top == expected,
                        "shots": total,
                        "role": run["role"],
                        "success_rate": profile.get("success_rate"),
                        "chance_corrected_success": profile.get("chance_corrected_success"),
                        "coarse_tvd_similarity": profile.get("coarse_tvd_similarity"),
                        "coarse_hellinger_fidelity": profile.get("coarse_hellinger_fidelity"),
                        "dsr_michelson": profile.get("dsr_michelson"),
                        "peak_mismatch": profile.get("peak_mismatch"),
                        "source_path": run["source_path"],
                    }
                )

            analysis_df = pd.DataFrame(analysis_rows)
            if analysis_df.empty:
                print(
                    "No runs loaded yet. After Phase 4, set NEW_JOB_IDS; "
                    "qward campaign files under QWARD_BV_RAW_DIR are auto-loaded when present."
                )
            else:
                display_cols = [
                    "role",
                    "n_secret",
                    "pattern_name",
                    "backend_name",
                    "success_rate",
                    "dsr_michelson",
                    "chance_corrected_success",
                    "top_matches_expected",
                    "source",
                ]
                print(analysis_df[display_cols].to_string(index=False))

            analysis_df
            """
        )
    )

    cells.append(
        code(
            r"""
            def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 42):
                values = np.asarray(values, dtype=float)
                if len(values) == 0:
                    return np.nan, np.nan, np.nan
                mean = float(np.mean(values))
                if len(values) < 3:
                    return mean, np.nan, np.nan
                rng = np.random.default_rng(seed)
                boot = np.array(
                    [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
                )
                lo, hi = np.percentile(boot, [2.5, 97.5])
                return mean, float(lo), float(hi)


            agg_rows = []
            if not analysis_df.empty:
                for (n, pattern), grp in analysis_df.groupby(["n_secret", "pattern_name"]):
                    for metric in (
                        "success_rate",
                        "dsr_michelson",
                        "chance_corrected_success",
                        "coarse_tvd_similarity",
                        "coarse_hellinger_fidelity",
                    ):
                        mean, lo, hi = bootstrap_mean_ci(grp[metric].dropna().values)
                        agg_rows.append(
                            {
                                "n_secret": n,
                                "pattern_name": pattern,
                                "metric": metric,
                                "n_runs": len(grp),
                                "mean": mean,
                                "std": float(grp[metric].std(ddof=1)) if len(grp) > 1 else 0.0,
                                "ci_lo": lo,
                                "ci_hi": hi,
                            }
                        )

            agg_df = pd.DataFrame(agg_rows)
            print("Per (n_secret, pattern, metric) aggregation:")
            agg_df.head(20) if not agg_df.empty else agg_df
            """
        )
    )

    cells.append(
        code(
            r"""
            stamp = ts()
            csv_path = RESULTS_DIR / f"bv_ibm_dsr_multirun_{stamp}.csv"
            json_path = RESULTS_DIR / f"bv_ibm_dsr_multirun_{stamp}.json"

            if not analysis_df.empty:
                analysis_df.to_csv(csv_path, index=False)
                payload = {
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "wall_n_from_phase2": int(wall_n) if "wall_n" in dir() else None,
                    "target_n": int(TARGET_N) if "TARGET_N" in dir() else None,
                    "new_job_ids": NEW_JOB_IDS,
                    "rows": analysis_rows,
                    "aggregation": agg_rows,
                    "counts_by_index": [r["counts"] for r in run_rows],
                }
                with open(json_path, "w") as f:
                    json.dump(payload, f, indent=2)
                print(f"Saved {csv_path}")
                print(f"Saved {json_path}")
            else:
                print("Nothing to save yet.")

            if not analysis_df.empty:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
                for ax, metric, title in zip(
                    axes,
                    ["dsr_michelson", "success_rate"],
                    ["DSR Michelson (multi-run)", "Success rate (multi-run)"],
                ):
                    for pattern_name, grp in analysis_df.groupby("pattern_name"):
                        ax.scatter(grp["n_secret"], grp[metric], label=pattern_name, alpha=0.75, s=40)
                    sub = agg_df[agg_df["metric"] == metric]
                    if not sub.empty:
                        for pattern_name, g2 in sub.groupby("pattern_name"):
                            g2 = g2.sort_values("n_secret")
                            ax.plot(g2["n_secret"], g2["mean"], "-", alpha=0.5)
                            if g2["ci_lo"].notna().any():
                                ax.fill_between(g2["n_secret"], g2["ci_lo"], g2["ci_hi"], alpha=0.15)
                    if "wall_n" in dir():
                        ax.axvline(wall_n, color="gray", linestyle="--", label=f"wall n={wall_n}")
                    ax.axvline(29, color="tab:purple", linestyle=":", label="target n=29")
                    ax.set_ylim(-0.05, 1.05)
                    ax.set_xlabel("n_secret")
                    ax.set_ylabel(metric)
                    ax.set_title(title)
                    ax.legend(fontsize=8)
                plt.tight_layout()
                plot_path = PLOTS_DIR / f"bv_ibm_dsr_multirun_{stamp}.png"
                plt.savefig(plot_path, dpi=150)
                print(f"Saved {plot_path}")
                plt.show()
            """
        )
    )

    cells.append(
        md(
            """
            ## Summary / next steps

            1. Phase 1 — analytic secret vs. statevector marginal.
            2. Phase 2 — wall locked at `n_secret=25` / 27 total qubits timeout.
            3. Phase 3 — pick `TARGET_N` (default 29 → 30 qubits).
            4. Phase 4 — `SUBMIT_JOB=True` to queue a **new** IBM job (1 job / 3 PUBs);
               optional wait (`JOB_TIMEOUT_S=600`); job id stored in registry.
               Re-run this cell anytime to submit another execution.
            5. Phase 5 — waits/retrieves **all** registry + pasted job ids, then DSR
               multi-run analysis (`results/bv_ibm_dsr_multirun_*.csv`).
            """
        )
    )

    return cells


def main() -> None:
    shutil.copy2(PLAN_SRC, PLAN_DST)
    print(f"Copied plan -> {PLAN_DST}")

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": build_cells(),
    }
    NB_PATH.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {NB_PATH} with {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
