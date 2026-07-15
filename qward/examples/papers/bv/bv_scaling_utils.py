"""Reusable utilities for the Bernstein-Vazirani DSR scaling study.

This module underpins ``bv_scaling_dsr.py``. It demonstrates the central claim
of the paper's scalability argument:

  * **DSR** (success rate, chance-corrected success, Michelson contrast, coarse
    HF/TVD) needs only the measured counts and the *analytic* expected outcome
    (``secret[::-1]``) -> O(1) classical cost, computable at any qubit count.
  * **Hellinger fidelity / TVD fidelity** need the ideal reference distribution
    over all ``2**n`` outcomes, produced by a statevector simulation. Past the
    simulation wall (~30 qubits on a laptop) that reference cannot be built, so
    ``qiskit.quantum_info.hellinger_fidelity`` cannot be evaluated even though
    the hardware counts are in hand.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Dict, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Official Qiskit counts-distribution fidelity/distance (we use the standard
# library implementation, not a hand-rolled one).
from qiskit.quantum_info import hellinger_distance, hellinger_fidelity  # noqa: F401

from qward.algorithms import BernsteinVazirani
from qward.metrics.differential_success_rate import (
    compute_dsr_michelson,
    compute_dsr_profile,
)

try:
    import psutil

    HAVE_PSUTIL = True
except ImportError:  # pragma: no cover
    HAVE_PSUTIL = False

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BYTES_PER_AMPLITUDE = 16  # complex128
MEMORY_SAFETY_FACTOR = 4
ASSUMED_AVAILABLE_MEMORY_GB = 8
DEFAULT_TIMEOUT_S = 180.0

# The three secret patterns swept throughout the BV study.
PATTERN_NAMES = ("ONES", "ALT", "SINGLE")


# --------------------------------------------------------------------------- #
# Bernstein-Vazirani circuit + analytic expected outcome
# --------------------------------------------------------------------------- #
def expected_outcome_from_secret(secret_string: str) -> str:
    """Analytic BV answer: Qiskit little-endian measurement of the query register.

    No simulation required -- this is what lets DSR scale past the wall.
    """
    return secret_string[::-1]


def secrets_for(n: int) -> Dict[str, str]:
    """Fixed secret patterns to sweep (ONES / ALT / SINGLE), matching bv_configs."""
    return {
        "ONES": "1" * n,
        "ALT": "".join("1" if i % 2 == 0 else "0" for i in range(n)),
        "SINGLE": "0" * (n - 1) + "1",
    }


def build_bv_circuit(secret_string: str, use_barriers: bool = True) -> QuantumCircuit:
    """Build the BV circuit via the qward algorithm implementation."""
    return BernsteinVazirani(secret_string=secret_string, use_barriers=use_barriers).circuit


def circuit_without_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    return circuit.remove_final_measurements(inplace=False)


def circuit_metrics(circuit: QuantumCircuit) -> Dict[str, int]:
    ops = circuit.count_ops()
    return {
        "num_qubits_total": circuit.num_qubits,
        "num_qubits_measured": circuit.num_clbits,
        "depth": circuit.depth(),
        "cx_count": int(ops.get("cx", 0)),
        "total_gates": int(sum(ops.values())),
    }


# --------------------------------------------------------------------------- #
# DSR (counts-only, no simulation)
# --------------------------------------------------------------------------- #
def dsr_profile_from_counts(counts: Dict[str, int], expected: str) -> Dict:
    """Return the flat DSR profile dict for a single expected outcome (K=1)."""
    counts = {str(k).replace(" ", ""): int(v) for k, v in counts.items()}
    expected = expected.replace(" ", "")
    michelson = float(compute_dsr_michelson(counts, {expected}))
    profile = compute_dsr_profile(counts, {expected}, include_michelson=True)
    flat = profile.to_flat_dict()
    flat.setdefault("dsr_michelson", michelson)
    if flat.get("success_rate") is None:
        total = sum(counts.values())
        flat["success_rate"] = counts.get(expected, 0) / total if total else 0.0
    return flat


# --------------------------------------------------------------------------- #
# Memory / statevector timeout machinery (used by the wall sweep)
# --------------------------------------------------------------------------- #
def available_memory_bytes() -> int:
    if HAVE_PSUTIL:
        return psutil.virtual_memory().available
    return int(ASSUMED_AVAILABLE_MEMORY_GB * (1024**3))


def predicted_statevector_bytes(total_qubits: int) -> int:
    return BYTES_PER_AMPLITUDE * (2**total_qubits) * MEMORY_SAFETY_FACTOR


def _bv_statevector_top_worker(secret: str) -> Tuple[str, float]:
    """Fork-safe worker: rebuild circuit, return the top measured outcome."""
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


def _ideal_measured_distribution_worker(secret: str) -> Dict[str, float]:
    """Fork-safe worker: full ideal probability distribution over the measured register.

    This is the ideal reference (``dist_q``) that ``hellinger_fidelity`` needs.
    Building it requires a statevector, so it blows up as ``2**(n+1)`` amplitudes.
    """
    from qiskit import QuantumCircuit as _QC
    from qiskit.quantum_info import Statevector as _SV

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
    return {format(idx, f"0{n}b"): float(p) for idx, p in enumerate(probs) if p > 1e-12}


def run_with_timeout(fn, args, timeout_s: float):
    """Run ``fn(*args)`` in a forked child process; terminate on timeout.

    Returns ``(result, None)`` on success, ``(None, "timeout")`` if killed, or
    ``(None, "err_message")`` if the child raised.
    """
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


def top_measured_outcome_from_statevector(
    circuit: QuantumCircuit, n_measured: int
) -> Tuple[str, float]:
    """Marginalize over the ancilla; return (bitstring, probability) on the query register."""
    bare = circuit_without_measurements(circuit)
    sv = Statevector.from_instruction(bare)
    probs = sv.probabilities(qargs=list(range(n_measured)))
    top_idx = int(np.argmax(probs))
    return format(top_idx, f"0{n_measured}b"), float(probs[top_idx])


# --------------------------------------------------------------------------- #
# HF / TVD PROOF: full-distribution fidelities need the ideal statevector
# --------------------------------------------------------------------------- #
class SimulationInfeasibleError(RuntimeError):
    """Raised when the ideal reference distribution cannot be produced.

    ``qiskit.quantum_info.hellinger_fidelity`` needs a *second* (ideal reference)
    distribution, and building it requires a statevector simulation that is
    infeasible past the wall.
    """

    def __init__(self, n_secret: int, predicted_bytes: int, kind: str = "memory", detail: str = ""):
        self.n_secret = n_secret
        self.predicted_bytes = predicted_bytes
        self.kind = kind
        gib = predicted_bytes / 1024**3
        super().__init__(
            f"ideal reference distribution for n_secret={n_secret} "
            f"({n_secret + 1} qubits) is infeasible [{kind}]: needs ~{gib:.1f} GiB and "
            f"2**{n_secret + 1} = {2 ** (n_secret + 1):,} amplitudes. {detail}".strip()
        )


def total_variation_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    """TVD = 0.5 * sum_i |p_i - q_i| over two probability distributions."""
    keys = set(p) | set(q)
    return float(0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys))


def ideal_reference_counts(
    secret: str,
    shots: int = 4096,
    allow_attempt: bool = False,
    timeout_s: float = 60.0,
    memory_use_fraction: float = 0.6,
    memory_budget_bytes: Optional[int] = None,
) -> Dict[str, int]:
    """Produce the ideal reference counts distribution needed for HF/TVD.

    Raises ``SimulationInfeasibleError`` when the reference cannot be built.
    Set ``allow_attempt=True`` to actually try the simulation under the hard
    timeout even when the memory pre-check says it is hopeless.
    """
    n = len(secret)
    total_qubits = n + 1
    predicted = predicted_statevector_bytes(total_qubits)
    if memory_budget_bytes is None:
        memory_budget_bytes = int(available_memory_bytes() * memory_use_fraction)

    if predicted > memory_budget_bytes and not allow_attempt:
        raise SimulationInfeasibleError(
            n,
            predicted,
            "memory",
            f"(> {memory_budget_bytes / 1024**3:.1f} GiB budget). "
            f"Set allow_attempt=True to try anyway under a {timeout_s:.0f}s timeout.",
        )

    ideal_probs, err = run_with_timeout(_ideal_measured_distribution_worker, (secret,), timeout_s)
    if err == "timeout":
        raise SimulationInfeasibleError(
            n, predicted, "timeout", f"(statevector exceeded {timeout_s:.0f}s hard timeout)."
        )
    if err is not None:
        raise SimulationInfeasibleError(n, predicted, "error", f"(statevector failed: {err}).")

    counts = {k: int(round(v * shots)) for k, v in ideal_probs.items()}
    return {k: c for k, c in counts.items() if c > 0}


def full_distribution_fidelities(
    counts: Dict[str, int],
    secret: str,
    shots: Optional[int] = None,
    timeout_s: float = 60.0,
    allow_attempt: bool = False,
    memory_use_fraction: float = 0.6,
    memory_budget_bytes: Optional[int] = None,
) -> Dict:
    """Compute HF / TVD-fidelity of ``counts`` vs the ideal reference, if possible.

    Uses the official ``qiskit.quantum_info.hellinger_fidelity``. Returns
    ``status="computed"`` below the wall and ``status="infeasible_*"`` (HF/TVD
    = ``None``) once the ideal statevector can no longer be produced.
    """
    n = len(secret)
    total_qubits = n + 1
    predicted = predicted_statevector_bytes(total_qubits)
    observed = {str(k).replace(" ", ""): int(v) for k, v in counts.items()}
    total = sum(observed.values())
    if shots is None:
        shots = total or 4096

    base = {
        "n_secret": n,
        "num_qubits_total": total_qubits,
        "predicted_statevector_bytes": predicted,
        "predicted_statevector_gib": predicted / 1024**3,
        "num_ideal_amplitudes": 2**total_qubits,
        "hellinger_fidelity": None,
        "tvd": None,
        "tvd_fidelity": None,
        "fidelity_backend": "qiskit.quantum_info.hellinger_fidelity",
    }

    try:
        ideal = ideal_reference_counts(
            secret,
            shots=shots,
            allow_attempt=allow_attempt,
            timeout_s=timeout_s,
            memory_use_fraction=memory_use_fraction,
            memory_budget_bytes=memory_budget_bytes,
        )
    except SimulationInfeasibleError as exc:
        base["status"] = f"infeasible_{exc.kind}"
        base["reason"] = str(exc)
        return base

    hf = float(hellinger_fidelity(observed, ideal))
    obs_p = {k: v / total for k, v in observed.items()} if total else {}
    ideal_total = sum(ideal.values())
    ideal_p = {k: v / ideal_total for k, v in ideal.items()} if ideal_total else {}
    tvd = total_variation_distance(obs_p, ideal_p)
    base.update(
        {
            "status": "computed",
            "reason": "ideal reference built; qiskit hellinger_fidelity applied.",
            "hellinger_fidelity": hf,
            "tvd": tvd,
            "tvd_fidelity": 1.0 - tvd,
            "ideal_support": len(ideal),
        }
    )
    return base


def synthetic_noisy_counts(expected: str, shots: int, flip_p: float, seed: int = 7) -> Dict[str, int]:
    """Mock a noisy hardware run below the wall (per-bit flips of the ideal outcome).

    Used only for the *computable* control points in the HF-vs-DSR figure; the
    beyond-wall points use the real IBM hardware counts.
    """
    rng = np.random.default_rng(seed)
    counts: Dict[str, int] = {}
    for _ in range(shots):
        bits = [("1" if b == "0" else "0") if rng.random() < flip_p else b for b in expected]
        bs = "".join(bits)
        counts[bs] = counts.get(bs, 0) + 1
    return counts


def run_wall_sweep(
    min_n: int = 2,
    max_n: int = 34,
    attempt_timeout_s: float = DEFAULT_TIMEOUT_S,
    time_budget_s: float = 600.0,
    memory_use_fraction: float = 0.6,
    pattern: str = "ALT",
    verbose: bool = True,
):
    """Statevector-time growth sweep that auto-stops at the simulation wall.

    Returns ``(rows, wall_n)`` where ``rows`` is a list of per-``n`` dicts and
    ``wall_n`` is the largest ``n_secret`` whose statevector simulated OK.
    """
    rows = []
    last_elapsed_s = None
    consecutive_skips = 0

    for n in range(min_n, max_n + 1):
        secret = secrets_for(n)[pattern]
        qc = build_bv_circuit(secret, use_barriers=False)
        metrics = circuit_metrics(qc)
        total_qubits = metrics["num_qubits_total"]
        predicted = predicted_statevector_bytes(total_qubits)
        available = available_memory_bytes()

        row = {
            "n_secret": n,
            "secret_pattern": pattern,
            **metrics,
            "predicted_bytes": predicted,
            "predicted_gib": predicted / 1024**3,
            "simulated": False,
            "elapsed_s": None,
            "status": None,
            "top_bitstring": None,
            "top_probability": None,
            "matches_expected": None,
        }

        if predicted > available * memory_use_fraction:
            row["status"] = "skipped_memory_prediction"
            if verbose:
                print(
                    f"n={n:3d} qubits={total_qubits:3d}  SKIP mem "
                    f"({predicted / 1024**3:.1f} GiB > {memory_use_fraction:.0%} of "
                    f"{available / 1024**3:.1f} GiB)",
                    flush=True,
                )
            rows.append(row)
            consecutive_skips += 1
            if consecutive_skips >= 2:
                break
            continue

        if last_elapsed_s is not None and last_elapsed_s * 2 > time_budget_s:
            row["status"] = "skipped_time_prediction"
            rows.append(row)
            break

        if verbose:
            print(
                f"n={n:3d} qubits={total_qubits:3d}  attempting statevector "
                f"(timeout {attempt_timeout_s:.0f}s)...",
                flush=True,
            )
        start = time.perf_counter()
        result, err = run_with_timeout(_bv_statevector_top_worker, (secret,), attempt_timeout_s)
        elapsed = time.perf_counter() - start

        if err == "timeout":
            row.update({"elapsed_s": elapsed, "status": "timeout"})
            if verbose:
                print(f"n={n:3d}  TIMEOUT after {elapsed:.1f}s -- treating as wall.", flush=True)
            rows.append(row)
            break
        if err is not None:
            row.update({"elapsed_s": elapsed, "status": f"error: {err}"})
            rows.append(row)
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
        if verbose:
            print(
                f"n={n:3d} qubits={total_qubits:3d}  OK  elapsed={elapsed:.3f}s  "
                f"match={row['matches_expected']}  p={prob:.6f}",
                flush=True,
            )
        rows.append(row)
        if elapsed > time_budget_s:
            break

    simulated = [r for r in rows if r["simulated"]]
    wall_n = (min_n - 1) if not simulated else max(r["n_secret"] for r in simulated)
    return rows, wall_n
