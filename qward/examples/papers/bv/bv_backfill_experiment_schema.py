#!/usr/bin/env python3
"""
Backfill BV IBM QPU JSONs to match the IBMExperimentBase / Grover schema.

Fills experiment fields that the notebook ingest omitted, without re-running
IBM jobs:

  * batch_id, backend_type
  * evaluate_result fields (advantage_ratio, quantum_advantage, thresholds, …)
  * DSR variant scores (michelson already from enrich_dsr_profile; also ratio /
    log-ratio / normalized-margin)
  * full batch_summary (mean/std/min/max, analysis block, mean DSR fields)
  * explicit markers that ideal_probs / HF / TVDF are infeasible past the wall

Does NOT invent:
  * backend_calibration / gate_error_characterization (need live IBM API)
  * ideal_probs / hellinger_* / tvd_* at n≈29 (statevector wall)
  * transpiled_depth (needs backend pass manager)

Usage:
  PYTHONPATH=. uv run python qward/examples/papers/bv/bv_backfill_experiment_schema.py
  PYTHONPATH=. uv run python qward/examples/papers/bv/bv_backfill_experiment_schema.py --force
"""

from __future__ import annotations

import argparse
import json
import statistics
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from qward.metrics.differential_success_rate import (
    compute_dsr_with_flags,
    compute_dsr_ratio,
    compute_dsr_log_ratio,
    compute_dsr_normalized_margin,
)

RAW_DIR = Path(__file__).resolve().parent / "data" / "qpu" / "raw"


def _normalize_counts(counts: Dict[str, int]) -> Dict[str, int]:
    return {str(k).replace(" ", ""): int(v) for k, v in counts.items()}


def _ci_95(values: List[float]) -> tuple[float, float]:
    if len(values) < 2:
        m = values[0] if values else 0.0
        return m, m
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    # Normal approx; matches the lightweight analysis block used elsewhere.
    half = 1.96 * std / (len(values) ** 0.5)
    return mean - half, mean + half


def _backfill_individual(ir: Dict[str, Any], config: Dict[str, Any]) -> None:
    counts = _normalize_counts(ir.get("counts") or {})
    ir["counts"] = counts
    shots = int(ir.get("shots") or sum(counts.values()) or 0)
    ir["shots"] = shots

    expected = (
        (ir.get("expected_outcomes") or [None])[0]
        or ir.get("expected_outcome")
        or config.get("expected_outcome")
    )
    if expected:
        expected = str(expected).replace(" ", "")
        ir["expected_outcome"] = expected
        ir["expected_outcomes"] = [expected]

    secret = ir.get("secret_string") or config.get("secret_string")
    if secret:
        ir["secret_string"] = secret

    random_chance = float(
        ir.get("random_chance")
        or config.get("classical_random_prob")
        or (1.0 / (2 ** len(expected)) if expected else 0.0)
    )
    ir["random_chance"] = random_chance
    ir["theoretical_success"] = float(config.get("theoretical_success", 1.0))

    if expected and shots > 0:
        success_count = int(counts.get(expected, 0))
        success_rate = success_count / shots
    else:
        success_count = int(ir.get("success_count") or 0)
        success_rate = float(ir.get("success_rate") or 0.0)

    ir["success_count"] = success_count
    ir["success_rate"] = success_rate
    advantage = success_rate / random_chance if random_chance > 0 else 0.0
    ir["advantage_ratio"] = advantage
    ir["quantum_advantage"] = advantage > 2.0
    ir["threshold_30"] = success_rate >= 0.30
    ir["threshold_50"] = success_rate >= 0.50
    ir["threshold_70"] = success_rate >= 0.70
    ir["threshold_90"] = success_rate >= 0.90

    if expected and counts:
        dsr_m, peak_mm = compute_dsr_with_flags(counts, [expected])
        ir["dsr_michelson"] = dsr_m
        ir["peak_mismatch"] = peak_mm
        ir["dsr_ratio"] = compute_dsr_ratio(counts, [expected])
        ir["dsr_log_ratio"] = compute_dsr_log_ratio(counts, [expected])
        ir["dsr_normalized_margin"] = compute_dsr_normalized_margin(counts, [expected])

    # Full-distribution metrics are intentionally absent past the wall.
    ir.setdefault("hellinger_fidelity", None)
    ir.setdefault("hellinger_distance", None)
    ir.setdefault("tvd", None)
    ir.setdefault("tvd_fidelity", None)

    ir.setdefault("backend_type", "qpu")
    if ir.get("transpiled_depth") is None:
        ir["transpiled_depth"] = None  # unknown without backend re-transpile


def _rebuild_batch_summary(payload: Dict[str, Any]) -> None:
    config = payload.get("config", {})
    results = payload.get("individual_results", [])
    rates = [float(r["success_rate"]) for r in results if r.get("success_rate") is not None]
    shots = [int(r.get("shots") or 0) for r in results]
    advantages = [float(r["advantage_ratio"]) for r in results if r.get("advantage_ratio") is not None]

    batch: Dict[str, Any] = dict(payload.get("batch_summary") or {})
    batch.update(
        {
            "config_id": payload.get("config_id") or config.get("config_id"),
            "noise_model": "IBM-QPU",
            "backend_name": payload.get("backend_name"),
            "num_runs": len(results),
            "shots_per_run": int(statistics.mean(shots)) if shots else None,
            "backend_type": "qpu",
        }
    )
    if rates:
        mean_r = statistics.mean(rates)
        std_r = statistics.stdev(rates) if len(rates) > 1 else 0.0
        ci_lo, ci_hi = _ci_95(rates)
        batch.update(
            {
                "mean_success_rate": mean_r,
                "std_success_rate": std_r,
                "min_success_rate": min(rates),
                "max_success_rate": max(rates),
                "median_success_rate": statistics.median(rates),
                "analysis": {
                    "config_id": batch["config_id"],
                    "noise_model": "IBM-QPU",
                    "num_runs": len(rates),
                    "mean": mean_r,
                    "std": std_r,
                    "median": statistics.median(rates),
                    "min": min(rates),
                    "max": max(rates),
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                },
            }
        )
    if advantages:
        mean_adv = statistics.mean(advantages)
        batch["mean_quantum_advantage_ratio"] = mean_adv
        batch["quantum_advantage_demonstrated"] = mean_adv > 2.0

    # Preserve / refresh profile means if present on individuals.
    def _mean_field(key: str) -> Optional[float]:
        vals = [float(r[key]) for r in results if r.get(key) is not None]
        return statistics.mean(vals) if vals else None

    for src, dst in (
        ("success_rate", "mean_success_rate_profile"),
        ("chance_corrected_success", "mean_chance_corrected_success"),
        ("coarse_tvd_similarity", "mean_coarse_tvd_similarity"),
        ("coarse_hellinger_fidelity", "mean_coarse_hellinger_fidelity"),
        ("dsr_michelson", "mean_dsr_michelson"),
        ("dsr_ratio", "mean_dsr_ratio"),
        ("dsr_log_ratio", "mean_dsr_log_ratio"),
        ("dsr_normalized_margin", "mean_dsr_normalized_margin"),
    ):
        val = _mean_field(src)
        if val is not None:
            batch[dst] = val

    # Explicit: full-distribution means are not available past the wall.
    batch["mean_hellinger_fidelity"] = None
    batch["mean_hellinger_distance"] = None
    batch["mean_tvd"] = None
    batch["mean_tvd_fidelity"] = None
    payload["batch_summary"] = batch


def backfill_file(path: Path, force: bool = False) -> str:
    payload = json.loads(path.read_text())
    if payload.get("algorithm") not in ("BERNSTEIN-VAZIRANI", "BV"):
        return "skipped_not_bv"

    # Idempotency: skip if already looks like a full experiment file.
    if not force and payload.get("batch_id") and payload.get("individual_results"):
        ir0 = payload["individual_results"][0]
        if ir0.get("advantage_ratio") is not None and ir0.get("qward_metrics"):
            return "skipped"

    config = payload.get("config", {})
    if not payload.get("batch_id"):
        # Stable-ish id from the first job_id when present.
        job_id = None
        if payload.get("individual_results"):
            job_id = payload["individual_results"][0].get("job_id")
        payload["batch_id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, job_id or path.name))

    for ir in payload.get("individual_results", []):
        _backfill_individual(ir, config)

    # Mark ideal simulation as infeasible for large BV (paper claim).
    n = int(config.get("num_qubits") or 0)
    if n >= 26:
        payload["ideal_probs"] = None
        payload["ideal_probs_status"] = "infeasible_beyond_simulation_wall"
        payload["ideal_probs_reason"] = (
            f"Statevector ideal histogram for {n}+1 total qubits exceeds laptop "
            "memory/time; HF/TVDF cannot be computed. DSR profile remains valid."
        )
    payload.setdefault("backend_calibration", None)
    payload.setdefault("gate_error_characterization", None)

    _rebuild_batch_summary(payload)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return "backfilled"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill BV IBM JSON experiment schema.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    files = sorted(RAW_DIR.glob("BV*_IBM_*.json"))
    print(f"Found {len(files)} BV IBM JSON file(s) in {RAW_DIR}")
    stats = {"backfilled": 0, "skipped": 0, "skipped_not_bv": 0}
    for path in files:
        status = backfill_file(path, force=args.force)
        key = "backfilled" if status == "backfilled" else status
        stats[key] = stats.get(key, 0) + 1
        print(f"  {path.name}: {status}")
    print(f"\nDone. {stats}")


if __name__ == "__main__":
    main()
