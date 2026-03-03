"""
Enrich QPU result JSONs with ideal distributions and fidelity metrics.

Adds to each JSON file:
  - top-level ``ideal_probs``: statevector-derived ideal distribution
  - per-result ``hellinger_fidelity``, ``hellinger_distance``
  - per-result ``tvd`` and ``tvd_fidelity`` (Total Variation Distance)
  - backfills DSR fields for IBM results that are missing them
  - updates ``batch_summary`` with mean Hellinger/TVD/DSR values

Idempotent: re-running skips files that already have all metrics.

Usage:
  PYTHONPATH=. uv run python qward/examples/papers/enrich_hellinger.py --dataset all
  PYTHONPATH=. uv run python qward/examples/papers/enrich_hellinger.py --dataset grover-aws
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from qiskit.quantum_info import Statevector, hellinger_fidelity, hellinger_distance

from qward.algorithms.grover import Grover
from qward.algorithms.qft import QFTCircuitGenerator
from qward.metrics.differential_success_rate import (
    compute_dsr_with_flags,
    compute_dsr_ratio,
    compute_dsr_log_ratio,
    compute_dsr_normalized_margin,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PAPERS_DIR = Path(__file__).resolve().parent

DATASETS = {
    "grover-aws": PAPERS_DIR / "grover" / "data" / "qpu" / "aws",
    "grover-ibm": PAPERS_DIR / "grover" / "data" / "qpu" / "raw",
    "qft-aws": PAPERS_DIR / "qft" / "data" / "qpu" / "aws",
    "qft-ibm": PAPERS_DIR / "qft" / "data" / "qpu" / "raw",
}

# ---------------------------------------------------------------------------
# Circuit reconstruction
# ---------------------------------------------------------------------------


def _build_grover_circuit(config: Dict):
    """Reconstruct a Grover circuit from config.

    Returns:
        (circuit, measured_qubits) — measured_qubits is None when all qubits
        are measured (the common case).
    """
    marked_states = config["marked_states"]
    grover = Grover(marked_states)
    return grover.circuit, None


def _build_qft_circuit(config: Dict):
    """Reconstruct a QFT circuit from config.

    Returns:
        (circuit, measured_qubits) — for period_detection the counting
        register qubits [1..num_qubits] are returned so the ancilla (qubit 0)
        is traced out when computing ideal probabilities.
    """
    num_qubits = config["num_qubits"]
    test_mode = config.get("test_mode", "roundtrip")

    if test_mode == "roundtrip":
        input_state = config.get("input_state", "0" * num_qubits)
        gen = QFTCircuitGenerator(
            num_qubits, test_mode="roundtrip", input_state=input_state
        )
        return gen.circuit, None
    elif test_mode == "period_detection":
        period = config["period"]
        gen = QFTCircuitGenerator(
            num_qubits, test_mode="period_detection", period=period
        )
        # Period detection has an ancilla at qubit 0; measurements are on
        # qubits 1..num_qubits mapped to classical bits 0..num_qubits-1.
        measured_qubits = list(range(1, num_qubits + 1))
        return gen.circuit, measured_qubits
    else:
        raise ValueError(f"Unknown QFT test_mode: {test_mode}")


def _compute_ideal_probs(circuit, measured_qubits=None) -> Dict[str, float]:
    """Compute ideal probability distribution via Statevector simulation.

    Strips measurements, simulates, and returns probabilities > 1e-10.
    When *measured_qubits* is given, computes marginal probabilities over
    only those qubits (tracing out ancillas).
    """
    circuit_no_meas = circuit.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(circuit_no_meas)
    if measured_qubits is not None:
        probs = sv.probabilities_dict(qargs=measured_qubits, decimals=10)
    else:
        probs = sv.probabilities_dict(decimals=10)
    # Filter to keep JSON compact
    return {k: v for k, v in probs.items() if v > 1e-10}


# ---------------------------------------------------------------------------
# Expected outcomes derivation (for DSR backfill)
# ---------------------------------------------------------------------------


def _expected_outcomes_grover(config: Dict) -> List[str]:
    """Derive expected outcomes for Grover from config."""
    return list(config["marked_states"])


def _expected_outcomes_qft(config: Dict) -> List[str]:
    """Derive expected outcomes for QFT from config."""
    test_mode = config.get("test_mode", "roundtrip")
    num_qubits = config["num_qubits"]

    if test_mode == "roundtrip":
        return [config.get("input_state", "0" * num_qubits)]

    if test_mode == "period_detection":
        period = config["period"]
        n = num_qubits
        size = 2 ** n
        outcomes = []
        seen = set()
        for k in range(period):
            peak = (k * size // period) % size
            bitstring = format(peak, f"0{n}b")
            if bitstring not in seen:
                seen.add(bitstring)
                outcomes.append(bitstring)
        return outcomes

    return []


# ---------------------------------------------------------------------------
# Enrichment logic
# ---------------------------------------------------------------------------


def _marginalize_counts(
    counts: Dict[str, int], ideal_probs: Dict[str, float]
) -> Dict[str, int]:
    """Marginalize counts to match ideal_probs bitstring length.

    When QPU counts include ancilla qubits (e.g. AWS Braket measures all
    qubits for QFT period_detection), we must drop the extra bits to align
    with the ideal distribution.  The ancilla is qubit 0 (rightmost bit in
    Qiskit's little-endian convention).
    """
    if not counts or not ideal_probs:
        return counts

    count_len = len(next(iter(counts)))
    ideal_len = len(next(iter(ideal_probs)))

    if count_len == ideal_len:
        return counts

    if count_len <= ideal_len:
        return counts

    # Number of ancilla bits to drop from the right
    drop = count_len - ideal_len
    marginal: Counter = Counter()
    for bitstr, cnt in counts.items():
        marginal[bitstr[:count_len - drop]] += cnt
    return dict(marginal)


def _normalize_counts(counts: Dict[str, int]) -> Dict[str, float]:
    """Convert raw counts to a probability distribution."""
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def _total_variation_distance(
    p: Dict[str, float], q: Dict[str, float]
) -> float:
    """Compute the Total Variation Distance between two distributions.

    TVD = 0.5 * sum_i |p_i - q_i|   over the union of all outcomes.

    Returns a value in [0, 1] where 0 = identical, 1 = no overlap.
    """
    all_keys = set(p.keys()) | set(q.keys())
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in all_keys)


def _needs_dsr_backfill(result: Dict) -> bool:
    """Check if a result is missing DSR fields."""
    return "dsr_michelson" not in result


def _backfill_dsr(result: Dict, expected_outcomes: List[str]) -> None:
    """Add DSR fields to a result that is missing them."""
    counts = result.get("counts", {})
    if not counts or not expected_outcomes:
        return

    dsr_michelson, peak_mismatch = compute_dsr_with_flags(counts, expected_outcomes)
    dsr_ratio = compute_dsr_ratio(counts, expected_outcomes)
    dsr_log_ratio = compute_dsr_log_ratio(counts, expected_outcomes)
    dsr_norm_margin = compute_dsr_normalized_margin(counts, expected_outcomes)

    result["dsr_michelson"] = dsr_michelson
    result["dsr_ratio"] = dsr_ratio
    result["dsr_log_ratio"] = dsr_log_ratio
    result["dsr_normalized_margin"] = dsr_norm_margin
    result["peak_mismatch"] = peak_mismatch
    result["expected_outcomes"] = expected_outcomes


def _enrich_file(
    path: Path, algorithm: str, dry_run: bool = False, force: bool = False
) -> Optional[Dict]:
    """Enrich a single JSON file. Returns stats dict or None if skipped."""
    payload = json.loads(path.read_text())
    config = payload.get("config", {})
    results = payload.get("individual_results", [])

    if not results:
        return None

    # Idempotency: skip if already fully enriched (unless forced)
    if not force:
        first_with_counts = next(
            (r for r in results if r.get("counts")), None
        )
        if (
            first_with_counts
            and "hellinger_fidelity" in first_with_counts
            and "tvd" in first_with_counts
        ):
            return None

    # Reconstruct circuit and compute ideal probs
    try:
        if algorithm == "GROVER":
            circuit, measured_qubits = _build_grover_circuit(config)
        elif algorithm == "QFT":
            circuit, measured_qubits = _build_qft_circuit(config)
        else:
            print(f"  SKIP {path.name}: unknown algorithm {algorithm}")
            return None
    except Exception as e:
        print(f"  ERROR {path.name}: circuit reconstruction failed: {e}")
        return None

    try:
        ideal_probs = _compute_ideal_probs(circuit, measured_qubits)
    except Exception as e:
        print(f"  ERROR {path.name}: statevector simulation failed: {e}")
        return None

    # Store ideal_probs at top level
    payload["ideal_probs"] = ideal_probs

    # Derive expected outcomes for DSR backfill
    if algorithm == "GROVER":
        expected_outcomes = _expected_outcomes_grover(config)
    else:
        expected_outcomes = _expected_outcomes_qft(config)

    # Enrich each individual result
    hellinger_fids = []
    hellinger_dists = []
    tvd_vals = []
    tvd_fid_vals = []
    dsr_michelson_vals = []
    dsr_ratio_vals = []
    dsr_log_ratio_vals = []
    dsr_norm_margin_vals = []
    dsr_backfilled = False

    for result in results:
        counts = result.get("counts")
        if not counts:
            continue

        # Marginalize counts if they include ancilla bits (AWS period_detection)
        counts_aligned = _marginalize_counts(counts, ideal_probs)
        counts_dist = _normalize_counts(counts_aligned)

        # Compute Hellinger fidelity and distance
        try:
            hf = hellinger_fidelity(ideal_probs, counts_dist)
            hd = hellinger_distance(ideal_probs, counts_dist)
        except Exception as e:
            print(f"  WARN {path.name}: hellinger computation failed: {e}")
            hf = None
            hd = None

        if hf is not None:
            result["hellinger_fidelity"] = hf
            result["hellinger_distance"] = hd
            hellinger_fids.append(hf)
            hellinger_dists.append(hd)

        # Compute Total Variation Distance
        tvd = _total_variation_distance(ideal_probs, counts_dist)
        tvd_fid = 1.0 - tvd
        result["tvd"] = tvd
        result["tvd_fidelity"] = tvd_fid
        tvd_vals.append(tvd)
        tvd_fid_vals.append(tvd_fid)

        # Backfill DSR if missing
        if _needs_dsr_backfill(result):
            _backfill_dsr(result, expected_outcomes)
            dsr_backfilled = True

        # Collect DSR values for batch summary
        if "dsr_michelson" in result:
            dsr_michelson_vals.append(result["dsr_michelson"])
            dsr_ratio_vals.append(result["dsr_ratio"])
            dsr_log_ratio_vals.append(result["dsr_log_ratio"])
            dsr_norm_margin_vals.append(result["dsr_normalized_margin"])

    # Update batch_summary
    batch = payload.get("batch_summary", {})
    if hellinger_fids:
        batch["mean_hellinger_fidelity"] = sum(hellinger_fids) / len(hellinger_fids)
        batch["mean_hellinger_distance"] = sum(hellinger_dists) / len(hellinger_dists)

    if tvd_vals:
        batch["mean_tvd"] = sum(tvd_vals) / len(tvd_vals)
        batch["mean_tvd_fidelity"] = sum(tvd_fid_vals) / len(tvd_fid_vals)

    if dsr_backfilled and dsr_michelson_vals:
        batch["mean_dsr_michelson"] = sum(dsr_michelson_vals) / len(dsr_michelson_vals)
        batch["mean_dsr_ratio"] = sum(dsr_ratio_vals) / len(dsr_ratio_vals)
        batch["mean_dsr_log_ratio"] = sum(dsr_log_ratio_vals) / len(dsr_log_ratio_vals)
        batch["mean_dsr_normalized_margin"] = (
            sum(dsr_norm_margin_vals) / len(dsr_norm_margin_vals)
        )

    payload["batch_summary"] = batch

    # Write back
    if not dry_run:
        path.write_text(json.dumps(payload, indent=2) + "\n")

    num_enriched = len(tvd_vals)
    return {
        "file": path.name,
        "num_results": num_enriched,
        "mean_hf": sum(hellinger_fids) / len(hellinger_fids) if hellinger_fids else 0,
        "mean_hd": sum(hellinger_dists) / len(hellinger_dists) if hellinger_dists else 0,
        "mean_tvd": sum(tvd_vals) / len(tvd_vals) if tvd_vals else 0,
        "mean_tvd_fid": sum(tvd_fid_vals) / len(tvd_fid_vals) if tvd_fid_vals else 0,
        "dsr_backfilled": dsr_backfilled,
    }


def _run_dataset(
    name: str, directory: Path, algorithm: str,
    dry_run: bool = False, force: bool = False,
):
    """Process all JSON files in a dataset directory."""
    if not directory.exists():
        print(f"\n--- {name}: directory not found ({directory}) ---")
        return

    json_files = sorted(directory.glob("*.json"))
    if not json_files:
        print(f"\n--- {name}: no JSON files found ---")
        return

    print(f"\n{'='*60}")
    print(f"  Dataset: {name} ({len(json_files)} files)")
    print(f"  Path: {directory}")
    print(f"{'='*60}")

    enriched = 0
    skipped = 0
    errors = 0
    all_hf = []
    all_hd = []
    all_tvd = []
    all_tvd_fid = []
    t0 = time.time()

    for i, path in enumerate(json_files, 1):
        stats = _enrich_file(path, algorithm, dry_run=dry_run, force=force)
        if stats is None:
            skipped += 1
            print(f"  [{i}/{len(json_files)}] SKIP  {path.name}")
        elif stats.get("num_results", 0) == 0:
            errors += 1
            print(f"  [{i}/{len(json_files)}] WARN  {path.name} (no results enriched)")
        else:
            enriched += 1
            all_hf.append(stats["mean_hf"])
            all_hd.append(stats["mean_hd"])
            all_tvd.append(stats["mean_tvd"])
            all_tvd_fid.append(stats["mean_tvd_fid"])
            dsr_tag = " +DSR" if stats["dsr_backfilled"] else ""
            print(
                f"  [{i}/{len(json_files)}] OK    {path.name}  "
                f"HF={stats['mean_hf']:.4f}  TVD={stats['mean_tvd']:.4f}"
                f"  ({stats['num_results']} results){dsr_tag}"
            )

    elapsed = time.time() - t0
    print(f"\n  Summary: {enriched} enriched, {skipped} skipped, {errors} errors")
    print(f"  Time: {elapsed:.1f}s")
    if all_hf:
        print(
            f"  Hellinger Fidelity:  mean={sum(all_hf)/len(all_hf):.4f}  "
            f"min={min(all_hf):.4f}  max={max(all_hf):.4f}"
        )
        print(
            f"  Hellinger Distance:  mean={sum(all_hd)/len(all_hd):.4f}  "
            f"min={min(all_hd):.4f}  max={max(all_hd):.4f}"
        )
        print(
            f"  TVD:                 mean={sum(all_tvd)/len(all_tvd):.4f}  "
            f"min={min(all_tvd):.4f}  max={max(all_tvd):.4f}"
        )
        print(
            f"  TVD Fidelity (1-TVD):mean={sum(all_tvd_fid)/len(all_tvd_fid):.4f}  "
            f"min={min(all_tvd_fid):.4f}  max={max(all_tvd_fid):.4f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Enrich QPU result JSONs with ideal probs and Hellinger fidelity."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["grover-aws", "grover-ibm", "qft-aws", "qft-ibm", "all"],
        help="Which dataset to process (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute but don't write changes to files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-enrich files even if already enriched",
    )
    args = parser.parse_args()

    dataset_algorithm = {
        "grover-aws": "GROVER",
        "grover-ibm": "GROVER",
        "qft-aws": "QFT",
        "qft-ibm": "QFT",
    }

    if args.dataset == "all":
        targets = list(DATASETS.keys())
    else:
        targets = [args.dataset]

    print(f"Enriching {len(targets)} dataset(s)...")
    if args.dry_run:
        print("(DRY RUN - no files will be modified)")
    if args.force:
        print("(FORCE - re-enriching all files)")

    for name in targets:
        _run_dataset(
            name,
            DATASETS[name],
            dataset_algorithm[name],
            dry_run=args.dry_run,
            force=args.force,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
