"""
Enrich QPU/simulator result JSONs with the DSR evaluation profile.

Histogram-free: computes the four-component DSR profile (success rate,
chance-corrected success, coarse TVD similarity, coarse Hellinger fidelity)
from measurement counts and the analytically known expected-outcome set
``E`` alone -- it never needs to simulate the full ideal distribution.  If a
file was *previously* enriched by ``enrich_hellinger.py`` and already carries
a top-level ``ideal_probs`` field, this script opportunistically reads the
exact expected-outcome weights from it (still no new simulation is run here);
otherwise it defaults to uniform weights over ``E``.

Adds to each ``individual_results`` entry:
  - ``success_rate``, ``chance_baseline``, ``chance_corrected_success``
  - ``coarse_tvd``, ``coarse_tvd_similarity``
  - ``coarse_hellinger_distance``, ``coarse_hellinger_fidelity``
  - refreshes the optional Michelson "peak-contrast" layer
    (``dsr_michelson``, ``peak_mismatch``) from the same code path so all DSR
    fields share one canonical implementation
  - updates ``batch_summary`` with the corresponding means

Idempotent: skips per-result entries that already carry
``chance_corrected_success`` (use ``--force`` to recompute anyway).

Usage:
  PYTHONPATH=. uv run python qward/examples/papers/enrich_dsr_profile.py --dataset all
  PYTHONPATH=. uv run python qward/examples/papers/enrich_dsr_profile.py --dataset grover-aws
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from enrich_hellinger import (
    DATASETS,
    _expected_outcomes_grover,
    _expected_outcomes_qft,
    _expected_outcomes_bv,
)

from qward.metrics.differential_success_rate import DSRProfiler
from qward.schemas.dsr_profile_schema import DSRProfileSchema

# ---------------------------------------------------------------------------
# Histogram-free helpers
# ---------------------------------------------------------------------------


def _marginalize_to_length(counts: Dict[str, int], target_len: int) -> Dict[str, int]:
    """Truncate wider bitstrings (e.g. ancilla qubits included by AWS) down
    to ``target_len`` bits, without needing any simulated reference.

    Mirrors ``enrich_hellinger._marginalize_counts`` but is keyed off the
    known measured-qubit count directly, so it needs no ``ideal_probs``.
    """
    if not counts:
        return counts
    count_len = len(next(iter(counts)))
    if count_len <= target_len:
        return counts
    drop = count_len - target_len
    marginal: Counter = Counter()
    for bitstr, cnt in counts.items():
        marginal[bitstr[: count_len - drop]] += cnt
    return dict(marginal)


def _expected_weights_from_payload(
    payload: Dict, expected_outcomes: List[str]
) -> Optional[Dict[str, float]]:
    """Opportunistically read exact expected-outcome weights from a
    previously-computed ``ideal_probs`` field (from ``enrich_hellinger.py``).

    Returns None (uniform default) when ``ideal_probs`` is absent -- this
    script never triggers a new statevector simulation itself.
    """
    ideal_probs = payload.get("ideal_probs")
    if not ideal_probs:
        return None

    weights = {outcome: ideal_probs.get(outcome, 0.0) for outcome in expected_outcomes}
    total = sum(weights.values())
    if total <= 0:
        return None
    return {outcome: weight / total for outcome, weight in weights.items()}


def _check_qft_period_divides_evenly(config: Dict) -> None:
    """QFT period-detection peaks are exactly uniform (1/period) only when
    ``period`` evenly divides ``2**num_qubits``. Warn loudly otherwise, since
    the analytic peak weights would then be non-uniform (Shor's sinc^2
    envelope) and the uniform default used below would bias the coarse
    metrics.
    """
    if config.get("test_mode") != "period_detection":
        return
    period = config.get("period")
    num_qubits = config.get("num_qubits")
    if not period or not num_qubits:
        return
    if (2**num_qubits) % period != 0:
        print(
            f"    WARN: period={period} does not evenly divide 2**{num_qubits}; "
            "QFT peaks are analytically non-uniform here and the uniform "
            "expected_weights default used by DSRProfiler will bias the "
            "coarse TVD/Hellinger components. Supply exact weights instead."
        )


# ---------------------------------------------------------------------------
# Enrichment logic
# ---------------------------------------------------------------------------


def _needs_profile_backfill(result: Dict) -> bool:
    return "chance_corrected_success" not in result


def _backfill_profile(
    result: Dict,
    expected_outcomes: List[str],
    expected_weights: Optional[Dict[str, float]],
) -> Optional[DSRProfileSchema]:
    """Compute and write the DSR profile fields into ``result`` in place."""
    counts = result.get("counts")
    if not counts or not expected_outcomes:
        return None

    num_measured_qubits = len(next(iter(expected_outcomes)))
    counts_aligned = _marginalize_to_length(counts, num_measured_qubits)

    weights = None
    if expected_weights is not None:
        weights = {o: expected_weights[o] for o in expected_outcomes if o in expected_weights}
        if set(weights) != set(expected_outcomes) or abs(sum(weights.values()) - 1.0) > 1e-6:
            weights = None  # fall back to uniform rather than pass a partial/invalid map

    try:
        profiler = DSRProfiler(
            counts_aligned,
            expected_outcomes,
            num_measured_qubits=num_measured_qubits,
            expected_weights=weights,
            include_michelson=True,
        )
        profile = profiler.profile()
    except ValueError as e:
        print(f"    WARN: could not compute DSR profile: {e}")
        return None

    result["expected_outcomes"] = list(expected_outcomes)
    result["success_rate"] = profile.success_rate
    result["chance_baseline"] = profile.chance_baseline
    result["chance_corrected_success"] = profile.chance_corrected_success
    result["coarse_tvd"] = profile.coarse_tvd
    result["coarse_tvd_similarity"] = profile.coarse_tvd_similarity
    result["coarse_hellinger_distance"] = profile.coarse_hellinger_distance
    result["coarse_hellinger_fidelity"] = profile.coarse_hellinger_fidelity
    result["dsr_michelson"] = profile.dsr_michelson
    result["peak_mismatch"] = profile.peak_mismatch
    return profile


def _enrich_file(
    path: Path, algorithm: str, dry_run: bool = False, force: bool = False
) -> Optional[Dict]:
    """Enrich a single JSON file's ``individual_results`` with the DSR
    profile. Returns stats dict, or None if skipped."""
    payload = json.loads(path.read_text())
    config = payload.get("config", {})
    results = payload.get("individual_results", [])

    if not results:
        return None

    if not force:
        first_with_counts = next((r for r in results if r.get("counts")), None)
        if first_with_counts and not _needs_profile_backfill(first_with_counts):
            return None

    if algorithm == "GROVER":
        expected_outcomes = _expected_outcomes_grover(config)
    elif algorithm == "QFT":
        expected_outcomes = _expected_outcomes_qft(config)
        _check_qft_period_divides_evenly(config)
    elif algorithm in ("BERNSTEIN-VAZIRANI", "BV"):
        expected_outcomes = _expected_outcomes_bv(config)
    else:
        print(f"  SKIP {path.name}: unknown algorithm {algorithm}")
        return None

    if not expected_outcomes:
        print(f"  SKIP {path.name}: could not derive expected_outcomes")
        return None

    expected_weights = _expected_weights_from_payload(payload, expected_outcomes)

    success_rates, chance_corrected, coarse_tvd_sims, coarse_hf_vals = [], [], [], []
    num_backfilled = 0

    for result in results:
        counts = result.get("counts")
        if not counts:
            continue

        result_expected = result.get("expected_outcomes")
        if not result_expected and result.get("expected_outcome"):
            result_expected = [str(result["expected_outcome"]).replace(" ", "")]
        if not result_expected:
            result_expected = expected_outcomes
        profile = _backfill_profile(result, result_expected, expected_weights)
        if profile is None:
            continue

        num_backfilled += 1
        success_rates.append(profile.success_rate)
        chance_corrected.append(profile.chance_corrected_success)
        coarse_tvd_sims.append(profile.coarse_tvd_similarity)
        coarse_hf_vals.append(profile.coarse_hellinger_fidelity)

    batch = payload.get("batch_summary", {})
    if success_rates:
        batch["mean_success_rate_profile"] = sum(success_rates) / len(success_rates)
        batch["mean_chance_corrected_success"] = sum(chance_corrected) / len(chance_corrected)
        batch["mean_coarse_tvd_similarity"] = sum(coarse_tvd_sims) / len(coarse_tvd_sims)
        batch["mean_coarse_hellinger_fidelity"] = sum(coarse_hf_vals) / len(coarse_hf_vals)
    payload["batch_summary"] = batch

    if not dry_run:
        path.write_text(json.dumps(payload, indent=2) + "\n")

    return {
        "file": path.name,
        "num_results": num_backfilled,
        "mean_success_rate": sum(success_rates) / len(success_rates) if success_rates else 0.0,
        "mean_chance_corrected_success": (
            sum(chance_corrected) / len(chance_corrected) if chance_corrected else 0.0
        ),
        "mean_coarse_tvd_similarity": (
            sum(coarse_tvd_sims) / len(coarse_tvd_sims) if coarse_tvd_sims else 0.0
        ),
        "mean_coarse_hellinger_fidelity": (
            sum(coarse_hf_vals) / len(coarse_hf_vals) if coarse_hf_vals else 0.0
        ),
    }


def _run_dataset(
    name: str,
    directory: Path,
    algorithm: str,
    dry_run: bool = False,
    force: bool = False,
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
    all_success = []
    all_chance_corrected = []
    all_tvd_sim = []
    all_hf = []
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
            all_success.append(stats["mean_success_rate"])
            all_chance_corrected.append(stats["mean_chance_corrected_success"])
            all_tvd_sim.append(stats["mean_coarse_tvd_similarity"])
            all_hf.append(stats["mean_coarse_hellinger_fidelity"])
            print(
                f"  [{i}/{len(json_files)}] OK    {path.name}  "
                f"success={stats['mean_success_rate']:.4f}  "
                f"chance_corrected={stats['mean_chance_corrected_success']:.4f}"
                f"  ({stats['num_results']} results)"
            )

    elapsed = time.time() - t0
    print(f"\n  Summary: {enriched} enriched, {skipped} skipped, {errors} errors")
    print(f"  Time: {elapsed:.1f}s")
    if all_success:
        print(
            f"  Success rate:             mean={sum(all_success)/len(all_success):.4f}  "
            f"min={min(all_success):.4f}  max={max(all_success):.4f}"
        )
        print(
            f"  Chance-corrected success: mean={sum(all_chance_corrected)/len(all_chance_corrected):.4f}  "
            f"min={min(all_chance_corrected):.4f}  max={max(all_chance_corrected):.4f}"
        )
        print(
            f"  Coarse TVD similarity:    mean={sum(all_tvd_sim)/len(all_tvd_sim):.4f}  "
            f"min={min(all_tvd_sim):.4f}  max={max(all_tvd_sim):.4f}"
        )
        print(
            f"  Coarse Hellinger fidelity:mean={sum(all_hf)/len(all_hf):.4f}  "
            f"min={min(all_hf):.4f}  max={max(all_hf):.4f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Enrich QPU/simulator result JSONs with the histogram-free DSR profile."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["grover-aws", "grover-ibm", "qft-aws", "qft-ibm", "bv-ibm", "all"],
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
        "bv-ibm": "BERNSTEIN-VAZIRANI",
    }

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    print(f"Enriching {len(targets)} dataset(s) with the DSR profile...")
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
