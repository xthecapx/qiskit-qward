"""
Build a unified DSR_result.csv from JSON files (Grover/QFT) and CSV files
(Teleportation).

JSON is the single source of truth for Grover and QFT experiments — all
pre-computed metrics (DSR variants, Hellinger fidelity, TVD) are read
directly from the enriched individual_results.

Teleportation data lives in CSV files (different schema, no JSON equivalent).
DSR is computed on the fly; Hellinger/TVD are computed from the delta ideal
distribution (the single expected state), so HF = TVD-F = p(expected_state).

This script replaces running ``differential_success_rate_experiment.py``
manually and supersedes the old two-CSV setup (DSR_result.csv +
DSR_result_aws.csv) with a single unified CSV.

Usage:
    PYTHONPATH=. uv run python qward/examples/papers/build_csv_from_json.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from differential_success_rate_experiment import (
    _compute_teleportation_dsr_rows,
    _collect_teleportation_paths,
    _expected_from_config,
)

# ── Column order ──────────────────────────────────────────────────────────────

FIELDNAMES = [
    "algorithm",
    "execution_type",
    "backend_name",
    "backend_type",
    "noise_model",
    "config_id",
    "result_id",
    "optimization_level",
    "num_qubits",
    "circuit_depth",
    "transpiled_depth",
    "total_gates",
    "shots",
    "expected_outcomes",
    "histogram",
    "peak_mismatch",
    "dsr_michelson",
    "dsr_ratio",
    "dsr_log_ratio",
    "dsr_normalized_margin",
    "hellinger_fidelity",
    "hellinger_distance",
    "tvd",
    "tvd_fidelity",
    "source_file",
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def _fmt(value) -> str:
    """Format a numeric value to 6 decimal places, or '' if missing."""
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""


# ── JSON extraction (Grover / QFT) ───────────────────────────────────────────


def _extract_rows_from_json(json_paths: List[Path]) -> List[Dict[str, str]]:
    """Extract rows from enriched JSON files.

    Reads pre-computed DSR, Hellinger, and TVD metrics directly from
    each ``individual_results`` entry.  Only entries with non-empty
    ``counts`` are included.
    """
    rows: List[Dict[str, str]] = []

    for path in json_paths:
        payload = json.loads(path.read_text())
        config = payload.get("config", {})
        algorithm = payload.get("algorithm", "unknown")
        execution_type = payload.get("execution_type", "")
        root_backend = payload.get("backend_name", payload.get("device_name", ""))
        root_config_id = payload.get("config_id", "")

        for result in payload.get("individual_results", []):
            counts = result.get("counts")
            if not counts:
                continue

            # expected_outcomes — prefer result-level, fall back to config
            expected = result.get("expected_outcomes")
            if not expected:
                expected = _expected_from_config(config, result)
            if not expected:
                continue

            shots = result.get("shots")
            if shots is None:
                shots = sum(counts.values())

            row = {
                "algorithm": str(algorithm),
                "execution_type": str(execution_type),
                "backend_name": str(result.get("backend_name", root_backend)),
                "backend_type": str(result.get("backend_type", "")),
                "noise_model": str(result.get("noise_model", payload.get("noise_id", ""))),
                "config_id": str(result.get("config_id", root_config_id)),
                "result_id": str(result.get("experiment_id", result.get("job_id", ""))),
                "optimization_level": str(result.get("optimization_level", "")),
                "num_qubits": str(result.get("num_qubits", config.get("num_qubits", ""))),
                "circuit_depth": str(result.get("circuit_depth", "")),
                "transpiled_depth": str(result.get("transpiled_depth", "")),
                "total_gates": str(result.get("total_gates", "")),
                "shots": str(shots),
                "expected_outcomes": ",".join(expected),
                "histogram": json.dumps(counts, sort_keys=True),
                "peak_mismatch": str(bool(result.get("peak_mismatch", False))),
                "dsr_michelson": _fmt(result.get("dsr_michelson")),
                "dsr_ratio": _fmt(result.get("dsr_ratio")),
                "dsr_log_ratio": _fmt(result.get("dsr_log_ratio")),
                "dsr_normalized_margin": _fmt(result.get("dsr_normalized_margin")),
                "hellinger_fidelity": _fmt(result.get("hellinger_fidelity")),
                "hellinger_distance": _fmt(result.get("hellinger_distance")),
                "tvd": _fmt(result.get("tvd")),
                "tvd_fidelity": _fmt(result.get("tvd_fidelity")),
                "source_file": path.name,
            }
            rows.append(row)

    return rows


# ── Path collection ───────────────────────────────────────────────────────────

PAPERS = Path(__file__).resolve().parent

DATASETS = {
    "grover-ibm": ("GROVER", PAPERS / "grover" / "data" / "qpu" / "raw"),
    "grover-aws": ("GROVER", PAPERS / "grover" / "data" / "qpu" / "aws"),
    "qft-ibm": ("QFT", PAPERS / "qft" / "data" / "qpu" / "raw"),
    "qft-aws": ("QFT", PAPERS / "qft" / "data" / "qpu" / "aws"),
}


def _collect_json_paths() -> List[Path]:
    """Collect all JSON result files from the four known directories."""
    paths: List[Path] = []
    for _label, (_algo, directory) in DATASETS.items():
        if directory.exists():
            paths.extend(sorted(directory.glob("*.json")))
    return paths


# ── Teleportation (CSV) ──────────────────────────────────────────────────────


def _compute_teleportation_fidelity(row: Dict[str, str]) -> None:
    """Compute Hellinger fidelity, Hellinger distance, TVD, and TVD fidelity
    for a teleportation row.

    For teleportation the ideal output is a single deterministic state
    (e.g. '000'), so the ideal distribution is a delta function. Under a delta
    ideal: HF = TVD-F = p(expected_state).

    Wide histograms (keys longer than num_qubits, caused by full-circuit
    measurement on AWS) are marginalized to the last *num_qubits* bits before
    computing the metrics.
    """
    import math

    histogram_str = row.get("histogram", "")
    expected_str = row.get("expected_outcomes", "")
    nq_str = row.get("num_qubits", "")
    if not histogram_str or not expected_str or not nq_str:
        return

    hist: Dict[str, int] = json.loads(histogram_str)
    total = sum(hist.values())
    if total == 0:
        return

    nq = int(nq_str)
    expected_state = expected_str.strip()

    # Marginalize wide histograms to payload qubits (last nq bits)
    key_len = len(next(iter(hist)))
    if key_len > nq:
        from collections import Counter

        marginal: Counter = Counter()
        for k, v in hist.items():
            marginal[k[-nq:]] += v
        hist = dict(marginal)

    # Build distributions over the full nq-bit state space
    all_states = [format(i, f"0{nq}b") for i in range(2**nq)]
    obs = {s: hist.get(s, 0) / total for s in all_states}
    # Delta ideal: P(expected_state) = 1, all others = 0
    ideal = {s: (1.0 if s == expected_state else 0.0) for s in all_states}

    # Hellinger: BC = sum sqrt(p_i * q_i); HF = BC^2; HD = sqrt(1 - BC)
    bc = sum(math.sqrt(obs[s] * ideal[s]) for s in all_states)
    hf = bc**2
    hd = math.sqrt(max(1.0 - bc, 0.0))

    # TVD = 0.5 * sum |p_i - q_i|; TVD-F = 1 - TVD
    tvd = 0.5 * sum(abs(obs[s] - ideal[s]) for s in all_states)
    tvd_fid = 1.0 - tvd

    row["hellinger_fidelity"] = _fmt(hf)
    row["hellinger_distance"] = _fmt(hd)
    row["tvd"] = _fmt(tvd)
    row["tvd_fidelity"] = _fmt(tvd_fid)


def _teleportation_rows_with_fidelity(
    teleportation_dir: Path,
) -> List[Dict[str, str]]:
    """Compute teleportation DSR rows and add Hellinger/TVD fidelity columns."""
    csv_paths = _collect_teleportation_paths(teleportation_dir)
    rows = _compute_teleportation_dsr_rows(csv_paths)

    for row in rows:
        _compute_teleportation_fidelity(row)
        # Ensure columns exist even if computation failed
        row.setdefault("hellinger_fidelity", "")
        row.setdefault("hellinger_distance", "")
        row.setdefault("tvd", "")
        row.setdefault("tvd_fidelity", "")

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    default_teleportation = PAPERS / "teleportation"
    default_output = PAPERS / "DSR_result.csv"

    parser = argparse.ArgumentParser(
        description="Build unified DSR_result.csv from JSON + teleportation CSV."
    )
    parser.add_argument("--teleportation-dir", type=Path, default=default_teleportation)
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()

    # ── Grover / QFT (JSON) ──
    json_paths = _collect_json_paths()
    json_rows = _extract_rows_from_json(json_paths)

    grover_count = sum(1 for r in json_rows if r["algorithm"] == "GROVER")
    qft_count = sum(1 for r in json_rows if r["algorithm"] == "QFT")

    # ── Teleportation (CSV) ──
    tp_rows = _teleportation_rows_with_fidelity(args.teleportation_dir)

    # ── Merge and write ──
    all_rows = json_rows + tp_rows

    if not all_rows:
        print("No rows generated. Check input paths.")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {args.output}")
    print(f"  Grover (JSON):       {grover_count:4d} rows  ({len(json_paths)} files total)")
    print(f"  QFT (JSON):          {qft_count:4d} rows")
    print(f"  Teleportation (CSV): {len(tp_rows):4d} rows")
    print(f"  Grand total:         {len(all_rows):4d} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
