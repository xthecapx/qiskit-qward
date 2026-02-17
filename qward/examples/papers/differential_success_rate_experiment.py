"""
Compute Differential Success Rate (DSR) variants from QPU histograms.

Two data schemas are supported:

1. **Grover / QFT** – JSON files with `individual_results` list.
   Each result carries counts, optimization_level, transpiled_depth, etc.
   Expected outcomes are derived from the top-level `config` dict.

2. **Teleportation** – CSV files with per-row histogram and payload_size.
   Expected outcome is `'0' * payload_size`.

Only results with **non-empty counts** are included (QUEUED / ERROR / empty
counts are silently skipped).
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from qward.metrics.differential_success_rate import (
    compute_dsr_with_flags,
    compute_dsr_ratio,
    compute_dsr_log_ratio,
    compute_dsr_normalized_margin,
)

# =============================================================================
# Helpers
# =============================================================================


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _counts_to_json(counts: Dict) -> str:
    return json.dumps(counts, sort_keys=True)


def _normalize_expected(expected: Optional[Iterable[str]]) -> List[str]:
    if expected is None:
        return []
    if isinstance(expected, str):
        return [expected]
    return list(expected)


def _parse_counts_string(counts_str: str) -> Optional[Dict[str, int]]:
    """Parse counts from CSV string representation."""
    if not counts_str or counts_str.strip() == "":
        return None
    try:
        return json.loads(counts_str)
    except json.JSONDecodeError:
        pass
    try:
        parsed = ast.literal_eval(counts_str)
        if isinstance(parsed, dict):
            return {str(k): int(v) for k, v in parsed.items()}
    except (ValueError, SyntaxError):
        pass
    return None


# =============================================================================
# Schema 1: Grover / QFT  (JSON with `individual_results`)
# =============================================================================


def _expected_from_config(config: Dict, result: Dict) -> List[str]:
    """Derive expected measurement outcomes from config and result metadata."""
    # Prefer result-level fields if present
    if result.get("marked_states"):
        return _normalize_expected(result["marked_states"])
    if result.get("input_state"):
        return _normalize_expected(result["input_state"])

    # Fall back to config-level fields
    if config.get("marked_states"):
        return _normalize_expected(config["marked_states"])
    if config.get("input_state"):
        return _normalize_expected(config["input_state"])

    # Period-detection fallback: expected peaks from period and num_qubits
    period = config.get("period")
    num_qubits = config.get("num_qubits")
    if period and num_qubits:
        try:
            period = int(period)
            num_qubits = int(num_qubits)
        except (TypeError, ValueError):
            return []
        if period > 0 and num_qubits > 0:
            size = 2**num_qubits
            step = size / period
            seen: set[str] = set()
            ordered: List[str] = []
            for k in range(period):
                peak = format(int(round(k * step)), f"0{num_qubits}b")
                if peak not in seen:
                    seen.add(peak)
                    ordered.append(peak)
            return ordered

    return []


def compute_dsr_rows(input_paths: List[Path]) -> List[Dict[str, str]]:
    """Compute DSR rows from Grover / QFT JSON files.

    Only results whose ``individual_results`` entries have non-empty
    ``counts`` are included.  Jobs that are QUEUED, ERROR, or have
    empty histograms are silently skipped.
    """
    rows: List[Dict[str, str]] = []

    for path in input_paths:
        payload = json.loads(path.read_text())
        config = payload.get("config", {})
        algorithm = payload.get("algorithm", "unknown")
        backend = payload.get("backend_name", "")
        config_id = payload.get("config_id", "")
        execution_type = payload.get("execution_type", "")

        # Unified schema: iterate over `individual_results`
        results = payload.get("individual_results", [])

        for result in results:
            counts = result.get("counts")
            if not counts:
                # Skip jobs without histogram data (QUEUED, ERROR, etc.)
                continue

            expected = _expected_from_config(config, result)
            if not expected:
                continue

            # Compute all DSR variants
            dsr_michelson, peak_mismatch = compute_dsr_with_flags(counts, expected)
            dsr_ratio = compute_dsr_ratio(counts, expected)
            dsr_log_ratio = compute_dsr_log_ratio(counts, expected)
            dsr_norm_margin = compute_dsr_normalized_margin(counts, expected)

            shots = result.get("shots")
            if shots is None:
                shots = sum(counts.values())

            num_qubits = result.get("num_qubits", config.get("num_qubits", ""))
            circuit_depth = result.get("circuit_depth", "")
            transpiled_depth = result.get("transpiled_depth", "")
            total_gates = result.get("total_gates", "")
            backend_type = result.get("backend_type", "")
            noise_model = result.get("noise_model", payload.get("noise_id", ""))

            row = {
                "algorithm": str(algorithm),
                "execution_type": str(execution_type),
                "backend_name": str(result.get("backend_name", backend)),
                "backend_type": str(backend_type),
                "noise_model": str(noise_model),
                "config_id": str(result.get("config_id", config_id)),
                "result_id": str(result.get("experiment_id", result.get("job_id", ""))),
                "optimization_level": str(result.get("optimization_level", "")),
                "num_qubits": str(num_qubits),
                "circuit_depth": str(circuit_depth),
                "transpiled_depth": str(transpiled_depth),
                "total_gates": str(total_gates),
                "shots": str(shots),
                "expected_outcomes": ",".join(expected),
                "histogram": _counts_to_json(counts),
                "peak_mismatch": str(bool(peak_mismatch)),
                "dsr_michelson": _format_float(dsr_michelson),
                "dsr_ratio": _format_float(dsr_ratio),
                "dsr_log_ratio": _format_float(dsr_log_ratio),
                "dsr_normalized_margin": _format_float(dsr_norm_margin),
                "source_file": path.name,
            }
            rows.append(row)

    return rows


# =============================================================================
# Schema 2: Teleportation  (CSV with per-row histogram)
# =============================================================================


def _compute_teleportation_dsr_rows(csv_paths: List[Path]) -> List[Dict[str, str]]:
    """Compute DSR rows from teleportation CSV files.

    For teleportation, expected outcome = ``'0' * payload_size``.
    """
    rows: List[Dict[str, str]] = []

    for csv_path in csv_paths:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status", "") != "completed":
                    continue

                counts = _parse_counts_string(row.get("counts", ""))
                if not counts:
                    continue

                payload_size_str = row.get("payload_size", "")
                if not payload_size_str:
                    continue
                try:
                    payload_size = int(payload_size_str)
                except ValueError:
                    continue
                if payload_size <= 0:
                    continue

                expected = ["0" * payload_size]

                # DSR variants
                dsr_michelson, peak_mismatch = compute_dsr_with_flags(counts, expected)
                dsr_ratio = compute_dsr_ratio(counts, expected)
                dsr_log_ratio = compute_dsr_log_ratio(counts, expected)
                dsr_norm_margin = compute_dsr_normalized_margin(counts, expected)

                shots = sum(counts.values())
                circuit_depth = row.get("circuit_depth", "")
                num_qubits = row.get("num_qubits", "")

                backend_name = row.get("ibm_backend", row.get("qbraid_device_id", ""))
                execution_type = row.get("execution_type", "")
                if any(
                    kw in (execution_type + backend_name).lower()
                    for kw in ("ibm", "rigetti", "qbraid")
                ):
                    backend_type = "qpu"
                else:
                    backend_type = row.get("backend_type", "qpu")

                num_gates = row.get("num_gates", "")

                result_row = {
                    "algorithm": "TELEPORTATION",
                    "execution_type": (execution_type.upper() if execution_type else "QPU"),
                    "backend_name": str(backend_name),
                    "backend_type": str(backend_type),
                    "noise_model": "",
                    "config_id": f"payload_{payload_size}",
                    "result_id": row.get("ibm_job_id", row.get("job_id", "")),
                    "optimization_level": "",
                    "num_qubits": str(payload_size),
                    "circuit_depth": str(circuit_depth),
                    "transpiled_depth": str(circuit_depth),
                    "total_gates": str(num_gates),
                    "shots": str(shots),
                    "expected_outcomes": ",".join(expected),
                    "histogram": _counts_to_json(counts),
                    "peak_mismatch": str(bool(peak_mismatch)),
                    "dsr_michelson": _format_float(dsr_michelson),
                    "dsr_ratio": _format_float(dsr_ratio),
                    "dsr_log_ratio": _format_float(dsr_log_ratio),
                    "dsr_normalized_margin": _format_float(dsr_norm_margin),
                    "source_file": csv_path.name,
                }
                rows.append(result_row)

    return rows


# =============================================================================
# Path Collection
# =============================================================================


def _collect_raw_dirs(data_path: Path, provider: Optional[str] = None) -> List[Path]:
    """Collect JSON result directories from a data root or explicit path.

    Supports:
    - Preferred layout: ``.../data/qpu/raw`` (IBM), ``.../data/qpu/aws`` (AWS).
    - Legacy layout: ``.../data/aws/raw``.
    - Legacy direct raw path ``.../data/qpu/raw``.
    - Explicit result path containing JSON files.
    """
    if not data_path.exists():
        return []

    # Explicit directory containing result JSON files
    if data_path.is_dir() and any(data_path.glob("*.json")):
        return [data_path]

    candidates: List[Path] = []
    if provider == "qpu":
        candidates.append(data_path / "qpu" / "raw")
    elif provider == "aws":
        candidates.extend([data_path / "qpu" / "aws", data_path / "aws" / "raw"])
    elif provider:
        candidates.extend([data_path / "qpu" / provider, data_path / provider / "raw"])
    else:
        candidates.extend([data_path / "qpu" / "raw", data_path / "qpu" / "aws"])
        for provider_dir in sorted(p for p in data_path.iterdir() if p.is_dir()):
            provider_raw = provider_dir / "raw"
            if provider_raw.exists():
                candidates.append(provider_raw)

    unique_existing: List[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path.exists() and path not in seen:
            seen.add(path)
            unique_existing.append(path)
    return unique_existing


def _collect_json_paths(
    grover_data: Path, qft_data: Path, provider: Optional[str] = None
) -> List[Path]:
    """Collect JSON input paths for Grover and QFT experiments."""
    paths: List[Path] = []

    for raw_dir in _collect_raw_dirs(grover_data, provider):
        paths.extend(sorted(raw_dir.glob("*.json")))
    for raw_dir in _collect_raw_dirs(qft_data, provider):
        paths.extend(sorted(raw_dir.glob("*.json")))

    return paths


def _collect_teleportation_paths(teleportation_dir: Path) -> List[Path]:
    """Collect CSV input paths for Teleportation experiments."""
    paths: List[Path] = []
    if not teleportation_dir.exists():
        return paths

    paths.extend(sorted(teleportation_dir.glob("*.csv")))

    for subdir_name in ("ibm", "aws"):
        subdir = teleportation_dir / subdir_name
        if subdir.exists():
            paths.extend(sorted(subdir.glob("*.csv")))

    return paths


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    default_grover = repo_root / "examples" / "papers" / "grover" / "data"
    default_qft = repo_root / "examples" / "papers" / "qft" / "data"
    default_teleportation = repo_root / "examples" / "papers" / "teleportation"
    default_output = repo_root / "examples" / "papers" / "DSR_result.csv"

    parser = argparse.ArgumentParser(description="Compute DSR variants from QPU histograms.")
    parser.add_argument("--grover-dir", type=Path, default=default_grover)
    parser.add_argument("--qft-dir", type=Path, default=default_qft)
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Provider filter under each data root (e.g. qpu, aws)",
    )
    parser.add_argument("--teleportation-dir", type=Path, default=default_teleportation)
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()

    # ----- Schema 1: Grover / QFT (JSON, individual_results) -----
    json_paths = _collect_json_paths(args.grover_dir, args.qft_dir, provider=args.provider)
    grover_qft_rows = compute_dsr_rows(json_paths)

    grover_count = sum(1 for r in grover_qft_rows if r["algorithm"] == "GROVER")
    qft_count = sum(1 for r in grover_qft_rows if r["algorithm"] == "QFT")

    # ----- Schema 2: Teleportation (CSV) -----
    teleportation_paths = _collect_teleportation_paths(args.teleportation_dir)
    teleportation_rows = _compute_teleportation_dsr_rows(teleportation_paths)

    # ----- Merge and write -----
    rows = grover_qft_rows + teleportation_rows

    if not rows:
        print("No rows generated. Check input paths or expected outcomes.")
        return 1

    fieldnames = list(rows[0].keys())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"  Schema 1 (JSON):")
    print(
        f"    Grover:        {grover_count} rows  ({len([p for p in json_paths if 'grover' in str(p)])} files)"
    )
    print(
        f"    QFT:           {qft_count} rows  ({len([p for p in json_paths if 'qft' in str(p)])} files)"
    )
    print(f"  Schema 2 (CSV):")
    print(f"    Teleportation: {len(teleportation_rows)} rows  ({len(teleportation_paths)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
