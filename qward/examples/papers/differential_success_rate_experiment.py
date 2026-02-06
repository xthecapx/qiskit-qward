"""
Compute Differential Success Rate (DSR) variants from QPU histograms.

Reads Grover, QFT, and Teleportation QPU data, extracts histograms, and writes
a CSV with DSR values from multiple separation formulas.
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


def _iter_results(payload: Dict) -> Iterable[Dict]:
    if "jobs" in payload and isinstance(payload["jobs"], list):
        for job in payload["jobs"]:
            yield job
    if "individual_results" in payload and isinstance(payload["individual_results"], list):
        for result in payload["individual_results"]:
            yield result


def _normalize_expected(expected: Optional[Iterable[str]]) -> List[str]:
    if expected is None:
        return []
    if isinstance(expected, str):
        return [expected]
    return list(expected)


def _expected_from_config(config: Dict, result: Dict) -> List[str]:
    # Prefer result-level fields if present
    if result.get("marked_states"):
        return _normalize_expected(result.get("marked_states"))
    if result.get("input_state"):
        return _normalize_expected(result.get("input_state"))

    # Then fall back to config-level fields
    if config.get("marked_states"):
        return _normalize_expected(config.get("marked_states"))
    if config.get("input_state"):
        return _normalize_expected(config.get("input_state"))

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
            size = 2 ** num_qubits
            step = size / period
            peaks = []
            for k in range(period):
                value = int(round(k * step))
                peaks.append(format(value, f"0{num_qubits}b"))
            # Keep unique peaks in order
            seen = set()
            ordered = []
            for peak in peaks:
                if peak not in seen:
                    seen.add(peak)
                    ordered.append(peak)
            return ordered
    return []


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _counts_to_json(counts: Dict) -> str:
    return json.dumps(counts, sort_keys=True)


def _parse_counts_string(counts_str: str) -> Optional[Dict[str, int]]:
    """Parse counts from CSV string representation."""
    if not counts_str or counts_str.strip() == "":
        return None
    try:
        # Try JSON first
        return json.loads(counts_str)
    except json.JSONDecodeError:
        pass
    try:
        # Try Python literal (e.g., "{'0': 100, '1': 50}")
        parsed = ast.literal_eval(counts_str)
        if isinstance(parsed, dict):
            return {str(k): int(v) for k, v in parsed.items()}
    except (ValueError, SyntaxError):
        pass
    return None


def _compute_teleportation_dsr_rows(csv_paths: List[Path]) -> List[Dict[str, str]]:
    """
    Compute DSR rows from teleportation CSV files.

    For teleportation, expected outcome = '0' * payload_size.
    """
    rows: List[Dict[str, str]] = []

    for csv_path in csv_paths:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip non-completed rows
                status = row.get("status", "")
                if status != "completed":
                    continue

                # Parse counts
                counts_str = row.get("counts", "")
                counts = _parse_counts_string(counts_str)
                if not counts:
                    continue

                # Get payload_size to determine expected outcome
                payload_size_str = row.get("payload_size", "")
                if not payload_size_str:
                    continue
                try:
                    payload_size = int(payload_size_str)
                except ValueError:
                    continue

                if payload_size <= 0:
                    continue

                # Expected outcome is all zeros with length = payload_size
                expected = ["0" * payload_size]

                # Compute DSR variants
                dsr_michelson, peak_mismatch = compute_dsr_with_flags(counts, expected)
                dsr_ratio = compute_dsr_ratio(counts, expected)
                dsr_log_ratio = compute_dsr_log_ratio(counts, expected)
                dsr_norm_margin = compute_dsr_normalized_margin(counts, expected)

                # Extract metadata
                shots = sum(counts.values())
                circuit_depth = row.get("circuit_depth", "")
                num_qubits = row.get("num_qubits", "")

                # Determine backend info
                backend_name = row.get("ibm_backend", row.get("qbraid_device_id", ""))
                execution_type = row.get("execution_type", "")
                if "ibm" in execution_type.lower() or "ibm" in backend_name.lower():
                    backend_type = "qpu"
                elif "qbraid" in execution_type.lower() or "rigetti" in backend_name.lower():
                    backend_type = "qpu"
                else:
                    backend_type = row.get("backend_type", "qpu")

                # Use num_gates or payload_size for additional grouping
                num_gates = row.get("num_gates", "")

                # Construct result row
                result_row = {
                    "algorithm": "TELEPORTATION",
                    "execution_type": execution_type.upper() if execution_type else "QPU",
                    "backend_name": str(backend_name),
                    "backend_type": str(backend_type),
                    "noise_model": "",
                    "config_id": f"payload_{payload_size}",
                    "result_id": row.get("ibm_job_id", row.get("job_id", "")),
                    "optimization_level": "",  # Not applicable for teleportation
                    "num_qubits": str(payload_size),  # Use payload_size as effective qubits
                    "circuit_depth": str(circuit_depth),
                    "transpiled_depth": str(circuit_depth),  # Use circuit_depth as transpiled
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


def compute_dsr_rows(input_paths: List[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in input_paths:
        payload = json.loads(path.read_text())
        config = payload.get("config", {})
        algorithm = payload.get("algorithm", "unknown")
        backend = payload.get("backend_name", "")
        config_id = payload.get("config_id", "")
        execution_type = payload.get("execution_type", "")

        for result in _iter_results(payload):
            counts = result.get("counts")
            if not counts:
                continue

            expected = _expected_from_config(config, result)
            if not expected:
                continue

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


def _collect_input_paths(grover_dir: Path, qft_dir: Path) -> List[Path]:
    """Collect JSON input paths for Grover and QFT experiments."""
    paths: List[Path] = []
    if grover_dir.exists():
        paths.extend(sorted(grover_dir.glob("*.json")))
    if qft_dir.exists():
        paths.extend(sorted(qft_dir.glob("*.json")))
    return paths


def _collect_teleportation_paths(teleportation_dir: Path) -> List[Path]:
    """Collect CSV input paths for Teleportation experiments."""
    paths: List[Path] = []
    if not teleportation_dir.exists():
        return paths

    # Collect top-level CSV files
    paths.extend(sorted(teleportation_dir.glob("*.csv")))

    # Collect CSV files from ibm/ and aws/ subdirectories
    ibm_dir = teleportation_dir / "ibm"
    if ibm_dir.exists():
        paths.extend(sorted(ibm_dir.glob("*.csv")))

    aws_dir = teleportation_dir / "aws"
    if aws_dir.exists():
        paths.extend(sorted(aws_dir.glob("*.csv")))

    return paths


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    default_grover = repo_root / "examples" / "papers" / "grover" / "data" / "qpu" / "raw"
    default_qft = repo_root / "examples" / "papers" / "qft" / "data" / "qpu" / "raw"
    default_teleportation = repo_root / "examples" / "papers" / "teleportation"
    default_output = repo_root / "examples" / "papers" / "DSR_result.csv"

    parser = argparse.ArgumentParser(description="Compute DSR variants from QPU histograms.")
    parser.add_argument("--grover-dir", type=Path, default=default_grover)
    parser.add_argument("--qft-dir", type=Path, default=default_qft)
    parser.add_argument("--teleportation-dir", type=Path, default=default_teleportation)
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()

    # Collect Grover and QFT rows from JSON files
    input_paths = _collect_input_paths(args.grover_dir, args.qft_dir)
    rows = compute_dsr_rows(input_paths)

    # Collect Teleportation rows from CSV files
    teleportation_paths = _collect_teleportation_paths(args.teleportation_dir)
    teleportation_rows = _compute_teleportation_dsr_rows(teleportation_paths)
    rows.extend(teleportation_rows)

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
    print(f"  - Grover/QFT: {len(rows) - len(teleportation_rows)} rows")
    print(f"  - Teleportation: {len(teleportation_rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
