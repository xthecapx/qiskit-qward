#!/usr/bin/env python3
"""
Enrich existing QPU result files with backend calibration from historical jobs.

Uses `job.properties()` from IBM Runtime API which returns the calibration
snapshot from when the job was executed — NOT current calibration.

Usage:
    PYTHONPATH=. uv run python qward/examples/papers/enrich_calibration.py --dataset all
    PYTHONPATH=. uv run python qward/examples/papers/enrich_calibration.py --dataset grover --dry-run
"""

import argparse
import glob
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# Load IBM credentials from .env
ENV_PATH = Path(__file__).resolve().parent.parent.parent.parent / ".env"
if ENV_PATH.exists():
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "IBM_QUANTUM" in line and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip('"'))

from qiskit_ibm_runtime import QiskitRuntimeService

DATA_DIRS = {
    "grover": "qward/examples/papers/grover/data/qpu/raw",
    "qft": "qward/examples/papers/qft/data/qpu/raw",
    "bv": "qward/examples/papers/bv/data/qpu/raw",
    "ghz": "qward/examples/papers/ghz/data/qpu/raw",
    "pe": "qward/examples/papers/phase_estimation/data/qpu/raw",
    "vtp": "qward/examples/papers/vtp/data/qpu/raw",
    "rv": "qward/examples/papers/random_volumetric/data/qpu/raw",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def get_service() -> QiskitRuntimeService:
    """Create IBM Runtime service from env vars."""
    return QiskitRuntimeService(
        channel=os.environ["IBM_QUANTUM_CHANNEL"],
        token=os.environ["IBM_QUANTUM_TOKEN"],
        instance=os.environ["IBM_QUANTUM_INSTANCE"],
    )


def extract_calibration_from_properties(props) -> Dict[str, Any]:
    """Extract calibration metrics from BackendProperties object."""
    num_qubits = len(props.qubits)

    t1_vals = []
    t2_vals = []
    readout_errors = []

    for q in range(num_qubits):
        for p in props.qubits[q]:
            if p.name == "T1" and p.value and p.value > 0:
                t1_vals.append(p.value)
            elif p.name == "T2" and p.value and p.value > 0:
                t2_vals.append(p.value)
            elif p.name == "readout_error" and p.value is not None:
                readout_errors.append(p.value)

    single_q_errors = []
    two_q_errors = []
    single_q_gates = {"sx", "x", "rx", "ry", "rz", "id"}

    for g in props.gates:
        for p in g.parameters:
            if p.name == "gate_error" and p.value and p.value > 0:
                if g.gate in single_q_gates:
                    single_q_errors.append(p.value)
                elif len(g.qubits) == 2:
                    two_q_errors.append(p.value)

    return {
        "median_single_qubit_gate_error": (
            statistics.median(single_q_errors) if single_q_errors else None
        ),
        "median_two_qubit_gate_error": statistics.median(two_q_errors) if two_q_errors else None,
        "median_readout_error": statistics.median(readout_errors) if readout_errors else None,
        "median_t1_us": statistics.median(t1_vals) if t1_vals else None,
        "median_t2_us": statistics.median(t2_vals) if t2_vals else None,
        "num_operational_qubits": num_qubits,
        "backend_name": props.backend_name,
        "calibration_timestamp": str(props.last_update_date),
        "provider": "ibm",
        "source": "job.properties()",
    }


def get_job_ids_from_file(data: Dict[str, Any]) -> List[str]:
    """Extract all job_ids from a result file."""
    job_ids = []
    for r in data.get("individual_results", []):
        jid = r.get("job_id")
        if jid:
            job_ids.append(jid)
    batch_id = data.get("batch_id")
    if batch_id and not job_ids:
        job_ids.append(batch_id)
    return job_ids


def enrich_file(filepath: str, service: QiskitRuntimeService, dry_run: bool = False) -> str:
    """Enrich a single file with calibration data.

    Returns: 'enriched', 'skipped' (already has cal), 'failed', or 'no_jobs'
    """
    with open(filepath) as f:
        data = json.load(f)

    if data.get("backend_calibration") and data["backend_calibration"].get("median_t1_us"):
        return "skipped"

    job_ids = get_job_ids_from_file(data)
    if not job_ids:
        return "no_jobs"

    if dry_run:
        return "would_enrich"

    # Try first job_id
    for jid in job_ids:
        try:
            job = service.job(jid)
            props = job.properties()
            if props is None:
                continue

            calibration = extract_calibration_from_properties(props)
            if calibration["median_t1_us"] is None:
                continue

            data["backend_calibration"] = calibration
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            return "enriched"
        except Exception as e:
            error_str = str(e)
            if "not found" in error_str.lower() or "404" in error_str:
                continue
            print(f"    Error for job {jid}: {error_str[:80]}")
            continue

    return "failed"


def main():
    parser = argparse.ArgumentParser(description="Enrich QPU results with historical calibration")
    parser.add_argument(
        "--dataset", type=str, default="all", choices=["all"] + list(DATA_DIRS.keys())
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Max files to process")
    args = parser.parse_args()

    datasets = list(DATA_DIRS.keys()) if args.dataset == "all" else [args.dataset]

    print("=" * 60)
    print("CALIBRATION ENRICHMENT")
    print(f"Datasets: {datasets}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)

    if not args.dry_run:
        service = get_service()
    else:
        service = None

    totals = {"enriched": 0, "skipped": 0, "failed": 0, "no_jobs": 0, "would_enrich": 0}

    for ds in datasets:
        dir_path = PROJECT_ROOT / DATA_DIRS[ds]
        if not dir_path.exists():
            print(f"\n[{ds}] Directory not found: {dir_path}")
            continue

        files = sorted(glob.glob(str(dir_path / "*.json")))
        print(f"\n[{ds}] {len(files)} files")

        count = 0
        for fp in files:
            if args.limit and count >= args.limit:
                break

            fname = Path(fp).name
            result = enrich_file(fp, service, dry_run=args.dry_run)
            totals[result] += 1
            count += 1

            if result == "enriched":
                print(f"  + {fname}")
            elif result == "failed":
                print(f"  ! {fname} (failed)")
            elif result == "no_jobs":
                print(f"  ? {fname} (no job_ids)")

            # Rate limit: IBM API
            if result in ("enriched", "failed") and not args.dry_run:
                time.sleep(0.5)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"  Enriched: {totals['enriched']}")
    print(f"  Skipped (already had cal): {totals['skipped']}")
    print(f"  Failed: {totals['failed']}")
    print(f"  No job_ids: {totals['no_jobs']}")
    if args.dry_run:
        print(f"  Would enrich: {totals['would_enrich']}")
    print(f"  Total processed: {sum(totals.values())}")


if __name__ == "__main__":
    main()
