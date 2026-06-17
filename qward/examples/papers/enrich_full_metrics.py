"""
Enrich QPU result JSONs with complete pre-runtime QWARD metrics.

Backfills qward_metrics using all 5 strategies:
  - QiskitMetrics, ComplexityMetrics, StructuralMetrics, BehavioralMetrics, ElementMetrics

Pre-runtime metrics are deterministic (depend only on the circuit), so safe to backfill.

Idempotent: re-running skips files that already have valid metrics (no "error" key).

Usage:
  PYTHONPATH=. uv run python qward/examples/papers/enrich_full_metrics.py --dataset all
  PYTHONPATH=. uv run python qward/examples/papers/enrich_full_metrics.py --dataset grover-ibm
  PYTHONPATH=. uv run python qward/examples/papers/enrich_full_metrics.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

from qiskit import QuantumCircuit

from qward import Scanner
from qward.metrics import (
    QiskitMetrics,
    ComplexityMetrics,
    StructuralMetrics,
    BehavioralMetrics,
    ElementMetrics,
)
from qward.algorithms.grover import Grover
from qward.algorithms.qft import QFTCircuitGenerator

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
# Circuit reconstruction (mirrors enrich_hellinger.py)
# ---------------------------------------------------------------------------


def _build_grover_circuit(config: Dict) -> QuantumCircuit:
    marked_states = config["marked_states"]
    grover = Grover(marked_states)
    return grover.circuit


def _build_qft_circuit(config: Dict) -> QuantumCircuit:
    num_qubits = config["num_qubits"]
    test_mode = config.get("test_mode", "roundtrip")

    if test_mode == "roundtrip":
        input_state = config.get("input_state", "0" * num_qubits)
        gen = QFTCircuitGenerator(num_qubits, test_mode="roundtrip", input_state=input_state)
        return gen.circuit
    elif test_mode == "period_detection":
        period = config["period"]
        gen = QFTCircuitGenerator(num_qubits, test_mode="period_detection", period=period)
        return gen.circuit
    else:
        raise ValueError(f"Unknown QFT test_mode: {test_mode}")


def _reconstruct_circuit(data: Dict) -> Optional[QuantumCircuit]:
    """Reconstruct circuit from JSON data config."""
    algorithm = data.get("algorithm", "").upper()
    config = data.get("config", {})

    if "GROVER" in algorithm:
        return _build_grover_circuit(config)
    elif "QFT" in algorithm:
        return _build_qft_circuit(config)
    else:
        return None


# ---------------------------------------------------------------------------
# Metrics calculation
# ---------------------------------------------------------------------------


def _calculate_full_metrics(circuit: QuantumCircuit) -> Dict[str, Any]:
    """Run all 5 pre-runtime metric strategies on a circuit.

    Returns dict of {strategy_name: {col: value, ...}} or {"error": msg}.
    """
    scanner = Scanner(circuit=circuit)
    scanner.add_strategy(QiskitMetrics(circuit))
    scanner.add_strategy(ComplexityMetrics(circuit))
    scanner.add_strategy(StructuralMetrics(circuit))
    scanner.add_strategy(BehavioralMetrics(circuit))
    scanner.add_strategy(ElementMetrics(circuit))

    metrics_dict = scanner.calculate_metrics()

    result: Dict[str, Any] = {}
    for metric_name, df in metrics_dict.items():
        if df is not None and hasattr(df, "empty") and not df.empty:
            row = df.iloc[0]
            result[metric_name] = {}
            for col in df.columns:
                val = row[col]
                if isinstance(val, (int, float, str, bool, type(None))):
                    result[metric_name][col] = val
                elif hasattr(val, "item"):
                    result[metric_name][col] = val.item()
                elif isinstance(val, (list, tuple)):
                    result[metric_name][col] = [str(v) for v in val]
                else:
                    result[metric_name][col] = str(val)

    return result


def _has_valid_metrics(qward_metrics: Any) -> bool:
    """Check if qward_metrics is already populated (no error, has actual data)."""
    if not isinstance(qward_metrics, dict):
        return False
    if "error" in qward_metrics:
        return False
    if len(qward_metrics) == 0:
        return False
    return True


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


def enrich_file(filepath: Path, dry_run: bool = False) -> str:
    """Enrich a single JSON file with full QWARD metrics.

    Returns status: "enriched", "skipped", or "error: <msg>".
    """
    with open(filepath) as f:
        data = json.load(f)

    individual_results = data.get("individual_results", [])
    if not individual_results:
        return "skipped"

    first_metrics = individual_results[0].get("qward_metrics", {})
    if _has_valid_metrics(first_metrics):
        return "skipped"

    circuit = _reconstruct_circuit(data)
    if circuit is None:
        return f"error: cannot reconstruct circuit for algorithm={data.get('algorithm')}"

    try:
        metrics = _calculate_full_metrics(circuit)
    except Exception as e:
        return f"error: metrics calculation failed: {e}"

    if not metrics or "error" in metrics:
        return f"error: empty metrics result"

    if dry_run:
        return "would_enrich"

    for ir in individual_results:
        ir["qward_metrics"] = metrics

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return "enriched"


def main():
    parser = argparse.ArgumentParser(description="Enrich QPU data with full QWARD metrics")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset(s) to enrich",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be enriched without modifying files",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        dirs = list(DATASETS.values())
    else:
        dirs = [DATASETS[args.dataset]]

    files = []
    for d in dirs:
        if d.exists():
            files.extend(sorted(d.glob("*.json")))

    print(f"Found {len(files)} JSON files to process")
    if args.dry_run:
        print("DRY RUN — no files will be modified\n")

    stats = {"enriched": 0, "skipped": 0, "errors": 0}
    start = time.time()

    for i, fp in enumerate(files, 1):
        status = enrich_file(fp, dry_run=args.dry_run)

        if status == "enriched" or status == "would_enrich":
            stats["enriched"] += 1
            print(f"  [{i}/{len(files)}] {fp.name}: {status}")
        elif status == "skipped":
            stats["skipped"] += 1
        else:
            stats["errors"] += 1
            print(f"  [{i}/{len(files)}] {fp.name}: {status}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Enriched: {stats['enriched']}")
    print(f"  Skipped:  {stats['skipped']}")
    print(f"  Errors:   {stats['errors']}")


if __name__ == "__main__":
    main()
