"""CLI for QWARD scan module.

Usage:
    uv run -m qward.scan job <JOB_ID> --target-state "101"
    uv run -m qward.scan batch <BATCH_ID> --target-state "1010" --aggregation merge
    uv run -m qward.scan pre --circuit circuit.qpy
    uv run -m qward.scan counts --counts '{"00": 900, "11": 100}' --target-state "00"
"""

import argparse
import json
import sys
from pathlib import Path


def _load_circuit(path: str):
    """Load circuit from .qpy or .qasm file."""
    from qiskit import QuantumCircuit
    from qiskit.qpy import load as qpy_load

    p = Path(path)
    if p.suffix == ".qpy":
        with open(p, "rb") as f:
            circuits = qpy_load(f)
            return circuits[0] if isinstance(circuits, list) else circuits
    elif p.suffix in (".qasm", ".qasm3"):
        return QuantumCircuit.from_qasm_file(str(p))
    else:
        raise ValueError(f"Unsupported circuit format: {p.suffix}. Use .qpy or .qasm")


def _output_results(results, output_path=None):
    """Print or save results."""
    import pandas as pd

    if output_path is None:
        for name, df in results.items():
            print(f"\n{'=' * 60}")
            print(f"  {name}")
            print(f"{'=' * 60}")
            print(df.to_string())
        return

    p = Path(output_path)
    if p.suffix == ".csv":
        combined = pd.concat(
            [df.assign(_metric=name) for name, df in results.items()],
            ignore_index=True,
        )
        combined.to_csv(p, index=False)
        print(f"Saved to {p}")
    elif p.suffix == ".json":
        data = {name: df.to_dict(orient="records") for name, df in results.items()}
        p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        print(f"Saved to {p}")
    else:
        raise ValueError(f"Unsupported output format: {p.suffix}. Use .csv or .json")


def cmd_job(args):
    from qward.scan import scan_job

    results = scan_job(
        args.job_id,
        expected_outcomes=args.expected_outcomes,
        target_state=args.target_state,
        target_histogram=json.loads(args.target_histogram) if args.target_histogram else None,
        trim_idle=not args.no_trim,
    )
    _output_results(results, args.output)


def cmd_batch(args):
    from qward.scan import scan_batch

    results = scan_batch(
        args.batch_id,
        expected_outcomes=args.expected_outcomes,
        target_state=args.target_state,
        target_histogram=json.loads(args.target_histogram) if args.target_histogram else None,
        aggregation=args.aggregation,
        trim_idle=not args.no_trim,
    )
    _output_results(results, args.output)


def cmd_pre(args):
    from qward.scan import scan_pre

    circuit = _load_circuit(args.circuit)
    results = scan_pre(circuit, include_quantum_specific=not args.no_quantum_specific)
    _output_results(results, args.output)


def cmd_counts(args):
    from qward.scan import scan_post

    circuit = _load_circuit(args.circuit)

    if args.counts_file:
        counts = json.loads(Path(args.counts_file).read_text(encoding="utf-8"))
    else:
        counts = json.loads(args.counts)

    results = scan_post(
        circuit,
        counts,
        expected_outcomes=args.expected_outcomes,
        target_state=args.target_state,
        target_histogram=json.loads(args.target_histogram) if args.target_histogram else None,
    )
    _output_results(results, args.output)


def main():
    parser = argparse.ArgumentParser(
        prog="qward.scan",
        description="QWARD scan — quantum circuit fidelity analysis",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common arguments
    def add_target_args(p):
        p.add_argument("--target-state", dest="target_state", help="Target bitstring (shortcut)")
        p.add_argument(
            "--expected-outcomes",
            dest="expected_outcomes",
            nargs="*",
            help="Expected bitstrings for DSR",
        )
        p.add_argument(
            "--target-histogram",
            dest="target_histogram",
            help="Ideal distribution as JSON string",
        )
        p.add_argument("--output", "-o", help="Output file (.csv or .json)")

    # job subcommand
    p_job = subparsers.add_parser("job", help="Scan IBM Quantum job by ID")
    p_job.add_argument("job_id", help="IBM Quantum job ID")
    p_job.add_argument("--no-trim", action="store_true", help="Skip idle qubit trimming")
    add_target_args(p_job)
    p_job.set_defaults(func=cmd_job)

    # batch subcommand
    p_batch = subparsers.add_parser("batch", help="Scan IBM Quantum batch by ID")
    p_batch.add_argument("batch_id", help="IBM batch/session ID")
    p_batch.add_argument(
        "--aggregation",
        choices=["last", "merge"],
        default="last",
        help="How to combine job counts (default: last)",
    )
    p_batch.add_argument("--no-trim", action="store_true", help="Skip idle qubit trimming")
    add_target_args(p_batch)
    p_batch.set_defaults(func=cmd_batch)

    # pre subcommand
    p_pre = subparsers.add_parser("pre", help="Pre-runtime metrics from circuit file")
    p_pre.add_argument("--circuit", required=True, help="Circuit file (.qpy or .qasm)")
    p_pre.add_argument(
        "--no-quantum-specific",
        action="store_true",
        help="Skip QuantumSpecificMetrics",
    )
    p_pre.add_argument("--output", "-o", help="Output file (.csv or .json)")
    p_pre.set_defaults(func=cmd_pre)

    # counts subcommand
    p_counts = subparsers.add_parser("counts", help="Post-runtime metrics from counts")
    p_counts.add_argument("--circuit", required=True, help="Circuit file (.qpy or .qasm)")
    p_counts.add_argument("--counts", help="Counts as JSON string")
    p_counts.add_argument("--counts-file", dest="counts_file", help="Counts from JSON file")
    add_target_args(p_counts)
    p_counts.set_defaults(func=cmd_counts)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
