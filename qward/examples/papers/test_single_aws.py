#!/usr/bin/env python3
"""
Quick single-shot test to validate the AWS Braket integration end-to-end.

By default runs on the local AerSimulator using the same transpiled circuit
that would be sent to AWS, so you can verify correctness without QPU cost.
Pass --qpu to submit to real hardware.

Usage:
    python test_single_aws.py                         # simulate S2-1
    python test_single_aws.py --config ASYM-1         # simulate ASYM-1
    python test_single_aws.py --qpu --config ASYM-1   # run on real QPU
    python test_single_aws.py --calibrate             # bit-order calibration
"""

import argparse
import os
import pprint
import sys
from dataclasses import asdict
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qward.algorithms import QuantumCircuitExecutor, Grover, AWSJobResult
from qward.algorithms.executor import QuantumCircuitExecutor as _Executor
from qward.examples.papers.grover.grover_configs import get_config
from qward.metrics.differential_success_rate import (
    compute_dsr_with_flags,
    compute_dsr_ratio,
    compute_dsr_log_ratio,
    compute_dsr_normalized_margin,
)


def _build_calibration_circuit() -> QuantumCircuit:
    """Build a 3-qubit circuit with unambiguous bit-order signature.

    Circuit:
        q0: X, measure
        q1: measure
        q2: measure

    Expected dominant outcome in Qiskit convention: ``001``.
    """
    circuit = QuantumCircuit(3, 3)
    circuit.x(0)
    circuit.measure([0, 1, 2], [0, 1, 2])
    return circuit


def _simulate(circuit_clean: QuantumCircuit, expected_outcomes, shots: int) -> None:
    """Run the transpiled circuit on AerSimulator and print results."""
    print("\n>>> Running on AerSimulator (noiseless)...")
    sim = AerSimulator()
    job = sim.run(circuit_clean, shots=shots)
    result = job.result()
    counts = result.get_counts()

    total = sum(counts.values())
    success_count = sum(v for k, v in counts.items() if k in expected_outcomes)
    success_rate = success_count / total if total > 0 else 0.0
    top5 = dict(sorted(counts.items(), key=lambda x: -x[1])[:5])

    print(f">>> Simulation complete ({total} shots)")
    print(f">>> Unique outcomes: {len(counts)}")
    print(f">>> Top 5: {top5}")
    print(f">>> Success count:  {success_count}")
    print(f">>> Success rate:   {success_rate:.2%}")

    # DSR
    try:
        dsr_m, peak_mm = compute_dsr_with_flags(counts, expected_outcomes)
        dsr_r = compute_dsr_ratio(counts, expected_outcomes)
        dsr_lr = compute_dsr_log_ratio(counts, expected_outcomes)
        dsr_nm = compute_dsr_normalized_margin(counts, expected_outcomes)
        print(f"\n>>> DSR Michelson:         {dsr_m:.4f}")
        print(f">>> DSR Ratio:             {dsr_r:.4f}")
        print(f">>> DSR Log-Ratio:         {dsr_lr:.4f}")
        print(f">>> DSR Normalized Margin: {dsr_nm:.4f}")
        print(f">>> Peak Mismatch:         {peak_mm}")
    except Exception as exc:
        print(f">>> DSR calculation error: {exc}")


def _run_on_qpu(circuit, expected_outcomes, args) -> None:
    """Submit the original circuit to AWS Braket QPU."""
    executor = QuantumCircuitExecutor(shots=args.shots)
    print("\nSubmitting to AWS Braket QPU...")
    aws_result: AWSJobResult = executor.run_aws(
        circuit,
        device_id=args.device,
        region=args.region,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        expected_outcomes=expected_outcomes,
        timeout=args.timeout,
        poll_interval=5,
        show_progress=True,
        wait_for_results=True,
    )

    print("\n" + "=" * 70)
    print("RAW AWSJobResult")
    print("=" * 70)
    result_dict = asdict(aws_result)
    if result_dict.get("qward_metrics"):
        result_dict["qward_metrics"] = {
            k: f"<DataFrame {v.shape if hasattr(v, 'shape') else type(v).__name__}>"
            for k, v in (aws_result.qward_metrics or {}).items()
        }
    pprint.pprint(result_dict, width=120)

    print("\n" + "=" * 70)
    print("QUICK EVALUATION")
    print("=" * 70)
    print(f"Status:          {aws_result.status}")
    print(f"Job ID:          {aws_result.job_id}")
    print(f"Device:          {aws_result.device_name}")
    print(f"Region:          {aws_result.region}")

    if aws_result.counts:
        total = sum(aws_result.counts.values())
        success_count = sum(
            v for k, v in aws_result.counts.items() if k in expected_outcomes
        )
        success_rate = success_count / total if total > 0 else 0.0
        print(f"Total shots:     {total}")
        print(f"Success count:   {success_count}")
        print(f"Success rate:    {success_rate:.2%}")
        print(f"Top 5 counts:    {dict(sorted(aws_result.counts.items(), key=lambda x: -x[1])[:5])}")
    else:
        print("No counts yet (job may still be running).")
        print(f"  Job ARN: {aws_result.job_id}")

    print(f"\nDSR Michelson:          {aws_result.dsr_michelson}")
    print(f"DSR Ratio:              {aws_result.dsr_ratio}")
    print(f"DSR Log-Ratio:          {aws_result.dsr_log_ratio}")
    print(f"DSR Normalized Margin:  {aws_result.dsr_normalized_margin}")
    print(f"Peak Mismatch:          {aws_result.peak_mismatch}")
    print("=" * 70)

    if aws_result.error:
        print(f"\nERROR: {aws_result.error}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick AWS Braket test (single config)")
    parser.add_argument("--config", "-c", default="S2-1", help="Config ID (default: S2-1)")
    parser.add_argument("--device", "-d", default="Ankaa-3", help="Device name")
    parser.add_argument("--region", "-r", default="us-west-1", help="AWS region")
    parser.add_argument("--shots", "-s", type=int, default=1024, help="Shots")
    parser.add_argument("--timeout", "-t", type=int, default=120, help="Timeout in seconds")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run 3-qubit bit-order calibration circuit (expected: 001)",
    )
    parser.add_argument(
        "--qpu",
        action="store_true",
        help="Submit to real QPU (default: simulate locally)",
    )
    args = parser.parse_args()

    if args.calibrate:
        circuit = _build_calibration_circuit()
        expected_outcomes = ["001"]
    else:
        config = get_config(args.config)
        expected_outcomes = list(config.marked_states)
        grover = Grover(marked_states=config.marked_states, use_barriers=True)
        circuit = grover.circuit

    mode = "QPU" if args.qpu else "LOCAL SIMULATOR"

    print("=" * 70)
    print(f"AWS BRAKET TEST — {mode}")
    print("=" * 70)
    if args.calibrate:
        print("Mode:                Bit-order calibration")
        print("Qubits:              3")
        print("Expected outcomes:   ['001']")
    else:
        print(f"Config:              {config.config_id}")
        print(f"Qubits:              {config.num_qubits}")
        print(f"Marked states:       {config.marked_states}")
        print(f"Theoretical success: {config.theoretical_success:.1%}")
    print(f"Shots:               {args.shots}")

    # Build circuit
    print(f"\nOriginal circuit depth: {circuit.depth()}")
    print(f"Original circuit gates: {circuit.size()}")
    if args.calibrate:
        print("\nCalibration circuit:")
        print(circuit.draw(output="text"))

    # Transpile using the same method that run_aws uses
    circuit_clean = _Executor._prepare_circuit_for_aws(circuit)
    print(f"Transpiled circuit depth: {circuit_clean.depth()}")
    print(f"Transpiled circuit gates: {circuit_clean.size()}")
    print("=" * 70)

    if args.qpu:
        _run_on_qpu(circuit, expected_outcomes, args)
    else:
        _simulate(circuit_clean, expected_outcomes, args.shots)


if __name__ == "__main__":
    main()
