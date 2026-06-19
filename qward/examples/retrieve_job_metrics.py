"""
Retrieve IBM Quantum job by ID and calculate QWARD pre/post metrics.

Usage:
    python -m qward.examples.retrieve_job_metrics --job-id <JOB_ID>

Requires:
    - IBM Quantum credentials in .env (IBM_QUANTUM_CHANNEL, IBM_QUANTUM_TOKEN, IBM_QUANTUM_INSTANCE)
    - Or credentials already saved via QiskitRuntimeService.save_account()
"""

import argparse
import io
import os
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from qiskit import QuantumCircuit
from qiskit.qpy import load
from qiskit_ibm_runtime import QiskitRuntimeService

from qward import Scanner
from qward.metrics import FidelityMetrics, ComplexityMetrics, QiskitMetrics
from qward.metrics.behavioral_metrics import BehavioralMetrics
from qward.metrics.element_metrics import ElementMetrics
from qward.metrics.structural_metrics import StructuralMetrics
from qward.metrics.quantum_specific_metrics import QuantumSpecificMetrics

MAX_QUBITS_FOR_UNITARY = 20


def get_service() -> QiskitRuntimeService:
    """Connect to IBM Quantum using .env credentials or saved account."""
    load_dotenv()

    channel = os.getenv("IBM_QUANTUM_CHANNEL")
    token = os.getenv("IBM_QUANTUM_TOKEN")
    instance = os.getenv("IBM_QUANTUM_INSTANCE")

    kwargs = {}
    if channel:
        kwargs["channel"] = channel
    if token:
        kwargs["token"] = token
    if instance:
        kwargs["instance"] = instance

    return QiskitRuntimeService(**kwargs)


def extract_circuit_from_job(job) -> QuantumCircuit:
    """Extract QuantumCircuit from job inputs (QPY-encoded ISA circuit)."""
    inputs = job.inputs
    pubs = inputs.get("pubs", [])
    if not pubs:
        raise ValueError(f"No pubs found in job inputs. Keys: {list(inputs.keys())}")

    circuit_data = pubs[0][0]

    if isinstance(circuit_data, bytes):
        buf = io.BytesIO(circuit_data)
        circuits = load(buf)
        return circuits[0] if isinstance(circuits, list) else circuits

    if isinstance(circuit_data, QuantumCircuit):
        return circuit_data

    raise TypeError(f"Unexpected circuit data type: {type(circuit_data)}")


def trim_idle_qubits(circuit: QuantumCircuit) -> QuantumCircuit:
    """Remove idle qubits from an ISA circuit, keeping only qubits with operations.

    ISA circuits are mapped to the full device topology (e.g. 156 qubits on ibm_fez)
    but typically only a few qubits have actual gates. This trims the circuit down
    to only active qubits so metrics reflect the real computation.

    Args:
        circuit: Full ISA circuit with potentially many idle qubits

    Returns:
        New QuantumCircuit with only active qubits and corresponding classical bits
    """
    from qiskit.circuit import Barrier, Measure

    # Find qubits that have at least one non-barrier gate
    active_qubit_indices = set()
    for instruction in circuit.data:
        gate = instruction[0] if isinstance(instruction, tuple) else instruction.operation
        qubits = instruction[1] if isinstance(instruction, tuple) else instruction.qubits

        if isinstance(gate, Barrier):
            continue

        for qubit in qubits:
            idx = circuit.qubits.index(qubit)
            active_qubit_indices.add(idx)

    active_qubit_indices = sorted(active_qubit_indices)

    if not active_qubit_indices:
        raise ValueError("No active qubits found in circuit")

    num_active = len(active_qubit_indices)
    # Count measurements to determine classical bits needed
    num_measurements = sum(
        1
        for inst in circuit.data
        if isinstance(inst[0] if isinstance(inst, tuple) else inst.operation, Measure)
    )
    num_clbits = max(num_measurements, circuit.num_clbits) if num_measurements > 0 else 0

    # Build trimmed circuit
    trimmed = QuantumCircuit(num_active, min(num_clbits, num_active))

    # Map old qubit index -> new qubit index
    qubit_map = {old_idx: new_idx for new_idx, old_idx in enumerate(active_qubit_indices)}

    clbit_counter = 0
    for instruction in circuit.data:
        gate = instruction[0] if isinstance(instruction, tuple) else instruction.operation
        qubits = instruction[1] if isinstance(instruction, tuple) else instruction.qubits
        clbits = instruction[2] if isinstance(instruction, tuple) else instruction.clbits

        if isinstance(gate, Barrier):
            continue

        # Map qubits
        qubit_indices = [circuit.qubits.index(q) for q in qubits]

        # Skip if any qubit not in active set (shouldn't happen but safety check)
        if not all(idx in qubit_map for idx in qubit_indices):
            continue

        new_qubits = [trimmed.qubits[qubit_map[idx]] for idx in qubit_indices]

        if isinstance(gate, Measure):
            if clbit_counter < trimmed.num_clbits:
                trimmed.append(gate, new_qubits, [trimmed.clbits[clbit_counter]])
                clbit_counter += 1
        else:
            trimmed.append(gate, new_qubits, [])

    return trimmed


def compute_pre_metrics(circuit: QuantumCircuit) -> Dict[str, pd.DataFrame]:
    """All pre-runtime metrics (circuit structure only, no execution needed)."""
    scanner = Scanner(circuit=circuit)
    scanner.add_strategy(QiskitMetrics(circuit))
    scanner.add_strategy(ComplexityMetrics(circuit))
    scanner.add_strategy(ElementMetrics(circuit))
    scanner.add_strategy(StructuralMetrics(circuit))
    scanner.add_strategy(BehavioralMetrics(circuit))

    if circuit.num_qubits <= MAX_QUBITS_FOR_UNITARY:
        scanner.add_strategy(QuantumSpecificMetrics(circuit))
    else:
        print(
            f"  [skipping QuantumSpecificMetrics: "
            f"{circuit.num_qubits} qubits exceeds unitary limit of {MAX_QUBITS_FOR_UNITARY}]"
        )

    return scanner.calculate_metrics()


def extract_counts_from_job(job) -> Dict[str, int]:
    """Extract measurement counts from a completed IBM Runtime job."""
    result = job.result()
    pub_result = result[0]
    for attr in ["c", "meas", "cr"]:
        if hasattr(pub_result.data, attr):
            bit_array = getattr(pub_result.data, attr)
            if hasattr(bit_array, "get_counts"):
                return dict(bit_array.get_counts())
    return {}


def infer_expected_outcomes(counts: Dict[str, int], threshold: float = 0.7) -> Optional[List[str]]:
    """Infer expected outcomes from counts if dominant outcome exceeds threshold.

    Args:
        counts: Measurement counts
        threshold: Minimum probability for an outcome to be considered "expected"

    Returns:
        List of expected bitstrings, or None if no outcome dominates
    """
    if not counts:
        return None
    total = sum(counts.values())
    dominant = max(counts, key=counts.get)
    if counts[dominant] / total >= threshold:
        return [dominant]
    return None


def compute_post_metrics(
    circuit: QuantumCircuit,
    job,
    expected_outcomes: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Post-runtime metrics (requires execution results).

    If expected_outcomes is not provided, attempts to infer from counts
    (dominant outcome > 70% probability).
    """
    if expected_outcomes is None:
        counts = extract_counts_from_job(job)
        expected_outcomes = infer_expected_outcomes(counts)
        if expected_outcomes:
            print(f"  [auto-detected expected_outcomes={expected_outcomes} from dominant count]")

    perf = FidelityMetrics(
        circuit=circuit,
        job=job,
        expected_outcomes=expected_outcomes,
    )
    scanner = Scanner(circuit=circuit)
    scanner.add_strategy(perf)
    return scanner.calculate_metrics()


def retrieve_and_analyze(
    job_id: str,
    expected_outcomes: Optional[List[str]] = None,
    show_counts: bool = True,
) -> Dict:
    """
    Main entry point: retrieve job by ID and compute all QWARD metrics.

    Args:
        job_id: IBM Quantum job ID
        expected_outcomes: Optional expected bitstrings for DSR calculation
        show_counts: Whether to print raw measurement counts

    Returns:
        Dictionary with keys: job_info, circuit, pre_metrics, post_metrics, counts
    """
    print(f"Retrieving job: {job_id}")
    service = get_service()
    job = service.job(job_id)

    job_info = {
        "job_id": job_id,
        "status": str(job.status()),
        "backend": job.backend().name,
        "primitive": job.primitive_id,
    }

    print(f"Status: {job_info['status']}")
    print(f"Backend: {job_info['backend']}")
    print(f"Primitive: {job_info['primitive']}")

    # Extract circuit
    print("\nExtracting circuit from job inputs...")
    isa_circuit = extract_circuit_from_job(job)
    print(f"ISA circuit: {isa_circuit.num_qubits} qubits, depth {isa_circuit.depth()}")

    # Trim idle qubits to get the actual computation
    circuit = trim_idle_qubits(isa_circuit)
    print(f"Trimmed circuit: {circuit.num_qubits} qubits, depth {circuit.depth()}")

    result = {
        "job_info": job_info,
        "circuit": circuit,
        "pre_metrics": None,
        "post_metrics": None,
        "counts": None,
    }

    # Pre-runtime metrics
    print("\n" + "=" * 60)
    print("PRE-RUNTIME METRICS (circuit structure)")
    print("=" * 60)
    pre_metrics = compute_pre_metrics(circuit)
    result["pre_metrics"] = pre_metrics
    for name, df in pre_metrics.items():
        print(f"\n--- {name} ---")
        print(df.to_string())

    # Post-runtime metrics
    print("\n" + "=" * 60)
    print("POST-RUNTIME METRICS (execution results)")
    print("=" * 60)
    try:
        post_metrics = compute_post_metrics(circuit, job, expected_outcomes)
        result["post_metrics"] = post_metrics
        for name, df in post_metrics.items():
            print(f"\n--- {name} ---")
            print(df.to_string())
    except Exception as e:
        print(f"Error computing post metrics: {e}")

    # Raw counts
    if show_counts:
        print("\n" + "=" * 60)
        print("RAW COUNTS")
        print("=" * 60)
        try:
            job_result = job.result()
            pub_result = job_result[0]

            counts = None
            for attr in ["c", "meas", "cr"]:
                if hasattr(pub_result.data, attr):
                    bit_array = getattr(pub_result.data, attr)
                    if hasattr(bit_array, "get_counts"):
                        counts = dict(bit_array.get_counts())
                        break

            if counts:
                result["counts"] = counts
                total = sum(counts.values())
                print(f"Total shots: {total}")
                print(f"Unique outcomes: {len(counts)}")
                sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
                print("Top 10 outcomes:")
                for outcome, count in sorted_counts:
                    pct = (count / total) * 100
                    print(f"  |{outcome}>: {count} ({pct:.1f}%)")
        except Exception as e:
            print(f"Error getting counts: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve IBM Quantum job and compute QWARD metrics"
    )
    parser.add_argument("--job-id", required=True, help="IBM Quantum job ID")
    parser.add_argument(
        "--expected-outcomes",
        nargs="*",
        default=None,
        help="Expected bitstrings for DSR (e.g. 00 101)",
    )
    parser.add_argument("--no-counts", action="store_true", help="Skip printing raw counts")
    args = parser.parse_args()

    retrieve_and_analyze(
        job_id=args.job_id,
        expected_outcomes=args.expected_outcomes,
        show_counts=not args.no_counts,
    )


if __name__ == "__main__":
    main()
