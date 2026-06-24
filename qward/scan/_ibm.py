"""IBM Quantum job/batch scan functions."""

import io
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import Barrier, Measure
from qiskit.qpy import load

from qward.scan._core import scan_post, scan_pre


def get_ibm_service(service=None):
    """Get or create QiskitRuntimeService instance.

    Args:
        service: Existing service instance, or None to create from .env.

    Returns:
        QiskitRuntimeService instance.
    """
    if service is not None:
        return service

    from dotenv import load_dotenv
    from qiskit_ibm_runtime import QiskitRuntimeService

    load_dotenv()

    kwargs = {}
    channel = os.getenv("IBM_QUANTUM_CHANNEL")
    token = os.getenv("IBM_QUANTUM_TOKEN")
    instance = os.getenv("IBM_QUANTUM_INSTANCE")

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
    """Remove idle qubits from an ISA circuit.

    ISA circuits map to the full device topology (e.g. 156 qubits on ibm_fez)
    but typically only a few qubits have actual gates.
    """
    active_qubit_indices = set()
    for instruction in circuit.data:
        gate = instruction[0] if isinstance(instruction, tuple) else instruction.operation
        qubits = instruction[1] if isinstance(instruction, tuple) else instruction.qubits

        if isinstance(gate, Barrier):
            continue

        for qubit in qubits:
            idx = circuit.qubits.index(qubit)
            active_qubit_indices.add(idx)

    sorted_indices = sorted(active_qubit_indices)

    if not sorted_indices:
        raise ValueError("No active qubits found in circuit")

    num_active = len(sorted_indices)
    num_measurements = sum(
        1
        for inst in circuit.data
        if isinstance(inst[0] if isinstance(inst, tuple) else inst.operation, Measure)
    )
    num_clbits = max(num_measurements, circuit.num_clbits) if num_measurements > 0 else 0

    trimmed = QuantumCircuit(num_active, min(num_clbits, num_active))
    qubit_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}

    clbit_counter = 0
    for instruction in circuit.data:
        gate = instruction[0] if isinstance(instruction, tuple) else instruction.operation
        qubits = instruction[1] if isinstance(instruction, tuple) else instruction.qubits

        if isinstance(gate, Barrier):
            continue

        qubit_indices = [circuit.qubits.index(q) for q in qubits]

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


def detect_primitive_type_from_job(job) -> str:
    """Detect whether a completed job produced Sampler or Estimator results."""
    try:
        result = job.result()
        pub_result = result[0]
        if hasattr(pub_result, "data") and hasattr(pub_result.data, "evs"):
            return "estimator"
    except Exception:
        pass
    return "sampler"


def extract_estimator_from_job(job) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract expectation values and standard deviations from an Estimator job."""
    result = job.result()
    pub_result = result[0]
    evs = np.atleast_1d(np.asarray(pub_result.data.evs, dtype=float))
    stds = None
    if hasattr(pub_result.data, "stds"):
        stds = np.atleast_1d(np.asarray(pub_result.data.stds, dtype=float))
    return evs, stds


def scan_job(
    job_id: str,
    *,
    expected_outcomes: Optional[List[str]] = None,
    target_histogram: Optional[Dict[str, float]] = None,
    target_state: Optional[str] = None,
    ideal_expectation_values: Optional[np.ndarray] = None,
    observable_labels: Optional[List[str]] = None,
    service=None,
    trim_idle: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Scan an IBM Quantum job by ID — compute pre + post metrics.

    Auto-detects primitive type (Sampler vs Estimator) and computes
    appropriate post-runtime metrics.

    Args:
        job_id: IBM Quantum job ID.
        expected_outcomes: Expected bitstrings for DSR/success_rate (Sampler).
        target_histogram: Ideal probability distribution for HF/TVD (Sampler).
        target_state: Shortcut — sets both expected_outcomes and target_histogram.
        ideal_expectation_values: Ideal values for fidelity computation (Estimator).
        observable_labels: Human-readable labels for observables (Estimator).
        service: QiskitRuntimeService instance (or None to create from .env).
        trim_idle: Remove idle qubits from ISA circuit before analysis.

    Returns:
        Dict mapping metric names to DataFrames. Includes pre-runtime metrics
        and "FidelityMetrics" for post-runtime fidelity.
    """
    svc = get_ibm_service(service)
    job = svc.job(job_id)

    circuit = extract_circuit_from_job(job)
    if trim_idle:
        circuit = trim_idle_qubits(circuit)

    results = scan_pre(circuit)

    ptype = detect_primitive_type_from_job(job)
    if ptype == "estimator":
        evs, stds = extract_estimator_from_job(job)
        post = scan_post(
            circuit,
            expectation_values=evs,
            standard_deviations=stds,
            ideal_expectation_values=ideal_expectation_values,
            observable_labels=observable_labels,
        )
        results.update(post)
    else:
        counts = extract_counts_from_job(job)
        if counts:
            post = scan_post(
                circuit,
                counts,
                expected_outcomes=expected_outcomes,
                target_histogram=target_histogram,
                target_state=target_state,
            )
            results.update(post)

    meta = pd.DataFrame(
        [
            {
                "job_id": job_id,
                "backend": (
                    job.backend().name if hasattr(job.backend(), "name") else str(job.backend())
                ),
                "status": str(job.status()),
                "num_qubits": circuit.num_qubits,
                "depth": circuit.depth(),
                "primitive_type": ptype,
            }
        ]
    )
    results["_meta"] = meta

    return results


def scan_batch(
    batch_id: str,
    *,
    expected_outcomes: Optional[List[str]] = None,
    target_histogram: Optional[Dict[str, float]] = None,
    target_state: Optional[str] = None,
    ideal_expectation_values: Optional[np.ndarray] = None,
    observable_labels: Optional[List[str]] = None,
    service=None,
    aggregation: str = "last",
    trim_idle: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Scan an IBM Quantum batch — compute pre + aggregated post metrics.

    Auto-detects primitive type (Sampler vs Estimator) from job results.

    Args:
        batch_id: IBM batch/session ID.
        expected_outcomes: Expected bitstrings for DSR/success_rate (Sampler).
        target_histogram: Ideal probability distribution for HF/TVD (Sampler).
        target_state: Shortcut — sets both expected_outcomes and target_histogram.
        ideal_expectation_values: Ideal values for fidelity (Estimator).
        observable_labels: Human-readable labels for observables (Estimator).
        service: QiskitRuntimeService instance (or None to create from .env).
        aggregation: How to combine results across jobs:
            - "last": Use last completed job's results.
            - "merge": Sampler: sum counts. Estimator: average evs.
        trim_idle: Remove idle qubits from ISA circuit.

    Returns:
        Dict mapping metric names to DataFrames.
    """
    if aggregation not in ("last", "merge"):
        raise ValueError(f"aggregation must be 'last' or 'merge', got '{aggregation}'")

    svc = get_ibm_service(service)
    jobs = svc.jobs(session_id=batch_id, limit=None)

    if not jobs:
        raise ValueError(f"No jobs found for batch_id '{batch_id}'")

    circuit = extract_circuit_from_job(jobs[0])
    if trim_idle:
        circuit = trim_idle_qubits(circuit)

    results = scan_pre(circuit)

    completed_jobs = [j for j in jobs if str(j.status()).upper() in ("DONE", "COMPLETED")]

    ptype = "sampler"
    if completed_jobs:
        ptype = detect_primitive_type_from_job(completed_jobs[0])

        if ptype == "estimator":
            if aggregation == "last":
                evs, stds = extract_estimator_from_job(completed_jobs[0])
            else:
                all_evs = []
                all_stds = []
                for j in completed_jobs:
                    e, s = extract_estimator_from_job(j)
                    all_evs.append(e)
                    if s is not None:
                        all_stds.append(s)
                evs = np.mean(all_evs, axis=0)
                stds = np.mean(all_stds, axis=0) if all_stds else None

            post = scan_post(
                circuit,
                expectation_values=evs,
                standard_deviations=stds,
                ideal_expectation_values=ideal_expectation_values,
                observable_labels=observable_labels,
            )
            results.update(post)
        else:
            if aggregation == "last":
                counts = extract_counts_from_job(completed_jobs[0])
            else:
                merged: Dict[str, int] = {}
                for j in completed_jobs:
                    for k, v in extract_counts_from_job(j).items():
                        merged[k] = merged.get(k, 0) + v
                counts = merged

            if counts:
                post = scan_post(
                    circuit,
                    counts,
                    expected_outcomes=expected_outcomes,
                    target_histogram=target_histogram,
                    target_state=target_state,
                )
                results.update(post)

    meta_rows = []
    for j in jobs:
        meta_rows.append(
            {
                "job_id": j.job_id,
                "status": str(j.status()),
                "backend": j.backend().name if hasattr(j.backend(), "name") else str(j.backend()),
                "primitive_type": ptype,
            }
        )
    results["_meta"] = pd.DataFrame(meta_rows)

    return results
