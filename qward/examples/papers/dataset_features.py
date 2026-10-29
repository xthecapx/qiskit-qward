"""Training-dataset feature and target groups for one QPU job.

Groups (see docs/ibm_fez_dataset_plan.md):

    A  logical circuit    QWARD pre-runtime metrics (+ QuantumSpecific)   prepare
    B  ISA circuit        same metrics on the executed circuit, layout     prepare
    C  hardware @ submit  used-qubit calibration, device medians           prepare
    D  hardware @ run     same as C from job.properties()                  recover
    E  baselines          M1 / M1-mixture / M2 error predictions           prepare
    F  targets            FidelityMetrics + HistogramAssessment            recover
    G  provenance         IDs, versions, seeds, options, timestamps        record

Every group is a dict. A group that cannot be computed is
``{"available": False, "reason": "..."}`` instead of zeros, so a model can
tell *missing* from *measured as zero*.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from qiskit import QuantumCircuit, QuantumRegister

from qward.examples.papers.experiment_helpers import calculate_qward_metrics
from qward.metrics import BackendCalibrationCollector, QuantumSpecificMetrics
from qward.metrics.circuit_error_prediction import (
    CONTROL_FLOW,
    CircuitErrorPrediction,
    circuit_hash,
    is_dynamic,
    snapshot_at_properties,
)
from qward.metrics.fidelity_metrics import FidelityMetrics
from qward.metrics.histogram_assessment import HistogramAssessment

FEATURES_SCHEMA_VERSION = "1.0"
QUANTUM_SPECIFIC_MAX_QUBITS = 8
QUANTUM_SPECIFIC_SEED = 11
QUANTUM_SPECIFIC_CACHE = Path(__file__).resolve().parent / "data" / "quantum_specific_cache"
MAX_REFERENCE_BITS = 12
MAX_STATEVECTOR_QUBITS = 20
DYNAMIC_REFERENCE_SHOTS = 200_000
REFERENCE_FLOOR = 1e-9


def missing(reason: str) -> Dict[str, Any]:
    """Placeholder for a group that was not computed."""
    return {"available": False, "reason": reason}


def _safe(group: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    try:
        return {"available": True, **group()}
    except Exception as exc:  # pylint: disable=broad-except
        return missing(f"error: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def measured_labels(circuit: QuantumCircuit) -> list:
    """All output labels in Qiskit count format (registers space-separated)."""
    sizes = [reg.size for reg in reversed(circuit.cregs)]
    width = sum(sizes)
    if width > MAX_REFERENCE_BITS:
        raise ValueError(f"Reference enumeration limited to {MAX_REFERENCE_BITS} bits")
    labels = []
    for value in range(2**width):
        bits, parts, start = format(value, f"0{width}b"), [], 0
        for size in sizes:
            parts.append(bits[start : start + size])
            start += size
        labels.append(" ".join(parts))
    return labels


def _format_labels(probabilities: Mapping[str, float], circuit: QuantumCircuit) -> Dict[str, float]:
    """Plain bitstrings (clbit 0 rightmost) to Qiskit count labels."""
    sizes = [reg.size for reg in reversed(circuit.cregs)]
    out: Dict[str, float] = {}
    for bits, value in probabilities.items():
        bits = str(bits)
        if value <= REFERENCE_FLOOR:
            continue
        parts, start = [], 0
        for size in sizes:
            parts.append(bits[start : start + size])
            start += size
        label = " ".join(parts)
        out[label] = out.get(label, 0.0) + float(value)
    total = sum(out.values())
    return {label: value / total for label, value in out.items()}


def ideal_reference(circuit: QuantumCircuit) -> Dict[str, float]:
    """Exact noiseless output distribution in count-label format.

    Static circuits whose measurements are terminal: statevector marginal over
    measured qubits (up to MAX_STATEVECTOR_QUBITS). Dynamic circuits or
    mid-circuit measurements: noiseless Aer sampling with
    DYNAMIC_REFERENCE_SHOTS shots (estimated, not exact; flagged by callers).
    """
    from qiskit.quantum_info import Statevector  # pylint: disable=import-outside-toplevel

    if circuit.num_clbits > MAX_REFERENCE_BITS:
        raise ValueError(
            f"No tractable ideal distribution: {circuit.num_clbits} measured bits "
            f"(limit {MAX_REFERENCE_BITS})"
        )
    measures = [i for i in circuit.data if i.operation.name == "measure"]
    clbit_of = {
        circuit.find_bit(i.clbits[0]).index: circuit.find_bit(i.qubits[0]).index for i in measures
    }
    terminal = _measurements_terminal(circuit)
    if (
        not is_dynamic(circuit)
        and terminal
        and len(clbit_of) == circuit.num_clbits
        and circuit.num_qubits <= MAX_STATEVECTOR_QUBITS
    ):
        body = circuit.remove_final_measurements(inplace=False)
        qargs = [clbit_of[c] for c in range(circuit.num_clbits)]
        return _format_labels(Statevector(body).probabilities_dict(qargs=qargs), circuit)
    from qiskit import transpile  # pylint: disable=import-outside-toplevel
    from qiskit_aer import AerSimulator  # pylint: disable=import-outside-toplevel

    simulator = AerSimulator(method="statevector")
    counts = (
        simulator.run(
            transpile(circuit, simulator), shots=DYNAMIC_REFERENCE_SHOTS, seed_simulator=11
        )
        .result()
        .get_counts()
    )
    total = sum(counts.values())
    return {label: count / total for label, count in counts.items()}


def reference_is_exact(circuit: QuantumCircuit) -> bool:
    """True when ``ideal_reference`` uses the exact statevector path."""
    return (
        not is_dynamic(circuit)
        and _measurements_terminal(circuit)
        and circuit.num_qubits <= MAX_STATEVECTOR_QUBITS
    )


def _measurements_terminal(circuit: QuantumCircuit) -> bool:
    """No gate acts on a qubit after it has been measured."""
    measured = set()
    for item in circuit.data:
        qubits = {circuit.find_bit(q).index for q in item.qubits}
        if item.operation.name == "measure":
            measured |= qubits
        elif item.operation.name != "barrier" and qubits & measured:
            return False
    return True


def success_labels(circuit: QuantumCircuit, success: Callable[[str], bool]) -> list:
    """Count labels accepted by a runner's success criteria."""
    return [label for label in measured_labels(circuit) if success(label)]


def reference_from_success(
    circuit: QuantumCircuit, success: Callable[[str], bool]
) -> Dict[str, float]:
    """Uniform reference over the success set (fallback when no ideal exists)."""
    winners = [label for label in measured_labels(circuit) if success(label)]
    if not winners:
        raise ValueError("Success criteria accept no output label")
    return {label: 1.0 / len(winners) for label in winners}


# ---------------------------------------------------------------------------
# Group A/B: circuit metrics
# ---------------------------------------------------------------------------


def compact_isa(isa: QuantumCircuit) -> QuantumCircuit:
    """ISA circuit restricted to qubits that carry operations, without delays.

    QWARD metrics expect a plain circuit; the device-wide ISA (156 qubits, a
    TranspileLayout and scheduling delays) is reduced to the active wires in
    physical order. Gate sequence, depth and 2Q count are preserved.
    """
    active = sorted(
        {
            isa.find_bit(q).index
            for item in isa.data
            if item.operation.name not in {"barrier", "delay"}
            for q in item.qubits
        }
    )
    position = {physical: i for i, physical in enumerate(active)}
    out = QuantumCircuit(QuantumRegister(len(active), "q"), *isa.cregs, name=isa.name)
    for item in isa.data:
        if item.operation.name == "delay":
            continue
        qubits = [isa.find_bit(q).index for q in item.qubits]
        if any(q not in position for q in qubits):
            continue  # Barrier touching idle wires only.
        out.append(
            item.operation,
            [out.qubits[position[q]] for q in qubits],
            [out.clbits[isa.find_bit(c).index] for c in item.clbits],
        )
    return out


def _qward(circuit: QuantumCircuit) -> Dict[str, Any]:
    metrics = calculate_qward_metrics(circuit)
    if "error" in metrics:
        raise RuntimeError(metrics["error"])
    return metrics


def _quantum_specific_key(circuit: QuantumCircuit) -> str:
    """Cache key: logical circuit, seed, and the metric's own source code."""
    import torch  # pylint: disable=import-outside-toplevel
    from qward.metrics import quantum_specific_metrics  # pylint: disable=import-outside-toplevel

    source = hashlib.sha256(inspect.getsource(quantum_specific_metrics).encode()).hexdigest()
    payload = f"{circuit_hash(circuit)}|{QUANTUM_SPECIFIC_SEED}|{source}|torch{torch.__version__}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


_DEFAULT_CACHE = object()


def quantum_specific(circuit: QuantumCircuit, *, cache_dir: Any = _DEFAULT_CACHE) -> Dict[str, Any]:
    """QuantumSpecificMetrics with explicit size/dynamic limits.

    The metric depends only on the logical circuit and takes up to ~75 s at
    8 qubits, so results are cached per circuit (``cache_dir=None`` disables).
    A cache hit is marked ``cached: true``; ``compute_seconds`` stays the
    original cost.
    """
    if is_dynamic(circuit):
        return missing("dynamic_circuit")
    if circuit.num_qubits > QUANTUM_SPECIFIC_MAX_QUBITS:
        return missing(f"size_limit>{QUANTUM_SPECIFIC_MAX_QUBITS}_qubits")
    if cache_dir is _DEFAULT_CACHE:
        cache_dir = QUANTUM_SPECIFIC_CACHE  # Read at call time so tests can redirect it.
    entry = None
    if cache_dir is not None:
        entry = Path(cache_dir) / f"{_quantum_specific_key(circuit)}.json"
        if entry.exists():
            return {**json.loads(entry.read_text(encoding="utf-8")), "cached": True}

    def compute():
        # The metric optimizes from random starts; seed it so one circuit always
        # yields the same features, then restore the caller's RNG state.
        import torch  # pylint: disable=import-outside-toplevel

        state = torch.get_rng_state()
        torch.manual_seed(QUANTUM_SPECIFIC_SEED)
        try:
            start = time.perf_counter()
            values = QuantumSpecificMetrics(circuit).get_metrics().model_dump()
        finally:
            torch.set_rng_state(state)
        return {
            **values,
            "seed": QUANTUM_SPECIFIC_SEED,
            "compute_seconds": round(time.perf_counter() - start, 3),
        }

    result = _safe(compute)
    if entry is not None and result.get("available"):
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**result, "cached": False}


def logical_features(
    circuit: QuantumCircuit, *, include_quantum_specific: bool = True
) -> Dict[str, Any]:
    """Group A."""
    group = _safe(lambda: {"qward_metrics": _qward(circuit)})
    group["num_qubits"] = circuit.num_qubits
    group["num_clbits"] = circuit.num_clbits
    group["depth"] = circuit.depth()
    group["size"] = circuit.size()
    group["dynamic"] = is_dynamic(circuit)
    group["quantum_specific"] = (
        quantum_specific(circuit) if include_quantum_specific else missing("disabled")
    )
    return group


def isa_features(isa: QuantumCircuit) -> Dict[str, Any]:
    """Group B."""
    compact = compact_isa(isa)
    group = _safe(lambda: {"qward_metrics": _qward(compact)})
    layout = isa.layout.final_index_layout(filter_ancillas=True) if isa.layout else None
    ops = dict(compact.count_ops())
    group.update(
        {
            "physical_layout": layout,
            "active_qubits": compact.num_qubits,
            "depth": compact.depth(),
            "size": compact.size(),
            "two_qubit_gates": sum(len(i.qubits) == 2 for i in compact.data),
            "two_qubit_depth": compact.depth(lambda i: len(i.qubits) == 2),
            "count_ops": ops,
        }
    )
    return group


# ---------------------------------------------------------------------------
# Group C/D: hardware
# ---------------------------------------------------------------------------


def _stats(values) -> Dict[str, Optional[float]]:
    clean = sorted(v for v in values if v is not None and math.isfinite(v))
    if not clean:
        return {"mean": None, "max": None, "min": None, "median": None}
    middle = len(clean) // 2
    median = clean[middle] if len(clean) % 2 else (clean[middle - 1] + clean[middle]) / 2
    return {"mean": sum(clean) / len(clean), "max": clean[-1], "min": clean[0], "median": median}


def hardware_features(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    """Group C or D from a calibration snapshot of the exact ISA circuit."""
    # IBM publishes no error for control flow; it is counted, not treated as missing.
    control_flow = [e for e in snapshot["instructions"] if e["name"] in CONTROL_FLOW]
    instructions = [
        e
        for e in snapshot["instructions"]
        if e["name"] not in {"delay", "barrier"} and e["name"] not in CONTROL_FLOW
    ]
    one_q = [e["error"] for e in instructions if len(e["qubits"]) == 1 and e["name"] != "measure"]
    two_q = [e["error"] for e in instructions if len(e["qubits"]) == 2]
    readout = [e["error"] for e in instructions if e["name"] == "measure"]
    qubits = list(snapshot["qubits"].values())
    known = [e for e in instructions if e["error"] is not None]

    def per_qubit(key, scale=1.0):
        return _stats(q[key] * scale for q in qubits if q.get(key) is not None)

    idle_t1 = [
        q["idle_s"] / q["t1_s"] for q in qubits if q.get("idle_s") is not None and q.get("t1_s")
    ]
    return {
        "available": True,
        "calibrated_at": snapshot.get("calibrated_at"),
        "collected_at": snapshot.get("collected_at"),
        "source": snapshot.get("source", "prepare_snapshot"),
        "calibration_coverage": len(known) / len(instructions) if instructions else 0.0,
        "uncalibrated_control_flow_ops": len(control_flow),
        "one_qubit_gate_error": _stats(one_q),
        "two_qubit_gate_error": _stats(two_q),
        "readout_error": _stats(readout),
        "readout_p1_given_0": per_qubit("prob_meas1_prep0"),
        "readout_p0_given_1": per_qubit("prob_meas0_prep1"),
        "t1_us": per_qubit("t1_s", 1e6),
        "t2_us": per_qubit("t2_s", 1e6),
        "idle_over_t1": _stats(idle_t1),
        "scheduled": snapshot.get("scheduled"),
        "duration_s": (
            max(((e["start_s"] or 0) + (e["duration_s"] or 0) for e in instructions), default=0)
            if snapshot.get("scheduled")
            else None
        ),
        "num_used_qubits": len(qubits),
    }


def device_features(backend: Any) -> Dict[str, Any]:
    """Device-wide medians at prepare time (context for group C)."""
    return _safe(lambda: BackendCalibrationCollector(backend).get_metrics().model_dump())


# ---------------------------------------------------------------------------
# Group E: baseline predictions
# ---------------------------------------------------------------------------


def baseline_predictions(
    isa: QuantumCircuit,
    snapshot: Mapping[str, Any],
    reference: Optional[Mapping[str, float]],
    *,
    success_set: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Group E. Static circuits with a known ideal distribution only.

    Comparable with the targets of group F:

    * ``*_tvd``: predicted TVD(predicted distribution, ideal), vs observed TVD.
    * ``*_error``: predicted 1 - P(success set), vs observed 1 - success_rate.
      M1 uses Q as P(no error) and adds the ideal's own mass on the success
      set; mixture/M2 sum their predicted distributions over the set.
    """
    if is_dynamic(isa):
        return missing("dynamic_circuit")
    if not reference:
        return missing("no_reference")
    flat = {k.replace(" ", ""): v for k, v in reference.items() if v > REFERENCE_FLOOR}
    total = sum(flat.values())
    flat = {k: v / total for k, v in flat.items()}
    winners = [k.replace(" ", "") for k in (success_set or flat)]

    def mass(distribution):
        if distribution is None:
            return None
        return sum(distribution.get(k, 0.0) for k in winners)

    def compute():
        values = CircuitErrorPrediction(isa, snapshot=snapshot, reference=flat).get_metrics()
        q = values.esp_score
        ideal_success = mass(flat)
        mixture_success = mass(values.mixture_probabilities)
        m2_success = mass(values.m2_probabilities)
        if m2_success is None and len(flat) == 1:
            m2_success = values.m2_success  # >12 bits: single-target point value.
        return {
            "esp_score": q,
            "m1_error": 1 - q * ideal_success if q is not None else None,
            "m1_tvd": (1 - q) if q is not None else None,
            "m1_mixture_error": 1 - mixture_success if mixture_success is not None else None,
            "m1_mixture_tvd": values.mixture_tvd,
            "m2_error": 1 - m2_success if m2_success is not None else None,
            "m2_tvd": values.m2_tvd,
            "m2_polarization": values.m2_polarization,
            "ideal_success_mass": ideal_success,
            "calibration_coverage": values.calibration_coverage,
        }

    return _safe(compute)


# ---------------------------------------------------------------------------
# Group F: targets
# ---------------------------------------------------------------------------


def targets(
    counts: Mapping[str, int],
    circuit: QuantumCircuit,
    reference: Optional[Mapping[str, float]],
    *,
    success: Optional[Callable[[str], bool]] = None,
    alpha: float = 0.01,
) -> Dict[str, Any]:
    """Group F: one FidelityMetrics pass + HistogramAssessment against the reference.

    ``reference`` is the ideal distribution (Hellinger/TVD target). The
    success set (DSR, success_rate) is the labels accepted by ``success``;
    without it, the reference support is used.
    """
    if not reference:
        return missing("no_reference")
    width = sum(reg.size for reg in circuit.cregs)
    # Multi-register labels ("00 01") are compared as plain bitstrings.
    flat_counts: Dict[str, int] = {}
    for label, count in counts.items():
        key = label.replace(" ", "")
        flat_counts[key] = flat_counts.get(key, 0) + count
    flat_reference = {k.replace(" ", ""): v for k, v in reference.items()}
    if success is not None:
        expected = [label.replace(" ", "") for label in success_labels(circuit, success)]
    else:
        expected = [label for label, p in flat_reference.items() if p > 0]
    if not expected:
        return missing("empty_success_set")
    flat_circuit = QuantumCircuit(circuit.num_qubits, width)

    def fidelity():
        values = FidelityMetrics(
            flat_circuit,
            counts=flat_counts,
            expected_outcomes=expected,
            target_histogram=flat_reference,
            num_measured_qubits=width,
        ).get_metrics()
        return values.model_dump(exclude_none=True, exclude={"expected_outcomes"})

    def assessment():
        values = HistogramAssessment(
            flat_circuit, counts=flat_counts, reference=flat_reference, alpha=alpha
        ).get_metrics()
        return values.model_dump(exclude={"error_budget", "m2_error_budget"})

    return {
        "available": True,
        "shots": sum(counts.values()),
        "fidelity": _safe(fidelity),
        "histogram_assessment": _safe(assessment),
    }


def execution_hardware(
    job: Any,
    snapshot: Mapping[str, Any],
    isa: QuantumCircuit,
    reference,
    *,
    success_set: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Group D (+ E re-predicted) from the calibration the job ran under."""
    properties = job.properties()
    if properties is None:
        return {"hardware_at_execution": missing("job.properties() returned None")}
    late = snapshot_at_properties(snapshot, properties)
    group = hardware_features(late)
    group["source"] = "job.properties()"
    group["changed_since_prepare"] = late["calibrated_at"] != snapshot.get("calibrated_at")
    return {
        "hardware_at_execution": group,
        "baselines_at_execution": baseline_predictions(
            isa, late, reference, success_set=success_set
        ),
    }
