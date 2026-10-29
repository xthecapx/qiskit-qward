"""Calibration-based experimental M1 baselines, without quantum simulation.

The full snapshot is separate from the existing backend-median collector.
It retains instruction occurrences, physical operands and missing calibrations.
Only small analytic distributions are materialized; deterministic predictions
remain scalable when a full output histogram would be exponentially large.
"""

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Set, Tuple

from qiskit import QuantumCircuit

from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsId, MetricsType
from qward.schemas.error_prediction_schema import ErrorPredictionSchema


def record_hash(record: Mapping[str, Any]) -> str:
    """Hash a JSON record using a stable canonical representation."""
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def circuit_hash(circuit: QuantumCircuit) -> str:
    """Fingerprint static instruction semantics including physical/classical operands."""
    instructions = []
    for item in circuit.data:
        if item.operation.name in {"if_else", "while_loop", "for_loop", "switch_case"}:
            raise ValueError("Prediction currently requires a static circuit")
        instructions.append(
            {
                "name": item.operation.name,
                "qubits": [circuit.find_bit(q).index for q in item.qubits],
                "clbits": [circuit.find_bit(c).index for c in item.clbits],
                "params": [str(p) for p in item.operation.params],
                "unit": getattr(item.operation, "unit", None),
            }
        )
    return record_hash(
        {"qubits": circuit.num_qubits, "clbits": circuit.num_clbits, "ops": instructions}
    )


def validate_distribution(reference: Mapping[str, float], width: int) -> Dict[str, float]:
    """Validate an analytic output reference without silently renormalizing it."""
    if not reference or width < 1:
        raise ValueError("A nonempty reference and positive output width are required")
    for key, value in reference.items():
        if len(key) != width or set(key) - {"0", "1"}:
            raise ValueError("Reference labels must be output-width binary strings")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Reference probabilities must be finite and in [0, 1]")
    if not math.isclose(sum(reference.values()), 1, abs_tol=1e-10):
        raise ValueError("Reference probabilities must sum to one")
    return {key: float(value) for key, value in reference.items() if value > 1e-14}


def flip_distribution(
    reference: Mapping[str, float], flips: Mapping[int, Tuple[float, float]], width: int
) -> Dict[str, float]:
    """Apply independent readout flips; clbit 0 is the rightmost label bit.

    ``flips[clbit] = (P(read 1 | prepared 0), P(read 0 | prepared 1))``.
    """
    result = dict(reference)
    for clbit, rates in flips.items():
        position = width - 1 - clbit
        updated: Dict[str, float] = {}
        for key, value in result.items():
            bit = key[position]
            rate = rates[0] if bit == "0" else rates[1]
            flipped = key[:position] + ("1" if bit == "0" else "0") + key[position + 1 :]
            updated[key] = updated.get(key, 0) + value * (1 - rate)
            updated[flipped] = updated.get(flipped, 0) + value * rate
        result = updated
    return result


def depolarizing_prediction(
    snapshot: Mapping[str, Any], reference: Mapping[str, float], width: int
) -> Dict[str, Any]:
    """M2: gate infidelities as global depolarization, readout as independent bit flips.

    A d-dimensional depolarizing channel with average infidelity r has
    depolarizing probability p = r*d/(d-1). The circuit polarization is the
    product of (1 - p) over native gates. Readout uses the backend's
    directional assignment errors when the snapshot has them (IBM
    prob_meas1_prep0 / prob_meas0_prep1), else the symmetric average. Hence
    P(x) = lambda * A(reference)(x) + (1 - lambda) * A(uniform)(x).
    Returns None-valued fields when any instruction or measurement lacks data.
    """
    polarization = 1.0
    flips: Dict[int, Tuple[float, float]] = {}
    qubits = snapshot.get("qubits", {})
    complete = True
    for entry in snapshot["instructions"]:
        if entry["name"] == "delay":
            continue
        error = entry["error"]
        if error is None:
            complete = False
            continue
        if entry["name"] == "measure":
            directional = qubits.get(str(entry["qubits"][0]), {})
            up, down = directional.get("prob_meas1_prep0"), directional.get("prob_meas0_prep1")
            rates = (up, down) if up is not None and down is not None else (error, error)
            for clbit in entry["clbits"]:
                flips[clbit] = rates
            continue
        dimension = 2 ** len(entry["qubits"])
        polarization *= max(0.0, 1 - error * dimension / (dimension - 1))
    if not complete or sorted(flips) != list(range(width)):
        return {"polarization": None, "success": None, "tvd": None, "probabilities": None}
    uniform = math.ldexp(1.0, -width)
    deterministic = len(reference) == 1
    success = None
    if deterministic:
        target = next(iter(reference))
        # A bit reads correctly with 1 - P(flip | its value); uniform input
        # reads as t with probability prod((1 - up - down) / 2 + (bit's in-rate)).
        readout_success = 1.0
        scrambled_success = 1.0
        for clbit, (up, down) in flips.items():
            bit = target[width - 1 - clbit]
            readout_success *= 1 - (up if bit == "0" else down)
            scrambled_success *= 0.5 * ((1 - up + down) if bit == "0" else (1 - down + up))
        success = polarization * readout_success + (1 - polarization) * scrambled_success
    probabilities = None
    tvd = None
    if width <= 12:
        flipped = flip_distribution(reference, flips, width)
        noisy_uniform = flip_distribution(
            {f"{x:0{width}b}": uniform for x in range(2**width)}, flips, width
        )
        probabilities = {
            f"{x:0{width}b}": polarization * flipped.get(f"{x:0{width}b}", 0)
            + (1 - polarization) * noisy_uniform.get(f"{x:0{width}b}", 0)
            for x in range(2**width)
        }
        tvd = 0.5 * sum(abs(probabilities[key] - reference.get(key, 0)) for key in probabilities)
    elif deterministic:
        tvd = 1 - success
    return {
        "polarization": polarization,
        "success": success,
        "tvd": min(1.0, max(0.0, tvd)) if tvd is not None else None,
        "probabilities": probabilities,
    }


def _add_directional_readout(qubit_properties: Dict[str, Dict], raw: Any) -> None:
    """Record IBM's P(1|0) and P(0|1) assignment errors when reported."""
    for qubit, values in qubit_properties.items():
        for name in ("prob_meas1_prep0", "prob_meas0_prep1"):
            try:
                values[name] = raw.qubit_property(int(qubit), name)[0]
            except Exception:  # pylint: disable=broad-except
                values[name] = None


def capture_calibration(circuit: QuantumCircuit, backend: Any) -> Dict[str, Any]:
    """Capture properties for the exact ISA circuit; never submit or simulate it.

    Collection time and backend calibration time are deliberately distinct.
    When available, the backend's complete properties record preserves individual
    calibration timestamps. Target-only backends leave those timestamps unknown.
    """
    target = backend.target
    dt = target.dt
    try:
        starts = circuit.op_start_times
    except AttributeError:
        starts = None
    entries = []
    # ALAP can pad the backend's completely unused wires with delays. They
    # belong in the archived instruction stream, not in the circuit's exposure.
    active: Set[int] = {
        circuit.find_bit(qubit).index
        for item in circuit.data
        if item.operation.name not in {"barrier", "delay"}
        for qubit in item.qubits
    }
    measured: Set[int] = set()
    last_end: Dict[int, float] = {}
    idle: Dict[int, float] = {}
    for index, item in enumerate(circuit.data):
        name = item.operation.name
        qubits = tuple(circuit.find_bit(q).index for q in item.qubits)
        if name == "barrier":
            continue
        props = target[name].get(qubits) if name in target.operation_names else None
        duration = props.duration if props is not None else None
        if name == "delay":
            unit = item.operation.unit
            scale = {"s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}.get(unit)
            if unit == "dt":
                scale = dt
            duration = float(item.operation.params[0]) * scale if scale is not None else None
        entry = {
            "index": index,
            "name": name,
            "qubits": list(qubits),
            "clbits": [circuit.find_bit(c).index for c in item.clbits],
            "error": props.error if props is not None else None,
            "duration_s": duration,
            "start_s": starts[index] * dt if starts is not None and dt is not None else None,
        }
        entries.append(entry)
        if entry["start_s"] is not None and duration is not None:
            for qubit in qubits:
                if qubit not in active or qubit in measured:
                    continue
                idle[qubit] = idle.get(qubit, 0) + max(0, entry["start_s"] - last_end.get(qubit, 0))
                if name == "delay":
                    idle[qubit] += duration
                last_end[qubit] = entry["start_s"] + duration
        if name == "measure":
            measured.update(qubits)
    qubit_properties = {}
    for qubit in sorted(active):
        props = target.qubit_properties[qubit] if target.qubit_properties else None
        qubit_properties[str(qubit)] = {
            "t1_s": props.t1 if props is not None else None,
            "t2_s": props.t2 if props is not None else None,
            "idle_s": idle.get(qubit) if starts is not None else None,
        }
    raw_properties = None
    if callable(getattr(backend, "properties", None)):
        try:
            raw = backend.properties(refresh=True)
        except TypeError:
            raw = backend.properties()
        if raw is not None:
            raw_properties = json.loads(json.dumps(raw.to_dict(), default=str))
            _add_directional_readout(qubit_properties, raw)
    snapshot = {
        "schema_version": "1.0",
        "backend_name": backend.name,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "calibrated_at": raw_properties.get("last_update_date") if raw_properties else None,
        "backend_properties": raw_properties,
        "circuit_hash": circuit_hash(circuit),
        "dt_s": dt,
        "scheduled": starts is not None and dt is not None,
        "instructions": entries,
        "qubits": qubit_properties,
    }
    snapshot["snapshot_hash"] = record_hash(snapshot)
    return snapshot


def snapshot_at_properties(snapshot: Mapping[str, Any], properties: Any) -> Dict[str, Any]:
    """Re-read errors for the same instructions from another calibration record.

    Used after execution with ``job.properties()`` (calibration in effect when
    the job ran) to separate calibration drift from model error. Instruction
    list, timing and circuit hash are unchanged; only error values, T1/T2 and
    readout rates are replaced. The result is diagnostic and never replaces
    the prediction frozen before submission.
    """
    updated = json.loads(json.dumps(snapshot))
    updated.pop("snapshot_hash", None)

    def lookup(read):
        try:
            value = read()
        except Exception:  # pylint: disable=broad-except
            return None
        return float(value) if value is not None else None

    for entry in updated["instructions"]:
        if entry["name"] in {"delay", "barrier"}:
            continue
        if entry["name"] == "measure":
            entry["error"] = lookup(lambda e=entry: properties.readout_error(e["qubits"][0]))
        else:
            entry["error"] = lookup(lambda e=entry: properties.gate_error(e["name"], e["qubits"]))
    for qubit, values in updated["qubits"].items():
        values["t1_s"] = lookup(lambda q=int(qubit): properties.t1(q))
        values["t2_s"] = lookup(lambda q=int(qubit): properties.t2(q))
        for name in ("prob_meas1_prep0", "prob_meas0_prep1"):
            values[name] = lookup(lambda q=int(qubit), n=name: properties.qubit_property(q, n)[0])
    updated["calibrated_at"] = str(getattr(properties, "last_update_date", None))
    updated["source"] = "execution_time_properties"
    updated["snapshot_hash"] = record_hash(updated)
    return updated


class CircuitErrorPrediction(MetricCalculator):
    """Opt-in PRE_RUNTIME strategy for experimental calibration-product baselines.

    ``snapshot`` must come from ``capture_calibration`` on the supplied circuit.
    Average calibration error is treated as a score, not a physical independent
    failure probability. The mixture is a separate, testable hypothesis.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        snapshot: Mapping[str, Any],
        reference: Mapping[str, float],
    ):
        super().__init__(circuit)
        self._snapshot = json.loads(json.dumps(snapshot))
        self._reference = validate_distribution(reference, circuit.num_clbits)

    def _get_metric_type(self) -> MetricsType:
        return MetricsType.PRE_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        return MetricsId.ERROR_PREDICTION

    def is_ready(self) -> bool:
        return self.circuit is not None and bool(self._snapshot)

    def get_metrics(self) -> ErrorPredictionSchema:
        snapshot = dict(self._snapshot)
        digest = snapshot.pop("snapshot_hash")
        if record_hash(snapshot) != digest or snapshot["circuit_hash"] != circuit_hash(
            self.circuit
        ):
            raise ValueError("Snapshot or circuit changed after calibration capture")
        gates: List[float] = []
        measures: List[float] = []
        missing = 0
        entries = snapshot["instructions"]
        for entry in entries:
            if entry["name"] == "delay":
                continue
            error = entry["error"]
            if error is None:
                missing += 1
                continue
            if not math.isfinite(error) or not 0 <= error <= 1:
                raise ValueError("Calibration errors must be finite and in [0, 1]")
            (measures if entry["name"] == "measure" else gates).append(error)

        def product(errors):
            return 0.0 if 1 in errors else math.exp(sum(math.log1p(-r) for r in errors))

        partial = product(gates + measures)
        total = len(gates) + len(measures) + missing
        measurement_entries = [e for e in entries if e["name"] == "measure"]
        measured_bits = [c for e in measurement_entries for c in e["clbits"]]
        complete = (
            missing == 0
            and len(measures) == self.circuit.num_clbits
            and sorted(measured_bits) == list(range(self.circuit.num_clbits))
        )
        score = partial if complete else None
        width = self.circuit.num_clbits
        uniform = math.ldexp(1.0, -width)
        probabilities = None
        if score is not None and width <= 12:
            probabilities = {
                f"{x:0{width}b}": score * self._reference.get(f"{x:0{width}b}", 0)
                + (1 - score) * uniform
                for x in range(2**width)
            }
        # Sparse ideal supports suffice to calculate TVD(p, uniform) exactly.
        distance = 0.5 * (
            sum(abs(p - uniform) for p in self._reference.values())
            + (1 - len(self._reference) * uniform)
        )
        features = {
            "gate_error_cost": (
                -math.log(product(gates)) if gates and product(gates) > 0 and not missing else None
            ),
            "readout_error_cost": (
                -math.log(product(measures))
                if len(measures) == width and product(measures) > 0
                else None
            ),
            "native_one_qubit_gates": sum(
                len(e["qubits"]) == 1 and e["name"] not in {"measure", "delay"} for e in entries
            ),
            "native_two_qubit_gates": sum(len(e["qubits"]) == 2 for e in entries),
            "transpiled_depth": self.circuit.depth(),
            "scheduled": snapshot["scheduled"],
            "duration_s": (
                max(((e["start_s"] or 0) + (e["duration_s"] or 0) for e in entries), default=0)
                if snapshot["scheduled"]
                else None
            ),
        }
        for name in ("t1", "t2"):
            values = list(snapshot["qubits"].values())
            features[f"idle_over_{name}"] = (
                sum(q["idle_s"] / q[f"{name}_s"] for q in values)
                if snapshot["scheduled"] and all(q[f"{name}_s"] for q in values)
                else None
            )
        m2 = depolarizing_prediction(snapshot, self._reference, width)
        return ErrorPredictionSchema(
            circuit_hash=snapshot["circuit_hash"],
            snapshot_hash=digest,
            backend_name=snapshot["backend_name"],
            collected_at=snapshot["collected_at"],
            output_width=width,
            calibration_coverage=(total - missing) / total if total else 0,
            missing_instructions=missing,
            partial_product=partial,
            gate_product=product(gates) if not missing else None,
            readout_product=product(measures) if len(measures) == width else None,
            esp_score=score,
            m1_success=score if len(self._reference) == 1 else None,
            effective_visibility=score,
            mixture_tvd=(1 - score) * distance if score is not None else None,
            mixture_probabilities=probabilities,
            m2_polarization=m2["polarization"],
            m2_success=m2["success"],
            m2_tvd=m2["tvd"],
            m2_probabilities=m2["probabilities"],
            features=features,
            reference_probabilities=self._reference,
            assumptions=[
                "Experimental M1 scores, not quantum-state fidelity or guaranteed error bounds",
                "M1-success: ESP directly predicts a deterministic reference's success",
                "M1-mixture: visibility=ESP in a global output-uniform mixture",
                "M2: depolarizing p=r*d/(d-1) per gate, global uniform scrambling, "
                "independent (directional when available) readout flips",
                "Readout included once as a scalar average-error product",
                "No extra total-duration decoherence multiplier; idle features reported separately",
                "Calibration uncertainty, coherent errors and crosstalk are not bounded by M1",
            ],
        )
