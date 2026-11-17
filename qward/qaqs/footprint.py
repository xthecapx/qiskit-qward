"""Pre-runtime circuit footprint used by the QAQS models.

The footprint is the set of metrics computed from a circuit *before* it runs: the logical
circuit (group ``a``) and the circuit compiled for a backend (group ``b``). Column names follow
the QAQS dataset (``a.StructuralMetrics.effort``, ``b.depth``, ...), so a footprint computed
here is directly comparable with the thesis data.

Quantum-specific metrics (``a.quantum_specific.magic`` and ``coherence``) need PyTorch and take
seconds at 6 qubits and over a minute at 8. They are optional; without them the models that use
them see a missing value, exactly like circuits above 8 qubits in the training data.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.transpiler import generate_preset_pass_manager

from qward.metrics import (
    BehavioralMetrics,
    ComplexityMetrics,
    ElementMetrics,
    QiskitMetrics,
    StructuralMetrics,
)
from qward.metrics.circuit_error_prediction import is_dynamic
from qward.scanner import Scanner
from qward.utils.flatten import flatten_dict

#: Largest logical width for which quantum-specific metrics were computed in training.
QUANTUM_SPECIFIC_MAX_QUBITS = 8
#: Seed that makes the quantum-specific metrics reproducible (same as the training data).
QUANTUM_SPECIFIC_SEED = 11
#: Compilation settings of every training collection.
OPTIMIZATION_LEVEL = 3
SEED_TRANSPILER = 11


def compact_isa(isa: QuantumCircuit) -> QuantumCircuit:
    """Compiled circuit restricted to the qubits that carry operations, without delays.

    A device-wide compiled circuit (156 qubits plus scheduling delays) is reduced to its active
    wires in physical order, so metrics describe the circuit and not the chip. Gate order,
    depth and two-qubit count are preserved.
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


def _qward_metrics(circuit: QuantumCircuit) -> Dict[str, Any]:
    """The five pre-runtime QWARD metric groups of one circuit, as nested plain values."""
    scanner = Scanner(circuit=circuit)
    for metric in (
        QiskitMetrics,
        ComplexityMetrics,
        StructuralMetrics,
        BehavioralMetrics,
        ElementMetrics,
    ):
        scanner.add_strategy(metric(circuit))
    out: Dict[str, Any] = {}
    for name, frame in scanner.calculate_metrics().items():
        if frame is None or getattr(frame, "empty", True):
            continue
        row = frame.iloc[0]
        out[name] = {}
        for column in frame.columns:
            value = row[column]
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, (int, float, str, bool, type(None))):
                out[name][column] = value
    return out


def logical_footprint(circuit: QuantumCircuit) -> Dict[str, Any]:
    """Group ``a`` without the quantum-specific metrics."""
    group = {f"a.{k}": v for k, v in flatten_dict(_qward_metrics(circuit)).items()}
    group["a.num_qubits"] = circuit.num_qubits
    return group


def compiled_footprint(isa: QuantumCircuit) -> Dict[str, Any]:
    """Group ``b``: metrics of the compiled circuit restricted to its active qubits.

    Raises:
        ValueError: if the compiled circuit has no gates (for example a circuit that cancels
            itself), so there is nothing to describe or predict.
    """
    compact = compact_isa(isa)
    if not any(item.operation.name not in {"measure", "barrier"} for item in compact.data):
        raise ValueError(
            "The circuit compiles to no gates (it only measures), so there is nothing to predict. "
            "A circuit that undoes itself, such as a transform followed by its inverse, is "
            "optimized away unless a barrier separates the two halves."
        )
    group = {f"b.{k}": v for k, v in flatten_dict(_qward_metrics(compact)).items()}
    group.update(
        {
            "b.depth": compact.depth(),
            "b.size": compact.size(),
            "b.two_qubit_gates": sum(len(i.qubits) == 2 for i in compact.data),
            "b.two_qubit_depth": compact.depth(lambda i: len(i.qubits) == 2),
            "b.count_ops.measure": dict(compact.count_ops()).get("measure", 0),
        }
    )
    return group


def quantum_specific(circuit: QuantumCircuit) -> Dict[str, Any]:
    """``a.quantum_specific.*`` with the same limits as the training data.

    Returns ``{"a.quantum_specific_present": 0.0}`` (and no metric values) for dynamic circuits,
    circuits wider than :data:`QUANTUM_SPECIFIC_MAX_QUBITS`, or when PyTorch is not installed.
    """
    absent = {"a.quantum_specific_present": 0.0}
    if is_dynamic(circuit) or circuit.num_qubits > QUANTUM_SPECIFIC_MAX_QUBITS:
        return absent
    try:
        import torch  # pylint: disable=import-outside-toplevel
        from qward.metrics import QuantumSpecificMetrics  # pylint: disable=import-outside-toplevel
    except ImportError:
        return absent
    state = torch.get_rng_state()
    torch.manual_seed(QUANTUM_SPECIFIC_SEED)
    try:
        started = time.perf_counter()
        values = QuantumSpecificMetrics(circuit).get_metrics().model_dump()
    finally:
        torch.set_rng_state(state)
    out = {f"a.quantum_specific.{k}": v for k, v in values.items() if v is not None}
    out["a.quantum_specific_present"] = 1.0
    out["a.quantum_specific.compute_seconds"] = round(time.perf_counter() - started, 3)
    return out


def ensure_barrier(circuit: QuantumCircuit) -> Tuple[QuantumCircuit, bool]:
    """Add one barrier before the final measurements when the circuit has no barrier at all.

    Every circuit in the training data had a barrier before its measurements, and the footprint
    is sensitive to it (the measurement-density metric is about 0.3 without one and about 0.04
    with one, against a training maximum of 0.14). A barrier has no effect on the computation;
    it only stops the compiler from optimizing across it. Returns the circuit and whether a
    barrier was added.
    """
    if any(item.operation.name == "barrier" for item in circuit.data):
        return circuit, False
    trailing = 0
    for item in reversed(circuit.data):
        if item.operation.name != "measure":
            break
        trailing += 1
    if trailing == 0:
        return circuit, False
    split = len(circuit.data) - trailing
    out = circuit.copy_empty_like()
    for item in circuit.data[:split]:
        out.append(item.operation, item.qubits, item.clbits)
    out.barrier()
    for item in circuit.data[split:]:
        out.append(item.operation, item.qubits, item.clbits)
    return out, True


def compile_circuit(circuit: QuantumCircuit, backend: Any) -> QuantumCircuit:
    """Compile like every training collection: optimization level 3, seed 11, ALAP scheduling.

    Dynamic circuits are compiled without scheduling (as in the training data).
    """
    manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=OPTIMIZATION_LEVEL,
        seed_transpiler=SEED_TRANSPILER,
        scheduling_method=None if is_dynamic(circuit) else "alap",
    )
    return manager.run(circuit)


def scheduled_duration_seconds(isa: QuantumCircuit, backend: Any) -> Optional[float]:
    """Scheduled duration of one shot, or ``None`` for dynamic / unscheduled circuits.

    Matches ``c.duration_s`` in the dataset: the end of the last scheduled operation.
    """
    try:
        starts = isa.op_start_times
    except AttributeError:
        return None
    dt = backend.target.dt
    if dt is None:
        return None
    end = 0.0
    for index, item in enumerate(isa.data):
        if item.operation.name in {"delay", "barrier"}:
            continue
        props = None
        if item.operation.name in backend.target.operation_names:
            qubits = tuple(isa.find_bit(q).index for q in item.qubits)
            props = backend.target[item.operation.name].get(qubits)
        duration = props.duration if props is not None and props.duration is not None else 0.0
        end = max(end, (starts[index] or 0) * dt + duration)
    return float(end)


def compute_footprint(
    circuit: QuantumCircuit,
    backend: Any,
    *,
    quantum_specific_metrics: bool = True,
    add_barrier: bool = True,
) -> Dict[str, Any]:
    """Footprint of ``circuit`` compiled for ``backend`` plus the runtime facts the cost model needs.

    Returns a flat dictionary of ``a.*`` / ``b.*`` metrics and the keys ``duration_s``,
    ``dynamic``, ``qubits`` and ``barrier_added`` (see :func:`ensure_barrier`).
    """
    barrier_added = False
    if add_barrier:
        circuit, barrier_added = ensure_barrier(circuit)
    isa = compile_circuit(circuit, backend)
    footprint: Dict[str, Any] = {}
    footprint.update(logical_footprint(circuit))
    footprint.update(
        quantum_specific(circuit)
        if quantum_specific_metrics
        else {"a.quantum_specific_present": 0.0}
    )
    footprint.update(compiled_footprint(isa))
    footprint["duration_s"] = scheduled_duration_seconds(isa, backend)
    footprint["dynamic"] = bool(is_dynamic(circuit))
    footprint["qubits"] = int(circuit.num_qubits)
    footprint["barrier_added"] = barrier_added
    return footprint


def as_row(footprint: Dict[str, Any], columns: list) -> np.ndarray:
    """Footprint values in ``columns`` order; missing or non-numeric values become NaN."""
    row = []
    for column in columns:
        value = footprint.get(column)
        row.append(float(value) if isinstance(value, (int, float, bool, np.number)) else np.nan)
    return np.asarray([row], dtype=float)
