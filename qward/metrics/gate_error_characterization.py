"""
Gate error characterization for transpiled circuits.

Extracts per-gate error rates from the backend for the specific physical
qubits assigned during transpilation.
"""

import statistics
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from qiskit import QuantumCircuit

from qward.metrics.backend_metric_base import BackendMetricCollector
from qward.schemas.gate_error_characterization_schema import (
    GateErrorCharacterizationSchema,
    GateErrorEntry,
)


class GateErrorCharacterization(BackendMetricCollector):
    """
    Characterizes gate errors for the specific qubits used in a transpiled circuit.

    Requires both a transpiled circuit (with layout info) and the backend
    that was targeted during transpilation.
    """

    def __init__(self, transpiled_circuit: QuantumCircuit, backend):
        super().__init__(backend)
        self._transpiled_circuit = transpiled_circuit

    def is_available(self) -> bool:
        """Check if backend has gate error data in target."""
        return hasattr(self._backend, "target") and self._backend.target is not None

    def get_metrics(self) -> GateErrorCharacterizationSchema:
        """Extract per-gate error characterization."""
        if not self.is_available():
            return GateErrorCharacterizationSchema()

        target = self._backend.target
        entries: List[GateErrorEntry] = []
        physical_qubits_used: Set[int] = set()

        for instruction in self._transpiled_circuit.data:
            op_name = instruction.operation.name
            if op_name in ("barrier", "measure", "delay", "reset"):
                continue

            physical_qubits = self._get_physical_qubits(instruction)
            if physical_qubits is None:
                continue

            physical_qubits_used.update(physical_qubits)
            error_rate, duration_ns = self._query_gate_error(target, op_name, physical_qubits)

            entries.append(
                GateErrorEntry(
                    gate_name=op_name,
                    physical_qubits=list(physical_qubits),
                    error_rate=error_rate,
                    duration_ns=duration_ns,
                )
            )

        return self._build_schema(entries, physical_qubits_used)

    def to_dict(self) -> Dict[str, Any]:
        """Return metrics as plain dictionary for JSON serialization."""
        return self.get_metrics().model_dump()

    def _get_physical_qubits(self, instruction) -> Optional[Tuple[int, ...]]:
        """Get physical qubit indices from instruction."""
        try:
            qubit_indices = []
            for qubit in instruction.qubits:
                idx = self._transpiled_circuit.find_bit(qubit).index
                qubit_indices.append(idx)
            return tuple(qubit_indices)
        except (ValueError, AttributeError):
            return None

    def _query_gate_error(
        self, target, op_name: str, physical_qubits: Tuple[int, ...]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Query backend target for gate error and duration."""
        error_rate = None
        duration_ns = None

        try:
            if op_name in target.operation_names:
                props_map = target[op_name]
                if props_map is not None and physical_qubits in props_map:
                    props = props_map[physical_qubits]
                    if props is not None:
                        if props.error is not None:
                            error_rate = props.error
                        if props.duration is not None:
                            duration_ns = props.duration * 1e9  # seconds → nanoseconds
        except (KeyError, TypeError, AttributeError):
            pass

        return error_rate, duration_ns

    def _build_schema(
        self, entries: List[GateErrorEntry], physical_qubits_used: Set[int]
    ) -> GateErrorCharacterizationSchema:
        """Build final schema with aggregate statistics."""
        single_q_errors: List[float] = []
        two_q_errors: List[float] = []
        all_errors: List[float] = []

        for entry in entries:
            if entry.error_rate is None:
                continue
            all_errors.append(entry.error_rate)
            if len(entry.physical_qubits) == 1:
                single_q_errors.append(entry.error_rate)
            elif len(entry.physical_qubits) == 2:
                two_q_errors.append(entry.error_rate)

        weighted_mean = statistics.mean(all_errors) if all_errors else None

        return GateErrorCharacterizationSchema(
            entries=entries,
            mean_single_qubit_error=(statistics.mean(single_q_errors) if single_q_errors else None),
            mean_two_qubit_error=(statistics.mean(two_q_errors) if two_q_errors else None),
            max_error=max(all_errors) if all_errors else None,
            weighted_mean_error=weighted_mean,
            num_distinct_physical_qubits=len(physical_qubits_used),
            physical_qubits_used=sorted(physical_qubits_used),
        )
