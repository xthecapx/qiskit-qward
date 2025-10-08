"""LOC-oriented metric strategy for QWARD.

Calculates the following metrics:

- ϕ1 (phi1_total_loc): Total LOC in the quantum program (estimated from QASM if available)
- ϕ2 (phi2_gate_loc): LOC related to quantum gate operations
- ϕ3 (phi3_measure_loc): LOC related to quantum measurements
- ϕ4 (phi4_quantum_total_loc): Total LOC related to quantum aspects (gates + measurements + other quantum ops)
- ϕ5 (phi5_num_qubits): Number of qubits used
- ϕ6 (phi6_num_gate_types): Number of distinct gate types used
"""

from __future__ import annotations

from typing import Any, Optional, Set

from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsType, MetricsId
from qward.schemas.loc_metrics_schema import LocMetricsSchema


class LocMetrics(MetricCalculator):
    """Concrete strategy that computes LOC-related metrics for a circuit/program."""

    def _get_metric_type(self) -> MetricsType:
        return MetricsType.PRE_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        return MetricsId.LOC

    def is_ready(self) -> bool:
        return self.circuit is not None

    def get_metrics(self) -> LocMetricsSchema:
        # Try to obtain a QASM representation
        qasm_text: Optional[str] = self._export_qasm_text()

        if qasm_text is not None:
            (
                phi1_total_loc,
                phi2_gate_loc,
                phi3_measure_loc,
                phi4_quantum_total_loc,
            ) = self._compute_loc_from_qasm(qasm_text)
        else:
            # Fallback: Estimate from circuit instructions if QASM is unavailable
            (
                phi1_total_loc,
                phi2_gate_loc,
                phi3_measure_loc,
                phi4_quantum_total_loc,
            ) = self._compute_loc_from_circuit()

        # ϕ5: Number of qubits
        phi5_num_qubits = getattr(self.circuit, "num_qubits", 0) or 0

        # ϕ6: Number of distinct quantum gate types (exclude non-gate ops)
        phi6_num_gate_types = len(self._distinct_gate_types())

        return LocMetricsSchema(
            phi1_total_loc=phi1_total_loc,
            phi2_gate_loc=phi2_gate_loc,
            phi3_measure_loc=phi3_measure_loc,
            phi4_quantum_total_loc=phi4_quantum_total_loc,
            phi5_num_qubits=phi5_num_qubits,
            phi6_num_gate_types=phi6_num_gate_types,
        )

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _export_qasm_text(self) -> Optional[str]:
        # Prefer QASM 3 if available (Qiskit >= 1.0)
        try:
            from qiskit.qasm3 import dumps as qasm3_dumps  # type: ignore

            return qasm3_dumps(self.circuit)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001 - best-effort QASM export
            pass

        # Fallback to legacy QASM (if available on the circuit)
        try:
            if hasattr(self.circuit, "qasm"):
                return self.circuit.qasm()  # type: ignore[no-any-return]
        except Exception:  # noqa: BLE001
            pass

        return None

    def _compute_loc_from_qasm(self, qasm_text: str) -> tuple[int, int, int, int]:
        lines = [line.strip() for line in qasm_text.splitlines()]

        # Exclude empty and comment lines
        content_lines = [
            ln
            for ln in lines
            if ln and not ln.startswith("//") and not ln.startswith("#")
        ]

        # ϕ1: total LOC (approx. program LOC from QASM)
        phi1_total_loc = len(content_lines)

        # Identify operational quantum lines
        quantum_gate_prefixes = (
            "x ",
            "y ",
            "z ",
            "h ",
            "s ",
            "sdg ",
            "t ",
            "tdg ",
            "rx ",
            "ry ",
            "rz ",
            "cx ",
            "cy ",
            "cz ",
            "swap ",
            "ccx ",
            "cswap ",
            "rxx ",
            "rzz ",
            "ryy ",
            "rzx ",
            "cp ",
            "crx ",
            "cry ",
            "crz ",
            "u ",
            "u1 ",
            "u2 ",
            "u3 ",
        )

        def is_declaration(ln: str) -> bool:
            return ln.startswith("OPENQASM") or ln.startswith("include ") or ln.startswith("qubit[") or ln.startswith("bit[") or ln.startswith("qreg ") or ln.startswith("creg ")

        def is_measure(ln: str) -> bool:
            return ln.startswith("measure ") or ln.startswith("meas")

        def is_other_quantum(ln: str) -> bool:
            # Non-gate quantum ops like barrier/reset/delay (names may vary between QASM versions)
            return ln.startswith("barrier") or ln.startswith("reset") or ln.startswith("delay ")

        def is_gate_line(ln: str) -> bool:
            return (not is_measure(ln)) and (not is_other_quantum(ln)) and ln.startswith(quantum_gate_prefixes)

        operational_lines = [ln for ln in content_lines if not is_declaration(ln)]

        phi2_gate_loc = sum(1 for ln in operational_lines if is_gate_line(ln))
        phi3_measure_loc = sum(1 for ln in operational_lines if is_measure(ln))
        other_quantum_loc = sum(1 for ln in operational_lines if is_other_quantum(ln))

        # ϕ4: total quantum aspect LOC = gate + measure + other quantum ops
        phi4_quantum_total_loc = phi2_gate_loc + phi3_measure_loc + other_quantum_loc

        return phi1_total_loc, phi2_gate_loc, phi3_measure_loc, phi4_quantum_total_loc

    def _compute_loc_from_circuit(self) -> tuple[int, int, int, int]:
        # Fallback approximation: treat each instruction as one LOC
        # Count by instruction categories from circuit.data
        gate_ops: Set[str] = set()
        measure_count = 0
        other_quantum_count = 0
        total_instruction_count = 0

        for instr, _q, _c in getattr(self.circuit, "data", []):
            name = getattr(instr, "name", "")
            total_instruction_count += 1
            if name == "measure":
                measure_count += 1
            elif name in {"barrier", "reset", "delay"}:
                other_quantum_count += 1
            else:
                gate_ops.add(name)

        phi2_gate_loc = len(gate_ops) if False else total_instruction_count - measure_count - other_quantum_count
        phi3_measure_loc = measure_count
        phi4_quantum_total_loc = total_instruction_count

        # Use instruction count as a conservative proxy for program LOC
        phi1_total_loc = total_instruction_count

        return phi1_total_loc, phi2_gate_loc, phi3_measure_loc, phi4_quantum_total_loc

    def _distinct_gate_types(self) -> Set[str]:
        non_gate_names = {"measure", "barrier", "reset", "delay"}
        types: Set[str] = set()
        for instr, _q, _c in getattr(self.circuit, "data", []):
            name = getattr(instr, "name", "")
            if name and name not in non_gate_names:
                types.add(name)
        return types



