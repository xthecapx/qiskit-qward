"""
Quantum Software Quality Metrics implementation for QWARD.

This module provides the ElementMetrics class for analyzing quantum circuits
and extracting fundamental element-level metrics used across the QWARD framework.
These metrics focus on quantifying the basic building blocks of a circuit,
including qubits and quantum gates, as well as their distributions and
relationships.

This module provides the ElementMetrics class for analyzing quantum circuits
and extracting comprehensive quality metrics as defined in the paper:

[ ] J. A. Cruz-Lemus, L. A. Marcelo, and M. Piattini, "Towards a set of metrics for
quantum circuits understandability," in *Quality of Information and Communications
Technology. QUATIC 2021 (Communications in Computer and Information Science, vol. 1439)
*, A. C. R. Paiva, A. R. Cavalli, P. Ventura Martins, and R. Pérez-Castillo, Eds. Cham:
 Springer, 2021, pp. 238–253. doi: 10.1007/978-3-030-85347-1_18.


These metrics provide detailed analysis of quantum circuit structure, gate distribution,
and oracle usage patterns for quality assessment and comparison.
"""

from typing import Dict, Any, List, Tuple

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.circuit import Instruction, ControlledGate

from qward.metrics import types
from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsType, MetricsId

# Import schemas for structured data validation
try:
    from qward.schemas.element_metrics_schema import ElementMetricsSchema

    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False

# =============================================================================
# Gate Classification Constants
# =============================================================================

# Single-qubit gates
SINGLE_QUBIT_GATES = {
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz",
    "u1",
    "u2",
    "u3",
    "p",
    "u",
    "sx",
    "sxdg",
    "rzx",
    "phase",
    "reset",
}

# Pauli gates
PAULI_GATES = {"x", "y", "z"}

# Controlled gates (any controlled gate)
CONTROLLED_GATE_PREFIXES = {"c", "mc"}

# Specific controlled gates
CONTROLLED_SINGLE_QUBIT_GATES = {
    "cy",
    "cz",
    "ch",
    "cs",
    "csdg",
    "ct",
    "ctdg",
    "crx",
    "cry",
    "crz",
    "cp",
    "cu1",
    "cu2",
    "cu3",
    "cu",
    "csx",
    "csxdg",
    "crzx",
    "cphase",
}

# Two-qubit gates
TWO_QUBIT_GATES = {
    "cx",
    "cy",
    "cz",
    "swap",
    "iswap",
    "dcx",
    "ecr",
    "rxx",
    "ryy",
    "rzz",
    "rzx",
    "xx_minus_yy",
    "xx_plus_yy",
}

# Multi-qubit gates (3+ qubits)
MULTI_QUBIT_GATES = {
    "ccx",
    "cswap",
    "mcx",
    "mcphase",
    "mcu1",
    "mcu2",
    "mcu3",
    "mcrx",
    "mcry",
    "mcrz",
    "mcp",
    "mcu",
    "mcswap",
}

# Oracle gates (custom gates that represent oracles)
ORACLE_GATES = {"oracle", "black_box", "unitary", "custom", "phase_oracle", "grover_oracle", "u"}

# Measurement gates
MEASUREMENT_GATES = {"measure", "reset"}

# Ignored gates (not considered in quality metrics)
IGNORED_GATES = {"barrier", "delay", "snapshot", "save", "initialize", "finalize"}


class ElementMetrics(MetricCalculator):
    """
    Quantum Element Metrics calculator for QWARD.

    This class implements element-based metrics for quantum circuits
    focusing on gate distribution, oracle usage patterns, and measurement analysis.
    """

    @property
    def id(self):
        return types.MetricsId.ELEMENT.value

    def __init__(self, circuit: QuantumCircuit):
        """
        Initialize the ElementMetrics calculator.

        Args:
            circuit: The quantum circuit to analyze
        """
        super().__init__(circuit)
        self._circuit_dag = circuit_to_dag(circuit) if circuit else None
        self._ensure_schemas_available()

    def _get_metric_type(self) -> MetricsType:
        """
        Get the type of this metric.

        Returns:
            MetricsType: PRE_RUNTIME (can be calculated without execution)
        """
        return MetricsType.PRE_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        """
        Get the ID of this metric.

        Returns:
            MetricsId: ELEMENT
        """
        return MetricsId.ELEMENT

    def is_ready(self) -> bool:
        """
        Check if the metric is ready to be calculated.

        Returns:
            bool: True if the circuit is available, False otherwise
        """
        return self.circuit is not None

    def get_metrics(self) -> ElementMetricsSchema:
        """
        Calculate and return quantum element metrics.

        Returns:
            ElementMetricsSchema: Validated schema with all metrics

        Raises:
            ImportError: If schemas are not available
        """
        if not SCHEMAS_AVAILABLE:
            raise ImportError("ElementMetricsSchema is not available")

        # Analyze the circuit to extract all metrics
        gate_counts = self._count_gates()
        _ = self._analyze_qubits()  # Used for internal analysis
        oracle_analysis = self._analyze_oracles()
        measurement_analysis = self._analyze_measurements()

        # Calculate Pauli gate metrics
        no_p_x = gate_counts.get("x", 0)
        no_p_y = gate_counts.get("y", 0)
        no_p_z = gate_counts.get("z", 0)
        t_no_p = no_p_x + no_p_y + no_p_z

        # Calculate single-qubit gate metrics
        no_h = gate_counts.get("h", 0)
        percent_sppos_q = self._calculate_superposition_ratio()
        no_other_sg = self._count_other_single_qubit_gates(gate_counts)
        t_no_csqg = self._count_controlled_single_qubit_gates(gate_counts)
        t_no_sqg = no_h + t_no_p + no_other_sg + t_no_csqg

        # Calculate controlled gate metrics
        no_c_any_g = self._count_all_controlled_gates(gate_counts)
        no_swap = gate_counts.get("swap", 0)
        no_cnot = gate_counts.get("cx", 0)
        percent_q_in_cnot = self._calculate_cnot_qubit_ratio()
        avg_cnot, max_cnot = self._calculate_cnot_stats()

        # Calculate Toffoli gate metrics
        no_toff = gate_counts.get("ccx", 0)
        percent_q_in_toff = self._calculate_toffoli_qubit_ratio()
        avg_toff, max_toff = self._calculate_toffoli_stats()

        # Calculate general gate metrics
        no_gates = self._count_total_gates(gate_counts)
        no_c_gates = t_no_csqg + no_toff + oracle_analysis["controlled_oracles"] + no_cnot
        percent_single_gates = (t_no_sqg) / no_gates if no_gates > 0 else 0.0

        # Oracle metrics
        no_or = oracle_analysis["oracles"] + oracle_analysis["controlled_oracles"]
        no_c_or = oracle_analysis["controlled_oracles"]
        percent_q_in_or = oracle_analysis["qubit_ratio"]
        percent_q_in_c_or = oracle_analysis["controlled_qubit_ratio"]
        avg_or_d = oracle_analysis["avg_depth"]
        max_or_d = oracle_analysis["max_depth"]

        # Measurement and ancilla metrics
        no_qm = measurement_analysis["measured_qubits"]
        percent_qm = measurement_analysis["measured_ratio"]
        percent_anc = len(self.circuit.ancillas) / self.circuit.num_qubits

        return ElementMetricsSchema(
            no_p_x=no_p_x,
            no_p_y=no_p_y,
            no_p_z=no_p_z,
            t_no_p=t_no_p,
            no_h=no_h,
            percent_sppos_q=percent_sppos_q,
            no_other_sg=no_other_sg,
            t_no_csqg=t_no_csqg,
            t_no_sqg=t_no_sqg,
            no_c_any_g=no_c_any_g,
            no_swap=no_swap,
            no_cnot=no_cnot,
            percent_q_in_cnot=percent_q_in_cnot,
            avg_cnot=avg_cnot,
            max_cnot=max_cnot,
            no_toff=no_toff,
            percent_q_in_toff=percent_q_in_toff,
            avg_toff=avg_toff,
            max_toff=max_toff,
            no_gates=no_gates,
            no_c_gates=no_c_gates,
            percent_single_gates=percent_single_gates,
            no_or=no_or,
            no_c_or=no_c_or,
            percent_q_in_or=percent_q_in_or,
            percent_q_in_c_or=percent_q_in_c_or,
            avg_or_d=avg_or_d,
            max_or_d=max_or_d,
            no_qm=no_qm,
            percent_qm=percent_qm,
            percent_anc=percent_anc,
        )

    def _count_gates(self) -> Dict[str, int]:
        """
        Count the occurrences of each gate type in the circuit.
        Excludes measurement and reset operations as they are not considered gates
        for quantum software quality metrics.

        Returns:
            Dict[str, int]: Dictionary mapping gate names to counts
        """
        gate_counts: Dict[str, int] = {}

        for instruction in self.circuit.data:
            gate_name = instruction.operation.name
            if gate_name not in MEASUREMENT_GATES and gate_name not in IGNORED_GATES:
                gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1

        return gate_counts

    def _analyze_qubits(self) -> Dict[str, Any]:
        """
        Analyze qubit usage patterns in the circuit.

        Returns:
            Dict[str, Any]: Dictionary with qubit analysis results
        """
        qubit_operations: Dict[int, List[str]] = {i: [] for i in range(self.circuit.num_qubits)}

        for instruction in self.circuit.data:
            for qubit in instruction.qubits:
                qubit_index = self.circuit.find_bit(qubit).index
                qubit_operations[qubit_index].append(instruction.operation.name)

        return {
            "qubit_operations": qubit_operations,
            "total_operations": sum(len(ops) for ops in qubit_operations.values()),
        }

    def _analyze_oracles(self) -> Dict[str, Any]:
        """
        Analiza el uso de oráculos en el circuito.

        Returns:
            Dict[str, Any]: métricas de análisis de oráculos.
        """

        def _is_oracle_gate(instr: Instruction) -> bool:
            """Verifica si una instrucción o su definición interna representa un oráculo."""
            name = getattr(instr, "name", "").lower()
            if any(key in name for key in ORACLE_GATES):
                return True

            class_name = instr.__class__.__name__.lower()
            if any(key in class_name for key in ORACLE_GATES):
                return True

            return False

        oracles = 0
        controlled_oracles = 0
        oracle_qubits = set()
        controlled_oracle_qubits = set()
        affected_qubits = set()
        oracle_depths = []

        # --- Iterar sobre las instrucciones del circuito ---
        for instruction in self.circuit.data:
            instr = instruction.operation
            qargs = instruction.qubits
            gate_name = instr.name.lower()

            # Ignorar operaciones no unitarias
            if gate_name in {"measure", "barrier", "reset", "delay"}:
                continue

            if _is_oracle_gate(instr):
                qubits = [self.circuit.find_bit(q).index for q in qargs]

                # Distinguir entre oráculos controlados y simples
                if (
                    gate_name.startswith(("c", "mc"))
                    or len(qargs) > 2
                    or isinstance(instr, ControlledGate)
                ):
                    if isinstance(instr, ControlledGate):
                        n_ctrl = instr.num_ctrl_qubits
                        # ctrl_qubits = qubits[:n_ctrl]
                        target_qubits = qubits[n_ctrl:]
                        controlled_oracles += 1
                        controlled_oracle_qubits.update(qubits)
                        affected_qubits.update(target_qubits)
                    else:
                        controlled_oracles += 1
                        controlled_oracle_qubits.update(qubits)

                        # Último qubit = target afectado
                        if qubits:
                            affected_qubits.add(qubits[-1])
                    oracle_depths.append(len(affected_qubits))
                else:
                    # Oráculo simple
                    oracles += 1
                    oracle_qubits.update(qubits)
                    affected_qubits.update(qubits)

                    oracle_depths.append(len(qargs))

        # --- Métricas finales ---
        total_qubits = self.circuit.num_qubits
        qubit_ratio = len(affected_qubits) / total_qubits if total_qubits else 0.0
        controlled_qubit_ratio = (
            len(controlled_oracle_qubits) / total_qubits if total_qubits else 0.0
        )

        return {
            "oracles": oracles,
            "controlled_oracles": controlled_oracles,
            "qubit_ratio": qubit_ratio,
            "controlled_qubit_ratio": controlled_qubit_ratio,
            "avg_depth": sum(oracle_depths) / len(oracle_depths) if oracle_depths else 0.0,
            "max_depth": max(oracle_depths) if oracle_depths else 0,
        }

    def _analyze_measurements(self) -> Dict[str, Any]:
        """
        Analyze measurement patterns in the circuit.

        Returns:
            Dict[str, Any]: Dictionary with measurement analysis results
        """
        measured_qubits = set()

        for instruction in self.circuit.data:
            if instruction.operation.name == "measure":
                for qubit in instruction.qubits:
                    measured_qubits.add(self.circuit.find_bit(qubit).index)

        total_qubits = self.circuit.num_qubits
        measured_ratio = len(measured_qubits) / total_qubits if total_qubits > 0 else 0.0

        return {"measured_qubits": len(measured_qubits), "measured_ratio": measured_ratio}

    def _calculate_superposition_ratio(self) -> float:
        """
        Calculate the ratio of qubits with a Hadamard gate as an initial gate.

        Returns:
            float: Ratio of qubits in superposition state
        """
        if self.circuit.num_qubits == 0:
            return 0.0

        qubits_with_initial_h = 0

        # Find the first gate applied to each qubit
        first_gates = {}

        for instruction in self.circuit.data:
            for qubit in instruction.qubits:
                qubit_index = self.circuit.find_bit(qubit).index
                if qubit_index not in first_gates:
                    first_gates[qubit_index] = instruction.operation.name

        # Count qubits that start with Hadamard
        for qubit_index, first_gate in first_gates.items():
            if first_gate == "h":
                qubits_with_initial_h += 1

        return qubits_with_initial_h / self.circuit.num_qubits

    def _count_other_single_qubit_gates(self, gate_counts: Dict[str, int]) -> int:
        """
        Count other single-qubit gates (excluding Pauli and Hadamard).

        Args:
            gate_counts: Dictionary of gate counts

        Returns:
            int: Number of other single-qubit gates
        """
        other_single_qubit = 0

        for gate_name, count in gate_counts.items():
            if (
                gate_name in SINGLE_QUBIT_GATES
                and gate_name not in PAULI_GATES
                and gate_name != "h"
            ):
                other_single_qubit += count

        return other_single_qubit

    def _count_controlled_single_qubit_gates(self, gate_counts: Dict[str, int]) -> int:
        """
        Count controlled single-qubit gates.

        Args:
            gate_counts: Dictionary of gate counts

        Returns:
            int: Number of controlled single-qubit gates
        """
        controlled_single_qubit = 0

        for gate_name, count in gate_counts.items():
            if gate_name in CONTROLLED_SINGLE_QUBIT_GATES:
                controlled_single_qubit += count

        return controlled_single_qubit

    def _count_all_controlled_gates(self, gate_counts: Dict[str, int]) -> int:
        """
        Count all controlled gates.

        Args:
            gate_counts: Dictionary of gate counts

        Returns:
            int: Number of controlled gates
        """
        controlled_gates = 0

        for gate_name, count in gate_counts.items():
            if (
                gate_name.startswith("c")
                or gate_name.startswith("mc")
                or gate_name in MULTI_QUBIT_GATES
            ):
                controlled_gates += count

        return controlled_gates

    def _calculate_cnot_qubit_ratio(self) -> float:
        """
        Calculate the ratio of qubits affected by CNOT gates.

        Returns:
            float: Ratio of qubits affected by CNOT gates
        """
        if self.circuit.num_qubits == 0:
            return 0.0

        cnot_qubits = set()

        for instruction in self.circuit.data:
            if instruction.operation.name == "cx":
                for qubit in instruction.qubits:
                    cnot_qubits.add(self.circuit.find_bit(qubit).index)

        return len(cnot_qubits) / self.circuit.num_qubits

    def _calculate_cnot_stats(self) -> Tuple[float, int]:
        """
        Calculate average and maximum number of times each qubit
        is affected (as target) by a CNOT gate.

        Returns:
            Tuple[float, int]: (average_cnot, max_cnot)
        """
        qubit_cnot_counts = {i: 0 for i in range(self.circuit.num_qubits)}

        for instruction in self.circuit.data:
            if instruction.operation.name == "cx":
                target_qubit = instruction.qubits[1]
                target_index = self.circuit.find_bit(target_qubit).index
                qubit_cnot_counts[target_index] += 1

        counts = list(qubit_cnot_counts.values())

        avg_cnot = sum(counts) / self.circuit.num_qubits if counts else 0.0
        max_cnot = max(counts) if counts else 0

        return avg_cnot, max_cnot

    def _calculate_toffoli_qubit_ratio(self) -> float:
        """
        Calculate the ratio of qubits affected by Toffoli gates.

        Returns:
            float: Ratio of qubits affected by Toffoli gates
        """
        if self.circuit.num_qubits == 0:
            return 0.0

        toffoli_qubits = set()

        for instruction in self.circuit.data:
            if instruction.operation.name == "ccx":
                for qubit in instruction.qubits:
                    toffoli_qubits.add(self.circuit.find_bit(qubit).index)

        return len(toffoli_qubits) / self.circuit.num_qubits

    def _calculate_toffoli_stats(self) -> Tuple[float, int]:
        """
        Calculate average and maximum Toffoli gates per qubit.

        Returns:
            Tuple[float, int]: (average_toffoli, max_toffoli)
        """
        num_qubits = self.circuit.num_qubits
        toffoli_counts = [0] * num_qubits

        for instruction in self.circuit.data:
            if instruction.operation.name.lower() in {"ccx", "toffoli"}:
                target_qubit = instruction.qubits[-1]
                target_index = self.circuit.find_bit(target_qubit).index
                toffoli_counts[target_index] += 1

        avg_toff = sum(toffoli_counts) / num_qubits if num_qubits else 0
        max_toff = max(toffoli_counts) if toffoli_counts else 0

        return (avg_toff, max_toff)

    def _count_total_gates(self, gate_counts: Dict[str, int]) -> int:
        """
        Count total number of gates in the circuit.

        Args:
            gate_counts: Dictionary of gate counts

        Returns:
            int: Total number of gates
        """
        return sum(gate_counts.values())

    def _ensure_schemas_available(self):
        """
        Ensure that the required schemas are available.

        Raises:
            ImportError: If schemas are not available
        """
        if not SCHEMAS_AVAILABLE:
            raise ImportError(
                "ElementMetricsSchema is not available. "
                "Please ensure that the schemas module is properly imported."
            )
