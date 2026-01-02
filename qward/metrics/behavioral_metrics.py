"""
Behavioral Metrics implementation for QWARD.

This module provides the BehavioralMetrics class for analyzing quantum circuit
execution patterns and behavioral characteristics from a static circuit analysis,
including:

- Normalized Depth: Circuit depth after transpilation to canonical gate set [1]
- Program Communication: Communication requirements based on interaction graph [2]
- Critical-Depth: Two-qubit operations on critical path analysis [2]
- Measurement: Mid-circuit measurement and reset operations [2]
- Liveness: Qubit activity patterns during execution [2]
- Parallelism: Cross-talk susceptibility metric [2]


[1] T. Lubinski et al., "Application-Oriented Performance Benchmarks for
Quantum Computing," in IEEE Transactions on Quantum Engineering, vol. 4,
pp. 1-32, 2023, Art no. 3100332, doi: 10.1109/TQE.2023.3253761.

[2] T. Tomesh, P. Gokhale, V. Omole, G. S. Ravi, K. N. Smith, J. Viszlai,
X.-C. Wu, N. Hardavellas, M. R. Martonosi y F. T. Chong, “SupermarQ: A scalable
quantum benchmark suite,” in Proc. 2022 IEEE International Symposium on
High-Performance Computer Architecture (HPCA), 2022, doi:
10.1109/HPCA53966.2022.00050.

"""

import networkx as nx

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate, Instruction
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler import CouplingMap

from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsType, MetricsId
import itertools

# Import schemas for structured data validation
try:
    from qward.schemas.behavioral_metrics_schema import BehavioralMetricsSchema

    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False

# =============================================================================
# Gate Classification Constants
# =============================================================================

# Two-qubit gates for critical path analysis
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

# Measurement and reset operations
MEASUREMENT_OPERATIONS = {"measure", "reset"}

# Canonical basis gates for normalized depth calculation
CANONICAL_BASIS_GATES = ["rx", "ry", "rz", "cx"]


class BehavioralMetrics(MetricCalculator):
    """
    Extract behavioral metrics from QuantumCircuit objects.

    This class analyzes quantum circuits to compute behavioral metrics that
    describe how the circuit behaves as a computational process. Behavioral
    metrics capture dynamic characteristics such as parallelism potential,
    liveness of qubits across the circuit timeline, normalized depth measures,
    communication flow between qubits, and other indicators of execution
    dynamics derived from the circuit’s DAG representation.

    Attributes:
        circuit (QuantumCircuit):
            The quantum circuit to analyze (inherited from MetricCalculator).

        _dag_circuit (DAGCircuit):
            DAG representation of the circuit, used to extract execution
            behavior, dependency structure, and parallelism characteristics.
    """

    def __init__(self, circuit: QuantumCircuit):
        """
        Initialize the BehavioralMetrics calculator.

        Args:
            circuit: The quantum circuit to analyze
        """
        super().__init__(circuit)
        self._ensure_schemas_available()
        self._dag_circuit = circuit_to_dag(circuit)

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
            MetricsId: BEHAVIORAL_METRICS
        """
        return MetricsId.BEHAVIORAL

    def is_ready(self) -> bool:
        """
        Check if the metric is ready to be calculated.

        Returns:
            bool: True if the circuit is available, False otherwise
        """
        return self.circuit is not None

    def get_metrics(self) -> BehavioralMetricsSchema:
        """
        Calculate and return behavioral metrics.

        Returns:
            BehavioralMetricsSchema: Validated schema with all metrics

        Raises:
            ImportError: If schemas are not available
        """
        if not SCHEMAS_AVAILABLE:
            raise ImportError("BehavioralMetricsSchema is not available")

        # Calculate all behavioral metrics
        normalized_depth = self._calculate_normalized_depth()
        program_communication = self._calculate_program_communication()
        critical_depth = self._calculate_critical_depth()
        measurement = self._calculate_measurement()
        liveness = self._calculate_liveness()
        parallelism = self._calculate_parallelism()

        return BehavioralMetricsSchema(
            normalized_depth=normalized_depth,
            program_communication=program_communication,
            critical_depth=critical_depth,
            measurement=measurement,
            liveness=liveness,
            parallelism=parallelism,
        )

    def _make_all_to_all_coupling(self, n_qubits: int):
        """Genera una coupling_map totalmente conectada (all-to-all)."""
        return [(i, j) for i, j in itertools.permutations(range(n_qubits), 2) if i != j]

    def _calculate_normalized_depth(self) -> float:
        """
        Calculate the normalized depth by transpiling to canonical gate set.

        The circuit is transpiled to the basis gates ['rx', 'ry', 'rz', 'cx']
        and the depth of the transpiled circuit is returned.

        Returns:
            float: Depth of the transpiled circuit
        """
        try:
            cm = CouplingMap(self._make_all_to_all_coupling(self.circuit.num_qubits))
            # Transpile to canonical basis gates

            transpiled_circuit = transpile(
                self.circuit,
                basis_gates=CANONICAL_BASIS_GATES,
                coupling_map=cm,
                optimization_level=0,
            )

            return float(transpiled_circuit.depth())
        except Exception as e:
            # If transpilation fails, return original depth
            print(f"Transpilation failed: {e}, returning original depth.")
            return float(self.circuit.depth())

    def _calculate_program_communication(self) -> float:
        """
        Calculate the program communication metric.

        This metric quantifies communication requirements by analyzing the
        interaction graph of the circuit. It computes the normalized average
        degree of the interaction graph.

        Formula: C = Σ(d(qi)) / (N(N-1))
        where d(qi) is the degree of qubit qi and N is the number of qubits.

        Returns:
            float: Normalized communication requirements (0-1)
        """
        n = self.circuit.num_qubits
        if n <= 1:
            return 0.0

        graph = nx.Graph()
        graph.add_nodes_from(range(n))

        for instr in self.circuit.data:
            qargs = instr.qubits
            if len(qargs) > 1:
                indices = [self.circuit.find_bit(q).index for q in qargs]
                for i, idx_i in enumerate(indices):
                    for idx_j in indices[i + 1 :]:
                        graph.add_edge(idx_i, idx_j)

        degrees = dict(graph.degree())
        total_degree = sum(degrees.values())

        communication = total_degree / (n * (n - 1))
        return communication

    def _calculate_critical_depth(self) -> float:
        """
        Calculate the critical depth metric.

        This metric analyzes the critical path of the circuit and computes
        the ratio of two-qubit operations on the critical path to total
        two-qubit operations.

        Formula: D = ned/ne
        where ned is the number of two-qubit interactions on the critical path
        and ne is the total number of two-qubit interactions.

        Returns:
            float: Critical depth ratio (0-1)
        """
        # Count total two-qubit operations
        two_qubit_nodes = [
            node
            for node in self._dag_circuit.topological_op_nodes()
            if isinstance(node, DAGOpNode) and len(node.qargs) == 2
        ]
        ne = len(two_qubit_nodes)
        if ne == 0:
            return 0.0  # no hay operaciones de 2 qubits → D = 0

        longest_path = self._dag_circuit.longest_path()

        critical_op_nodes = [node for node in longest_path if isinstance(node, DAGOpNode)]

        ned = sum(1 for node in critical_op_nodes if len(node.qargs) == 2)

        critical_depth = ned / ne
        return critical_depth

    def _calculate_measurement(self) -> float:
        """
        Calculate the measurement metric.

        This metric focuses on mid-circuit measurement and reset operations.
        It computes the ratio of layers containing measurement/reset operations
        to the total circuit depth.

        Formula: M = lmcm/d
        where lmcm is the number of layers with measurement/reset operations
        and d is the circuit depth.

        Returns:
            float: Measurement ratio (0-1)
        """
        layers = list(self._dag_circuit.layers())
        d = len(layers)

        if d == 0:
            return 0.0

        l_mcm = 0

        # Recorremos las capas del DAG
        for layer in layers:
            # Cada capa contiene nodos (operaciones)
            ops = list(layer["graph"].op_nodes())
            if any(node.name in ("measure", "reset") for node in ops):
                l_mcm += 1

        measurement_ratio = l_mcm / d
        return measurement_ratio

    def _calculate_liveness(self) -> float:
        """
        Calculate the liveness metric.

        This metric captures qubit activity patterns during circuit execution.
        It computes the ratio of active qubit-time steps to total qubit-time steps.

        Formula: L = Σ(Aij) / (n*d)
        where A is the liveness matrix (1 if qubit i is active at time j, 0 otherwise),
        n is the number of qubits, and d is the circuit depth.

        Returns:
            float: Liveness ratio (0-1)
        """
        if self.circuit.num_qubits == 0 or self.circuit.depth() == 0:
            return 0.0

        layers = list(self._dag_circuit.layers())  # cada elemento representa un paso del circuito
        n_qubits = self.circuit.num_qubits
        depth = len(layers)

        if depth == 0 or n_qubits == 0:
            return 0.0

        # Inicializar matriz A (n_qubits x depth)
        active_counts = 0

        for layer in layers:
            # qubits activos en este paso
            active_qubits = set()
            for op in layer["graph"].op_nodes():
                if op.name in {"barrier"}:
                    continue  # no cuentan como actividad útil
                for qarg in op.qargs:
                    active_qubits.add(self.circuit.find_bit(qarg).index)
            active_counts += len(active_qubits)

        # L = promedio de actividad sobre todos los qubits y pasos
        liveness = active_counts / (n_qubits * depth)
        return liveness

    def _calculate_parallelism(self) -> float:
        """
        Calculate the parallelism metric.

        This metric represents cross-talk susceptibility by comparing the ratios
        of qubits, gates, and circuit depth. It indicates how susceptible a
        circuit is to cross-talk effects.

        Formula: P = (ng/d - 1) / (n - 1)
        where ng is the number of gates, d is the circuit depth, and n is the number of qubits.

        Returns:
            float: Parallelism factor (0-1)
        """
        n = self.circuit.num_qubits
        d = self.circuit.depth()
        ng = len(
            [
                inst
                for inst in self.circuit.data
                if inst.operation.name not in ("barrier", "measure", "reset")
            ]
        )

        if n <= 1 or d <= 0:
            return 0.0

        # Calculate parallelism factor
        gates_per_depth = ng / d
        parallelism = (gates_per_depth - 1) / (n - 1)

        # Clamp to [0, 1] range
        return max(0.0, min(1.0, parallelism))

    def _ensure_schemas_available(self):
        """
        Ensure that the required schemas are available.

        Raises:
            ImportError: If schemas are not available
        """
        if not SCHEMAS_AVAILABLE:
            raise ImportError(
                "BehavioralMetricsSchema is not available. "
                "Please ensure that the schemas module is properly imported."
            )
