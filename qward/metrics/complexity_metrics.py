"""
Complexity metrics implementation for QWARD.

This module provides the ComplexityMetrics class for analyzing quantum circuits
and extracting various complexity metrics as described in research literature.
The metrics include gate-based metrics, entanglement metrics, standardized metrics,
advanced metrics, and derived metrics.

Reference:
    [1] D. Shami, "Character Complexity: A Novel Measure for Quantum Circuit Analysis,"
        Sep. 18, 2024, arXiv: arXiv:2408.09641. doi: 10.48550/arXiv.2408.09641.
"""

from typing import Any, Dict, List

from qiskit import QuantumCircuit

from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsType, MetricsId

# Import schemas for structured data validation
try:
    from qward.metrics.schemas import (
        ComplexityMetricsSchema,
        GateBasedMetricsSchema,
        EntanglementMetricsSchema,
        StandardizedMetricsSchema,
        AdvancedMetricsSchema,
        DerivedMetricsSchema,
    )

    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False

# =============================================================================
# Gate Classification Constants
# =============================================================================

# Single-qubit gates
SINGLE_QUBIT_GATES = [
    "id",
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
]

# Two-qubit gates
TWO_QUBIT_GATES = [
    "cx",
    "cz",
    "swap",
    "iswap",
    "cp",
    "cu",
    "rxx",
    "ryy",
    "rzz",
    "crx",
    "cry",
    "crz",
]

# Clifford gates (can be efficiently simulated classically)
CLIFFORD_GATES = ["h", "s", "sdg", "cx", "cz", "x", "y", "z"]

# Non-computational gates (don't affect quantum state)
NON_COMPUTATIONAL_GATES = ["barrier", "measure"]

# Gate complexity weights for weighted complexity calculation
GATE_WEIGHTS = {
    # Single-qubit gates (basic)
    "id": 1,
    "x": 1,
    "y": 1,
    "z": 1,
    "h": 1,
    "s": 1,
    "sdg": 1,
    # Single-qubit gates (parameterized)
    "t": 2,
    "tdg": 2,
    "rx": 2,
    "ry": 2,
    "rz": 2,
    "p": 2,
    "u1": 2,
    "u2": 3,
    "u3": 4,
    # Two-qubit gates
    "cx": 10,
    "cz": 10,
    "swap": 12,
    "cp": 12,
    # Multi-qubit gates
    "ccx": 30,
    "cswap": 32,
    "mcx": 40,
    # Default weight for unknown gates
}
DEFAULT_GATE_WEIGHT = 5


class ComplexityMetrics(MetricCalculator):
    """
    Calculate complexity metrics from QuantumCircuit objects.

    This class provides methods for analyzing quantum circuits and extracting
    various complexity metrics including gate-based metrics, entanglement metrics,
    standardized metrics, advanced metrics, and derived metrics.

    The complexity metrics help characterize the computational difficulty and
    resource requirements of quantum circuits.
    """

    def _get_metric_type(self) -> MetricsType:
        """Get the type of this metric."""
        return MetricsType.PRE_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        """Get the ID of this metric."""
        return MetricsId.COMPLEXITY

    def is_ready(self) -> bool:
        """Check if the metric is ready to be calculated."""
        return self.circuit is not None

    def _ensure_schemas_available(self) -> None:
        """Ensure Pydantic schemas are available, raise ImportError if not."""
        if not SCHEMAS_AVAILABLE:
            raise ImportError(
                "Pydantic schemas are not available. Install pydantic to use structured metrics."
            )

    # =============================================================================
    # Main API Methods
    # =============================================================================

    def get_metrics(self) -> "ComplexityMetricsSchema":
        """
        Get all complexity metrics as a structured, validated schema object.

        Returns:
            ComplexityMetricsSchema: Complete validated complexity metrics schema

        Raises:
            ImportError: If Pydantic schemas are not available
            ValidationError: If metrics data doesn't match schema constraints
        """
        self._ensure_schemas_available()

        return ComplexityMetricsSchema(
            gate_based_metrics=self.get_gate_based_metrics(),
            entanglement_metrics=self.get_entanglement_metrics(),
            standardized_metrics=self.get_standardized_metrics(),
            advanced_metrics=self.get_advanced_metrics(),
            derived_metrics=self.get_derived_metrics(),
        )

    # =============================================================================
    # Gate-Based Metrics
    # =============================================================================

    def get_gate_based_metrics(self) -> "GateBasedMetricsSchema":
        """
        Get gate-based complexity metrics as a validated schema object.

        Returns:
            GateBasedMetricsSchema: Validated gate-based metrics
        """
        self._ensure_schemas_available()

        op_counts = self.circuit.count_ops()
        gate_count = self.circuit.size()
        circuit_depth = self.circuit.depth()

        # T-count (number of T gates) - important for fault-tolerant quantum computing
        t_count = op_counts.get("t", 0) + op_counts.get("tdg", 0)

        # CNOT count - most common two-qubit gate
        cnot_count = op_counts.get("cx", 0)

        # Two-qubit gate count
        two_qubit_count = self._count_gates_by_type(op_counts, TWO_QUBIT_GATES)

        # Multi-qubit gate ratio
        single_qubit_count = self._count_gates_by_type(op_counts, SINGLE_QUBIT_GATES)
        multi_qubit_count = self._calculate_multi_qubit_count(
            gate_count, single_qubit_count, op_counts
        )
        multi_qubit_ratio = multi_qubit_count / gate_count if gate_count > 0 else 0

        return GateBasedMetricsSchema(
            gate_count=gate_count,
            circuit_depth=circuit_depth,
            t_count=t_count,
            cnot_count=cnot_count,
            two_qubit_count=two_qubit_count,
            multi_qubit_ratio=round(multi_qubit_ratio, 3),
        )

    def get_gate_based_metrics_dict(self) -> Dict[str, Any]:
        """
        Get gate-based complexity metrics as a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing gate counts and ratios
        """
        return self.get_gate_based_metrics().model_dump()

    # =============================================================================
    # Entanglement Metrics
    # =============================================================================

    def get_entanglement_metrics(self) -> "EntanglementMetricsSchema":
        """
        Get entanglement-based complexity metrics as a validated schema object.

        Returns:
            EntanglementMetricsSchema: Validated entanglement metrics
        """
        self._ensure_schemas_available()

        op_counts = self.circuit.count_ops()
        gate_count = self.circuit.size()
        width = self.circuit.num_qubits

        # Two-qubit gate count (entangling operations)
        two_qubit_count = self._count_gates_by_type(op_counts, TWO_QUBIT_GATES)

        # Entangling gate density
        entangling_gate_density = two_qubit_count / gate_count if gate_count > 0 else 0

        # Entangling width (estimate of qubits involved in entanglement)
        entangling_width = min(width, two_qubit_count + 1) if two_qubit_count > 0 else 1

        return EntanglementMetricsSchema(
            entangling_gate_density=round(entangling_gate_density, 3),
            entangling_width=entangling_width,
        )

    def get_entanglement_metrics_dict(self) -> Dict[str, Any]:
        """
        Get entanglement-based complexity metrics as a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing entanglement-related metrics
        """
        return self.get_entanglement_metrics().model_dump()

    # =============================================================================
    # Standardized Metrics
    # =============================================================================

    def get_standardized_metrics(self) -> "StandardizedMetricsSchema":
        """
        Get standardized complexity metrics as a validated schema object.

        Returns:
            StandardizedMetricsSchema: Validated standardized metrics
        """
        self._ensure_schemas_available()

        depth = self.circuit.depth()
        width = self.circuit.num_qubits
        op_counts = self.circuit.count_ops()

        # Use sum of operation counts instead of circuit.size() to handle all operations consistently
        total_operations = sum(op_counts.values())

        # Circuit volume (depth × width)
        circuit_volume = depth * width

        # Gate density (gates per qubit-time-step)
        gate_density = total_operations / circuit_volume if circuit_volume > 0 else 0

        # Clifford vs non-Clifford ratio (only considering computational gates)
        clifford_count = self._count_gates_by_type(op_counts, CLIFFORD_GATES)
        non_computational_count = self._count_gates_by_type(op_counts, NON_COMPUTATIONAL_GATES)
        computational_gate_count = total_operations - non_computational_count
        non_clifford_count = max(0, computational_gate_count - clifford_count)

        # Calculate ratios based on computational gates only
        clifford_ratio = (
            clifford_count / computational_gate_count if computational_gate_count > 0 else 0
        )
        non_clifford_ratio = (
            non_clifford_count / computational_gate_count if computational_gate_count > 0 else 0
        )

        return StandardizedMetricsSchema(
            circuit_volume=circuit_volume,
            gate_density=round(gate_density, 3),
            clifford_ratio=round(clifford_ratio, 3),
            non_clifford_ratio=round(non_clifford_ratio, 3),
        )

    def get_standardized_metrics_dict(self) -> Dict[str, Any]:
        """
        Get standardized complexity metrics as a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing standardized circuit metrics
        """
        return self.get_standardized_metrics().model_dump()

    # =============================================================================
    # Advanced Metrics
    # =============================================================================

    def get_advanced_metrics(self) -> "AdvancedMetricsSchema":
        """
        Get advanced complexity metrics as a validated schema object.

        Returns:
            AdvancedMetricsSchema: Validated advanced metrics
        """
        self._ensure_schemas_available()

        depth = self.circuit.depth()
        width = self.circuit.num_qubits
        gate_count = self.circuit.size()

        # Parallelism factor (average gates per time step)
        parallelism_factor = gate_count / depth if depth > 0 else 0
        max_parallelism = width  # Maximum gates that could be executed in parallel
        parallelism_efficiency = parallelism_factor / max_parallelism if max_parallelism > 0 else 0

        # Circuit efficiency (utilization of qubit-time resources)
        circuit_efficiency = gate_count / (width * depth) if (width * depth) > 0 else 0

        # Quantum resource utilization (balanced metric considering both dimensions)
        quantum_resource_utilization = 0.5 * (
            gate_count / (width * width) if width > 0 else 0
        ) + 0.5 * (gate_count / (depth * depth) if depth > 0 else 0)

        return AdvancedMetricsSchema(
            parallelism_factor=round(parallelism_factor, 3),
            parallelism_efficiency=round(parallelism_efficiency, 3),
            circuit_efficiency=round(circuit_efficiency, 3),
            quantum_resource_utilization=round(quantum_resource_utilization, 3),
        )

    def get_advanced_metrics_dict(self) -> Dict[str, Any]:
        """
        Get advanced complexity metrics as a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing advanced circuit analysis metrics
        """
        return self.get_advanced_metrics().model_dump()

    # =============================================================================
    # Derived Metrics
    # =============================================================================

    def get_derived_metrics(self) -> "DerivedMetricsSchema":
        """
        Get derived complexity metrics as a validated schema object.

        Returns:
            DerivedMetricsSchema: Validated derived metrics
        """
        self._ensure_schemas_available()

        depth = self.circuit.depth()
        width = self.circuit.num_qubits
        op_counts = self.circuit.count_ops()

        # Square circuit factor (how close to square the circuit is)
        square_ratio = min(depth, width) / max(depth, width) if max(depth, width) > 0 else 1.0

        # Weighted gate complexity
        weighted_complexity = sum(
            count * GATE_WEIGHTS.get(gate, DEFAULT_GATE_WEIGHT) for gate, count in op_counts.items()
        )

        # Normalized weighted complexity (per qubit)
        normalized_weighted_complexity = weighted_complexity / width if width > 0 else 0

        return DerivedMetricsSchema(
            square_ratio=round(square_ratio, 3),
            weighted_complexity=weighted_complexity,
            normalized_weighted_complexity=round(normalized_weighted_complexity, 3),
        )

    def get_derived_metrics_dict(self) -> Dict[str, Any]:
        """
        Get derived complexity metrics as a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing derived circuit metrics
        """
        return self.get_derived_metrics().model_dump()

    # =============================================================================
    # Helper Methods
    # =============================================================================

    def _count_gates_by_type(self, op_counts: Dict[str, int], gate_list: List[str]) -> int:
        """Count gates of specific types from operation counts."""
        return sum(op_counts.get(gate, 0) for gate in gate_list)

    def _calculate_multi_qubit_count(
        self, gate_count: int, single_qubit_count: int, op_counts: Dict[str, int]
    ) -> int:
        """Calculate the number of multi-qubit gates."""
        non_computational_count = self._count_gates_by_type(op_counts, NON_COMPUTATIONAL_GATES)
        return gate_count - single_qubit_count - non_computational_count
