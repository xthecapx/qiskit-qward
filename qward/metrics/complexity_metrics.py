"""
Complexity metrics implementation for QWARD.
"""

from typing import Any, Dict

from qiskit import QuantumCircuit

from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsType, MetricsId


class ComplexityMetrics(MetricCalculator):
    """
    Class for calculating complexity metrics from QuantumCircuit objects.

    This class provides methods for analyzing quantum circuits and extracting
    various complexity metrics as described in the research paper:

    [1] D. Shami, "Character Complexity: A Novel Measure for Quantum Circuit Analysis,"
    Sep. 18, 2024, arXiv: arXiv:2408.09641. doi: 10.48550/arXiv.2408.09641.

    The metrics include gate-based metrics, entanglement metrics, standardized metrics,
    advanced metrics, and derived metrics.
    """

    def _get_metric_type(self) -> MetricsType:
        """
        Get the type of this metric.

        Returns:
            MetricsType: The type of this metric
        """
        return MetricsType.PRE_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        """
        Get the ID of this metric.

        Returns:
            MetricsId: The ID of this metric
        """
        return MetricsId.COMPLEXITY

    def is_ready(self) -> bool:
        """
        Check if the metric is ready to be calculated.

        Returns:
            bool: True if the metric is ready to be calculated, False otherwise
        """
        return self.circuit is not None

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the metrics.

        Returns:
            Dict[str, Any]: Dictionary containing the metrics
        """

        return {
            "gate_based_metrics": self.get_gate_based_metrics(),
            "entanglement_metrics": self.get_entanglement_metrics(),
            "standardized_metrics": self.get_standardized_metrics(),
            "advanced_metrics": self.get_advanced_metrics(),
            "derived_metrics": self.get_derived_metrics(),
            "quantum_volume": self.estimate_quantum_volume(),
        }

    def get_gate_based_metrics(self) -> Dict[str, Any]:
        """
        Get gate-based metrics about the circuit.

        Returns:
            Dict[str, Any]: Dictionary containing gate-based metrics
        """
        op_counts = self._circuit.count_ops()
        gate_count = self._circuit.size()
        circuit_depth = self._circuit.depth()

        # T-count (number of T gates)
        t_count = op_counts.get("t", 0) + op_counts.get("tdg", 0)

        # CNOT count
        cnot_count = op_counts.get("cx", 0)

        # Two-qubit gate count
        two_qubit_gates = [
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
        two_qubit_count = sum(op_counts.get(gate, 0) for gate in two_qubit_gates)

        # Multi-qubit gate ratio
        single_qubit_gates = [
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
        single_qubit_count = sum(op_counts.get(gate, 0) for gate in single_qubit_gates)
        multi_qubit_count = (
            gate_count
            - single_qubit_count
            - op_counts.get("barrier", 0)
            - op_counts.get("measure", 0)
        )
        multi_qubit_ratio = multi_qubit_count / gate_count if gate_count > 0 else 0

        return {
            "gate_count": gate_count,
            "circuit_depth": circuit_depth,
            "t_count": t_count,
            "cnot_count": cnot_count,
            "two_qubit_count": two_qubit_count,
            "multi_qubit_ratio": round(multi_qubit_ratio, 3),
        }

    def get_entanglement_metrics(self) -> Dict[str, Any]:
        """
        Get entanglement-based metrics about the circuit.

        Returns:
            Dict[str, Any]: Dictionary containing entanglement metrics
        """
        op_counts = self._circuit.count_ops()
        gate_count = self._circuit.size()
        width = self._circuit.num_qubits

        # Two-qubit gate count
        two_qubit_gates = [
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
        two_qubit_count = sum(op_counts.get(gate, 0) for gate in two_qubit_gates)

        # Entangling gate density
        entangling_gate_density = two_qubit_count / gate_count if gate_count > 0 else 0

        # Entangling width
        entangling_width = min(width, two_qubit_count + 1) if two_qubit_count > 0 else 1

        return {
            "entangling_gate_density": round(entangling_gate_density, 3),
            "entangling_width": entangling_width,
        }

    def get_standardized_metrics(self) -> Dict[str, Any]:
        """
        Get standardized metrics about the circuit.

        Returns:
            Dict[str, Any]: Dictionary containing standardized metrics
        """
        depth = self._circuit.depth()
        width = self._circuit.num_qubits
        gate_count = self._circuit.size()
        op_counts = self._circuit.count_ops()

        # Circuit volume (depth × width)
        circuit_volume = depth * width

        # Gate density (gates per qubit-time-step)
        gate_density = gate_count / circuit_volume if circuit_volume > 0 else 0

        # Clifford vs non-Clifford ratio
        clifford_gates = ["h", "s", "sdg", "cx", "cz", "x", "y", "z"]
        clifford_count = sum(op_counts.get(gate, 0) for gate in clifford_gates)
        non_clifford_count = (
            gate_count - clifford_count - op_counts.get("barrier", 0) - op_counts.get("measure", 0)
        )
        clifford_ratio = clifford_count / gate_count if gate_count > 0 else 0
        non_clifford_ratio = non_clifford_count / gate_count if gate_count > 0 else 0

        return {
            "circuit_volume": circuit_volume,
            "gate_density": round(gate_density, 3),
            "clifford_ratio": round(clifford_ratio, 3),
            "non_clifford_ratio": round(non_clifford_ratio, 3),
        }

    def get_advanced_metrics(self) -> Dict[str, Any]:
        """
        Get advanced metrics about the circuit.

        Returns:
            Dict[str, Any]: Dictionary containing advanced metrics
        """
        depth = self._circuit.depth()
        width = self._circuit.num_qubits
        gate_count = self._circuit.size()

        # Parallelism factor
        parallelism_factor = gate_count / depth if depth > 0 else 0
        max_parallelism = width  # Maximum gates that could be executed in parallel
        parallelism_efficiency = parallelism_factor / max_parallelism if max_parallelism > 0 else 0

        # Circuit efficiency
        circuit_efficiency = gate_count / (width * depth) if (width * depth) > 0 else 0

        # Quantum resource utilization
        quantum_resource_utilization = 0.5 * (
            gate_count / (width * width) if width > 0 else 0
        ) + 0.5 * (gate_count / (depth * depth) if depth > 0 else 0)

        return {
            "parallelism_factor": round(parallelism_factor, 3),
            "parallelism_efficiency": round(parallelism_efficiency, 3),
            "circuit_efficiency": round(circuit_efficiency, 3),
            "quantum_resource_utilization": round(quantum_resource_utilization, 3),
        }

    def get_derived_metrics(self) -> Dict[str, Any]:
        """
        Get derived metrics about the circuit.

        Returns:
            Dict[str, Any]: Dictionary containing derived metrics
        """
        depth = self._circuit.depth()
        width = self._circuit.num_qubits
        op_counts = self._circuit.count_ops()

        # Square circuit factor
        square_ratio = min(depth, width) / max(depth, width) if max(depth, width) > 0 else 1.0

        # Weighted gate complexity
        gate_weights = {
            # Single-qubit gates
            "id": 1,
            "x": 1,
            "y": 1,
            "z": 1,
            "h": 1,
            "s": 1,
            "sdg": 1,
            # More complex single-qubit gates
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
            # Others default to 5
        }

        weighted_complexity = sum(
            count * gate_weights.get(gate, 5) for gate, count in op_counts.items()
        )

        # Normalized weighted complexity (per qubit)
        normalized_weighted_complexity = weighted_complexity / width if width > 0 else 0

        return {
            "square_ratio": round(square_ratio, 3),
            "weighted_complexity": weighted_complexity,
            "normalized_weighted_complexity": round(normalized_weighted_complexity, 3),
        }

    def estimate_quantum_volume(self) -> Dict[str, Any]:
        """
        Estimate the quantum volume of the current circuit.

        This is a circuit complexity metric based on the existing circuit's
        characteristics rather than the formal IBM Quantum Volume protocol.

        Returns:
            Dict[str, Any]: Dictionary containing quantum volume estimates
        """
        # Get circuit metrics
        depth = self._circuit.depth()
        width = self._circuit.width()
        num_qubits = self._circuit.num_qubits
        size = self._circuit.size()
        op_counts = self._circuit.count_ops()

        # Start with baseline QV calculation based on effective square size
        effective_depth = min(depth, num_qubits)

        # Calculate standard QV base as 2^n where n is effective depth
        standard_qv = 2**effective_depth

        # Calculate complexity factors

        # 1. Square circuit factor - how close is it to a square circuit?
        square_ratio = min(depth, width) / max(depth, width) if max(depth, width) > 0 else 1.0

        # 2. Circuit density - how many operations per qubit-timestep?
        max_possible_ops = depth * width
        density = size / max_possible_ops if max_possible_ops > 0 else 0.0

        # 3. Gate complexity - multi-qubit operations are more complex
        multi_qubit_ops = sum(
            count
            for gate, count in op_counts.items()
            if gate
            not in [
                "barrier",
                "measure",
                "id",
                "u1",
                "u2",
                "u3",
                "rx",
                "ry",
                "rz",
                "h",
                "x",
                "y",
                "z",
                "s",
                "t",
            ]
        )
        multi_qubit_ratio = multi_qubit_ops / size if size > 0 else 0.0

        # 4. Connectivity factor - for current circuit
        connectivity_factor = 0.5 + 0.5 * (multi_qubit_ratio > 0)

        # Calculate the enhanced quantum volume
        # Use factors to adjust the standard QV
        enhanced_factor = (
            0.4 * square_ratio  # Square circuits are foundational to QV
            + 0.3 * density  # Dense circuits are more complex
            + 0.2 * multi_qubit_ratio  # Multi-qubit operations increase complexity
            + 0.1 * connectivity_factor  # Connectivity affects feasibility
        )

        # Enhanced QV: apply enhancement factor to standard QV
        enhanced_qv = standard_qv * (1 + enhanced_factor)

        # Round to significant figures for clarity
        enhanced_qv_rounded = round(enhanced_qv, 2)

        return {
            "standard_quantum_volume": standard_qv,
            "enhanced_quantum_volume": enhanced_qv_rounded,
            "effective_depth": effective_depth,
            "factors": {
                "square_ratio": round(square_ratio, 2),
                "circuit_density": round(density, 2),
                "multi_qubit_ratio": round(multi_qubit_ratio, 2),
                "connectivity_factor": round(connectivity_factor, 2),
                "enhancement_factor": round(enhanced_factor, 2),
            },
            "circuit_metrics": {
                "depth": depth,
                "width": width,
                "size": size,
                "num_qubits": num_qubits,
                "operation_counts": op_counts,
            },
        }
