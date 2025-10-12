"""
Structural Metrics implementation for QWARD.

This module provides the StructuralMetrics class that unifies:
- LOC metrics (Lines of Code related metrics)
- Halstead metrics (complexity metrics) 
- Circuit structure metrics (width, depth, density, size)

This provides a comprehensive structural analysis of quantum circuits.

[1] J. Zhao, “Some Size and Structure Metrics for Quantum Software,” 
arXiv preprint arXiv:2103.08815, 2021.

"""

import math
from typing import Dict, Set, Any, List, Tuple, Optional

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.circuit import Parameter

from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsType, MetricsId

# Import schemas for structured data validation
try:
    from qward.schemas.structural_metrics_schema import StructuralMetricsSchema
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False

# =============================================================================
# Quantum Gate Classification Constants (from Halstead metrics)
# =============================================================================

# Quantum operators (gates)
QUANTUM_OPERATORS = {
    # Single-qubit gates
    "x", "y", "z", "h", "s", "sdg", "t", "tdg",
    "rx", "ry", "rz", "u1", "u2", "u3", "p", "u",
    "sx", "sxdg", "rzx", "phase", "reset",
    # Two-qubit gates
    "cx", "cy", "cz", "swap", "iswap", "dcx", "ecr",
    "rxx", "ryy", "rzz", "rzx", "xx_minus_yy", "xx_plus_yy",
    # Multi-qubit gates
    "ccx", "cswap", "mcx", "mcphase", "mcu1", "mcu2", "mcu3",
    "mcrx", "mcry", "mcrz", "mcp", "mcu", "mcswap"
}

# Classical operators
CLASSICAL_OPERATORS = {"measure", "barrier", "delay", "snapshot", "save", "initialize", "finalize"}

# Quantum operands (qubits)
QUANTUM_OPERANDS = {"q", "qubit"}  # Qubit references

# Classical operands
CLASSICAL_OPERANDS = {"c", "clbit", "cbit"}  # Classical bit references

# Parameter operands
PARAMETER_OPERANDS = {"theta", "phi", "lambda", "gamma", "beta", "alpha"}


class StructuralMetrics(MetricCalculator):
    """
    Structural Metrics calculator for QWARD.
    
    This class unifies LOC, Halstead, and circuit structure metrics
    to provide comprehensive structural analysis of quantum circuits.
    """

    @property 
    def id(self):
        return MetricsId.STRUCTURAL.value

    def __init__(self, circuit: QuantumCircuit):
        """
        Initialize the StructuralMetrics calculator.
        
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
            MetricsId: STRUCTURAL
        """
        return MetricsId.STRUCTURAL

    def is_ready(self) -> bool:
        """
        Check if the metric is ready to be calculated.
        
        Returns:
            bool: True if the circuit is available, False otherwise
        """
        return self.circuit is not None

    def get_metrics(self) -> StructuralMetricsSchema:
        """
        Calculate and return structural metrics combining LOC, Halstead, and circuit structure.
        
        Returns:
            StructuralMetricsSchema: Validated schema with all metrics
            
        Raises:
            ImportError: If schemas are not available
        """
        if not SCHEMAS_AVAILABLE:
            raise ImportError("StructuralMetricsSchema is not available")

        # Calculate LOC metrics
        loc_metrics = self._calculate_loc_metrics()
        
        # Calculate Halstead metrics
        halstead_metrics = self._calculate_halstead_metrics()
        
        # Calculate circuit structure metrics
        structure_metrics = self._calculate_structure_metrics()

        return StructuralMetricsSchema(
            # LOC Metrics
            phi1_total_loc=loc_metrics["phi1_total_loc"],
            phi2_gate_loc=loc_metrics["phi2_gate_loc"],
            phi3_measure_loc=loc_metrics["phi3_measure_loc"],
            phi4_quantum_total_loc=loc_metrics["phi4_quantum_total_loc"],
            phi5_num_qubits=loc_metrics["phi5_num_qubits"],
            phi6_num_gate_types=loc_metrics["phi6_num_gate_types"],
            
            # Halstead Metrics
            unique_operators=halstead_metrics["unique_operators"],
            unique_operands=halstead_metrics["unique_operands"],
            total_operators=halstead_metrics["total_operators"],
            total_operands=halstead_metrics["total_operands"],
            program_length=halstead_metrics["program_length"],
            vocabulary=halstead_metrics["vocabulary"],
            estimated_length=halstead_metrics["estimated_length"],
            volume=halstead_metrics["volume"],
            difficulty=halstead_metrics["difficulty"],
            effort=halstead_metrics["effort"],
            quantum_operators=halstead_metrics["quantum_operators"],
            classical_operators=halstead_metrics["classical_operators"],
            quantum_operands=halstead_metrics["quantum_operands"],
            classical_operands=halstead_metrics["classical_operands"],
            gate_types=halstead_metrics["gate_types"],
            qubit_operands=halstead_metrics["qubit_operands"],
            classical_bit_operands=halstead_metrics["classical_bit_operands"],
            parameter_operands=halstead_metrics["parameter_operands"],
            
            # Circuit Structure Metrics
            width=structure_metrics["width"],
            depth=structure_metrics["depth"],
            max_dens=structure_metrics["max_dens"],
            avg_dens=structure_metrics["avg_dens"],
            size=structure_metrics["size"]
        )

    def _calculate_loc_metrics(self) -> Dict[str, Any]:
        """
        Calculate LOC (Lines of Code) related metrics.
        
        Returns:
            Dict[str, Any]: Dictionary with LOC metrics
        """
            # Fallback: Estimate from circuit instructions if QASM is unavailable
        (
            phi1_total_loc,
            phi2_gate_loc,
            phi3_measure_loc,
            phi4_quantum_total_loc,
            ) = self._estimate_loc_from_circuit()

        # ϕ5: Número de cúbits usados
        phi5_num_qubits = self.circuit.num_qubits

        # ϕ6: Número del tipo de compuertas usadas
        unique_gates = set()
        for instruction in self.circuit.data:
            gate_name = instruction.operation.name
            if gate_name not in {"measure", "barrier", "reset"}:
                unique_gates.add(gate_name)
        phi6_num_gate_types = len(unique_gates)

        return {
            "phi1_total_loc": phi1_total_loc,
            "phi2_gate_loc": phi2_gate_loc,
            "phi3_measure_loc": phi3_measure_loc,
            "phi4_quantum_total_loc": phi4_quantum_total_loc,
            "phi5_num_qubits": phi5_num_qubits,
            "phi6_num_gate_types": phi6_num_gate_types,
        }

    def _calculate_halstead_metrics(self) -> Dict[str, Any]:
        """
        Calculate Halstead complexity metrics.
        
        Returns:
            Dict[str, Any]: Dictionary with Halstead metrics
        """
        # Initialize counters
        operators = set()
        operands = set()
        total_operators = 0
        total_operands = 0
        
        quantum_operators = set()
        classical_operators = set()
        quantum_operands = set()
        classical_operands = set()
        parameter_operands = set()
        
        gate_types = {}
        qubit_operands = set()
        classical_bit_operands = set()

        # Analyze circuit instructions
        for instruction in self.circuit.data:
            gate_name = instruction.operation.name.lower()
            
            # Count gate types
            gate_types[gate_name] = gate_types.get(gate_name, 0) + 1
            
            # Classify operators
            if gate_name in QUANTUM_OPERATORS:
                operators.add(gate_name)
                quantum_operators.add(gate_name)
            elif gate_name in CLASSICAL_OPERATORS:
                operators.add(gate_name)
                classical_operators.add(gate_name)
            else:
                # Unknown gate, treat as quantum operator
                operators.add(gate_name)
                quantum_operators.add(gate_name)
            
            total_operators += 1
            
            # Analyze operands (qubits and classical bits)
            for qubit in instruction.qubits:
                qubit_index = self.circuit.find_bit(qubit).index
                qubit_ref = f"q{qubit_index}"
                operands.add(qubit_ref)
                quantum_operands.add(qubit_ref)
                qubit_operands.add(qubit_ref)
                total_operands += 1
            
            for clbit in instruction.clbits:
                clbit_index = self.circuit.find_bit(clbit).index
                clbit_ref = f"c{clbit_index}"
                operands.add(clbit_ref)
                classical_operands.add(clbit_ref)
                classical_bit_operands.add(clbit_ref)
                total_operands += 1
            
            # Analyze parameters
            for param in instruction.operation.params:
                if isinstance(param, Parameter):
                    param_name = param.name.lower()
                    # Check if parameter name matches known patterns
                    for pattern in PARAMETER_OPERANDS:
                        if pattern in param_name:
                            parameter_operands.add(param_name)
                            operands.add(param_name)
                            total_operands += 1
                            break

        # Calculate derived metrics
        unique_operators = len(operators)
        unique_operands = len(operands)
        program_length = total_operators + total_operands
        vocabulary = unique_operators + unique_operands
        
        # Estimated length calculation
        if unique_operators > 0 and unique_operands > 0:
            estimated_length = (
                unique_operators * math.log2(unique_operators) +
                unique_operands * math.log2(unique_operands)
            )
        else:
            estimated_length = 0.0
        
        # Volume calculation
        if vocabulary > 0:
            volume = program_length * math.log2(vocabulary)
        else:
            volume = 0.0
        
        # Difficulty calculation
        if len(quantum_operands) > 0:
            difficulty = (len(quantum_operators) / 2) * (len(quantum_operands) / len(quantum_operands))
        else:
            difficulty = 0.0
        
        # Effort calculation
        effort = difficulty * volume

        return {
            "unique_operators": unique_operators,
            "unique_operands": unique_operands,
            "total_operators": total_operators,
            "total_operands": total_operands,
            "program_length": program_length,
            "vocabulary": vocabulary,
            "estimated_length": estimated_length,
            "volume": volume,
            "difficulty": difficulty,
            "effort": effort,
            "quantum_operators": len(quantum_operators),
            "classical_operators": len(classical_operators),
            "quantum_operands": len(quantum_operands),
            "classical_operands": len(classical_operands),
            "gate_types": gate_types,
            "qubit_operands": len(qubit_operands),
            "classical_bit_operands": len(classical_bit_operands),
            "parameter_operands": len(parameter_operands),
        }

    def _calculate_structure_metrics(self) -> Dict[str, Any]:
        """
        Calculate circuit structure metrics (width, depth, density, size).
        
        Returns:
            Dict[str, Any]: Dictionary with structure metrics
        """
        # Width: Number of qubits
        width = self.circuit.num_qubits
        
        # Depth: Maximum number of operations applied to any qubit
        depth = self.circuit.depth()
        
        # Size: Total number of operations/gates
        size = len(self.circuit.data)
        
        # Density calculations
        max_dens, avg_dens = self._calculate_circuit_density()

        return {
            "width": width,
            "depth": depth,
            "max_dens": max_dens,
            "avg_dens": avg_dens,
            "size": size
        }

    def _calculate_circuit_density(self) -> Tuple[int, float]:
        """
        Calculate the maximum and average number of operations applied to qubits.
        
        Returns:
            Tuple[int, float]: (maximum_density, average_density)
        """
        layer_densities = []

        for layer in self._circuit_dag.layers():
            ops = [
                node for node in layer['graph'].op_nodes()
                if node.op.name not in ["measure", "barrier", "reset"]    
            ]
            
            layer_densities.append(len(ops))

        if not layer_densities:
            return 0, 0.0

        max_dens = max(layer_densities)
        avg_dens = sum(layer_densities) / len(layer_densities)
        
        return max_dens, avg_dens


    def _estimate_loc_from_circuit(self) -> Tuple[int, int, int, int]:
        """
        Estimate LOC metrics from circuit instructions
        
        Returns:
            Tuple[int, int, int, int]: (total_loc, gate_loc, measure_loc, quantum_total_loc)
        """
        total_loc = len(self.circuit.data)
        gate_loc = 0
        measure_loc = 0
        
        for instruction in self.circuit.data:
            gate_name = instruction.operation.name.lower()
            if gate_name == "measure":
                measure_loc += 1
            elif gate_name in QUANTUM_OPERATORS:
                gate_loc += 1
            else:
                # Treat unknown gates as quantum gates
                gate_loc += 1
        
        quantum_total_loc = gate_loc + measure_loc
        
        return total_loc, gate_loc, measure_loc, quantum_total_loc

    def _ensure_schemas_available(self):
        """
        Ensure that the required schemas are available.
        
        Raises:
            ImportError: If schemas are not available
        """
        if not SCHEMAS_AVAILABLE:
            raise ImportError(
                "StructuralMetricsSchema is not available. "
                "Please ensure that the schemas module is properly imported."
            )
