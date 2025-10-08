"""
Quantum Halstead metrics implementation for QWARD.

This module provides the QuantumHalsteadMetrics class for analyzing quantum circuits
and extracting Halstead metrics adapted for quantum software. These metrics extend
classical Halstead metrics to quantify various aspects of quantum software complexity
by considering both classical and quantum operators and operands.

The metrics are based on the following definitions:
- η1: number of unique classical and quantum operators
- η2: number of unique classical and quantum operands  
- M1: total occurrences of classical and quantum operators
- M2: total occurrences of classical and quantum operands

Derived metrics include:
- Program length: M = M1 + M2
- Vocabulary: η = η1 + η2
- Estimated length: ME = η1*log2(η1) + η2*log2(η2)
- Volume: VQ = M × log2(η)
- Difficulty: DQ = (η1/2) × (M2/η2)
- Effort: EQ = DQ × VQ

Reference: Extended Halstead metrics for quantum software analysis
            Zhao, J. (2021). Some size and structure metrics for quantum software. 
            arXiv. https://arxiv.org/abs/2103.08815
"""

import math
from typing import Dict, Set, Any

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsType, MetricsId

# Import schemas for structured data validation
try:
    from qward.schemas.quantum_halstead_metrics_schema \
    import QuantumHalsteadMetricsSchema
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False

# =============================================================================
# Quantum Gate Classification Constants
# =============================================================================

# Quantum operators (gates)
QUANTUM_OPERATORS = {
    # Single-qubit gates
    "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg",
    "rx", "ry", "rz", "u1", "u2", "u3", "p", "u",
    # Two-qubit gates  
    "cx", "cz", "swap", "iswap", "cp", "cu", "rxx", "ryy", "rzz",
    "crx", "cry", "crz", "ccx", "cswap", "csx", "csdg", "ct",
    # Multi-qubit gates
    "mcx", "mcphase", "mcu1", "mcu2", "mcu3",
    # Measurement and reset
    "measure", "reset"
}

# Classical operators (control flow, classical operations)
CLASSICAL_OPERATORS = {
    "if_else", "for_loop", "while_loop", "break", "continue",
    "classical_and", "classical_or", "classical_not", "classical_xor",
    "classical_assign", "classical_measure", "classical_reset"
}

# Quantum operands (qubits, classical bits, parameters)
QUANTUM_OPERANDS = {
    "qubit", "ancilla_qubit", "parameter", "angle", "phase"
}

# Classical operands (classical bits, variables, constants)
CLASSICAL_OPERANDS = {
    "classical_bit", "classical_variable", "constant", "integer", "float"
}


class QuantumHalsteadMetrics(MetricCalculator):
    """
    Quantum Halstead metrics calculator for QWARD.
    
    This class implements Halstead metrics adapted for quantum software,
    considering both classical and quantum components of quantum circuits.
    """

    def __init__(self, circuit: QuantumCircuit):
        """
        Initialize the QuantumHalsteadMetrics calculator.
        
        Args:
            circuit: The quantum circuit to analyze
        """
        super().__init__(circuit)
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
            MetricsId: QUANTUM_HALSTEAD
        """
        return MetricsId.QUANTUM_HALSTEAD

    def is_ready(self) -> bool:
        """
        Check if the metric is ready to be calculated.
        
        Returns:
            bool: True if the circuit is available, False otherwise
        """
        return self.circuit is not None

    def get_metrics(self) -> QuantumHalsteadMetricsSchema:
        """
        Calculate and return quantum Halstead metrics.
        
        Returns:
            QuantumHalsteadMetricsSchema: Validated schema with all metrics
            
        Raises:
            ImportError: If schemas are not available
        """
        if not SCHEMAS_AVAILABLE:
            raise ImportError("QuantumHalsteadMetricsSchema is not available")

        # Analyze the circuit to extract operators and operands
        operators, operands = self._analyze_circuit()
        # Calculate basic counts
        unique_operators = len(operators["unique"])
        unique_operands = len(operands["unique"])
        total_operators = operators["total"]
        total_operands = operands["total"]
        
        # Calculate derived metrics
        program_length = total_operators + total_operands
        vocabulary = unique_operators + unique_operands
        
        # Handle edge cases for logarithmic calculations
        if unique_operators > 0 and unique_operands > 0:
            estimated_length = (
                unique_operators * math.log2(unique_operators) +
                unique_operands * math.log2(unique_operands)
            )
        else:
            estimated_length = 0.0
        
        if vocabulary > 0:
            volume = program_length * math.log2(vocabulary)
        else:
            volume = 0.0
        
        if unique_operands > 0:
            difficulty = (unique_operators / 2) * (total_operands / unique_operands)
        else:
            difficulty = 0.0
        
        effort = difficulty * volume
        
        
        # Quantum-specific classifications
        quantum_operators = len([op for op in operators["unique"] if op in QUANTUM_OPERATORS])
        classical_operators = len([op for op in operators["unique"] if op in CLASSICAL_OPERATORS])
        quantum_operands = len([op for op in operands["unique"] if op in QUANTUM_OPERANDS])
        classical_operands = len([op for op in operands["unique"] if op in CLASSICAL_OPERANDS])
        
        # Circuit-specific metrics
        gate_types = self._count_gate_types()
        qubit_operands = self.circuit.num_qubits
        classical_bit_operands = self.circuit.num_clbits
        parameter_operands = len(self.circuit.parameters)
        
        return QuantumHalsteadMetricsSchema(
            unique_operators=unique_operators,
            unique_operands=unique_operands,
            total_operators=total_operators,
            total_operands=total_operands,
            program_length=program_length,
            vocabulary=vocabulary,
            estimated_length=estimated_length,
            volume=volume,
            difficulty=difficulty,
            effort=effort,
            quantum_operators=quantum_operators,
            classical_operators=classical_operators,
            quantum_operands=quantum_operands,
            classical_operands=classical_operands,
            gate_types=gate_types,
            qubit_operands=qubit_operands,
            classical_bit_operands=classical_bit_operands,
            parameter_operands=parameter_operands
        )

    def _analyze_circuit(self) -> Dict[str, Any]:
        """
        Analyze the quantum circuit to extract operators and operands.
        
        Returns:
            Dict[str, Any]: Dictionary with unique and total counts for operators and operands
        """
        operators = {"unique": set(), "total": 0}
        operands = {"unique": set(), "total": 0}
        
        # Analyze each instruction in the circuit
        for instruction in self.circuit.data:
            # Extract operator (gate name)
            gate_name = instruction.operation.name
            if gate_name not in ["barrier", "delay"]:
                operators["unique"].add(gate_name)
                operators["total"] += 1
            
                # Extract operands (qubits, classical bits, parameters)
                for qubit in instruction.qubits:
                    operands["unique"].add(qubit)
                    operands["total"] += 1
                
                for clbit in instruction.clbits:
                    operands["unique"].add(clbit)
                    operands["total"] += 1
                
                # Extract parameters
                for param in instruction.operation.params:
                    if isinstance(param, Parameter):
                        operands["unique"].add(param)
                        operands["total"] += 1
                    elif isinstance(param, (int, float)):
                        operands["unique"].add(param)
                        operands["total"] += 1
        
        # Add circuit-level operands
        """if self.circuit.num_qubits > 0:
            operands["unique"].add("qubit")
            operands["total"] += self.circuit.num_qubits
        
        if self.circuit.num_clbits > 0:
            operands["unique"].add("classical_bit")
            operands["total"] += self.circuit.num_clbits
        
        if self.circuit.parameters:
            operands["unique"].add("parameter")
            operands["total"] += len(self.circuit.parameters)
        """
        return operators, operands

    def _count_gate_types(self) -> Dict[str, int]:
        """
        Count the occurrences of each gate type in the circuit.
        
        Returns:
            Dict[str, int]: Dictionary mapping gate names to counts
        """
        gate_counts = {}
        
        for instruction in self.circuit.data:
            gate_name = instruction.operation.name
            if gate_name in QUANTUM_OPERATORS:
                gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        
        return gate_counts

    def _ensure_schemas_available(self):
        """
        Ensure that the required schemas are available.
        
        Raises:
            ImportError: If schemas are not available
        """
        if not SCHEMAS_AVAILABLE:
            raise ImportError(
                "QuantumHalsteadMetricsSchema is not available. "
                "Please ensure that the schemas module is properly imported."
            )
