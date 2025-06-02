"""
Metric schemas for QWARD using Pydantic for structured data validation.

This module provides schema classes that define the structure and validation
rules for different types of metrics in QWARD, inspired by dataframely's
approach to data validation.

The schemas provide:
- Type validation for all fields
- Automatic constraint checking (e.g., non-negative values)
- Clear documentation of expected data types
- JSON schema generation for API documentation
- Better IDE support with autocomplete
"""

from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, field_validator, ConfigDict
from qiskit.circuit import CircuitInstruction
from qiskit.transpiler import Layout


# =============================================================================
# Basic Circuit Metrics Schema
# =============================================================================


class BasicMetricsSchema(BaseModel):
    """
    Schema for basic quantum circuit metrics.

    This schema validates basic metrics that can be extracted directly
    from a QuantumCircuit object, such as depth, width, and gate counts.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "depth": 3,
                "width": 6,
                "size": 4,
                "num_qubits": 2,
                "num_clbits": 4,
                "num_ancillas": 0,
                "num_parameters": 0,
                "has_calibrations": False,
                "has_layout": False,
                "count_ops": {"h": 1, "cx": 1, "measure": 2, "barrier": 1},
            }
        }
    )

    depth: int = Field(..., ge=0, description="Circuit depth (number of time steps)")
    width: int = Field(..., ge=0, description="Circuit width (total qubits and classical bits)")
    size: int = Field(..., ge=0, description="Total number of operations in the circuit")
    num_qubits: int = Field(..., ge=0, description="Number of quantum bits")
    num_clbits: int = Field(..., ge=0, description="Number of classical bits")
    num_ancillas: int = Field(..., ge=0, description="Number of ancilla qubits")
    num_parameters: int = Field(..., ge=0, description="Number of parameters in the circuit")
    has_calibrations: bool = Field(..., description="Whether the circuit has calibration data")
    has_layout: bool = Field(..., description="Whether the circuit has layout information")
    count_ops: Dict[str, int] = Field(..., description="Count of each operation type")

    @field_validator("count_ops")
    @classmethod
    def validate_count_ops(cls, v):
        """Validate that all operation counts are non-negative."""
        for op_name, count in v.items():
            if count < 0:
                raise ValueError(
                    f"Operation count for '{op_name}' must be non-negative, got {count}"
                )
        return v


# =============================================================================
# Instruction Metrics Schema
# =============================================================================


class InstructionMetricsSchema(BaseModel):
    """
    Schema for quantum circuit instruction metrics.

    This schema validates metrics related to the instructions and
    connectivity of a quantum circuit.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow CircuitInstruction objects

    num_connected_components: int = Field(..., ge=1, description="Number of connected components")
    num_nonlocal_gates: int = Field(..., ge=0, description="Number of non-local gates")
    num_tensor_factors: int = Field(..., ge=1, description="Number of tensor factors")
    num_unitary_factors: int = Field(..., ge=1, description="Number of unitary factors")
    instructions: Dict[str, List[CircuitInstruction]] = Field(
        ..., description="Instructions grouped by operation name"
    )

    @field_validator("instructions")
    @classmethod
    def validate_instructions(cls, v):
        """Validate that instruction lists are not empty."""
        for op_name, instructions in v.items():
            if not instructions:
                raise ValueError(f"Instruction list for '{op_name}' cannot be empty")
        return v


# =============================================================================
# Scheduling Metrics Schema
# =============================================================================


class SchedulingMetricsSchema(BaseModel):
    """
    Schema for quantum circuit scheduling metrics.

    This schema validates metrics related to circuit scheduling and
    timing information.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow Layout objects

    is_scheduled: bool = Field(..., description="Whether the circuit is scheduled")
    layout: Optional[Layout] = Field(None, description="Circuit layout (if scheduled)")
    op_start_times: Optional[List[int]] = Field(
        None, description="Operation start times (if scheduled)"
    )
    qubit_duration: Optional[int] = Field(None, ge=0, description="Qubit duration (if scheduled)")
    qubit_start_time: Optional[int] = Field(
        None, ge=0, description="Qubit start time (if scheduled)"
    )
    qubit_stop_time: Optional[int] = Field(None, ge=0, description="Qubit stop time (if scheduled)")

    @field_validator("qubit_stop_time")
    @classmethod
    def validate_stop_after_start(cls, v, info):
        """Validate that stop time is after start time."""
        if (
            v is not None
            and "qubit_start_time" in info.data
            and info.data["qubit_start_time"] is not None
        ):
            if v < info.data["qubit_start_time"]:
                raise ValueError("Qubit stop time must be >= start time")
        return v


# =============================================================================
# Complete Qiskit Metrics Schema
# =============================================================================


class QiskitMetricsSchema(BaseModel):
    """
    Complete schema for all Qiskit-derived metrics.

    This schema combines all individual metric schemas into a single
    comprehensive structure for Qiskit metrics.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "basic_metrics": {
                    "depth": 3,
                    "width": 6,
                    "size": 4,
                    "num_qubits": 2,
                    "num_clbits": 4,
                    "num_ancillas": 0,
                    "num_parameters": 0,
                    "has_calibrations": False,
                    "has_layout": False,
                    "count_ops": {"h": 1, "cx": 1, "measure": 2, "barrier": 1},
                },
                "instruction_metrics": {
                    "num_connected_components": 1,
                    "num_nonlocal_gates": 1,
                    "num_tensor_factors": 1,
                    "num_unitary_factors": 1,
                    "instructions": {"h": [], "cx": []},
                },
                "scheduling_metrics": {
                    "is_scheduled": False,
                    "layout": None,
                    "op_start_times": None,
                    "qubit_duration": None,
                    "qubit_start_time": None,
                    "qubit_stop_time": None,
                },
            }
        }
    )

    basic_metrics: BasicMetricsSchema = Field(..., description="Basic circuit metrics")
    instruction_metrics: InstructionMetricsSchema = Field(
        ..., description="Instruction-related metrics"
    )
    scheduling_metrics: SchedulingMetricsSchema = Field(
        ..., description="Scheduling-related metrics"
    )

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert the schema to a flattened dictionary for DataFrame compatibility.

        Returns:
            Dict[str, Any]: Flattened dictionary with dot-notation keys
        """
        result: Dict[str, Any] = {}

        # Flatten basic metrics
        basic_dict = self.basic_metrics.model_dump()  # pylint: disable=no-member
        count_ops = basic_dict.pop("count_ops")
        for key, value in basic_dict.items():
            result[f"basic_metrics.{key}"] = value
        for op_name, count in count_ops.items():
            result[f"basic_metrics.count_ops.{op_name}"] = count

        # Flatten instruction metrics
        instruction_dict = self.instruction_metrics.model_dump()  # pylint: disable=no-member
        instructions = instruction_dict.pop("instructions")
        for key, value in instruction_dict.items():
            result[f"instruction_metrics.{key}"] = value
        for op_name, instruction_list in instructions.items():
            result[f"instruction_metrics.instructions.{op_name}"] = instruction_list

        # Flatten scheduling metrics
        scheduling_dict = self.scheduling_metrics.model_dump()  # pylint: disable=no-member
        for key, value in scheduling_dict.items():
            result[f"scheduling_metrics.{key}"] = value

        return result

    @classmethod
    def from_flat_dict(cls, flat_dict: Dict[str, Any]) -> "QiskitMetricsSchema":
        """
        Create a schema instance from a flattened dictionary.

        Args:
            flat_dict: Flattened dictionary with dot-notation keys

        Returns:
            QiskitMetricsSchema: Schema instance
        """
        # Reconstruct nested structure
        basic_metrics = {}
        instruction_metrics = {}
        scheduling_metrics = {}
        count_ops = {}
        instructions = {}

        for key, value in flat_dict.items():
            if key.startswith("basic_metrics."):
                if key.startswith("basic_metrics.count_ops."):
                    op_name = key.replace("basic_metrics.count_ops.", "")
                    count_ops[op_name] = value
                else:
                    metric_name = key.replace("basic_metrics.", "")
                    basic_metrics[metric_name] = value
            elif key.startswith("instruction_metrics."):
                if key.startswith("instruction_metrics.instructions."):
                    op_name = key.replace("instruction_metrics.instructions.", "")
                    instructions[op_name] = value
                else:
                    metric_name = key.replace("instruction_metrics.", "")
                    instruction_metrics[metric_name] = value
            elif key.startswith("scheduling_metrics."):
                metric_name = key.replace("scheduling_metrics.", "")
                scheduling_metrics[metric_name] = value

        # Add count_ops and instructions back
        basic_metrics["count_ops"] = count_ops
        instruction_metrics["instructions"] = instructions

        return cls(
            basic_metrics=BasicMetricsSchema(**basic_metrics),
            instruction_metrics=InstructionMetricsSchema(**instruction_metrics),
            scheduling_metrics=SchedulingMetricsSchema(**scheduling_metrics),
        )


# =============================================================================
# Complexity Metrics Schemas
# =============================================================================


class GateBasedMetricsSchema(BaseModel):
    """
    Schema for gate-based complexity metrics.

    This schema validates metrics related to gate counts and types
    in quantum circuits.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gate_count": 10,
                "circuit_depth": 5,
                "t_count": 2,
                "cnot_count": 3,
                "two_qubit_count": 3,
                "multi_qubit_ratio": 0.3,
            }
        }
    )

    gate_count: int = Field(..., ge=0, description="Total number of gates in the circuit")
    circuit_depth: int = Field(..., ge=0, description="Circuit depth (number of time steps)")
    t_count: int = Field(..., ge=0, description="Number of T gates (important for fault tolerance)")
    cnot_count: int = Field(..., ge=0, description="Number of CNOT gates")
    two_qubit_count: int = Field(..., ge=0, description="Total number of two-qubit gates")
    multi_qubit_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Ratio of multi-qubit gates to total gates"
    )

    @field_validator("multi_qubit_ratio")
    @classmethod
    def validate_ratio_bounds(cls, v):
        """Validate that ratio is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Multi-qubit ratio must be between 0.0 and 1.0, got {v}")
        return v


class EntanglementMetricsSchema(BaseModel):
    """
    Schema for entanglement-based complexity metrics.

    This schema validates metrics related to entanglement generation
    and quantum correlations in circuits.
    """

    model_config = ConfigDict(
        json_schema_extra={"example": {"entangling_gate_density": 0.3, "entangling_width": 4}}
    )

    entangling_gate_density: float = Field(
        ..., ge=0.0, le=1.0, description="Density of entangling gates"
    )
    entangling_width: int = Field(..., ge=1, description="Estimated width of entanglement")

    @field_validator("entangling_gate_density")
    @classmethod
    def validate_density_bounds(cls, v):
        """Validate that density is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Entangling gate density must be between 0.0 and 1.0, got {v}")
        return v


class StandardizedMetricsSchema(BaseModel):
    """
    Schema for standardized complexity metrics.

    This schema validates standardized metrics for comparing
    circuits of different sizes and structures.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "circuit_volume": 20,
                "gate_density": 0.5,
                "clifford_ratio": 0.7,
                "non_clifford_ratio": 0.3,
            }
        }
    )

    circuit_volume: int = Field(..., ge=0, description="Circuit volume (depth × width)")
    gate_density: float = Field(..., ge=0.0, description="Gate density (gates per qubit-time-step)")
    clifford_ratio: float = Field(..., ge=0.0, le=1.0, description="Ratio of Clifford gates")
    non_clifford_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Ratio of non-Clifford gates"
    )

    @field_validator("clifford_ratio", "non_clifford_ratio")
    @classmethod
    def validate_ratio_bounds(cls, v):
        """Validate that ratios are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Ratio must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("non_clifford_ratio")
    @classmethod
    def validate_ratio_sum(cls, v, info):
        """Validate that Clifford and non-Clifford ratios sum to approximately 1."""
        if "clifford_ratio" in info.data:
            total = v + info.data["clifford_ratio"]
            if not 0.99 <= total <= 1.01:  # Allow small floating point errors
                raise ValueError(f"Clifford and non-Clifford ratios must sum to 1.0, got {total}")
        return v


class AdvancedMetricsSchema(BaseModel):
    """
    Schema for advanced complexity metrics.

    This schema validates advanced metrics for circuit analysis
    including parallelism and resource utilization.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "parallelism_factor": 2.5,
                "parallelism_efficiency": 0.8,
                "circuit_efficiency": 0.75,
                "quantum_resource_utilization": 0.9,
            }
        }
    )

    parallelism_factor: float = Field(..., ge=0.0, description="Average gates per time step")
    parallelism_efficiency: float = Field(
        ..., ge=0.0, le=1.0, description="Efficiency of parallelism"
    )
    circuit_efficiency: float = Field(
        ..., ge=0.0, le=1.0, description="Circuit resource efficiency"
    )
    quantum_resource_utilization: float = Field(
        ..., ge=0.0, le=1.0, description="Quantum resource utilization"
    )

    @field_validator("parallelism_efficiency", "circuit_efficiency")
    @classmethod
    def validate_efficiency_bounds(cls, v):
        """Validate that efficiency metrics are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Efficiency must be between 0.0 and 1.0, got {v}")
        return v


class DerivedMetricsSchema(BaseModel):
    """
    Schema for derived complexity metrics.

    This schema validates derived metrics that combine multiple
    circuit characteristics into composite measures.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "square_ratio": 0.8,
                "weighted_complexity": 25,
                "normalized_weighted_complexity": 5.5,
            }
        }
    )

    square_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="How close the circuit is to square"
    )
    weighted_complexity: int = Field(..., ge=0, description="Weighted gate complexity score")
    normalized_weighted_complexity: float = Field(
        ..., ge=0.0, description="Normalized weighted complexity per qubit"
    )

    @field_validator("square_ratio")
    @classmethod
    def validate_square_ratio(cls, v):
        """Validate that square ratio is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Square ratio must be between 0.0 and 1.0, got {v}")
        return v


class ComplexityMetricsSchema(BaseModel):
    """
    Complete schema for all complexity metrics.

    This schema combines all individual complexity metric schemas into
    a single comprehensive structure for circuit complexity analysis.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gate_based_metrics": {
                    "gate_count": 10,
                    "circuit_depth": 5,
                    "t_count": 2,
                    "cnot_count": 3,
                    "two_qubit_count": 3,
                    "multi_qubit_ratio": 0.3,
                },
                "entanglement_metrics": {"entangling_gate_density": 0.3, "entangling_width": 4},
                "standardized_metrics": {
                    "circuit_volume": 20,
                    "gate_density": 0.5,
                    "clifford_ratio": 0.7,
                    "non_clifford_ratio": 0.3,
                },
                "advanced_metrics": {
                    "parallelism_factor": 2.5,
                    "parallelism_efficiency": 0.8,
                    "circuit_efficiency": 0.75,
                    "quantum_resource_utilization": 0.9,
                },
                "derived_metrics": {
                    "square_ratio": 0.8,
                    "weighted_complexity": 25,
                    "normalized_weighted_complexity": 5.5,
                },
            }
        }
    )

    gate_based_metrics: GateBasedMetricsSchema = Field(
        ..., description="Gate-based complexity metrics"
    )
    entanglement_metrics: EntanglementMetricsSchema = Field(
        ..., description="Entanglement-related metrics"
    )
    standardized_metrics: StandardizedMetricsSchema = Field(
        ..., description="Standardized comparison metrics"
    )
    advanced_metrics: AdvancedMetricsSchema = Field(..., description="Advanced analysis metrics")
    derived_metrics: DerivedMetricsSchema = Field(..., description="Derived composite metrics")

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert the schema to a flattened dictionary for DataFrame compatibility.

        Returns:
            Dict[str, Any]: Flattened dictionary with dot-notation keys
        """
        result = {}

        # Flatten each metric category
        categories = [
            ("gate_based_metrics", self.gate_based_metrics),
            ("entanglement_metrics", self.entanglement_metrics),
            ("standardized_metrics", self.standardized_metrics),
            ("advanced_metrics", self.advanced_metrics),
            ("derived_metrics", self.derived_metrics),
        ]

        for category_name, category_data in categories:
            category_dict = category_data.model_dump()  # pylint: disable=no-member
            for key, value in category_dict.items():
                result[f"{category_name}.{key}"] = value

        return result

    @classmethod
    def from_flat_dict(cls, flat_dict: Dict[str, Any]) -> "ComplexityMetricsSchema":
        """
        Create a schema instance from a flattened dictionary.

        Args:
            flat_dict: Flattened dictionary with dot-notation keys

        Returns:
            ComplexityMetricsSchema: Schema instance
        """
        # Initialize category dictionaries
        categories: Dict[str, Dict[str, Any]] = {
            "gate_based_metrics": {},
            "entanglement_metrics": {},
            "standardized_metrics": {},
            "advanced_metrics": {},
            "derived_metrics": {},
        }

        # Parse flattened keys
        for key, value in flat_dict.items():
            # Handle metric categories
            for category_name in categories:  # pylint: disable=consider-using-dict-items
                if key.startswith(f"{category_name}."):
                    metric_name = key.replace(f"{category_name}.", "")
                    categories[category_name][metric_name] = value
                    break

        return cls(
            gate_based_metrics=GateBasedMetricsSchema(**categories["gate_based_metrics"]),
            entanglement_metrics=EntanglementMetricsSchema(**categories["entanglement_metrics"]),
            standardized_metrics=StandardizedMetricsSchema(**categories["standardized_metrics"]),
            advanced_metrics=AdvancedMetricsSchema(**categories["advanced_metrics"]),
            derived_metrics=DerivedMetricsSchema(**categories["derived_metrics"]),
        )


# =============================================================================
# Circuit Performance Metrics Schemas
# =============================================================================


class SuccessMetricsSchema(BaseModel):
    """
    Schema for success rate metrics.

    This schema validates success rate metrics including
    success rate, error rate, and shot analysis for both
    single job and multiple jobs cases.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success_rate": 0.85,
                "error_rate": 0.15,
                "total_shots": 1024,
                "successful_shots": 870,
                "mean_success_rate": 0.82,
                "std_success_rate": 0.05,
                "min_success_rate": 0.75,
                "max_success_rate": 0.90,
                "total_trials": 3072,
            }
        }
    )

    # Single job fields
    success_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Success rate (0.0 to 1.0)"
    )
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Error rate (0.0 to 1.0)")
    total_shots: Optional[int] = Field(None, ge=0, description="Total number of shots")
    successful_shots: Optional[int] = Field(None, ge=0, description="Number of successful shots")

    # Multiple jobs aggregate fields
    mean_success_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Mean success rate across jobs"
    )
    std_success_rate: Optional[float] = Field(
        None, ge=0.0, description="Standard deviation of success rates"
    )
    min_success_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum success rate"
    )
    max_success_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Maximum success rate"
    )
    total_trials: Optional[int] = Field(
        None, ge=0, description="Total number of trials across all jobs"
    )

    @field_validator("successful_shots")
    @classmethod
    def validate_successful_shots(cls, v, info):
        """Validate that successful shots <= total shots."""
        if v is not None and "total_shots" in info.data and info.data["total_shots"] is not None:
            if v > info.data["total_shots"]:
                raise ValueError("Successful shots cannot exceed total shots")
        return v


class FidelityMetricsSchema(BaseModel):
    """
    Schema for fidelity metrics.

    This schema validates fidelity-related metrics including
    quantum fidelity and related measures for both single
    job and multiple jobs cases.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "fidelity": 0.92,
                "has_expected_distribution": True,
                "method": "theoretical_comparison",
                "confidence": "high",
                "mean_fidelity": 0.89,
                "std_fidelity": 0.03,
                "min_fidelity": 0.85,
                "max_fidelity": 0.94,
            }
        }
    )

    # Single job fields
    fidelity: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Quantum fidelity (0.0 to 1.0)"
    )
    has_expected_distribution: Optional[bool] = Field(
        None, description="Whether expected distribution was provided for fidelity calculation"
    )
    method: str = Field(
        ...,
        description="Method used for fidelity calculation (theoretical_comparison or success_based)",
    )
    confidence: str = Field(
        ..., description="Confidence level of the fidelity calculation (high, medium, low)"
    )

    # Multiple jobs aggregate fields
    mean_fidelity: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Mean fidelity across jobs"
    )
    std_fidelity: Optional[float] = Field(
        None, ge=0.0, description="Standard deviation of fidelities"
    )
    min_fidelity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum fidelity")
    max_fidelity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Maximum fidelity")


class StatisticalMetricsSchema(BaseModel):
    """
    Schema for statistical analysis metrics.

    This schema validates statistical metrics derived from
    measurement outcome distributions for both single job
    and multiple jobs cases.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "entropy": 1.8,
                "uniformity": 0.75,
                "concentration": 0.25,
                "dominant_outcome_probability": 0.6,
                "num_unique_outcomes": 4,
                "mean_entropy": 1.7,
                "mean_uniformity": 0.72,
                "mean_concentration": 0.28,
                "mean_dominant_probability": 0.58,
                "std_entropy": 0.1,
                "std_uniformity": 0.05,
            }
        }
    )

    # Single job fields
    entropy: Optional[float] = Field(
        None, ge=0.0, description="Shannon entropy of the distribution"
    )
    uniformity: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Uniformity measure (0.0 to 1.0)"
    )
    concentration: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Concentration measure (0.0 to 1.0)"
    )
    dominant_outcome_probability: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Probability of the most frequent outcome"
    )
    num_unique_outcomes: Optional[int] = Field(
        None, ge=0, description="Number of unique measurement outcomes"
    )

    # Multiple jobs aggregate fields
    mean_entropy: Optional[float] = Field(None, ge=0.0, description="Mean entropy across jobs")
    mean_uniformity: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Mean uniformity across jobs"
    )
    mean_concentration: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Mean concentration across jobs"
    )
    mean_dominant_probability: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Mean dominant probability across jobs"
    )
    std_entropy: Optional[float] = Field(None, ge=0.0, description="Standard deviation of entropy")
    std_uniformity: Optional[float] = Field(
        None, ge=0.0, description="Standard deviation of uniformity"
    )

    @field_validator(
        "uniformity",
        "concentration",
        "dominant_outcome_probability",
        "mean_uniformity",
        "mean_concentration",
        "mean_dominant_probability",
    )
    @classmethod
    def validate_ratio_bounds(cls, v):
        """Validate that probability/ratio metrics are between 0 and 1."""
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError(f"Probability/ratio must be between 0.0 and 1.0, got {v}")
        return v


class CircuitPerformanceSchema(BaseModel):
    """
    Complete schema for circuit performance metrics.

    This schema combines all individual performance metric schemas into
    a single comprehensive structure for circuit performance analysis.
    """

    success_metrics: SuccessMetricsSchema = Field(..., description="Success rate metrics")
    fidelity_metrics: FidelityMetricsSchema = Field(..., description="Fidelity metrics")
    statistical_metrics: StatisticalMetricsSchema = Field(..., description="Statistical metrics")

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert the schema to a flattened dictionary for DataFrame compatibility.

        Returns:
            Dict[str, Any]: Flattened dictionary with dot-notation keys
        """
        result: Dict[str, Any] = {}

        # Flatten success metrics
        success_dict = self.success_metrics.model_dump()  # pylint: disable=no-member
        for key, value in success_dict.items():
            result[f"success_metrics.{key}"] = value

        # Flatten fidelity metrics
        fidelity_dict = self.fidelity_metrics.model_dump()  # pylint: disable=no-member
        for key, value in fidelity_dict.items():
            result[f"fidelity_metrics.{key}"] = value

        # Flatten statistical metrics
        statistical_dict = self.statistical_metrics.model_dump()  # pylint: disable=no-member
        for key, value in statistical_dict.items():
            result[f"statistical_metrics.{key}"] = value

        return result

    @classmethod
    def from_flat_dict(cls, flat_dict: Dict[str, Any]) -> "CircuitPerformanceSchema":
        """
        Create a schema instance from a flattened dictionary.

        Args:
            flat_dict: Flattened dictionary with dot-notation keys

        Returns:
            CircuitPerformanceSchema: Schema instance
        """
        # Reconstruct nested structure
        success_metrics = {}
        fidelity_metrics = {}
        statistical_metrics = {}

        for key, value in flat_dict.items():
            if key.startswith("success_metrics."):
                metric_name = key.replace("success_metrics.", "")
                success_metrics[metric_name] = value
            elif key.startswith("fidelity_metrics."):
                metric_name = key.replace("fidelity_metrics.", "")
                fidelity_metrics[metric_name] = value
            elif key.startswith("statistical_metrics."):
                metric_name = key.replace("statistical_metrics.", "")
                statistical_metrics[metric_name] = value

        return cls(
            success_metrics=SuccessMetricsSchema(**success_metrics),
            fidelity_metrics=FidelityMetricsSchema(**fidelity_metrics),
            statistical_metrics=StatisticalMetricsSchema(**statistical_metrics),
        )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success_metrics": {
                    "success_rate": 0.85,
                    "error_rate": 0.15,
                    "total_shots": 1024,
                    "successful_shots": 870,
                    "mean_success_rate": 0.82,
                    "std_success_rate": 0.05,
                    "min_success_rate": 0.75,
                    "max_success_rate": 0.90,
                    "total_trials": 3072,
                },
                "fidelity_metrics": {
                    "fidelity": 0.92,
                    "has_expected_distribution": True,
                    "method": "theoretical_comparison",
                    "confidence": "high",
                    "mean_fidelity": 0.89,
                    "std_fidelity": 0.03,
                    "min_fidelity": 0.85,
                    "max_fidelity": 0.94,
                },
                "statistical_metrics": {
                    "entropy": 1.8,
                    "uniformity": 0.75,
                    "concentration": 0.25,
                    "dominant_outcome_probability": 0.6,
                    "num_unique_outcomes": 4,
                    "mean_entropy": 1.7,
                    "mean_uniformity": 0.72,
                    "mean_concentration": 0.28,
                    "mean_dominant_probability": 0.58,
                    "std_entropy": 0.1,
                    "std_uniformity": 0.05,
                },
            },
        }
    )


# =============================================================================
# Legacy Schemas (for backward compatibility)
# =============================================================================


class CircuitPerformanceJobSchema(BaseModel):
    """
    Schema for individual job circuit performance metrics.

    This schema validates circuit performance metrics from a single
    quantum job execution.

    DEPRECATED: Use CircuitPerformanceSchema instead.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "job_12345",
                "success_rate": 0.85,
                "error_rate": 0.15,
                "fidelity": 0.92,
                "total_shots": 1024,
                "successful_shots": 870,
            }
        }
    )

    job_id: str = Field(..., description="Unique identifier for the job")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate (0.0 to 1.0)")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Error rate (0.0 to 1.0)")
    fidelity: float = Field(..., ge=0.0, le=1.0, description="Fidelity (0.0 to 1.0)")
    total_shots: int = Field(..., ge=0, description="Total number of shots")
    successful_shots: int = Field(..., ge=0, description="Number of successful shots")

    @field_validator("error_rate")
    @classmethod
    def validate_error_rate(cls, v, info):
        """Validate that error_rate = 1 - success_rate."""
        if "success_rate" in info.data:
            expected_error_rate = 1.0 - info.data["success_rate"]
            if abs(v - expected_error_rate) > 0.001:  # Allow small floating point errors
                raise ValueError(
                    f"Error rate ({v}) must equal 1 - success_rate ({expected_error_rate})"
                )
        return v

    @field_validator("successful_shots")
    @classmethod
    def validate_successful_shots(cls, v, info):
        """Validate that successful_shots <= total_shots."""
        if "total_shots" in info.data and v > info.data["total_shots"]:
            raise ValueError(
                f"Successful shots ({v}) cannot exceed total shots ({info.data['total_shots']})"
            )
        return v


class CircuitPerformanceAggregateSchema(BaseModel):
    """
    Schema for aggregate circuit performance metrics across multiple jobs.

    This schema validates aggregate circuit performance metrics computed
    across multiple job executions.

    DEPRECATED: Use CircuitPerformanceSchema instead.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mean_success_rate": 0.82,
                "std_success_rate": 0.05,
                "min_success_rate": 0.75,
                "max_success_rate": 0.90,
                "total_trials": 3072,
                "fidelity": 0.89,
                "error_rate": 0.18,
            }
        }
    )

    mean_success_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Mean success rate across jobs"
    )
    std_success_rate: float = Field(..., ge=0.0, description="Standard deviation of success rates")
    min_success_rate: float = Field(..., ge=0.0, le=1.0, description="Minimum success rate")
    max_success_rate: float = Field(..., ge=0.0, le=1.0, description="Maximum success rate")
    total_trials: int = Field(..., ge=0, description="Total number of trials across all jobs")
    fidelity: float = Field(..., ge=0.0, le=1.0, description="Average fidelity across jobs")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Average error rate across jobs")

    @field_validator("min_success_rate")
    @classmethod
    def validate_min_max_order(cls, v, info):
        """Validate that min <= mean <= max."""
        if "mean_success_rate" in info.data and v > info.data["mean_success_rate"]:
            raise ValueError("Minimum success rate cannot be greater than mean")
        return v

    @field_validator("max_success_rate")
    @classmethod
    def validate_max_bounds(cls, v, info):
        """Validate that max >= mean >= min."""
        if "mean_success_rate" in info.data and v < info.data["mean_success_rate"]:
            raise ValueError("Maximum success rate cannot be less than mean")
        if "min_success_rate" in info.data and v < info.data["min_success_rate"]:
            raise ValueError("Maximum success rate cannot be less than minimum")
        return v

    @field_validator("error_rate")
    @classmethod
    def validate_error_rate_consistency(cls, v, info):
        """Validate that error_rate = 1 - mean_success_rate."""
        if "mean_success_rate" in info.data:
            expected_error_rate = 1.0 - info.data["mean_success_rate"]
            if abs(v - expected_error_rate) > 1e-10:  # Allow for floating point precision
                raise ValueError(
                    f"Error rate {v} should equal 1 - mean_success_rate ({expected_error_rate})"
                )
        return v
