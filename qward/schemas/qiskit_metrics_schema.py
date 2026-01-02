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
