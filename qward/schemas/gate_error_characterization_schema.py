"""Pydantic schema for gate error characterization metrics."""

from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict


class GateErrorEntry(BaseModel):
    """Error data for a single gate instance on specific physical qubits."""

    gate_name: str = Field(..., description="Gate operation name (e.g., cx, ecr, sx)")
    physical_qubits: List[int] = Field(..., description="Physical qubit indices")
    error_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Gate error rate")
    duration_ns: Optional[float] = Field(None, ge=0.0, description="Gate duration in nanoseconds")


class GateErrorCharacterizationSchema(BaseModel):
    """Schema for per-gate error characterization of a transpiled circuit."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "entries": [
                    {
                        "gate_name": "ecr",
                        "physical_qubits": [0, 1],
                        "error_rate": 0.005,
                        "duration_ns": 660.0,
                    }
                ],
                "mean_single_qubit_error": 0.00023,
                "mean_two_qubit_error": 0.0045,
                "max_error": 0.012,
                "weighted_mean_error": 0.003,
                "num_distinct_physical_qubits": 5,
                "physical_qubits_used": [0, 1, 2, 14, 15],
            }
        }
    )

    entries: List[GateErrorEntry] = Field(
        default_factory=list, description="Per-gate error entries"
    )
    mean_single_qubit_error: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Mean error across single-qubit gates used"
    )
    mean_two_qubit_error: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Mean error across two-qubit gates used"
    )
    max_error: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Maximum gate error in the circuit"
    )
    weighted_mean_error: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Error weighted by gate count"
    )
    num_distinct_physical_qubits: int = Field(
        0, ge=0, description="Number of distinct physical qubits used"
    )
    physical_qubits_used: List[int] = Field(
        default_factory=list, description="List of physical qubit indices used"
    )
