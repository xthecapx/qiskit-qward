"""Pydantic schema for backend calibration metrics."""

from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class BackendCalibrationSchema(BaseModel):
    """Schema for backend calibration data captured at execution time."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "median_single_qubit_gate_error": 0.00023,
                "median_two_qubit_gate_error": 0.0045,
                "median_readout_error": 0.012,
                "median_t1_us": 280.5,
                "median_t2_us": 120.3,
                "num_operational_qubits": 156,
                "backend_name": "ibm_fez",
                "calibration_timestamp": "2026-02-05T10:30:00Z",
                "provider": "ibm",
            }
        }
    )

    median_single_qubit_gate_error: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Median error rate across all single-qubit gates"
    )
    median_two_qubit_gate_error: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Median error rate across all two-qubit gates"
    )
    median_readout_error: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Median measurement/readout error"
    )
    median_t1_us: Optional[float] = Field(
        None, ge=0.0, description="Median T1 relaxation time in microseconds"
    )
    median_t2_us: Optional[float] = Field(
        None, ge=0.0, description="Median T2 dephasing time in microseconds"
    )
    num_operational_qubits: int = Field(
        ..., ge=0, description="Number of qubits with valid calibration data"
    )
    backend_name: str = Field(..., description="Name of the backend")
    calibration_timestamp: Optional[str] = Field(
        None, description="ISO timestamp of calibration data"
    )
    provider: str = Field(..., description="Provider identifier: ibm, aws, ionq")
