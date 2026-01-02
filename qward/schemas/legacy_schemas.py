from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, field_validator, ConfigDict

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
