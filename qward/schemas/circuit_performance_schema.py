from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, field_validator, ConfigDict

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
