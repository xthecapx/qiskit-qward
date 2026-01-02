from typing import Dict, Any

from pydantic import BaseModel, Field, ConfigDict

# =============================================================================
# Behavioral Metrics Schema
# =============================================================================


class BehavioralMetricsSchema(BaseModel):
    """
    Schema for behavioral metrics that analyze quantum circuit execution patterns.

    This schema validates metrics related to circuit behavior during execution,
    including communication requirements, critical path analysis, measurement
    patterns, qubit activity, and parallelism characteristics.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "normalized_depth": 8.5,
                "program_communication": 0.4,
                "critical_depth": 0.6,
                "measurement": 0.2,
                "liveness": 0.75,
                "parallelism": 0.8,
            }
        }
    )

    # Normalized depth after transpilation to canonical gate set
    normalized_depth: float = Field(
        ...,
        ge=0.0,
        description="Depth of circuit after transpilation to basis gates ['rx', 'ry', 'rz', 'cx']",
    )

    # Program communication requirements
    program_communication: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized average degree of interaction graph (communication requirements)",
    )

    # Critical path analysis
    critical_depth: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio of two-qubit interactions on critical path to total two-qubit interactions",
    )

    # Measurement operations
    measurement: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio of layers with mid-circuit measurement/reset operations to total circuit depth",
    )

    # Qubit activity patterns
    liveness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio of active qubit-time steps to total qubit-time steps",
    )

    # Parallelism and cross-talk susceptibility
    parallelism: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Parallelism factor indicating susceptibility to cross-talk",
    )

    @classmethod
    def from_flat_dict(cls, data: Dict[str, Any]) -> "BehavioralMetricsSchema":
        """
        Create a schema instance from a flat dictionary.

        Args:
            data: Flat dictionary with metric values

        Returns:
            BehavioralMetricsSchema: Schema instance
        """
        return cls(**data)
