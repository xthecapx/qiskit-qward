from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, field_validator, ConfigDict


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
