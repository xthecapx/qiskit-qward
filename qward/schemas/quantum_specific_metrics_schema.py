"""
Schema for Quantum Specific Metrics.

This module provides the QuantumSpecificMetricsSchema class for validating
and structuring quantum-specific metrics data.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict


class QuantumSpecificMetricsSchema(BaseModel):
    """
    Schema for Quantum Specific Metrics that provides quantum-specific analysis.

    This schema includes:
    - %SpposQ: Ratio of qubits with a Hadamard gate as initial gate
    - Magic: Quantum magic measure (non-Cliffordness)
    - Coherence: Coherence power measure
    - Sensitivity: Circuit sensitivity measure
    - Entanglement-Ratio: Ratio of two-qubit interactions to total operations
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "spposq_ratio": 0.6,
                "magic": 0.85,
                "coherence": 1.2,
                "sensitivity": 0.45,
                "entanglement_ratio": 0.3,
            }
        }
    )

    # =============================================================================
    # Quantum Specific Metrics
    # =============================================================================

    # %SpposQ: Ratio of qubits with a Hadamard gate as initial gate
    spposq_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio of qubits with a Hadamard gate as initial gate (%SpposQ)",
    )

    # Magic: Quantum magic measure (non-Cliffordness)
    magic: float = Field(
        ..., ge=0.0, description="Quantum magic measure indicating non-Cliffordness of the circuit"
    )

    # Coherence: Coherence power measure
    coherence: float = Field(
        ...,
        ge=0.0,
        description="Coherence power measure indicating the circuit's ability to generate coherence",
    )

    # Sensitivity: Circuit sensitivity measure
    sensitivity: float = Field(
        ...,
        ge=0.0,
        description="Circuit sensitivity measure indicating sensitivity to perturbations",
    )

    # Entanglement-Ratio: Ratio of two-qubit interactions to total operations
    entanglement_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio of two-qubit interactions (ne) to total gate operations (ng): E = ne/ng",
    )

    # =============================================================================
    # Validation Methods
    # =============================================================================

    @field_validator("spposq_ratio")
    @classmethod
    def validate_spposq_ratio(cls, v):
        """Validate that %SpposQ ratio is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"%SpposQ ratio ({v}) must be between 0.0 and 1.0")
        return v

    @field_validator("magic")
    @classmethod
    def validate_magic(cls, v):
        """Validate that magic value is non-negative."""
        if v < 0.0:
            raise ValueError(f"Magic value ({v}) must be non-negative")
        return v

    @field_validator("coherence")
    @classmethod
    def validate_coherence(cls, v):
        """Validate that coherence value is non-negative."""
        if v < 0.0:
            raise ValueError(f"Coherence value ({v}) must be non-negative")
        return v

    @field_validator("sensitivity")
    @classmethod
    def validate_sensitivity(cls, v):
        """Validate that sensitivity value is non-negative."""
        if v < 0.0:
            raise ValueError(f"Sensitivity value ({v}) must be non-negative")
        return v

    @field_validator("entanglement_ratio")
    @classmethod
    def validate_entanglement_ratio(cls, v):
        """Validate that entanglement ratio is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Entanglement ratio ({v}) must be between 0.0 and 1.0")
        return v

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert the schema to a flat dictionary for DataFrame compatibility.

        Returns:
            Dict[str, Any]: Flattened dictionary representation
        """
        return {
            "spposq_ratio": self.spposq_ratio,
            "magic": self.magic,
            "coherence": self.coherence,
            "sensitivity": self.sensitivity,
            "entanglement_ratio": self.entanglement_ratio,
        }

    @classmethod
    def from_flat_dict(cls, data: Dict[str, Any]) -> "QuantumSpecificMetricsSchema":
        """
        Create a schema instance from a flat dictionary.

        Args:
            data: Flat dictionary with metric values

        Returns:
            QuantumSpecificMetricsSchema: Schema instance
        """
        # Map flat dictionary keys to schema fields
        mapped_data = {
            "spposq_ratio": data.get("quantum_specific_spposq_ratio", 0.0),
            "magic": data.get("quantum_specific_magic", 0.0),
            "coherence": data.get("quantum_specific_coherence", 0.0),
            "sensitivity": data.get("quantum_specific_sensitivity", 0.0),
            "entanglement_ratio": data.get("quantum_specific_entanglement_ratio", 0.0),
        }

        return cls(**mapped_data)
