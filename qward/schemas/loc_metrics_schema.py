from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, field_validator, ConfigDict

# =============================================================================
# LOC Metrics Schema
# =============================================================================


class LocMetricsSchema(BaseModel):
    """
    Schema for Lines-of-Code (LOC) oriented quantum metrics.

    This captures code size and quantum-specific LOC counts for a program/circuit.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "phi1_total_loc": 120,
                "phi2_gate_loc": 35,
                "phi3_measure_loc": 8,
                "phi4_quantum_total_loc": 43,
                "phi5_num_qubits": 5,
                "phi6_num_gate_types": 6,
            }
        }
    )

    # ϕ1: Número de LOC totales en un programa cuántico
    phi1_total_loc: int = Field(..., ge=0, description="Total LOC in the quantum program")

    # ϕ2: Número de LOC relacionadas a operaciones con compuertas cuánticas
    phi2_gate_loc: int = Field(..., ge=0, description="LOC containing quantum gate operations")

    # ϕ3: Número de LOC relacionadas a mediciones cuánticas
    phi3_measure_loc: int = Field(..., ge=0, description="LOC containing quantum measurements")

    # ϕ4: Tamaño total de LOC relacionadas a aspectos cuánticos
    phi4_quantum_total_loc: int = Field(
        ...,
        ge=0,
        description="Total LOC related to quantum aspects (gates + measurements + other quantum ops)",
    )

    # ϕ5: Número de cúbits usados
    phi5_num_qubits: int = Field(..., ge=0, description="Number of qubits used by the circuit")

    # ϕ6: Número del tipo de compuertas usadas
    phi6_num_gate_types: int = Field(
        ..., ge=0, description="Number of distinct quantum gate types used"
    )

    @field_validator("phi2_gate_loc", "phi3_measure_loc", "phi4_quantum_total_loc")
    @classmethod
    def validate_quantum_loc_consistency(cls, v):
        # Individual fields already constrained to be non-negative
        return v

    def to_flat_dict(self) -> Dict[str, Any]:
        data = self.model_dump()  # pylint: disable=no-member
        return {f"loc_metrics.{k}": v for k, v in data.items()}
