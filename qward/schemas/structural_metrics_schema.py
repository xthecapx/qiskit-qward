from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, field_validator, ConfigDict


class StructuralMetricsSchema(BaseModel):
    """
    Schema for Structural Metrics that unifies LOC, Halstead, and circuit structure metrics.

    This schema combines:
    - LOC metrics (Lines of Code related metrics)
    - Halstead metrics (complexity metrics)
    - Circuit structure metrics (width, depth, density, size)
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                # LOC Metrics
                "phi1_total_loc": 120,
                "phi2_gate_loc": 35,
                "phi3_measure_loc": 8,
                "phi4_quantum_total_loc": 43,
                "phi5_num_qubits": 5,
                "phi6_num_gate_types": 6,
                # Halstead Metrics
                "unique_operators": 5,
                "unique_operands": 8,
                "total_operators": 12,
                "total_operands": 20,
                "program_length": 32,
                "vocabulary": 13,
                "estimated_length": 28.5,
                "volume": 147.2,
                "difficulty": 3.2,
                "effort": 471.0,
                # Circuit Structure Metrics
                "width": 5,
                "depth": 8,
                "max_dens": 12,
                "avg_dens": 6.4,
                "size": 32,
            }
        }
    )

    # =============================================================================
    # LOC Metrics (Lines of Code)
    # =============================================================================

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

    # =============================================================================
    # Halstead Metrics
    # =============================================================================

    # Basic counts (η1, η2, M1, M2)
    unique_operators: int = Field(
        ..., ge=0, description="Number of unique classical and quantum operators (η1)"
    )
    unique_operands: int = Field(
        ..., ge=0, description="Number of unique classical and quantum operands (η2)"
    )
    total_operators: int = Field(
        ..., ge=0, description="Total occurrences of classical and quantum operators (M1)"
    )
    total_operands: int = Field(
        ..., ge=0, description="Total occurrences of classical and quantum operands (M2)"
    )

    # Derived metrics
    program_length: int = Field(
        ..., ge=0, description="Total length of the quantum program (M = M1 + M2)"
    )
    vocabulary: int = Field(
        ..., ge=0, description="Total vocabulary of the quantum program (η = η1 + η2)"
    )
    estimated_length: float = Field(
        ...,
        ge=0,
        description="Estimated length of the quantum program (ME = η1*log2(η1) + η2*log2(η2))",
    )
    volume: float = Field(..., ge=0, description="Volume of the quantum program (VQ = M × log2(η))")
    difficulty: float = Field(
        ..., ge=0, description="Difficulty of the quantum program (DQ = (η1/2) × (M2/η2))"
    )
    effort: float = Field(
        ..., ge=0, description="Effort required to implement the quantum program (EQ = DQ × VQ)"
    )

    # =============================================================================
    # Circuit Structure Metrics
    # =============================================================================

    # Circuit dimensions
    width: int = Field(..., ge=0, description="Number of qubits in the circuit")
    depth: int = Field(
        ..., ge=0, description="Maximum number of operations applied to a qubit in the circuit"
    )

    # Density metrics
    max_dens: int = Field(
        ..., ge=0, description="Maximum number of operations applied to the qubits in any layer"
    )
    avg_dens: float = Field(
        ...,
        ge=0.0,
        description="Average number of operations applied to the qubits across all layers",
    )

    # Size metric (total number of operations/gates in the circuit)
    size: int = Field(..., ge=0, description="Total number of operations/gates in the circuit")

    # =============================================================================
    # Validation Methods
    # =============================================================================

    @field_validator("phi4_quantum_total_loc")
    @classmethod
    def validate_quantum_total_loc(cls, v, info):
        """Validate that quantum total LOC is at least the sum of gate and measure LOC."""
        if hasattr(info, "data"):
            gate_loc = info.data.get("phi2_gate_loc", 0)
            measure_loc = info.data.get("phi3_measure_loc", 0)
            min_expected = gate_loc + measure_loc
            if v < min_expected:
                raise ValueError(
                    f"Quantum total LOC ({v}) should be at least gate LOC + measure LOC ({min_expected})"
                )
        return v

    @field_validator("program_length")
    @classmethod
    def validate_program_length(cls, v, info):
        """Validate that program length equals total operators + total operands."""
        if hasattr(info, "data"):
            total_ops = info.data.get("total_operators", 0) + info.data.get("total_operands", 0)
            if v != total_ops:
                raise ValueError(
                    f"Program length ({v}) must equal total_operators + total_operands ({total_ops})"
                )
        return v

    @field_validator("vocabulary")
    @classmethod
    def validate_vocabulary(cls, v, info):
        """Validate that vocabulary equals unique operators + unique operands."""
        if hasattr(info, "data"):
            unique_ops = info.data.get("unique_operators", 0) + info.data.get("unique_operands", 0)
            if v != unique_ops:
                raise ValueError(
                    f"Vocabulary ({v}) must equal unique_operators + unique_operands ({unique_ops})"
                )
        return v

    @field_validator("volume")
    @classmethod
    def validate_volume(cls, v, info):
        """Validate that volume is positive and reasonable."""
        if v < 0:
            raise ValueError("Volume must be positive")
        return v

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v, info):
        """Validate that difficulty is positive and reasonable."""
        if v < 0:
            raise ValueError("Difficulty must be positive")
        if v > 1000:  # Reasonable upper bound for quantum circuits
            raise ValueError(f"Difficulty ({v}) seems unreasonably high for a quantum circuit")
        return v

    @field_validator("effort")
    @classmethod
    def validate_effort(cls, v, info):
        """Validate that effort is positive and reasonable."""
        if v < 0:
            raise ValueError("Effort must be positive")
        if v > 100000000:  # Reasonable upper bound for quantum circuits
            raise ValueError(f"Effort ({v}) seems unreasonably high for a quantum circuit")
        return v

    @field_validator("width")
    @classmethod
    def validate_width_consistency(cls, v, info):
        """Validate that width is consistent with num_qubits from LOC metrics."""
        if hasattr(info, "data"):
            num_qubits = info.data.get("phi5_num_qubits", 0)
            if v != num_qubits:
                raise ValueError(f"Width ({v}) must equal phi5_num_qubits ({num_qubits})")
        return v

    # Se comenta ya que total_operators no cuenta barreras y size sí
    # @field_validator("size")
    # @classmethod
    # def validate_size_consistency(cls, v, info):
    #     """Validate that size is consistent with total operators from Halstead metrics."""
    #     if hasattr(info, 'data'):
    #         total_operators = info.data.get('total_operators', 0)
    #         if v != total_operators:
    #             raise ValueError(f"Size ({v}) must equal total_operators ({total_operators})")
    #     return v

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert the schema to a flat dictionary for DataFrame compatibility.

        Returns:
            Dict[str, Any]: Flattened dictionary representation
        """
        # LOC metrics with prefix
        loc_data = {
            f"loc_{k}": v
            for k, v in {
                "phi1_total_loc": self.phi1_total_loc,
                "phi2_gate_loc": self.phi2_gate_loc,
                "phi3_measure_loc": self.phi3_measure_loc,
                "phi4_quantum_total_loc": self.phi4_quantum_total_loc,
                "phi5_num_qubits": self.phi5_num_qubits,
                "phi6_num_gate_types": self.phi6_num_gate_types,
            }.items()
        }

        # Halstead metrics
        halstead_data = {
            "unique_operators": self.unique_operators,
            "unique_operands": self.unique_operands,
            "total_operators": self.total_operators,
            "total_operands": self.total_operands,
            "program_length": self.program_length,
            "vocabulary": self.vocabulary,
            "estimated_length": self.estimated_length,
            "volume": self.volume,
            "difficulty": self.difficulty,
            "effort": self.effort,
        }

        # Circuit structure metrics
        structure_data = {
            "width": self.width,
            "depth": self.depth,
            "max_dens": self.max_dens,
            "avg_dens": self.avg_dens,
            "size": self.size,
        }

        return {**loc_data, **halstead_data, **structure_data}
