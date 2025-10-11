from typing import Dict, List, Optional, Any
import warnings

from pydantic import BaseModel, Field, field_validator, ConfigDict


# =============================================================================
# Quantum Halstead Metrics Schema
# =============================================================================


class QuantumHalsteadMetricsSchema(BaseModel):
    """
    Schema for quantum Halstead metrics adapted for quantum software.
    
    These metrics extend classical Halstead metrics to quantify various aspects
    of quantum software complexity by considering both classical and quantum
    operators and operands.
    
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
            }
        }
    )

    # Basic counts (η1, η2, M1, M2)
    unique_operators: int = Field(..., ge=0, description="Number of unique classical and quantum operators (η1)")
    unique_operands: int = Field(..., ge=0, description="Number of unique classical and quantum operands (η2)")
    total_operators: int = Field(..., ge=0, description="Total occurrences of classical and quantum operators (M1)")
    total_operands: int = Field(..., ge=0, description="Total occurrences of classical and quantum operands (M2)")
    
    # Derived metrics
    program_length: int = Field(..., ge=0, description="Total length of the quantum program (M = M1 + M2)")
    vocabulary: int = Field(..., ge=0, description="Total vocabulary of the quantum program (η = η1 + η2)")
    estimated_length: float = Field(..., ge=0, description="Estimated length of the quantum program (ME = η1*log2(η1) + η2*log2(η2))")
    volume: float = Field(..., ge=0, description="Volume of the quantum program (VQ = M × log2(η))")
    difficulty: float = Field(..., ge=0, description="Difficulty of the quantum program (DQ = (η1/2) × (M2/η2))")
    effort: float = Field(..., ge=0, description="Effort required to implement the quantum program (EQ = DQ × VQ)")
    
    
    # Quantum-specific classifications
    quantum_operators: int = Field(..., ge=0, description="Number of unique quantum operators")
    classical_operators: int = Field(..., ge=0, description="Number of unique classical operators")
    quantum_operands: int = Field(..., ge=0, description="Number of unique quantum operands")
    classical_operands: int = Field(..., ge=0, description="Number of unique classical operands")
    
    # Circuit-specific metrics
    gate_types: Dict[str, int] = Field(..., description="Count of each gate type in the circuit")
    qubit_operands: int = Field(..., ge=0, description="Number of unique qubit operands")
    classical_bit_operands: int = Field(..., ge=0, description="Number of unique classical bit operands")
    parameter_operands: int = Field(..., ge=0, description="Number of unique parameter operands")

    @field_validator("program_length")
    @classmethod
    def validate_program_length(cls, v, info):
        """Validate that program length equals total operators + total operands."""
        if hasattr(info, 'data'):
            total_ops = info.data.get('total_operators', 0) + info.data.get('total_operands', 0)
            if v != total_ops:
                raise ValueError(f"Program length ({v}) must equal total_operators + total_operands ({total_ops})")
        return v

    @field_validator("vocabulary")
    @classmethod
    def validate_vocabulary(cls, v, info):
        """Validate that vocabulary equals unique operators + unique operands."""
        if hasattr(info, 'data'):
            unique_ops = info.data.get('unique_operators', 0) + info.data.get('unique_operands', 0)
            if v != unique_ops:
                raise ValueError(f"Vocabulary ({v}) must equal unique_operators + unique_operands ({unique_ops})")
        return v

    @field_validator("estimated_length")
    @classmethod
    def validate_estimated_length(cls, v, info):
        """Validate that estimated length is reasonable compared to actual length."""
        if hasattr(info, 'data'):
            actual_length = info.data.get('program_length', 0)
            if actual_length > 0 and (v < 0.1 * actual_length or v > 10 * actual_length):
                warnings.warn(f"Estimated length ({v}) differs significantly from actual length ({actual_length})")
        return v

    @field_validator("volume")
    @classmethod
    def validate_volume(cls, v, info):
        """Validate that volume is positive and reasonable."""
        if v <= 0:
            raise ValueError("Volume must be positive")
        if hasattr(info, 'data'):
            program_length = info.data.get('program_length', 0)
            vocabulary = info.data.get('vocabulary', 0)
            if vocabulary > 0:
                expected_volume = program_length * (vocabulary.bit_length() if vocabulary > 0 else 0)
                if v < 0.1 * expected_volume or v > 10 * expected_volume:
                    raise ValueError(f"Volume ({v}) seems unreasonable for given program length and vocabulary")
        return v

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v, info):
        """Validate that difficulty is positive and reasonable."""
        if v <= 0:
            raise ValueError("Difficulty must be positive")
        if v > 1000:  # Reasonable upper bound for quantum circuits
            raise ValueError(f"Difficulty ({v}) seems unreasonably high for a quantum circuit")
        return v

    @field_validator("effort")
    @classmethod
    def validate_effort(cls, v, info):
        """Validate that effort is positive and reasonable."""
        if v <= 0:
            raise ValueError("Effort must be positive")
        if v > 100000000:  # Reasonable upper bound for quantum circuits
            raise ValueError(f"Effort ({v}) seems unreasonably high for a quantum circuit")
        return v

    @field_validator("gate_types")
    @classmethod
    def validate_gate_types(cls, v):
        """Validate that all gate type counts are non-negative."""
        for gate_name, count in v.items():
            if count < 0:
                raise ValueError(f"Gate count for '{gate_name}' must be non-negative, got {count}")
        return v

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert the schema to a flat dictionary for DataFrame compatibility.
        
        Returns:
            Dict[str, Any]: Flattened dictionary representation
        """
        return {
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
            "quantum_operators": self.quantum_operators,
            "classical_operators": self.classical_operators,
            "quantum_operands": self.quantum_operands,
            "classical_operands": self.classical_operands,
            "qubit_operands": self.qubit_operands,
            "classical_bit_operands": self.classical_bit_operands,
            "parameter_operands": self.parameter_operands,
            **{f"gate_{gate}": count for gate, count in self.gate_types.items()}
        }

    @classmethod
    def from_flat_dict(cls, data: Dict[str, Any]) -> "QuantumHalsteadMetricsSchema":
        """
        Create a schema instance from a flat dictionary.
        
        Args:
            data: Flat dictionary with metric values
            
        Returns:
            QuantumHalsteadMetricsSchema: Schema instance
        """
        # Extract gate types from flat dictionary
        gate_types = {}
        for key, value in data.items():
            if key.startswith("gate_"):
                gate_name = key[5:]  # Remove "gate_" prefix
                gate_types[gate_name] = value
        
        # Remove gate keys from data to avoid conflicts
        filtered_data = {k: v for k, v in data.items() if not k.startswith("gate_")}
        filtered_data["gate_types"] = gate_types
        
        return cls(**filtered_data)