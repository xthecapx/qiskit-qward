from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, field_validator, ConfigDict


class ElementMetricsSchema(BaseModel):
    """
    Schema for Quantum Element Metrics.

    This schema validates comprehensive metrics for analyzing quantum circuit elements
    including gate distribution, oracle analysis, and measurement patterns.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "no_p_x": 2,
                "no_p_y": 1,
                "no_p_z": 0,
                "t_no_p": 3,
                "no_h": 4,
                "percent_sppos_q": 0.8,
                "no_other_sg": 3,
                "t_no_csqg": 6,
                "t_no_sqg": 10,
                "no_c_any_g": 6,
                "no_swap": 0,
                "no_cnot": 4,
                "percent_q_in_cnot": 0.8,
                "avg_cnot": 2.0,
                "max_cnot": 3,
                "no_toff": 1,
                "percent_q_in_toff": 0.6,
                "avg_toff": 0.6,
                "max_toff": 1,
                "no_gates": 16,
                "no_c_gates": 6,
                "percent_single_gates": 0.625,
                "no_or": 0,
                "no_c_or": 0,
                "percent_q_in_or": 0.0,
                "percent_q_in_c_or": 0.0,
                "avg_or_d": 0.0,
                "max_or_d": 0.0,
                "no_qm": 3,
                "percent_qm": 0.6,
                "percent_anc": 0.2,
            }
        }
    )

    # Pauli gate metrics
    no_p_x: int = Field(..., ge=0, description="Number of Pauli-X gates (NOT)")
    no_p_y: int = Field(..., ge=0, description="Number of Pauli-Y gates")
    no_p_z: int = Field(..., ge=0, description="Number of Pauli-Z gates")
    t_no_p: int = Field(..., ge=0, description="Total number of Pauli gates in the circuit")

    # Single-qubit gate metrics
    no_h: int = Field(..., ge=0, description="Number of Hadamard gates")
    percent_sppos_q: float = Field(
        ..., ge=0.0, le=1.0, description="Ratio of qubits with a Hadamard gate as an initial gate"
    )
    no_other_sg: int = Field(
        ..., ge=0, description="Number of other single-qubit gates in the circuit"
    )
    t_no_csqg: int = Field(..., ge=0, description="Total number of controlled single-qubit gates")
    t_no_sqg: int = Field(..., ge=0, description="Total number of single-qubit gates")

    # Controlled gate metrics
    no_c_or: int = Field(..., ge=0, description="Number of controlled oracles in the circuit")
    no_c_any_g: int = Field(..., ge=0, description="Number of controlled gates (any)")
    no_swap: int = Field(..., ge=0, description="Number of exchange gates")
    no_cnot: int = Field(..., ge=0, description="Number of NOT controlled gates (CNOT)")
    percent_q_in_cnot: float = Field(
        ..., ge=0.0, le=1.0, description="Ratio of qubits affected by CNOT gates"
    )
    avg_cnot: float = Field(
        ..., ge=0.0, description="Average number of CNOT gates directed to any qubit in a circuit"
    )
    max_cnot: int = Field(
        ..., ge=0, description="Maximum number of CNOT gates directed to any qubit of a circuit"
    )

    # Toffoli gate metrics
    no_toff: int = Field(..., ge=0, description="Number of Toffoli gates")
    percent_q_in_toff: float = Field(
        ..., ge=0.0, le=1.0, description="Ratio of qubits affected by Toffoli gates"
    )
    avg_toff: float = Field(
        ..., ge=0.0, description="Average number of Toffoli gates targeting any qubit of a circuit"
    )
    max_toff: int = Field(
        ..., ge=0, description="Maximum number of Toffoli gates targeting any qubit of a circuit"
    )

    # General gate metrics
    no_gates: int = Field(..., ge=0, description="Total number of gates in the circuit")
    no_c_gates: int = Field(
        ..., ge=0, description="Total number of controlled gates in the circuit"
    )
    percent_single_gates: float = Field(
        ..., ge=0.0, le=1.0, description="Ratio of single gates to total gates"
    )

    # Oracle metrics
    no_or: int = Field(..., ge=0, description="Number of oracles in the circuit")
    percent_q_in_or: float = Field(
        ..., ge=0.0, le=1.0, description="Proportion of qubits affected by the oracles"
    )
    percent_q_in_c_or: float = Field(
        ..., ge=0.0, le=1.0, description="Ratio of qubits affected by controlled oracles"
    )
    avg_or_d: float = Field(..., ge=0.0, description="Average depth of an oracle in the circuit")
    max_or_d: int = Field(..., ge=0, description="Maximum depth of an oracle in the circuit")

    # Measurement and ancilla metrics
    no_qm: int = Field(..., ge=0, description="Number of measured qubits")
    percent_qm: float = Field(..., ge=0.0, le=1.0, description="Ratio of measured qubits")
    percent_anc: float = Field(
        ..., ge=0.0, le=1.0, description="Ratio of ancilla (auxiliary) qubits in the circuit"
    )

    @field_validator("t_no_p")
    @classmethod
    def validate_t_no_p(cls, v, info):
        """Validate that total Pauli gates equals sum of individual Pauli gates."""
        if hasattr(info, "data"):
            expected = (
                info.data.get("no_p_x", 0) + info.data.get("no_p_y", 0) + info.data.get("no_p_z", 0)
            )
            if v != expected:
                raise ValueError(
                    f"Total Pauli gates ({v}) must equal no_p_x + no_p_y + no_p_z ({expected})"
                )
        return v

    @field_validator("t_no_sqg")
    @classmethod
    def validate_t_no_sqg(cls, v, info):
        """Validate that total single-qubit gates equals sum of individual single-qubit gates."""
        if hasattr(info, "data"):
            expected = (
                info.data.get("no_h", 0)
                + info.data.get("no_p_x", 0)
                + info.data.get("no_p_y", 0)
                + info.data.get("no_p_z", 0)
                + info.data.get("no_other_sg", 0)
                + info.data.get("t_no_csqg", 0)
            )
            if v != expected:
                raise ValueError(
                    f"Total single-qubit gates ({v}) must equal sum of individual single-qubit gates ({expected})"
                )
        return v

    @field_validator("no_c_gates")
    @classmethod
    def validate_no_c_gates(cls, v, info):
        """Validate that total controlled gates equals sum of individual controlled gates."""
        if hasattr(info, "data"):
            expected = (
                info.data.get("t_no_csqg", 0)
                + info.data.get("no_toff", 0)
                + info.data.get("no_c_or", 0)
                + info.data.get("no_cnot", 0)
            )
            if v != expected:
                raise ValueError(
                    f"Total controlled gates ({v}) must equal sum of individual controlled gates ({expected})"
                )
        return v

    @field_validator("percent_single_gates")
    @classmethod
    def validate_percent_single_gates(cls, v, info):
        """Validate that single gates percentage is calculated correctly."""
        if hasattr(info, "data"):
            total_gates = info.data.get("no_gates", 0)
            single_gates = info.data.get("t_no_sqg", 0)
            if total_gates > 0:
                expected = single_gates / total_gates
                if abs(v - expected) > 0.001:  # Allow small floating point errors
                    raise ValueError(
                        f"Single gates percentage ({v}) must equal t_no_sqg / no_gates ({expected})"
                    )
        return v

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert the schema to a flat dictionary for DataFrame compatibility.

        Returns:
            Dict[str, Any]: Flattened dictionary representation
        """
        return self.model_dump()  # pylint: disable=no-member

    @classmethod
    def from_flat_dict(cls, data: Dict[str, Any]) -> "ElementMetricsSchema":
        """
        Create a schema instance from a flat dictionary.

        Args:
            data: Flat dictionary with metric values

        Returns:
            ElementMetricsSchema: Schema instance
        """
        return cls(**data)
