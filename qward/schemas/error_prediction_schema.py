"""Experimental calibration-product prediction; scores are not state fidelities."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ErrorPredictionSchema(BaseModel):
    """Named M1/M2 hypotheses and calibration coverage on one frozen ISA circuit."""

    model_config = ConfigDict(allow_inf_nan=False)
    schema_version: str = "1.1"
    model_version: str = "calibration-product-v1"
    circuit_hash: str
    snapshot_hash: str
    backend_name: str
    collected_at: str
    output_width: int = Field(gt=0)
    calibration_coverage: float = Field(ge=0, le=1)
    missing_instructions: int = Field(ge=0)
    partial_product: float = Field(ge=0, le=1)
    gate_product: Optional[float] = Field(None, ge=0, le=1)
    readout_product: Optional[float] = Field(None, ge=0, le=1)
    esp_score: Optional[float] = Field(None, ge=0, le=1)
    m1_success: Optional[float] = Field(None, ge=0, le=1)
    effective_visibility: Optional[float] = Field(None, ge=0, le=1)
    mixture_tvd: Optional[float] = Field(None, ge=0, le=1)
    mixture_probabilities: Optional[Dict[str, float]] = None
    m2_polarization: Optional[float] = Field(None, ge=0, le=1)
    m2_success: Optional[float] = Field(None, ge=0, le=1)
    m2_reference_mass: Optional[float] = Field(None, ge=0, le=1)
    m2_tvd: Optional[float] = Field(None, ge=0, le=1)
    m2_probabilities: Optional[Dict[str, float]] = None
    reference_probabilities: Dict[str, float]
    features: Dict[str, Any] = Field(default_factory=dict)
    assumptions: List[str] = Field(default_factory=list)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Expose scalar scores and features to Scanner DataFrames."""
        fields = self.model_dump(
            exclude={"features", "mixture_probabilities", "m2_probabilities", "assumptions"}
        )
        fields.update({f"features.{key}": value for key, value in self.features.items()})
        return fields
