"""Independent task, model and rank evidence for a measured histogram."""

from typing import Any, Dict, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class BinomialPredictionCheckSchema(BaseModel):
    """Exact count envelope conditional on a fixed, independent probability.

    The envelope covers shot variation only. Calibration uncertainty, model
    discrepancy, drift and correlated shots require additional modeling.
    """

    model_config = ConfigDict(allow_inf_nan=False)
    scope: Literal["conditional_on_fixed_probability"] = "conditional_on_fixed_probability"
    shots: int = Field(gt=0)
    count: int = Field(ge=0)
    alpha: float = Field(gt=0, lt=1)
    predicted_probability: float = Field(ge=0, le=1)
    observed_probability: float = Field(ge=0, le=1)
    observed_shot_standard_error: float = Field(ge=0)
    predicted_shot_standard_error: float = Field(ge=0)
    residual_shot_z: Optional[float] = None
    predicted_count_lower: int = Field(ge=0)
    predicted_count_upper: int = Field(ge=0)
    upper_count_budget: int = Field(ge=0)
    upper_tail_pvalue: float = Field(ge=0, le=1)
    lower_tail_pvalue: float = Field(ge=0, le=1)
    range_status: Literal["below_range", "within_range", "above_range"]
    exceeds_upper_budget: bool


class HistogramAssessmentSchema(BaseModel):
    """Finite-shot intervals are distinct from unbounded model uncertainty."""

    model_config = ConfigDict(allow_inf_nan=False)
    schema_version: str = "1.3"
    shots: int = Field(gt=0)
    alpha: float = Field(gt=0, lt=1)
    tvd: float = Field(ge=0, le=1)
    tvd_interval: Tuple[float, float]
    success_rate: Optional[float] = Field(None, ge=0, le=1)
    success_interval: Optional[Tuple[float, float]] = None
    model_status: Literal["consistent", "mismatch", "insufficient_calibration", "not_applicable"]
    rank_status: Literal["verified", "unresolved"]
    normalized_entropy: float = Field(ge=0, le=1)
    gap_standard_error: float = Field(ge=0)
    rank_pvalue: float = Field(ge=0, le=1)
    prediction_residual: Optional[float] = None
    prediction_snapshot_hash: Optional[str] = None
    interval_method: str
    error_budget: Optional[BinomialPredictionCheckSchema] = None
    m2_model_status: Literal[
        "consistent", "mismatch", "insufficient_calibration", "not_applicable"
    ] = "not_applicable"
    m2_error_budget: Optional[BinomialPredictionCheckSchema] = None

    def to_flat_dict(self) -> Dict[str, Any]:
        """Export scalar values and named interval endpoints for Scanner."""
        fields = self.model_dump(
            exclude={"success_interval", "tvd_interval", "error_budget", "m2_error_budget"}
        )
        fields.update(tvd_lower=self.tvd_interval[0], tvd_upper=self.tvd_interval[1])
        if self.success_interval:
            fields.update(
                success_lower=self.success_interval[0], success_upper=self.success_interval[1]
            )
        for prefix, budget in (
            ("error_budget", self.error_budget),
            ("m2_error_budget", self.m2_error_budget),
        ):
            if budget is not None:
                fields.update(
                    {f"{prefix}_{key}": value for key, value in budget.model_dump().items()}
                )
        return fields
