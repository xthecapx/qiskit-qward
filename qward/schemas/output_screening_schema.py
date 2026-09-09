"""Schema for histogram-only candidate screening, independent of target-aware DSR."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OutputScreeningSchema(BaseModel):
    """Observed peak separation and finite-shot rank evidence.

    ``leading_outcome`` is absent when leaders tie; ``leading_outcomes`` lists
    all observed leaders without choosing one by bitstring order. A verified
    rank concerns the population mode under IID multinomial sampling, not
    application correctness. No significance level or acceptance rule is implicit.
    """

    shots: int = Field(..., gt=0)
    leading_outcome: Optional[str] = None
    leading_outcomes: List[str] = Field(..., min_length=1)
    has_unique_peak: bool
    top_count: int = Field(..., gt=0)
    runner_up_count: int = Field(..., ge=0)
    top_probability: float = Field(..., gt=0.0, le=1.0)
    runner_up_probability: float = Field(..., ge=0.0, le=1.0)
    top_two_contrast: float = Field(..., ge=0.0, le=1.0)
    absolute_gap: float = Field(..., ge=0.0, le=1.0)
    pair_probability: float = Field(..., gt=0.0, le=1.0)
    rank_pvalue: float = Field(..., ge=0.0, le=1.0)
    significance_level: Optional[float] = Field(None, gt=0.0, lt=1.0)
    rank_verified: Optional[bool] = None

    def to_flat_dict(self) -> Dict[str, Any]:
        """Return fields for DataFrame creation, omitting unspecified values."""
        return {key: value for key, value in self.model_dump().items() if value is not None}
