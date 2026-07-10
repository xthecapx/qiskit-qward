"""Pydantic schema for FidelityMetrics output."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FidelitySchema(BaseModel):
    """Schema for fidelity metrics (DSR profile, Michelson DSR, HF, TVD,
    success rate).

    All fidelity scores are in [0, 1] where 1 indicates perfect fidelity.

    ``dsr`` is the optional Michelson "peak-contrast" layer of the DSR
    evaluation framework (see ``qward.metrics.differential_success_rate``).
    ``chance_corrected_success``, ``coarse_tvd_similarity``, and
    ``coarse_hellinger_fidelity`` are the other three components of the
    histogram-free DSR profile; ``success_rate`` doubles as the profile's
    first component. ``hellinger_fidelity`` / ``tvd`` / ``tvd_fidelity`` are
    the *full*-distribution fidelity metrics, requiring a simulated ideal
    histogram (``target_histogram``) rather than only the expected-outcome
    set.
    """

    shots: Optional[int] = Field(None, ge=0)
    unique_outcomes: Optional[int] = Field(None, ge=0)
    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    dsr: Optional[float] = Field(None, ge=0.0, le=1.0)
    peak_mismatch: Optional[bool] = None
    hellinger_fidelity: Optional[float] = Field(None, ge=0.0, le=1.0)
    tvd: Optional[float] = Field(None, ge=0.0, le=1.0)
    tvd_fidelity: Optional[float] = Field(None, ge=0.0, le=1.0)
    expected_outcomes: Optional[List[str]] = None

    # DSR profile (histogram-free, task-level evaluation).
    chance_baseline: Optional[float] = Field(None, ge=0.0, le=1.0)
    chance_corrected_success: Optional[float] = Field(None, ge=0.0, le=1.0)
    coarse_tvd: Optional[float] = Field(None, ge=0.0, le=1.0)
    coarse_tvd_similarity: Optional[float] = Field(None, ge=0.0, le=1.0)
    coarse_hellinger_distance: Optional[float] = Field(None, ge=0.0, le=1.0)
    coarse_hellinger_fidelity: Optional[float] = Field(None, ge=0.0, le=1.0)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for DataFrame creation."""
        return {k: v for k, v in self.model_dump().items() if v is not None}
