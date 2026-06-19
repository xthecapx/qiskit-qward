"""Pydantic schema for FidelityMetrics output."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FidelitySchema(BaseModel):
    """Schema for fidelity metrics (DSR, HF, TVD, success rate).

    All fidelity scores are in [0, 1] where 1 indicates perfect fidelity.
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

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for DataFrame creation."""
        return {k: v for k, v in self.model_dump().items() if v is not None}
