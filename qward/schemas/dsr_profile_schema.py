"""Pydantic schema reporting DSR and companion measures."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DSRProfileSchema(BaseModel):
    """Schema reporting DSR and companion evaluation measures.

    ``Profile`` is retained in the API name for compatibility. DSR itself is
    the clipped Michelson contrast stored in ``dsr_michelson``. The remaining
    fields are companion measures or metadata; they are not components or
    alternative definitions of DSR. All fields are computed from measurement
    counts and an analytically known expected-outcome set ``E``.

    Components:
        - ``success_rate``: raw fraction of shots landing in ``E``.
        - ``chance_corrected_success``: success rescaled so that random
          guessing maps to 0 and perfect success maps to 1.
        - ``coarse_tvd_similarity`` / ``coarse_hellinger_fidelity``: "higher
          is better" agreement, on a coarse ``K + 1``-bin distribution
          (one bin per element of ``E`` plus a single "other" bin), between
          the observed counts and a task-reference distribution over ``E``
          (uniform ``1/K`` by default, or an explicit ``expected_weights``).

    Score vs. metric terminology (see project math notes):
        - ``success_rate`` and ``chance_corrected_success`` are *scores*,
          not metrics, in the formal sense.
        - ``coarse_tvd`` and ``coarse_hellinger_distance`` are genuine
          mathematical *metrics* on the coarse ``K + 1``-bin distribution.
        - ``coarse_tvd_similarity`` and ``coarse_hellinger_fidelity`` are the
          "higher is better" complements of those metrics and are therefore
          scores, not metrics, in the strict sense.

    The companion measures are reported separately. They are
    intentionally NOT averaged into one number: each answers a different
    question, and (for ``K = 1``) the two coarse-similarity components
    collapse exactly to ``success_rate``, so treating all four as
    independent evidence would double-count that redundancy.

    DSR may be included via ``dsr_michelson`` together with the diagnostic
    ``peak_mismatch`` flag. It compares the mean expected peak with the
    strongest competing peak.
    """

    shots: Optional[int] = Field(None, ge=0)
    num_measured_qubits: Optional[int] = Field(None, ge=0)
    expected_outcomes: Optional[List[str]] = None
    expected_weights: Optional[Dict[str, float]] = None
    num_expected_outcomes: Optional[int] = Field(None, ge=0)

    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    chance_baseline: Optional[float] = Field(None, ge=0.0, le=1.0)
    chance_corrected_success: Optional[float] = Field(None, ge=0.0, le=1.0)

    coarse_tvd: Optional[float] = Field(None, ge=0.0, le=1.0)
    coarse_tvd_similarity: Optional[float] = Field(None, ge=0.0, le=1.0)
    coarse_hellinger_distance: Optional[float] = Field(None, ge=0.0, le=1.0)
    coarse_hellinger_fidelity: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Optional fifth "peak-contrast" layer (legacy Michelson-contrast DSR).
    dsr_michelson: Optional[float] = Field(None, ge=0.0, le=1.0)
    peak_mismatch: Optional[bool] = None

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to a flat dictionary for DataFrame creation."""
        return {k: v for k, v in self.model_dump().items() if v is not None}
