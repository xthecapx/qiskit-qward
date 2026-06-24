"""Pydantic schema for EstimatorMetrics output."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EstimatorSchema(BaseModel):
    """Schema for Estimator primitive metrics.

    Captures expectation values from quantum observables and derived
    quality metrics (fidelity, SNR, success probabilities).
    """

    num_observables: Optional[int] = Field(None, ge=1)
    expectation_values: Optional[List[float]] = None
    standard_deviations: Optional[List[float]] = None
    observable_labels: Optional[List[str]] = None

    mean_expectation_value: Optional[float] = Field(None, ge=-1.0, le=1.0)
    min_expectation_value: Optional[float] = Field(None, ge=-1.0, le=1.0)
    max_expectation_value: Optional[float] = Field(None, ge=-1.0, le=1.0)

    success_probabilities: Optional[List[float]] = None
    mean_success_probability: Optional[float] = Field(None, ge=0.0, le=1.0)

    ideal_expectation_values: Optional[List[float]] = None
    observable_fidelities: Optional[List[float]] = None
    mean_observable_fidelity: Optional[float] = Field(None, ge=0.0, le=1.0)
    relative_errors: Optional[List[float]] = None
    mean_relative_error: Optional[float] = Field(None, ge=0.0)

    signal_to_noise_ratios: Optional[List[float]] = None
    mean_snr: Optional[float] = None
    depolarization_factor: Optional[float] = Field(None, ge=0.0, le=1.0)

    _LIST_PREFIXES = {
        "expectation_values": "evs",
        "standard_deviations": "stds",
        "success_probabilities": "success_prob",
        "observable_fidelities": "obs_fidelity",
        "relative_errors": "rel_error",
        "signal_to_noise_ratios": "snr",
    }

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for DataFrame creation.

        List fields are expanded to indexed columns (evs_0, evs_1, ...).
        """
        result: Dict[str, Any] = {}
        data = self.model_dump()

        for key, value in data.items():
            if value is None:
                continue
            if isinstance(value, list):
                prefix = self._LIST_PREFIXES.get(key)
                if prefix:
                    for i, v in enumerate(value):
                        result[f"{prefix}_{i}"] = v
                else:
                    result[key] = value
            else:
                result[key] = value

        return result
