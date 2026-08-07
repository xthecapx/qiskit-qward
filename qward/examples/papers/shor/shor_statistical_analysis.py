"""Statistical helpers for Shor experiment batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class ConfigAnalysis:
    config_id: str
    noise_model: str
    num_runs: int
    mean: float
    std: float
    median: float
    min_val: float
    max_val: float
    ci_lower: float
    ci_upper: float


def analyze_config_results(
    success_rates: List[float],
    config_id: str,
    noise_model: str,
    ideal_rates: Optional[List[float]] = None,
) -> ConfigAnalysis:
    arr = np.array(success_rates, dtype=float)
    n = len(arr)
    mean = float(arr.mean()) if n else 0.0
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    return ConfigAnalysis(
        config_id=config_id,
        noise_model=noise_model,
        num_runs=n,
        mean=mean,
        std=std,
        median=float(np.median(arr)) if n else 0.0,
        min_val=float(arr.min()) if n else 0.0,
        max_val=float(arr.max()) if n else 0.0,
        ci_lower=mean - 1.96 * se,
        ci_upper=mean + 1.96 * se,
    )


def print_analysis_summary(analysis: ConfigAnalysis) -> None:
    print(
        f"[{analysis.config_id}/{analysis.noise_model}] "
        f"n={analysis.num_runs} mean={analysis.mean:.3f} "
        f"std={analysis.std:.3f} CI=[{analysis.ci_lower:.3f},{analysis.ci_upper:.3f}]"
    )
