"""
Grover Experiment Statistical Analysis

This module provides Grover-specific statistical analysis built on the shared
analysis utilities in qward.algorithms.experiment_analysis.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from qward.algorithms.experiment_analysis import (
    compute_descriptive_stats,
    NormalityTestResult,
    test_shapiro_wilk,
    test_dagostino_pearson,
    test_anderson_darling,
    test_kolmogorov_smirnov,
    test_normality,
    analyze_noise_impact,
    characterize_distribution,
    compare_noise_models,
    analyze_config_results_base,
    load_latest_batch_files,
    load_batch_results,
    extract_success_rates,
    build_results_by_config,
    build_noise_means,
    generate_campaign_report,
)

# =============================================================================
# Aggregate Analysis for Experiment
# =============================================================================


@dataclass
class ConfigAnalysis:
    """Complete statistical analysis for a single configuration."""

    config_id: str
    noise_model: str
    num_runs: int

    # Descriptive statistics
    mean: float
    std: float
    median: float
    min_val: float
    max_val: float
    ci_lower: float
    ci_upper: float

    # Shape
    skewness: float
    kurtosis: float

    # Normality
    is_normal: bool
    normality_pvalue: Optional[float]

    # Noise impact (vs IDEAL)
    degradation_from_ideal: float = 0.0
    cohens_d_vs_ideal: float = 0.0
    effect_size_vs_ideal: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for data storage."""
        return {
            "config_id": self.config_id,
            "noise_model": self.noise_model,
            "num_runs": self.num_runs,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "min": self.min_val,
            "max": self.max_val,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "is_normal": self.is_normal,
            "normality_pvalue": self.normality_pvalue,
            "degradation_from_ideal": self.degradation_from_ideal,
            "cohens_d_vs_ideal": self.cohens_d_vs_ideal,
            "effect_size_vs_ideal": self.effect_size_vs_ideal,
        }


def analyze_config_results(
    success_rates: List[float],
    config_id: str,
    noise_model: str,
    ideal_rates: Optional[List[float]] = None,
) -> ConfigAnalysis:
    """
    Complete statistical analysis for a configuration's results.
    """
    stats_data = analyze_config_results_base(success_rates, noise_model, ideal_rates)

    return ConfigAnalysis(
        config_id=config_id,
        noise_model=noise_model,
        num_runs=stats_data["num_runs"],
        mean=stats_data["mean"],
        std=stats_data["std"],
        median=stats_data["median"],
        min_val=stats_data["min_val"],
        max_val=stats_data["max_val"],
        ci_lower=stats_data["ci_lower"],
        ci_upper=stats_data["ci_upper"],
        skewness=stats_data["skewness"],
        kurtosis=stats_data["kurtosis"],
        is_normal=stats_data["is_normal"],
        normality_pvalue=stats_data["normality_pvalue"],
        degradation_from_ideal=stats_data["degradation_from_ideal"],
        cohens_d_vs_ideal=stats_data["cohens_d_vs_ideal"],
        effect_size_vs_ideal=stats_data["effect_size_vs_ideal"],
    )


# =============================================================================
# Utility Functions
# =============================================================================


def print_analysis_summary(analysis: ConfigAnalysis) -> None:
    """Print formatted analysis summary."""
    print(f"\n{'=' * 60}")
    print(f"STATISTICAL ANALYSIS: {analysis.config_id} ({analysis.noise_model})")
    print(f"{'=' * 60}")

    print(f"\nDescriptive Statistics (n={analysis.num_runs}):")
    print(f"  Mean:   {analysis.mean:.4f} ± {analysis.std:.4f}")
    print(f"  Median: {analysis.median:.4f}")
    print(f"  Range:  [{analysis.min_val:.4f}, {analysis.max_val:.4f}]")
    print(f"  95% CI: [{analysis.ci_lower:.4f}, {analysis.ci_upper:.4f}]")

    print(f"\nDistribution Shape:")
    print(f"  Skewness: {analysis.skewness:.3f}")
    print(f"  Kurtosis: {analysis.kurtosis:.3f}")
    print(f"  Normal:   {'Yes' if analysis.is_normal else 'No'}", end="")
    if analysis.normality_pvalue:
        print(f" (p={analysis.normality_pvalue:.4f})")
    else:
        print()

    if analysis.noise_model != "IDEAL" and analysis.degradation_from_ideal != 0:
        print(f"\nNoise Impact (vs IDEAL):")
        print(f"  Degradation: {analysis.degradation_from_ideal:.1f}%")
        print(
            f"  Effect Size: {analysis.effect_size_vs_ideal} "
            f"(d={analysis.cohens_d_vs_ideal:.2f})"
        )


def print_comparison_table(analyses: List[ConfigAnalysis]) -> None:
    """Print comparison table for multiple analyses."""
    print(f"\n{'=' * 100}")
    print("COMPARISON TABLE")
    print(f"{'=' * 100}")
    print(
        f"{'Config':<10} {'Noise':<12} {'Mean':>8} {'Std':>8} {'Median':>8} "
        f"{'Normal':>8} {'Degrad%':>8} {'Effect':>10}"
    )
    print(f"{'-' * 100}")

    for a in analyses:
        print(
            f"{a.config_id:<10} {a.noise_model:<12} {a.mean:>8.4f} {a.std:>8.4f} "
            f"{a.median:>8.4f} {'Yes' if a.is_normal else 'No':>8} "
            f"{a.degradation_from_ideal:>7.1f}% {a.effect_size_vs_ideal:>10}"
        )

    print(f"{'=' * 100}")


__all__ = [
    # Shared analysis utilities (re-exported)
    "compute_descriptive_stats",
    "NormalityTestResult",
    "test_shapiro_wilk",
    "test_dagostino_pearson",
    "test_anderson_darling",
    "test_kolmogorov_smirnov",
    "test_normality",
    "analyze_noise_impact",
    "characterize_distribution",
    "compare_noise_models",
    "analyze_config_results_base",
    "load_latest_batch_files",
    "load_batch_results",
    "extract_success_rates",
    "build_results_by_config",
    "build_noise_means",
    "generate_campaign_report",
    # Grover-specific analysis
    "ConfigAnalysis",
    "analyze_config_results",
    "print_analysis_summary",
    "print_comparison_table",
]
