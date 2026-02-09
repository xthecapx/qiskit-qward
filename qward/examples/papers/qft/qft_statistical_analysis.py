"""
QFT Experiment Statistical Analysis

This module implements QFT-specific statistical analysis functions on top of
the shared analysis utilities in qward.algorithms.experiment_analysis.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

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
# QFT-Specific Analysis Functions
# =============================================================================


def analyze_scalability(
    results_by_qubits: Dict[int, Dict[str, List[float]]],
    noise_model: str = "IDEAL",
) -> Dict[str, Any]:
    """
    Analyze how QFT success rate scales with qubit count.
    """
    qubits = sorted(results_by_qubits.keys())
    means = []
    stds = []

    for n in qubits:
        rates = results_by_qubits[n].get(noise_model, [])
        if rates:
            means.append(np.mean(rates))
            stds.append(np.std(rates, ddof=1) if len(rates) > 1 else 0)
        else:
            means.append(np.nan)
            stds.append(np.nan)

    valid_mask = ~np.isnan(means)
    valid_qubits = np.array(qubits)[valid_mask]
    valid_means = np.array(means)[valid_mask]

    decay_fit = None
    if len(valid_qubits) >= 3 and all(m > 0 for m in valid_means):
        try:
            log_means = np.log(valid_means)
            slope, intercept, r_value, p_value, std_err = stats.linregress(valid_qubits, log_means)
            decay_fit = {
                "decay_rate": -slope,
                "initial_value": np.exp(intercept),
                "r_squared": r_value**2,
                "p_value": p_value,
                "half_life_qubits": np.log(2) / (-slope) if slope < 0 else float("inf"),
            }
        except Exception:
            pass

    degradations = []
    for i in range(1, len(means)):
        if not np.isnan(means[i]) and not np.isnan(means[i - 1]) and means[i - 1] > 0:
            degradations.append((means[i - 1] - means[i]) / means[i - 1] * 100)

    return {
        "noise_model": noise_model,
        "qubits": qubits,
        "means": means,
        "stds": stds,
        "decay_fit": decay_fit,
        "avg_degradation_per_qubit": np.mean(degradations) if degradations else 0,
        "max_qubits_above_50pct": max(
            [q for q, m in zip(qubits, means) if not np.isnan(m) and m >= 0.5],
            default=0,
        ),
        "max_qubits_above_90pct": max(
            [q for q, m in zip(qubits, means) if not np.isnan(m) and m >= 0.9],
            default=0,
        ),
    }


def analyze_period_impact(
    results_by_period: Dict[int, Dict[str, List[float]]],
    noise_model: str = "IDEAL",
) -> Dict[str, Any]:
    """
    Analyze how period affects QFT success rate in period detection mode.
    """
    periods = sorted(results_by_period.keys())
    means = []
    stds = []

    for p in periods:
        rates = results_by_period[p].get(noise_model, [])
        if rates:
            means.append(np.mean(rates))
            stds.append(np.std(rates, ddof=1) if len(rates) > 1 else 0)
        else:
            means.append(np.nan)
            stds.append(np.nan)

    valid_mask = ~np.isnan(means)
    valid_periods = np.array(periods)[valid_mask]
    valid_means = np.array(means)[valid_mask]

    correlation = None
    if len(valid_periods) >= 3:
        try:
            r, p = stats.pearsonr(valid_periods, valid_means)
            correlation = {
                "pearson_r": r,
                "p_value": p,
                "interpretation": (
                    "Larger periods → higher success"
                    if r > 0.5 and p < 0.05
                    else (
                        "Larger periods → lower success"
                        if r < -0.5 and p < 0.05
                        else "No clear trend"
                    )
                ),
            }
        except Exception:
            pass

    return {
        "noise_model": noise_model,
        "periods": periods,
        "means": means,
        "stds": stds,
        "correlation": correlation,
        "best_period": periods[np.nanargmax(means)] if means else None,
        "worst_period": periods[np.nanargmin(means)] if means else None,
    }


def compare_test_modes(
    roundtrip_rates: List[float],
    period_rates: List[float],
    noise_model: str = "unknown",
) -> Dict[str, Any]:
    """
    Compare roundtrip vs period detection mode performance.
    """
    rt_stats = compute_descriptive_stats(roundtrip_rates)
    pd_stats = compute_descriptive_stats(period_rates)

    if len(roundtrip_rates) >= 3 and len(period_rates) >= 3:
        stat, p = stats.mannwhitneyu(roundtrip_rates, period_rates, alternative="two-sided")
        test_result = {
            "statistic": stat,
            "p_value": p,
            "significant_difference": p < 0.05,
        }
    else:
        test_result = None

    return {
        "noise_model": noise_model,
        "roundtrip": {
            "mean": rt_stats["mean"],
            "std": rt_stats["std"],
            "median": rt_stats["median"],
        },
        "period_detection": {
            "mean": pd_stats["mean"],
            "std": pd_stats["std"],
            "median": pd_stats["median"],
        },
        "difference": rt_stats["mean"] - pd_stats["mean"],
        "better_mode": "roundtrip" if rt_stats["mean"] > pd_stats["mean"] else "period_detection",
        "statistical_test": test_result,
    }


# =============================================================================
# Aggregate Analysis for Experiment
# =============================================================================


@dataclass
class QFTConfigAnalysis:
    """Complete statistical analysis for a single QFT configuration."""

    config_id: str
    noise_model: str
    test_mode: str  # "roundtrip" or "period_detection"
    num_qubits: int
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

    # Mode-specific
    input_state: Optional[str] = None  # For roundtrip
    period: Optional[int] = None  # For period detection

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for data storage."""
        return {
            "config_id": self.config_id,
            "noise_model": self.noise_model,
            "test_mode": self.test_mode,
            "num_qubits": self.num_qubits,
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
            "input_state": self.input_state,
            "period": self.period,
        }


def analyze_qft_config_results(
    success_rates: List[float],
    config_id: str,
    noise_model: str,
    test_mode: str,
    num_qubits: int,
    input_state: Optional[str] = None,
    period: Optional[int] = None,
    ideal_rates: Optional[List[float]] = None,
) -> QFTConfigAnalysis:
    """
    Complete statistical analysis for a QFT configuration's results.
    """
    stats_data = analyze_config_results_base(success_rates, noise_model, ideal_rates)

    return QFTConfigAnalysis(
        config_id=config_id,
        noise_model=noise_model,
        test_mode=test_mode,
        num_qubits=num_qubits,
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
        input_state=input_state,
        period=period,
    )


# =============================================================================
# Utility Functions
# =============================================================================


def print_qft_analysis_summary(analysis: QFTConfigAnalysis) -> None:
    """Print formatted analysis summary for QFT."""
    print(f"\n{'=' * 60}")
    print(f"QFT ANALYSIS: {analysis.config_id} ({analysis.noise_model})")
    print(f"{'=' * 60}")

    print(f"\nConfiguration:")
    print(f"  Mode:   {analysis.test_mode}")
    print(f"  Qubits: {analysis.num_qubits}")
    if analysis.input_state:
        print(f"  Input:  |{analysis.input_state}⟩")
    if analysis.period:
        print(f"  Period: {analysis.period}")

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


def print_qft_comparison_table(analyses: List[QFTConfigAnalysis]) -> None:
    """Print comparison table for multiple QFT analyses."""
    print(f"\n{'=' * 110}")
    print("QFT COMPARISON TABLE")
    print(f"{'=' * 110}")
    print(
        f"{'Config':<12} {'Mode':<10} {'Noise':<12} {'Mean':>8} {'Std':>8} "
        f"{'Normal':>8} {'Degrad%':>8} {'Effect':>10}"
    )
    print(f"{'-' * 110}")

    for a in analyses:
        mode_short = "RT" if a.test_mode == "roundtrip" else "PD"
        print(
            f"{a.config_id:<12} {mode_short:<10} {a.noise_model:<12} {a.mean:>8.4f} "
            f"{a.std:>8.4f} {'Yes' if a.is_normal else 'No':>8} "
            f"{a.degradation_from_ideal:>7.1f}% {a.effect_size_vs_ideal:>10}"
        )

    print(f"{'=' * 110}")


def generate_qft_statistical_report(
    results_by_config: Dict[str, Dict[str, List[float]]],
    config_metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate comprehensive statistical report for QFT experiment.
    """
    all_analyses = []
    by_test_mode = {"roundtrip": [], "period_detection": []}
    by_noise = {}

    for config_id, noise_results in results_by_config.items():
        meta = config_metadata.get(config_id, {})
        ideal_rates = noise_results.get("IDEAL")

        for noise_model, rates in noise_results.items():
            analysis = analyze_qft_config_results(
                success_rates=rates,
                config_id=config_id,
                noise_model=noise_model,
                test_mode=meta.get("test_mode", "unknown"),
                num_qubits=meta.get("num_qubits", 0),
                input_state=meta.get("input_state"),
                period=meta.get("period"),
                ideal_rates=ideal_rates,
            )
            all_analyses.append(analysis)

            if analysis.test_mode in by_test_mode:
                by_test_mode[analysis.test_mode].append(analysis)

            if noise_model not in by_noise:
                by_noise[noise_model] = []
            by_noise[noise_model].append(analysis)

    return {
        "all_analyses": [a.to_dict() for a in all_analyses],
        "summary": {
            "total_configs": len(results_by_config),
            "total_analyses": len(all_analyses),
            "noise_models_tested": list(by_noise.keys()),
        },
        "by_test_mode": {
            mode: {
                "count": len(analyses),
                "avg_mean": np.mean([a.mean for a in analyses]) if analyses else 0,
                "avg_degradation": (
                    np.mean(
                        [a.degradation_from_ideal for a in analyses if a.noise_model != "IDEAL"]
                    )
                    if analyses
                    else 0
                ),
            }
            for mode, analyses in by_test_mode.items()
        },
        "by_noise_model": {
            noise: {
                "count": len(analyses),
                "avg_mean": np.mean([a.mean for a in analyses]),
                "avg_std": np.mean([a.std for a in analyses]),
            }
            for noise, analyses in by_noise.items()
        },
    }


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
    # QFT-specific analysis
    "analyze_scalability",
    "analyze_period_impact",
    "compare_test_modes",
    "QFTConfigAnalysis",
    "analyze_qft_config_results",
    "print_qft_analysis_summary",
    "print_qft_comparison_table",
    "generate_qft_statistical_report",
]
