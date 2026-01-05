"""
Grover Experiment Statistical Analysis

This module implements statistical analysis functions for understanding
the distribution of success rates and characterizing noise effects.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from scipy import stats
import warnings


# =============================================================================
# Descriptive Statistics
# =============================================================================

def compute_descriptive_stats(success_rates: List[float]) -> Dict[str, float]:
    """
    Compute comprehensive descriptive statistics for success rates.
    
    Args:
        success_rates: List of success rates from multiple runs
        
    Returns:
        dict: Comprehensive descriptive statistics
    """
    arr = np.array(success_rates)
    n = len(arr)
    
    if n == 0:
        return {"error": "Empty data"}
    
    mean = np.mean(arr)
    
    # Handle edge cases for standard deviation
    std = np.std(arr, ddof=1) if n > 1 else 0.0
    variance = np.var(arr, ddof=1) if n > 1 else 0.0
    sem = stats.sem(arr) if n > 1 else 0.0
    
    # Coefficient of variation (handle zero mean)
    cv = std / mean if mean > 0 else float('inf')
    
    # Skewness and kurtosis (need at least 3 points)
    if n >= 3:
        skewness = stats.skew(arr)
        kurtosis = stats.kurtosis(arr)  # Excess kurtosis (0 = normal)
    else:
        skewness = 0.0
        kurtosis = 0.0
    
    # Mode (most frequent value - may not be meaningful for continuous data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mode_result = stats.mode(arr, keepdims=True)
        mode = mode_result.mode[0] if len(mode_result.mode) > 0 else mean
    
    # Percentiles
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    
    # Confidence interval (95%)
    ci_margin = 1.96 * sem
    
    return {
        # Central Tendency
        "mean": mean,
        "median": np.median(arr),
        "mode": mode,
        
        # Spread
        "std": std,
        "variance": variance,
        "sem": sem,  # Standard error of the mean
        "range": np.max(arr) - np.min(arr),
        "iqr": iqr,
        "min": np.min(arr),
        "max": np.max(arr),
        "q1": q1,
        "q3": q3,
        
        # Relative Spread
        "cv": cv,  # Coefficient of variation
        
        # Shape
        "skewness": skewness,
        "kurtosis": kurtosis,
        
        # Confidence Interval (95%)
        "ci_lower": mean - ci_margin,
        "ci_upper": mean + ci_margin,
        "ci_margin": ci_margin,
        
        # Sample size
        "n": n,
    }


# =============================================================================
# Normality Tests
# =============================================================================

@dataclass
class NormalityTestResult:
    """Result of a normality test."""
    test_name: str
    statistic: float
    p_value: Optional[float]
    is_normal: bool
    interpretation: str
    critical_value: Optional[float] = None


def test_shapiro_wilk(data: List[float], alpha: float = 0.05) -> NormalityTestResult:
    """
    Shapiro-Wilk test for normality.
    
    Best for small samples (n < 50). Most powerful test for detecting
    departures from normality.
    
    Args:
        data: Sample data
        alpha: Significance level
        
    Returns:
        NormalityTestResult
    """
    arr = np.array(data)
    
    if len(arr) < 3:
        return NormalityTestResult(
            test_name="Shapiro-Wilk",
            statistic=0.0,
            p_value=None,
            is_normal=True,
            interpretation="Insufficient data (n < 3)"
        )
    
    # Shapiro-Wilk has a limit of 5000 samples
    if len(arr) > 5000:
        arr = np.random.choice(arr, 5000, replace=False)
    
    stat, p = stats.shapiro(arr)
    is_normal = p >= alpha
    
    return NormalityTestResult(
        test_name="Shapiro-Wilk",
        statistic=stat,
        p_value=p,
        is_normal=is_normal,
        interpretation=f"{'Normal' if is_normal else 'Not Normal'} (p={p:.4f})"
    )


def test_dagostino_pearson(data: List[float], alpha: float = 0.05) -> NormalityTestResult:
    """
    D'Agostino-Pearson test for normality.
    
    Based on skewness and kurtosis. Good for n >= 20.
    
    Args:
        data: Sample data
        alpha: Significance level
        
    Returns:
        NormalityTestResult
    """
    arr = np.array(data)
    
    if len(arr) < 20:
        return NormalityTestResult(
            test_name="D'Agostino-Pearson",
            statistic=0.0,
            p_value=None,
            is_normal=True,
            interpretation="Insufficient data (n < 20)"
        )
    
    stat, p = stats.normaltest(arr)
    is_normal = p >= alpha
    
    return NormalityTestResult(
        test_name="D'Agostino-Pearson",
        statistic=stat,
        p_value=p,
        is_normal=is_normal,
        interpretation=f"{'Normal' if is_normal else 'Not Normal'} (p={p:.4f})"
    )


def test_anderson_darling(data: List[float], alpha: float = 0.05) -> NormalityTestResult:
    """
    Anderson-Darling test for normality.
    
    More sensitive to tails than other tests.
    
    Args:
        data: Sample data
        alpha: Significance level (uses 5% critical value)
        
    Returns:
        NormalityTestResult
    """
    arr = np.array(data)
    
    if len(arr) < 3:
        return NormalityTestResult(
            test_name="Anderson-Darling",
            statistic=0.0,
            p_value=None,
            is_normal=True,
            interpretation="Insufficient data (n < 3)"
        )
    
    result = stats.anderson(arr, dist='norm')
    
    # Critical values at 15%, 10%, 5%, 2.5%, 1% significance
    # Index 2 is the 5% level
    critical_5pct = result.critical_values[2]
    is_normal = result.statistic < critical_5pct
    
    return NormalityTestResult(
        test_name="Anderson-Darling",
        statistic=result.statistic,
        p_value=None,  # Anderson-Darling doesn't give a p-value directly
        is_normal=is_normal,
        interpretation=f"{'Normal' if is_normal else 'Not Normal'} (stat={result.statistic:.4f} vs crit={critical_5pct:.4f})",
        critical_value=critical_5pct
    )


def test_kolmogorov_smirnov(data: List[float], alpha: float = 0.05) -> NormalityTestResult:
    """
    Kolmogorov-Smirnov test for normality.
    
    Compares data to a fitted normal distribution. Less powerful but more general.
    
    Args:
        data: Sample data
        alpha: Significance level
        
    Returns:
        NormalityTestResult
    """
    arr = np.array(data)
    
    if len(arr) < 3:
        return NormalityTestResult(
            test_name="Kolmogorov-Smirnov",
            statistic=0.0,
            p_value=None,
            is_normal=True,
            interpretation="Insufficient data (n < 3)"
        )
    
    # Fit normal distribution to data
    fitted_mean, fitted_std = stats.norm.fit(arr)
    
    # KS test against fitted normal
    stat, p = stats.kstest(arr, 'norm', args=(fitted_mean, fitted_std))
    is_normal = p >= alpha
    
    return NormalityTestResult(
        test_name="Kolmogorov-Smirnov",
        statistic=stat,
        p_value=p,
        is_normal=is_normal,
        interpretation=f"{'Normal' if is_normal else 'Not Normal'} (p={p:.4f})"
    )


def test_normality(data: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Apply all normality tests and provide overall verdict.
    
    Args:
        data: Sample data
        alpha: Significance level
        
    Returns:
        dict: Results from all tests plus overall verdict
    """
    results = {
        "shapiro_wilk": test_shapiro_wilk(data, alpha),
        "dagostino_pearson": test_dagostino_pearson(data, alpha),
        "anderson_darling": test_anderson_darling(data, alpha),
        "kolmogorov_smirnov": test_kolmogorov_smirnov(data, alpha),
    }
    
    # Count how many tests passed
    valid_tests = [r for r in results.values() if r.p_value is not None or r.critical_value is not None]
    tests_normal = sum(1 for r in valid_tests if r.is_normal)
    
    # Majority vote
    verdict = "Normal" if tests_normal > len(valid_tests) / 2 else "Not Normal"
    
    return {
        "tests": {name: {
            "statistic": r.statistic,
            "p_value": r.p_value,
            "is_normal": r.is_normal,
            "interpretation": r.interpretation,
        } for name, r in results.items()},
        "tests_passed": tests_normal,
        "total_valid_tests": len(valid_tests),
        "verdict": verdict,
        "alpha": alpha,
    }


# =============================================================================
# Noise Impact Analysis
# =============================================================================

def analyze_noise_impact(
    ideal_rates: List[float], 
    noisy_rates: List[float],
    noise_model: str = "unknown"
) -> Dict[str, Any]:
    """
    Compare ideal vs noisy distributions to quantify noise impact.
    
    Args:
        ideal_rates: Success rates from ideal (no-noise) simulation
        noisy_rates: Success rates from noisy simulation
        noise_model: Name of the noise model for labeling
        
    Returns:
        dict: Comprehensive noise impact analysis
    """
    ideal = np.array(ideal_rates)
    noisy = np.array(noisy_rates)
    
    ideal_mean = np.mean(ideal)
    noisy_mean = np.mean(noisy)
    ideal_std = np.std(ideal, ddof=1) if len(ideal) > 1 else 0
    noisy_std = np.std(noisy, ddof=1) if len(noisy) > 1 else 0
    
    results = {
        "noise_model": noise_model,
        
        # Basic comparison
        "ideal_mean": ideal_mean,
        "noisy_mean": noisy_mean,
        "mean_degradation": ideal_mean - noisy_mean,
        "degradation_percent": (
            (ideal_mean - noisy_mean) / ideal_mean * 100 
            if ideal_mean > 0 else 0
        ),
        
        # Variance comparison
        "ideal_std": ideal_std,
        "noisy_std": noisy_std,
        "variance_ratio": (
            noisy_std**2 / ideal_std**2 
            if ideal_std > 0 else float('inf')
        ),
        "std_increase": noisy_std - ideal_std,
        
        # Signal-to-Noise Ratio
        "snr_ideal": ideal_mean / ideal_std if ideal_std > 0 else float('inf'),
        "snr_noisy": noisy_mean / noisy_std if noisy_std > 0 else float('inf'),
        "snr_degradation": (
            (ideal_mean / ideal_std - noisy_mean / noisy_std)
            if ideal_std > 0 and noisy_std > 0 else 0
        ),
    }
    
    # Statistical tests
    
    # Mann-Whitney U Test (non-parametric)
    if len(ideal) >= 3 and len(noisy) >= 3:
        stat, p = stats.mannwhitneyu(ideal, noisy, alternative='greater')
        results["mannwhitney"] = {
            "statistic": stat,
            "p_value": p,
            "significant": p < 0.05,
            "interpretation": (
                "Ideal significantly better than Noisy" 
                if p < 0.05 else "No significant difference"
            )
        }
    
    # Levene's Test for equality of variances
    if len(ideal) >= 3 and len(noisy) >= 3:
        stat, p = stats.levene(ideal, noisy)
        results["levene_variance"] = {
            "statistic": stat,
            "p_value": p,
            "equal_variance": p >= 0.05,
            "interpretation": (
                "Equal variance" 
                if p >= 0.05 else "Noise increases variance"
            )
        }
    
    # Effect Size (Cohen's d)
    if ideal_std > 0 or noisy_std > 0:
        pooled_std = np.sqrt((ideal_std**2 + noisy_std**2) / 2)
        cohens_d = (ideal_mean - noisy_mean) / pooled_std if pooled_std > 0 else 0
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_interp = "Negligible"
        elif abs(cohens_d) < 0.5:
            effect_interp = "Small"
        elif abs(cohens_d) < 0.8:
            effect_interp = "Medium"
        else:
            effect_interp = "Large"
        
        results["effect_size"] = {
            "cohens_d": cohens_d,
            "interpretation": effect_interp
        }
    
    # Kolmogorov-Smirnov 2-sample test
    if len(ideal) >= 3 and len(noisy) >= 3:
        stat, p = stats.ks_2samp(ideal, noisy)
        results["ks_2sample"] = {
            "statistic": stat,
            "p_value": p,
            "same_distribution": p >= 0.05,
            "interpretation": (
                "Same distribution" 
                if p >= 0.05 else "Different distributions"
            )
        }
    
    return results


# =============================================================================
# Distribution Characterization
# =============================================================================

def characterize_distribution(data: List[float], label: str = "") -> Dict[str, Any]:
    """
    Comprehensive characterization of a distribution.
    
    Args:
        data: Sample data
        label: Label for the distribution
        
    Returns:
        dict: Complete distribution characterization
    """
    desc_stats = compute_descriptive_stats(data)
    normality = test_normality(data)
    
    # Determine distribution shape
    skew = desc_stats["skewness"]
    kurt = desc_stats["kurtosis"]
    
    if abs(skew) < 0.5:
        skew_desc = "Symmetric"
    elif skew > 0:
        skew_desc = "Right-skewed (positive)"
    else:
        skew_desc = "Left-skewed (negative)"
    
    if abs(kurt) < 0.5:
        kurt_desc = "Normal tails (mesokurtic)"
    elif kurt > 0:
        kurt_desc = "Heavy tails (leptokurtic)"
    else:
        kurt_desc = "Light tails (platykurtic)"
    
    return {
        "label": label,
        "descriptive_stats": desc_stats,
        "normality": normality,
        "shape": {
            "skewness": skew,
            "skewness_interpretation": skew_desc,
            "kurtosis": kurt,
            "kurtosis_interpretation": kurt_desc,
        },
        "summary": {
            "is_normal": normality["verdict"] == "Normal",
            "central_tendency": f"Mean={desc_stats['mean']:.4f}, Median={desc_stats['median']:.4f}",
            "spread": f"Std={desc_stats['std']:.4f}, IQR={desc_stats['iqr']:.4f}",
            "shape": f"{skew_desc}, {kurt_desc}",
        }
    }


# =============================================================================
# Multi-Noise Comparison
# =============================================================================

def compare_noise_models(
    results_by_noise: Dict[str, List[float]],
    baseline_key: str = "IDEAL"
) -> Dict[str, Any]:
    """
    Compare success rate distributions across multiple noise models.
    
    Args:
        results_by_noise: Dictionary mapping noise model names to success rate lists
        baseline_key: Key for the baseline (no-noise) condition
        
    Returns:
        dict: Comprehensive comparison across all noise models
    """
    comparison = {
        "noise_models": list(results_by_noise.keys()),
        "baseline": baseline_key,
        "by_noise_model": {},
        "ranking": [],
    }
    
    # Get baseline if available
    baseline_rates = results_by_noise.get(baseline_key)
    
    for noise_model, rates in results_by_noise.items():
        # Characterize this distribution
        char = characterize_distribution(rates, noise_model)
        
        model_results = {
            "mean": char["descriptive_stats"]["mean"],
            "std": char["descriptive_stats"]["std"],
            "median": char["descriptive_stats"]["median"],
            "is_normal": char["normality"]["verdict"] == "Normal",
            "skewness": char["shape"]["skewness"],
            "kurtosis": char["shape"]["kurtosis"],
        }
        
        # Compare to baseline if this isn't the baseline
        if baseline_rates is not None and noise_model != baseline_key:
            impact = analyze_noise_impact(baseline_rates, rates, noise_model)
            model_results["vs_baseline"] = {
                "degradation_percent": impact["degradation_percent"],
                "cohens_d": impact.get("effect_size", {}).get("cohens_d", 0),
                "effect_size": impact.get("effect_size", {}).get("interpretation", "Unknown"),
            }
        
        comparison["by_noise_model"][noise_model] = model_results
    
    # Rank noise models by mean success rate (best to worst)
    ranked = sorted(
        comparison["by_noise_model"].items(),
        key=lambda x: x[1]["mean"],
        reverse=True
    )
    comparison["ranking"] = [name for name, _ in ranked]
    comparison["most_detrimental"] = ranked[-1][0] if ranked else None
    
    # Find crossover point (where mean drops below 50%)
    crossover = None
    for name, data in ranked:
        if data["mean"] < 0.5:
            crossover = name
            break
    comparison["crossover_noise_model"] = crossover
    
    return comparison


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
    ideal_rates: Optional[List[float]] = None
) -> ConfigAnalysis:
    """
    Complete statistical analysis for a configuration's results.
    
    Args:
        success_rates: List of success rates from multiple runs
        config_id: Configuration identifier
        noise_model: Noise model identifier
        ideal_rates: Optional ideal (no-noise) rates for comparison
        
    Returns:
        ConfigAnalysis: Complete analysis
    """
    desc = compute_descriptive_stats(success_rates)
    norm = test_normality(success_rates)
    
    # Get Shapiro-Wilk p-value if available
    shapiro_p = norm["tests"]["shapiro_wilk"]["p_value"]
    
    # Calculate noise impact if ideal rates provided
    degradation = 0.0
    cohens_d = 0.0
    effect_size = ""
    
    if ideal_rates is not None and noise_model != "IDEAL":
        impact = analyze_noise_impact(ideal_rates, success_rates, noise_model)
        degradation = impact["degradation_percent"]
        if "effect_size" in impact:
            cohens_d = impact["effect_size"]["cohens_d"]
            effect_size = impact["effect_size"]["interpretation"]
    
    return ConfigAnalysis(
        config_id=config_id,
        noise_model=noise_model,
        num_runs=len(success_rates),
        mean=desc["mean"],
        std=desc["std"],
        median=desc["median"],
        min_val=desc["min"],
        max_val=desc["max"],
        ci_lower=desc["ci_lower"],
        ci_upper=desc["ci_upper"],
        skewness=desc["skewness"],
        kurtosis=desc["kurtosis"],
        is_normal=norm["verdict"] == "Normal",
        normality_pvalue=shapiro_p,
        degradation_from_ideal=degradation,
        cohens_d_vs_ideal=cohens_d,
        effect_size_vs_ideal=effect_size,
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
        print(f"  Effect Size: {analysis.effect_size_vs_ideal} (d={analysis.cohens_d_vs_ideal:.2f})")


def print_comparison_table(analyses: List[ConfigAnalysis]) -> None:
    """Print comparison table for multiple analyses."""
    print(f"\n{'=' * 100}")
    print("COMPARISON TABLE")
    print(f"{'=' * 100}")
    print(f"{'Config':<10} {'Noise':<12} {'Mean':>8} {'Std':>8} {'Median':>8} "
          f"{'Normal':>8} {'Degrad%':>8} {'Effect':>10}")
    print(f"{'-' * 100}")
    
    for a in analyses:
        print(f"{a.config_id:<10} {a.noise_model:<12} {a.mean:>8.4f} {a.std:>8.4f} "
              f"{a.median:>8.4f} {'Yes' if a.is_normal else 'No':>8} "
              f"{a.degradation_from_ideal:>7.1f}% {a.effect_size_vs_ideal:>10}")
    
    print(f"{'=' * 100}")

