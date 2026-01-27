"""
Shared statistical analysis utilities for experiment campaigns.

This module centralizes the analysis logic used by Grover, QFT, and future
experiments to avoid duplication and ensure consistent metrics across runs.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Iterable

import numpy as np
from scipy import stats


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
    cv = std / mean if mean > 0 else float("inf")

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
    """
    arr = np.array(data)

    if len(arr) < 3:
        return NormalityTestResult(
            test_name="Shapiro-Wilk",
            statistic=0.0,
            p_value=None,
            is_normal=True,
            interpretation="Insufficient data (n < 3)",
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
        interpretation=f"{'Normal' if is_normal else 'Not Normal'} (p={p:.4f})",
    )


def test_dagostino_pearson(data: List[float], alpha: float = 0.05) -> NormalityTestResult:
    """
    D'Agostino-Pearson test for normality.

    Based on skewness and kurtosis. Good for n >= 20.
    """
    arr = np.array(data)

    if len(arr) < 20:
        return NormalityTestResult(
            test_name="D'Agostino-Pearson",
            statistic=0.0,
            p_value=None,
            is_normal=True,
            interpretation="Insufficient data (n < 20)",
        )

    stat, p = stats.normaltest(arr)
    is_normal = p >= alpha

    return NormalityTestResult(
        test_name="D'Agostino-Pearson",
        statistic=stat,
        p_value=p,
        is_normal=is_normal,
        interpretation=f"{'Normal' if is_normal else 'Not Normal'} (p={p:.4f})",
    )


def test_anderson_darling(data: List[float], alpha: float = 0.05) -> NormalityTestResult:
    """
    Anderson-Darling test for normality.

    More sensitive to tails than other tests.
    """
    arr = np.array(data)

    if len(arr) < 3:
        return NormalityTestResult(
            test_name="Anderson-Darling",
            statistic=0.0,
            p_value=None,
            is_normal=True,
            interpretation="Insufficient data (n < 3)",
        )

    result = stats.anderson(arr, dist="norm")

    # Critical values at 15%, 10%, 5%, 2.5%, 1% significance
    # Index 2 is the 5% level
    critical_5pct = result.critical_values[2]
    is_normal = result.statistic < critical_5pct

    return NormalityTestResult(
        test_name="Anderson-Darling",
        statistic=result.statistic,
        p_value=None,  # Anderson-Darling doesn't give a p-value directly
        is_normal=is_normal,
        interpretation=(
            f"{'Normal' if is_normal else 'Not Normal'} "
            f"(stat={result.statistic:.4f} vs crit={critical_5pct:.4f})"
        ),
        critical_value=critical_5pct,
    )


def test_kolmogorov_smirnov(data: List[float], alpha: float = 0.05) -> NormalityTestResult:
    """
    Kolmogorov-Smirnov test for normality.

    Compares data to a fitted normal distribution. Less powerful but more general.
    """
    arr = np.array(data)

    if len(arr) < 3:
        return NormalityTestResult(
            test_name="Kolmogorov-Smirnov",
            statistic=0.0,
            p_value=None,
            is_normal=True,
            interpretation="Insufficient data (n < 3)",
        )

    # Fit normal distribution to data
    fitted_mean, fitted_std = stats.norm.fit(arr)

    # KS test against fitted normal
    stat, p = stats.kstest(arr, "norm", args=(fitted_mean, fitted_std))
    is_normal = p >= alpha

    return NormalityTestResult(
        test_name="Kolmogorov-Smirnov",
        statistic=stat,
        p_value=p,
        is_normal=is_normal,
        interpretation=f"{'Normal' if is_normal else 'Not Normal'} (p={p:.4f})",
    )


def test_normality(data: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Apply all normality tests and provide overall verdict.

    Returns a dict with test results and a majority-vote verdict.
    """
    results = {
        "shapiro_wilk": test_shapiro_wilk(data, alpha),
        "dagostino_pearson": test_dagostino_pearson(data, alpha),
        "anderson_darling": test_anderson_darling(data, alpha),
        "kolmogorov_smirnov": test_kolmogorov_smirnov(data, alpha),
    }

    # Count how many tests passed
    valid_tests = [
        r for r in results.values() if r.p_value is not None or r.critical_value is not None
    ]
    tests_normal = sum(1 for r in valid_tests if r.is_normal)

    # Majority vote
    verdict = "Normal" if tests_normal > len(valid_tests) / 2 else "Not Normal"

    return {
        "tests": {
            name: {
                "statistic": r.statistic,
                "p_value": r.p_value,
                "is_normal": r.is_normal,
                "interpretation": r.interpretation,
            }
            for name, r in results.items()
        },
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
    noise_model: str = "unknown",
) -> Dict[str, Any]:
    """
    Compare ideal vs noisy distributions to quantify noise impact.
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
            (ideal_mean - noisy_mean) / ideal_mean * 100 if ideal_mean > 0 else 0
        ),
        # Variance comparison
        "ideal_std": ideal_std,
        "noisy_std": noisy_std,
        "variance_ratio": (noisy_std**2 / ideal_std**2 if ideal_std > 0 else float("inf")),
        "std_increase": noisy_std - ideal_std,
        # Signal-to-Noise Ratio
        "snr_ideal": ideal_mean / ideal_std if ideal_std > 0 else float("inf"),
        "snr_noisy": noisy_mean / noisy_std if noisy_std > 0 else float("inf"),
        "snr_degradation": (
            (ideal_mean / ideal_std - noisy_mean / noisy_std)
            if ideal_std > 0 and noisy_std > 0
            else 0
        ),
    }

    # Statistical tests
    if len(ideal) >= 3 and len(noisy) >= 3:
        stat, p = stats.mannwhitneyu(ideal, noisy, alternative="greater")
        results["mannwhitney"] = {
            "statistic": stat,
            "p_value": p,
            "significant": p < 0.05,
            "interpretation": (
                "Ideal significantly better than Noisy" if p < 0.05 else "No significant difference"
            ),
        }

        stat, p = stats.levene(ideal, noisy)
        results["levene_variance"] = {
            "statistic": stat,
            "p_value": p,
            "equal_variance": p >= 0.05,
            "interpretation": "Equal variance" if p >= 0.05 else "Noise increases variance",
        }

        stat, p = stats.ks_2samp(ideal, noisy)
        results["ks_2sample"] = {
            "statistic": stat,
            "p_value": p,
            "same_distribution": p >= 0.05,
            "interpretation": "Same distribution" if p >= 0.05 else "Different distributions",
        }

    # Effect Size (Cohen's d)
    if ideal_std > 0 or noisy_std > 0:
        pooled_std = np.sqrt((ideal_std**2 + noisy_std**2) / 2)
        cohens_d = (ideal_mean - noisy_mean) / pooled_std if pooled_std > 0 else 0

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
            "interpretation": effect_interp,
        }

    return results


# =============================================================================
# Distribution Characterization
# =============================================================================


def characterize_distribution(data: List[float], label: str = "") -> Dict[str, Any]:
    """
    Comprehensive characterization of a distribution.
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
        },
    }


# =============================================================================
# Multi-Noise Comparison
# =============================================================================


def compare_noise_models(
    results_by_noise: Dict[str, List[float]],
    baseline_key: str = "IDEAL",
) -> Dict[str, Any]:
    """
    Compare success rate distributions across multiple noise models.
    """
    comparison = {
        "noise_models": list(results_by_noise.keys()),
        "baseline": baseline_key,
        "by_noise_model": {},
        "ranking": [],
    }

    baseline_rates = results_by_noise.get(baseline_key)

    for noise_model, rates in results_by_noise.items():
        char = characterize_distribution(rates, noise_model)

        model_results = {
            "mean": char["descriptive_stats"]["mean"],
            "std": char["descriptive_stats"]["std"],
            "median": char["descriptive_stats"]["median"],
            "is_normal": char["normality"]["verdict"] == "Normal",
            "skewness": char["shape"]["skewness"],
            "kurtosis": char["shape"]["kurtosis"],
        }

        if baseline_rates is not None and noise_model != baseline_key:
            impact = analyze_noise_impact(baseline_rates, rates, noise_model)
            model_results["vs_baseline"] = {
                "degradation_percent": impact["degradation_percent"],
                "cohens_d": impact.get("effect_size", {}).get("cohens_d", 0),
                "effect_size": impact.get("effect_size", {}).get("interpretation", "Unknown"),
            }

        comparison["by_noise_model"][noise_model] = model_results

    ranked = sorted(
        comparison["by_noise_model"].items(),
        key=lambda x: x[1]["mean"],
        reverse=True,
    )
    comparison["ranking"] = [name for name, _ in ranked]
    comparison["most_detrimental"] = ranked[-1][0] if ranked else None

    crossover = None
    for name, data in ranked:
        if data["mean"] < 0.5:
            crossover = name
            break
    comparison["crossover_noise_model"] = crossover

    return comparison


# =============================================================================
# Shared Config-Level Analysis
# =============================================================================


def analyze_config_results_base(
    success_rates: List[float],
    noise_model: str,
    ideal_rates: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Compute shared config-level statistics used by multiple experiments.
    """
    desc = compute_descriptive_stats(success_rates)
    norm = test_normality(success_rates)

    shapiro_p = norm["tests"]["shapiro_wilk"]["p_value"]

    degradation = 0.0
    cohens_d = 0.0
    effect_size = ""

    if ideal_rates is not None and noise_model != "IDEAL":
        impact = analyze_noise_impact(ideal_rates, success_rates, noise_model)
        degradation = impact["degradation_percent"]
        if "effect_size" in impact:
            cohens_d = impact["effect_size"]["cohens_d"]
            effect_size = impact["effect_size"]["interpretation"]

    return {
        "num_runs": len(success_rates),
        "mean": desc["mean"],
        "std": desc["std"],
        "median": desc["median"],
        "min_val": desc["min"],
        "max_val": desc["max"],
        "ci_lower": desc["ci_lower"],
        "ci_upper": desc["ci_upper"],
        "skewness": desc["skewness"],
        "kurtosis": desc["kurtosis"],
        "is_normal": norm["verdict"] == "Normal",
        "normality_pvalue": shapiro_p,
        "degradation_from_ideal": degradation,
        "cohens_d_vs_ideal": cohens_d,
        "effect_size_vs_ideal": effect_size,
    }


# =============================================================================
# Campaign Data Loading (Raw JSON)
# =============================================================================


def _parse_pair_from_filename(path: Path, raw_root: Path) -> Optional[Tuple[str, str]]:
    parts = path.stem.split("_")

    # Session subdir: <config>_<noise>.json
    if path.parent != raw_root:
        if len(parts) < 2:
            return None
        config_id = "_".join(parts[:-1])
        noise_id = parts[-1]
        return config_id, noise_id

    # Flat file: <config>_<noise>_YYYYMMDD_HHMMSS.json
    if len(parts) < 4:
        return None
    config_id = "_".join(parts[:-3])
    noise_id = parts[-3]
    return config_id, noise_id


def load_latest_batch_files(
    raw_dir: Path,
    ignore_names: Optional[Iterable[str]] = None,
) -> Dict[Tuple[str, str], Path]:
    """
    Load the latest raw JSON file per (config_id, noise_id).
    """
    ignore_names = set(ignore_names or [])
    raw_dir = Path(raw_dir)

    latest_files: Dict[Tuple[str, str], Path] = {}
    for path in raw_dir.rglob("*.json"):
        if path.name in ignore_names:
            continue

        pair = _parse_pair_from_filename(path, raw_dir)
        if pair is None:
            continue

        current = latest_files.get(pair)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            latest_files[pair] = path

    return latest_files


def load_batch_results(
    raw_dir: Path,
    ignore_names: Optional[Iterable[str]] = None,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Load latest raw JSON content per (config_id, noise_id).
    """
    latest_files = load_latest_batch_files(raw_dir, ignore_names=ignore_names)
    batch_results: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for pair, path in latest_files.items():
        with open(path, "r") as f:
            batch_results[pair] = json.load(f)

    return batch_results


def extract_success_rates(batch_data: Dict[str, Any]) -> List[float]:
    """
    Extract success rates from a raw batch JSON structure.
    """
    results = batch_data.get("individual_results", [])
    rates = [r.get("success_rate") for r in results if "success_rate" in r]
    return [r for r in rates if r is not None]


def build_results_by_config(
    batch_results: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, Dict[str, List[float]]]:
    """
    Build {config_id: {noise_id: [success_rates]}} from batch results.
    """
    results_by_config: Dict[str, Dict[str, List[float]]] = {}

    for (config_id, noise_id), data in batch_results.items():
        rates = extract_success_rates(data)
        results_by_config.setdefault(config_id, {})[noise_id] = rates

    return results_by_config


def build_noise_means(
    results_by_config: Dict[str, Dict[str, List[float]]],
) -> Dict[str, List[float]]:
    """
    Build {noise_id: [mean_success_rate per config]} for cross-noise comparison.
    """
    results_by_noise: Dict[str, List[float]] = {}
    for config_id, noise_results in results_by_config.items():
        for noise_id, rates in noise_results.items():
            if rates:
                results_by_noise.setdefault(noise_id, []).append(float(np.mean(rates)))
    return results_by_noise


def generate_campaign_report(
    raw_dir: Path,
    baseline_noise: str = "IDEAL",
    config_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ignore_names: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Generate a generic campaign analysis report from raw JSON results.
    """
    batch_results = load_batch_results(raw_dir, ignore_names=ignore_names)
    results_by_config = build_results_by_config(batch_results)
    results_by_noise = build_noise_means(results_by_config)

    analyses: List[Dict[str, Any]] = []
    for config_id, noise_results in results_by_config.items():
        ideal_rates = noise_results.get(baseline_noise)
        for noise_id, rates in noise_results.items():
            stats_data = analyze_config_results_base(rates, noise_id, ideal_rates)
            analysis = {
                "config_id": config_id,
                "noise_model": noise_id,
                **stats_data,
            }
            if config_metadata and config_id in config_metadata:
                analysis.update(config_metadata[config_id])
            analyses.append(analysis)

    comparison = compare_noise_models(results_by_noise, baseline_key=baseline_noise)

    return {
        "summary": {
            "total_configs": len(results_by_config),
            "total_analyses": len(analyses),
            "noise_models_tested": list(results_by_noise.keys()),
        },
        "analyses": analyses,
        "noise_comparison": comparison,
    }


__all__ = [
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
]
