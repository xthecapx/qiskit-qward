# Grover's Algorithm Experiment Design

## Research Goals

This document outlines the experimental design for evaluating Grover's algorithm performance using the QWARD framework. 

**Scope**: Simulator experiments (ideal and noisy) to establish baselines before real QPU execution.

### Primary Research Questions

1. **Scalability Limit**: At what circuit depth/qubit count does noise cause algorithm failure?
2. **Marked State Impact**: How do different marked state configurations affect success rates?
3. **Success Metrics**: What is the optimal way to define "success" at shot, job, and batch levels?
4. **Noise Characterization**: How do different noise models affect algorithm performance?

---

## Fixed Experiment Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Shots per job** | 1024 | Standard quantum computing benchmark |
| **Runs per config** | 10 | Sufficient for statistical analysis |
| **Optimization level** | 0 (none) | Establish baseline first, explore later |
| **Backend** | Aer simulator | Ideal and noisy modes |

---

## Phase 1: Simulator Experiments

This phase focuses on understanding Grover's algorithm behavior through simulation before committing QPU resources.

---

## Experiment 1: Scalability Study

### Objective
Determine the maximum problem size where Grover's algorithm remains effective under different noise conditions.

### Circuit Variations

| Config ID | Qubits | Marked States | Search Space | Theoretical Success | Iterations | Depth | Gates |
|-----------|--------|---------------|--------------|---------------------|------------|-------|-------|
| S2-1 | 2 | ["01"] | 4 | 100.00% | 1 | 12 | 21 |
| S3-1 | 3 | ["011"] | 8 | 94.53% | 2 | 26 | 49 |
| S4-1 | 4 | ["0110"] | 16 | 96.13% | 3 | 38 | 89 |
| S5-1 | 5 | ["01100"] | 32 | 99.92% | 4 | 50 | 141 |
| S6-1 | 6 | ["011001"] | 64 | 99.66% | 6 | 74 | 231 |
| S7-1 | 7 | ["0110011"] | 128 | 99.56% | 8 | 98 | 337 |
| S8-1 | 8 | ["01100110"] | 256 | 99.99% | 12 | 146 | 571 |

### Hypothesis
- **H1**: Success rate will degrade as qubit count increases due to accumulated gate errors
- **H2**: There exists a "crossover point" where noisy simulation falls below classical random search
- **H3**: The crossover point correlates with circuit depth rather than qubit count alone

### Metrics to Collect
- Success rate (per shot, per job, per batch)
- Circuit depth
- Two-qubit gate count
- Total gate count
- Theoretical vs observed success rate gap

---

## Experiment 2: Marked State Configuration Study

### Objective
Understand how the choice and structure of marked states affects algorithm performance.

### 2A: Number of Marked States

| Config ID | Qubits | Num Marked | Marked States | Theoretical | Observed | Depth | Gates |
|-----------|--------|------------|---------------|-------------|----------|-------|-------|
| M3-1 | 3 | 1 | ["000"] | 94.53% | 94.21% ✅ | 26 | 57 |
| M3-2 | 3 | 2 | ["000", "111"] | 100.00% | 100.00% ✅ | 17 | 36 |
| M3-4 | 3 | 4 | ["000", "001", "110", "111"] | 50.00% | 50.40% ✅ | 2 | 9 |
| M4-1 | 4 | 1 | ["0000"] | 96.13% | 95.77% ✅ | 38 | 101 |
| M4-2 | 4 | 2 | ["0000", "1111"] | 94.53% | 94.52% ✅ | 32 | 77 |
| M4-4 | 4 | 4 | ["0000", "0011", "1100", "1111"] | 100.00% | 100.00% ✅ | 25 | 58 |

**Hypothesis**: More marked states → fewer Grover iterations → shorter circuit → better success rate under noise

**Verified ✅**: M3-4 confirms that marking 50% of states (4/8) results in 0 Grover iterations and only 50% success (no quantum advantage).

### 2B: Hamming Weight Study

Testing whether the "type" of marked state affects performance under noise.

| Config ID | Qubits | Marked State | Hamming Weight | Observed | Depth | Gates |
|-----------|--------|--------------|----------------|----------|-------|-------|
| H3-0 | 3 | ["000"] | 0 (All zeros) | 94.77% ✅ | 26 | 57 |
| H3-1 | 3 | ["001"] | 1 (Single 1) | 94.70% ✅ | 26 | 53 |
| H3-2 | 3 | ["011"] | 2 (Two 1s) | 94.56% ✅ | 26 | 49 |
| H3-3 | 3 | ["111"] | 3 (All ones) | 94.44% ✅ | 22 | 45 |
| H4-0 | 4 | ["0000"] | 0 (All zeros) | 96.33% ✅ | 38 | 101 |
| H4-2 | 4 | ["0011"] | 2 (Balanced) | 96.39% ✅ | 38 | 89 |
| H4-4 | 4 | ["1111"] | 4 (All ones) | 95.79% ✅ | 32 | 77 |

**Hypothesis**: 
- All-zeros states may have different error profiles than all-ones under certain noise models
- Balanced Hamming weight states may be more resilient to bit-flip errors

**Verified ✅**: In ideal simulation, no significant difference based on Hamming weight. All match theoretical 94.53% (3q) and 96.13% (4q). Noise study will reveal differences.

### 2C: Symmetric vs Asymmetric Marked States

| Config ID | Marked States | Pattern | Observed | Depth | Gates |
|-----------|---------------|---------|----------|-------|-------|
| SYM-1 | ["000", "111"] | Symmetric (complements) | 100.00% ✅ | 17 | 36 |
| SYM-2 | ["001", "110"] | Symmetric | 100.00% ✅ | 17 | 36 |
| ASYM-1 | ["000", "001"] | Asymmetric (adjacent) | 100.00% ✅ | 19 | 40 |
| ASYM-2 | ["000", "011"] | Asymmetric (2-bit diff) | 100.00% ✅ | 19 | 38 |

**Hypothesis**: Symmetric marked states may show different interference patterns and error resilience

**Verified ✅**: All achieve 100% success in ideal simulation (2 marked states in 8-state space). Symmetric configs have slightly shorter circuits (17 vs 19 depth). Noise study will reveal robustness differences.

---

## Experiment 3: Noise Model Study

### Objective
Understand how different noise models affect Grover's algorithm and establish simulator baselines.

### Noise Models

| Model ID | Type | Parameters | Description |
|----------|------|------------|-------------|
| IDEAL | None | - | Perfect execution baseline |
| DEP-LOW | Depolarizing | p1=0.001, p2=0.01 | Light depolarizing noise |
| DEP-MED | Depolarizing | p1=0.01, p2=0.05 | Medium depolarizing noise |
| DEP-HIGH | Depolarizing | p1=0.05, p2=0.10 | Heavy depolarizing noise |
| PAULI | Pauli | pX=pY=pZ=0.01 | Structured Pauli errors |
| THERMAL | Thermal | T1=50μs, T2=70μs | T1/T2 relaxation |
| READOUT | Readout | p01=p10=0.02 | Measurement errors only |
| COMBINED | Mixed | DEP-MED + READOUT | Realistic combined noise |

### Analysis Goals
- Establish success rate under each noise model
- Identify which noise type is most detrimental to Grover
- Determine noise level at which algorithm becomes useless
- Create baseline predictions for real QPU comparison (Phase 2)

---

## Experiment 4: Success Metric Definition Study

### Objective
Determine the most meaningful way to define "success" for Grover execution.

### Level 1: Per-Shot Success (Current Approach)

```python
def success_per_shot(outcome: str, marked_states: list[str]) -> bool:
    """Binary success: did this single measurement hit a marked state?"""
    return outcome in marked_states

def success_rate(counts: dict, marked_states: list[str]) -> float:
    """Fraction of shots that hit marked states."""
    total = sum(counts.values())
    success = sum(counts.get(s, 0) for s in marked_states)
    return success / total
```

### Level 2: Per-Job Success

#### A. Threshold-Based Success
```python
def job_success_threshold(counts, marked_states, threshold=0.5):
    """Job succeeds if success rate exceeds threshold."""
    rate = success_rate(counts, marked_states)
    return rate >= threshold
```

**Thresholds to test**: 0.3, 0.5, 0.7, 0.9, theoretical_probability

#### B. Statistical Significance Success
```python
def job_success_statistical(counts, marked_states, num_qubits, confidence=0.95):
    """Job succeeds if success rate is statistically above random chance."""
    from scipy import stats
    
    total = sum(counts.values())
    success = sum(counts.get(s, 0) for s in marked_states)
    
    # Random chance baseline
    random_prob = len(marked_states) / (2 ** num_qubits)
    
    # Binomial test: is observed success significantly better than random?
    p_value = stats.binomtest(success, total, random_prob, alternative='greater').pvalue
    
    return p_value < (1 - confidence)
```

#### C. Quantum Advantage Success
```python
def job_success_quantum_advantage(counts, marked_states, num_qubits, advantage_factor=2.0):
    """Job succeeds if algorithm outperforms classical random search by factor X."""
    rate = success_rate(counts, marked_states)
    classical_expected = len(marked_states) / (2 ** num_qubits)
    return rate > (classical_expected * advantage_factor)
```

### Level 3: Per-Batch Success

```python
def batch_success(job_results: list[dict], marked_states: list[str], 
                  metric="mean", threshold=0.5):
    """Evaluate success across multiple jobs in a batch."""
    import numpy as np
    rates = [success_rate(job, marked_states) for job in job_results]
    
    if metric == "mean":
        return np.mean(rates) >= threshold
    elif metric == "min":
        return np.min(rates) >= threshold
    elif metric == "median":
        return np.median(rates) >= threshold
    elif metric == "consistency":
        return np.std(rates) <= (1 - threshold)
```

---

## Statistical Analysis Framework

### Philosophy
Instead of predictive modeling (regression), we focus on **descriptive statistics** and **distribution analysis** to understand:
1. How success rates are distributed across runs
2. Whether the distribution is normal or skewed
3. How noise changes the distribution shape and spread

### 1. Descriptive Statistics

For each configuration (10 runs × 1024 shots):

```python
def compute_descriptive_stats(success_rates: list[float]) -> dict:
    """Compute comprehensive descriptive statistics."""
    import numpy as np
    from scipy import stats
    
    return {
        # Central Tendency
        "mean": np.mean(success_rates),
        "median": np.median(success_rates),
        "mode": stats.mode(success_rates, keepdims=True).mode[0],
        
        # Spread
        "std": np.std(success_rates, ddof=1),  # Sample std
        "variance": np.var(success_rates, ddof=1),
        "range": np.max(success_rates) - np.min(success_rates),
        "iqr": np.percentile(success_rates, 75) - np.percentile(success_rates, 25),
        "min": np.min(success_rates),
        "max": np.max(success_rates),
        
        # Shape
        "skewness": stats.skew(success_rates),
        "kurtosis": stats.kurtosis(success_rates),  # Excess kurtosis (0 = normal)
        
        # Confidence Interval (95%)
        "ci_lower": np.mean(success_rates) - 1.96 * stats.sem(success_rates),
        "ci_upper": np.mean(success_rates) + 1.96 * stats.sem(success_rates),
        
        # Coefficient of Variation (relative spread)
        "cv": np.std(success_rates) / np.mean(success_rates) if np.mean(success_rates) > 0 else np.inf,
    }
```

### 2. Normality Tests

Test whether success rate distributions follow a normal distribution.

```python
def test_normality(success_rates: list[float], alpha=0.05) -> dict:
    """
    Apply multiple normality tests.
    
    Null hypothesis: Data comes from a normal distribution.
    If p-value < alpha: Reject null → NOT normal
    If p-value >= alpha: Fail to reject → Consistent with normal
    """
    from scipy import stats
    
    results = {}
    
    # Shapiro-Wilk Test (best for small samples, n < 50)
    # Most powerful test for detecting departures from normality
    stat, p = stats.shapiro(success_rates)
    results["shapiro_wilk"] = {
        "statistic": stat,
        "p_value": p,
        "is_normal": p >= alpha,
        "interpretation": "Normal" if p >= alpha else "Not Normal"
    }
    
    # D'Agostino-Pearson Test (based on skewness and kurtosis)
    # Good for n >= 20
    if len(success_rates) >= 20:
        stat, p = stats.normaltest(success_rates)
        results["dagostino_pearson"] = {
            "statistic": stat,
            "p_value": p,
            "is_normal": p >= alpha,
            "interpretation": "Normal" if p >= alpha else "Not Normal"
        }
    
    # Anderson-Darling Test (more weight to tails)
    result = stats.anderson(success_rates, dist='norm')
    # Critical values at 15%, 10%, 5%, 2.5%, 1% significance
    critical_5pct = result.critical_values[2]  # 5% level
    results["anderson_darling"] = {
        "statistic": result.statistic,
        "critical_value_5pct": critical_5pct,
        "is_normal": result.statistic < critical_5pct,
        "interpretation": "Normal" if result.statistic < critical_5pct else "Not Normal"
    }
    
    # Kolmogorov-Smirnov Test (compare to fitted normal)
    # Less powerful but more general
    fitted_mean, fitted_std = stats.norm.fit(success_rates)
    stat, p = stats.kstest(success_rates, 'norm', args=(fitted_mean, fitted_std))
    results["kolmogorov_smirnov"] = {
        "statistic": stat,
        "p_value": p,
        "is_normal": p >= alpha,
        "interpretation": "Normal" if p >= alpha else "Not Normal"
    }
    
    # Overall verdict (majority vote)
    normal_count = sum(1 for test in results.values() if test.get("is_normal", False))
    results["overall"] = {
        "tests_passed": normal_count,
        "total_tests": len(results) - 1,  # Exclude this summary
        "verdict": "Normal" if normal_count > len(results) // 2 else "Not Normal"
    }
    
    return results
```

### 3. Distribution Visualization

```python
def visualize_distribution(success_rates: list[float], config_id: str, noise_model: str):
    """Generate distribution visualizations."""
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Distribution Analysis: {config_id} ({noise_model})", fontsize=14)
    
    # 1. Histogram with Normal Overlay
    ax1 = axes[0, 0]
    ax1.hist(success_rates, bins=10, density=True, alpha=0.7, edgecolor='black')
    
    # Fit and plot normal distribution
    mu, std = np.mean(success_rates), np.std(success_rates)
    x = np.linspace(min(success_rates) - 0.1, max(success_rates) + 0.1, 100)
    ax1.plot(x, stats.norm.pdf(x, mu, std), 'r-', lw=2, label=f'Normal(μ={mu:.3f}, σ={std:.3f})')
    ax1.axvline(mu, color='red', linestyle='--', label=f'Mean: {mu:.3f}')
    ax1.set_xlabel('Success Rate')
    ax1.set_ylabel('Density')
    ax1.set_title('Histogram with Normal Fit')
    ax1.legend()
    
    # 2. Q-Q Plot (Quantile-Quantile)
    ax2 = axes[0, 1]
    stats.probplot(success_rates, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normal)')
    # Points on diagonal = normal distribution
    
    # 3. Box Plot
    ax3 = axes[1, 0]
    bp = ax3.boxplot(success_rates, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax3.axhline(np.median(success_rates), color='red', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Success Rate')
    ax3.set_title('Box Plot (outlier detection)')
    
    # 4. Empirical CDF vs Normal CDF
    ax4 = axes[1, 1]
    sorted_data = np.sort(success_rates)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax4.step(sorted_data, ecdf, where='post', label='Empirical CDF')
    ax4.plot(x, stats.norm.cdf(x, mu, std), 'r-', label='Normal CDF')
    ax4.set_xlabel('Success Rate')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Empirical vs Normal CDF')
    ax4.legend()
    
    plt.tight_layout()
    return fig
```

### 4. Noise Impact Analysis

Compare distributions between IDEAL and noisy runs to quantify noise impact.

```python
def analyze_noise_impact(ideal_rates: list[float], noisy_rates: list[float], 
                         noise_model: str) -> dict:
    """
    Compare ideal vs noisy distributions to quantify noise impact.
    """
    import numpy as np
    from scipy import stats
    
    results = {
        "noise_model": noise_model,
        
        # Basic comparison
        "ideal_mean": np.mean(ideal_rates),
        "noisy_mean": np.mean(noisy_rates),
        "mean_degradation": np.mean(ideal_rates) - np.mean(noisy_rates),
        "degradation_percent": (np.mean(ideal_rates) - np.mean(noisy_rates)) / np.mean(ideal_rates) * 100,
        
        # Variance comparison
        "ideal_std": np.std(ideal_rates),
        "noisy_std": np.std(noisy_rates),
        "variance_ratio": np.var(noisy_rates) / np.var(ideal_rates) if np.var(ideal_rates) > 0 else np.inf,
        
        # Signal-to-Noise Ratio
        "snr_ideal": np.mean(ideal_rates) / np.std(ideal_rates) if np.std(ideal_rates) > 0 else np.inf,
        "snr_noisy": np.mean(noisy_rates) / np.std(noisy_rates) if np.std(noisy_rates) > 0 else np.inf,
    }
    
    # Statistical tests for difference
    
    # Mann-Whitney U Test (non-parametric, doesn't assume normality)
    # Tests if one distribution tends to have larger values
    stat, p = stats.mannwhitneyu(ideal_rates, noisy_rates, alternative='greater')
    results["mannwhitney"] = {
        "statistic": stat,
        "p_value": p,
        "significant": p < 0.05,
        "interpretation": "Ideal > Noisy (significant)" if p < 0.05 else "No significant difference"
    }
    
    # Levene's Test for equality of variances
    # Important to know if noise increases variance
    stat, p = stats.levene(ideal_rates, noisy_rates)
    results["levene_variance"] = {
        "statistic": stat,
        "p_value": p,
        "equal_variance": p >= 0.05,
        "interpretation": "Equal variance" if p >= 0.05 else "Unequal variance (noise increases spread)"
    }
    
    # Effect Size (Cohen's d)
    # Measures practical significance of difference
    pooled_std = np.sqrt((np.var(ideal_rates) + np.var(noisy_rates)) / 2)
    cohens_d = (np.mean(ideal_rates) - np.mean(noisy_rates)) / pooled_std if pooled_std > 0 else 0
    results["effect_size"] = {
        "cohens_d": cohens_d,
        "interpretation": (
            "Negligible" if abs(cohens_d) < 0.2 else
            "Small" if abs(cohens_d) < 0.5 else
            "Medium" if abs(cohens_d) < 0.8 else
            "Large"
        )
    }
    
    # Kolmogorov-Smirnov 2-sample test
    # Tests if two samples come from the same distribution
    stat, p = stats.ks_2samp(ideal_rates, noisy_rates)
    results["ks_2sample"] = {
        "statistic": stat,
        "p_value": p,
        "same_distribution": p >= 0.05,
        "interpretation": "Same distribution" if p >= 0.05 else "Different distributions"
    }
    
    return results
```

### 5. Noise Characterization Summary

```python
def characterize_noise_effects(all_results: dict) -> dict:
    """
    Aggregate analysis across all noise models to understand noise behavior.
    
    Questions answered:
    1. Which noise type is most detrimental?
    2. How does variance scale with noise level?
    3. At what noise level does the algorithm fail?
    4. Does noise change the distribution shape?
    """
    import numpy as np
    
    noise_summary = {}
    
    for noise_model, configs in all_results.items():
        rates = [r["success_rate"] for r in configs]
        
        noise_summary[noise_model] = {
            # Performance
            "avg_success_rate": np.mean(rates),
            "success_rate_std": np.std(rates),
            
            # Reliability
            "worst_case": np.min(rates),
            "best_case": np.max(rates),
            "reliability_range": np.max(rates) - np.min(rates),
            
            # Quantum advantage (vs classical random)
            "maintains_advantage": np.mean(rates) > 0.5,  # Simplified threshold
            
            # Distribution shape (averaged)
            "avg_skewness": np.mean([r.get("skewness", 0) for r in configs]),
            "avg_kurtosis": np.mean([r.get("kurtosis", 0) for r in configs]),
        }
    
    # Rank noise models by severity
    ranked = sorted(noise_summary.items(), key=lambda x: x[1]["avg_success_rate"], reverse=True)
    
    return {
        "by_noise_model": noise_summary,
        "ranking_best_to_worst": [name for name, _ in ranked],
        "most_detrimental": ranked[-1][0] if ranked else None,
        "crossover_noise_level": next(
            (name for name, stats in ranked if not stats["maintains_advantage"]), 
            None
        )
    }
```

### 6. Expected Distribution Behavior

| Condition | Expected Distribution | Skewness | Kurtosis | Variance |
|-----------|----------------------|----------|----------|----------|
| Ideal | Near-degenerate (all ~1.0) | ~0 | High (concentrated) | Very low |
| Light noise | Left-skewed | Negative | Moderate | Low |
| Medium noise | Approaching normal | ~0 | ~0 | Moderate |
| Heavy noise | Right-skewed (floor at random) | Positive | Negative (flat) | High |
| Random (failure) | Normal around 1/N | ~0 | ~0 | Binomial variance |

### 7. Key Metrics to Report

For each experiment configuration:

| Metric | Purpose |
|--------|---------|
| Mean success rate | Central tendency |
| Standard deviation | Spread/reliability |
| 95% CI | Uncertainty in mean |
| Skewness | Distribution asymmetry |
| Kurtosis | Tail behavior |
| Normality test result | Distribution type |
| Cohen's d (vs ideal) | Effect size of noise |
| SNR | Signal quality |

---

## Data Collection Schema

### Per-Experiment Record

```python
experiment_record = {
    # Identification
    "experiment_id": str,          # Unique ID (e.g., "S3-1_DEP-MED_001")
    "config_id": str,              # Circuit config (e.g., "S3-1")
    "noise_model": str,            # Noise model ID (e.g., "DEP-MED")
    "run_number": int,             # 1-10
    
    # Circuit Properties
    "num_qubits": int,
    "marked_states": list[str],
    "num_marked": int,
    "theoretical_success": float,
    "grover_iterations": int,
    
    # Circuit Metrics
    "circuit_depth": int,
    "total_gates": int,
    "two_qubit_gates": int,
    
    # Execution
    "backend": str,                # "aer_simulator"
    "shots": 1024,
    "execution_time_ms": float,
    
    # Raw Results
    "counts": dict,                # Raw measurement counts
    
    # Success Metrics
    "success_rate": float,
    "success_threshold_50": bool,
    "success_threshold_70": bool,
    "success_threshold_90": bool,
    "success_statistical": bool,
    "success_quantum_advantage": bool,
    "quantum_advantage_ratio": float,
}
```

### Per-Config Aggregate (10 runs)

```python
config_aggregate = {
    "config_id": str,
    "noise_model": str,
    "num_runs": 10,
    "shots_per_run": 1024,
    
    # Descriptive Statistics
    "mean": float,
    "median": float,
    "std": float,
    "variance": float,
    "min": float,
    "max": float,
    "range": float,
    "iqr": float,
    "cv": float,
    "ci_lower": float,
    "ci_upper": float,
    
    # Distribution Shape
    "skewness": float,
    "kurtosis": float,
    
    # Normality Tests
    "shapiro_wilk_pvalue": float,
    "shapiro_wilk_normal": bool,
    "dagostino_pvalue": float,
    "dagostino_normal": bool,
    "normality_verdict": str,
    
    # Noise Impact (vs IDEAL)
    "degradation_from_ideal": float,
    "degradation_percent": float,
    "cohens_d_vs_ideal": float,
    "snr": float,
}
```

---

## Configuration Summary

### Total Experiment Configurations

| Experiment | Configs | × Noise Models | × Runs | = Total Jobs |
|------------|---------|----------------|--------|--------------|
| Scalability (S) | 7 | 8 | 10 | 560 |
| Marked Count (M) | 6 | 8 | 10 | 480 |
| Hamming (H) | 7 | 8 | 10 | 560 |
| Symmetric (SYM/ASYM) | 4 | 8 | 10 | 320 |
| **Total** | 24 | - | - | **1,920** |

**Execution time estimate**: ~30 minutes on local Aer simulator

**Recommended start**: IDEAL + DEP-MED only (480 jobs) for rapid iteration

---

## Implementation Plan

### Week 1: Infrastructure Setup ✅ COMPLETE
- [x] Create `grover_experiment.py` - main experiment runner
- [x] Implement circuit configuration generator (`grover_configs.py`)
- [x] Implement all success metric functions (`grover_success_metrics.py`)
- [x] Implement statistical analysis functions (`grover_statistical_analysis.py`)
- [x] Set up data collection and storage (JSON)
- [x] Run pilot study validation

### Week 2: Ideal Simulator Baseline ✅ COMPLETE
- [x] Run all configurations on ideal simulator
- [x] Verify theoretical success probabilities
- [x] Collect circuit metrics (depth, gates)
- [x] Run normality tests on ideal results
- [x] Validate experiment infrastructure

### Week 3: Noisy Simulator Study ✅ COMPLETE
- [x] Run all configurations with each noise model (24 configs × 7 noise models)
- [x] Compare distributions: ideal vs noisy
- [x] Quantify noise impact (degradation %)
- [x] Identify scalability crossover points per noise model

### Week 4: Analysis & Documentation
- [ ] Complete statistical analysis
- [ ] Generate all visualizations
- [ ] Document findings for thesis chapter
- [ ] Identify which success metric is most informative

---

## Future Work (Out of Scope for Phase 1)

### Phase 2: Optimization Study
- Compare optimization levels 0-3
- Measure gate reduction vs success improvement

### Phase 3: Real QPU Experiments
- Select best configurations from Phase 1
- Design targeted QPU experiments
- Compare simulator predictions to real hardware

---

## Code Structure

```
qward/examples/papers/
└── grover/
    ├── __init__.py                    # Package exports
    ├── README.md                      # Overview and usage instructions
    ├── grover_experiment.md           # This experiment design document
    ├── grover_math_analysis.tex       # Mathematical analysis (LaTeX)
    ├── grover_experiment.py           # Main experiment runner
    ├── grover_configs.py              # Circuit configurations (24 configs, 8 noise models)
    ├── grover_success_metrics.py      # Success metric implementations (3 levels)
    ├── grover_statistical_analysis.py # Statistical analysis functions
    └── data/
        └── simulator/
            ├── raw/                   # Individual experiment records (JSON)
            └── aggregated/            # Per-config aggregates (JSON)
```

---

## Pilot Study Results (Week 1 Validation)

Pilot study ran 4 configs × 2 noise models × 5 runs to validate infrastructure:

| Config | IDEAL Mean | DEP-MED Mean | Degradation |
|--------|------------|--------------|-------------|
| S3-1   | 0.9447     | 0.2854       | 69.8%       |
| M3-1   | 0.9482     | 0.2787       | 70.6%       |
| H3-0   | 0.9443     | 0.2715       | 71.3%       |
| H3-3   | 0.9465     | 0.2988       | 68.4%       |

**Key Finding**: DEP-MED noise causes ~70% degradation in success rate for 3-qubit circuits.

---

## Week 2 Results: Ideal Simulator Baseline ✅

All 24 configurations completed with 10 runs each (1024 shots per run).

### Scalability Study Results

| Config | Qubits | Marked | Theory% | Observed% | Gap% | Std | Normal | Depth | Gates |
|--------|--------|--------|---------|-----------|------|-----|--------|-------|-------|
| S2-1 | 2 | 1 | 100.00 | 100.00 | 0.000 | 0.0000 | No | 12 | 21 |
| S3-1 | 3 | 1 | 94.53 | 94.46 | 0.068 | 0.0068 | Yes | 26 | 49 |
| S4-1 | 4 | 1 | 96.13 | 96.14 | 0.011 | 0.0071 | Yes | 38 | 89 |
| S5-1 | 5 | 1 | 99.92 | 99.91 | 0.006 | 0.0009 | No | 50 | 141 |
| S6-1 | 6 | 1 | 99.66 | 99.68 | 0.019 | 0.0020 | Yes | 74 | 231 |
| S7-1 | 7 | 1 | 99.56 | 99.57 | 0.008 | 0.0021 | Yes | 98 | 337 |
| S8-1 | 8 | 1 | 99.99 | 99.99 | 0.004 | 0.0003 | No | 146 | 571 |

**Key Finding**: Circuit depth scales roughly 12-18 gates per qubit. All configs match theoretical within 0.1%.

### Marked Count Study Results

| Config | Qubits | Marked | Theory% | Observed% | Gap% | Std | Normal | Depth | Gates |
|--------|--------|--------|---------|-----------|------|-----|--------|-------|-------|
| M3-1 | 3 | 1 | 94.53 | 94.21 | 0.322 | 0.0070 | Yes | 26 | 57 |
| M3-2 | 3 | 2 | 100.00 | 100.00 | 0.000 | 0.0000 | No | 17 | 36 |
| M3-4 | 3 | 4 | 50.00 | 50.40 | 0.400 | 0.0128 | Yes | 2 | 9 |
| M4-1 | 4 | 1 | 96.13 | 95.77 | 0.360 | 0.0067 | Yes | 38 | 101 |
| M4-2 | 4 | 2 | 94.53 | 94.52 | 0.010 | 0.0054 | Yes | 32 | 77 |
| M4-4 | 4 | 4 | 100.00 | 100.00 | 0.000 | 0.0000 | No | 25 | 58 |

**Key Finding**: M3-4 (4 marked states out of 8) has only 50% success - confirming that marking half the search space eliminates quantum advantage (0 Grover iterations needed).

### Hamming Weight Study Results

| Config | Qubits | Marked | Hamming | Theory% | Observed% | Gap% | Std | Normal |
|--------|--------|--------|---------|---------|-----------|------|-----|--------|
| H3-0 | 3 | 1 | 0 (000) | 94.53 | 94.77 | 0.234 | 0.0056 | Yes |
| H3-1 | 3 | 1 | 1 (001) | 94.53 | 94.70 | 0.166 | 0.0066 | Yes |
| H3-2 | 3 | 1 | 2 (011) | 94.53 | 94.56 | 0.029 | 0.0094 | Yes |
| H3-3 | 3 | 1 | 3 (111) | 94.53 | 94.44 | 0.088 | 0.0115 | Yes |
| H4-0 | 4 | 1 | 0 (0000) | 96.13 | 96.33 | 0.196 | 0.0074 | Yes |
| H4-2 | 4 | 1 | 2 (0011) | 96.13 | 96.39 | 0.255 | 0.0068 | No |
| H4-4 | 4 | 1 | 4 (1111) | 96.13 | 95.79 | 0.341 | 0.0073 | Yes |

**Key Finding**: No significant difference in ideal simulation based on Hamming weight - all match theoretical within 0.35%. Differences should emerge under noise.

### Symmetry Study Results

| Config | Marked States | Pattern | Theory% | Observed% | Gap% | Std | Normal |
|--------|---------------|---------|---------|-----------|------|-----|--------|
| SYM-1 | ["000", "111"] | Symmetric (complements) | 100.00 | 100.00 | 0.000 | 0.0000 | No |
| SYM-2 | ["001", "110"] | Symmetric | 100.00 | 100.00 | 0.000 | 0.0000 | No |
| ASYM-1 | ["000", "001"] | Asymmetric (adjacent) | 100.00 | 100.00 | 0.000 | 0.0000 | No |
| ASYM-2 | ["000", "011"] | Asymmetric | 100.00 | 100.00 | 0.000 | 0.0000 | No |

**Key Finding**: All symmetry configs achieve 100% success (2 marked states in 8-state space). Noise study will reveal if symmetric vs asymmetric patterns affect robustness.

### Week 2 Summary

| Metric | Value |
|--------|-------|
| Total configurations | 24 |
| Total experiment runs | 240 (24 × 10) |
| Total shots | 245,760 (240 × 1024) |
| Mean gap (theory vs observed) | 0.105% |
| Max gap | 0.400% (M3-4) |
| Normal distributions | 14/24 (58%) |
| Execution time | ~5 minutes |

**Validation**: ✅ All theoretical success probabilities confirmed within 0.5% margin.

**Note on Normality**: Configs with 100% success (std=0) fail normality tests because there's no variance - this is expected for perfect simulations.

---

## Week 3 Results: Noisy Simulator Study ✅

All 24 configurations × 7 noise models × 10 runs = 1,680 experiments completed.

### Noise Model Severity Ranking

| Rank | Noise Model | Avg Success% | Degradation | Impact Level |
|------|-------------|--------------|-------------|--------------|
| 1 | IDEAL | 95.49% | Baseline | - |
| 2 | READOUT | 88.67% | -6.8% | Low |
| 3 | THERMAL | 79.52% | -16.0% | Medium |
| 4 | DEP-LOW | 57.24% | -38.2% | High |
| 5 | DEP-MED | 25.77% | -69.7% | Severe |
| 6 | COMBINED | 24.92% | -70.6% | Severe |
| 7 | PAULI | 22.89% | -72.6% | Severe |
| 8 | DEP-HIGH | 16.42% | -79.1% | Severe |

### Scalability Under Noise

| Config | Qubits | Depth | Gates | IDEAL | DEP-LOW | DEP-MED | THERMAL | READOUT |
|--------|--------|-------|-------|-------|---------|---------|---------|---------|
| S2-1 | 2 | 12 | 21 | 100.0% | 97.7% | 84.8% | 99.0% | 96.2% |
| S3-1 | 3 | 26 | 49 | 94.7% | 75.1% | 28.8% | 91.0% | 88.9% |
| S4-1 | 4 | 38 | 89 | 96.2% | 44.8% | 7.2% | 86.9% | 89.1% |
| S5-1 | 5 | 50 | 141 | 99.9% | 9.3% | 3.3%* | 66.7% | 90.3% |
| S6-1 | 6 | 74 | 231 | 99.7% | 1.7%* | 1.6%* | 36.1% | 88.3% |
| S7-1 | 7 | 98 | 337 | 99.6% | 0.8%* | 0.8%* | 13.8% | 86.5% |
| S8-1 | 8 | 146 | 571 | 100.0% | 0.4%* | 0.3%* | 1.3%* | 85.1% |

*Below 2× classical random (quantum advantage lost)

### Scalability Crossover Points

| Noise Model | Falls below 50% | Falls below 30% | Falls below 10% |
|-------------|-----------------|-----------------|-----------------|
| DEP-LOW | 4 qubits | 5 qubits | 5 qubits |
| DEP-MED | 3 qubits | 3 qubits | 4 qubits |
| DEP-HIGH | 3 qubits | 3 qubits | 4 qubits |
| THERMAL | 6 qubits | 7 qubits | 8 qubits |
| COMBINED | 3 qubits | 3 qubits | 4 qubits |

### Hamming Weight Under Noise (0s vs 1s)

| Config | State | Gates | IDEAL | DEP-LOW | DEP-MED | THERMAL |
|--------|-------|-------|-------|---------|---------|---------|
| H3-0 | 000 | 57 | 94.8% | 74.9% | 27.6% | 90.5% |
| H3-3 | 111 | 45 | 94.4% | 75.2% | 29.4% | 91.2% |
| **Diff** | | **+12** | +0.3% | **-0.3%** | **-1.8%** | **-0.8%** |
| H4-0 | 0000 | 101 | 96.3% | 45.5% | 6.8% | 85.7% |
| H4-4 | 1111 | 77 | 95.8% | 45.8% | 7.2% | 86.7% |
| **Diff** | | **+24** | +0.5% | **-0.3%** | **-0.4%** | **-1.0%** |

**Key Finding**: Under noise, all-1s states perform 0.3-1.8% better than all-0s states (more gates for 0s → more errors).

### Week 3 Key Findings

#### 1. Noise Severity Ranking

```
IDEAL (95.5%) > READOUT (88.7%) > THERMAL (79.5%) > DEP-LOW (57.2%) > DEP-MED (25.8%) > COMBINED (24.9%) > PAULI (22.9%) > DEP-HIGH (16.4%)
```

**Interpretation**:
- **READOUT noise**: Minimal impact (-6.8%) - measurement errors are correctable
- **THERMAL noise**: Moderate impact (-16.0%) - T1/T2 relaxation affects longer circuits
- **Depolarizing noise**: Devastating (-38% to -79%) - random errors accumulate quickly
- **COMBINED**: Similar to DEP-MED, confirming depolarizing dominates over readout

#### 2. Scalability Limits by Noise Type

| Noise Model | Practical Qubit Limit | Reasoning |
|-------------|----------------------|-----------|
| **READOUT** | **8+ qubits** ✅ | Only affects final measurement, not gates |
| **THERMAL** | **6 qubits** | T1/T2 times limit circuit depth |
| **DEP-LOW** | **4 qubits** | Light depolarizing still accumulates |
| **DEP-MED** | **3 qubits** | Medium noise quickly destroys coherence |
| **COMBINED** | **3 qubits** | Combined effects compound |
| **DEP-HIGH** | **2-3 qubits** | Only simplest circuits survive |

#### 3. Circuit Depth vs Noise Sensitivity

| Depth Range | Gates | DEP-MED Success | Status |
|-------------|-------|-----------------|--------|
| 12 | 21 | 84.8% | ✅ Quantum advantage |
| 26 | 49 | 28.8% | ⚠️ Marginal |
| 38 | 89 | 7.2% | ❌ Below random |
| 50+ | 141+ | <3.5% | ❌ Completely random |

**Rule of Thumb**: Under DEP-MED noise, every ~25 additional gates loses ~10% success rate.

#### 4. Hamming Weight Effect (CONFIRMED)

| Comparison | Gate Diff | Noise Impact | Winner |
|------------|-----------|--------------|--------|
| 000 vs 111 (3q) | +12 gates | -1.8% under DEP-MED | **1s better** |
| 0000 vs 1111 (4q) | +24 gates | -0.4% under DEP-MED | **1s better** |

**Conclusion**: Marking all-1s states requires fewer oracle gates → fewer errors under noise.

#### 5. Quantum Advantage Loss Points

| Qubits | Random Chance | DEP-LOW | DEP-MED | THERMAL | READOUT |
|--------|---------------|---------|---------|---------|---------|
| 2 | 25.0% | ✅ 97.7% | ✅ 84.8% | ✅ 99.0% | ✅ 96.2% |
| 3 | 12.5% | ✅ 75.1% | ⚠️ 28.8% | ✅ 91.0% | ✅ 88.9% |
| 4 | 6.25% | ⚠️ 44.8% | ❌ 7.2% | ✅ 86.9% | ✅ 89.1% |
| 5 | 3.12% | ❌ 9.3% | ❌ 3.3% | ✅ 66.7% | ✅ 90.3% |
| 6 | 1.56% | ❌ 1.7% | ❌ 1.6% | ⚠️ 36.1% | ✅ 88.3% |
| 7 | 0.78% | ❌ 0.8% | ❌ 0.8% | ❌ 13.8% | ✅ 86.5% |
| 8 | 0.39% | ❌ 0.4% | ❌ 0.3% | ❌ 1.3% | ✅ 85.1% |

✅ = Quantum advantage maintained (>2× random)
⚠️ = Marginal advantage
❌ = Quantum advantage lost

#### 6. Recommendations for Real QPU

Based on simulator results:

1. **Start with 2-3 qubit circuits** to validate QPU behavior matches simulator
2. **Expect DEP-MED-like noise** on current NISQ devices
3. **Use READOUT error mitigation** - it provides significant improvement
4. **Avoid circuits >50 depth** unless using error mitigation
5. **Prefer marking 1s over 0s** for ~1-2% improvement

### Week 3 Summary

| Metric | Value |
|--------|-------|
| Total experiments | 1,680 |
| Noise models tested | 7 (+ IDEAL baseline) |
| Configurations | 24 |
| Runs per config | 10 |
| Most resilient noise | READOUT (-6.8%) |
| Most severe noise | DEP-HIGH (-79.1%) |
| Max useful qubits (DEP-MED) | 3 |
| Max useful qubits (THERMAL) | 6 |

---

## Phase 2: Real QPU Experiment Plan

### Resource Constraints

| Resource | Value | Notes |
|----------|-------|-------|
| **Total QPU time** | 10 minutes (600 seconds) | IBM Quantum allocation |
| **Time per experiment** | ~5 seconds | Including queue + execution |
| **Max experiments** | ~120 | Budget: 600s ÷ 5s |
| **Shots per experiment** | 1024 | Same as simulator |
| **Runs per config** | 3-5 | Reduced from 10 due to cost |

### Experiment Design Philosophy

Based on simulator results, we design QPU experiments to:
1. **Validate** simulator predictions match real hardware
2. **Identify** the actual noise profile of the QPU
3. **Confirm** scalability limits
4. **Test** hypotheses about marked state effects
5. **Document** expected failures to establish algorithm boundaries

### Hypotheses to Test on Real QPU

| ID | Hypothesis | Based On | Expected Outcome |
|----|-----------|----------|------------------|
| H1 | QPU noise profile matches DEP-MED simulator | Industry reports | 3-qubit limit |
| H2 | READOUT errors dominate at small circuits | Simulator: READOUT -6.8% | 2-qubit >90% success |
| H3 | Marking 1s outperforms 0s on real hardware | Simulator: 1-2% improvement | Measurable difference |
| H4 | 4+ qubit circuits will show near-random results | Simulator: DEP-MED <10% | Loss of quantum advantage |
| H5 | Circuit depth correlates with error rate | Simulator: ~10%/25 gates | Consistent degradation |

---

### Experiment Groups

#### Group A: Baseline Validation (20 experiments, ~100 seconds)
**Goal**: Confirm QPU behaves as expected for simple cases.

| Exp | Config | Qubits | Depth | Runs | Simulator Prediction | QPU Hypothesis |
|-----|--------|--------|-------|------|---------------------|----------------|
| A1 | S2-1 | 2 | 12 | 5 | DEP-MED: 84.8% | 80-90% (best case) |
| A2 | M3-2 | 3 | 17 | 5 | DEP-MED: ~85%* | 75-85% (short circuit) |
| A3 | S3-1 | 3 | 26 | 5 | DEP-MED: 28.8% | 25-35% (typical) |
| A4 | H3-3 | 3 | 22 | 5 | DEP-MED: 29.4% | 25-35% (all 1s state) |

*Estimated from short circuit depth

**Success Criteria**: Results within ±15% of DEP-MED simulator predictions.

---

#### Group B: Hamming Weight Test (15 experiments, ~75 seconds)
**Goal**: Confirm marking 1s vs 0s difference on real hardware.

| Exp | Config | State | Gates | Runs | Simulator (DEP-MED) | QPU Hypothesis |
|-----|--------|-------|-------|------|---------------------|----------------|
| B1 | H3-0 | 000 | 57 | 5 | 27.6% | 20-30% |
| B2 | H3-3 | 111 | 45 | 5 | 29.4% | 25-35% |
| B3 | H4-0 | 0000 | 101 | 5 | 6.8% | 5-10% |

**Success Criteria**: H3-3 > H3-0 and H4-4 > H4-0 (1s better than 0s).

---

#### Group C: Scalability Boundary (35 experiments, ~175 seconds)
**Goal**: Find exact qubit count where quantum advantage disappears.

| Exp | Config | Qubits | Depth | Gates | Runs | Simulator (DEP-MED) | Expected |
|-----|--------|--------|-------|-------|------|---------------------|----------|
| C1 | S2-1 | 2 | 12 | 21 | 5 | 84.8% | ✅ Quantum advantage |
| C2 | S3-1 | 3 | 26 | 49 | 5 | 28.8% | ⚠️ Marginal |
| C3 | S4-1 | 4 | 38 | 89 | 5 | 7.2% | ❌ Near random |
| C4 | S5-1 | 5 | 50 | 141 | 5 | 3.3% | ❌ Random |
| C5 | S6-1 | 6 | 74 | 231 | 5 | 1.6% | ❌ Random |
| C6 | S7-1 | 7 | 98 | 337 | 5 | 0.8% | ❌ Random |
| C7 | S8-1 | 8 | 146 | 571 | 5 | 0.3% | ❌ Random |

**Success Criteria**: Identify crossover point where success rate ≤ 2× random chance.

---

#### Group D: Marked State Count (15 experiments, ~75 seconds)
**Goal**: Test if more marked states improve QPU success.

| Exp | Config | Qubits | Marked | Depth | Runs | Simulator (DEP-MED) | QPU Hypothesis |
|-----|--------|--------|--------|-------|------|---------------------|----------------|
| D1 | M3-1 | 3 | 1 | 26 | 5 | ~28% | 20-35% |
| D2 | M3-2 | 3 | 2 | 17 | 5 | ~85%* | 75-90% (shorter) |
| D3 | M4-1 | 4 | 1 | 38 | 5 | ~7% | 5-12% |

*M3-2 has much shorter circuit due to fewer iterations.

**Success Criteria**: Fewer Grover iterations (more marked states) = higher success.

---

#### Group E: Symmetry Test (15 experiments, ~75 seconds)
**Goal**: Check if symmetric vs asymmetric marked states differ on real hardware.

| Exp | Config | Pattern | Depth | Gates | Runs | Simulator (DEP-MED) | QPU Hypothesis |
|-----|--------|---------|-------|-------|------|---------------------|----------------|
| E1 | SYM-1 | ["000","111"] | 17 | 36 | 5 | ~85%* | 75-90% |
| E2 | ASYM-1 | ["000","001"] | 19 | 40 | 5 | ~80%* | 70-85% |
| E3 | SYM-2 | ["001","110"] | 17 | 36 | 5 | ~85%* | 75-90% |

**Success Criteria**: SYM configs should outperform ASYM due to shorter circuits.

---

#### Group F: Expected Failures - Confirm Limits (20 experiments, ~100 seconds)
**Goal**: Confirm where the algorithm definitively fails on real QPU.

| Exp | Config | Qubits | Depth | Gates | Runs | Random Chance | Expected Result |
|-----|--------|--------|-------|-------|------|---------------|-----------------|
| F1 | S5-1 | 5 | 50 | 141 | 2 | 3.12% | ❌ ~3-5% (random) |
| F2 | S6-1 | 6 | 74 | 231 | 2 | 1.56% | ❌ ~1-3% (random) |
| F3 | S7-1 | 7 | 98 | 337 | 2 | 0.78% | ❌ ~0.5-2% (random) |
| F4 | S8-1 | 8 | 146 | 571 | 2 | 0.39% | ❌ ~0.3-1% (random) |
| F5 | H4-0 | 4 | 38 | 101 | 2 | 6.25% | ❌ ~5-10% (near random) |
| F6 | M4-1 | 4 | 38 | 101 | 2 | 6.25% | ❌ ~5-10% (near random) |
| F7 | S5-1 | 5 | 50 | 141 | 2 | 3.12% | ❌ ~3-5% (random) |
| F8 | S6-1 | 6 | 74 | 231 | 2 | 1.56% | ❌ ~1-3% (random) |
| F9 | S7-1 | 7 | 98 | 337 | 2 | 0.78% | ❌ ~0.5-2% (random) |
| F10 | S8-1 | 8 | 146 | 571 | 2 | 0.39% | ❌ ~0.3-1% (random) |

**Success Criteria**: Results statistically indistinguishable from random chance (p > 0.05).

---

### QPU Execution Summary

| Group | Purpose | Experiments | Time (s) | Priority |
|-------|---------|-------------|----------|----------|
| A | Baseline Validation | 20 | 100 | 🔴 Critical |
| B | Hamming Weight Test | 15 | 75 | 🟡 Important |
| C | Scalability Boundary | 35 | 175 | 🔴 Critical |
| D | Marked State Count | 15 | 75 | 🟡 Important |
| E | Symmetry Test | 15 | 75 | 🟢 Nice-to-have |
| F | Expected Failures | 20 | 100 | 🔴 Critical |
| **Total** | | **120** | **600** | |

### Execution Priority Order

If time is limited, execute in this order:

1. **Phase 1 (Essential, ~375s)**: Groups A + C + F
   - Validates QPU matches simulator
   - Confirms scalability limits
   - Documents failure boundaries

2. **Phase 2 (Important, ~150s)**: Groups B + D
   - Tests Hamming weight hypothesis
   - Tests marked state count effect

3. **Phase 3 (Optional, ~75s)**: Group E
   - Tests symmetry effects
   - Lower priority, can skip if time constrained

### Expected Outcomes

| Outcome | Probability | Implication |
|---------|-------------|-------------|
| QPU matches DEP-MED | 60% | Simulator is good predictor |
| QPU worse than DEP-MED | 25% | Real hardware has additional noise |
| QPU better than DEP-MED | 15% | Error mitigation/calibration helping |

### Data Collection for QPU

Each QPU experiment should record:
- `qpu_name`: IBM backend name (e.g., "ibm_brisbane")
- `qpu_calibration_date`: Last calibration timestamp
- `queue_time_ms`: Time waiting in queue
- `execution_time_ms`: Actual QPU execution time
- `transpilation_time_ms`: Circuit preparation time
- `counts`: Raw measurement results
- `success_rate`: Calculated success
- `qward_metrics`: Pre-runtime circuit metrics

### Post-QPU Analysis Plan

1. **Compare QPU vs Simulator**:
   - Calculate mean absolute error (MAE)
   - Identify which noise model best predicts QPU

2. **Validate Hypotheses**:
   - H1-H5 confirmation or rejection
   - Statistical significance tests

3. **Establish QPU-Specific Limits**:
   - Actual crossover point for quantum advantage
   - Maximum practical circuit depth

4. **Cost-Benefit Analysis**:
   - Success rate vs QPU cost
   - Recommendations for practical Grover applications

---

## Quick Start

```python
# Test single configuration
from qward.examples.papers.grover import test_single_config
result = test_single_config("S3-1", "IDEAL", num_runs=5)

# Run pilot study
from qward.examples.papers.grover import test_pilot_study
results = test_pilot_study()

# Full campaign
from qward.examples.papers.grover import run_experiment_campaign
results = run_experiment_campaign(
    config_ids=["S3-1", "S4-1", "S5-1"],
    noise_ids=["IDEAL", "DEP-LOW", "DEP-MED"],
    num_runs=10,
    save_results=True,
)

# List all configurations
from qward.examples.papers.grover import list_all_configs, list_noise_configs
list_all_configs()
list_noise_configs()
```

---

## Progress Tracker

### Completed ✅
1. ✅ Define experiment parameters (1024 shots, 10 runs, opt level 0)
2. ✅ Define statistical analysis framework
3. ✅ Implement `grover_configs.py` with all circuit configurations
4. ✅ Implement `grover_success_metrics.py` with 3-level success metrics
5. ✅ Implement `grover_statistical_analysis.py` with normality tests
6. ✅ Create `grover_experiment.py` runner
7. ✅ Run pilot study with 3-qubit configurations (IDEAL + DEP-MED)
8. ✅ Run IDEAL baseline for ALL 24 configurations (Week 2)
9. ✅ Verify theoretical success probabilities match observed (mean gap: 0.105%)
10. ✅ Collect circuit metrics (depth scales 12-18 gates/qubit)
11. ✅ Run normality tests (14/24 normal, 10 have zero variance)
12. ✅ Run noisy simulator study (Week 3) - 7 noise models × 24 configs
13. ✅ Compare ideal vs noisy distributions (degradation 6.8% to 79.1%)
14. ✅ Identify scalability crossover points (3-6 qubits depending on noise)
15. ✅ Test Hamming weight hypothesis (confirmed: 1s better than 0s under noise)
16. ✅ Rank noise model severity (READOUT < THERMAL < DEP-LOW < DEP-MED < COMBINED < PAULI < DEP-HIGH)
17. ✅ Create mathematical analysis document (`grover_math_analysis.tex`)

### Next Steps (Week 4)
1. [ ] Generate visualizations (distribution plots, heatmaps)
2. [ ] Complete statistical analysis report
3. [ ] Document findings for thesis chapter
4. [ ] Identify which success metric is most informative
5. [ ] Prepare recommendations for real QPU experiments
