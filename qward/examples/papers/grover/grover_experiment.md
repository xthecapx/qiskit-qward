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

| Config ID | Qubits | Marked States | Search Space | Theoretical Success | Expected Iterations |
|-----------|--------|---------------|--------------|---------------------|---------------------|
| S2-1 | 2 | ["01"] | 4 | 1.00 | 1 |
| S3-1 | 3 | ["011"] | 8 | 1.00 | 2 |
| S4-1 | 4 | ["0110"] | 16 | 0.96 | 3 |
| S5-1 | 5 | ["01100"] | 32 | 0.99 | 4 |
| S6-1 | 6 | ["011001"] | 64 | 0.98 | 6 |
| S7-1 | 7 | ["0110011"] | 128 | 0.99 | 8 |
| S8-1 | 8 | ["01100110"] | 256 | 0.99 | 12 |

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

| Config ID | Qubits | Num Marked | Marked States | Theoretical Success |
|-----------|--------|------------|---------------|---------------------|
| M3-1 | 3 | 1 | ["000"] | 1.00 |
| M3-2 | 3 | 2 | ["000", "111"] | 1.00 |
| M3-4 | 3 | 4 | ["000", "001", "110", "111"] | 1.00 |
| M4-1 | 4 | 1 | ["0000"] | 0.96 |
| M4-2 | 4 | 2 | ["0000", "1111"] | 1.00 |
| M4-4 | 4 | 4 | ["0000", "0011", "1100", "1111"] | 1.00 |

**Hypothesis**: More marked states → fewer Grover iterations → shorter circuit → better success rate under noise

### 2B: Hamming Weight Study

Testing whether the "type" of marked state affects performance under noise.

| Config ID | Qubits | Marked State | Hamming Weight | Description |
|-----------|--------|--------------|----------------|-------------|
| H3-0 | 3 | ["000"] | 0 | All zeros |
| H3-1 | 3 | ["001"] | 1 | Single 1 |
| H3-2 | 3 | ["011"] | 2 | Two 1s |
| H3-3 | 3 | ["111"] | 3 | All ones |
| H4-0 | 4 | ["0000"] | 0 | All zeros |
| H4-2 | 4 | ["0011"] | 2 | Balanced |
| H4-4 | 4 | ["1111"] | 4 | All ones |

**Hypothesis**: 
- All-zeros states may have different error profiles than all-ones under certain noise models
- Balanced Hamming weight states may be more resilient to bit-flip errors

### 2C: Symmetric vs Asymmetric Marked States

| Config ID | Marked States | Pattern | Notes |
|-----------|---------------|---------|-------|
| SYM-1 | ["000", "111"] | Symmetric (complement pairs) | Both extremes |
| SYM-2 | ["001", "110"] | Symmetric | Single bit flip from extremes |
| ASYM-1 | ["000", "001"] | Asymmetric | Adjacent states (1-bit difference) |
| ASYM-2 | ["000", "011"] | Asymmetric | Two-bit difference |

**Hypothesis**: Symmetric marked states may show different interference patterns and error resilience

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

### Week 2: Ideal Simulator Baseline
- [ ] Run all configurations on ideal simulator
- [ ] Verify theoretical success probabilities
- [ ] Collect circuit metrics (depth, gates)
- [ ] Run normality tests on ideal results
- [ ] Validate experiment infrastructure

### Week 3: Noisy Simulator Study
- [ ] Run all configurations with each noise model
- [ ] Compare distributions: ideal vs noisy
- [ ] Quantify noise impact (Cohen's d, degradation %)
- [ ] Identify scalability crossover points per noise model

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

### Next Steps
1. [ ] Run full ideal simulator baseline (Week 2)
2. [ ] Run noisy simulator study (Week 3)
3. [ ] Complete analysis and documentation (Week 4)
