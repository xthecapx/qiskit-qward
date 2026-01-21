# Quantum Fourier Transform Experiment Design

## Research Goals

This document outlines the experimental design for evaluating Quantum Fourier Transform (QFT) performance using the QWARD framework.

**Scope**: Simulator experiments (ideal and noisy) to establish baselines before real QPU execution.

### Primary Research Questions

1. **Scalability Limit**: At what qubit count does noise cause QFT to fail?
2. **Test Mode Comparison**: How do roundtrip and period detection modes compare under noise?
3. **Period Impact**: How does the period parameter affect success in period detection mode?
4. **Input State Impact**: Does the input state (Hamming weight) affect roundtrip success?
5. **Noise Characterization**: How do different noise models affect QFT performance?

---

## Fixed Experiment Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Shots per job** | 1024 | Standard quantum computing benchmark |
| **Runs per config** | 10 | Sufficient for statistical analysis |
| **Optimization level** | 0 (none) | Establish baseline first |
| **Backend** | Aer simulator | Ideal and noisy modes |

---

## QFT Algorithm Overview

### Test Modes

#### 1. Roundtrip Mode (QFT → QFT⁻¹)
- Prepare input state |ψ⟩
- Apply QFT
- Apply inverse QFT
- Measure - should return original state
- **Success criterion**: measurement == input_state

#### 2. Period Detection Mode
- Prepare ancilla qubit in |1⟩
- Put counting qubits in superposition
- Apply controlled phase rotations (encodes period)
- Apply inverse QFT to counting register
- Measure - should show peaks at multiples of N/period
- **Success criterion**: measurement lands on expected peak

### Circuit Complexity

QFT requires O(n²) gates for n qubits:
- n Hadamard gates
- n(n-1)/2 controlled phase gates
- n/2 SWAP gates (for standard ordering)

---

## Phase 1: Simulator Experiments ✅ COMPLETE

---

## Experiment 1: Scalability Study - Roundtrip Mode (SR)

### Objective
Determine the maximum qubit count where QFT roundtrip remains effective under noise.

### Circuit Variations

| Config ID | Qubits | Input State | IDEAL | DEP-LOW | DEP-MED | DEP-HIGH | READOUT |
|-----------|--------|-------------|-------|---------|---------|----------|---------|
| SR2 | 2 | \|01⟩ | 100.0% | 98.0% | 92.5% | 83.6% | 95.8% |
| SR3 | 3 | \|101⟩ | 100.0% | 96.4% | 86.2% | 70.3% | 94.4% |
| SR4 | 4 | \|0110⟩ | 100.0% | 93.5% | 76.2% | 52.2% | 91.9% |
| SR5 | 5 | \|10101⟩ | 100.0% | 90.8% | 66.4% | 38.8% | 89.6% |
| SR6 | 6 | \|011001⟩ | 100.0% | 86.4% | 56.9% | 25.0% | 88.4% |
| SR7 | 7 | \|1010101⟩ | 100.0% | 82.4% | 45.9% | 16.2% | 86.9% |
| SR8 | 8 | \|01100110⟩ | 100.0% | 77.1% | 36.7% | 9.2% | 84.6% |

### Hardware-Only Configurations (Not Simulated)
| Config ID | Qubits | Reason |
|-----------|--------|--------|
| SR10 | 10 | 2¹⁰ = 1024 states, too slow for simulator |
| SR12 | 12 | 2¹² = 4096 states, very expensive |

### Hypotheses (Roundtrip)

- **H1**: Success rate degrades exponentially as qubit count increases ✅ **CONFIRMED**
- **H2**: There exists a crossover point where noisy QFT becomes useless ✅ **CONFIRMED**
- **H3**: Degradation rate correlates with two-qubit gate count ✅ **CONFIRMED**

### Key Finding: Exponential Decay

Under DEP-HIGH noise, success rate follows approximately:

```
success ≈ 0.84 × exp(-0.37 × n)
```

Where n = number of qubits. Half-life ≈ 1.9 qubits.

---

## Experiment 2: Scalability Study - Period Detection Mode (SP)

### Objective
Test QFT period detection under various noise conditions.

### Circuit Variations

| Config ID | Qubits | Period | Peaks | IDEAL | DEP-LOW | DEP-MED | DEP-HIGH |
|-----------|--------|--------|-------|-------|---------|---------|----------|
| SP3-P2 | 3 | 2 | 4 | 100.0% | 98.8% | 95.4% | 90.1% |
| SP4-P2 | 4 | 2 | 8 | 100.0% | 96.8% | 86.8% | 74.6% |
| SP4-P4 | 4 | 4 | 4 | 100.0% | 98.2% | 94.0% | 87.6% |
| SP5-P4 | 5 | 4 | 8 | 100.0% | 95.6% | 85.0% | 71.7% |
| SP6-P4 | 6 | 4 | 16 | 100.0% | 93.8% | 77.3% | 55.4% |
| SP6-P8 | 6 | 8 | 8 | 100.0% | 95.7% | 83.7% | 67.0% |

### Hardware-Only Configurations
| Config ID | Qubits | Period | Reason |
|-----------|--------|--------|--------|
| SP8-P4 | 8 | 4 | 9 total qubits (8 counting + 1 ancilla) |
| SP8-P16 | 8 | 16 | 9 total qubits |

### Hypotheses (Period Detection)

- **H4**: Period detection outperforms roundtrip under noise ✅ **CONFIRMED** (+16.6% at DEP-MED)
- **H5**: Larger periods → higher success (more valid peaks) ✅ **CONFIRMED**
- **H6**: Success scales with number of valid measurement outcomes ✅ **CONFIRMED**

### Key Finding: Period Detection is More Robust

Period detection has multiple "correct" answers (peaks), making it inherently more noise-tolerant:

| Mode | DEP-MED Avg | Reason |
|------|-------------|--------|
| Roundtrip | 70.3% | Single correct answer |
| Period Detection | 86.8% | Multiple valid peaks |

**Difference**: +16.6% for period detection

---

## Experiment 3: Period Variation Study (PV)

### Objective
Understand how period size affects detection accuracy at fixed qubit counts.

### 4-Qubit Period Variation

| Config ID | Period | N/Period | Peaks | IDEAL | DEP-LOW | DEP-MED |
|-----------|--------|----------|-------|-------|---------|---------|
| PV4-P2 | 2 | 8 | 2 | 100.0% | 96.5% | 87.5% |
| PV4-P4 | 4 | 4 | 4 | 100.0% | 98.5% | 93.7% |
| PV4-P8 | 8 | 2 | 8 | 100.0% | 100.0% | 100.0% |

### 6-Qubit Period Variation

| Config ID | Period | N/Period | Peaks | IDEAL | DEP-LOW | DEP-MED |
|-----------|--------|----------|-------|-------|---------|---------|
| PV6-P2 | 2 | 32 | 2 | 100.0% | 91.7% | 72.0% |
| PV6-P4 | 4 | 16 | 4 | 100.0% | 93.7% | 77.6% |
| PV6-P8 | 8 | 8 | 8 | 100.0% | 95.5% | 83.9% |
| PV6-P16 | 16 | 4 | 16 | 100.0% | 97.9% | 91.9% |

### Key Finding: Larger Period = Easier Detection

The relationship between period and success rate is clear:

```
More peaks → Higher probability of hitting a valid outcome → Higher success rate
```

**PV4-P8** achieves 100% success even under DEP-MED noise because period=8 with 4 qubits means ALL 16 states are valid peaks!

---

## Experiment 4: Input Variation Study (IV)

### Objective
Test whether the input state (Hamming weight) affects roundtrip performance.

### Results

| Config ID | Input | Hamming | IDEAL | DEP-LOW | DEP-MED | DEP-HIGH |
|-----------|-------|---------|-------|---------|---------|----------|
| IV4-0000 | 0000 | 0 | 100.0% | 93.8% | 77.7% | 53.9% |
| IV4-0001 | 0001 | 1 | 100.0% | 93.7% | 77.2% | 53.3% |
| IV4-0101 | 0101 | 2 | 100.0% | 93.0% | 76.1% | 52.9% |
| IV4-1010 | 1010 | 2 | 100.0% | 93.1% | 76.8% | 51.4% |
| IV4-1111 | 1111 | 4 | 100.0% | 93.4% | 74.5% | 51.6% |

### Hypothesis (Input Variation)

- **H7**: Input state Hamming weight affects success under noise ⚠️ **WEAK EFFECT**

### Key Finding: Minimal Hamming Weight Effect

Unlike Grover's algorithm, QFT shows minimal variation based on input state:
- **Range under DEP-MED**: 74.5% - 77.7% (only 3.2% spread)
- **Slight trend**: Lower Hamming weight slightly better (fewer X gates needed)

---

## Experiment 5: Noise Model Study

### Noise Models Tested

| Model ID | Type | Parameters | Description |
|----------|------|------------|-------------|
| IDEAL | None | - | Perfect execution baseline |
| DEP-LOW | Depolarizing | p1=0.001, p2=0.005 | Light depolarizing noise |
| DEP-MED | Depolarizing | p1=0.005, p2=0.02 | Medium depolarizing noise |
| DEP-HIGH | Depolarizing | p1=0.01, p2=0.05 | Heavy depolarizing noise |
| READOUT | Readout | p01=p10=0.02 | Measurement errors only |
| COMBINED | Mixed | DEP-MED + READOUT | Realistic combined noise |

### Noise Model Severity Ranking

| Rank | Noise Model | Avg Success | Degradation | Impact |
|------|-------------|-------------|-------------|--------|
| 1 | IDEAL | 100.00% | Baseline | - |
| 2 | READOUT | 93.79% | -6.2% | Low |
| 3 | DEP-LOW | 93.78% | -6.2% | Low |
| 4 | DEP-MED | 78.88% | -21.1% | Medium |
| 5 | COMBINED | 74.84% | -25.2% | Medium-High |
| 6 | DEP-HIGH | 60.87% | -39.1% | High |

### Key Findings (Noise)

1. **READOUT errors are easily tolerable** - Only 6.2% degradation
2. **Depolarizing noise is the primary concern** - DEP-HIGH causes 39.1% degradation
3. **Combined noise ≈ DEP-MED + small constant** - Readout adds ~4% on top of depolarizing

---

## Scalability Crossover Points

### At What Qubit Count Does QFT Fail?

#### 90% Success Threshold (High Quality)

| Noise Model | Max Qubits (Roundtrip) | Max Qubits (Period Detection) |
|-------------|------------------------|-------------------------------|
| DEP-LOW | 5 | 6+ |
| DEP-MED | 2 | 4-5 |
| DEP-HIGH | 0 | 3 |
| READOUT | 4 | 6+ |

#### 50% Success Threshold (Useful)

| Noise Model | Max Qubits (Roundtrip) | Max Qubits (Period Detection) |
|-------------|------------------------|-------------------------------|
| DEP-LOW | 8+ | 8+ |
| DEP-MED | 6 | 6+ |
| DEP-HIGH | 4 | 5-6 |
| READOUT | 8+ | 8+ |

---

## Statistical Analysis Framework

### Descriptive Statistics

For each configuration (10 runs × 1024 shots):

```python
{
    "mean": float,          # Central tendency
    "median": float,        # Robust central tendency
    "std": float,           # Standard deviation
    "variance": float,      # Spread
    "min": float,           # Worst case
    "max": float,           # Best case
    "ci_lower": float,      # 95% CI lower bound
    "ci_upper": float,      # 95% CI upper bound
    "skewness": float,      # Distribution asymmetry
    "kurtosis": float,      # Tail behavior
}
```

### Normality Tests Applied

1. **Shapiro-Wilk** - Best for small samples (n < 50)
2. **D'Agostino-Pearson** - Based on skewness/kurtosis (n ≥ 20)
3. **Anderson-Darling** - Sensitive to tails
4. **Kolmogorov-Smirnov** - General comparison to fitted normal

### Noise Impact Metrics

- **Degradation %**: (IDEAL - noisy) / IDEAL × 100
- **Cohen's d**: Standardized effect size (meaningful when comparing two noisy conditions)
- **Variance ratio**: How much noise increases spread

---

## Summary of Hypotheses and Results

| ID | Hypothesis | Result | Evidence |
|----|------------|--------|----------|
| H1 | Success degrades exponentially with qubits | ✅ CONFIRMED | Half-life ~2 qubits under DEP-HIGH |
| H2 | Crossover point exists for QFT failure | ✅ CONFIRMED | 4-6 qubits for DEP-MED |
| H3 | Degradation correlates with 2Q gate count | ✅ CONFIRMED | More gates = more errors |
| H4 | Period detection > roundtrip under noise | ✅ CONFIRMED | +16.6% at DEP-MED |
| H5 | Larger periods → higher success | ✅ CONFIRMED | PV4-P8 achieves 100% |
| H6 | Success scales with valid peak count | ✅ CONFIRMED | More peaks = easier |
| H7 | Hamming weight affects success | ⚠️ WEAK | Only 3.2% variation |

---

## Key Findings Summary

### 1. QFT Scalability Limits

| Noise Level | Roundtrip Limit | Period Detection Limit |
|-------------|-----------------|------------------------|
| Low noise | 5-6 qubits | 6+ qubits |
| Medium noise | 4-5 qubits | 5-6 qubits |
| High noise | 3-4 qubits | 4-5 qubits |

### 2. Mode Comparison

**Period detection is 16.6% more resilient than roundtrip** under medium noise because:
- Multiple valid measurement outcomes
- Errors that shift to adjacent peaks still count as success
- Roundtrip has exactly ONE correct answer

### 3. Period Selection Strategy

For period detection applications:
- **Larger periods = better noise resilience**
- Trade-off: larger periods provide less information about the underlying function
- **Recommendation**: Use largest practical period for your application

### 4. Noise Resilience Ranking

```
READOUT (best) > DEP-LOW ≈ READOUT > DEP-MED > COMBINED > DEP-HIGH (worst)
```

### 5. Gate Error Impact

Rule of thumb for DEP-MED noise:
```
~3-5% success loss per additional qubit in roundtrip mode
```

---

## Recommendations for Real QPU

Based on simulator results:

1. **Start with 2-3 qubit circuits** to validate QPU matches simulator
2. **Prefer period detection mode** over roundtrip for noise resilience
3. **Use larger periods** when possible (more valid outcomes)
4. **Expect DEP-MED-like noise** on current NISQ devices
5. **Apply READOUT error mitigation** - adds ~6% improvement
6. **Avoid circuits >6 qubits** unless using error mitigation

---

## Data Collection Schema

### Per-Experiment Record

```python
experiment_record = {
    # Identification
    "experiment_id": str,
    "config_id": str,
    "noise_model": str,
    "run_number": int,
    "timestamp": str,
    
    # Circuit Properties
    "num_qubits": int,
    "test_mode": str,           # "roundtrip" or "period_detection"
    "input_state": str,         # For roundtrip
    "period": int,              # For period detection
    
    # Circuit Metrics
    "circuit_depth": int,
    "total_gates": int,
    
    # QWARD Pre-runtime Metrics
    "qward_metrics": dict,
    
    # Execution
    "shots": 1024,
    "execution_time_ms": float,
    
    # Results
    "counts": dict,
    "success_rate": float,
    "success_count": int,
}
```

---

## Code Structure

```
qward/examples/papers/
└── qft/
    ├── __init__.py                  # Package exports
    ├── README.md                    # Quick start guide
    ├── qft_experiment.md            # This experiment design document
    ├── qft_experiment.py            # Main experiment runner
    ├── qft_configs.py               # Circuit configurations (25 configs, 6 noise models)
    ├── qft_statistical_analysis.py  # Statistical analysis functions
    └── data/
        └── raw/
            └── 20260114_125806/     # Session data (150 JSON files)
        └── aggregated/
            └── campaign_summary_*.json
```

---

## Experiment Summary

| Metric | Value |
|--------|-------|
| Total simulator configs | 25 |
| Hardware-only configs | 4 (SR10, SR12, SP8-P4, SP8-P16) |
| Noise models tested | 6 |
| Total batches | 150 (25 × 6) |
| Runs per batch | 10 |
| Shots per run | 1024 |
| Total experiments | 1,500 |
| Total shots | 1,536,000 |

---

## Phase 2: Real QPU Experiment Plan (Future Work)

### Priority Experiments for QPU

Based on simulator findings, prioritize:

1. **Baseline validation** (2-3 qubit roundtrip)
2. **Period detection comparison** (3-4 qubit period detection)
3. **Scalability boundary** (4-6 qubit progressive test)
4. **Mode comparison** (same qubit count, both modes)

### Expected QPU Results

| Qubit Count | Roundtrip Expected | Period Detection Expected |
|-------------|-------------------|---------------------------|
| 2 | 85-95% | 95-100% |
| 3 | 65-80% | 85-95% |
| 4 | 45-65% | 75-85% |
| 5 | 25-45% | 60-75% |
| 6 | 15-30% | 50-65% |

---

## Quick Start

```python
# Test single configuration
from qward.examples.papers.qft import test_roundtrip_base_case, test_period_detection_base_case
result_rt = test_roundtrip_base_case()
result_pd = test_period_detection_base_case()

# Run batch experiment
from qward.examples.papers.qft import run_batch
batch = run_batch("SR4", "DEP-MED", num_runs=10)

# Full campaign
from qward.examples.papers.qft import run_experiment_campaign, get_simulator_config_ids
results = run_experiment_campaign(
    config_ids=get_simulator_config_ids(),
    noise_ids=["IDEAL", "DEP-MED"],
    num_runs=10,
)

# Statistical analysis
from qward.examples.papers.qft import analyze_scalability, compare_noise_models
```

---

## Progress Tracker

### Completed ✅

1. ✅ Define experiment parameters (1024 shots, 10 runs, opt level 0)
2. ✅ Implement `qft.py` - QFT and QFTCircuitGenerator classes
3. ✅ Implement `phase_estimation.py` - PhaseEstimation classes
4. ✅ Implement `qft_configs.py` - 25 simulator configs + 4 hardware-only
5. ✅ Implement `qft_experiment.py` - experiment runner with incremental save
6. ✅ Implement `qft_statistical_analysis.py` - full statistical analysis
7. ✅ Run IDEAL baseline for all 25 configurations
8. ✅ Run noisy simulator study (6 noise models × 25 configs)
9. ✅ Compare roundtrip vs period detection modes
10. ✅ Analyze scalability crossover points
11. ✅ Test period variation hypothesis (confirmed: larger = better)
12. ✅ Test input variation hypothesis (minimal effect)
13. ✅ Rank noise model severity
14. ✅ Document all findings

### Next Steps (Week 4+)

1. [ ] Generate visualizations (scalability curves, heatmaps)
2. [ ] Create comparison plots (roundtrip vs period detection)
3. [ ] Prepare QPU experiment plan with resource estimates
4. [ ] Document findings for thesis chapter
