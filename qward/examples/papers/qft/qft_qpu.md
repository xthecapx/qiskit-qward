# Quantum Fourier Transform - QPU Experiment Selection Guide

## Goal

Use simulator data to identify which QFT configurations are worth running on real QPU hardware, focusing on the unique characteristics of QFT (roundtrip vs period detection modes).

---

## The Three Regions Framework

Based on simulator results, we classify algorithm behavior into three regions:

### Region 1: Signal Dominant (Algorithm Works)
- **Definition**: Success rate > 80% for roundtrip, > 90% for period detection
- **Value**: Validate QPU matches simulator predictions
- **Expected behavior**: Clear reconstruction of input state (roundtrip) or sharp peaks (period detection)

### Region 2: Signal + Noise (Partial Data)
- **Definition**: 50% < Success rate < 80% (roundtrip) or 70-90% (period detection)
- **Value**: Understand noise transition, find practical limits
- **Expected behavior**: Correct answer visible but with significant noise

### Region 3: Noise Dominant (Algorithm Fails)
- **Definition**: Success rate < 50% (roundtrip) or < 70% (period detection)
- **Criterion**: Approaching random chance behavior
- **Value**: Confirm failure boundaries, document algorithm limits
- **Expected behavior**: Uniform or noisy distribution

---

## QFT Success Criteria

### Roundtrip Mode
- **Success**: Measure the exact input state after QFT → QFT⁻¹
- **Random chance**: 1/(2^n) for n qubits
- **Quantum advantage**: Success >> random chance

### Period Detection Mode
- **Success**: Measure any valid peak (multiple correct answers)
- **Number of peaks**: N / period (where N = 2^n)
- **Random chance**: peaks / N = 1 / period
- **Key insight**: More peaks = more noise-tolerant

---

## QFT Configuration Analysis

### Random Chance Baselines

| Qubits | States | Roundtrip Random | Period=2 Random | Period=4 Random | Period=8 Random |
|--------|--------|------------------|-----------------|-----------------|-----------------|
| 2 | 4 | 25.00% | 50.00% | 100.00%* | - |
| 3 | 8 | 12.50% | 50.00% | 50.00% | 100.00%* |
| 4 | 16 | 6.25% | 50.00% | 25.00% | 50.00% |
| 5 | 32 | 3.12% | 50.00% | 12.50% | 25.00% |
| 6 | 64 | 1.56% | 50.00% | 6.25% | 12.50% |
| 7 | 128 | 0.78% | 50.00% | 3.12% | 6.25% |
| 8 | 256 | 0.39% | 50.00% | 1.56% | 3.12% |

*When period = states, all measurements are "correct"

---

## Region Classification from Simulator Data

### Using RIGETTI-ANKAA3 as QPU Proxy

From the QFT campaign summary, here's the classification:

### Classification Table: Scalability - Roundtrip Mode (SR-series)

| Config | Qubits | Input | RIGETTI | Region | Notes |
|--------|--------|-------|---------|--------|-------|
| SR2 | 2 | 01 | 94.3% | **Region 1** ✅ | Baseline |
| SR3 | 3 | 101 | 90.9% | **Region 1** ✅ | Still good |
| SR4 | 4 | 0110 | 86.1% | **Region 1** ✅ | Marginal R1 |
| SR5 | 5 | 10101 | 81.8% | **Region 1-2** ⚠️ | Transition |
| SR6 | 6 | 011001 | 75.9% | **Region 2** ⚠️ | Degraded |
| SR7 | 7 | 1010101 | 71.5% | **Region 2** ⚠️ | Degraded |
| SR8 | 8 | 01100110 | 65.4% | **Region 2** ⚠️ | Near failure |

### Classification Table: Scalability - Period Detection (SP-series)

| Config | Qubits | Period | Peaks | RIGETTI | Region | Notes |
|--------|--------|--------|-------|---------|--------|-------|
| SP3-P2 | 3 | 2 | 4 | 96.7% | **Region 1** ✅ | Excellent |
| SP4-P2 | 4 | 2 | 8 | 92.3% | **Region 1** ✅ | Good |
| SP4-P4 | 4 | 4 | 4 | 96.7% | **Region 1** ✅ | More peaks |
| SP5-P4 | 5 | 4 | 8 | 92.2% | **Region 1** ✅ | Good |
| SP6-P4 | 6 | 4 | 16 | 88.2% | **Region 1** ✅ | Marginal R1 |
| SP6-P8 | 6 | 8 | 8 | 92.1% | **Region 1** ✅ | More peaks helps |

### Classification Table: Period Variation (PV-series)

| Config | Qubits | Period | Peaks | RIGETTI | Region | Notes |
|--------|--------|--------|-------|---------|--------|-------|
| PV4-P2 | 4 | 2 | 8 | 92.5% | **Region 1** ✅ | - |
| PV4-P4 | 4 | 4 | 4 | 96.3% | **Region 1** ✅ | Larger period better |
| PV4-P8 | 4 | 8 | 2 | 100.0% | **Region 1** ✅ | Perfect! |
| PV6-P2 | 6 | 2 | 32 | 84.6% | **Region 1** ✅ | - |
| PV6-P4 | 6 | 4 | 16 | 88.1% | **Region 1** ✅ | - |
| PV6-P8 | 6 | 8 | 8 | 91.9% | **Region 1** ✅ | - |
| PV6-P16 | 6 | 16 | 4 | 95.8% | **Region 1** ✅ | Larger period wins |

### Classification Table: Input Variation (IV-series)

| Config | Input | Hamming | RIGETTI | Region | Notes |
|--------|-------|---------|---------|--------|-------|
| IV4-0000 | 0000 | 0 | 86.3% | **Region 1** ✅ | Best |
| IV4-0001 | 0001 | 1 | 86.5% | **Region 1** ✅ | Similar |
| IV4-0101 | 0101 | 2 | 86.5% | **Region 1** ✅ | Similar |
| IV4-1010 | 1010 | 2 | 86.3% | **Region 1** ✅ | Similar |
| IV4-1111 | 1111 | 4 | 85.6% | **Region 1** ✅ | Slightly worse |

---

## Key Observations from Simulator Data

### 1. QFT is More Noise-Resilient than Grover

| Algorithm | 4-qubit DEP-MED | 4-qubit RIGETTI | Status |
|-----------|-----------------|-----------------|--------|
| Grover (1 marked) | ~7% | ~35% | Near failure |
| QFT Roundtrip | ~76% | ~86% | Still working |
| QFT Period Detection | ~87-94% | ~92-96% | Robust |

**Why?** QFT has gradual phase rotations; Grover has delicate interference patterns.

### 2. Period Detection >> Roundtrip

At 6 qubits under realistic noise:
- **Roundtrip**: 75.9% success
- **Period Detection (P8)**: 92.1% success
- **Difference**: +16.2%

### 3. Larger Periods = More Robust

For 6-qubit period detection:
- Period 2: 84.6%
- Period 4: 88.1%
- Period 8: 91.9%
- Period 16: 95.8%

**More valid outcomes = higher probability of correct answer**

### 4. Input State Has Minimal Impact

For 4-qubit roundtrip, Hamming weight 0 vs 4:
- Difference: only 0.7%
- **Conclusion**: Input state doesn't matter for QFT planning

---

## Analysis Methodology

### Step 1: Region Classification

```python
def classify_qft_region(success_rate: float, mode: str) -> str:
    """
    Classify QFT configuration into regions.
    
    Different thresholds for roundtrip vs period detection
    because period detection has multiple correct answers.
    """
    if mode == "roundtrip":
        if success_rate > 0.80:
            return "Region 1: Signal Dominant"
        elif success_rate > 0.50:
            return "Region 2: Signal + Noise"
        else:
            return "Region 3: Noise Dominant"
    
    elif mode == "period_detection":
        if success_rate > 0.90:
            return "Region 1: Signal Dominant"
        elif success_rate > 0.70:
            return "Region 2: Signal + Noise"
        else:
            return "Region 3: Noise Dominant"
```

### Step 2: Compare Modes at Same Qubit Count

```python
def compare_modes(roundtrip_results: list, period_results: list):
    """
    Compare roundtrip vs period detection at the same qubit count.
    
    Key question: How much does period detection help?
    """
    comparisons = []
    
    for qubit_count in [3, 4, 5, 6]:
        rt = next((r for r in roundtrip_results if r["num_qubits"] == qubit_count), None)
        pd = [r for r in period_results if r["num_qubits"] == qubit_count]
        
        if rt and pd:
            best_pd = max(pd, key=lambda x: x["mean_success_rate"])
            comparisons.append({
                "qubits": qubit_count,
                "roundtrip_success": rt["mean_success_rate"],
                "period_detection_success": best_pd["mean_success_rate"],
                "period_used": best_pd["period"],
                "improvement": best_pd["mean_success_rate"] - rt["mean_success_rate"],
            })
    
    return comparisons
```

### Step 3: Find Optimal Period for Each Qubit Count

```python
def find_optimal_period(results: list, qubit_count: int) -> dict:
    """
    For a given qubit count, find the period that maximizes success.
    
    Trade-off: larger period = more noise-tolerant but less informative
    """
    candidates = [r for r in results if r["num_qubits"] == qubit_count]
    
    if not candidates:
        return None
    
    # Sort by success rate
    sorted_candidates = sorted(candidates, key=lambda x: x["mean_success_rate"], reverse=True)
    
    return {
        "optimal": sorted_candidates[0],
        "all_periods": [(c["period"], c["mean_success_rate"]) for c in sorted_candidates],
    }
```

---

## QPU Experiment Prioritization

### Priority 1: Mode Comparison (High Value)

**Goal**: Validate that period detection outperforms roundtrip on real QPU

| Priority | Configs | Qubits | Runs | Expected | Value |
|----------|---------|--------|------|----------|-------|
| 1.1 | SR4 vs SP4-P4 | 4 | 5 each | 86% vs 96% | Mode comparison |
| 1.2 | SR6 vs SP6-P8 | 6 | 5 each | 76% vs 92% | Larger scale |

**Success Criteria**: Period detection shows >10% improvement

### Priority 2: Scalability Validation (High Value)

**Goal**: Confirm scalability limits match simulator

| Priority | Config | Mode | Qubits | Runs | Expected | Value |
|----------|--------|------|--------|------|----------|-------|
| 2.1 | SR2 | Roundtrip | 2 | 5 | 90-98% | Baseline |
| 2.2 | SR4 | Roundtrip | 4 | 5 | 80-92% | Mid-range |
| 2.3 | SR6 | Roundtrip | 6 | 5 | 65-82% | Near boundary |
| 2.4 | SR8 | Roundtrip | 8 | 3 | 55-75% | Boundary test |

**Success Criteria**: Degradation pattern matches simulator (exponential decay)

### Priority 3: Period Optimization (Medium Value)

**Goal**: Validate larger period = better success on QPU

| Priority | Config | Period | Qubits | Runs | Expected | Value |
|----------|--------|--------|--------|------|----------|-------|
| 3.1 | PV4-P2 | 2 | 4 | 3 | 88-96% | Small period |
| 3.2 | PV4-P4 | 4 | 4 | 3 | 92-100% | Medium period |
| 3.3 | PV4-P8 | 8 | 4 | 3 | 98-100% | Large period |

**Success Criteria**: Clear trend P8 > P4 > P2

### Priority 4: Hardware-Only Configurations (Exploratory)

**Goal**: Test configurations too expensive for full simulator study

| Priority | Config | Mode | Qubits | Runs | Expected | Value |
|----------|--------|------|--------|------|----------|-------|
| 4.1 | SR10 | Roundtrip | 10 | 2 | 40-60%? | Extrapolation test |
| 4.2 | SP8-P16 | Period | 8 (9 total) | 2 | 75-90%? | Large circuit |

**Success Criteria**: Understand behavior beyond simulator data

---

## Data Analysis Plan

### 1. Pre-QPU Analysis

```python
def analyze_qft_regions(campaign_data_path: str, noise_model: str = "RIGETTI-ANKAA3"):
    """
    Analyze QFT simulator data to plan QPU experiments.
    
    Output:
    - Region classification for each config
    - Mode comparison summary
    - Optimal periods by qubit count
    - QPU experiment recommendations
    """
    with open(campaign_data_path) as f:
        data = json.load(f)
    
    # Separate by mode
    roundtrip = [d for d in data if d["config_id"].startswith("SR") or d["config_id"].startswith("IV")]
    period_detection = [d for d in data if d["config_id"].startswith("SP") or d["config_id"].startswith("PV")]
    
    # Classify regions
    for config in roundtrip:
        config["region"] = classify_qft_region(config["mean_success_rate"], "roundtrip")
    
    for config in period_detection:
        config["region"] = classify_qft_region(config["mean_success_rate"], "period_detection")
    
    # Mode comparison
    mode_comparison = compare_modes(roundtrip, period_detection)
    
    return {
        "roundtrip": roundtrip,
        "period_detection": period_detection,
        "mode_comparison": mode_comparison,
    }
```

### 2. Post-QPU Analysis

```python
def analyze_qft_qpu_results(qpu_results: list, simulator_results: list):
    """
    Compare QPU results to simulator predictions.
    
    Key analyses:
    1. Does mode comparison hold? (period > roundtrip)
    2. Does period trend hold? (larger = better)
    3. What's the actual scalability limit?
    4. Which noise model best predicts QPU?
    """
    analyses = {}
    
    # 1. Mode comparison
    rt_qpu = [r for r in qpu_results if r["mode"] == "roundtrip"]
    pd_qpu = [r for r in qpu_results if r["mode"] == "period_detection"]
    
    mode_diff = []
    for qubit_count in [4, 6]:
        rt = next((r for r in rt_qpu if r["num_qubits"] == qubit_count), None)
        pd = max([r for r in pd_qpu if r["num_qubits"] == qubit_count], 
                 key=lambda x: x["mean_success_rate"])
        if rt and pd:
            mode_diff.append(pd["mean_success_rate"] - rt["mean_success_rate"])
    
    analyses["mode_comparison"] = {
        "period_detection_advantage": np.mean(mode_diff),
        "hypothesis_confirmed": np.mean(mode_diff) > 0.10,
    }
    
    # 2. Period trend
    period_configs = [r for r in qpu_results if r["config_id"].startswith("PV4")]
    if period_configs:
        sorted_by_period = sorted(period_configs, key=lambda x: x["period"])
        analyses["period_trend"] = {
            "by_period": [(c["period"], c["mean_success_rate"]) for c in sorted_by_period],
            "trend_confirmed": all(
                sorted_by_period[i]["mean_success_rate"] <= sorted_by_period[i+1]["mean_success_rate"]
                for i in range(len(sorted_by_period)-1)
            ),
        }
    
    # 3. Scalability
    sr_configs = sorted([r for r in qpu_results if r["config_id"].startswith("SR")],
                       key=lambda x: x["num_qubits"])
    analyses["scalability"] = {
        "by_qubits": [(c["num_qubits"], c["mean_success_rate"]) for c in sr_configs],
        "50_percent_threshold": next(
            (c["num_qubits"] for c in sr_configs if c["mean_success_rate"] < 0.50), 
            "beyond 8 qubits"
        ),
    }
    
    return analyses
```

---

## Visualization Plan

### 1. Mode Comparison Chart

```python
def plot_mode_comparison(results: dict):
    """
    Bar chart comparing roundtrip vs period detection by qubit count.
    
    X-axis: Qubit count
    Y-axis: Success rate
    Bars: Roundtrip (blue) vs Period Detection (green)
    """
    pass
```

### 2. Period Optimization Curves

```python
def plot_period_optimization(results: list, qubit_count: int):
    """
    Line plot showing success vs period at fixed qubit count.
    
    X-axis: Period
    Y-axis: Success rate
    Show both simulator prediction and QPU actual
    """
    pass
```

### 3. Scalability Decay Curves

```python
def plot_scalability_decay(results: list):
    """
    Line plot showing success vs qubit count.
    
    Separate lines for:
    - Roundtrip
    - Period Detection (best period)
    - Random chance (for reference)
    
    Highlight Region boundaries
    """
    pass
```

### 4. Simulator vs QPU Scatter

```python
def plot_simulator_vs_qpu(simulator: list, qpu: list):
    """
    Scatter plot comparing predictions to actual.
    
    X-axis: Simulator prediction
    Y-axis: QPU actual
    Color: Mode (roundtrip vs period detection)
    Shape: Region
    Diagonal line = perfect prediction
    """
    pass
```

---

## Decision Checklist

### Before Running QPU Experiments

- [ ] Simulator data analyzed for all target configs
- [ ] Region classification complete
- [ ] Mode comparison documented
- [ ] Period optimization trend confirmed
- [ ] Expected success ranges calculated
- [ ] Priority order established
- [ ] Hardware-only configs identified for exploration

### After Running QPU Experiments

- [ ] Compare QPU to simulator predictions
- [ ] Confirm mode comparison (period > roundtrip)
- [ ] Confirm period trend (larger = better)
- [ ] Identify scalability limits
- [ ] Determine best-matching noise model
- [ ] Document hardware-only config results

---

## Resource Estimate

Based on priority tiers:

| Tier | Experiments | Runs | Time Est | Value |
|------|-------------|------|----------|-------|
| Priority 1 (Mode) | 4 configs | 5 each = 20 | ~100s | High |
| Priority 2 (Scale) | 4 configs | 3-5 each = 18 | ~90s | High |
| Priority 3 (Period) | 3 configs | 3 each = 9 | ~45s | Medium |
| Priority 4 (Hardware) | 2 configs | 2 each = 4 | ~20s | Exploratory |
| **Total** | 13 configs | 51 runs | **~255s** | |

This stays within a 10-minute QPU budget with margin for reruns.

---

## Key Questions to Answer

### From Simulator Data (Pre-QPU)

1. **Where are the region boundaries for QFT?**
   - Roundtrip: ~8 qubits before Region 3
   - Period Detection: 8+ qubits still in Region 1-2

2. **How much does period detection help?**
   - Average: +16% improvement over roundtrip

3. **What's the optimal period strategy?**
   - Use largest period practical for your application

### From QPU Data (Post-QPU)

1. **Does QPU match simulator predictions?**
   - Expected: Within ±10-15%

2. **Which noise model best predicts QPU?**
   - Candidates: RIGETTI-ANKAA3, IBM-HERON-R2

3. **Are there unexpected QPU behaviors?**
   - Watch for: crosstalk, calibration drift, qubit-specific errors

4. **What are the practical QFT limits on this QPU?**
   - Expected: 6-8 qubits for roundtrip, 8+ for period detection
