# Grover's Algorithm - QPU Experiment Selection Guide

## Goal

Use simulator data to identify which Grover configurations are worth running on real QPU hardware, minimizing expensive QPU time while maximizing scientific value.

---

## The Three Regions Framework

Based on simulator results, we classify algorithm behavior into three regions:

### Region 1: Signal Dominant (Algorithm Works)
- **Definition**: Success rate significantly above random chance, quantum advantage clear
- **Criterion**: Success rate > 2× random chance AND > 50%
- **Value**: Validate QPU matches simulator predictions
- **Expected behavior**: Clear peak in measurement histogram at marked state

### Region 2: Signal + Noise (Partial Data)
- **Definition**: Success rate above random but degraded, quantum advantage marginal
- **Criterion**: 2× random chance < Success rate < 50%
- **Value**: Understand noise transition, find practical limits
- **Expected behavior**: Marked state visible but surrounded by noise

### Region 3: Noise Dominant (Algorithm Fails)
- **Definition**: Success rate at or near random chance, no quantum advantage
- **Criterion**: Success rate ≤ 2× random chance
- **Value**: Confirm failure boundaries, document algorithm limits
- **Expected behavior**: Uniform or near-uniform distribution

---

## Grover Configuration Analysis

### Random Chance Baselines

| Qubits | Search Space | Marked States | Random Chance | 2× Random |
|--------|--------------|---------------|---------------|-----------|
| 2 | 4 | 1 | 25.00% | 50.00% |
| 3 | 8 | 1 | 12.50% | 25.00% |
| 3 | 8 | 2 | 25.00% | 50.00% |
| 4 | 16 | 1 | 6.25% | 12.50% |
| 5 | 32 | 1 | 3.12% | 6.25% |
| 6 | 64 | 1 | 1.56% | 3.12% |
| 7 | 128 | 1 | 0.78% | 1.56% |
| 8 | 256 | 1 | 0.39% | 0.78% |

---

## Region Classification from Simulator Data

### Using Hardware-Calibrated Noise Models

The most realistic QPU predictions come from hardware-calibrated noise models:
- **IBM-HERON-R3**: Best IBM hardware (ibm_boston)
- **IBM-HERON-R2**: Mid-tier IBM (ibm_marrakesh) 
- **IBM-HERON-R1**: Older IBM (ibm_torino)
- **RIGETTI-ANKAA3**: Rigetti comparison

### Classification Table: Scalability Study (S-series)

| Config | Qubits | Depth | Gates | IBM-R3 | IBM-R2 | IBM-R1 | Region (R2) |
|--------|--------|-------|-------|--------|--------|--------|-------------|
| S2-1 | 2 | 12 | 21 | ~95% | ~90% | ~85% | **Region 1** ✅ |
| S3-1 | 3 | 26 | 49 | ~75% | ~60% | ~50% | **Region 1-2** ⚠️ |
| S4-1 | 4 | 38 | 89 | ~50% | ~35% | ~25% | **Region 2** ⚠️ |
| S5-1 | 5 | 50 | 141 | ~25% | ~15% | ~10% | **Region 2-3** ⚠️ |
| S6-1 | 6 | 74 | 231 | ~10% | ~5% | ~3% | **Region 3** ❌ |
| S7-1 | 7 | 98 | 337 | ~5% | ~2% | ~1% | **Region 3** ❌ |
| S8-1 | 8 | 146 | 571 | ~2% | ~1% | <1% | **Region 3** ❌ |

### Classification Table: Symmetry Study (SYM/ASYM-series)

| Config | Marked States | Depth | Gates | IBM-R2 Est | Region |
|--------|---------------|-------|-------|------------|--------|
| SYM-1 | ["000", "111"] | 17 | 36 | ~75% | **Region 1** ✅ |
| SYM-2 | ["001", "110"] | 17 | 36 | ~75% | **Region 1** ✅ |
| ASYM-1 | ["000", "001"] | 19 | 40 | ~70% | **Region 1** ✅ |
| ASYM-2 | ["000", "011"] | 19 | 38 | ~72% | **Region 1** ✅ |

### Classification Table: Marked Count Study (M-series)

| Config | Qubits | Marked | Depth | Gates | IBM-R2 Est | Region |
|--------|--------|--------|-------|-------|------------|--------|
| M3-1 | 3 | 1 | 26 | 57 | ~55% | **Region 1-2** ⚠️ |
| M3-2 | 3 | 2 | 17 | 36 | ~75% | **Region 1** ✅ |
| M3-4 | 3 | 4 | 2 | 9 | ~50%* | **Region 2** (no advantage) |
| M4-1 | 4 | 1 | 38 | 101 | ~30% | **Region 2** ⚠️ |
| M4-2 | 4 | 2 | 32 | 77 | ~45% | **Region 2** ⚠️ |
| M4-4 | 4 | 4 | 25 | 58 | ~60% | **Region 1** ✅ |

*M3-4 marks 50% of states, so 50% is the theoretical maximum (no iterations needed)

### Classification Table: Hamming Weight Study (H-series)

| Config | Qubits | State | Hamming | Gates | IBM-R2 Est | Region |
|--------|--------|-------|---------|-------|------------|--------|
| H3-0 | 3 | 000 | 0 | 57 | ~52% | **Region 1-2** ⚠️ |
| H3-1 | 3 | 001 | 1 | 53 | ~55% | **Region 1-2** ⚠️ |
| H3-2 | 3 | 011 | 2 | 49 | ~58% | **Region 1** ✅ |
| H3-3 | 3 | 111 | 3 | 45 | ~62% | **Region 1** ✅ |
| H4-0 | 4 | 0000 | 0 | 101 | ~28% | **Region 2** ⚠️ |
| H4-2 | 4 | 0011 | 2 | 89 | ~33% | **Region 2** ⚠️ |
| H4-4 | 4 | 1111 | 4 | 77 | ~38% | **Region 2** ⚠️ |

---

## Analysis Methodology

### Step 1: Extract Data from Simulator Results

For each configuration, extract from campaign summary JSON:

```python
def classify_region(success_rate: float, num_qubits: int, num_marked: int) -> str:
    """Classify a configuration into one of three regions."""
    search_space = 2 ** num_qubits
    random_chance = num_marked / search_space
    double_random = 2 * random_chance
    
    if success_rate > double_random and success_rate > 0.50:
        return "Region 1: Signal Dominant"
    elif success_rate > double_random:
        return "Region 2: Signal + Noise"
    else:
        return "Region 3: Noise Dominant"
```

### Step 2: Create Region Summary

```python
def create_region_summary(results: list[dict], noise_model: str) -> dict:
    """Summarize configurations by region."""
    summary = {
        "Region 1": [],  # Good candidates for validation
        "Region 2": [],  # Interesting transition zone
        "Region 3": []   # Document failure (fewer runs needed)
    }
    
    for config in results:
        if config["noise_model"] == noise_model:
            region = classify_region(
                config["mean_success_rate"],
                config["num_qubits"],
                config["num_marked"]
            )
            summary[region].append(config)
    
    return summary
```

### Step 3: Calculate Confidence Intervals

For QPU planning, we need confidence intervals from simulator data:

```python
def calculate_expected_range(mean: float, std: float, n_runs: int = 10) -> tuple:
    """Calculate expected range for QPU results."""
    # 95% confidence interval
    se = std / np.sqrt(n_runs)
    ci_lower = mean - 1.96 * se
    ci_upper = mean + 1.96 * se
    
    # Add hardware uncertainty (typically ±10% more variance than simulator)
    hardware_factor = 1.10
    expected_lower = max(0, ci_lower * (1 - 0.10))
    expected_upper = min(1, ci_upper * (1 + 0.10))
    
    return expected_lower, expected_upper
```

---

## QPU Experiment Prioritization

### Priority 1: Region 1 Validation (High Value, High Confidence)

**Goal**: Confirm QPU matches simulator predictions for working algorithms

| Priority | Config | Qubits | Runs | Expected Success | Value |
|----------|--------|--------|------|------------------|-------|
| 1.1 | S2-1 | 2 | 5 | 80-95% | Baseline validation |
| 1.2 | SYM-1 | 3 | 5 | 65-80% | Short circuit (2 marked) |
| 1.3 | M3-2 | 3 | 5 | 65-80% | Multiple marked states |
| 1.4 | H3-3 | 3 | 5 | 55-70% | All-1s state (fewer gates) |

**Success Criteria**: QPU results within ±15% of simulator predictions

### Priority 2: Region 2 Boundary Exploration (Medium Value)

**Goal**: Find exact transition point where algorithm becomes marginal

| Priority | Config | Qubits | Runs | Expected Success | Value |
|----------|--------|--------|------|------------------|-------|
| 2.1 | S3-1 | 3 | 5 | 45-65% | Scalability boundary |
| 2.2 | S4-1 | 4 | 5 | 25-45% | Near-marginal performance |
| 2.3 | H3-0 | 3 | 3 | 45-60% | Compare to H3-3 |
| 2.4 | M4-2 | 4 | 3 | 35-55% | Multiple marked at 4q |

**Success Criteria**: Determine if quantum advantage persists

### Priority 3: Region 3 Failure Documentation (Low Value but Necessary)

**Goal**: Confirm algorithm failure at larger scales

| Priority | Config | Qubits | Runs | Expected Success | Value |
|----------|--------|--------|------|------------------|-------|
| 3.1 | S5-1 | 5 | 2 | 5-15% | Document failure onset |
| 3.2 | S6-1 | 6 | 2 | 1-5% | Confirm random behavior |
| 3.3 | S7-1 | 7 | 2 | 0.5-2% | Deep failure region |

**Success Criteria**: Results statistically indistinguishable from random

---

## Data Analysis Plan

### 1. Pre-QPU Analysis (Before Running)

```python
def analyze_simulator_regions(campaign_data_path: str, noise_model: str = "IBM-HERON-R2"):
    """
    Analyze simulator data to plan QPU experiments.
    
    Output:
    - Region classification for each config
    - Recommended QPU experiments by priority
    - Expected success ranges
    - Total QPU time estimate
    """
    # Load data
    with open(campaign_data_path) as f:
        data = json.load(f)
    
    # Filter by noise model
    filtered = [d for d in data if d["noise_model"] == noise_model]
    
    # Classify regions
    for config in filtered:
        config["region"] = classify_region(
            config["mean_success_rate"],
            config["num_qubits"],
            config["num_marked"]
        )
    
    # Sort by region and then by expected value
    region_1 = sorted([c for c in filtered if "Region 1" in c["region"]], 
                      key=lambda x: x["mean_success_rate"], reverse=True)
    region_2 = sorted([c for c in filtered if "Region 2" in c["region"]], 
                      key=lambda x: x["mean_success_rate"], reverse=True)
    region_3 = sorted([c for c in filtered if "Region 3" in c["region"]], 
                      key=lambda x: x["num_qubits"])
    
    return {
        "region_1": region_1,
        "region_2": region_2,
        "region_3": region_3,
    }
```

### 2. Post-QPU Analysis (After Running)

```python
def compare_qpu_to_simulator(qpu_results: list, simulator_predictions: list):
    """
    Compare actual QPU results to simulator predictions.
    
    Metrics:
    - Mean Absolute Error (MAE)
    - Which noise model best predicts QPU?
    - Did region classifications hold?
    """
    comparisons = []
    
    for qpu in qpu_results:
        config_id = qpu["config_id"]
        sim = next(s for s in simulator_predictions if s["config_id"] == config_id)
        
        comparison = {
            "config_id": config_id,
            "qpu_success": qpu["mean_success_rate"],
            "simulator_success": sim["mean_success_rate"],
            "absolute_error": abs(qpu["mean_success_rate"] - sim["mean_success_rate"]),
            "predicted_region": sim["region"],
            "actual_region": classify_region(
                qpu["mean_success_rate"], 
                qpu["num_qubits"],
                qpu["num_marked"]
            ),
            "region_match": sim["region"] == classify_region(...)
        }
        comparisons.append(comparison)
    
    # Summary statistics
    mae = np.mean([c["absolute_error"] for c in comparisons])
    region_accuracy = np.mean([c["region_match"] for c in comparisons])
    
    return {
        "comparisons": comparisons,
        "mae": mae,
        "region_classification_accuracy": region_accuracy,
    }
```

### 3. Noise Model Identification

```python
def identify_qpu_noise_profile(qpu_results: list, all_simulator_results: dict):
    """
    Determine which simulator noise model best matches QPU behavior.
    
    Compare QPU results against each noise model to find best match.
    """
    noise_models = ["IBM-HERON-R3", "IBM-HERON-R2", "IBM-HERON-R1", 
                    "DEP-LOW", "DEP-MED", "RIGETTI-ANKAA3"]
    
    model_errors = {}
    
    for noise_model in noise_models:
        sim_results = all_simulator_results[noise_model]
        errors = []
        
        for qpu in qpu_results:
            sim = next((s for s in sim_results if s["config_id"] == qpu["config_id"]), None)
            if sim:
                errors.append(abs(qpu["mean_success_rate"] - sim["mean_success_rate"]))
        
        model_errors[noise_model] = {
            "mae": np.mean(errors),
            "max_error": np.max(errors),
            "rmse": np.sqrt(np.mean(np.array(errors)**2)),
        }
    
    # Rank models by MAE
    ranked = sorted(model_errors.items(), key=lambda x: x[1]["mae"])
    
    return {
        "best_match": ranked[0][0],
        "all_models": model_errors,
        "ranking": [m[0] for m in ranked],
    }
```

---

## Visualization Plan

### 1. Region Map Heatmap

```python
def plot_region_map(results: list, noise_model: str):
    """
    Create heatmap showing success rate by qubits and config type.
    
    X-axis: Qubit count (2-8)
    Y-axis: Config type (S, M, H, SYM)
    Color: Success rate (green=Region1, yellow=Region2, red=Region3)
    """
    pass
```

### 2. Transition Curves

```python
def plot_transition_curves(results: list, noise_models: list):
    """
    Plot success rate vs qubit count for different noise models.
    
    Shows where algorithms transition between regions.
    Highlight 2× random threshold line.
    """
    pass
```

### 3. QPU vs Simulator Comparison

```python
def plot_qpu_vs_simulator(qpu_results: list, simulator_results: list):
    """
    Scatter plot: simulator prediction (x) vs QPU actual (y)
    
    Perfect prediction = diagonal line
    Color by region
    Size by number of qubits
    """
    pass
```

---

## Decision Checklist

Before running QPU experiments, confirm:

- [ ] Simulator data loaded for all target configs
- [ ] Region classification completed
- [ ] Expected success ranges calculated
- [ ] Priority order established
- [ ] Total QPU time estimated (stay within budget)
- [ ] Data collection pipeline ready
- [ ] Analysis scripts tested

After running QPU experiments:

- [ ] Compare QPU to simulator predictions
- [ ] Calculate MAE and region accuracy
- [ ] Identify best-matching noise model
- [ ] Document any unexpected behaviors
- [ ] Update region boundaries if needed

---

## Resource Estimate

Based on priority tiers:

| Tier | Experiments | Runs | Time Est | Value |
|------|-------------|------|----------|-------|
| Priority 1 | 4 configs | 5 each = 20 | ~100s | High |
| Priority 2 | 4 configs | 3-5 each = 14 | ~70s | Medium |
| Priority 3 | 3 configs | 2 each = 6 | ~30s | Low but needed |
| **Total** | 11 configs | 40 runs | **~200s** | |

This leaves ~400s buffer for additional experiments or reruns if needed.
