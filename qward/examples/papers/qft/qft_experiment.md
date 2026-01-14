# Quantum Fourier Transform (QFT) Experiment Design

## Research Goals

This document outlines the experimental design for evaluating QFT performance using the QWARD framework.

**Scope**: Simulator experiments (ideal and noisy) to understand QFT scalability and noise sensitivity.

### Primary Research Questions

1. **Scalability Limit**: At what circuit depth/qubit count does noise cause QFT to fail?
2. **Test Mode Comparison**: How do round-trip and period detection success rates compare?
3. **Period Sensitivity**: How does the choice of period affect detection accuracy?
4. **QWARD Correlation**: Can pre-runtime QWARD metrics predict success rate?

---

## Understanding the QFT Algorithm

### Mathematical Background

The QFT transforms computational basis states as:

$$F_{2^n}|k\rangle = \frac{1}{\sqrt{2^n}} \sum_{\ell=0}^{2^n-1} e^{2\pi i k\ell / 2^n} |\ell\rangle$$

### Circuit Structure

QFT on n qubits uses:
- **n Hadamard gates**: Create superposition
- **n(n-1)/2 controlled phase rotations**: R_k gates with k=2,3,...,n
- **⌊n/2⌋ SWAP gates**: Reverse qubit order

**Gate count formula**: `n(n+1)/2 + floor(n/2)`

### Key Challenge: Small Rotation Gates

For large k, the rotation R_k has angle 2π/2^k:
- R_10 ≈ 0.006 radians
- These become noise-dominated on real hardware
- This is the primary scalability limit

---

## Test Modes Explained

### Mode 1: Round-Trip (QFT → QFT⁻¹)

**Concept**: Apply QFT then inverse QFT - should return to original state.

```
|input⟩ → [QFT] → |fourier⟩ → [QFT⁻¹] → |input⟩ (ideally)
```

**Example** (3 qubits, input |101⟩):
1. Prepare |101⟩ (X gates on qubits 0 and 2)
2. Apply QFT
3. Apply QFT⁻¹
4. Measure → expect "101"

**Success Criteria**: `measurement == input_state`

**Pros**:
- Simple to understand
- Works with any input state
- Good for noise characterization

**Cons**:
- Tests QFT⁻¹ ∘ QFT = I, not QFT individually

### Mode 2: Period Detection

**Concept**: Encode a periodic signal, use QFT to detect the period (like in Shor's algorithm).

```
|0⟩ → [superposition] → [encode period T] → [QFT⁻¹] → |peaks at 2^n/T⟩
```

**Example** (4 qubits, period=4, N=16):
- Expected peaks: 16/4 = 4 → peaks at 0, 4, 8, 12
- Binary: 0000, 0100, 1000, 1100

**Success Criteria**: `measurement % (N/period) ≤ tolerance`

**Pros**:
- Tests actual QFT functionality
- Relevant for real algorithms (Shor)
- Multiple valid outcomes

**Cons**:
- More complex setup
- Requires understanding of period encoding

---

## Fixed Experiment Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Shots per job** | 1024 | Standard quantum computing benchmark |
| **Runs per config** | 10 | Sufficient for statistical analysis |
| **Optimization level** | 0 (none) | Establish baseline first |
| **Backend** | Aer simulator | Ideal and noisy modes |

---

## Experiment 1: Scalability - Round-Trip

### Objective
Determine the maximum qubit count where QFT→QFT⁻¹ remains effective.

### Circuit Configurations

| Config ID | Qubits | Input State | Gates (est.) | Expected |
|-----------|--------|-------------|--------------|----------|
| SR2 | 2 | 01 | 4 | ~99% |
| SR3 | 3 | 101 | 8 | ~99% |
| SR4 | 4 | 0110 | 13 | ~98% |
| SR5 | 5 | 10101 | 19 | ~95% |
| SR6 | 6 | 011001 | 27 | ~90% |
| SR7 | 7 | 1010101 | 36 | ~85% |
| SR8 | 8 | 01100110 | 46 | ~75% |
| SR10 | 10 | 0110011001 | 70 | ~50% |
| SR12 | 12 | 011001100110 | 99 | ~30% |

### Hypotheses

- **H1**: Success rate decreases with O(n²) gate count
- **H2**: Crossover point (success < 50%) occurs around 10-12 qubits
- **H3**: Success rate correlates with QWARD complexity metrics

---

## Experiment 2: Scalability - Period Detection

### Objective
Understand how period detection accuracy scales with qubits.

### Circuit Configurations

| Config ID | Qubits | Period | Peaks | Expected |
|-----------|--------|--------|-------|----------|
| SP3-P2 | 3 | 2 | 0, 4 | ~99% |
| SP4-P2 | 4 | 2 | 0, 8 | ~99% |
| SP4-P4 | 4 | 4 | 0,4,8,12 | ~99% |
| SP5-P4 | 5 | 4 | 0,8,16,24 | ~95% |
| SP6-P4 | 6 | 4 | ... | ~90% |
| SP6-P8 | 6 | 8 | ... | ~90% |
| SP8-P4 | 8 | 4 | ... | ~75% |
| SP8-P16 | 8 | 16 | ... | ~70% |

---

## Experiment 3: Period Variation

### Objective
Understand how period choice affects detection at fixed qubit count.

### Hypothesis
- Smaller periods → fewer peaks → harder to hit
- Larger periods → peaks closer together → harder to distinguish

| Config ID | Qubits | Period | Num Peaks | Peak Spacing |
|-----------|--------|--------|-----------|--------------|
| PV4-P2 | 4 | 2 | 2 | 8 |
| PV4-P4 | 4 | 4 | 4 | 4 |
| PV4-P8 | 4 | 8 | 8 | 2 |
| PV6-P2 | 6 | 2 | 2 | 32 |
| PV6-P4 | 6 | 4 | 4 | 16 |
| PV6-P8 | 6 | 8 | 8 | 8 |
| PV6-P16 | 6 | 16 | 16 | 4 |

---

## Experiment 4: Input State Variation

### Objective
Test whether different input states affect round-trip success.

### Hypothesis
- All zeros/ones might have different error profiles
- Alternating patterns might be more/less stable

| Config ID | Input | Pattern |
|-----------|-------|---------|
| IV4-0000 | 0000 | All zeros |
| IV4-1111 | 1111 | All ones |
| IV4-0101 | 0101 | Alternating |
| IV4-1010 | 1010 | Alternating (opposite) |
| IV4-0001 | 0001 | Single one |

---

## Noise Models

| ID | Type | Parameters | Description |
|----|------|------------|-------------|
| IDEAL | none | - | Perfect simulation |
| DEP-LOW | depolarizing | p1=0.1%, p2=0.5% | Low noise |
| DEP-MED | depolarizing | p1=0.5%, p2=2% | Medium noise |
| DEP-HIGH | depolarizing | p1=1%, p2=5% | High noise |
| READOUT | readout | p=2% | Measurement errors only |
| COMBINED | combined | dep + readout | Realistic noise |

---

## QWARD Integration

### Pre-Runtime Metrics

Using QWARD Scanner with:
- `QiskitMetrics`: Basic circuit properties
- `ComplexityMetrics`: Gate counts, depth ratios
- `StructuralMetrics`: Entanglement patterns
- `QuantumSpecificMetrics`: Quantum-specific measures

### Correlation Analysis Goals

1. Which metrics best predict success rate?
2. Can we build a success predictor from pre-runtime metrics?
3. Is there a "complexity threshold" for reliable QFT?

---

## Success Metrics

### Round-Trip Mode
- **Success**: `measurement == input_state`
- **Rate**: `successful_shots / total_shots`

### Period Detection Mode
- **Success**: `|measurement - nearest_peak| ≤ tolerance`
- **Rate**: `shots_at_peaks / total_shots`

### Thresholds
- **90%**: Acceptable for practical use
- **95%**: Good performance
- **99%**: Excellent (near-ideal)

---

## Data Collection

### Per-Run Data
```json
{
  "experiment_id": "SR4_IDEAL_001",
  "config_id": "SR4",
  "noise_model": "IDEAL",
  "run_number": 1,
  "num_qubits": 4,
  "test_mode": "roundtrip",
  "input_state": "0110",
  "circuit_depth": 15,
  "total_gates": 13,
  "qward_metrics": {...},
  "shots": 1024,
  "execution_time_ms": 45.2,
  "counts": {"0110": 1020, "0111": 2, ...},
  "success_rate": 0.996,
  "success_count": 1020
}
```

---

## Quick Start

### Test Base Cases

```python
from qft_experiment import test_roundtrip_base_case, test_period_detection_base_case

# Understand round-trip mode
result = test_roundtrip_base_case()

# Understand period detection mode
result = test_period_detection_base_case()
```

### Run Single Configuration

```python
from qft_experiment import test_single_config

result = test_single_config(config_id="SR4", noise_id="IDEAL", num_runs=5)
```

### Run Pilot Study

```python
from qft_experiment import test_pilot_study

results = test_pilot_study()
```

### Run Full Campaign

```python
from qft_experiment import run_experiment_campaign

results = run_experiment_campaign(
    config_ids=["SR3", "SR4", "SR5", "SR6"],
    noise_ids=["IDEAL", "DEP-LOW", "DEP-MED"],
    num_runs=10
)
```

---

## Expected Outputs

### Data Files
```
data/
├── raw/                    # Individual experiment JSONs
│   ├── SR3_IDEAL_20260113_143022.json
│   └── ...
└── aggregated/             # Campaign summaries
    └── campaign_summary_20260113_143022.json
```

### Key Findings to Report
1. Success rate vs qubit count curve
2. Noise sensitivity analysis
3. Round-trip vs period detection comparison
4. QWARD metric correlations
5. Practical scalability limit

---

*Document created: January 2026*
*Author: qWard Development Team*
