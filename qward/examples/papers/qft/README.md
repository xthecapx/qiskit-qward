# Quantum Fourier Transform Experiment

This folder contains the experimental framework for evaluating QFT performance using the QWARD library.

## Purpose

Systematic study of QFT behavior under different conditions:
- Circuit scalability (2-12 qubits)
- Two test modes: round-trip and period detection
- Noise sensitivity analysis
- QWARD metrics correlation

## Files

| File | Description |
|------|-------------|
| `qft_experiment.md` | Experiment design document |
| `qft_experiment.py` | Main experiment runner |
| `qft_configs.py` | Circuit configurations |
| `README.md` | This file |

## Quick Start

### Understand the Algorithm

```python
from qft_experiment import test_roundtrip_base_case, test_period_detection_base_case

# Round-trip: QFT → QFT⁻¹ should return to input
result = test_roundtrip_base_case()

# Period detection: QFT extracts period from encoded state
result = test_period_detection_base_case()
```

### Run Experiments

```python
from qft_experiment import test_single_config, run_batch

# Quick test
result = test_single_config(config_id="SR4", noise_id="IDEAL", num_runs=3)

# Full batch
batch = run_batch(config_id="SR4", noise_id="DEP-MED", num_runs=10)
```

## Test Modes

### Round-Trip
- Input: basis state (e.g., "0110")
- Process: QFT → QFT⁻¹
- Success: returns to input state

### Period Detection  
- Input: period value (e.g., period=4)
- Process: encode period → QFT⁻¹
- Success: measurement at expected peaks

## Parameters

- **Shots per job**: 1024
- **Runs per config**: 10
- **Optimization level**: 0 (none)

## Data Structure

```
data/
├── raw/          # Individual experiment records (JSON)
└── aggregated/   # Per-config aggregates
```
