# Quantum Fourier Transform Experiment

This folder contains the experimental framework for evaluating QFT performance using the QWARD library.

## Key Results Summary

| Finding | Value |
|---------|-------|
| **Period detection vs roundtrip** | +16.6% better under medium noise |
| **Max qubits (DEP-MED, roundtrip)** | 4-5 qubits for >50% success |
| **Max qubits (DEP-MED, period detection)** | 6+ qubits for >50% success |
| **Most resilient noise** | READOUT (-6.2% degradation) |
| **Most severe noise** | DEP-HIGH (-39.1% degradation) |

## Files

| File | Description |
|------|-------------|
| `qft_experiment.md` | **Complete experiment design document with all results** |
| `qft_experiment.py` | Main experiment runner |
| `qft_configs.py` | Circuit configurations (25 simulator + 4 hardware-only) |
| `qft_statistical_analysis.py` | Statistical analysis functions |
| `__init__.py` | Package exports |
| `README.md` | This file |

## Quick Start

### Understand the Algorithm

```python
from qward.examples.papers.qft import test_roundtrip_base_case, test_period_detection_base_case

# Round-trip: QFT → QFT⁻¹ should return to input
result = test_roundtrip_base_case()

# Period detection: QFT extracts period from encoded state
result = test_period_detection_base_case()
```

### Run Experiments

```python
from qward.examples.papers.qft import (
    run_batch,
    run_experiment_campaign,
    get_simulator_config_ids,
)

# Quick single config test
batch = run_batch("SR4", "DEP-MED", num_runs=10)

# Full campaign (incremental save enabled)
results = run_experiment_campaign(
    config_ids=get_simulator_config_ids(),
    noise_ids=["IDEAL", "DEP-LOW", "DEP-MED"],
    num_runs=10,
    incremental_save=True,
)
```

### Statistical Analysis

```python
from qward.examples.papers.qft import (
    aggregate_session_results,
    analyze_scalability,
    compare_noise_models,
    analyze_qft_config_results,
)

# Load existing results
results = aggregate_session_results('20260114_125806')

# Analyze scalability
scalability = analyze_scalability(by_qubits, 'DEP-MED')
print(f"Half-life: {scalability['decay_fit']['half_life_qubits']:.1f} qubits")
```

## Test Modes

### Round-Trip Mode
- **Input**: basis state (e.g., "0110")
- **Process**: QFT → QFT⁻¹
- **Success**: measurement == input state
- **Use case**: Testing QFT reversibility

### Period Detection Mode
- **Input**: period value (e.g., period=4)
- **Process**: encode period → inverse QFT
- **Success**: measurement lands on expected peak
- **Use case**: Shor's algorithm building block

## Configuration Summary

| Experiment Type | Configs | Description |
|-----------------|---------|-------------|
| Scalability Roundtrip (SR) | 7 | 2-8 qubits |
| Scalability Period (SP) | 6 | 3-6 qubits, various periods |
| Period Variation (PV) | 7 | Fixed qubits, varying period |
| Input Variation (IV) | 5 | 4 qubits, different input states |
| **Total Simulator** | **25** | |
| Hardware Only | 4 | SR10, SR12, SP8-P4, SP8-P16 |

## Noise Models

| Model | Type | Avg Degradation |
|-------|------|-----------------|
| IDEAL | None | Baseline |
| READOUT | Measurement only | -6.2% |
| DEP-LOW | Light depolarizing | -6.2% |
| DEP-MED | Medium depolarizing | -21.1% |
| COMBINED | DEP-MED + READOUT | -25.2% |
| DEP-HIGH | Heavy depolarizing | -39.1% |

## Data Structure

```
data/
├── raw/
│   └── 20260114_125806/     # Session with 150 JSON files
│       ├── SR2_IDEAL.json
│       ├── SR2_DEP-LOW.json
│       └── ...
└── aggregated/
    └── campaign_summary_20260114_125806.json
```

## Experiment Parameters

| Parameter | Value |
|-----------|-------|
| Shots per job | 1024 |
| Runs per config | 10 |
| Optimization level | 0 (none) |
| Total experiments | 1,500 |
| Total shots | 1,536,000 |

## See Also

- `qft_experiment.md` - Full experiment design and results
- `../../qft.py` - QFT and QFTCircuitGenerator classes
- `../../phase_estimation.py` - PhaseEstimation classes
