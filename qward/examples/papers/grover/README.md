# Grover's Algorithm Experiment

This folder contains the experimental framework for evaluating Grover's algorithm performance using the QWARD library.

## Purpose

Systematic study of Grover's algorithm behavior under different conditions:
- Circuit scalability (2-8 qubits)
- Marked state configurations
- Noise models
- Success metric definitions

## Files

| File | Description |
|------|-------------|
| `grover_experiment.md` | Experiment design document |
| `grover_experiment.py` | Main experiment runner |
| `grover_configs.py` | Circuit configurations |
| `grover_success_metrics.py` | Success metric implementations |
| `grover_statistical_analysis.py` | Statistical analysis functions |
| `grover_visualization.py` | Plotting functions |

## Data Structure

```
data/
└── simulator/
    ├── raw/          # Individual experiment records (JSON)
    └── aggregated/   # Per-config aggregates (CSV)
```

## Quick Start

```python
from grover_experiment import run_experiment

# Run a single configuration
results = run_experiment(
    config_id="S3-1",
    noise_model="IDEAL",
    num_runs=10
)
```

## Parameters

- **Shots per job**: 1024
- **Runs per config**: 10
- **Optimization level**: 0 (none)

