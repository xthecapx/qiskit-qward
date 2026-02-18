---
name: qward-development
description: QWARD library for quantum circuit analysis, metrics extraction, performance evaluation, and visualization. Use when analyzing quantum circuits with Scanner, extracting metrics (QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics, ElementMetrics, StructuralMetrics, BehavioralMetrics, QuantumSpecificMetrics), visualizing results with the Visualizer API, implementing custom metric strategies, running experiments with BaseExperimentRunner, using noise model presets (IBM Heron, Rigetti Ankaa), or extending QWARD with custom metrics. Covers the Strategy pattern architecture, Pydantic schema validation, fluent API chaining, and the type-safe visualization system.
---

# QWARD Development

## Overview

QWARD (Quantum Circuit Analysis and Runtime Development) is a Python library for analyzing quantum circuits and their execution results. It provides a comprehensive framework for extracting metrics, validating data with Pydantic schemas, and visualizing analysis results.

**Key capabilities:**
- **Scanner**: Strategy pattern-based circuit analysis with fluent API
- **8 Metric Types**: QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics, ElementMetrics, StructuralMetrics, BehavioralMetrics, QuantumSpecificMetrics, DifferentialSuccessRate
- **Schema Validation**: Pydantic-based type safety with IDE autocomplete
- **Visualization**: Type-safe constants, granular plot control, dashboards
- **Algorithms**: Grover, QFT, Phase Estimation, Teleportation implementations
- **Executor**: Unified interface for simulators, IBM Quantum, AWS Braket
- **Experiment Framework**: Campaign runners with statistical analysis

## Quick Start

```python
from qiskit import QuantumCircuit
from qward import Scanner

# One-liner: analyze with all pre-runtime metrics
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

Scanner(circuit).scan().summary().visualize(save=True, show=False)
```

## Core Architecture

QWARD uses the **Strategy Pattern** for extensible metric calculation:

```
Scanner (Context)
    │
    ├── QiskitMetrics ──────► QiskitMetricsSchema
    ├── ComplexityMetrics ──► ComplexityMetricsSchema
    ├── CircuitPerformanceMetrics ──► CircuitPerformanceSchema
    ├── ElementMetrics ─────► ElementMetricsSchema
    ├── StructuralMetrics ──► StructuralMetricsSchema
    ├── BehavioralMetrics ──► BehavioralMetricsSchema
    └── QuantumSpecificMetrics ► QuantumSpecificMetricsSchema
```

## Reference Documentation

Load these as needed based on your task:

- **`references/scanner.md`** - Scanner class, fluent API, ScanResult wrapper
- **`references/metrics.md`** - All 8 metric types with schema details
- **`references/visualization.md`** - Visualizer, type-safe constants, dashboards
- **`references/algorithms.md`** - Grover, QFT, Phase Estimation, Teleportation
- **`references/executor.md`** - QuantumCircuitExecutor, noise models, experiments
- **`references/custom-metrics.md`** - Creating custom MetricCalculator subclasses
- **`references/development.md`** - Code quality, testing, contributing

## Workflow Decision Guide

- Analyze a quantum circuit → `references/scanner.md`
- Extract specific metrics → `references/metrics.md`
- Visualize analysis results → `references/visualization.md`
- Use built-in algorithms → `references/algorithms.md`
- Run on hardware/simulators → `references/executor.md`
- Create custom metrics → `references/custom-metrics.md`
- Contribute to QWARD → `references/development.md`

## Common Patterns

### Pattern 1: Full Circuit Analysis

```python
from qiskit import QuantumCircuit
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

# Add specific strategies
scanner = Scanner(circuit)
scanner.add_strategy(QiskitMetrics(circuit))
scanner.add_strategy(ComplexityMetrics(circuit))

# Get DataFrames
results = scanner.calculate_metrics()
print(results["QiskitMetrics"])
print(results["ComplexityMetrics"])
```

### Pattern 2: Fluent API with All Pre-Runtime Metrics

```python
from qward import Scanner

# Auto-adds all pre-runtime metrics, displays summary, saves visualizations
Scanner(circuit).scan().summary().visualize(save=True, show=False)

# Or pick specific metrics
from qward.metrics import QiskitMetrics, ComplexityMetrics
results = Scanner(circuit, strategies=[QiskitMetrics, ComplexityMetrics]).scan()
```

### Pattern 3: Schema-Based Type-Safe Access

```python
from qward.metrics import ComplexityMetrics

complexity = ComplexityMetrics(circuit)
metrics = complexity.get_metrics()  # Returns ComplexityMetricsSchema

# Full IDE autocomplete and type safety
print(f"Gate count: {metrics.gate_based_metrics.gate_count}")
print(f"T-gate count: {metrics.gate_based_metrics.t_count}")
print(f"Circuit efficiency: {metrics.advanced_metrics.circuit_efficiency:.3f}")
```

### Pattern 4: Performance Analysis with Custom Success Criteria

```python
from qiskit_aer import AerSimulator
from qward.metrics import CircuitPerformanceMetrics

# Run circuit
simulator = AerSimulator()
job = simulator.run(circuit, shots=1024)

# Custom success criteria for Bell state
def bell_success(result: str) -> bool:
    return result.replace(" ", "") in ["00", "11"]

perf = CircuitPerformanceMetrics(circuit=circuit, job=job, success_criteria=bell_success)
metrics = perf.get_metrics()
print(f"Success rate: {metrics.success_metrics.success_rate:.1%}")
```

### Pattern 5: Type-Safe Visualization

```python
from qward.visualization import Visualizer
from qward.visualization.constants import Metrics, Plots

visualizer = Visualizer(scanner=scanner, output_dir="analysis")

# Generate specific plots with type-safe constants
# Note: Metrics use UPPER_CASE, Plots use PascalCase.UPPER_CASE
visualizer.generate_plots({
    Metrics.QISKIT: [Plots.Qiskit.CIRCUIT_STRUCTURE, Plots.Qiskit.GATE_DISTRIBUTION],
    Metrics.COMPLEXITY: [Plots.Complexity.COMPLEXITY_RADAR]
}, save=True, show=False)

# Or generate all plots for a metric
visualizer.generate_plots({Metrics.QISKIT: None}, save=True)

# Create comprehensive dashboards
visualizer.create_dashboard(save=True, show=False)
```

### Pattern 6: Experiment Campaign

```python
from qward.algorithms import BaseExperimentRunner, get_preset_noise_config

class MyExperimentRunner(BaseExperimentRunner):
    # Implement abstract methods for your algorithm
    pass

runner = MyExperimentRunner()
results = runner.run_campaign(
    config_ids=["config-1", "config-2"],
    noise_ids=["IDEAL", "IBM-HERON-R2", "RIGETTI-ANKAA3"],
    num_runs=10
)
```

## Available Metrics Summary

| Metric Class | Type | Key Measurements |
|-------------|------|------------------|
| `QiskitMetrics` | Pre-runtime | depth, width, size, gate counts, connectivity |
| `ComplexityMetrics` | Pre-runtime | T-count, CNOT-count, circuit volume, efficiency |
| `ElementMetrics` | Pre-runtime | gate frequencies, operand analysis, parameters |
| `StructuralMetrics` | Pre-runtime | layering, graph connectivity, topology |
| `BehavioralMetrics` | Pre-runtime | state evolution, interference patterns |
| `QuantumSpecificMetrics` | Pre-runtime | entanglement, non-classicality |
| `CircuitPerformanceMetrics` | Post-runtime | success rate, error rate, statistics |
| `DifferentialSuccessRate` | Post-runtime | DSR analysis, ideal vs noisy comparison |

## Noise Model Presets

```python
from qward.algorithms import get_preset_noise_config, NoiseModelGenerator

# Available presets (hardware-calibrated)
# "IBM-HERON-R1", "IBM-HERON-R2", "IBM-HERON-R3", "RIGETTI-ANKAA3"

config = get_preset_noise_config("IBM-HERON-R2")
noise_model = NoiseModelGenerator.create_from_config(config)
```

## Best Practices

1. **Use Scanner for analysis** - Provides consistent DataFrame output
2. **Use schemas for type safety** - Get IDE autocomplete and validation
3. **Use fluent API for quick analysis** - `Scanner(circuit).scan().summary().visualize()`
4. **Use type-safe constants for visualization** - Prevents typos, enables autocomplete
5. **Define success criteria** - Required for meaningful CircuitPerformanceMetrics
6. **Use noise presets** - Hardware-calibrated for realistic simulation
7. **Run experiments systematically** - Use BaseExperimentRunner for campaigns

## Integration with Qiskit

QWARD integrates seamlessly with Qiskit:

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics

# Build circuit with Qiskit
circuit = QuantumCircuit(3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)
circuit.measure_all()

# Analyze with QWARD (pre-runtime)
scanner = Scanner(circuit)
scanner.add_strategy(QiskitMetrics(circuit))
scanner.add_strategy(ComplexityMetrics(circuit))

# Execute with Qiskit Aer
simulator = AerSimulator()
job = simulator.run(circuit, shots=1024)

# Analyze performance (post-runtime)
scanner.add_strategy(CircuitPerformanceMetrics(circuit=circuit, job=job))

# Get all results
results = scanner.calculate_metrics()
```
