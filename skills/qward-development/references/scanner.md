# Scanner Reference

The Scanner class is the main entry point for analyzing quantum circuits in QWARD.

## Basic Usage

```python
from qiskit import QuantumCircuit
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

# Create scanner
scanner = Scanner(circuit=circuit)

# Add strategies
scanner.add_strategy(QiskitMetrics(circuit))
scanner.add_strategy(ComplexityMetrics(circuit))

# Calculate metrics (returns Dict[str, pd.DataFrame])
results = scanner.calculate_metrics()
```

## Fluent API

```python
# One-liner with all pre-runtime metrics
Scanner(circuit).scan().summary().visualize(save=True, show=False)

# Chain methods
Scanner(circuit).add(QiskitMetrics).add(ComplexityMetrics).scan().summary()

# Pass strategy classes (auto-instantiated with circuit)
Scanner(circuit, strategies=[QiskitMetrics, ComplexityMetrics]).scan()
```

## Constructor

```python
Scanner(
    circuit: Optional[QuantumCircuit] = None,
    *,
    job: Optional[Union[AerJob, QiskitJob]] = None,
    strategies: Optional[list] = None,
)
```

**Parameters:**
- `circuit`: The quantum circuit to analyze
- `job`: Optional job for post-runtime metrics (CircuitPerformanceMetrics)
- `strategies`: List of metric classes or instances

## Methods

### add_strategy(strategy)
Add a metric strategy instance.

```python
scanner.add_strategy(QiskitMetrics(circuit))
```

### add(strategy, **kwargs) -> Scanner
Fluent method for adding strategies. Returns self for chaining.

```python
scanner.add(QiskitMetrics).add(ComplexityMetrics)
```

### calculate_metrics() -> Dict[str, pd.DataFrame]
Calculate all metrics and return DataFrames.

```python
results = scanner.calculate_metrics()
# results["QiskitMetrics"] -> DataFrame
# results["ComplexityMetrics"] -> DataFrame
# results["CircuitPerformance.individual_jobs"] -> DataFrame
# results["CircuitPerformance.aggregate"] -> DataFrame (if multiple jobs)
```

### scan(include_all_pre_runtime=True) -> ScanResult
Calculate metrics and return a ScanResult wrapper.

```python
result = scanner.scan()
result.summary()      # Print summary
result.visualize()    # Generate visualizations
result.to_dict()      # Get raw dictionary
```

### display_summary(metrics_dict=None)
Print a formatted summary of metrics.

## ScanResult Wrapper

ScanResult provides fluent post-processing:

```python
result = Scanner(circuit).scan()

# Fluent chaining
result.summary().visualize(save=True, show=False)

# Dictionary-like access
result["QiskitMetrics"]  # Returns DataFrame
result.keys()            # Available metrics
result.items()           # Iterate over metrics

# Convert to dict
raw_dict = result.to_dict()
```

### visualize() Parameters

```python
result.visualize(
    save=False,                    # Save plots to disk
    show=True,                     # Display plots
    output_dir="qward/examples/img",
    config=None,                   # PlotConfig instance
    selections=None,               # Dict for specific plots
)
```

## Pre-runtime vs Post-runtime Metrics

**Pre-runtime** (circuit only):
- QiskitMetrics
- ComplexityMetrics
- ElementMetrics
- StructuralMetrics
- BehavioralMetrics
- QuantumSpecificMetrics

**Post-runtime** (requires job):
- CircuitPerformanceMetrics
- DifferentialSuccessRate

```python
# Auto-add all pre-runtime
Scanner(circuit).scan(include_all_pre_runtime=True)

# Get pre-runtime strategy classes
from qward.metrics import get_all_pre_runtime_strategies
strategies = get_all_pre_runtime_strategies()
```

## Multiple Jobs Support

CircuitPerformanceMetrics supports multiple jobs:

```python
from qward.metrics import CircuitPerformanceMetrics

# Multiple jobs
perf = CircuitPerformanceMetrics(circuit=circuit)
perf.add_job(job1)
perf.add_job(job2)
perf.add_job(job3)

scanner.add_strategy(perf)
results = scanner.calculate_metrics()

# Returns two DataFrames:
results["CircuitPerformance.individual_jobs"]  # Per-job metrics
results["CircuitPerformance.aggregate"]        # Aggregate statistics
```
