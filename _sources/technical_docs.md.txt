# Technical Documentation

This document provides detailed technical information about the Qward library's architecture, components, and advanced usage patterns. For a structural overview and class diagrams, please refer to `docs/architecture.md`.

## Core Architecture

Qward is designed around a `Scanner` that applies various metric strategy objects to Qiskit `QuantumCircuit`s and their execution results.

### 1. `Scanner` (`qward.Scanner`)

The `Scanner` is the central class for orchestrating analysis. 
-   **Initialization**: `Scanner(circuit: Optional[QuantumCircuit], *, job: Optional[Union[AerJob, QiskitJob]], result: Optional[qward.Result], strategies: Optional[list])`
    -   It takes an optional Qiskit `QuantumCircuit`.
    -   Optionally, a Qiskit `Job` (from Aer or a provider) and/or a `qward.Result` object (which wraps execution counts and metadata).
    -   Optionally, a list of metric strategy *classes* or *instances* can be provided at initialization.
-   **Adding Strategies**: Strategies are added using `scanner.add_strategy(strategy_instance: MetricCalculator)`.
-   **Calculating Metrics**: `scanner.calculate_metrics() -> Dict[str, pd.DataFrame]`
    -   This method iterates through all added strategies, calls their `get_metrics()` method, and compiles the results into a dictionary. The keys are typically the strategy class names (or a modified name for `SuccessRate` with multiple jobs), and values are pandas DataFrames containing the metric data.
    -   For `SuccessRate` with multiple jobs, two DataFrames are produced: `SuccessRate.individual_jobs` and `SuccessRate.aggregate`.

### 2. Metric Strategy System (`qward.metrics`)

Metric strategies are classes responsible for specific calculations or data extraction.

-   **Base Class**: `qward.metrics.base_metric.MetricCalculator`
    -   All metric strategies inherit from this abstract base class.
    -   Requires implementation of:
        -   `_get_metric_type() -> MetricsType`: Returns `MetricsType.PRE_RUNTIME` (if only circuit is needed) or `MetricsType.POST_RUNTIME` (if job/result data is needed).
        -   `_get_metric_id() -> MetricsId`: Returns a unique `MetricsId` enum value.
        -   `is_ready() -> bool`: Checks if the strategy has sufficient data to be calculated.
        -   `get_metrics() -> Dict[str, Any]`: Performs the calculation and returns a dictionary of results.
    -   The constructor `MetricCalculator(circuit: QuantumCircuit)` stores the circuit and initializes `_metric_type` and `_id` by calling the abstract getter methods.

-   **Built-in Strategies**:
    -   **`QiskitMetrics` (`qward.metrics.qiskit_metrics.QiskitMetrics`)**
        -   Type: `PRE_RUNTIME`
        -   ID: `MetricsId.QISKIT`
        -   Extracts data directly from the `QuantumCircuit` object (e.g., depth, width, operations count, instruction details, basic scheduling info if available).
        -   `get_metrics()` returns a flattened dictionary of these properties.

    -   **`ComplexityMetrics` (`qward.metrics.complexity_metrics.ComplexityMetrics`)**
        -   Type: `PRE_RUNTIME`
        -   ID: `MetricsId.COMPLEXITY`
        -   Calculates a comprehensive set of complexity metrics based on D. Shami's "Character Complexity" paper, including gate-based, entanglement, standardized, advanced, and derived metrics.
        -   Also includes Quantum Volume estimation (`standard_quantum_volume`, `enhanced_quantum_volume`, and contributing factors).
        -   `get_metrics()` returns a dictionary where top-level keys correspond to these categories (e.g., "gate_based_metrics", "quantum_volume"), and values are sub-dictionaries of specific metrics.

    -   **`SuccessRate` (`qward.metrics.success_rate.SuccessRate`)**
        -   Type: `POST_RUNTIME`
        -   ID: `MetricsId.SUCCESS_RATE`
        -   Constructor: `SuccessRate(circuit: QuantumCircuit, job: Optional[Job]=None, jobs: Optional[List[Job]]=None, result: Optional[Dict]=None, success_criteria: Optional[Callable]=None)`
            -   Requires a `QuantumCircuit`.
            -   Needs execution data, provided via a single `job`, a list of `jobs`, or a `result` dictionary (though `job`/`jobs` are preferred for direct access to Qiskit's result objects).
            -   `success_criteria` is a callable `(bitstring: str) -> bool` that defines a successful outcome. Defaults to `bitstring == "0"`.
        -   `add_job(job_or_jobs)`: Allows adding more jobs after instantiation.
        -   `get_metrics()`: Calculates success rate, error rate, fidelity, total shots, successful shots. If multiple jobs are provided, it returns individual job stats and aggregate stats.

-   **Strategy Types (`qward.metrics.types`)**
    -   `MetricsId(Enum)`: Defines unique IDs for strategy types (e.g., `QISKIT`, `COMPLEXITY`, `SUCCESS_RATE`).
    -   `MetricsType(Enum)`: Defines when a strategy is calculated (`PRE_RUNTIME`, `POST_RUNTIME`).

-   **Default Strategies (`qward.metrics.defaults`)**
    -   `get_default_strategies() -> List[Type[MetricCalculator]]`: Returns a list of default metric strategy classes (`[QiskitMetrics, ComplexityMetrics, SuccessRate]`).

### 3. `Result` (`qward.Result`)

A utility class for encapsulating job execution data.
-   Constructor: `Result(job: Optional[Job]=None, counts: Optional[Dict]=None, metadata: Optional[Dict]=None)`.
-   Primarily stores Qiskit `Job` object, `counts` (measurement outcomes), and `metadata`.
-   Provides `save()` and `load()` methods for serialization to/from JSON.
-   `update_from_job()`: If a job is stored, this method (re-)populates counts and basic metadata from `job.result()`.




## Circuit Complexity Metrics (via `ComplexityMetrics`)

The `ComplexityMetrics` class, when used with the `Scanner`, provides a detailed breakdown of circuit complexity. The output DataFrame from `scanner.calculate_metrics()["ComplexityMetrics"]` will have columns corresponding to flattened metric names (e.g., `gate_based_metrics.gate_count`, `entanglement_metrics.entangling_gate_density`).

Refer to the `qward/metrics/complexity_metrics.py` source or its docstrings for a full list of all calculated metrics. The categories include:
-   Gate-based Metrics
-   Entanglement Metrics
-   Standardized Metrics
-   Advanced Metrics
-   Derived Metrics

Example access:
```python
# results = scanner.calculate_metrics()
# complexity_df = results.get("ComplexityMetrics")
# if complexity_df is not None:
#     gate_count = complexity_df['gate_based_metrics.gate_count'].iloc[0]
#     t_count = complexity_df['gate_based_metrics.t_count'].iloc[0]
#     # ... and so on
```

## Quantum Volume Estimation (via `ComplexityMetrics`)

The `ComplexityMetrics` class also performs Quantum Volume (QV) estimation based on the circuit's structure.

-   **Standard Quantum Volume**: `quantum_volume.standard_quantum_volume` (2^effective_depth)
-   **Enhanced Quantum Volume**: `quantum_volume.enhanced_quantum_volume` (adjusts standard QV by factors like square ratio, density, multi-qubit ratio, connectivity).
-   **Effective Depth**: `quantum_volume.effective_depth`
-   **Contributing Factors**: `quantum_volume.factors.square_ratio`, etc.

Example access from `ComplexityMetrics` DataFrame:
```python
# enhanced_qv = complexity_df['quantum_volume.enhanced_quantum_volume'].iloc[0]
```

## Technical Guidelines for Custom Strategies

When creating a new metric strategy by inheriting from `qward.metrics.base_metric.MetricCalculator`:
1.  **Implement Abstract Methods**: `_get_metric_type`, `_get_metric_id`, `is_ready`, `get_metrics`.
2.  **Strategy ID**: If your strategy is conceptually new, consider if `MetricsId` enum in `qward.metrics.types` needs to be extended. For purely external or highly specific custom strategies, you might need a strategy if modifying the core enum is not desired (though the base class expects a `MetricsId` enum member).
3.  **Data Requirements**: Clearly define if your strategy is `PRE_RUNTIME` (only needs circuit) or `POST_RUNTIME` (needs job/results). If `POST_RUNTIME`, your `__init__` should accept job(s) or result data, and `is_ready` should check for their presence.
4.  **Return Format**: `get_metrics()` must return a dictionary where keys are string metric names and values are the calculated metric values.

## API Flow and Usage Patterns

### Pattern 1: Basic Circuit Analysis (No Execution Results)
```python
from qiskit import QuantumCircuit
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics

qc = QuantumCircuit(2); qc.h(0); qc.cx(0,1)

scanner = Scanner(circuit=qc)
scanner.add_strategy(QiskitMetrics(qc))
scanner.add_strategy(ComplexityMetrics(qc))

results = scanner.calculate_metrics()
# print(results["QiskitMetrics"])
# print(results["ComplexityMetrics"])
```

### Pattern 2: Using Constructor with Strategies
```python
from qiskit import QuantumCircuit
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics

qc = QuantumCircuit(2); qc.h(0); qc.cx(0,1)

# Using strategy classes (instantiated automatically)
scanner = Scanner(circuit=qc, strategies=[QiskitMetrics, ComplexityMetrics])

# Using strategy instances
qm = QiskitMetrics(qc)
cm = ComplexityMetrics(qc)
scanner = Scanner(circuit=qc, strategies=[qm, cm])

results = scanner.calculate_metrics()
```

### Pattern 3: Analysis with Execution Results
```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner, Result
from qward.metrics import QiskitMetrics, ComplexityMetrics, SuccessRate

qc = QuantumCircuit(2,2); qc.h(0); qc.cx(0,1); qc.measure_all()

sim = AerSimulator()
job = sim.run(qc).result()
counts = job.get_counts()

# For SuccessRate, we need the Qiskit Job object (not just its result)
qiskit_job_obj = sim.run(qc) # Re-run to get a job object to pass to SuccessRate

qward_res = Result(job=qiskit_job_obj, counts=counts)

scanner = Scanner(circuit=qc, job=qiskit_job_obj, result=qward_res)
scanner.add_strategy(QiskitMetrics(qc))
scanner.add_strategy(ComplexityMetrics(qc))

def criteria(s): return s == '00' or s == '11' # GHZ state
scanner.add_strategy(SuccessRate(circuit=qc, job=qiskit_job_obj, success_criteria=criteria))

results = scanner.calculate_metrics()
# print(results["SuccessRate.aggregate"])
```

### Pattern 4: Using Standard Qiskit Runtime Services
```python
# from qiskit import QuantumCircuit
# from qiskit_ibm_runtime import QiskitRuntimeService
# from qward import Scanner, Result
# from qward.metrics import SuccessRate

# qc = QuantumCircuit(2,2); qc.h(0); qc.cx(0,1); qc.measure_all()

# Note: Ensure IBM Quantum credentials are set up
# service = QiskitRuntimeService()
# backend = service.backend('ibmq_qasm_simulator')
# job = service.run(qc, backend=backend)
# result = job.result()
# counts = result.get_counts()

# qward_result = Result(job=job, counts=counts)
# scanner = Scanner(circuit=qc, job=job, result=qward_result)
# def criteria(s): return s == '00' or s == '11'
# scanner.add_strategy(SuccessRate(circuit=qc, job=job, success_criteria=criteria))
# metrics = scanner.calculate_metrics()
# print(metrics["SuccessRate.aggregate"])
```

## Conclusion

Qward provides a structured and extensible approach to quantum circuit analysis. By understanding the `Scanner`, the metric strategy system, and associated helper classes like `Result`, developers can gain significant insights into their quantum circuits and execution outcomes.

## Visualization System

QWARD includes a comprehensive visualization module (`qward.visualization`) for creating publication-quality plots of quantum circuit metrics.

### Architecture

The visualization system follows a hierarchical class structure:

1. **`BaseVisualizer`** (`qward.visualization.base.BaseVisualizer`)
   - Abstract base class for all visualizers
   - Manages output directories and plot configuration
   - Provides `save_plot()` and `show_plot()` methods
   - Requires implementation of `create_plot()` method

2. **`MetricVisualizer`** (`qward.visualization.base.MetricVisualizer`)
   - Abstract class for metric-specific visualizers
   - Extends `BaseVisualizer` with metric data handling
   - Constructor: `MetricVisualizer(metrics_dict: Dict[str, pd.DataFrame], output_dir: str = "img", config: Optional[PlotConfig] = None)`
   - Provides helper methods: `get_metric_data()`, `validate_columns()`, `add_value_labels()`

3. **`PlotConfig`** (`qward.visualization.base.PlotConfig`)
   - Dataclass for plot configuration
   - Parameters:
     - `figsize: Tuple[int, int] = (10, 6)`
     - `dpi: int = 300`
     - `style: str = "default"` (options: "default", "quantum", "minimal")
     - `color_palette: List[str] = None` (defaults to ColorBrewer-inspired palette)
     - `save_format: str = "png"` (options: "png", "svg", "pdf", "eps")
     - `grid: bool = True`
     - `alpha: float = 0.7`

### Built-in Visualizers

#### `SuccessRateVisualizer` (`qward.visualization.success_rate_visualizer.SuccessRateVisualizer`)

Specialized visualizer for `SuccessRate` metric outputs.

**Constructor**: `SuccessRateVisualizer(metrics_dict: Dict[str, pd.DataFrame], output_dir: str = "img", config: Optional[PlotConfig] = None)`

**Methods**:
- `plot_all(save: bool = True, show: bool = True) -> List[Figure]`: Creates all available plots
- `create_dashboard(save: bool = True, show: bool = True) -> Figure`: Creates a comprehensive dashboard
- `plot_success_rate_comparison(save: bool = True, show: bool = True) -> Figure`: Bar chart of success rates
- `plot_error_rate_comparison(save: bool = True, show: bool = True) -> Figure`: Bar chart of error rates
- `plot_fidelity_comparison(save: bool = True, show: bool = True) -> Figure`: Bar chart of fidelity values
- `plot_shot_distribution(save: bool = True, show: bool = True) -> Figure`: Stacked bar chart of measurement outcomes
- `plot_aggregate_summary(save: bool = True, show: bool = True) -> Figure`: Summary statistics visualization

### Usage Examples

#### Basic Visualization
```python
from qward.visualization import SuccessRateVisualizer

# After calculating metrics
metrics_dict = scanner.calculate_metrics()

# Create visualizer
visualizer = SuccessRateVisualizer(metrics_dict, output_dir="img/analysis")

# Generate all plots
figures = visualizer.plot_all(save=True, show=False)
```

#### Custom Configuration
```python
from qward.visualization import SuccessRateVisualizer, PlotConfig

# Custom configuration
config = PlotConfig(
    figsize=(12, 8),
    style="quantum",
    dpi=150,
    save_format="svg",
    color_palette=["#1f77b4", "#ff7f0e", "#2ca02c"]
)

# Create visualizer with custom config
visualizer = SuccessRateVisualizer(
    metrics_dict,
    output_dir="img/custom",
    config=config
)

# Create specific plots
visualizer.plot_success_rate_comparison(save=True)
visualizer.plot_fidelity_comparison(save=True)
```

#### Dashboard Creation
```python
# Create comprehensive dashboard
visualizer.create_dashboard(save=True, show=True)
```

### Creating Custom Visualizers

To create a custom visualizer for your own metrics:

```python
from qward.visualization.base import MetricVisualizer
import matplotlib.pyplot as plt

class MyCustomVisualizer(MetricVisualizer):
    def create_plot(self) -> plt.Figure:
        """Create the main plot."""
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Get your metric data
        data = self.get_metric_data("MyCustomMetric")
        if data is None:
            raise ValueError("MyCustomMetric data not found")
        
        # Create visualization
        # ... your plotting code ...
        
        return fig
    
    def plot_custom_analysis(self, save=True, show=True) -> plt.Figure:
        """Create a custom analysis plot."""
        fig = self.create_plot()
        
        if save:
            self.save_plot(fig, "custom_analysis")
        if show:
            self.show_plot(fig)
            
        return fig
```

### Integration with Scanner

The visualization system is designed to work seamlessly with Scanner output:

```python
# Complete workflow
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner
from qward.metrics import SuccessRate
from qward.visualization import SuccessRateVisualizer

# Create and run circuit
circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

simulator = AerSimulator()
job = simulator.run(circuit, shots=1024)

# Calculate metrics
scanner = Scanner(circuit=circuit)
success_rate = SuccessRate(circuit=circuit)
success_rate.add_job(job)
scanner.add_strategy(success_rate)

metrics_dict = scanner.calculate_metrics()

# Visualize results
visualizer = SuccessRateVisualizer(metrics_dict)
visualizer.plot_all(save=True)
```

### Best Practices

1. **Output Organization**: Use descriptive output directories
   ```python
   output_dir = f"img/{experiment_name}/{timestamp}"
   visualizer = SuccessRateVisualizer(metrics_dict, output_dir=output_dir)
   ```

2. **Batch Processing**: For many plots, use `show=False` to avoid display overhead
   ```python
   figures = visualizer.plot_all(save=True, show=False)
   ```

3. **Publication Quality**: Use high DPI and vector formats
   ```python
   config = PlotConfig(dpi=300, save_format="svg")
   ```

4. **Memory Management**: Close figures after batch processing
   ```python
   import matplotlib.pyplot as plt
   figures = visualizer.plot_all(save=True, show=False)
   plt.close('all')  # Free memory
   ```

## Conclusion

Qward provides a structured and extensible approach to quantum circuit analysis. By understanding the `Scanner`, the metric strategy system, associated helper classes like `Result`, and the visualization system, developers can gain significant insights into their quantum circuits and execution outcomes while presenting results in a clear, professional manner.
