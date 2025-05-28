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

QWARD includes a comprehensive visualization module (`qward.visualization`) for creating publication-quality plots of quantum circuit metrics. The visualization module follows a structured approach that integrates seamlessly with the metric calculation system.

### Architecture

The visualization system is built on the following key components:

- **`PlotConfig`** (`qward.visualization.base.PlotConfig`):
    - A dataclass holding all plot appearance and saving configurations.
    - Parameters include `figsize`, `dpi`, `style` (e.g., "default", "quantum", "minimal"), `color_palette`, `save_format` (e.g., "png", "svg"), `grid`, and `alpha`.

- **`BaseVisualizer`** (`qward.visualization.base.BaseVisualizer`):
    - An abstract base class for all visualizers.
    - Handles common setup: output directory creation, applying plot styles using `PlotConfig`.
    - Provides `save_plot()` and `show_plot()` utility methods.
    - Requires subclasses to implement the `create_plot()` method for their specific visualization logic.

- **`SuccessRateVisualizer`** (`qward.visualization.success_rate_visualizer.SuccessRateVisualizer`):
    - A concrete visualizer inheriting directly from `BaseVisualizer`.
    - Responsible for generating various plots related to the metrics produced by the `SuccessRate` strategy (e.g., success vs. error rates, fidelity, shot distributions, aggregate summaries).
    - Internally, it uses the **Strategy pattern** to manage the generation of these different plot types. Each specific plot (e.g., fidelity comparison) is handled by a dedicated strategy class.

- **`PlotStrategy`** (internal to `qward.visualization.success_rate_visualizer`):
    - An interface (abstract base class) defining a common contract for different plot generation algorithms within `SuccessRateVisualizer`.
    - Concrete strategies (e.g., `FidelityPlotStrategy`, `ShotDistributionPlotStrategy`) implement this interface to create specific charts.
    - `SuccessRateVisualizer` delegates plotting tasks to these strategies.

- **(Conceptual) `MetricPlottingUtils`**:
    - While not a specific class shown in a high-level diagram, it's important to note that a utility class or module would ideally contain static helper methods for common tasks related to plotting metric data (e.g., extracting specific data from `metrics_dict`, validating DataFrame columns, adding standard value labels or summaries to plots). `SuccessRateVisualizer` and its internal plot strategies would leverage such utilities to avoid code duplication and maintain consistency.

Visually, the core relationships (focusing on `SuccessRateVisualizer`) can be represented as:

```{mermaid}
classDiagram
    class BaseVisualizer {
        <<abstract>>
        +output_dir: str
        +config: PlotConfig
        +save_plot(fig, filename)
        +show_plot(fig)
        +create_plot()*
    }

    class PlotConfig {
        +figsize: Tuple[int, int]
        +dpi: int
        +style: str
        +color_palette: List[str]
        +save_format: str
        +grid: bool
        +alpha: float
    }

    class SuccessRateVisualizer {
        +metrics_dict: Dict[str, DataFrame]
        # +_is_dashboard_context: bool
        +plot_success_error_comparison()
        +plot_fidelity_comparison()
        +plot_shot_distribution()
        +plot_aggregate_summary()
        +create_dashboard()
        +plot_all()
    }

    class PlotStrategy {
        <<Interface>>
        #visualizer: SuccessRateVisualizer
        #config: PlotConfig
        +plot(ax: Axes)*
    }

    class SuccessErrorPlotStrategy {
        +plot(ax: Axes)
    }
    class FidelityPlotStrategy {
        +plot(ax: Axes)
    }
    class ShotDistributionPlotStrategy {
        +plot(ax: Axes)
    }
    class AggregateSummaryPlotStrategy {
        +plot(ax: Axes)
    }
    
    note for PlotStrategy "Each concrete strategy implements a specific plot type (e.g., success vs error, fidelity)."

    BaseVisualizer <|-- SuccessRateVisualizer
    BaseVisualizer --> PlotConfig : uses
    SuccessRateVisualizer o--> PlotStrategy : uses (delegates to)

    PlotStrategy <|.. SuccessErrorPlotStrategy : implements
    PlotStrategy <|.. FidelityPlotStrategy : implements
    PlotStrategy <|.. ShotDistributionPlotStrategy : implements
    PlotStrategy <|.. AggregateSummaryPlotStrategy : implements
```

### Built-in Visualizers

#### `SuccessRateVisualizer` (`qward.visualization.success_rate_visualizer.SuccessRateVisualizer`)

Specialized visualizer for `SuccessRate` metric outputs. It provides methods like:
- `plot_all()`: Generates all standard individual plots.
- `create_dashboard()`: Creates a consolidated dashboard view.
- Individual plot methods (e.g., `plot_fidelity_comparison()`, `plot_shot_distribution()`).

It takes a `metrics_dict` (as produced by the `Scanner`) and an optional `PlotConfig` instance for customization.

For more details on usage, available plot types, and customization, refer to the [Visualization Guide](visualization_guide.md).
