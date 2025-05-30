# Technical Documentation

This document provides detailed technical information about QWARD's components and their implementation.

## Scanner

The `Scanner` class is the main orchestrator for metric calculation. It maintains a collection of metric calculators and coordinates their execution.

### Key Features
- Manages quantum circuit and execution results
- Coordinates multiple metric calculators
- Returns consolidated results as DataFrames
- Supports both class-based and instance-based metric addition

### Output Format
The `calculate_metrics()` method returns a dictionary where keys are metric calculator class names (or a modified name for `CircuitPerformance` with multiple jobs), and values are pandas DataFrames containing the metric data.
-   For `CircuitPerformance` with multiple jobs, two DataFrames are produced: `CircuitPerformance.individual_jobs` and `CircuitPerformance.aggregate`.

## Metric Calculators

QWARD provides several built-in metric calculators:

### QiskitMetrics
-   **Purpose**: Extracts metrics directly from QuantumCircuit objects
-   **Type**: `MetricsType.PRE_RUNTIME`
-   **ID**: `MetricsId.QISKIT`
-   **Constructor**: `QiskitMetrics(circuit: QuantumCircuit)`

### ComplexityMetrics
-   **Purpose**: Calculates circuit complexity metrics based on research literature
-   **Type**: `MetricsType.PRE_RUNTIME`
-   **ID**: `MetricsId.COMPLEXITY`
-   **Constructor**: `ComplexityMetrics(circuit: QuantumCircuit)`

### CircuitPerformance
-   **Purpose**: Calculates performance metrics from execution results
-   **Type**: `MetricsType.POST_RUNTIME`
-   **ID**: `MetricsId.CIRCUIT_PERFORMANCE`
-   **Constructor**: `CircuitPerformance(circuit: QuantumCircuit, job: Optional[Job]=None, jobs: Optional[List[Job]]=None, result: Optional[Dict]=None, success_criteria: Optional[Callable]=None)`

### Metric Types and IDs
-   `MetricsType(Enum)`: Defines when metrics can be calculated (`PRE_RUNTIME`, `POST_RUNTIME`).
-   `MetricsId(Enum)`: Defines unique IDs for strategy types (e.g., `QISKIT`, `COMPLEXITY`, `CIRCUIT_PERFORMANCE`).

## Usage Examples

### Basic Usage
```python
from qiskit import QuantumCircuit
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformance

# Create circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Method 1: Add metrics individually
scanner = Scanner(circuit=qc)
scanner.add_metric(QiskitMetrics(qc))
scanner.add_metric(ComplexityMetrics(qc))

# Method 2: Add metrics via constructor (using classes)
scanner = Scanner(circuit=qc, metrics=[QiskitMetrics, ComplexityMetrics, CircuitPerformance])

# Calculate metrics
results = scanner.calculate_metrics()
```

### Working with Execution Results
```python
from qiskit_aer import AerSimulator

# Execute circuit
sim = AerSimulator()
job = sim.run(qc, shots=1024)

# For CircuitPerformance, we need the Qiskit Job object (not just its result)
qiskit_job_obj = sim.run(qc) # Re-run to get a job object to pass to CircuitPerformance

# Define success criteria
def criteria(outcome):
    return outcome == "00"  # Success if both qubits measure 0

# Add CircuitPerformance with job
scanner.add_metric(CircuitPerformance(circuit=qc, job=qiskit_job_obj, success_criteria=criteria))

# Calculate all metrics
# print(results["CircuitPerformance.aggregate"])
```

### Alternative Approach with Result Object
```python
from qward import Result

# Create QWARD Result object
# from qward.metrics import CircuitPerformance

# qward_result = Result(job=job, counts=job.result().get_counts())

# # Create scanner with result
# scanner = Scanner(circuit=qc, result=qward_result)

# # Add metrics
# scanner.add_metric(CircuitPerformance(circuit=qc, job=job, success_criteria=criteria))

# metrics = scanner.calculate_metrics()
# print(metrics["CircuitPerformance.aggregate"])
```

## Schema Validation

QWARD includes comprehensive schema-based validation using Pydantic for enhanced data integrity and type safety.

### Key Benefits
- **Type Safety**: Automatic validation of data types and constraints
- **Business Rules**: Cross-field validation (e.g., error_rate = 1 - success_rate)
- **IDE Support**: Full autocomplete and type hints
- **API Documentation**: Automatic JSON schema generation

### Usage
Each metric calculator provides both traditional dictionary outputs and structured schema outputs:

```python
# Traditional approach
metrics_dict = calculator.get_metrics()

# Schema-based approach
structured_metrics = calculator.get_structured_metrics()
```

## Visualization System

QWARD includes a comprehensive visualization system for analyzing quantum circuit metrics.

### Architecture

The visualization system follows a clean architecture with the following components:

- **`BaseVisualizer`** (`qward.visualization.base.BaseVisualizer`):
  - Abstract base class for all visualizers
  - Handles common functionality like output directory management, plot styling, and saving/showing plots
  - Provides a consistent interface through the `create_plot()` abstract method

- **`CircuitPerformanceVisualizer`** (`qward.visualization.circuit_performance_visualizer.CircuitPerformanceVisualizer`):
  - Concrete visualizer for circuit performance metrics
  - Responsible for generating various plots related to the metrics produced by the `CircuitPerformance` strategy (e.g., success vs. error rates, fidelity, shot distributions, aggregate summaries).

- **`PlotConfig`** (`qward.visualization.base.PlotConfig`):
  - Configuration dataclass for plot appearance and saving options
  - Allows customization of figure size, DPI, color palettes, styles, etc.

- **`PlotStrategy`** (internal to `qward.visualization.circuit_performance_visualizer`):
  - An interface (abstract base class) defining a common contract for different plot generation algorithms within `CircuitPerformanceVisualizer`.
  - Concrete implementations handle specific plot types (e.g., success vs. error comparison, fidelity analysis, etc.)
  - `CircuitPerformanceVisualizer` delegates plotting tasks to these strategies.

- **Utility Functions** (within `qward.visualization.utils` if it exists):
  - Helper functions for common plotting tasks (e.g., applying consistent styling, adding value labels or summaries to plots). `CircuitPerformanceVisualizer` and its internal plot strategies would leverage such utilities to avoid code duplication and maintain consistency.

Visually, the core relationships (focusing on `CircuitPerformanceVisualizer`) can be represented as:

```mermaid
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
    
    class CircuitPerformanceVisualizer {
        +metrics_dict: Dict[str, DataFrame]
        +plot_success_error_comparison()
        +plot_fidelity_comparison()
        +plot_shot_distribution()
        +plot_aggregate_summary()
        +create_dashboard()
        +plot_all()
    }

    class PlotStrategy {
        <<interface>>
        +create_plot()*
    }

    BaseVisualizer <|-- CircuitPerformanceVisualizer
    BaseVisualizer --> PlotConfig : uses
    CircuitPerformanceVisualizer o--> PlotStrategy : uses (delegates to)
    
    note for BaseVisualizer "Abstract base class providing core visualization functionality"
    note for CircuitPerformanceVisualizer "Concrete implementation with direct plotting methods for simplicity"
    note for PlotConfig "Configuration dataclass for plot appearance and saving options"
```

### Key Components

#### `BaseVisualizer` (`qward.visualization.base.BaseVisualizer`)

Abstract base class that provides common functionality for all visualizers:

- **Output Management**: Handles output directory creation and file path management
- **Plot Configuration**: Integrates with `PlotConfig` for consistent styling
- **Save/Show Operations**: Provides `save_plot()` and `show_plot()` methods
- **Abstract Interface**: Defines `create_plot()` method that subclasses must implement

#### `CircuitPerformanceVisualizer` (`qward.visualization.circuit_performance_visualizer.CircuitPerformanceVisualizer`)

Specialized visualizer for `CircuitPerformance` metric outputs. It provides methods like:

- `plot_success_error_comparison()`: Creates bar charts comparing success vs. error rates across jobs
- `plot_fidelity_comparison()`: Visualizes fidelity metrics across different jobs
- `plot_shot_distribution()`: Shows distribution of successful vs. failed shots
- `plot_aggregate_summary()`: Creates summary plots of aggregate statistics
- `create_dashboard()`: Generates a comprehensive dashboard with multiple plot types
- `plot_all()`: Convenience method to generate all available plots

#### `PlotConfig` (`qward.visualization.base.PlotConfig`)

Configuration dataclass that allows customization of plot appearance:

```python
@dataclass
class PlotConfig:
    figsize: Tuple[int, int] = (10, 6)
    dpi: int = 300
    style: str = "default"
    color_palette: List[str] = field(default_factory=lambda: ["#1f77b4", "#ff7f0e", "#2ca02c"])
    save_format: str = "png"
    grid: bool = True
    alpha: float = 0.8
```

### Usage Examples

#### Basic Visualization
```python
from qward.visualization import CircuitPerformanceVisualizer

# Assuming you have metrics_dict from scanner.calculate_metrics()
visualizer = CircuitPerformanceVisualizer(metrics_dict, output_dir="plots")

# Generate all plots
figures = visualizer.plot_all(save=True, show=False)
```

#### Custom Configuration
```python
from qward.visualization import CircuitPerformanceVisualizer, PlotConfig

# Custom plot configuration
config = PlotConfig(
    figsize=(12, 8),
    dpi=150,
    style="seaborn",
    color_palette=["#ff6b6b", "#4ecdc4", "#45b7d1"],
    save_format="svg"
)

visualizer = CircuitPerformanceVisualizer(
    metrics_dict, 
    output_dir="custom_plots", 
    config=config
)

# Create specific plots
visualizer.plot_success_error_comparison(save=True)
visualizer.plot_fidelity_comparison(save=True)
```

#### Dashboard Creation
```python
# Create comprehensive dashboard
dashboard_fig = visualizer.create_dashboard(save=True, show=True)
```

### Integration with Scanner

The visualization system seamlessly integrates with the Scanner workflow:

```python
from qward import Scanner
from qward.metrics import CircuitPerformance
from qward.visualization import CircuitPerformanceVisualizer

# Calculate metrics
scanner = Scanner(circuit=circuit)
scanner.add_metric(CircuitPerformance(circuit=circuit, job=job))
metrics_dict = scanner.calculate_metrics()

# Create visualizations
visualizer = CircuitPerformanceVisualizer(metrics_dict)
visualizer.plot_all(save=True)
```

### Extensibility

The visualization system is designed for easy extension:

1. **Custom Visualizers**: Create new visualizers by inheriting from `BaseVisualizer`
2. **Custom Plot Types**: Add new methods to existing visualizers
3. **Custom Styling**: Define new `PlotConfig` presets for different use cases

For detailed usage examples and advanced features, see the [Visualization Guide](visualization_guide.md).

## Conclusion

QWARD provides a structured and extensible approach to quantum circuit analysis. By understanding the `Scanner`, the metric strategy system, and associated helper classes like `Result`, developers can gain significant insights into their quantum circuits and execution outcomes.

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
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformance

qc = QuantumCircuit(2,2); qc.h(0); qc.cx(0,1); qc.measure_all()

sim = AerSimulator()
job = sim.run(qc).result()
counts = job.get_counts()

# For CircuitPerformance, we need the Qiskit Job object (not just its result)
qiskit_job_obj = sim.run(qc) # Re-run to get a job object to pass to CircuitPerformance

qward_res = Result(job=qiskit_job_obj, counts=counts)

scanner = Scanner(circuit=qc, job=qiskit_job_obj, result=qward_res)
scanner.add_strategy(QiskitMetrics(qc))
scanner.add_strategy(ComplexityMetrics(qc))

def criteria(s): return s == '00' or s == '11' # GHZ state
scanner.add_strategy(CircuitPerformance(circuit=qc, job=qiskit_job_obj, success_criteria=criteria))

results = scanner.calculate_metrics()
# print(results["CircuitPerformance.aggregate"])
```

### Pattern 4: Using Standard Qiskit Runtime Services
```python
# from qiskit import QuantumCircuit
# from qiskit_ibm_runtime import QiskitRuntimeService
# from qward import Scanner, Result
# from qward.metrics import CircuitPerformance

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
# scanner.add_strategy(CircuitPerformance(circuit=qc, job=job, success_criteria=criteria))
# metrics = scanner.calculate_metrics()
# print(metrics["CircuitPerformance.aggregate"])
```

## Conclusion

QWARD provides a structured and extensible approach to quantum circuit analysis. By understanding the `Scanner`, the metric strategy system, and associated helper classes like `Result`, developers can gain significant insights into their quantum circuits and execution outcomes.

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

- **`CircuitPerformanceVisualizer`** (`qward.visualization.circuit_performance_visualizer.CircuitPerformanceVisualizer`):
    - A concrete visualizer inheriting directly from `BaseVisualizer`.
    - Responsible for generating various plots related to the metrics produced by the `CircuitPerformance` strategy (e.g., success vs. error rates, fidelity, shot distributions, aggregate summaries).
    - Internally, it uses the **Strategy pattern** to manage the generation of these different plot types. Each specific plot (e.g., fidelity comparison) is handled by a dedicated strategy class.

- **`PlotStrategy`** (internal to `qward.visualization.circuit_performance_visualizer`):
    - An interface (abstract base class) defining a common contract for different plot generation algorithms within `CircuitPerformanceVisualizer`.
    - Concrete strategies (e.g., `FidelityPlotStrategy`, `ShotDistributionPlotStrategy`) implement this interface to create specific charts.
    - `CircuitPerformanceVisualizer` delegates plotting tasks to these strategies.

- **(Conceptual) `MetricPlottingUtils`**:
    - While not a specific class shown in a high-level diagram, it's important to note that a utility class or module would ideally contain static helper methods for common tasks related to plotting metric data (e.g., extracting specific data from `metrics_dict`, validating DataFrame columns, adding standard value labels or summaries to plots). `CircuitPerformanceVisualizer` and its internal plot strategies would leverage such utilities to avoid code duplication and maintain consistency.

Visually, the core relationships (focusing on `CircuitPerformanceVisualizer`) can be represented as:

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

    class CircuitPerformanceVisualizer {
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
        #visualizer: CircuitPerformanceVisualizer
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

    BaseVisualizer <|-- CircuitPerformanceVisualizer
    BaseVisualizer --> PlotConfig : uses
    CircuitPerformanceVisualizer o--> PlotStrategy : uses (delegates to)

    PlotStrategy <|.. SuccessErrorPlotStrategy : implements
    PlotStrategy <|.. FidelityPlotStrategy : implements
    PlotStrategy <|.. ShotDistributionPlotStrategy : implements
    PlotStrategy <|.. AggregateSummaryPlotStrategy : implements
```

### Built-in Visualizers

#### `CircuitPerformanceVisualizer` (`qward.visualization.circuit_performance_visualizer.CircuitPerformanceVisualizer`)

Specialized visualizer for `CircuitPerformance` metric outputs. It provides methods like:
- `plot_all()`: Generates all standard individual plots.
- `create_dashboard()`: Creates a consolidated dashboard view.
- Individual plot methods (e.g., `plot_fidelity_comparison()`, `plot_shot_distribution()`).

It takes a `metrics_dict` (as produced by the `Scanner`) and an optional `PlotConfig` instance for customization.

For more details on usage, available plot types, and customization, refer to the [Visualization Guide](visualization_guide.md).
