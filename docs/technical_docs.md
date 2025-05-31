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

# For CircuitPerformance, we need the Qiskit Job object
qiskit_job_obj = sim.run(qc) # Re-run to get a job object to pass to CircuitPerformance

# Define success criteria
def criteria(outcome):
    return outcome == "00"  # Success if both qubits measure 0

# Add CircuitPerformance with job
scanner.add_metric(CircuitPerformance(circuit=qc, job=qiskit_job_obj, success_criteria=criteria))

# Calculate all metrics
# print(results["CircuitPerformance.aggregate"])
```

### Alternative Approach with Job Object
```python
from qiskit_aer import AerSimulator

# Execute circuit
sim = AerSimulator()
job = sim.run(qc, shots=1024)

# For CircuitPerformance, we need the Qiskit Job object
qiskit_job_obj = sim.run(qc) # Re-run to get a job object to pass to CircuitPerformance

# Define success criteria
def criteria(outcome):
    return outcome == "00"  # Success if both qubits measure 0

# Add CircuitPerformance with job
scanner.add_metric(CircuitPerformance(circuit=qc, job=qiskit_job_obj, success_criteria=criteria))

# Calculate all metrics
# print(results["CircuitPerformance.aggregate"])
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

QWARD includes a comprehensive visualization module (`qward.visualization`) for creating publication-quality plots of quantum circuit metrics. The visualization module follows a structured approach that integrates seamlessly with the metric calculation system.

### Architecture

The visualization system is built on the following key components:

- **`PlotConfig`** (`qward.visualization.base.PlotConfig`):
    - A dataclass holding all plot appearance and saving configurations.
    - Parameters include `figsize`, `dpi`, `style` (e.g., "default", "quantum", "minimal"), `color_palette`, `save_format` (e.g., "png", "svg"), `grid`, and `alpha`.

- **`BaseVisualizer`** (`qward.visualization.base.BaseVisualizer`):
    - An abstract base class for all visualizers.
    - Handles common setup: output directory creation, applying plot styles using `PlotConfig`.
    - Provides utility methods for data validation, plot creation, and formatting.
    - Provides `save_plot()` and `show_plot()` utility methods.
    - Requires subclasses to implement the `create_plot()` method for their specific visualization logic.

- **Individual Visualizers**: Three concrete visualizers inheriting from `BaseVisualizer`:
    - **`QiskitMetricsVisualizer`** (`qward.visualization.qiskit_metrics_visualizer.QiskitMetricsVisualizer`): Visualizes circuit structure, instruction breakdown, and scheduling metrics.
    - **`ComplexityMetricsVisualizer`** (`qward.visualization.complexity_metrics_visualizer.ComplexityMetricsVisualizer`): Visualizes complexity analysis with radar charts, gate metrics, and efficiency analysis.
    - **`CircuitPerformanceVisualizer`** (`qward.visualization.circuit_performance_visualizer.CircuitPerformanceVisualizer`): Visualizes performance metrics with success rates, fidelity, and shot distributions.

- **`Visualizer`** (`qward.visualization.visualizer.Visualizer`):
    - A unified entry point that automatically detects available metrics and provides appropriate visualizations.
    - Can work with Scanner instances or custom metrics data.
    - Provides methods for creating individual plots, dashboards, or complete visualization suites.

```{mermaid}
classDiagram
    class BaseVisualizer {
        <<abstract>>
        +output_dir: str
        +config: PlotConfig
        +save_plot(fig, filename)
        +show_plot(fig)
        +create_plot()*
        +_validate_required_columns()
        +_extract_metrics_from_columns()
        +_create_bar_plot_with_labels()
        +_add_value_labels_to_bars()
        +_show_no_data_message()
        +_setup_plot_axes()
        +_finalize_plot()
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

    class QiskitMetricsVisualizer {
        +metrics_dict: Dict[str, DataFrame]
        +plot_circuit_structure()
        +plot_instruction_breakdown()
        +plot_scheduling_metrics()
        +create_dashboard()
        +plot_all()
    }

    class ComplexityMetricsVisualizer {
        +metrics_dict: Dict[str, DataFrame]
        +plot_gate_based_metrics()
        +plot_complexity_radar()
        +plot_quantum_volume_analysis()
        +plot_efficiency_metrics()
        +create_dashboard()
        +plot_all()
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

    class Visualizer {
        +scanner: Optional[Scanner]
        +metrics_data: Dict[str, DataFrame]
        +registered_visualizers: Dict[str, Type[BaseVisualizer]]
        +register_visualizer()
        +get_available_metrics()
        +visualize_metric()
        +create_dashboard()
        +visualize_all()
    }
    
    note for BaseVisualizer "Abstract base class providing core visualization functionality with common utilities"
    note for QiskitMetricsVisualizer "Visualizes circuit structure and instruction analysis"
    note for ComplexityMetricsVisualizer "Visualizes complexity analysis with radar charts and efficiency metrics"
    note for CircuitPerformanceVisualizer "Visualizes performance metrics with success rates and fidelity analysis"
    note for Visualizer "Unified entry point with auto-detection and comprehensive visualization capabilities"

    BaseVisualizer <|-- QiskitMetricsVisualizer
    BaseVisualizer <|-- ComplexityMetricsVisualizer
    BaseVisualizer <|-- CircuitPerformanceVisualizer
    BaseVisualizer --> PlotConfig : uses
    Visualizer --> BaseVisualizer : manages
```

### Built-in Visualizers

#### `QiskitMetricsVisualizer` (`qward.visualization.qiskit_metrics_visualizer.QiskitMetricsVisualizer`)

Specialized visualizer for `QiskitMetrics` outputs. It provides methods like:
- `plot_circuit_structure()`: Visualizes basic circuit structure (depth, width, size, qubits)
- `plot_instruction_breakdown()`: Shows instruction and gate type analysis
- `plot_scheduling_metrics()`: Displays scheduling and timing information
- `create_dashboard()`: Creates a comprehensive dashboard with all QiskitMetrics plots
- `plot_all()`: Generates all individual plots

#### `ComplexityMetricsVisualizer` (`qward.visualization.complexity_metrics_visualizer.ComplexityMetricsVisualizer`)

Specialized visualizer for `ComplexityMetrics` outputs. It provides methods like:
- `plot_gate_based_metrics()`: Visualizes gate counts and circuit depth metrics
- `plot_complexity_radar()`: Creates radar chart for normalized complexity indicators
- `plot_quantum_volume_analysis()`: Shows Quantum Volume estimation and factors
- `plot_efficiency_metrics()`: Displays parallelism and circuit efficiency analysis
- `create_dashboard()`: Creates a comprehensive dashboard with all ComplexityMetrics plots
- `plot_all()`: Generates all individual plots

#### `CircuitPerformanceVisualizer` (`qward.visualization.circuit_performance_visualizer.CircuitPerformanceVisualizer`)

Specialized visualizer for `CircuitPerformance` metric outputs. It provides methods like:
- `plot_success_error_comparison()`: Creates bar charts comparing success vs. error rates across jobs
- `plot_fidelity_comparison()`: Visualizes fidelity metrics across different jobs
- `plot_shot_distribution()`: Shows distribution of successful vs. failed shots
- `plot_aggregate_summary()`: Creates summary plots of aggregate statistics
- `create_dashboard()`: Generates a comprehensive dashboard with multiple plot types
- `plot_all()`: Convenience method to generate all available plots

#### `Visualizer` (`qward.visualization.visualizer.Visualizer`)

Unified entry point for all visualizations. It provides methods like:
- `register_visualizer()`: Register custom visualizers for specific metrics
- `get_available_metrics()`: Get list of metrics available for visualization
- `visualize_metric()`: Create visualizations for a specific metric type
- `create_dashboard()`: Create dashboards for all available metrics
- `visualize_all()`: Generate all individual plots for all available metrics
- `get_metric_summary()`: Get summary information about available metrics

For more details on usage, available plot types, and customization, refer to the [Visualization Guide](visualization_guide.md).

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
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformance

qc = QuantumCircuit(2,2); qc.h(0); qc.cx(0,1); qc.measure_all()

sim = AerSimulator()
job = sim.run(qc, shots=1024)

# Create scanner with circuit and job
scanner = Scanner(circuit=qc, job=job)
scanner.add_strategy(QiskitMetrics(qc))
scanner.add_strategy(ComplexityMetrics(qc))

def criteria(s): return s == '00' or s == '11' # GHZ state
scanner.add_strategy(CircuitPerformance(circuit=qc, job=job, success_criteria=criteria))

results = scanner.calculate_metrics()
# print(results["CircuitPerformance.aggregate"])
```

### Pattern 4: Using Standard Qiskit Runtime Services
```python
# from qiskit import QuantumCircuit
# from qiskit_ibm_runtime import QiskitRuntimeService
# from qward import Scanner
# from qward.metrics import CircuitPerformance

# qc = QuantumCircuit(2,2); qc.h(0); qc.cx(0,1); qc.measure_all()

# Note: Ensure IBM Quantum credentials are set up
# service = QiskitRuntimeService()
# backend = service.backend('ibmq_qasm_simulator')
# job = service.run(qc, backend=backend)

# scanner = Scanner(circuit=qc, job=job)
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
    - Provides utility methods for data validation, plot creation, and formatting.
    - Provides `save_plot()` and `show_plot()` utility methods.
    - Requires subclasses to implement the `create_plot()` method for their specific visualization logic.

- **Individual Visualizers**: Three concrete visualizers inheriting from `BaseVisualizer`:
    - **`QiskitMetricsVisualizer`** (`qward.visualization.qiskit_metrics_visualizer.QiskitMetricsVisualizer`): Visualizes circuit structure, instruction breakdown, and scheduling metrics.
    - **`ComplexityMetricsVisualizer`** (`qward.visualization.complexity_metrics_visualizer.ComplexityMetricsVisualizer`): Visualizes complexity analysis with radar charts, gate metrics, and efficiency analysis.
    - **`CircuitPerformanceVisualizer`** (`qward.visualization.circuit_performance_visualizer.CircuitPerformanceVisualizer`): Visualizes performance metrics with success rates, fidelity, and shot distributions.

- **`Visualizer`** (`qward.visualization.visualizer.Visualizer`):
    - A unified entry point that automatically detects available metrics and provides appropriate visualizations.
    - Can work with Scanner instances or custom metrics data.
    - Provides methods for creating individual plots, dashboards, or complete visualization suites.

```{mermaid}
classDiagram
    class BaseVisualizer {
        <<abstract>>
        +output_dir: str
        +config: PlotConfig
        +save_plot(fig, filename)
        +show_plot(fig)
        +create_plot()*
        +_validate_required_columns()
        +_extract_metrics_from_columns()
        +_create_bar_plot_with_labels()
        +_add_value_labels_to_bars()
        +_show_no_data_message()
        +_setup_plot_axes()
        +_finalize_plot()
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

    class QiskitMetricsVisualizer {
        +metrics_dict: Dict[str, DataFrame]
        +plot_circuit_structure()
        +plot_instruction_breakdown()
        +plot_scheduling_metrics()
        +create_dashboard()
        +plot_all()
    }

    class ComplexityMetricsVisualizer {
        +metrics_dict: Dict[str, DataFrame]
        +plot_gate_based_metrics()
        +plot_complexity_radar()
        +plot_quantum_volume_analysis()
        +plot_efficiency_metrics()
        +create_dashboard()
        +plot_all()
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

    class Visualizer {
        +scanner: Optional[Scanner]
        +metrics_data: Dict[str, DataFrame]
        +registered_visualizers: Dict[str, Type[BaseVisualizer]]
        +register_visualizer()
        +get_available_metrics()
        +visualize_metric()
        +create_dashboard()
        +visualize_all()
    }
    
    note for BaseVisualizer "Abstract base class providing core visualization functionality with common utilities"
    note for QiskitMetricsVisualizer "Visualizes circuit structure and instruction analysis"
    note for ComplexityMetricsVisualizer "Visualizes complexity analysis with radar charts and efficiency metrics"
    note for CircuitPerformanceVisualizer "Visualizes performance metrics with success rates and fidelity analysis"
    note for Visualizer "Unified entry point with auto-detection and comprehensive visualization capabilities"

    BaseVisualizer <|-- QiskitMetricsVisualizer
    BaseVisualizer <|-- ComplexityMetricsVisualizer
    BaseVisualizer <|-- CircuitPerformanceVisualizer
    BaseVisualizer --> PlotConfig : uses
    Visualizer --> BaseVisualizer : manages
```

### Built-in Visualizers

#### `QiskitMetricsVisualizer` (`qward.visualization.qiskit_metrics_visualizer.QiskitMetricsVisualizer`)

Specialized visualizer for `QiskitMetrics` outputs. It provides methods like:
- `plot_circuit_structure()`: Visualizes basic circuit structure (depth, width, size, qubits)
- `plot_instruction_breakdown()`: Shows instruction and gate type analysis
- `plot_scheduling_metrics()`: Displays scheduling and timing information
- `create_dashboard()`: Creates a comprehensive dashboard with all QiskitMetrics plots
- `plot_all()`: Generates all individual plots

#### `ComplexityMetricsVisualizer` (`qward.visualization.complexity_metrics_visualizer.ComplexityMetricsVisualizer`)

Specialized visualizer for `ComplexityMetrics` outputs. It provides methods like:
- `plot_gate_based_metrics()`: Visualizes gate counts and circuit depth metrics
- `plot_complexity_radar()`: Creates radar chart for normalized complexity indicators
- `plot_quantum_volume_analysis()`: Shows Quantum Volume estimation and factors
- `plot_efficiency_metrics()`: Displays parallelism and circuit efficiency analysis
- `create_dashboard()`: Creates a comprehensive dashboard with all ComplexityMetrics plots
- `plot_all()`: Generates all individual plots

#### `CircuitPerformanceVisualizer` (`qward.visualization.circuit_performance_visualizer.CircuitPerformanceVisualizer`)

Specialized visualizer for `CircuitPerformance` metric outputs. It provides methods like:
- `plot_success_error_comparison()`: Creates bar charts comparing success vs. error rates across jobs
- `plot_fidelity_comparison()`: Visualizes fidelity metrics across different jobs
- `plot_shot_distribution()`: Shows distribution of successful vs. failed shots
- `plot_aggregate_summary()`: Creates summary plots of aggregate statistics
- `create_dashboard()`: Generates a comprehensive dashboard with multiple plot types
- `plot_all()`: Convenience method to generate all available plots

#### `Visualizer` (`qward.visualization.visualizer.Visualizer`)

Unified entry point for all visualizations. It provides methods like:
- `register_visualizer()`: Register custom visualizers for specific metrics
- `get_available_metrics()`: Get list of metrics available for visualization
- `visualize_metric()`: Create visualizations for a specific metric type
- `create_dashboard()`: Create dashboards for all available metrics
- `visualize_all()`: Generate all individual plots for all available metrics
- `get_metric_summary()`: Get summary information about available metrics

For more details on usage, available plot types, and customization, refer to the [Visualization Guide](visualization_guide.md).
