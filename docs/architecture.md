# QWARD Architecture

This document outlines the architecture of the QWARD library and provides usage examples.

## Overview

QWARD is designed with a clear separation between execution and analysis components. The architecture consists of four main components and follows the Strategy pattern for extensible metric calculation.

## Simplified Architecture

This simplified view shows the core Strategy pattern implementation in QWARD, focusing on the essential components and their relationships.

```{mermaid}
classDiagram
    %% Simplified Strategy Pattern for QWARD
    
    class Scanner {
        <<Context>>
        +circuit: QuantumCircuit
        +calculators: List[MetricCalculator]
        +calculate_metrics() DataFrame
    }

    class MetricCalculator {
        <<Strategy Interface>>
        +get_metrics() Dict
    }

    class QiskitMetrics {
        <<Concrete Strategy>>
        +get_metrics() Dict
    }

    class ComplexityMetrics {
        <<Concrete Strategy>>
        +get_metrics() Dict
    }

    class SuccessRate {
        <<Concrete Strategy>>
        +get_metrics() Dict
    }

    %% Strategy Pattern Relationships
    Scanner --> MetricCalculator : uses strategies
    
    %% Strategy Interface Implementation
    MetricCalculator <|.. QiskitMetrics : implements
    MetricCalculator <|.. ComplexityMetrics : implements
    MetricCalculator <|.. SuccessRate : implements

    %% Pattern Notes
    note for Scanner "Context: Orchestrates metric calculation and returns consolidated DataFrame"
    note for MetricCalculator "Strategy Interface: Common interface for all metric calculation algorithms"
    note for QiskitMetrics "Returns Dict of Qiskit-native metrics"
    note for ComplexityMetrics "Returns Dict of complexity analysis metrics"
    note for SuccessRate "Returns Dict of execution success metrics"
```

### Key Points

- **Scanner (Context)**: Maintains metric calculators and delegates calculation work
- **MetricCalculator (Interface)**: Defines common `get_metrics()` method returning Dict
- **Concrete Strategies**: Each implements different metric calculation algorithms
- **Strategy Pattern Benefits**: Runtime strategy switching, extensibility, separation of concerns

## Detailed Architecture

The complete architecture with all implementation details:

```{mermaid}
classDiagram
    %% Strategy Pattern Implementation for Quantum Circuit Metrics Analysis
    
    class Scanner {
        <<Context>>
        +circuit: QuantumCircuit
        +job: Union[AerJob, QiskitJob]
        +result: Result
        +metrics: List[MetricStrategy]
        +__init__(circuit, job, result, metrics)
        +add_metric(metric_strategy)
        +calculate_metrics()
        +set_circuit(circuit)
        +set_job(job)
        +set_result(result)
    }



    class Result {
        <<Data>>
        +job: Union[AerJob, QiskitJob]
        +counts: Dict[str, int]
        +metadata: Dict[str, Any]
        +__init__(job, counts, metadata)
        +save(path)
        +load(path)
        +update_from_job()
    }

    class MetricStrategy {
        <<Strategy Interface>>
        # Attributes initialized in __init__
        _circuit: QuantumCircuit
        _metric_type: MetricsType
        _id: MetricsId
        +__init__(circuit)
        +metric_type: MetricsType
        +id: MetricsId
        +name: str
        +circuit: QuantumCircuit
        +_get_metric_type()* 
        +_get_metric_id()*
        +is_ready()*
        +get_metrics()*
    }

    class QiskitMetricsStrategy {
        <<Concrete Strategy>>
        +__init__(circuit)
        +_get_metric_type()
        +_get_metric_id()
        +is_ready()
        +get_metrics()
        +get_basic_metrics()
        +get_instruction_metrics()
        +get_scheduling_metrics()
    }

    class ComplexityMetricsStrategy {
        <<Concrete Strategy>>
        +__init__(circuit)
        +_get_metric_type()
        +_get_metric_id()
        +is_ready()
        +get_metrics()
        +get_gate_based_metrics()
        +get_entanglement_metrics()
        +get_standardized_metrics()
        +get_advanced_metrics()
        +get_derived_metrics()
        +estimate_quantum_volume()
    }

    class SuccessRateStrategy {
        <<Concrete Strategy>>
        _job: Optional[Union[AerJob, QiskitJob]]
        _jobs: List[Union[AerJob, QiskitJob]]
        _result: Optional[Dict]
        success_criteria: Callable
        +__init__(circuit, job, jobs, result, success_criteria)
        +_get_metric_type()
        +_get_metric_id()
        +is_ready()
        +get_metrics()
        +add_job(job)
    }

    class MetricsType {
        <<enumeration>>
        PRE_RUNTIME
        POST_RUNTIME
    }

    class MetricsId {
        <<enumeration>>
        QISKIT
        COMPLEXITY
        SUCCESS_RATE
    }

    %% Strategy Pattern Relationships
    Scanner --> MetricStrategy : uses strategies
    Scanner --> Result : manages
    
    %% Strategy Interface Implementation
    MetricStrategy <|.. QiskitMetricsStrategy : implements
    MetricStrategy <|.. ComplexityMetricsStrategy : implements
    MetricStrategy <|.. SuccessRateStrategy : implements
    
    %% Strategy Interface Dependencies
    MetricStrategy --> MetricsType : defines type
    MetricStrategy --> MetricsId : defines identifier

    %% Notes about Strategy Pattern
    note for Scanner "Context: Maintains references to metric strategies and delegates metric calculation work to them. Can switch strategies at runtime via add_metric()."
    
    note for MetricStrategy "Strategy Interface: Defines common interface for all metric calculation algorithms. Each strategy bundles related metrics."
    
    note for QiskitMetricsStrategy "Concrete Strategy: Bundles Qiskit-native metrics (basic, instruction, scheduling)"
    
    note for ComplexityMetricsStrategy "Concrete Strategy: Bundles complexity analysis metrics (gate-based, entanglement, standardized, advanced, derived)"
    
    note for SuccessRateStrategy "Concrete Strategy: Bundles execution success metrics (success rate, fidelity, error rate)"
```

## Folder Structure

The QWARD library is organized into the following folder structure:

```
/qward/
├── __init__.py                 # Main package initialization
├── scanner.py                  # Scanner class implementation
├── runtime/
│   └── __init__.py
├── result.py                   # Result class implementation
├── metrics/
│   ├── __init__.py
│   ├── base_metric.py          # Base Metric class
│   ├── types.py                # MetricsType and MetricsId enums
│   ├── defaults.py             # Default metric configurations
│   ├── qiskit_metrics.py       # QiskitMetrics implementation
│   ├── complexity_metrics.py   # ComplexityMetrics implementation
│   └── success_rate.py         # SuccessRate implementation
├── utils/
│   ├── __init__.py
│   ├── flatten.py              # Utility for flattening nested lists
│   └── helpers.py              # Utility functions
└── examples/
    ├── __init__.py
    ├── utils.py                # Utilities for examples
    ├── run_on_aer.ipynb        # Example notebook for running on Aer simulator
    ├── aer.py                  # Example Aer simulator usage
    └── example_metrics_constructor.py # Example for custom metrics constructor
```

This structure provides a clean organization for the code, with:

1. **Main Package**: Core classes at the top level for easy imports
2. **Runtime Module**: Handles execution of quantum circuits
3. **Metrics Module**: Contains all metric implementations
4. **Utils Module**: Helper functions and utilities
5. **Examples Module**: Working code examples demonstrating library usage

## Components

### Scanner
The Scanner class is the main entry point for analyzing quantum circuits. It can be initialized with a quantum circuit, job, result, and an optional list of metric classes or instances. It allows users to add further metrics and calculate them.



### Result
The Result class represents the output of a quantum circuit execution. It includes the job information, measurement counts, and metadata. It provides methods for saving and loading results, as well as updating results from a job.

### Metric
The Metric class is an abstract base class that defines the interface for all metrics. It includes the circuit attribute, properties for metric type and ID, and abstract methods for metric calculation. Concrete implementations include QiskitMetrics, ComplexityMetrics, and SuccessRate. Default metric classes can be obtained using the `get_default_metrics()` function from the `qward.metrics.defaults` module.

### SuccessRate
The SuccessRate class calculates success rate metrics for quantum circuits, such as success rate, fidelity, and error rate. It is initialized with a `QuantumCircuit`, and can optionally take a single `job` or a list of `jobs`, a `result` dictionary (containing counts), and a custom `success_criteria` function. Metrics are calculated based on the execution counts from the provided job(s) or result. Additional jobs can be added using the `add_job` method for aggregate analysis.

## Usage Examples

### Basic Circuit Analysis
```python
from qiskit import QuantumCircuit
from qward import Scanner, QiskitMetrics

# Create a quantum circuit
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)

# Create a scanner with the circuit
scanner = Scanner(circuit=circuit)

# Add a metric
scanner.add_metric(QiskitMetrics(circuit))

# Calculate metrics
results = scanner.calculate_metrics()
```



### Analyzing Results
```python
from qward import Scanner, QiskitMetrics, ComplexityMetrics

# Create a scanner with a result
scanner = Scanner(result=result)

# Add multiple metrics
scanner.add_metric(QiskitMetrics(circuit))
scanner.add_metric(ComplexityMetrics(circuit))

# Calculate metrics
results = scanner.calculate_metrics()
```

### Using Custom Metrics
```python
from qward import Metric, MetricsType, MetricsId
from qiskit import QuantumCircuit

class MyCustomMetric(Metric):
    def __init__(self, circuit: QuantumCircuit):
        super().__init__(circuit)
    
    def _get_metric_type(self) -> MetricsType:
        """
        Get the type of this metric.
        
        Returns:
            MetricsType: The type of this metric
        """
        return MetricsType.PRE_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        """
        Get the ID of this metric.
        For custom metrics, you might extend MetricsId or use a general ID.
        Returns:
            MetricsId: The ID of this metric (e.g., reusing an existing one for simplicity)
        """
        return MetricsId.QISKIT
    
    def is_ready(self) -> bool:
        return True
    
    def get_metrics(self) -> dict:
        # Custom metric calculation
        value = 42
        return {"my_metric": value}

# Example usage (assuming 'circuit' is a QuantumCircuit instance)
# circuit = QuantumCircuit(1)
# scanner = Scanner(circuit=circuit)
# scanner.add_metric(MyCustomMetric(circuit))
# results = scanner.calculate_metrics()
# print(results)
```

## Best Practices

1. **Circuit Analysis**
   - Use the Scanner class for all circuit analysis
   - Add metrics before calculating results
   - Consider using multiple metrics for comprehensive analysis

2. **Execution**
   - Handle job and result errors appropriately
   - Use appropriate Qiskit runtime services for backend execution

3. **Result Management**
   - Save results for later analysis
   - Include relevant metadata with results
   - Use consistent naming conventions for saved results

4. **Custom Metrics**
   - Inherit from the Metric base class
   - Implement the required abstract methods
   - Return results in a consistent format
   - Document metric calculation methodology

## Visualization System

QWARD includes a comprehensive visualization system for analyzing and presenting quantum circuit metrics. The visualization module follows a structured approach that integrates seamlessly with the metric calculation system.

### Architecture Overview

The visualization system is built on the following key components:

- **`PlotConfig`**: A dataclass holding all plot appearance and saving configurations.
- **`BaseVisualizer`**: An abstract base class for all visualizers. It handles common setup (output directory, styling via `PlotConfig`) and provides `save_plot`/`show_plot` methods. Subclasses must implement `create_plot()` for their specific visualization logic.
- **`SuccessRateVisualizer`**: A concrete visualizer inheriting from `BaseVisualizer`. It's responsible for generating various plots related to success rate metrics. Internally, it uses the **Strategy pattern** to manage different types of plots.
- **`PlotStrategy`**: An interface (abstract base class) defining a contract for different plot generation algorithms. Concrete strategies (e.g., `SuccessErrorPlotStrategy`, `FidelityPlotStrategy`) implement this interface to create specific charts. `SuccessRateVisualizer` delegates plotting tasks to these strategies.
- **(Conceptual) `MetricPlottingUtils`**: A utility class or module (not shown in the diagram for simplicity but important for implementation) would contain static helper methods for common tasks related to plotting metric data (e.g., extracting data, validating columns, adding standard labels). Both `SuccessRateVisualizer` and its strategies might use these utilities.

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

### Key Components (Summary)

1. **`BaseVisualizer`**: Abstract base class providing core visualization functionality.
2. **`SuccessRateVisualizer`**: Concrete implementation for `SuccessRate` metrics, using the Strategy pattern internally for different plot types.
3. **`PlotStrategy`**: Interface for different plotting algorithms, allowing `SuccessRateVisualizer` to be flexible in how it generates plots.
4. **`PlotConfig`**: Dataclass for easy plot appearance customization.

### Integration with Scanner

The visualization system seamlessly integrates with the Scanner output:

```python
# Calculate metrics
scanner = Scanner(circuit=circuit)
scanner.add_strategy(SuccessRate(circuit=circuit))
metrics_dict = scanner.calculate_metrics()

# Create visualizations
visualizer = SuccessRateVisualizer(metrics_dict)
figures = visualizer.plot_all(save=True)
```

### Extensibility

The visualization system is designed for easy extension:

1. **Custom Visualizers**: Create new visualizers by inheriting from `MetricVisualizer`
2. **Custom Styles**: Define new plot styles and color palettes
3. **New Plot Types**: Add new visualization methods to existing visualizers

For detailed usage and examples, see the [Visualization Guide](visualization_guide.md). 