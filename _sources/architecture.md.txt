# QWARD Architecture

This document outlines the architecture of the QWARD library and provides usage examples.

## Overview

QWARD is designed with a clear separation between execution and analysis components. The architecture consists of four main components and follows the Strategy pattern for extensible metric calculation, enhanced with Pydantic-based schema validation for data integrity and type safety.

## Simplified Architecture

This simplified view shows the core Strategy pattern implementation in QWARD, focusing on the essential components and their relationships.

```{mermaid}
classDiagram
    %% Simplified Strategy Pattern for QWARD with Schema Validation
    
    class Scanner {
        <<Context>>
        +circuit: QuantumCircuit
        +calculators: List[MetricCalculator]
        +calculate_metrics() Dict[str, DataFrame]
    }

    class MetricCalculator {
        <<Strategy Interface>>
        +get_metrics() BaseModel
    }

    class QiskitMetrics {
        <<Concrete Strategy>>
        +circuit: QuantumCircuit
        +get_metrics() QiskitMetricsSchema
        +is_ready() bool
    }

    class ComplexityMetrics {
        <<Concrete Strategy>>
        +get_metrics() ComplexityMetricsSchema
    }

    class CircuitPerformanceMetrics {
        <<Concrete Strategy>>
        +get_metrics() CircuitPerformanceSchema
    }

    class SchemaModule {
        <<Validation Layer>>
        +QiskitMetricsSchema
        +ComplexityMetricsSchema
        +CircuitPerformanceSchema
        +validate_data()
        +generate_json_schema()
        +to_flat_dict()
    }

    %% Strategy Pattern Relationships
    Scanner --> MetricCalculator : uses strategies
    
    %% Strategy Interface Implementation
    MetricCalculator <|.. QiskitMetrics : implements
    MetricCalculator <|.. ComplexityMetrics : implements
    MetricCalculator <|.. CircuitPerformanceMetrics : implements
    
    %% Schema Integration
    QiskitMetrics --> SchemaModule : validates with
    ComplexityMetrics --> SchemaModule : validates with
    CircuitPerformanceMetrics --> SchemaModule : validates with

    %% Pattern Notes
    note for Scanner "Context: Orchestrates metric calculation and returns dictionary of DataFrames"
    note for MetricCalculator "Strategy Interface: Common interface returning validated schema objects"
    note for QiskitMetrics "Returns validated Qiskit-native metrics with type safety"
    note for ComplexityMetrics "Returns validated complexity analysis metrics with constraints"
    note for CircuitPerformanceMetrics "Returns validated execution performance metrics with cross-field validation"
    note for SchemaModule "Pydantic-based validation with automatic type checking and JSON schema generation"
```

### Key Points

- **Scanner (Context)**: Maintains metric calculators and delegates calculation work
- **MetricCalculator (Interface)**: Defines common interface returning validated schema objects
- **Concrete Strategies**: Each implements different metric calculation algorithms with validation
- **Schema Validation**: Pydantic-based data validation ensures type safety and constraint checking
- **Unified API**: All metric classes use `get_metrics()` to return schema objects directly
- **Scanner Integration**: Scanner automatically calls `to_flat_dict()` for DataFrame conversion
- **Strategy Pattern Benefits**: Runtime strategy switching, extensibility, separation of concerns

## Schema-Based Data Validation

QWARD includes comprehensive schema-based validation using Pydantic, providing:

```{mermaid}
classDiagram
    %% Schema Validation Architecture
    
    class BaseModel {
        <<Pydantic Base>>
        +model_validate()
        +model_json_schema()
        +model_dump()
        +to_flat_dict()
    }
    
    class QiskitMetricsSchema {
        +basic_metrics: BasicMetricsSchema
        +instruction_metrics: InstructionMetricsSchema
        +scheduling_metrics: SchedulingMetricsSchema
        +to_flat_dict() Dict[str, Any]
        +from_flat_dict() QiskitMetricsSchema
    }
    
    class ComplexityMetricsSchema {
        +gate_based_metrics: GateBasedMetricsSchema
        +entanglement_metrics: EntanglementMetricsSchema
        +standardized_metrics: StandardizedMetricsSchema
        +advanced_metrics: AdvancedMetricsSchema
        +derived_metrics: DerivedMetricsSchema
        +to_flat_dict() Dict[str, Any]
        +from_flat_dict() ComplexityMetricsSchema
    }
    
    class CircuitPerformanceSchema {
        +success_metrics: SuccessMetricsSchema
        +fidelity_metrics: FidelityMetricsSchema
        +statistical_metrics: StatisticalMetricsSchema
        +validate_error_rate()
        +validate_successful_shots()
        +to_flat_dict() Dict[str, Any]
    }

    %% Inheritance
    BaseModel <|-- QiskitMetricsSchema
    BaseModel <|-- ComplexityMetricsSchema
    BaseModel <|-- CircuitPerformanceSchema

    %% Validation Features
    note for QiskitMetricsSchema "Validates circuit metrics with type checking and constraints"
    note for ComplexityMetricsSchema "Validates complexity metrics with range and cross-field validation"
    note for CircuitPerformanceSchema "Validates performance metrics with rate consistency checks and statistical constraints"
```

### Schema Benefits

1. **Type Safety**: Automatic type checking and validation
2. **Constraint Validation**: Range checks, cross-field validation, and business rules
3. **IDE Support**: Full autocomplete and type hints
4. **API Documentation**: Automatic JSON schema generation
5. **DataFrame Compatibility**: Easy conversion to/from flat dictionaries via `to_flat_dict()`
6. **Error Prevention**: Catch data inconsistencies early

## Detailed Architecture

The complete architecture with all implementation details:

```{mermaid}
classDiagram
    %% Complete QWARD Architecture with Schema Validation
    
    class Scanner {
        <<Context>>
        +circuit: QuantumCircuit
        +job: Union[AerJob, QiskitJob]
        +strategies: List[MetricCalculator]
        +__init__(circuit, job, strategies)
        +add_strategy(metric_calculator)
        +calculate_metrics() Dict[str, DataFrame]
        +set_circuit(circuit)
        +set_job(job)
    }

    class MetricCalculator {
        <<Strategy Interface>>
        #circuit: QuantumCircuit
        #_metric_type: MetricsType
        #_id: MetricsId
        +__init__(circuit)
        +metric_type: MetricsType
        +id: MetricsId
        +name: str
        +circuit: QuantumCircuit
        +_get_metric_type()* MetricsType
        +_get_metric_id()* MetricsId
        +is_ready()* bool
        +get_metrics()* BaseModel
        +_ensure_schemas_available()
    }

    class QiskitMetrics {
        <<Concrete Strategy>>
        +__init__(circuit)
        +_get_metric_type() MetricsType
        +_get_metric_id() MetricsId
        +is_ready() bool
        +get_metrics() QiskitMetricsSchema
        +get_basic_metrics() BasicMetricsSchema
        +get_instruction_metrics() InstructionMetricsSchema
        +get_scheduling_metrics() SchedulingMetricsSchema
    }

    class ComplexityMetrics {
        <<Concrete Strategy>>
        +__init__(circuit)
        +_get_metric_type() MetricsType
        +_get_metric_id() MetricsId
        +is_ready() bool
        +get_metrics() ComplexityMetricsSchema
        +get_gate_based_metrics() GateBasedMetricsSchema
        +get_entanglement_metrics() EntanglementMetricsSchema
        +get_standardized_metrics() StandardizedMetricsSchema
        +get_advanced_metrics() AdvancedMetricsSchema
        +get_derived_metrics() DerivedMetricsSchema
    }

    class CircuitPerformanceMetrics {
        <<Concrete Strategy>>
        #_job: Optional[JobType]
        #_jobs: List[JobType]
        #success_criteria: Callable[[str], bool]
        +__init__(circuit, job, jobs, success_criteria)
        +_get_metric_type() MetricsType
        +_get_metric_id() MetricsId
        +is_ready() bool
        +get_metrics() CircuitPerformanceSchema
        +get_success_metrics() SuccessMetricsSchema
        +get_fidelity_metrics() FidelityMetricsSchema
        +get_statistical_metrics() StatisticalMetricsSchema
        +add_job(job)
        +_default_success_criteria() Callable[[str], bool]
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
        CIRCUIT_PERFORMANCE
    }

    %% Strategy Pattern Relationships
    Scanner --> MetricCalculator : uses strategies
    
    %% Strategy Interface Implementation
    MetricCalculator <|.. QiskitMetrics : implements
    MetricCalculator <|.. ComplexityMetrics : implements
    MetricCalculator <|.. CircuitPerformanceMetrics : implements
    
    %% Strategy Interface Dependencies
    MetricCalculator --> MetricsType : defines type
    MetricCalculator --> MetricsId : defines identifier

    %% Notes about Enhanced Architecture
    note for Scanner "Context: Maintains references to metric calculators and delegates work. Returns dictionary of DataFrames for analysis."
    
    note for MetricCalculator "Strategy Interface: Common interface returning validated schema objects. Includes schema availability checking."
    
    note for QiskitMetrics "Concrete Strategy: Qiskit-native metrics with validated schema output"
    
    note for ComplexityMetrics "Concrete Strategy: Comprehensive complexity analysis with validated schemas and constraint checking"
    
    note for CircuitPerformanceMetrics "Concrete Strategy: Circuit performance analysis with validated schema output and custom criteria"
```

## Folder Structure

The QWARD library is organized into the following folder structure:

```
/qward/
├── __init__.py                 # Main package initialization
├── scanner.py                  # Scanner class implementation
├── runtime/
│   └── __init__.py
├── metrics/
│   ├── __init__.py
│   ├── base_metric.py          # Base MetricCalculator class
│   ├── types.py                # MetricsType and MetricsId enums
│   ├── defaults.py             # Default metric configurations
│   ├── schemas.py              # Pydantic schema definitions
│   ├── qiskit_metrics.py       # QiskitMetrics implementation
│   ├── complexity_metrics.py   # ComplexityMetrics implementation
│   └── circuit_performance.py  # CircuitPerformanceMetrics implementation
├── visualization/
│   ├── __init__.py
│   ├── base.py                 # VisualizationStrategy base class and PlotConfig
│   ├── visualizer.py           # Unified Visualizer class
│   ├── qiskit_metrics_visualizer.py      # QiskitVisualizer implementation
│   ├── complexity_metrics_visualizer.py  # ComplexityVisualizer implementation
│   └── circuit_performance_visualizer.py # CircuitPerformanceVisualizer implementation
├── utils/
│   ├── __init__.py
│   ├── flatten.py              # Utility for flattening nested lists
│   └── helpers.py              # Utility functions
└── examples/
    ├── __init__.py
    ├── utils.py                # Utilities for examples
    ├── schema_demo.py          # Schema validation demonstration
    ├── circuit_performance_demo.py # Circuit performance metrics demonstration
    ├── run_on_aer.ipynb        # Example notebook for running on Aer simulator
    ├── aer.py                  # Example Aer simulator usage
    ├── example_visualizer.py   # Comprehensive visualization examples
    ├── visualization_demo.py   # CircuitPerformanceVisualizer demo
    ├── direct_strategy_example.py # Direct strategy usage examples
    └── visualization_quickstart.py # Quick visualization example
```

This structure provides a clean organization for the code, with:

1. **Main Package**: Core classes at the top level for easy imports
2. **Runtime Module**: Handles execution of quantum circuits
3. **Metrics Module**: Contains all metric implementations and schema definitions
4. **Visualization Module**: Contains all visualization strategies and unified visualizer
5. **Utils Module**: Helper functions and utilities
6. **Examples Module**: Working code examples demonstrating library usage including schema validation and visualization

## Components

### Scanner
The Scanner class is the main entry point for analyzing quantum circuits. It can be initialized with a quantum circuit, job, and an optional list of metric classes or instances. It allows users to add further metrics and calculate them, returning results as a dictionary of DataFrames for easy analysis.

### MetricCalculator
The MetricCalculator class is an abstract base class that defines the interface for all metrics. It includes the circuit attribute, properties for metric type and ID, and abstract methods for metric calculation. All concrete implementations return validated schema objects directly through `get_metrics()`. Default metric classes can be obtained using the `get_default_strategies()` function from the `qward.metrics.defaults` module.

### Schema Validation
The schema validation system provides:
- **Type Safety**: Automatic validation of data types and constraints
- **Business Rules**: Cross-field validation (e.g., error_rate = 1 - success_rate)
- **Range Validation**: Ensures values are within expected bounds
- **Documentation**: Automatic JSON schema generation for API documentation
- **IDE Support**: Full autocomplete and type hints for better developer experience

### QiskitMetrics
The QiskitMetrics class extracts metrics directly from QuantumCircuit objects, including basic metrics (depth, width, gate counts), instruction metrics (connectivity, factors), and scheduling metrics (timing information). It returns a validated `QiskitMetricsSchema` object with type-safe access to all metric categories.

### ComplexityMetrics
The ComplexityMetrics class calculates comprehensive circuit complexity metrics based on research literature, including gate-based metrics, entanglement metrics, standardized metrics, advanced metrics, and derived metrics. All metrics are validated through a `ComplexityMetricsSchema` with appropriate constraints and cross-field validation.

### CircuitPerformanceMetrics
The CircuitPerformanceMetrics class calculates performance metrics for quantum circuits, such as success rate, fidelity, and error rate. It supports both single job and multiple job analysis with customizable success criteria. The class returns a validated `CircuitPerformanceSchema` object with validation for both individual job metrics and aggregate statistics across multiple jobs.

## Usage Examples

### Basic Circuit Analysis with Schema Validation
```python
from qiskit import QuantumCircuit
from qward import Scanner
from qward.metrics import QiskitMetrics

# Create a quantum circuit
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)

# Create a scanner with the circuit
scanner = Scanner(circuit=circuit)

# Add a metric
scanner.add_strategy(QiskitMetrics(circuit))

# Calculate metrics (Scanner returns dictionary of DataFrames)
results = scanner.calculate_metrics()

# Use schema-based metrics directly
qiskit_metrics = QiskitMetrics(circuit)
metrics = qiskit_metrics.get_metrics()  # Returns QiskitMetricsSchema

# Access validated data with full type safety
print(f"Circuit depth: {metrics.basic_metrics.depth}")
print(f"Gate count: {metrics.basic_metrics.size}")
print(f"Number of qubits: {metrics.basic_metrics.num_qubits}")
```

### Comprehensive Analysis with Multiple Metrics
```python
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics

# Create a scanner with multiple metrics
scanner = Scanner(circuit=circuit)
scanner.add_strategy(QiskitMetrics(circuit))
scanner.add_strategy(ComplexityMetrics(circuit))

# For circuit performance, you need job execution results
if job:  # Assuming you have a job from circuit execution
    scanner.add_strategy(CircuitPerformanceMetrics(circuit, job=job))

# Calculate all metrics (returns dictionary of DataFrames)
results = scanner.calculate_metrics()

# Access schema-based metrics for detailed analysis
complexity_metrics = ComplexityMetrics(circuit)
complexity_schema = complexity_metrics.get_metrics()

print(f"Gate Count: {complexity_schema.gate_based_metrics.gate_count}")
print(f"Gate Density: {complexity_schema.standardized_metrics.gate_density}")
print(f"Parallelism Efficiency: {complexity_schema.advanced_metrics.parallelism_efficiency}")
```

### Circuit Performance Analysis with Custom Criteria
```python
from qward.metrics import CircuitPerformanceMetrics

# Define custom success criteria
def bell_state_success(result: str) -> bool:
    clean_result = result.replace(" ", "")
    return clean_result in ["0000", "1111"]  # |00⟩ or |11⟩ states

# Create circuit performance metric with custom criteria
circuit_performance = CircuitPerformanceMetrics(
    circuit=circuit, 
    job=job, 
    success_criteria=bell_state_success
)

# Get validated schema with automatic constraint checking
metrics = circuit_performance.get_metrics()
print(f"Success Rate: {metrics.success_metrics.success_rate:.3f}")
print(f"Fidelity: {metrics.fidelity_metrics.fidelity:.3f}")
print(f"Error Rate: {metrics.success_metrics.error_rate:.3f}")  # Automatically validated
```

### Schema Validation and JSON Generation
```python
from qward.metrics.schemas import ComplexityMetricsSchema
import json

# Generate JSON schema for API documentation
complexity_metrics = ComplexityMetrics(circuit)
schema_obj = complexity_metrics.get_metrics()

# Get JSON schema for documentation
json_schema = ComplexityMetricsSchema.model_json_schema()
print(json.dumps(json_schema, indent=2))

# Convert to flat dictionary for DataFrame compatibility
flat_dict = schema_obj.to_flat_dict()
print(f"Flattened metrics: {list(flat_dict.keys())}")

# Validate data integrity
try:
    # This will raise ValidationError if data is invalid
    validated_schema = ComplexityMetricsSchema.model_validate(schema_obj.model_dump())
    print("✅ Data validation passed")
except Exception as e:
    print(f"❌ Validation error: {e}")
```

### Using Custom Metrics with Schema Support
```python
from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsType, MetricsId
from qiskit import QuantumCircuit
from pydantic import BaseModel

class MyCustomMetricsSchema(BaseModel):
    custom_complexity: float
    circuit_signature: str

class MyCustomMetric(MetricCalculator):
    def __init__(self, circuit: QuantumCircuit):
        super().__init__(circuit)
    
    def _get_metric_type(self) -> MetricsType:
        return MetricsType.PRE_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        return MetricsId.QISKIT  # Or define a new ID
    
    def is_ready(self) -> bool:
        return self.circuit is not None
    
    def get_metrics(self) -> MyCustomMetricsSchema:
        # Custom metric calculation
        custom_value = self.circuit.depth() * self.circuit.num_qubits
        return MyCustomMetricsSchema(
            custom_complexity=custom_value,
            circuit_signature=f"{self.circuit.num_qubits}q_{self.circuit.depth()}d"
        )

# Usage
scanner = Scanner(circuit=circuit)
scanner.add_strategy(MyCustomMetric(circuit))
results = scanner.calculate_metrics()
```

## Best Practices

1. **Circuit Analysis**
   - Use the Scanner class for all circuit analysis
   - Add metrics before calculating results
   - Consider using multiple metrics for comprehensive analysis
   - All metrics return validated schema objects by default

2. **Schema Usage**
   - All metric classes return schema objects directly via `get_metrics()`
   - Take advantage of IDE autocomplete with schema objects
   - Use `to_flat_dict()` when you need dictionary format for DataFrames
   - Leverage JSON schema generation for API documentation

3. **Execution**
   - Handle job and result errors appropriately
   - Use appropriate Qiskit runtime services for backend execution
   - Validate success criteria for CircuitPerformanceMetrics metrics

4. **Data Management**
   - Use consistent naming conventions for analysis results
   - Schema validation ensures data integrity automatically
   - Scanner handles schema-to-DataFrame conversion seamlessly

5. **Custom Metrics**
   - Inherit from the MetricCalculator base class
   - Implement the required abstract methods
   - Return validated schema objects from `get_metrics()`
   - Document metric calculation methodology
   - Consider adding custom Pydantic schemas for validation

6. **Performance**
   - Schema validation adds minimal overhead but provides significant benefits
   - Scanner automatically handles `to_flat_dict()` conversion for DataFrames
   - All validation happens at metric calculation time

## Visualization System

QWARD includes a comprehensive visualization system for analyzing and presenting quantum circuit metrics. The visualization module follows a structured approach that integrates seamlessly with the metric calculation system and provides a modern, type-safe API.

### New API Features (v0.9.0)

#### 🎯 **Type-Safe Constants System**
- **`Metrics`** constants: `Metrics.QISKIT`, `Metrics.COMPLEXITY`, `Metrics.CIRCUIT_PERFORMANCE`
- **`Plots`** constants: `Plots.QISKIT.CIRCUIT_STRUCTURE`, `Plots.COMPLEXITY.COMPLEXITY_RADAR`, etc.
- **IDE Autocompletion**: Full IntelliSense support for all plot names
- **Error Prevention**: Compile-time detection of typos in metric and plot names

#### 🔍 **Rich Plot Metadata System**
- **Plot Descriptions**: Detailed information about what each plot shows
- **Plot Types**: Categorized as bar charts, radar charts, line plots, etc.
- **Dependencies**: Information about required data columns
- **Categories**: Organized by analysis type (structure, performance, complexity)

#### ⚡ **Granular Plot Control**
- **Single Plot Generation**: `generate_plot(metric, plot_name)`
- **Selected Plots**: `generate_plots({metric: [plot1, plot2]})`
- **All Plots**: `generate_plots({metric: None})`
- **Memory Efficient**: Default `save=False, show=False` for batch processing

### Architecture Overview

The visualization system is built on the following key components:

- **`PlotConfig`**: A dataclass holding all plot appearance and saving configurations.
- **`VisualizationStrategy`**: An abstract base class for all visualizers with plot registry system. It handles common setup (output directory, styling via `PlotConfig`) and provides utility methods like `save_plot`/`show_plot`, data validation, and plot creation helpers. Subclasses must implement plot registry and new API methods.
- **Individual Visualizers**: Three concrete visualizers inheriting from `VisualizationStrategy`:
  - **`QiskitVisualizer`**: Visualizes circuit structure and instruction metrics (4 plots)
  - **`ComplexityVisualizer`**: Visualizes complexity analysis with radar charts and efficiency metrics (3 plots)
  - **`CircuitPerformanceVisualizer`**: Visualizes performance metrics with success rates, fidelity, and shot distributions (4 plots)
- **`Visualizer`**: A unified entry point that automatically detects available metrics and provides type-safe visualization capabilities.

```{mermaid}
classDiagram
    class VisualizationStrategy {
        <<abstract>>
        +output_dir: str
        +config: PlotConfig
        +PLOT_REGISTRY: Dict[str, PlotMetadata]
        +save_plot(fig, filename)
        +show_plot(fig)
        +get_available_plots()* List[str]
        +get_plot_metadata(plot_name)* PlotMetadata
        +generate_plot(plot_name, save, show)* Figure
        +generate_plots(plot_names, save, show)* List[Figure]
        +generate_all_plots(save, show) List[Figure]
        +create_dashboard(save, show)*
        +_validate_required_columns()
        +_extract_metrics_from_columns()
        +_create_bar_plot_with_labels()
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
        +__post_init__()
    }
    
    class PlotMetadata {
        +name: str
        +method_name: str
        +description: str
        +plot_type: PlotType
        +filename: str
        +dependencies: List[str]
        +category: str
    }
    
    class QiskitVisualizer {
        +PLOT_REGISTRY: Dict[str, PlotMetadata]
        +generate_plot(plot_name, save, show)
        +get_available_plots() List[str]
        +get_plot_metadata(plot_name) PlotMetadata
        +plot_circuit_structure()
        +plot_gate_distribution()
        +plot_instruction_metrics()
        +plot_circuit_summary()
        +create_dashboard()
    }
    
    class ComplexityVisualizer {
        +PLOT_REGISTRY: Dict[str, PlotMetadata]
        +generate_plot(plot_name, save, show)
        +get_available_plots() List[str]
        +get_plot_metadata(plot_name) PlotMetadata
        +plot_gate_based_metrics()
        +plot_complexity_radar()
        +plot_efficiency_metrics()
        +create_dashboard()
    }
    
    class CircuitPerformanceVisualizer {
        +PLOT_REGISTRY: Dict[str, PlotMetadata]
        +generate_plot(plot_name, save, show)
        +get_available_plots() List[str]
        +get_plot_metadata(plot_name) PlotMetadata
        +plot_success_error_comparison()
        +plot_fidelity_comparison()
        +plot_shot_distribution()
        +plot_aggregate_summary()
        +create_dashboard()
    }
    
    class Visualizer {
        +scanner: Optional[Scanner]
        +metrics_data: Dict[str, DataFrame]
        +registered_strategies: Dict[str, Type[VisualizationStrategy]]
        +get_available_plots() Dict[str, List[str]]
        +get_plot_metadata(metric, plot_name) PlotMetadata
        +generate_plot(metric, plot_name, save, show) Figure
        +generate_plots(selections, save, show) Dict[str, List[Figure]]
        +create_dashboard(save, show) Dict[str, Figure]
        +register_strategy()
        +get_available_metrics()
        +get_metric_summary()
        +print_available_metrics()
    }

    VisualizationStrategy <|-- QiskitVisualizer
    VisualizationStrategy <|-- ComplexityVisualizer
    VisualizationStrategy <|-- CircuitPerformanceVisualizer
    VisualizationStrategy --> PlotConfig : uses
    VisualizationStrategy --> PlotMetadata : defines
    Visualizer --> VisualizationStrategy : manages
    Visualizer --> Scanner : optional integration
    
    note for VisualizationStrategy "Abstract base class with plot registry and metadata system"
    note for QiskitVisualizer "4 plots: circuit_structure, gate_distribution, instruction_metrics, circuit_summary"
    note for ComplexityVisualizer "3 plots: gate_based_metrics, complexity_radar, efficiency_metrics"
    note for CircuitPerformanceVisualizer "4 plots: success_error_comparison, fidelity_comparison, shot_distribution, aggregate_summary"
    note for Visualizer "Unified entry point with type-safe constants and granular control"
```

### Key Components

#### `VisualizationStrategy`
Abstract base class providing core visualization functionality with new API:
- **Output Management**: Handles output directory creation and file path management
- **Plot Configuration**: Integrates with `PlotConfig` for consistent styling
- **Plot Registry**: Each visualizer defines a `PLOT_REGISTRY` with rich metadata
- **New API Methods**: `get_available_plots()`, `get_plot_metadata()`, `generate_plot()`, `generate_plots()`, `generate_all_plots()`
- **Common Utilities**: Provides reusable methods for data validation, plot creation, and formatting
- **Save/Show Operations**: Provides `save_plot()` and `show_plot()` methods
- **Abstract Interface**: Defines `create_dashboard()` method that subclasses must implement

#### Individual Visualizers
Three specialized visualizers for different metric types with plot registries:

1. **`QiskitVisualizer`**: Handles circuit structure and instruction analysis (4 plots)
   - `circuit_structure`: Basic circuit metrics (depth, width, size, qubits)
   - `gate_distribution`: Gate type analysis and instruction distribution
   - `instruction_metrics`: Instruction-related metrics and connectivity analysis
   - `circuit_summary`: Derived metrics and summary information

2. **`ComplexityVisualizer`**: Handles complexity analysis visualization (3 plots)
   - `gate_based_metrics`: Gate counts, depth, T-gates, and CNOT gates
   - `complexity_radar`: Radar chart for normalized complexity indicators
   - `efficiency_metrics`: Parallelism efficiency and circuit efficiency analysis

3. **`CircuitPerformanceVisualizer`**: Handles execution performance visualization (4 plots)
   - `success_error_comparison`: Success vs error rate comparisons across jobs
   - `fidelity_comparison`: Fidelity analysis across different executions
   - `shot_distribution`: Distribution of successful vs failed shots
   - `aggregate_summary`: Statistical summary across multiple jobs

#### `Visualizer` (Unified Entry Point)
Main class that provides a single interface for all visualizations with new API:
- **Type-Safe Constants**: Works with `Metrics` and `Plots` constants for error prevention
- **Auto-Detection**: Automatically detects available metrics and registers appropriate visualizers
- **Granular Control**: Generate single plots, selected plots, or all plots with precise control
- **Rich Metadata**: Access detailed information about each plot's purpose and requirements
- **Flexible Input**: Works with Scanner instances or custom metrics data
- **Memory Efficient**: Default `save=False, show=False` for batch processing

#### `PlotConfig`
Configuration dataclass for plot appearance and behavior:
- **Appearance**: Figure size, DPI, color palettes, transparency
- **Styling**: Plot styles (default, quantum, minimal)
- **Output**: Save format (PNG, SVG), grid settings
- **Extensible**: Easy to customize for different visualization needs

### Integration with Scanner (New API)

The visualization system seamlessly integrates with the Scanner output using the new type-safe API:

```python
from qward.visualization.constants import Metrics, Plots

# Calculate metrics
scanner = Scanner(circuit=circuit)
scanner.add_strategy(QiskitMetrics(circuit))
scanner.add_strategy(ComplexityMetrics(circuit))
scanner.add_strategy(CircuitPerformanceMetrics(circuit=circuit, job=job))
metrics_dict = scanner.calculate_metrics()

# Option 1: Use unified Visualizer with type-safe constants (recommended)
visualizer = Visualizer(scanner=scanner)

# NEW API: Generate specific plots with type-safe constants
selected_plots = visualizer.generate_plots({
    Metrics.QISKIT: [
        Plots.QISKIT.CIRCUIT_STRUCTURE,
        Plots.QISKIT.GATE_DISTRIBUTION
    ],
    Metrics.COMPLEXITY: [
        Plots.COMPLEXITY.COMPLEXITY_RADAR
    ]
}, save=True, show=False)

# NEW API: Generate all plots for specific metrics
all_qiskit_plots = visualizer.generate_plots({
    Metrics.QISKIT: None  # None = all plots
}, save=True, show=False)

# NEW API: Generate single plot
single_plot = visualizer.generate_plot(
    Metrics.CIRCUIT_PERFORMANCE, 
    Plots.CIRCUIT_PERFORMANCE.SUCCESS_ERROR_COMPARISON, 
    save=True, 
    show=False
)

# NEW API: Explore available plots and metadata
available_plots = visualizer.get_available_plots()
for metric_name, plot_names in available_plots.items():
    print(f"\n{metric_name} ({len(plot_names)} plots):")
    for plot_name in plot_names:
        metadata = visualizer.get_plot_metadata(metric_name, plot_name)
        print(f"  - {plot_name}: {metadata.description} ({metadata.plot_type.value})")

# Create comprehensive dashboards (unchanged)
dashboards = visualizer.create_dashboard(save=True, show=False)

# Option 2: Use individual visualizers directly with new API
from qward.visualization import QiskitVisualizer, ComplexityVisualizer, CircuitPerformanceVisualizer

qiskit_viz = QiskitVisualizer({Metrics.QISKIT: metrics_dict[Metrics.QISKIT]})

# NEW API: Generate specific plots
qiskit_viz.generate_plot(Plots.QISKIT.CIRCUIT_STRUCTURE, save=True, show=False)

# NEW API: Generate selected plots
selected_figures = qiskit_viz.generate_plots([
    Plots.QISKIT.CIRCUIT_STRUCTURE,
    Plots.QISKIT.GATE_DISTRIBUTION
], save=True, show=False)

# NEW API: Generate all plots
all_figures = qiskit_viz.generate_all_plots(save=True, show=False)

# NEW API: Get plot metadata
metadata = qiskit_viz.get_plot_metadata(Plots.QISKIT.CIRCUIT_STRUCTURE)
print(f"Plot description: {metadata.description}")
print(f"Plot type: {metadata.plot_type.value}")
print(f"Dependencies: {metadata.dependencies}")
```

### Extensibility

The visualization system is designed for easy extension with the new plot registry system:

1. **Custom Visualizers**: Create new visualizers by inheriting from `VisualizationStrategy` and defining a `PLOT_REGISTRY`
2. **Plot Metadata**: Rich metadata system provides detailed information about each plot
3. **Type Safety**: Constants system prevents errors and provides IDE autocompletion
4. **Custom Plot Types**: Add new visualization methods with proper metadata registration
5. **Custom Styling**: Define new `PlotConfig` presets for different use cases
6. **Registration System**: Register custom visualizers with the unified `Visualizer` class

### Benefits of New API

1. **Type Safety**: Constants prevent typos and provide IDE autocompletion
2. **Rich Metadata**: Detailed information about each plot's purpose and requirements
3. **Granular Control**: Generate exactly the plots you need
4. **Memory Efficient**: Default parameters optimized for batch processing
5. **Error Prevention**: Validation of metric and plot combinations
6. **Discoverability**: Easy exploration of available plots and their capabilities
7. **Future-Proof**: Extensible design for adding new visualization types

For detailed usage and examples, see the [Visualization Guide](visualization_guide.md) and explore the comprehensive examples in [`qward/examples/`](../qward/examples/). 