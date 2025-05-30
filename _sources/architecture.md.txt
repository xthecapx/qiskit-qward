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
        +calculate_metrics() DataFrame
    }

    class MetricCalculator {
        <<Strategy Interface>>
        +get_metrics() Dict[str, Any]
        +get_structured_metrics() BaseModel
    }

    class QiskitMetrics {
        <<Concrete Strategy>>
        +get_metrics() Dict[str, Any]
        +get_structured_metrics() QiskitMetricsSchema
        +get_structured_basic_metrics() BasicMetricsSchema
        +get_structured_instruction_metrics() InstructionMetricsSchema
        +get_structured_scheduling_metrics() SchedulingMetricsSchema
    }

    class ComplexityMetrics {
        <<Concrete Strategy>>
        +get_metrics() Dict[str, Any]
        +get_structured_metrics() ComplexityMetricsSchema
        +get_structured_gate_based_metrics() GateBasedMetricsSchema
        +get_structured_entanglement_metrics() EntanglementMetricsSchema
        +get_structured_standardized_metrics() StandardizedMetricsSchema
        +get_structured_advanced_metrics() AdvancedMetricsSchema
        +get_structured_derived_metrics() DerivedMetricsSchema
        +get_structured_quantum_volume() QuantumVolumeSchema
    }

    class CircuitPerformance {
        <<Concrete Strategy>>
        +get_metrics() Dict[str, Any]
        +get_structured_metrics() Union[CircuitPerformanceJobSchema, CircuitPerformanceAggregateSchema]
        +get_structured_single_job_metrics() CircuitPerformanceJobSchema
        +get_structured_multiple_jobs_metrics() CircuitPerformanceAggregateSchema
    }

    class SchemaModule {
        <<Validation Layer>>
        +QiskitMetricsSchema
        +ComplexityMetricsSchema
        +CircuitPerformanceJobSchema
        +CircuitPerformanceAggregateSchema
        +validate_data()
        +generate_json_schema()
    }

    %% Strategy Pattern Relationships
    Scanner --> MetricCalculator : uses strategies
    
    %% Strategy Interface Implementation
    MetricCalculator <|.. QiskitMetrics : implements
    MetricCalculator <|.. ComplexityMetrics : implements
    MetricCalculator <|.. CircuitPerformance : implements
    
    %% Schema Integration
    QiskitMetrics --> SchemaModule : validates with
    ComplexityMetrics --> SchemaModule : validates with
    CircuitPerformance --> SchemaModule : validates with

    %% Pattern Notes
    note for Scanner "Context: Orchestrates metric calculation and returns consolidated DataFrame"
    note for MetricCalculator "Strategy Interface: Common interface with both Dict and Schema outputs"
    note for QiskitMetrics "Returns validated Qiskit-native metrics with type safety"
    note for ComplexityMetrics "Returns validated complexity analysis metrics with constraints"
    note for CircuitPerformance "Returns validated execution performance metrics with cross-field validation"
    note for SchemaModule "Pydantic-based validation with automatic type checking and JSON schema generation"
```

### Key Points

- **Scanner (Context)**: Maintains metric calculators and delegates calculation work
- **MetricCalculator (Interface)**: Defines common interface with both dictionary and schema outputs
- **Concrete Strategies**: Each implements different metric calculation algorithms with validation
- **Schema Validation**: Pydantic-based data validation ensures type safety and constraint checking
- **Dual API**: Both traditional dictionary and modern schema-based approaches supported
- **Strategy Pattern Benefits**: Runtime strategy switching, extensibility, separation of concerns

## Schema-Based Data Validation

QWARD now includes comprehensive schema-based validation using Pydantic, providing:

```{mermaid}
classDiagram
    %% Schema Validation Architecture
    
    class BaseModel {
        <<Pydantic Base>>
        +model_validate()
        +model_json_schema()
        +model_dump()
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
        +quantum_volume: QuantumVolumeSchema
        +to_flat_dict() Dict[str, Any]
        +from_flat_dict() ComplexityMetricsSchema
    }
    
    class CircuitPerformanceJobSchema {
        +job_id: str
        +success_rate: float [0.0-1.0]
        +error_rate: float [0.0-1.0]
        +fidelity: float [0.0-1.0]
        +total_shots: int [≥0]
        +successful_shots: int [≥0]
        +validate_error_rate()
        +validate_successful_shots()
    }
    
    class CircuitPerformanceAggregateSchema {
        +mean_success_rate: float [0.0-1.0]
        +std_success_rate: float [≥0.0]
        +min_success_rate: float [0.0-1.0]
        +max_success_rate: float [0.0-1.0]
        +total_trials: int [≥0]
        +fidelity: float [0.0-1.0]
        +error_rate: float [0.0-1.0]
        +validate_min_max_order()
        +validate_error_rate_consistency()
    }

    %% Inheritance
    BaseModel <|-- QiskitMetricsSchema
    BaseModel <|-- ComplexityMetricsSchema
    BaseModel <|-- CircuitPerformanceJobSchema
    BaseModel <|-- CircuitPerformanceAggregateSchema

    %% Validation Features
    note for QiskitMetricsSchema "Validates circuit metrics with type checking and constraints"
    note for ComplexityMetricsSchema "Validates complexity metrics with range and cross-field validation"
    note for CircuitPerformanceJobSchema "Validates single job metrics with rate consistency checks"
    note for CircuitPerformanceAggregateSchema "Validates aggregate metrics with statistical constraints"
```

### Schema Benefits

1. **Type Safety**: Automatic type checking and validation
2. **Constraint Validation**: Range checks, cross-field validation, and business rules
3. **IDE Support**: Full autocomplete and type hints
4. **API Documentation**: Automatic JSON schema generation
5. **DataFrame Compatibility**: Easy conversion to/from flat dictionaries
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
        +result: Result
        +metrics: List[MetricCalculator]
        +__init__(circuit, job, result, metrics)
        +add_metric(metric_calculator)
        +calculate_metrics() Dict[str, DataFrame]
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
        +get_metrics()* Dict[str, Any]
        +_ensure_schemas_available()
    }

    class QiskitMetrics {
        <<Concrete Strategy>>
        +__init__(circuit)
        +_get_metric_type() MetricsType
        +_get_metric_id() MetricsId
        +is_ready() bool
        +get_metrics() Dict[str, Any]
        +get_structured_metrics() QiskitMetricsSchema
        +get_basic_metrics() Dict[str, Any]
        +get_instruction_metrics() Dict[str, Any]
        +get_scheduling_metrics() Dict[str, Any]
        +get_structured_basic_metrics() BasicMetricsSchema
        +get_structured_instruction_metrics() InstructionMetricsSchema
        +get_structured_scheduling_metrics() SchedulingMetricsSchema
    }

    class ComplexityMetrics {
        <<Concrete Strategy>>
        +__init__(circuit)
        +_get_metric_type() MetricsType
        +_get_metric_id() MetricsId
        +is_ready() bool
        +get_metrics() Dict[str, Any]
        +get_structured_metrics() ComplexityMetricsSchema
        +get_gate_based_metrics() Dict[str, Any]
        +get_entanglement_metrics() Dict[str, Any]
        +get_standardized_metrics() Dict[str, Any]
        +get_advanced_metrics() Dict[str, Any]
        +get_derived_metrics() Dict[str, Any]
        +estimate_quantum_volume() Dict[str, Any]
        +get_structured_gate_based_metrics() GateBasedMetricsSchema
        +get_structured_entanglement_metrics() EntanglementMetricsSchema
        +get_structured_standardized_metrics() StandardizedMetricsSchema
        +get_structured_advanced_metrics() AdvancedMetricsSchema
        +get_structured_derived_metrics() DerivedMetricsSchema
        +get_structured_quantum_volume() QuantumVolumeSchema
    }

    class CircuitPerformance {
        <<Concrete Strategy>>
        #_job: Optional[JobType]
        #_jobs: List[JobType]
        #_result: Optional[Dict]
        #success_criteria: Callable[[str], bool]
        +__init__(circuit, job, jobs, result, success_criteria)
        +_get_metric_type() MetricsType
        +_get_metric_id() MetricsId
        +is_ready() bool
        +get_metrics() Dict[str, Any]
        +get_structured_metrics() Union[CircuitPerformanceJobSchema, CircuitPerformanceAggregateSchema]
        +get_single_job_metrics(job) Dict[str, Any]
        +get_multiple_jobs_metrics() Dict[str, Any]
        +get_structured_single_job_metrics(job) CircuitPerformanceJobSchema
        +get_structured_multiple_jobs_metrics() CircuitPerformanceAggregateSchema
        +add_job(job)
        +_calculate_success_metrics(counts, job_id) Dict[str, Any]
        +_calculate_aggregate_metrics(rates, fidelities, shots) Dict[str, Any]
        +_extract_job_id(job) str
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
    Scanner --> Result : manages
    
    %% Strategy Interface Implementation
    MetricCalculator <|.. QiskitMetrics : implements
    MetricCalculator <|.. ComplexityMetrics : implements
    MetricCalculator <|.. CircuitPerformance : implements
    
    %% Strategy Interface Dependencies
    MetricCalculator --> MetricsType : defines type
    MetricCalculator --> MetricsId : defines identifier

    %% Notes about Enhanced Architecture
    note for Scanner "Context: Maintains references to metric calculators and delegates work. Returns DataFrames for analysis."
    
    note for MetricCalculator "Strategy Interface: Common interface with both Dict and Schema outputs. Includes schema availability checking."
    
    note for QiskitMetrics "Concrete Strategy: Qiskit-native metrics with granular structured methods for each category"
    
    note for ComplexityMetrics "Concrete Strategy: Comprehensive complexity analysis with validated schemas and constraint checking"
    
    note for CircuitPerformance "Concrete Strategy: Circuit performance analysis with single/multiple job support and custom criteria"
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
│   ├── base_metric.py          # Base MetricCalculator class
│   ├── types.py                # MetricsType and MetricsId enums
│   ├── defaults.py             # Default metric configurations
│   ├── schemas.py              # Pydantic schema definitions
│   ├── qiskit_metrics.py       # QiskitMetrics implementation
│   ├── complexity_metrics.py   # ComplexityMetrics implementation
│   └── circuit_performance.py  # CircuitPerformance implementation
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
    └── example_metrics_constructor.py # Example for custom metrics constructor
```

This structure provides a clean organization for the code, with:

1. **Main Package**: Core classes at the top level for easy imports
2. **Runtime Module**: Handles execution of quantum circuits
3. **Metrics Module**: Contains all metric implementations and schema definitions
4. **Utils Module**: Helper functions and utilities
5. **Examples Module**: Working code examples demonstrating library usage including schema validation

## Components

### Scanner
The Scanner class is the main entry point for analyzing quantum circuits. It can be initialized with a quantum circuit, job, result, and an optional list of metric classes or instances. It allows users to add further metrics and calculate them, returning results as DataFrames for easy analysis.

### Result
The Result class represents the output of a quantum circuit execution. It includes the job information, measurement counts, and metadata. It provides methods for saving and loading results, as well as updating results from a job.

### MetricCalculator
The MetricCalculator class is an abstract base class that defines the interface for all metrics. It includes the circuit attribute, properties for metric type and ID, and abstract methods for metric calculation. All concrete implementations now support both traditional dictionary outputs and modern schema-based validation. Default metric classes can be obtained using the `get_default_metrics()` function from the `qward.metrics.defaults` module.

### Schema Validation
The schema validation system provides:
- **Type Safety**: Automatic validation of data types and constraints
- **Business Rules**: Cross-field validation (e.g., error_rate = 1 - success_rate)
- **Range Validation**: Ensures values are within expected bounds
- **Documentation**: Automatic JSON schema generation for API documentation
- **IDE Support**: Full autocomplete and type hints for better developer experience

### QiskitMetrics
The QiskitMetrics class extracts metrics directly from QuantumCircuit objects, including basic metrics (depth, width, gate counts), instruction metrics (connectivity, factors), and scheduling metrics (timing information). It now provides both dictionary and structured schema outputs with granular access to each metric category.

### ComplexityMetrics
The ComplexityMetrics class calculates comprehensive circuit complexity metrics based on research literature, including gate-based metrics, entanglement metrics, standardized metrics, advanced metrics, derived metrics, and quantum volume estimation. All metrics are validated through schemas with appropriate constraints and cross-field validation.

### CircuitPerformance
The CircuitPerformance class calculates performance metrics for quantum circuits, such as success rate, fidelity, and error rate. It supports both single job and multiple job analysis with customizable success criteria. The class now provides structured outputs with validation for both individual job metrics and aggregate statistics across multiple jobs.

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
scanner.add_metric(QiskitMetrics(circuit))

# Calculate metrics (traditional approach)
results = scanner.calculate_metrics()

# Use structured metrics (new schema-based approach)
qiskit_metrics = QiskitMetrics(circuit)
structured_metrics = qiskit_metrics.get_structured_metrics()

# Access validated data with full type safety
print(f"Circuit depth: {structured_metrics.basic_metrics.depth}")
print(f"Gate count: {structured_metrics.basic_metrics.size}")
print(f"Number of qubits: {structured_metrics.basic_metrics.num_qubits}")
```

### Comprehensive Analysis with Multiple Metrics
```python
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformance

# Create a scanner with multiple metrics
scanner = Scanner(circuit=circuit)
scanner.add_metric(QiskitMetrics(circuit))
scanner.add_metric(ComplexityMetrics(circuit))

# For circuit performance, you need job execution results
if job:  # Assuming you have a job from circuit execution
    scanner.add_metric(CircuitPerformance(circuit, job=job))

# Calculate all metrics
results = scanner.calculate_metrics()

# Access structured metrics for detailed analysis
complexity_metrics = ComplexityMetrics(circuit)
complexity_schema = complexity_metrics.get_structured_metrics()

print(f"Quantum Volume: {complexity_schema.quantum_volume.enhanced_quantum_volume}")
print(f"Gate Density: {complexity_schema.standardized_metrics.gate_density}")
print(f"Parallelism Efficiency: {complexity_schema.advanced_metrics.parallelism_efficiency}")
```

### Circuit Performance Analysis with Custom Criteria
```python
from qward.metrics import CircuitPerformance

# Define custom success criteria
def bell_state_success(result: str) -> bool:
    clean_result = result.replace(" ", "")
    return clean_result in ["0000", "1111"]  # |00⟩ or |11⟩ states

# Create circuit performance metric with custom criteria
circuit_performance = CircuitPerformance(
    circuit=circuit, 
    job=job, 
    success_criteria=bell_state_success
)

# Get structured metrics with validation
if len(circuit_performance.runtime_jobs) == 1:
    job_schema = circuit_performance.get_structured_single_job_metrics()
    print(f"Success Rate: {job_schema.success_rate:.3f}")
    print(f"Fidelity: {job_schema.fidelity:.3f}")
else:
    aggregate_schema = circuit_performance.get_structured_multiple_jobs_metrics()
    print(f"Mean Success Rate: {aggregate_schema.mean_success_rate:.3f}")
    print(f"Standard Deviation: {aggregate_schema.std_success_rate:.3f}")
```

### Schema Validation and JSON Generation
```python
from qward.metrics.schemas import ComplexityMetricsSchema
import json

# Generate JSON schema for API documentation
complexity_metrics = ComplexityMetrics(circuit)
schema_obj = complexity_metrics.get_structured_metrics()

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
from typing import Dict, Any

class MyCustomMetric(MetricCalculator):
    def __init__(self, circuit: QuantumCircuit):
        super().__init__(circuit)
    
    def _get_metric_type(self) -> MetricsType:
        return MetricsType.PRE_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        return MetricsId.QISKIT  # Or define a new ID
    
    def is_ready(self) -> bool:
        return self.circuit is not None
    
    def get_metrics(self) -> Dict[str, Any]:
        # Custom metric calculation
        custom_value = self.circuit.depth() * self.circuit.num_qubits
        return {
            "custom_complexity": custom_value,
            "circuit_signature": f"{self.circuit.num_qubits}q_{self.circuit.depth()}d"
        }

# Usage
scanner = Scanner(circuit=circuit)
scanner.add_metric(MyCustomMetric(circuit))
results = scanner.calculate_metrics()
```

## Best Practices

1. **Circuit Analysis**
   - Use the Scanner class for all circuit analysis
   - Add metrics before calculating results
   - Consider using multiple metrics for comprehensive analysis
   - Prefer structured metrics for type safety and validation

2. **Schema Usage**
   - Use structured methods when you need type safety and validation
   - Use traditional dictionary methods for backward compatibility
   - Leverage JSON schema generation for API documentation
   - Take advantage of IDE autocomplete with schema objects

3. **Execution**
   - Handle job and result errors appropriately
   - Use appropriate Qiskit runtime services for backend execution
   - Validate success criteria for CircuitPerformance metrics

4. **Result Management**
   - Save results for later analysis
   - Include relevant metadata with results
   - Use consistent naming conventions for saved results
   - Leverage schema validation to ensure data integrity

5. **Custom Metrics**
   - Inherit from the MetricCalculator base class
   - Implement the required abstract methods
   - Return results in a consistent format
   - Document metric calculation methodology
   - Consider adding schema validation for custom metrics

6. **Performance**
   - Schema validation adds minimal overhead but provides significant benefits
   - Use flat dictionary conversion for DataFrame operations
   - Cache structured metrics when performing multiple analyses

## Visualization System

QWARD includes a comprehensive visualization system for analyzing and presenting quantum circuit metrics. The visualization module follows a structured approach that integrates seamlessly with the metric calculation system.

### Architecture Overview

The visualization system is built on the following key components:

- **`PlotConfig`**: A dataclass holding all plot appearance and saving configurations.
- **`BaseVisualizer`**: An abstract base class for all visualizers. It handles common setup (output directory, styling via `PlotConfig`) and provides `save_plot`/`show_plot` methods. Subclasses must implement `create_plot()` for their specific visualization logic.
- **`CircuitPerformanceVisualizer`**: A concrete visualizer inheriting from `BaseVisualizer`. It's responsible for generating various plots related to circuit performance metrics with direct plotting methods (no longer using Strategy pattern for simplicity).

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
        +plot_success_error_comparison()
        +plot_fidelity_comparison()
        +plot_shot_distribution()
        +plot_aggregate_summary()
        +create_dashboard()
        +plot_all()
    }

    BaseVisualizer <|-- CircuitPerformanceVisualizer
    BaseVisualizer --> PlotConfig : uses
    
    note for BaseVisualizer "Abstract base class providing core visualization functionality"
    note for CircuitPerformanceVisualizer "Concrete implementation with direct plotting methods for simplicity"
    note for PlotConfig "Configuration dataclass for plot appearance and saving options"
```

### Key Components (Summary)

1. **`BaseVisualizer`**: Abstract base class providing core visualization functionality.
2. **`CircuitPerformanceVisualizer`**: Concrete implementation for `CircuitPerformance` metrics with simplified direct plotting methods.
3. **`PlotConfig`**: Dataclass for easy plot appearance customization.

### Integration with Scanner

The visualization system seamlessly integrates with the Scanner output:

```python
# Calculate metrics
scanner = Scanner(circuit=circuit)
scanner.add_metric(CircuitPerformance(circuit=circuit, job=job))
metrics_dict = scanner.calculate_metrics()

# Create visualizations
visualizer = CircuitPerformanceVisualizer(metrics_dict)
figures = visualizer.plot_all(save=True)
```

### Extensibility

The visualization system is designed for easy extension:

1. **Custom Visualizers**: Create new visualizers by inheriting from `BaseVisualizer`
2. **Custom Styles**: Define new plot styles and color palettes
3. **New Plot Types**: Add new visualization methods to existing visualizers

For detailed usage and examples, see the [Visualization Guide](visualization_guide.md). 