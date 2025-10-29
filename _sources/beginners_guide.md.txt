# QWARD Beginner's Guide

This guide provides a comprehensive introduction to QWARD, a Python library for analyzing quantum circuits and their execution results.

## What is QWARD?

QWARD is a library built on top of Qiskit that helps quantum developers understand how their quantum algorithms perform on both simulators and real quantum hardware. It provides tools to:

1. Define and execute quantum circuits using Qiskit
2. Collect execution data like counts from simulators or hardware jobs
3. Analyze circuits and results using built-in **metric calculators**
4. Assess circuit properties, complexity, and estimate potential performance
5. Validate data integrity with schema-based validation using Pydantic

## Key Concepts

### Scanner
The `qward.Scanner` class is the central component for orchestrating circuit analysis. You provide it with a `QuantumCircuit` and optionally an execution `Job`. You then add various **metric calculator** objects to the `Scanner` to perform different types of analysis.

### Metric Calculators
Metric calculators are classes that perform specific calculations or data extraction based on a circuit, a job, or a result. QWARD provides several built-in metric calculators:

- `QiskitMetrics`: Extracts basic properties directly available from a `QuantumCircuit` object (e.g., depth, width, gate counts)
- `ComplexityMetrics`: Calculates a wide range of complexity indicators, including those from "Character Complexity: A Novel Measure for Quantum Circuit Analysis" by D. Shami
- `CircuitPerformanceMetrics`: Calculates success rates, error rates, and fidelity based on execution counts from a job, given a user-defined success criterion

You can also create your own custom metric calculators by subclassing `qward.metrics.base_metric.MetricCalculator`.

### Unified API
All metric calculators use the same simple interface:

```python
calculator = QiskitMetrics(circuit)        # or ComplexityMetrics(circuit)
metrics = calculator.get_metrics()         # Returns validated schema object
depth = metrics.basic_metrics.depth       # Type-safe access with IDE support
```

### Schema-Based Validation
QWARD provides comprehensive data validation using Pydantic schemas, offering:

- **Type Safety**: Automatic validation of data types and constraints
- **Business Rules**: Cross-field validation (e.g., error_rate = 1 - success_rate)
- **Range Validation**: Ensures values are within expected bounds (e.g., success rates between 0.0-1.0)
- **IDE Support**: Full autocomplete and type hints for better developer experience
- **API Documentation**: Automatic JSON schema generation

All metric calculators return validated schema objects directly through `get_metrics()`, providing type safety and data integrity by default.

## Getting Started

### Installation

You can set up QWARD in two ways:

#### Option 1: Local Installation

```bash
# Clone the repository
git clone https://github.com/your-org/qiskit-qward.git
cd qiskit-qward

# Install in development mode
pip install -e .

# Set up IBM Quantum credentials
cp .env.example .env
# Edit .env with your IBM Quantum token
```

#### Option 2: Using Docker

```bash
# Clone the repository
git clone https://github.com/your-org/qiskit-qward.git
cd qiskit-qward

# Copy and edit .env file
cp .env.example .env
# Edit .env with your IBM Quantum token

# Start Docker container with Jupyter Lab
chmod +x start.sh
./start.sh
```

This will open a Jupyter Lab interface in your browser where you can run the examples and tutorials.

### First Steps: The Quantum Coin Flip

Let's analyze a simple quantum coin flip circuit. This uses a single qubit in superposition.

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics
from qward.examples.utils import create_example_circuit, get_display

display = get_display()

# 1. Create a quantum circuit (2-qubit GHZ state from examples.utils)
circuit = create_example_circuit()

print("Quantum Circuit (2-qubit GHZ from examples):")
display(circuit.draw(output='mpl'))

# 2. Simulate the circuit to get results (optional for some calculators)
print("\nRunning quantum simulation...")
simulator = AerSimulator()
job = simulator.run(circuit, shots=1024)

# 3. Create a Scanner instance
scanner = Scanner(circuit=circuit, job=job)

# 4. Add Metric Calculators
scanner.add_strategy(QiskitMetrics(circuit=circuit))
scanner.add_strategy(ComplexityMetrics(circuit=circuit))

# Define success for the first qubit being '0'
def coin_flip_success_q0_is_0(bitstring):
    return bitstring.endswith('0')  # True if q0 is '0'

scanner.add_strategy(CircuitPerformanceMetrics(
    circuit=circuit, 
    job=job, 
    success_criteria=coin_flip_success_q0_is_0
))

# 5. Calculate all added calculators
print("\nCalculating metrics...")
all_metrics_results = scanner.calculate_metrics()

# 6. Display results
print("\n--- Metric Results ---")
for metric_name, df in all_metrics_results.items():
    print(f"\n{metric_name} DataFrame:")
    display(df)

# 7. Use schema-based API for type-safe access
print("\n--- Schema-Based API (Type-Safe Access) ---")

# QiskitMetrics with schema validation
qiskit_metrics = QiskitMetrics(circuit)
metrics = qiskit_metrics.get_metrics()  # Returns QiskitMetricsSchema

print("✅ QiskitMetrics Schema:")
print(f"  Circuit Depth: {metrics.basic_metrics.depth}")
print(f"  Number of Qubits: {metrics.basic_metrics.num_qubits}")
print(f"  Gate Count: {metrics.basic_metrics.size}")
print(f"  Width: {metrics.basic_metrics.width}")

# ComplexityMetrics with schema validation
complexity_metrics = ComplexityMetrics(circuit)
complexity_schema = complexity_metrics.get_metrics()  # Returns ComplexityMetricsSchema

print("\n✅ ComplexityMetrics Schema:")
print(f"  Gate Count: {complexity_schema.gate_based_metrics.gate_count}")
print(f"  T-gate Count: {complexity_schema.gate_based_metrics.t_count}")
print(f"  Circuit Volume: {complexity_schema.standardized_metrics.circuit_volume}")
print(f"  Parallelism Factor: {complexity_schema.advanced_metrics.parallelism_factor:.3f}")
print(f"  Weighted Complexity: {complexity_schema.derived_metrics.weighted_complexity}")

# CircuitPerformance with schema validation
circuit_performance = CircuitPerformanceMetrics(
    circuit=circuit, 
    job=job, 
    success_criteria=coin_flip_success_q0_is_0
)
performance_schema = circuit_performance.get_metrics()  # Returns CircuitPerformanceSchema

print("\n✅ CircuitPerformanceMetrics Schema:")
print(f"  Success Rate: {performance_schema.success_metrics.success_rate:.3f}")
print(f"  Error Rate: {performance_schema.success_metrics.error_rate:.3f}")  # Automatically validated
print(f"  Fidelity: {performance_schema.fidelity_metrics.fidelity:.3f}")
print(f"  Total Shots: {performance_schema.success_metrics.total_shots}")
print(f"  Successful Shots: {performance_schema.success_metrics.successful_shots}")
```

### Understanding Schema Validation

QWARD provides structured, validated metrics through schema objects:

```python
# Schema validation in action
try:
    from qward.metrics.schemas import CircuitPerformanceSchema
    
    print("\n🔍 Schema Validation Demo:")
    # This will raise ValidationError because success_rate > 1.0
    invalid_data = CircuitPerformanceSchema(
        success_metrics={
            "success_rate": 1.5,  # Invalid!
            "error_rate": 0.25,
            "total_shots": 1000,
            "successful_shots": 800
        },
        fidelity_metrics={
            "fidelity": 0.8,
            "method": "theoretical_comparison",
            "confidence": "high"
        },
        statistical_metrics={
            "entropy": 0.5,
            "uniformity": 0.6,
            "concentration": 0.4,
            "dominant_outcome_probability": 0.7,
            "num_unique_outcomes": 2
        }
    )
except Exception as e:
    print(f"✅ Validation caught error: {type(e).__name__} - Success rate cannot exceed 1.0")

# Benefits of schema approach:
# 1. Type safety and IDE autocomplete
# 2. Automatic validation (e.g., depth must be >= 0)
# 3. Cross-field validation (e.g., error_rate = 1 - success_rate)
# 4. JSON schema generation for API documentation
```

### Alternative: Using Constructor with Calculators

You can also provide calculators directly in the Scanner constructor:

```python
# Using calculator classes (will be instantiated automatically)
scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics])

# Using calculator instances
qm = QiskitMetrics(circuit)
cm = ComplexityMetrics(circuit)
scanner = Scanner(circuit=circuit, strategies=[qm, cm])

# Calculate metrics
all_metrics_results = scanner.calculate_metrics()
```

### Understanding the Circuit

The example `create_example_circuit()` prepares a 2-qubit GHZ state:
```
     ┌───┐     ┌─┐   
q_0: ┤ H ├──■──┤M├───
     └───┘┌─┴─┐└╥┘┌─┐
q_1: ─────┤ X ├─╫─┤M├
          └───┘ ║ └╥┘
c: 2/═══════════╩══╩═
                0  1 
```

1. **H gate on q_0**: Puts the first qubit into superposition
2. **CX gate (CNOT)**: Entangles q_0 and q_1. If q_0 is |1⟩, q_1 is flipped
3. **Measurement**: Collapses the superposition. The expected outcomes are |00⟩ and |11⟩ with roughly equal probability

## Going Further: A More Complex Circuit

Let's analyze a 3-qubit GHZ state circuit:

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics
from qward.examples.utils import get_display

display = get_display()

# 1. Create a more complex quantum circuit (3-qubit GHZ state)
circuit = QuantumCircuit(3, 3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(0, 2)
circuit.measure([0,1,2], [0,1,2])

print("3-qubit GHZ Circuit:")
display(circuit.draw(output='mpl'))

# 2. Simulate (optional, needed for CircuitPerformanceMetrics)
simulator = AerSimulator()
job = simulator.run(circuit, shots=1024)

# 3. Create Scanner and add Calculators
scanner = Scanner(circuit=circuit, job=job)
scanner.add_strategy(QiskitMetrics(circuit))
scanner.add_strategy(ComplexityMetrics(circuit))

# Example CircuitPerformanceMetrics: success if all qubits are '0' (state '000')
def all_zeros(bitstring):
    return bitstring.replace(" ", "") == '000'

scanner.add_strategy(CircuitPerformanceMetrics(
    circuit=circuit, 
    job=job, 
    success_criteria=all_zeros
))

# 4. Calculate and display metrics
all_metrics_results = scanner.calculate_metrics()
print("\n--- Metrics for 3-qubit GHZ ---")
for metric_name, df in all_metrics_results.items():
    print(f"\n{metric_name} DataFrame:")
    display(df)

# 5. Use schema-based metrics for type-safe access
complexity_metrics = ComplexityMetrics(circuit)
complexity_schema = complexity_metrics.get_metrics()

print(f"\nSchema-Based Complexity Analysis:")
print(f"  Gate Count: {complexity_schema.gate_based_metrics.gate_count}")
print(f"  Circuit Depth: {complexity_schema.gate_based_metrics.circuit_depth}")
print(f"  CNOT Count: {complexity_schema.gate_based_metrics.cnot_count}")
print(f"  Gate Density: {complexity_schema.standardized_metrics.gate_density:.3f}")
print(f"  Parallelism Efficiency: {complexity_schema.advanced_metrics.parallelism_efficiency:.3f}")
print(f"  Entangling Gate Density: {complexity_schema.entanglement_metrics.entangling_gate_density:.3f}")
```

## Understanding Circuit Complexity

QWARD's `ComplexityMetrics` class provides comprehensive circuit complexity analysis. The resulting schema contains various sub-categories of metrics:

### Schema-Based Access (Recommended)
```python
# Using schema-based metrics for type safety and validation
complexity_metrics = ComplexityMetrics(circuit)
schema = complexity_metrics.get_metrics()

# Type-safe access with IDE autocomplete
print(f"Gate Count: {schema.gate_based_metrics.gate_count}")
print(f"Circuit Volume: {schema.standardized_metrics.circuit_volume}")
print(f"T-gate Count: {schema.gate_based_metrics.t_count}")
print(f"Parallelism Factor: {schema.advanced_metrics.parallelism_factor}")
print(f"Multi-qubit Ratio: {schema.gate_based_metrics.multi_qubit_ratio:.3f}")
print(f"Entangling Gate Density: {schema.entanglement_metrics.entangling_gate_density:.3f}")
```

### DataFrame Access (for analysis and visualization)
```python
# Scanner returns DataFrames for analysis
if "ComplexityMetrics" in all_metrics_results:
    complexity_df = all_metrics_results["ComplexityMetrics"]
    
    # Example: Access gate count and circuit volume
    gate_count = complexity_df['gate_based_metrics.gate_count'].iloc[0]
    circuit_volume = complexity_df['standardized_metrics.circuit_volume'].iloc[0]
    
    print(f"Gate Count: {gate_count}")
    print(f"Circuit Volume: {circuit_volume}")
```

## Circuit Performance Analysis with Schema Validation

The `CircuitPerformanceMetrics` calculator provides comprehensive validation for both single job and multiple job scenarios:

```python
from qward.metrics import CircuitPerformanceMetrics

# Create circuit performance calculator
circuit_performance = CircuitPerformanceMetrics(circuit=circuit, job=job)

# Schema-based approach with validation
metrics = circuit_performance.get_metrics()
print(f"Success Rate: {metrics.success_metrics.success_rate:.3f}")
print(f"Error Rate: {metrics.success_metrics.error_rate:.3f}")  # Automatically validated: error_rate = 1 - success_rate
print(f"Fidelity: {metrics.fidelity_metrics.fidelity:.3f}")
print(f"Successful Shots: {metrics.success_metrics.successful_shots}/{metrics.success_metrics.total_shots}")

# Schema validation catches errors automatically
try:
    from qward.metrics.schemas import CircuitPerformanceSchema
    
    # This will raise ValidationError
    invalid_schema = CircuitPerformanceSchema(
        success_metrics={
            "success_rate": 0.75,
            "error_rate": 0.30,  # Should be 0.25!
            "total_shots": 1000,
            "successful_shots": 750
        },
        fidelity_metrics={
            "fidelity": 0.8,
            "method": "theoretical_comparison",
            "confidence": "high"
        },
        statistical_metrics={
            "entropy": 0.5,
            "uniformity": 0.6,
            "concentration": 0.4,
            "dominant_outcome_probability": 0.7,
            "num_unique_outcomes": 2
        }
    )
except Exception as e:
    print(f"Schema validation caught inconsistency: {type(e).__name__}")
```

## Creating Your Own Custom Calculators

To create your own custom metric calculator, you need to inherit from `qward.metrics.base_metric.MetricCalculator` and implement its abstract methods.

```python
from qiskit import QuantumCircuit
from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsType, MetricsId
from pydantic import BaseModel

class MyCustomMetricsSchema(BaseModel):
    custom_complexity: float
    circuit_signature: str
    parameter_used: int
    gates_per_qubit: float

class MyCustomCalculator(MetricCalculator):
    def __init__(self, circuit: QuantumCircuit, an_extra_parameter: int = 0):
        super().__init__(circuit)
        self.an_extra_parameter = an_extra_parameter

    def _get_metric_type(self) -> MetricsType:
        """Return PRE_RUNTIME if it only needs the circuit, or POST_RUNTIME if it needs job results."""
        return MetricsType.PRE_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        """Return a unique identifier for this calculator."""
        return MetricsId.QISKIT  # Using existing ID for example purposes

    def is_ready(self) -> bool:
        """Return True if the calculator can be executed (e.g., circuit is present)."""
        return self.circuit is not None

    def get_metrics(self) -> MyCustomMetricsSchema:
        """Perform the custom metric calculation and return results as a validated schema."""
        # Example: Calculate something based on the circuit and the extra parameter
        custom_value = self.circuit.depth() * self.an_extra_parameter
        circuit_signature = f"{self.circuit.num_qubits}q_{self.circuit.depth()}d_{self.circuit.size()}g"
        
        return MyCustomMetricsSchema(
            custom_complexity=custom_value,
            circuit_signature=circuit_signature,
            parameter_used=self.an_extra_parameter,
            gates_per_qubit=self.circuit.size() / self.circuit.num_qubits if self.circuit.num_qubits > 0 else 0
        )

# How to use your custom calculator:
my_circuit = QuantumCircuit(2)
my_circuit.h(0)
my_circuit.cx(0,1)

custom_calculator = MyCustomCalculator(circuit=my_circuit, an_extra_parameter=5)

scanner = Scanner(circuit=my_circuit)
scanner.add_strategy(custom_calculator)

results = scanner.calculate_metrics()
print(results['MyCustomCalculator'])

# Access schema object directly
custom_metrics = custom_calculator.get_metrics()
print(f"Custom complexity: {custom_metrics.custom_complexity}")
print(f"Circuit signature: {custom_metrics.circuit_signature}")
```

## Schema Validation and JSON Generation

One of the powerful features of the schema system is automatic JSON schema generation for API documentation:

```python
from qward.metrics.schemas import ComplexityMetricsSchema, CircuitPerformanceSchema
import json

# Generate JSON schemas for documentation
complexity_json_schema = ComplexityMetricsSchema.model_json_schema()
circuit_performance_json_schema = CircuitPerformanceSchema.model_json_schema()

print("Complexity Metrics JSON Schema:")
print(json.dumps(complexity_json_schema, indent=2))

print("\nCircuit Performance JSON Schema:")
print(json.dumps(circuit_performance_json_schema, indent=2))

# These schemas can be used for:
# 1. API documentation generation
# 2. Frontend form validation
# 3. Database schema definition
# 4. Integration with other systems
```

## Best Practices

### 1. Choose the Right Approach
- **Use schema objects** for type safety, validation, and IDE support (recommended)
- **Use Scanner DataFrames** for analysis and visualization
- **Combine both approaches** as needed in your workflow

### 2. Validation and Error Handling
```python
# Always handle potential validation errors
try:
    metrics = calculator.get_metrics()
    # Use validated data with confidence
    print(f"Validated depth: {metrics.basic_metrics.depth}")
except Exception as e:
    print(f"Validation error: {e}")
```

### 3. Performance Considerations
- Schema validation adds minimal overhead but provides significant benefits
- Use `to_flat_dict()` for DataFrame operations when needed
- Cache schema objects when performing multiple analyses

### 4. Custom Success Criteria
```python
# Define robust success criteria that handle different measurement formats
def robust_success_criteria(result: str) -> bool:
    # Remove spaces and handle different formats
    clean_result = result.replace(" ", "")
    # Define your success condition
    return clean_result.startswith("00")  # Example: first two qubits are 0

# Use with CircuitPerformanceMetrics calculator
circuit_performance = CircuitPerformanceMetrics(
    circuit=circuit, 
    job=job, 
    success_criteria=robust_success_criteria
)
```

## Visualization System

QWARD includes a comprehensive visualization system that makes it easy to create beautiful, informative plots of your quantum circuit analysis.

### Quick Visualization Example

```python
from qward.visualization import Visualizer

# Calculate metrics first
scanner = Scanner(circuit=circuit, job=job)
scanner.add_strategy(QiskitMetrics(circuit))
scanner.add_strategy(ComplexityMetrics(circuit))
scanner.add_strategy(CircuitPerformanceMetrics(circuit=circuit, job=job))

# Create unified visualizer
visualizer = Visualizer(scanner=scanner, output_dir="my_analysis")

# Create comprehensive dashboards
dashboards = visualizer.create_dashboard(save=True, show=False)

# Create all individual plots
all_plots = visualizer.visualize_all(save=True, show=False)

print(f"Created {len(dashboards)} dashboards and {len(all_plots)} individual plots")
```

### Available Visualizations

QWARD provides three specialized visualizers:

#### 1. QiskitMetrics Visualizations
- **Circuit Structure**: Basic metrics like depth, width, size, and qubit count
- **Instruction Breakdown**: Gate type analysis and instruction distribution
- **Scheduling Metrics**: Timing and scheduling information

#### 2. ComplexityMetrics Visualizations
- **Gate-Based Metrics**: Gate counts, depth, and T-gate analysis
- **Complexity Radar Chart**: Normalized complexity indicators in a radar plot
- **Efficiency Metrics**: Parallelism and circuit efficiency analysis

#### 3. CircuitPerformanceMetrics Visualizations
- **Success vs Error Rates**: Comparison across different jobs
- **Fidelity Analysis**: Fidelity metrics visualization
- **Shot Distribution**: Successful vs failed shots as stacked bars
- **Aggregate Summary**: Statistical summary across multiple jobs

### Customizing Plot Appearance

```python
from qward.visualization import Visualizer, PlotConfig

# Create custom plot configuration
config = PlotConfig(
    figsize=(12, 8),           # Larger figures
    dpi=150,                   # Lower DPI for faster rendering
    style="quantum",           # Use quantum-themed styling
    color_palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],  # Custom colors
    save_format="svg",         # Save as SVG instead of PNG
    grid=True,                 # Show grid lines
    alpha=0.8                  # Transparency level
)

# Use custom configuration
visualizer = Visualizer(scanner=scanner, config=config, output_dir="custom_plots")
dashboards = visualizer.create_dashboard(save=True, show=False)
```

## Next Steps

- Explore the example scripts and notebooks in the `qward/examples/` directory:
  - `qward/examples/aer.py` - Basic Aer simulator usage
  - `qward/examples/run_on_aer.ipynb` - Interactive notebook example
  - `qward/examples/schema_demo.py` - Schema validation demonstration
  - `qward/examples/circuit_performance_demo.py` - Circuit performance analysis examples
- Check the [Technical Documentation](technical_docs.md) for more in-depth information about components
- Read the [API Documentation](apidocs/index.rst) for a complete reference to all classes and methods
- Review the [Architecture Documentation](architecture.md) to understand the library's design patterns and schema system
