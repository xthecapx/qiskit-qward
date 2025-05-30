# Quickstart Guide

Qward is a Python library for analyzing quantum circuits and their execution quality on quantum processing units (QPUs) or simulators. This guide will help you quickly get started with both traditional dictionary-based metrics and the new schema-based validation system.

## Installation

### Option 1: Local Installation

```bash
# Clone the repository
# git clone https://github.com/your-org/qiskit-qward.git # Replace with actual URL
# cd qiskit-qward

# Install in development mode (assuming you are in the project root)
pip install -e .

# Set up IBM Quantum credentials (optional, for running on IBM hardware)
# cp .env.example .env
# Edit .env with your IBM Quantum token
```

### Option 2: Using Docker

(Assuming Docker setup is configured in the project)
```bash
# Clone the repository
# git clone https://github.com/your-org/qiskit-qward.git # Replace with actual URL
# cd qiskit-qward

# Copy and edit .env file (optional, for IBM hardware)
# cp .env.example .env
# Edit .env with your IBM Quantum token

# Start Docker container with Jupyter Lab (example script name)
# chmod +x start.sh
# ./start.sh
```

## Usage

Qward revolves around the `Scanner` class, which uses various metric calculator objects to analyze Qiskit `QuantumCircuit` objects and their execution results. The library now provides both traditional dictionary outputs and modern schema-based validation for enhanced type safety and data integrity.

### Core Workflow

1.  **Create/Load a `QuantumCircuit`**: Use Qiskit to define your circuit.
2.  **(Optional) Execute the Circuit**: Run your circuit on a simulator or quantum hardware to get a Qiskit `Job` and its `Result` (containing counts).
3.  **Instantiate `qward.Scanner`**: Provide the circuit, and optionally the Qiskit `Job` and `qward.Result` (which wraps Qiskit's job result/counts).
4.  **Add Metric Calculators**: Instantiate and add desired metric calculator classes from `qward.metrics` (e.g., `QiskitMetrics`, `ComplexityMetrics`, `CircuitPerformance`) to the scanner.
5.  **Calculate Metrics**: Call `scanner.calculate_metrics()`.
6.  **Interpret Results**: The result is a dictionary of pandas DataFrames, one for each metric type.
7.  **(New) Use Schema Validation**: Access structured, validated metrics through schema objects for enhanced type safety.

### Example: Analyzing a Simple Circuit

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner, Result # QWARD classes
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformance # QWARD calculators
from qward.examples.utils import get_display # For pretty printing in notebooks

display = get_display()

# 1. Create a Quantum Circuit (e.g., 2-qubit Bell state)
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0,1], [0,1])

print("Quantum Circuit:")
display(qc.draw(output='mpl'))

# 2. Execute the Circuit (e.g., using AerSimulator)
simulator = AerSimulator()
job = simulator.run(qc, shots=2048)
qiskit_job_result = job.result()
counts = qiskit_job_result.get_counts(qc)

# 3. Instantiate QWARD Scanner and Result
# QWARD's Result can wrap Qiskit's counts and job metadata
qward_result = Result(job=job, counts=counts, metadata=qiskit_job_result.to_dict()) 

scanner = Scanner(circuit=qc, job=job, result=qward_result)

# 4. Add Metric Calculators
scanner.add_metric(QiskitMetrics(circuit=qc))
scanner.add_metric(ComplexityMetrics(circuit=qc))

# For CircuitPerformance, define what a "successful" measurement is
def success_if_00(bitstring):
    # Handle measurement results with spaces
    clean_result = bitstring.replace(" ", "")
    return clean_result == "00"

# CircuitPerformance needs a job to get counts from
scanner.add_metric(CircuitPerformance(circuit=qc, job=job, success_criteria=success_if_00))

# 5. Calculate Metrics (Traditional Approach)
all_metric_data = scanner.calculate_metrics()

# 6. Interpret Results
print("\n--- All Calculated Metrics (Traditional DataFrames) ---")
for metric_name, df in all_metric_data.items():
    print(f"\nMetric: {metric_name}")
    display(df)

# Example: Accessing specific data from ComplexityMetrics output
if "ComplexityMetrics" in all_metric_data:
    complexity_df = all_metric_data["ComplexityMetrics"]
    print("\nSelected Complexity Data:")
    print(f"  Gate Count: {complexity_df['gate_based_metrics.gate_count'].iloc[0]}")
    print(f"  Circuit Depth: {complexity_df['gate_based_metrics.circuit_depth'].iloc[0]}")
    print(f"  Enhanced QV Estimate: {complexity_df['quantum_volume.enhanced_quantum_volume'].iloc[0]}")

# Example: Accessing specific data from CircuitPerformance output
if "CircuitPerformance.aggregate" in all_metric_data:
    success_df = all_metric_data["CircuitPerformance.aggregate"]
    print("\nSuccess Rate Data (for '00'):")
    print(f"  Mean Success Rate: {success_df['mean_success_rate'].iloc[0]:.2%}")
    print(f"  Total Shots: {success_df['total_trials'].iloc[0]}")
```

### New Feature: Schema-Based Validation

Qward now provides structured, validated metrics through Pydantic schemas:

```python
# 7. Use Schema-Based Validation (New Feature)
print("\n--- Schema-Based Validation (Type-Safe Access) ---")

# QiskitMetrics with schema validation
qiskit_metrics = QiskitMetrics(qc)
try:
    # Get structured metrics with automatic validation
    qiskit_schema = qiskit_metrics.get_structured_metrics()
    
    print("✅ QiskitMetrics Schema Validation:")
    print(f"  Circuit Depth: {qiskit_schema.basic_metrics.depth}")
    print(f"  Number of Qubits: {qiskit_schema.basic_metrics.num_qubits}")
    print(f"  Gate Count: {qiskit_schema.basic_metrics.size}")
    print(f"  Multi-qubit Gates: {qiskit_schema.instruction_metrics.multi_qubit_gate_count}")
    
    # Access granular schemas
    basic_metrics = qiskit_metrics.get_structured_basic_metrics()
    print(f"  Width: {basic_metrics.width}")
    
except ImportError:
    print("❌ Pydantic not available - install pydantic for schema validation")

# ComplexityMetrics with schema validation
complexity_metrics = ComplexityMetrics(qc)
try:
    complexity_schema = complexity_metrics.get_structured_metrics()
    
    print("\n✅ ComplexityMetrics Schema Validation:")
    print(f"  Enhanced Quantum Volume: {complexity_schema.quantum_volume.enhanced_quantum_volume:.2f}")
    print(f"  Gate Density: {complexity_schema.standardized_metrics.gate_density:.3f}")
    print(f"  Parallelism Efficiency: {complexity_schema.advanced_metrics.parallelism_efficiency:.3f}")
    print(f"  T-gate Count: {complexity_schema.gate_based_metrics.t_count}")
    
    # Access individual category schemas
    gate_metrics = complexity_metrics.get_structured_gate_based_metrics()
    print(f"  Multi-qubit Ratio: {gate_metrics.multi_qubit_ratio:.3f}")
    
    qv_metrics = complexity_metrics.get_structured_quantum_volume()
    print(f"  QV Enhancement Factor: {qv_metrics.factors.enhancement_factor:.2f}")
    
except ImportError:
    print("❌ Pydantic not available - install pydantic for schema validation")

# CircuitPerformance with schema validation
circuit_performance = CircuitPerformance(circuit=qc, job=job, success_criteria=success_if_00)
try:
    # Single job analysis
    job_schema = circuit_performance.get_structured_single_job_metrics()
    
    print("\n✅ CircuitPerformance Schema Validation:")
    print(f"  Job ID: {job_schema.job_id}")
    print(f"  Success Rate: {job_schema.success_rate:.3f}")
    print(f"  Error Rate: {job_schema.error_rate:.3f}")  # Automatically validated
    print(f"  Fidelity: {job_schema.fidelity:.3f}")
    print(f"  Successful Shots: {job_schema.successful_shots}/{job_schema.total_shots}")
    
except ImportError:
    print("❌ Pydantic not available - install pydantic for schema validation")

# Demonstrate validation in action
try:
    from qward.metrics.schemas import CircuitPerformanceJobSchema
    
    print("\n🔍 Schema Validation Demo:")
    # This will raise ValidationError because success_rate > 1.0
    invalid_data = CircuitPerformanceJobSchema(
        job_id="test",
        success_rate=1.5,  # Invalid!
        error_rate=0.25,
        fidelity=0.8,
        total_shots=1000,
        successful_shots=800
    )
except Exception as e:
    print(f"✅ Validation caught error: {type(e).__name__} - Success rate cannot exceed 1.0")
```

### Alternative: Using Constructor with Calculators

You can also provide calculators directly in the Scanner constructor:

```python
# Using calculator classes (will be instantiated automatically)
scanner = Scanner(circuit=qc, metrics=[QiskitMetrics, ComplexityMetrics])

# Using calculator instances
qm = QiskitMetrics(qc)
cm = ComplexityMetrics(qc)
scanner = Scanner(circuit=qc, metrics=[qm, cm])

# Calculate metrics
all_metric_data = scanner.calculate_metrics()
```

### Creating Custom Calculators

To create your own metric calculator, inherit from `qward.metrics.base_metric.MetricCalculator` and implement the required abstract methods:

```python
from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsType, MetricsId
from qiskit import QuantumCircuit
from typing import Dict, Any

class MySimpleCustomCalculator(MetricCalculator):
    def __init__(self, circuit: QuantumCircuit):
        super().__init__(circuit)

    def _get_metric_type(self) -> MetricsType:
        return MetricsType.PRE_RUNTIME # Only needs the circuit

    def _get_metric_id(self) -> MetricsId:
        # Ideally, add a new ID to MetricsId enum. For now, reuse for example.
        return MetricsId.QISKIT # Placeholder

    def is_ready(self) -> bool:
        return self.circuit is not None

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "custom_depth_plus_width": self.circuit.depth() + self.circuit.width(),
            "gates_per_qubit": self.circuit.size() / self.circuit.num_qubits if self.circuit.num_qubits > 0 else 0,
            "circuit_signature": f"{self.circuit.num_qubits}q_{self.circuit.depth()}d_{self.circuit.size()}g"
        }

# Usage:
custom_calculator = MySimpleCustomCalculator(qc)
scanner.add_metric(custom_calculator)
results = scanner.calculate_metrics() 
print(results['MySimpleCustomCalculator'])

# You can also add schema validation to your custom calculator
# by creating Pydantic schemas and implementing get_structured_metrics()
```

## Key Metrics Provided

Qward, through its built-in metric calculator classes, offers insights into:

### 1. Circuit Structure (`QiskitMetrics`)
   - **Basic metrics**: Depth, width, number of qubits/clbits, operations count
   - **Instruction metrics**: Multi-qubit gate count, connectivity analysis
   - **Scheduling metrics**: Timing and resource information
   - **Schema support**: Type-safe access with `get_structured_basic_metrics()`, `get_structured_instruction_metrics()`, etc.

### 2. Circuit Complexity (`ComplexityMetrics`)
   - **Gate-based metrics**: Gate counts, T-count, CNOT count, multi-qubit ratios
   - **Entanglement metrics**: Entangling gate density, entangling width
   - **Standardized metrics**: Circuit volume, gate density, Clifford ratio
   - **Advanced metrics**: Parallelism factor, circuit efficiency
   - **Derived metrics**: Weighted complexity scores
   - **Schema support**: Comprehensive validation with range checks and cross-field validation

### 3. Quantum Volume Estimation (within `ComplexityMetrics`)
   - **Standard QV**: 2^effective_depth calculation
   - **Enhanced QV**: Factoring in density, squareness, and multi-qubit gate ratios
   - **Contributing factors**: Detailed breakdown of enhancement calculations
   - **Schema support**: Nested validation for all QV components

### 4. Execution Success (`CircuitPerformance`)
   - **Single job analysis**: Success rate, error rate, fidelity for individual executions
   - **Multiple job analysis**: Aggregate statistics across multiple runs
   - **Custom criteria**: User-defined success conditions
   - **Schema support**: Automatic validation of rate consistency (e.g., error_rate = 1 - success_rate)

## Schema Benefits

The new schema-based validation system provides:

1. **Type Safety**: Automatic validation of data types and constraints
2. **Business Rules**: Cross-field validation (e.g., successful_shots ≤ total_shots)
3. **Range Validation**: Ensures values are within expected bounds (e.g., rates between 0.0-1.0)
4. **IDE Support**: Full autocomplete and type hints for better developer experience
5. **API Documentation**: Automatic JSON schema generation for documentation
6. **Error Prevention**: Catch data inconsistencies early in the analysis pipeline

### JSON Schema Generation

Generate API documentation automatically:

```python
from qward.metrics.schemas import ComplexityMetricsSchema, CircuitPerformanceJobSchema
import json

# Generate JSON schemas for documentation
complexity_json_schema = ComplexityMetricsSchema.model_json_schema()
circuit_performance_json_schema = CircuitPerformanceJobSchema.model_json_schema()

print("Complexity Metrics JSON Schema:")
print(json.dumps(complexity_json_schema, indent=2))

# Use for API documentation, frontend validation, etc.
```

## Best Practices

1. **Choose the Right Approach**:
   - Use schema-based methods for new code requiring type safety
   - Use traditional dictionary methods for backward compatibility
   - Combine both approaches as needed

2. **Error Handling**:
   ```python
   try:
       structured_metrics = calculator.get_structured_metrics()
       # Use validated data with confidence
   except ImportError:
       # Fallback to traditional approach if Pydantic not available
       traditional_metrics = calculator.get_metrics()
   except Exception as e:
       print(f"Validation error: {e}")
   ```

3. **Performance**: Schema validation adds minimal overhead but provides significant benefits

## Using Jupyter Notebooks

The easiest way to work with Qward is often using Jupyter notebooks. If you use the Docker setup (if available and configured with `./start.sh`), or a local Python environment with Jupyter installed, you can explore the examples in `qward/examples/` such as:

- `run_on_aer.ipynb` - Interactive circuit analysis
- `schema_demo.py` - Schema validation demonstration
- `circuit_performance_demo.py` - Circuit performance analysis examples

## Next Steps

- Explore the [Beginner's Guide](beginners_guide.md) for detailed tutorials
- Check the [Architecture Documentation](architecture.md) to understand the design patterns
- Review the [API Documentation](apidocs/index.rst) for complete reference
- Try the examples in `qward/examples/` to see real-world usage
