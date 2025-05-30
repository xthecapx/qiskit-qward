# Qward Beginner's Guide

This guide provides a comprehensive introduction to Qward, a Python library for analyzing quantum circuits and their execution results.

## What is Qward?

Qward is a library built on top of Qiskit that helps quantum developers understand how their quantum algorithms perform on both simulators and real quantum hardware. It provides tools to:

1.  Define and execute quantum circuits using Qiskit.
2.  Collect execution data like counts from simulators or hardware jobs.
3.  Analyze circuits and results using a variety of built-in **metric calculators**.
4.  Assess circuit properties, complexity, and estimate potential performance.
5.  Validate data integrity with schema-based validation using Pydantic.

## Key Concepts

### Scanner
The `qward.Scanner` class is the central component for orchestrating circuit analysis. You provide it with a `QuantumCircuit` and optionally an execution `Job` or `Result`. You then add various **metric calculator** objects to the `Scanner` to perform different types of analysis.

### Metric Calculators
Metric calculators are classes that perform specific calculations or data extraction based on a circuit, a job, or a result. Qward provides several built-in metric calculators:
-   `QiskitMetrics`: Extracts basic properties directly available from a `QuantumCircuit` object (e.g., depth, width, gate counts).
-   `ComplexityMetrics`: Calculates a wide range of complexity indicators, including those from "Character Complexity: A Novel Measure for Quantum Circuit Analysis" by D. Shami, and also provides Quantum Volume estimation.
-   `CircuitPerformance`: Calculates success rates, error rates, and fidelity based on execution counts from a job, given a user-defined success criterion.

You can also create your own custom metric calculators by subclassing `qward.metrics.base_metric.MetricCalculator`.

### Schema-Based Validation
Qward now includes comprehensive data validation using Pydantic schemas, providing:
- **Type Safety**: Automatic validation of data types and constraints
- **Business Rules**: Cross-field validation (e.g., error_rate = 1 - success_rate)
- **Range Validation**: Ensures values are within expected bounds (e.g., success rates between 0.0-1.0)
- **IDE Support**: Full autocomplete and type hints for better developer experience
- **API Documentation**: Automatic JSON schema generation

Each metric calculator provides both traditional dictionary outputs and modern schema-based validation through structured methods.

## Getting Started

### Installation

You can set up Qward in two ways:

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
from qiskit_aer import AerSimulator # For running the circuit
from qward import Scanner, Result   # QWARD's Scanner and Result
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformance # QWARD calculators
from qward.examples.utils import create_example_circuit, get_display # Example helper

display = get_display()

# 1. Create a quantum circuit (2-qubit GHZ state from examples.utils)
# This circuit prepares a |00> + |11> state and measures both qubits.
# For a "coin flip" on the first qubit, we can define success as measuring '0' or '1'.
circuit = create_example_circuit() # This is a 2-qubit circuit

print("Quantum Circuit (2-qubit GHZ from examples):")
display(circuit.draw(output='mpl'))

# 2. Simulate the circuit to get results (optional for some calculators)
print("\nRunning quantum simulation...")
simulator = AerSimulator()
job = simulator.run(circuit, shots=1024)
qiskit_job_result = job.result()
counts = qiskit_job_result.get_counts()

# Wrap Qiskit job and counts in QWARD's Result object
qward_result_obj = Result(job=job, counts=counts) # Pass the actual job and counts

# 3. Create a Scanner instance
# We can initialize it with the circuit and the QWARD Result object
scanner = Scanner(circuit=circuit, result=qward_result_obj, job=job)

# 4. Add Metric Calculators
# QiskitMetrics and ComplexityMetrics only need the circuit
scanner.add_metric(QiskitMetrics(circuit=circuit))
scanner.add_metric(ComplexityMetrics(circuit=circuit))

# CircuitPerformance needs the circuit and the job (or jobs) to get counts
# Let's define success for the first qubit being '0' (e.g., "tails" if '00' or '01')
# The example circuit measures two qubits. Bitstrings are read right-to-left (q1q0).
# So, '00' means qubit 0 is '0', qubit 1 is '0'.
# '10' means qubit 0 is '0', qubit 1 is '1'.
def coin_flip_success_q0_is_0(bitstring):
    # bitstring is like '00', '01', '10', '11'
    # We are interested in the first qubit (q0) state.
    return bitstring.endswith('0') # True if q0 is '0'

scanner.add_metric(CircuitPerformance(circuit=circuit, job=job, success_criteria=coin_flip_success_q0_is_0))
# For multiple jobs, you can pass a list of jobs or use circuit_performance_calculator.add_job()

# 5. Calculate all added calculators
print("\nCalculating metrics...")
all_metrics_results = scanner.calculate_metrics()

# 6. Display results
print("\n--- Metric Results ---")
for metric_name, df in all_metrics_results.items():
    print(f"\n{metric_name} DataFrame:")
    display(df)

# Access specific data from ComplexityMetrics, for example:
if "ComplexityMetrics" in all_metrics_results:
    complexity_df = all_metrics_results["ComplexityMetrics"]
    print("\nSelected Complexity Metrics:")
    print(f"  Gate count: {complexity_df['gate_based_metrics.gate_count'].iloc[0]}")
    print(f"  Circuit depth: {complexity_df['gate_based_metrics.circuit_depth'].iloc[0]}")
    print(f"  Circuit volume: {complexity_df['standardized_metrics.circuit_volume'].iloc[0]}")
    print(f"  Standard Quantum Volume: {complexity_df['quantum_volume.standard_quantum_volume'].iloc[0]}")
    print(f"  Enhanced Quantum Volume: {complexity_df['quantum_volume.enhanced_quantum_volume'].iloc[0]}")

if "CircuitPerformance.aggregate" in all_metrics_results:
    success_df = all_metrics_results["CircuitPerformance.aggregate"]
    print("\nCoin Flip (q0 is '0') Success Rate:")
    print(f"  Mean success rate: {success_df['mean_success_rate'].iloc[0]:.2%}")
    print(f"  Total shots: {success_df['total_trials'].iloc[0]}")
```

### Using Schema-Based Validation (New Feature)

Qward now provides structured, validated metrics through schema objects:

```python
# Traditional approach (returns dictionaries)
qiskit_metrics = QiskitMetrics(circuit)
traditional_metrics = qiskit_metrics.get_metrics()
print(f"Traditional: {traditional_metrics['basic_metrics']['depth']}")

# Schema-based approach (returns validated objects)
structured_metrics = qiskit_metrics.get_structured_metrics()
print(f"Schema-based: {structured_metrics.basic_metrics.depth}")

# Benefits of schema approach:
# 1. Type safety and IDE autocomplete
# 2. Automatic validation (e.g., depth must be >= 0)
# 3. Cross-field validation (e.g., error_rate = 1 - success_rate)
# 4. JSON schema generation for API documentation

# Access granular structured metrics
basic_schema = qiskit_metrics.get_structured_basic_metrics()
print(f"Circuit has {basic_schema.num_qubits} qubits and depth {basic_schema.depth}")

# Validation in action
try:
    from qward.metrics.schemas import CircuitPerformanceJobSchema
    
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
    print(f"Validation caught error: {type(e).__name__}")
```

### Alternative: Using Constructor with Calculators

You can also provide calculators directly in the Scanner constructor:

```python
# Using calculator classes (will be instantiated automatically)
scanner = Scanner(circuit=circuit, metrics=[QiskitMetrics, ComplexityMetrics])

# Using calculator instances
qm = QiskitMetrics(circuit)
cm = ComplexityMetrics(circuit)
scanner = Scanner(circuit=circuit, metrics=[qm, cm])

# Calculate metrics
all_metrics_results = scanner.calculate_metrics()
```

This example shows how to:
1.  Create a quantum circuit.
2.  Simulate it using Qiskit Aer and obtain results (counts).
3.  Use `qward.Scanner` to analyze the circuit and its results.
4.  Add various calculator types (`QiskitMetrics`, `ComplexityMetrics`, `CircuitPerformance`).
5.  Calculate and interpret the metrics. For `ComplexityMetrics`, this includes gate counts, depth, and Quantum Volume estimates. For `CircuitPerformance`, it includes the mean success based on your criteria.
6.  Use both traditional dictionary and modern schema-based approaches.

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
1.  **H gate on q_0**: Puts the first qubit into superposition.
2.  **CX gate (CNOT)**: Entangles q_0 and q_1. If q_0 is |1⟩, q_1 is flipped.
3.  **Measurement**: Collapses the superposition. The expected outcomes are |00⟩ and |11⟩ with roughly equal probability.

The results should show approximately 50% "00" and 50% "11". Our "coin flip" success criteria for q_0 being '0' would count outcomes like "00" and potentially "10" (if noise occurs) as success.

## Going Further: A More Complex Circuit

Instead of a specific named enigma, let's focus on how you would analyze any custom or more complex circuit. You would follow a similar pattern: create your circuit, simulate if needed for `CircuitPerformance`, then use the `Scanner` with appropriate calculators.

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner, Result
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformance
from qward.examples.utils import get_display

display = get_display()

# 1. Create a more complex quantum circuit (e.g., a 3-qubit GHZ state)
circuit = QuantumCircuit(3, 3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(0, 2)
circuit.measure([0,1,2], [0,1,2])

print("3-qubit GHZ Circuit:")
display(circuit.draw(output='mpl'))

# 2. Simulate (optional, needed for CircuitPerformance)
simulator = AerSimulator()
job = simulator.run(circuit, shots=1024)
qiskit_job_result = job.result()
counts = qiskit_job_result.get_counts()
qward_result_obj = Result(job=job, counts=counts)

# 3. Create Scanner and add Calculators
scanner = Scanner(circuit=circuit, result=qward_result_obj, job=job)
scanner.add_metric(QiskitMetrics(circuit))
scanner.add_metric(ComplexityMetrics(circuit))

# Example CircuitPerformance: success if all qubits are '0' (state '000')
def all_zeros(bitstring):
    return bitstring.replace(" ", "") == '000'
scanner.add_metric(CircuitPerformance(circuit=circuit, job=job, success_criteria=all_zeros))

# 4. Calculate and display metrics
all_metrics_results = scanner.calculate_metrics()
print("\n--- Metrics for 3-qubit GHZ ---")
for metric_name, df in all_metrics_results.items():
    print(f"\n{metric_name} DataFrame:")
    display(df)

# 5. Use structured metrics for type-safe access
complexity_metrics = ComplexityMetrics(circuit)
complexity_schema = complexity_metrics.get_structured_metrics()

print(f"\nStructured Complexity Analysis:")
print(f"  Enhanced Quantum Volume: {complexity_schema.quantum_volume.enhanced_quantum_volume}")
print(f"  Gate Density: {complexity_schema.standardized_metrics.gate_density:.3f}")
print(f"  Parallelism Efficiency: {complexity_schema.advanced_metrics.parallelism_efficiency:.3f}")
print(f"  Entangling Gate Density: {complexity_schema.entanglement_metrics.entangling_gate_density:.3f}")

# You can then access specific values from the DataFrames as shown previously.
```

This example demonstrates how to apply the QWARD workflow to a different circuit, showcasing its flexibility and the benefits of schema validation.

## Understanding Circuit Complexity

Qward's `ComplexityMetrics` class provides comprehensive circuit complexity analysis. When you add an instance of `ComplexityMetrics(circuit)` to your `Scanner` and call `scanner.calculate_metrics()`, the resulting DataFrame for `ComplexityMetrics` will contain various sub-categories of metrics based on the research "Character Complexity: A Novel Measure for Quantum Circuit Analysis" by D. Shami.

Key metric categories available under `ComplexityMetrics` include:

1.  **Gate-based Metrics** (e.g., `gate_based_metrics.gate_count`, `gate_based_metrics.circuit_depth`, `gate_based_metrics.t_count`, `gate_based_metrics.cnot_count`)
2.  **Entanglement Metrics** (e.g., `entanglement_metrics.entangling_gate_density`, `entanglement_metrics.entangling_width`)
3.  **Standardized Metrics** (e.g., `standardized_metrics.circuit_volume`, `standardized_metrics.gate_density`, `standardized_metrics.clifford_ratio`)
4.  **Advanced Metrics** (e.g., `advanced_metrics.parallelism_factor`, `advanced_metrics.circuit_efficiency`)
5.  **Derived Metrics** (e.g., `derived_metrics.weighted_complexity`)

### Traditional DataFrame Access
```python
# Assuming 'all_metrics_results' is from scanner.calculate_metrics()
if "ComplexityMetrics" in all_metrics_results:
    complexity_df = all_metrics_results["ComplexityMetrics"]
    
    # Example: Access gate count and circuit volume
    gate_count = complexity_df['gate_based_metrics.gate_count'].iloc[0]
    circuit_volume = complexity_df['standardized_metrics.circuit_volume'].iloc[0]
    
    print(f"Gate Count: {gate_count}")
    print(f"Circuit Volume: {circuit_volume}")
    # ... and so on for other metrics
```

### Schema-Based Access (Recommended)
```python
# Using structured metrics for type safety and validation
complexity_metrics = ComplexityMetrics(circuit)
schema = complexity_metrics.get_structured_metrics()

# Type-safe access with IDE autocomplete
print(f"Gate Count: {schema.gate_based_metrics.gate_count}")
print(f"Circuit Volume: {schema.standardized_metrics.circuit_volume}")
print(f"T-gate Count: {schema.gate_based_metrics.t_count}")
print(f"Parallelism Factor: {schema.advanced_metrics.parallelism_factor}")

# Access individual categories
gate_metrics = complexity_metrics.get_structured_gate_based_metrics()
print(f"Multi-qubit Ratio: {gate_metrics.multi_qubit_ratio:.3f}")

entanglement_metrics = complexity_metrics.get_structured_entanglement_metrics()
print(f"Entangling Gate Density: {entanglement_metrics.entangling_gate_density:.3f}")
```

## Quantum Volume Estimation

Quantum Volume (QV) is an important benchmark for a quantum computer's capabilities. Qward's `ComplexityMetrics` also includes an estimation of Quantum Volume for a given circuit. This is not a formal QV benchmark execution but an estimation based on the circuit's structure.

The `ComplexityMetrics` output (under the `quantum_volume` prefix) provides:

1.  **Standard Quantum Volume** (e.g., `quantum_volume.standard_quantum_volume`): Calculated as 2^n where n is the effective depth (min(depth, num_qubits)).
2.  **Enhanced Quantum Volume** (e.g., `quantum_volume.enhanced_quantum_volume`): An adjusted QV estimate that considers factors like square ratio, density, and multi-qubit gate ratio.
3.  **Contributing Factors** (e.g., `quantum_volume.factors.square_ratio`): Details on the factors used in the enhanced QV calculation.

### Traditional Access
```python
# Assuming 'all_metrics_results' is from scanner.calculate_metrics()
if "ComplexityMetrics" in all_metrics_results:
    complexity_df = all_metrics_results["ComplexityMetrics"]
    
    std_qv = complexity_df['quantum_volume.standard_quantum_volume'].iloc[0]
    enhanced_qv = complexity_df['quantum_volume.enhanced_quantum_volume'].iloc[0]
    effective_depth = complexity_df['quantum_volume.effective_depth'].iloc[0]
    
    print(f"Standard QV (Circuit Estimate): {std_qv}")
    print(f"Enhanced QV (Circuit Estimate): {enhanced_qv}")
    print(f"Effective Depth for QV Estimate: {effective_depth}")
```

### Schema-Based Access (Recommended)
```python
# Using structured metrics for validated access
complexity_metrics = ComplexityMetrics(circuit)
qv_schema = complexity_metrics.get_structured_quantum_volume()

print(f"Standard QV: {qv_schema.standard_quantum_volume}")
print(f"Enhanced QV: {qv_schema.enhanced_quantum_volume:.2f}")
print(f"Effective Depth: {qv_schema.effective_depth}")

# Access contributing factors with validation
factors = qv_schema.factors
print(f"Square Ratio: {factors.square_ratio:.2f}")
print(f"Circuit Density: {factors.circuit_density:.2f}")
```

## Circuit Performance Analysis with Schema Validation

The `CircuitPerformance` calculator now provides comprehensive validation for both single job and multiple job scenarios:

```python
from qward.metrics import CircuitPerformance

# Create circuit performance calculator
circuit_performance = CircuitPerformance(circuit=circuit, job=job)

# Traditional approach
traditional_metrics = circuit_performance.get_metrics()
print(f"Traditional success rate: {traditional_metrics['success_rate']}")

# Schema-based approach with validation
if len(circuit_performance.runtime_jobs) == 1:
    # Single job analysis
    job_schema = circuit_performance.get_structured_single_job_metrics()
    print(f"Job ID: {job_schema.job_id}")
    print(f"Success Rate: {job_schema.success_rate:.3f}")
    print(f"Error Rate: {job_schema.error_rate:.3f}")  # Automatically validated: error_rate = 1 - success_rate
    print(f"Fidelity: {job_schema.fidelity:.3f}")
    print(f"Successful Shots: {job_schema.successful_shots}/{job_schema.total_shots}")
else:
    # Multiple jobs analysis
    aggregate_schema = circuit_performance.get_structured_multiple_jobs_metrics()
    print(f"Mean Success Rate: {aggregate_schema.mean_success_rate:.3f}")
    print(f"Standard Deviation: {aggregate_schema.std_success_rate:.3f}")
    print(f"Range: {aggregate_schema.min_success_rate:.3f} - {aggregate_schema.max_success_rate:.3f}")
    print(f"Total Trials: {aggregate_schema.total_trials}")

# Schema validation catches errors automatically
try:
    from qward.metrics.schemas import CircuitPerformanceJobSchema
    
    # This will raise ValidationError
    invalid_schema = CircuitPerformanceJobSchema(
        job_id="test",
        success_rate=0.75,
        error_rate=0.30,  # Should be 0.25!
        fidelity=0.8,
        total_shots=1000,
        successful_shots=750
    )
except Exception as e:
    print(f"Schema validation caught inconsistency: {type(e).__name__}")
```

## Creating Your Own Custom Calculators

To create your own custom metric calculator, you need to inherit from `qward.metrics.base_metric.MetricCalculator` and implement its abstract methods.

```python
from qiskit import QuantumCircuit
from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsType, MetricsId # Enums for type and ID
from typing import Dict, Any

class MyCustomCalculator(MetricCalculator):
    def __init__(self, circuit: QuantumCircuit, an_extra_parameter: int = 0):
        super().__init__(circuit) # Call base class constructor
        self.an_extra_parameter = an_extra_parameter

    def _get_metric_type(self) -> MetricsType:
        """Return PRE_RUNTIME if it only needs the circuit, or POST_RUNTIME if it needs job results."""
        return MetricsType.PRE_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        """Return a unique identifier for this calculator."""
        return MetricsId.QISKIT # Using existing ID for example purposes

    def is_ready(self) -> bool:
        """Return True if the calculator can be executed (e.g., circuit is present)."""
        return self.circuit is not None

    def get_metrics(self) -> Dict[str, Any]:
        """Perform the custom metric calculation and return results as a dictionary."""
        # Example: Calculate something based on the circuit and the extra parameter
        custom_value = self.circuit.depth() * self.an_extra_parameter
        circuit_signature = f"{self.circuit.num_qubits}q_{self.circuit.depth()}d_{self.circuit.size()}g"
        
        return {
            "custom_complexity": custom_value,
            "circuit_signature": circuit_signature,
            "parameter_used": self.an_extra_parameter,
            "gates_per_qubit": self.circuit.size() / self.circuit.num_qubits if self.circuit.num_qubits > 0 else 0
        }

# How to use your custom calculator:
my_circuit = QuantumCircuit(2)
my_circuit.h(0)
my_circuit.cx(0,1)

custom_calculator = MyCustomCalculator(circuit=my_circuit, an_extra_parameter=5)

scanner = Scanner(circuit=my_circuit)
scanner.add_metric(custom_calculator)

results = scanner.calculate_metrics()
print(results['MyCustomCalculator'])

# You can also add schema validation to your custom calculator
# by creating Pydantic schemas and implementing get_structured_metrics()
```

## Schema Validation and JSON Generation

One of the powerful features of the new schema system is automatic JSON schema generation for API documentation:

```python
from qward.metrics.schemas import ComplexityMetricsSchema, CircuitPerformanceJobSchema
import json

# Generate JSON schemas for documentation
complexity_json_schema = ComplexityMetricsSchema.model_json_schema()
circuit_performance_json_schema = CircuitPerformanceJobSchema.model_json_schema()

print("Complexity Metrics JSON Schema:")
print(json.dumps(complexity_json_schema, indent=2))

print("\nCircuit Performance Job JSON Schema:")
print(json.dumps(circuit_performance_json_schema, indent=2))

# These schemas can be used for:
# 1. API documentation generation
# 2. Frontend form validation
# 3. Database schema definition
# 4. Integration with other systems
```

## Best Practices

### 1. Choose the Right Approach
- **Use schema-based methods** when you need type safety, validation, and IDE support
- **Use traditional dictionary methods** for backward compatibility or when working with existing code
- **Combine both approaches** as needed in your workflow

### 2. Validation and Error Handling
```python
# Always handle potential validation errors
try:
    structured_metrics = calculator.get_structured_metrics()
    # Use validated data with confidence
    print(f"Validated depth: {structured_metrics.basic_metrics.depth}")
except ImportError:
    # Fallback to traditional approach if Pydantic not available
    traditional_metrics = calculator.get_metrics()
    print(f"Traditional depth: {traditional_metrics['basic_metrics']['depth']}")
except Exception as e:
    print(f"Validation error: {e}")
```

### 3. Performance Considerations
- Schema validation adds minimal overhead but provides significant benefits
- Use flat dictionary conversion for DataFrame operations when needed
- Cache structured metrics when performing multiple analyses

### 4. Custom Success Criteria
```python
# Define robust success criteria that handle different measurement formats
def robust_success_criteria(result: str) -> bool:
    # Remove spaces and handle different formats
    clean_result = result.replace(" ", "")
    # Define your success condition
    return clean_result.startswith("00")  # Example: first two qubits are 0

# Use with CircuitPerformance calculator
circuit_performance = CircuitPerformance(
    circuit=circuit, 
    job=job, 
    success_criteria=robust_success_criteria
)
```

## Next Steps

-   Explore the example scripts and notebooks in the `qward/examples/` directory:
    - `qward/examples/aer.py` - Basic Aer simulator usage
    - `qward/examples/run_on_aer.ipynb` - Interactive notebook example
    - `qward/examples/schema_demo.py` - Schema validation demonstration
    - `qward/examples/circuit_performance_demo.py` - Circuit performance analysis examples
-   Check the [Technical Documentation](technical_docs.md) for more in-depth information about components
-   Read the [API Documentation](apidocs/index.rst) for a complete reference to all classes and methods
-   Review the [Architecture Documentation](architecture.md) to understand the library's design patterns and schema system
