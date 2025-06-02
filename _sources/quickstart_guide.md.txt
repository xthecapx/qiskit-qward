# QWARD Quickstart Guide

Welcome to QWARD! This guide will get you up and running with quantum circuit analysis in just a few minutes.

## What is QWARD?

QWARD is a Python library for analyzing quantum circuits and their execution results. It provides:

- **Circuit Analysis**: Extract metrics from quantum circuits (depth, complexity, gate counts)
- **Performance Analysis**: Analyze execution results (success rates, fidelity, error rates)
- **Schema Validation**: Type-safe, validated metrics with IDE support
- **Visualization**: Beautiful plots and dashboards for your analysis

## Installation

```bash
# Clone and install
git clone https://github.com/your-org/qiskit-qward.git
cd qiskit-qward
pip install -e .
```

## Quick Example

Let's analyze a simple Bell state circuit:

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics

# 1. Create a Bell state circuit
circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

# 2. Run on simulator (optional, needed for CircuitPerformanceMetrics)
simulator = AerSimulator()
job = simulator.run(circuit, shots=1000)

# 3. Analyze with Scanner (returns DataFrames)
scanner = Scanner(circuit=circuit, job=job)
scanner.add_strategy(QiskitMetrics(circuit))
scanner.add_strategy(ComplexityMetrics(circuit))
scanner.add_strategy(CircuitPerformanceMetrics(circuit=circuit, job=job))

results = scanner.calculate_metrics()
print("Available metrics:", list(results.keys()))

# 4. Use schema-based API for type-safe access
qiskit_metrics = QiskitMetrics(circuit)
metrics = qiskit_metrics.get_metrics()  # Returns QiskitMetricsSchema

print(f"Circuit depth: {metrics.basic_metrics.depth}")
print(f"Gate count: {metrics.basic_metrics.size}")
print(f"Number of qubits: {metrics.basic_metrics.num_qubits}")
```

## Core Concepts

### Unified API
All metric classes use the same simple interface:

```python
# All metric classes work the same way
calculator = QiskitMetrics(circuit)        # or ComplexityMetrics(circuit)
metrics = calculator.get_metrics()         # Returns validated schema object
depth = metrics.basic_metrics.depth       # Type-safe access with IDE support
```

### Schema Validation
QWARD provides automatic data validation:

```python
# Type safety and validation built-in
complexity_metrics = ComplexityMetrics(circuit)
schema = complexity_metrics.get_metrics()

# IDE autocomplete and type checking
print(f"Gate count: {schema.gate_based_metrics.gate_count}")
print(f"T-gate count: {schema.gate_based_metrics.t_count}")
print(f"Circuit efficiency: {schema.advanced_metrics.circuit_efficiency:.3f}")

# Validation catches errors automatically
# (e.g., efficiency values must be between 0.0-1.0)
```

### Scanner Integration
Scanner automatically handles schema-to-DataFrame conversion:

```python
# Scanner works seamlessly with schema objects
scanner = Scanner(circuit=circuit)
scanner.add_strategy(QiskitMetrics(circuit))
scanner.add_strategy(ComplexityMetrics(circuit))

# Returns DataFrames for analysis and visualization
dataframes = scanner.calculate_metrics()
```

## Available Metrics

### QiskitMetrics
Extracts basic circuit properties:

```python
qiskit_metrics = QiskitMetrics(circuit)
metrics = qiskit_metrics.get_metrics()

# Basic circuit properties
print(f"Depth: {metrics.basic_metrics.depth}")
print(f"Width: {metrics.basic_metrics.width}")
print(f"Size: {metrics.basic_metrics.size}")
print(f"Qubits: {metrics.basic_metrics.num_qubits}")

# Instruction analysis
print(f"Connected components: {metrics.instruction_metrics.num_connected_components}")
print(f"Non-local gates: {metrics.instruction_metrics.num_nonlocal_gates}")
```

### ComplexityMetrics
Analyzes circuit complexity:

```python
complexity_metrics = ComplexityMetrics(circuit)
metrics = complexity_metrics.get_metrics()

# Gate-based complexity
print(f"Gate count: {metrics.gate_based_metrics.gate_count}")
print(f"T-gate count: {metrics.gate_based_metrics.t_count}")
print(f"CNOT count: {metrics.gate_based_metrics.cnot_count}")

# Advanced complexity indicators
print(f"Circuit volume: {metrics.standardized_metrics.circuit_volume}")
print(f"Parallelism factor: {metrics.advanced_metrics.parallelism_factor:.3f}")
print(f"Weighted complexity: {metrics.derived_metrics.weighted_complexity}")
```

### CircuitPerformanceMetrics
Analyzes execution performance:

```python
# Requires job execution results
circuit_performance = CircuitPerformanceMetrics(circuit=circuit, job=job)
metrics = circuit_performance.get_metrics()

# Success analysis
print(f"Success rate: {metrics.success_metrics.success_rate:.3f}")
print(f"Error rate: {metrics.success_metrics.error_rate:.3f}")
print(f"Successful shots: {metrics.success_metrics.successful_shots}")

# Fidelity analysis
print(f"Fidelity: {metrics.fidelity_metrics.fidelity:.3f}")

# Statistical analysis
print(f"Entropy: {metrics.statistical_metrics.entropy:.3f}")
print(f"Uniformity: {metrics.statistical_metrics.uniformity:.3f}")
```

## Custom Success Criteria

Define custom success criteria for CircuitPerformanceMetrics:

```python
# Custom success criteria for Bell state
def bell_state_success(result: str) -> bool:
    clean_result = result.replace(" ", "")
    return clean_result in ["00", "11"]  # |00⟩ or |11⟩ states

# Use with CircuitPerformanceMetrics
circuit_performance = CircuitPerformanceMetrics(
    circuit=circuit, 
    job=job, 
    success_criteria=bell_state_success
)

metrics = circuit_performance.get_metrics()
print(f"Bell state success rate: {metrics.success_metrics.success_rate:.3f}")
```

## Visualization

Create beautiful visualizations of your analysis:

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

## Error Handling

Handle validation and execution errors gracefully:

```python
try:
    # Calculate metrics with validation
    metrics = calculator.get_metrics()
    
    # Use validated data with confidence
    depth = metrics.basic_metrics.depth
    print(f"Circuit depth: {depth}")
    
except ImportError as e:
    print(f"Missing dependencies: {e}")
except Exception as e:
    print(f"Calculation error: {e}")
```

## Next Steps

- **Learn More**: Check out the [Beginner's Guide](beginners_guide.md) for detailed examples
- **Advanced Usage**: See [Technical Documentation](technical_docs.md) for in-depth information
- **API Reference**: Browse the [API Documentation](apidocs/index.rst) for complete method references
- **Examples**: Explore working examples in `qward/examples/`

## Key Benefits

✅ **Simple API**: All metric classes use `get_metrics()` for consistent interface  
✅ **Type Safety**: Schema validation with IDE autocomplete and error prevention  
✅ **Flexible Analysis**: Scanner for DataFrames, schemas for type-safe access  
✅ **Rich Visualization**: Automatic plot generation with customizable styling  
✅ **Extensible**: Easy to add custom metrics and success criteria  

Start analyzing your quantum circuits today with QWARD!
