# Quickstart Guide

Qward is a Python library for analyzing quantum circuits and their execution quality on quantum processing units (QPUs) or simulators. This guide will help you quickly get started with the unified schema-based API that provides type safety and data validation by default.

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

Qward revolves around the `Scanner` class, which uses various metric calculator objects to analyze Qiskit `QuantumCircuit` objects and their execution results. The library provides a unified schema-based API that returns validated objects with full type safety and IDE support.

### Core Workflow

1.  **Create/Load a `QuantumCircuit`**: Use Qiskit to define your circuit.
2.  **(Optional) Execute the Circuit**: Run your circuit on a simulator or quantum hardware to get a Qiskit `Job` and its `Result` (containing counts).
3.  **Instantiate `qward.Scanner`**: Provide the circuit, and optionally the Qiskit `Job`.
4.  **Add Metric Calculators**: Instantiate and add desired metric calculator classes from `qward.metrics` (e.g., `QiskitMetrics`, `ComplexityMetrics`, `CircuitPerformanceMetrics`) to the scanner.
5.  **Calculate Metrics**: Call `scanner.calculate_metrics()`.
6.  **Interpret Results**: The result is a dictionary of pandas DataFrames, one for each metric type.
7.  **Use Schema Objects**: Access validated metrics directly through schema objects for enhanced type safety.

### Example: Analyzing a Simple Circuit

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner # QWARD classes
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics # QWARD calculators
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

# 3. Instantiate QWARD Scanner
scanner = Scanner(circuit=qc, job=job)

# 4. Add Metric Calculators
scanner.add_strategy(QiskitMetrics(circuit=qc))
scanner.add_strategy(ComplexityMetrics(circuit=qc))

# For CircuitPerformance, define what a "successful" measurement is
def success_if_00(bitstring):
    # Handle measurement results with spaces
    clean_result = bitstring.replace(" ", "")
    return clean_result == "00"

# CircuitPerformance needs a job to get counts from
scanner.add_strategy(CircuitPerformanceMetrics(circuit=qc, job=job, success_criteria=success_if_00))

# 5. Calculate Metrics (Scanner returns DataFrames)
all_metric_data = scanner.calculate_metrics()

# 6. Interpret Results
print("\n--- All Calculated Metrics (DataFrames from Scanner) ---")
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

# Example: Accessing specific data from CircuitPerformanceMetrics output
if "CircuitPerformance.aggregate" in all_metric_data:
    success_df = all_metric_data["CircuitPerformance.aggregate"]
    print("\nSuccess Rate Data (for '00'):")
    print(f"  Mean Success Rate: {success_df['mean_success_rate'].iloc[0]:.2%}")
    print(f"  Total Shots: {success_df['total_trials'].iloc[0]}")
```

### Schema-Based API (Type-Safe Access)

Qward provides validated schema objects for direct access to metrics with full type safety:

```python
# 7. Use Schema-Based API (Type-Safe Access)
print("\n--- Schema-Based API (Type-Safe Access) ---")

# QiskitMetrics with schema validation
qiskit_metrics = QiskitMetrics(qc)
qiskit_schema = qiskit_metrics.get_metrics()  # Returns QiskitMetricsSchema

print("✅ QiskitMetrics Schema:")
print(f"  Circuit Depth: {qiskit_schema.basic_metrics.depth}")
print(f"  Number of Qubits: {qiskit_schema.basic_metrics.num_qubits}")
print(f"  Gate Count: {qiskit_schema.basic_metrics.size}")
print(f"  Multi-qubit Gates: {qiskit_schema.instruction_metrics.multi_qubit_gate_count}")
print(f"  Width: {qiskit_schema.basic_metrics.width}")

# ComplexityMetrics with schema validation
complexity_metrics = ComplexityMetrics(qc)
complexity_schema = complexity_metrics.get_metrics()  # Returns ComplexityMetricsSchema

print("\n✅ ComplexityMetrics Schema:")
print(f"  Enhanced Quantum Volume: {complexity_schema.quantum_volume.enhanced_quantum_volume:.2f}")
print(f"  Gate Density: {complexity_schema.standardized_metrics.gate_density:.3f}")
print(f"  Parallelism Efficiency: {complexity_schema.advanced_metrics.parallelism_efficiency:.3f}")
print(f"  T-gate Count: {complexity_schema.gate_based_metrics.t_count}")
print(f"  Multi-qubit Ratio: {complexity_schema.gate_based_metrics.multi_qubit_ratio:.3f}")
print(f"  QV Enhancement Factor: {complexity_schema.quantum_volume.factors.enhancement_factor:.2f}")

# CircuitPerformance with schema validation
circuit_performance = CircuitPerformanceMetrics(circuit=qc, job=job, success_criteria=success_if_00)
performance_schema = circuit_performance.get_metrics()  # Returns CircuitPerformanceSchema

print("\n✅ CircuitPerformanceMetrics Schema:")
print(f"  Success Rate: {performance_schema.success_metrics.success_rate:.3f}")
print(f"  Error Rate: {performance_schema.success_metrics.error_rate:.3f}")  # Automatically validated
print(f"  Fidelity: {performance_schema.fidelity_metrics.fidelity:.3f}")
print(f"  Total Shots: {performance_schema.success_metrics.total_shots}")
print(f"  Successful Shots: {performance_schema.success_metrics.successful_shots}")

# Demonstrate validation in action
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

### Visualization: Creating Beautiful Plots

Qward includes a comprehensive visualization system that automatically creates publication-quality plots of your metrics:

```python
# 8. Create Visualizations
from qward.visualization import Visualizer

print("\n--- Creating Visualizations ---")

# Create unified visualizer (recommended approach)
visualizer = Visualizer(scanner=scanner, output_dir="quickstart_plots")

# Option 1: Create comprehensive dashboards for all metrics
print("Creating dashboards...")
dashboards = visualizer.create_dashboard(save=True, show=False)
print(f"✅ Created {len(dashboards)} dashboards")

# Option 2: Create all individual plots
print("Creating individual plots...")
all_plots = visualizer.visualize_all(save=True, show=False)
print(f"✅ Created plots for {len(all_plots)} metric types")

# Option 3: Visualize specific metrics
print("Creating QiskitMetrics plots...")
qiskit_plots = visualizer.visualize_metric("QiskitMetrics", save=True, show=False)
print(f"✅ Created {len(qiskit_plots)} QiskitMetrics plots")

# Print summary of what was created
visualizer.print_available_metrics()
```

#### Available Visualizations

1. **QiskitMetrics Visualizations**:
   - Circuit structure (depth, width, size, qubits)
   - Instruction breakdown and gate analysis
   - Scheduling and timing metrics

2. **ComplexityMetrics Visualizations**:
   - Gate-based complexity metrics
   - Radar chart for normalized complexity indicators
   - Quantum Volume analysis and factors
   - Efficiency and parallelism metrics

3. **CircuitPerformanceMetrics Visualizations**:
   - Success vs error rate comparisons
   - Fidelity analysis across jobs
   - Shot distribution (successful vs failed)
   - Aggregate statistics summary

#### Custom Visualization Configuration

```python
from qward.visualization import PlotConfig

# Create custom plot configuration
config = PlotConfig(
    figsize=(12, 8),           # Larger figures
    style="quantum",           # Quantum-themed styling
    color_palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    save_format="svg",         # High-quality vector graphics
    dpi=150,                   # Good balance of quality and speed
    alpha=0.8                  # Transparency level
)

# Use custom configuration
custom_visualizer = Visualizer(
    scanner=scanner, 
    config=config, 
    output_dir="custom_quickstart_plots"
)
custom_dashboards = custom_visualizer.create_dashboard(save=True, show=False)
print(f"✅ Created custom-styled dashboards: {list(custom_dashboards.keys())}")
```

### Alternative: Using Constructor with Calculators

You can also provide calculators directly in the Scanner constructor:

```python
# Using calculator classes (will be instantiated automatically)
scanner = Scanner(circuit=qc, strategies=[QiskitMetrics, ComplexityMetrics])

# Using calculator instances
qm = QiskitMetrics(qc)
cm = ComplexityMetrics(qc)
scanner = Scanner(circuit=qc, strategies=[qm, cm])

# Calculate metrics
all_metric_data = scanner.calculate_metrics()
```

### Creating Custom Calculators

To create your own metric calculator, inherit from `qward.metrics.base_metric.MetricCalculator` and implement the required abstract methods:

```python
from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsType, MetricsId
from qiskit import QuantumCircuit
from pydantic import BaseModel

class MyCustomMetricsSchema(BaseModel):
    custom_depth_plus_width: float
    gates_per_qubit: float
    circuit_signature: str

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

    def get_metrics(self) -> MyCustomMetricsSchema:
        return MyCustomMetricsSchema(
            custom_depth_plus_width=self.circuit.depth() + self.circuit.width(),
            gates_per_qubit=self.circuit.size() / self.circuit.num_qubits if self.circuit.num_qubits > 0 else 0,
            circuit_signature=f"{self.circuit.num_qubits}q_{self.circuit.depth()}d_{self.circuit.size()}g"
        )

# Usage:
custom_calculator = MySimpleCustomCalculator(qc)
scanner.add_strategy(custom_calculator)
results = scanner.calculate_metrics() 
print(results['MySimpleCustomCalculator'])

# Access schema object directly
custom_metrics = custom_calculator.get_metrics()
print(f"Custom complexity: {custom_metrics.custom_depth_plus_width}")
```

## Key Metrics Provided

Qward, through its built-in metric calculator classes, offers insights into:

### 1. Circuit Structure (`QiskitMetrics`)
   - **Basic metrics**: Depth, width, number of qubits/clbits, operations count
   - **Instruction metrics**: Multi-qubit gate count, connectivity analysis
   - **Scheduling metrics**: Timing and resource information
   - **Schema support**: Type-safe access with full IDE autocomplete

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

### 4. Execution Success (`CircuitPerformanceMetrics`)
   - **Single job analysis**: Success rate, error rate, fidelity for individual executions
   - **Multiple job analysis**: Aggregate statistics across multiple runs
   - **Custom criteria**: User-defined success conditions
   - **Schema support**: Automatic validation of rate consistency (e.g., error_rate = 1 - success_rate)

## Schema Benefits

The schema-based validation system provides:

1. **Type Safety**: Automatic validation of data types and constraints
2. **Business Rules**: Cross-field validation (e.g., successful_shots ≤ total_shots)
3. **Range Validation**: Ensures values are within expected bounds (e.g., rates between 0.0-1.0)
4. **IDE Support**: Full autocomplete and type hints for better developer experience
5. **API Documentation**: Automatic JSON schema generation for documentation
6. **Error Prevention**: Catch data inconsistencies early in the analysis pipeline

### JSON Schema Generation

Generate API documentation automatically:

```python
from qward.metrics.schemas import ComplexityMetricsSchema, CircuitPerformanceSchema
import json

# Generate JSON schemas for documentation
complexity_json_schema = ComplexityMetricsSchema.model_json_schema()
circuit_performance_json_schema = CircuitPerformanceSchema.model_json_schema()

print("Complexity Metrics JSON Schema:")
print(json.dumps(complexity_json_schema, indent=2))

# Use for API documentation, frontend validation, etc.
```

## Best Practices

1. **Choose the Right Approach**:
   - Use schema objects for type safety and validation (recommended)
   - Use Scanner DataFrames for analysis and visualization
   - Combine both approaches as needed

2. **Error Handling**:
   ```python
   try:
       metrics = calculator.get_metrics()
       # Use validated data with confidence
       depth = metrics.basic_metrics.depth
   except Exception as e:
       print(f"Metric calculation failed: {e}")
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
- Review the [Visualization Guide](visualization_guide.md) for comprehensive plotting capabilities
- Review the [API Documentation](apidocs/index.rst) for complete reference
- Try the examples in `qward/examples/` to see real-world usage
