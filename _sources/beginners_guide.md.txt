# Qward Beginner's Guide

This guide provides a comprehensive introduction to Qward, a Python library for analyzing quantum circuits and their execution results.

## What is Qward?

Qward is a library built on top of Qiskit that helps quantum developers understand how their quantum algorithms perform on both simulators and real quantum hardware. It provides tools to:

1.  Define and execute quantum circuits using Qiskit.
2.  Collect execution data like counts from simulators or hardware jobs.
3.  Analyze circuits and results using a variety of built-in **metric strategies**.
4.  Assess circuit properties, complexity, and estimate potential performance.

## Key Concepts

### Scanner
The `qward.Scanner` class is the central component for orchestrating circuit analysis. You provide it with a `QuantumCircuit` and optionally an execution `Job` or `Result`. You then add various **metric strategy** objects to the `Scanner` to perform different types of analysis.

### Metric Strategies
Metric strategies are classes that perform specific calculations or data extraction based on a circuit, a job, or a result. Qward provides several built-in metric strategies:
-   `QiskitMetrics`: Extracts basic properties directly available from a `QuantumCircuit` object (e.g., depth, width, gate counts).
-   `ComplexityMetrics`: Calculates a wide range of complexity indicators, including those from "Character Complexity: A Novel Measure for Quantum Circuit Analysis" by D. Shami, and also provides Quantum Volume estimation.
-   `SuccessRate`: Calculates success rates, error rates, and fidelity based on execution counts from a job, given a user-defined success criterion.

You can also create your own custom metric strategies by subclassing `qward.metrics.base_metric.MetricCalculator`.

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
from qward.metrics import QiskitMetrics, ComplexityMetrics, SuccessRate # QWARD strategies
from qward.examples.utils import create_example_circuit, get_display # Example helper

display = get_display()

# 1. Create a quantum circuit (2-qubit GHZ state from examples.utils)
# This circuit prepares a |00> + |11> state and measures both qubits.
# For a "coin flip" on the first qubit, we can define success as measuring '0' or '1'.
circuit = create_example_circuit() # This is a 2-qubit circuit

print("Quantum Circuit (2-qubit GHZ from examples):")
display(circuit.draw(output='mpl'))

# 2. Simulate the circuit to get results (optional for some strategies)
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

# 4. Add Strategies
# QiskitMetrics and ComplexityMetrics only need the circuit
scanner.add_strategy(QiskitMetrics(circuit=circuit))
scanner.add_strategy(ComplexityMetrics(circuit=circuit))

# SuccessRate needs the circuit and the job (or jobs) to get counts
# Let's define success for the first qubit being '0' (e.g., "tails" if '00' or '01')
# The example circuit measures two qubits. Bitstrings are read right-to-left (q1q0).
# So, '00' means qubit 0 is '0', qubit 1 is '0'.
# '10' means qubit 0 is '0', qubit 1 is '1'.
def coin_flip_success_q0_is_0(bitstring):
    # bitstring is like '00', '01', '10', '11'
    # We are interested in the first qubit (q0) state.
    return bitstring.endswith('0') # True if q0 is '0'

scanner.add_strategy(SuccessRate(circuit=circuit, job=job, success_criteria=coin_flip_success_q0_is_0))
# For multiple jobs, you can pass a list of jobs or use success_rate_strategy.add_job()

# 5. Calculate all added strategies
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

if "SuccessRate.aggregate" in all_metrics_results:
    success_df = all_metrics_results["SuccessRate.aggregate"]
    print("\nCoin Flip (q0 is '0') Success Rate:")
    print(f"  Mean success rate: {success_df['mean_success_rate'].iloc[0]:.2%}")
    print(f"  Total shots: {success_df['total_trials'].iloc[0]}")
```

### Alternative: Using Constructor with Strategies

You can also provide strategies directly in the Scanner constructor:

```python
# Using strategy classes (will be instantiated automatically)
scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics])

# Using strategy instances
qm = QiskitMetrics(circuit)
cm = ComplexityMetrics(circuit)
scanner = Scanner(circuit=circuit, strategies=[qm, cm])

# Calculate metrics
all_metrics_results = scanner.calculate_metrics()
```

This example shows how to:
1.  Create a quantum circuit.
2.  Simulate it using Qiskit Aer and obtain results (counts).
3.  Use `qward.Scanner` to analyze the circuit and its results.
4.  Add various strategy types (`QiskitMetrics`, `ComplexityMetrics`, `SuccessRate`).
5.  Calculate and interpret the metrics. For `ComplexityMetrics`, this includes gate counts, depth, and Quantum Volume estimates. For `SuccessRate`, it includes the mean success based on your criteria.

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

Instead of a specific named enigma, let's focus on how you would analyze any custom or more complex circuit. You would follow a similar pattern: create your circuit, simulate if needed for `SuccessRate`, then use the `Scanner` with appropriate strategies.

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner, Result
from qward.metrics import QiskitMetrics, ComplexityMetrics, SuccessRate
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

# 2. Simulate (optional, needed for SuccessRate)
simulator = AerSimulator()
job = simulator.run(circuit, shots=1024)
qiskit_job_result = job.result()
counts = qiskit_job_result.get_counts()
qward_result_obj = Result(job=job, counts=counts)

# 3. Create Scanner and add Strategies
scanner = Scanner(circuit=circuit, result=qward_result_obj, job=job)
scanner.add_strategy(QiskitMetrics(circuit))
scanner.add_strategy(ComplexityMetrics(circuit))

# Example SuccessRate: success if all qubits are '0' (state '000')
def all_zeros(bitstring):
    return bitstring == '000'
scanner.add_strategy(SuccessRate(circuit=circuit, job=job, success_criteria=all_zeros))

# 4. Calculate and display metrics
all_metrics_results = scanner.calculate_metrics()
print("\n--- Metrics for 3-qubit GHZ ---")
for metric_name, df in all_metrics_results.items():
    print(f"\n{metric_name} DataFrame:")
    display(df)

# You can then access specific values from the DataFrames as shown previously.
```

This example demonstrates how to apply the QWARD workflow to a different circuit, showcasing its flexibility.

## Understanding Circuit Complexity

Qward's `ComplexityMetrics` class provides comprehensive circuit complexity analysis. When you add an instance of `ComplexityMetrics(circuit)` to your `Scanner` and call `scanner.calculate_metrics()`, the resulting DataFrame for `ComplexityMetrics` will contain various sub-categories of metrics based on the research "Character Complexity: A Novel Measure for Quantum Circuit Analysis" by D. Shami.

Key metric categories available under `ComplexityMetrics` include:

1.  **Gate-based Metrics** (e.g., `gate_based_metrics.gate_count`, `gate_based_metrics.circuit_depth`, `gate_based_metrics.t_count`, `gate_based_metrics.cnot_count`)
2.  **Entanglement Metrics** (e.g., `entanglement_metrics.entangling_gate_density`, `entanglement_metrics.entangling_width`)
3.  **Standardized Metrics** (e.g., `standardized_metrics.circuit_volume`, `standardized_metrics.gate_density`, `standardized_metrics.clifford_ratio`)
4.  **Advanced Metrics** (e.g., `advanced_metrics.parallelism_factor`, `advanced_metrics.circuit_efficiency`)
5.  **Derived Metrics** (e.g., `derived_metrics.weighted_complexity`)

To access these, you would typically retrieve the `ComplexityMetrics` DataFrame from the `Scanner`'s results:

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

## Quantum Volume Estimation

Quantum Volume (QV) is an important benchmark for a quantum computer's capabilities. Qward's `ComplexityMetrics` also includes an estimation of Quantum Volume for a given circuit. This is not a formal QV benchmark execution but an estimation based on the circuit's structure.

The `ComplexityMetrics` output (under the `quantum_volume` prefix) provides:

1.  **Standard Quantum Volume** (e.g., `quantum_volume.standard_quantum_volume`): Calculated as 2^n where n is the effective depth (min(depth, num_qubits)).
2.  **Enhanced Quantum Volume** (e.g., `quantum_volume.enhanced_quantum_volume`): An adjusted QV estimate that considers factors like square ratio, density, and multi-qubit gate ratio.
3.  **Contributing Factors** (e.g., `quantum_volume.factors.square_ratio`): Details on the factors used in the enhanced QV calculation.

Example usage:

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
    
    # Example of accessing a contributing factor
    # square_ratio = complexity_df['quantum_volume.factors.square_ratio'].iloc[0]
    # print(f"Square Ratio Factor: {square_ratio}")
```

## Creating Your Own Custom Strategies

To create your own custom metric strategy, you need to inherit from `qward.metrics.base_metric.MetricCalculator` and implement its abstract methods.

```python
from qiskit import QuantumCircuit
from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsType, MetricsId # Enums for type and ID

class MyCustomStrategy(MetricCalculator):
    def __init__(self, circuit: QuantumCircuit, an_extra_parameter: int = 0):
        super().__init__(circuit) # Call base class constructor
        self.an_extra_parameter = an_extra_parameter
        # _metric_type and _id are set in the base class by calling _get_metric_type and _get_metric_id

    def _get_metric_type(self) -> MetricsType:
        """Return PRE_RUNTIME if it only needs the circuit, or POST_RUNTIME if it needs job results."""
        return MetricsType.PRE_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        """Return a unique identifier for this strategy. 
           You might need to add a new value to the MetricsId enum in types.py or manage custom IDs.
           For this example, let's assume we are re-using one for simplicity, but this is not ideal.
        """
        # For a real custom strategy, you'd likely define a new MetricsId or handle it
        # return MetricsId.COMPLEXITY # Placeholder - AVOID REUSING IDs like this in practice
        # A better approach for truly custom IDs might involve a string or a new Enum value if you modify types.py
        # For now, let's imagine a hypothetical CUSTOM_ID if we had added it to the Enum.
        # Since we can't modify the Enum here, we just return a string for the example, 
        # but the base class expects a MetricsId enum. 
        # This part of the example highlights a design consideration for custom strategy IDs.
        # To make this runnable with current enums, we would pick an existing one, e.g., MetricsId.QISKIT
        return MetricsId.QISKIT # Using an existing ID for example purposes ONLY.

    def is_ready(self) -> bool:
        """Return True if the strategy can be calculated (e.g., circuit is present)."""
        return self.circuit is not None

    def get_metrics(self) -> dict:
        """Perform the custom metric calculation and return results as a dictionary."""
        # Example: Calculate something based on the circuit and the extra parameter
        custom_value = self.circuit.depth() * self.an_extra_parameter
        return {"my_custom_metric_value": custom_value, "parameter_used": self.an_extra_parameter}

# How to use your custom strategy:
# my_circuit = QuantumCircuit(2)
# my_circuit.h(0)
# my_circuit.cx(0,1)

# custom_strategy_instance = MyCustomStrategy(circuit=my_circuit, an_extra_parameter=5)

# scanner = Scanner(circuit=my_circuit)
# scanner.add_strategy(custom_strategy_instance)

# results = scanner.calculate_metrics()
# print(results['MyCustomStrategy'])
```

## Next Steps

-   Explore the example scripts and notebooks in the `qward/examples/` directory (e.g., `qward/examples/aer.py`, `qward/examples/run_on_aer.ipynb`).
-   Check the [Technical Documentation](technical_docs.md) (`docs/technical_docs.md`) for more in-depth information about components (once updated).
-   Read the [API Documentation](apidocs/index.rst) (`docs/apidocs/index.rst`) for a complete reference to all classes and methods (once generated/updated).

(Note: Example notebook paths like `flip_coin/notebook_demo.ipynb` need to be updated or removed if they don't exist or are not aligned with the new structure. The `qward/examples/run_on_aer.ipynb` is a relevant current example.)
