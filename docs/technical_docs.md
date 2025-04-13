# Technical Documentation

This document provides detailed technical information about the Qward framework's architecture, components, and usage patterns.

## Framework Architecture

Qward is organized into several key components:

### 1. Validators

The validator system is the core of Qward, built around extending Qiskit's QuantumCircuit:

- **ScanningQuantumCircuit**: The foundation class that all validators inherit from. It provides core functionality for circuit creation, validation, execution, and analysis.
  
- **Example Validators**: Specific implementations for different quantum algorithms:
  - **FlipCoinValidator**: Implements a simple quantum coin flip using a Hadamard gate
  - **QuantumEnigmaValidator**: Implements the quantum solution to the "two doors enigma" problem

Validators handle:
- Circuit construction
- Simulation configuration
- IBM Quantum execution
- Results collection
- Analysis integration
- Complexity metrics calculation
- Quantum volume estimation

### 2. Analysis Framework

The analysis framework processes execution results:

- **Analysis**: Base class for all analyzers with core functionality for processing results
- **SuccessRate**: Calculates and analyzes success rates for circuit executions based on custom criteria
- **Complexity Metrics**: Provides comprehensive circuit complexity analysis
- **Quantum Volume**: Estimates quantum volume metrics for circuits

### 3. Implementation Details

#### ScanningQuantumCircuit

The `ScanningQuantumCircuit` extends Qiskit's `QuantumCircuit` and adds:

```python
def __init__(
    self, num_qubits: int = 1, num_clbits: int = 1, use_barriers: bool = True, name: str = None
):
    # Create quantum and classical registers
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(num_clbits, "c")
    
    # Initialize the quantum circuit
    super().__init__(qr, cr, name=name)
    
    # Store additional attributes
    self.use_barriers = use_barriers
    self.analyzers: List[Analysis] = []
```

Key methods include:
- `add_analyzer(analyzer)`: Adds an analysis module to the validator
- `run_simulation(show_histogram, num_jobs, shots_per_job)`: Runs local simulations
- `run_on_ibm()`: Executes the circuit on IBM Quantum hardware
- `plot_analysis()`: Visualizes analysis results
- `calculate_complexity_metrics()`: Calculates detailed circuit complexity metrics
- `estimate_quantum_volume()`: Estimates the quantum volume of the circuit

#### Success Rate Analyzer

The `SuccessRate` analyzer evaluates circuit execution outcomes:

```python
def __init__(self, results_df: pd.DataFrame = None):
    super().__init__(results_df)
    self.success_criteria = self._default_success_criteria()

def set_success_criteria(self, criteria: Callable[[str], bool]):
    self.success_criteria = criteria
```

Key features:
- Custom success criteria definition
- Statistical analysis (mean, std dev, min/max)
- Visualization of success rate distributions

## Circuit Complexity Metrics

The `ScanningQuantumCircuit` now includes comprehensive circuit complexity analysis capabilities through the `calculate_complexity_metrics()` method. This feature implements metrics described in "Character Complexity: A Novel Measure for Quantum Circuit Analysis" by Daksh Shami.

### Gate-based Metrics

- **Gate Count**: Total number of gates in the circuit
- **Circuit Depth**: Longest path through the circuit
- **T-count**: Number of T gates (costly in fault-tolerant implementations)
- **CNOT Count**: Number of CNOT gates (important for entanglement)
- **Two-qubit Gate Count**: Total count of two-qubit operations
- **Multi-qubit Gate Ratio**: Proportion of multi-qubit gates to total gates

### Entanglement Metrics

- **Entangling Gate Density**: Ratio of entangling gates to total gates
- **Entangling Width**: Estimate of maximum number of qubits that could be entangled

### Standardized Metrics

- **Circuit Volume**: Product of depth and width (depth × width)
- **Gate Density**: Gates per qubit-time-step
- **Clifford Ratio**: Proportion of Clifford gates
- **Non-Clifford Ratio**: Proportion of non-Clifford gates

### Advanced Metrics

- **Parallelism Factor**: Average gates executable in parallel
- **Parallelism Efficiency**: Actual parallelism relative to maximum possible
- **Circuit Efficiency**: How efficiently the circuit uses available qubits
- **Quantum Resource Utilization**: Efficiency of both space (qubits) and time (depth)

### Derived Metrics

- **Square Ratio**: How close the circuit is to a square circuit (depth ≈ width)
- **Weighted Complexity**: Gates weighted by their implementation complexity
- **Normalized Weighted Complexity**: Weighted complexity per qubit

Example usage:
```python
from qward.examples.flip_coin.scanner import ScanningQuantumFlipCoin

# Create scanner
scanner = ScanningQuantumFlipCoin()

# Calculate complexity metrics
metrics = scanner.calculate_complexity_metrics()

# Access specific metrics
gate_count = metrics["gate_based_metrics"]["gate_count"]
entangling_density = metrics["entanglement_metrics"]["entangling_gate_density"]
circuit_volume = metrics["standardized_metrics"]["circuit_volume"]
```

## Quantum Volume Estimation

The `estimate_quantum_volume()` method provides an analysis of a circuit's quantum volume, an important metric for understanding the computational capacity of a quantum circuit.

### Standard Quantum Volume

The standard quantum volume is calculated as 2^n where n is the effective depth (minimum of depth and number of qubits).

### Enhanced Quantum Volume

Qward extends the standard quantum volume with an enhanced estimate that factors in:

1. **Square Ratio**: How close the circuit is to a square circuit (depth ≈ width)
2. **Circuit Density**: How many operations per qubit-timestep
3. **Multi-qubit Operation Ratio**: Proportion of entangling operations
4. **Connectivity Factor**: Presence of entangling operations

The method returns a comprehensive dictionary with:
- Standard quantum volume
- Enhanced quantum volume
- Effective depth
- Contributing factors
- Circuit metrics

Example usage:
```python
from qward.examples.flip_coin.scanner import ScanningQuantumFlipCoin

# Create scanner
scanner = ScanningQuantumFlipCoin()

# Estimate quantum volume
qv = scanner.estimate_quantum_volume()

# Access quantum volume metrics
standard_qv = qv["standard_quantum_volume"]
enhanced_qv = qv["enhanced_quantum_volume"]
square_ratio = qv["factors"]["square_ratio"]
```

## Example Use Cases

### 1. Quantum Coin Flip

The `FlipCoinValidator` implements a simple quantum coin toss:

```python
def _build_circuit(self):
    # Apply Hadamard gate to create superposition
    self.h(0)

    if self.use_barriers:
        self.barrier()

    # Measure the qubit
    self.measure(0, 0)
```

Success criteria:
```python
# Define success criteria: tails (1) is considered success
def success_criteria(state):
    return state == "1"
```

### 2. Two Doors Enigma

The `QuantumEnigmaValidator` implements a more complex problem involving:
- Multiple qubits (3 qubits)
- Superposition with Hadamard gates
- CNOT entanglement
- Complex measurement interpretation

Success criteria focus on detecting when both guardians point to the same door.

## Execution Pipeline

1. **Circuit Creation**: Validator constructs the quantum circuit
2. **Simulation Execution**: Circuit is submitted to a simulator or IBM Quantum hardware
3. **Results Collection**: Outcomes are gathered with execution metrics
4. **Analysis Processing**: Results are passed to analyzers
5. **Complexity Analysis**: Circuit structure is analyzed for complexity metrics
6. **Quantum Volume Estimation**: Circuit's quantum volume is calculated
7. **Visualization**: Analysis results are plotted and interpreted

## Technical Guidelines

When extending Qward with custom validators:

1. Inherit from ScanningQuantumCircuit
2. Initialize with appropriate number of qubits and classical bits
3. Define custom success criteria for your algorithm
4. Add appropriate analyzers
5. Implement your circuit-building logic
6. Use the built-in methods for simulation, execution, and analysis
7. Leverage complexity metrics and quantum volume estimation for deeper insights

## API Flow and Usage Patterns

### Pattern 1: Using Existing Validators

```python
# Import a validator
from qward.examples.flip_coin.scanner import ScanningQuantumFlipCoin

# Create validator instance
scanner = ScanningQuantumFlipCoin(use_barriers=True)

# Run simulation
results = scanner.run_simulation(show_histogram=True, num_jobs=100, shots_per_job=1024)

# Access analysis results
analysis = results["analysis"]["analyzer_0"]
print(f"Mean success rate: {analysis['mean_success_rate']:.2%}")

# Access complexity metrics
complexity = results["complexity_metrics"]
print(f"Circuit depth: {complexity['gate_based_metrics']['circuit_depth']}")

# Estimate quantum volume
qv = scanner.estimate_quantum_volume()
print(f"Quantum Volume: {qv['standard_quantum_volume']}")
```

### Pattern 2: Creating Custom Validators

```python
from qward.scanning_quantum_circuit import ScanningQuantumCircuit
from qward.analysis.success_rate import SuccessRate

class MyCustomValidator(ScanningQuantumCircuit):
    def __init__(self, use_barriers=True):
        super().__init__(num_qubits=2, num_clbits=2, use_barriers=use_barriers, name="my_circuit")
        
        # Define success criteria
        def success_criteria(state):
            return state == "00"
        
        # Add success rate analyzer
        success_analyzer = SuccessRate()
        success_analyzer.set_success_criteria(success_criteria)
        self.add_analyzer(success_analyzer)
        
        # Build the circuit
        self._build_circuit()
    
    def _build_circuit(self):
        # Implement your quantum circuit here
        self.h(0)
        self.cx(0, 1)
        
        if self.use_barriers:
            self.barrier()
            
        self.measure([0, 1], [0, 1])
```

## Conclusion

Qward provides a comprehensive framework for analyzing and validating quantum circuits. By leveraging the validator system, analysis framework, and advanced metrics, you can gain deep insights into your quantum algorithms' performance and resource requirements.
