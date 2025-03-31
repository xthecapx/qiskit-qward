# Technical Documentation

This document provides detailed technical information about the Qiskit Qward framework's architecture, components, and usage patterns.

## Framework Architecture

Qiskit Qward is organized into several key components:

### 1. Validators

The validator system is the core of Qiskit Qward, built around extending Qiskit's QuantumCircuit:

- **BaseValidator**: The foundation class that all validators inherit from. It provides core functionality for circuit creation, validation, execution, and analysis.
  
- **Example Validators**: Specific implementations for different quantum algorithms:
  - **FlipCoinValidator**: Implements a simple quantum coin flip using a Hadamard gate
  - **QuantumEnigmaValidator**: Implements the quantum solution to the "two doors enigma" problem

Validators handle:
- Circuit construction
- Simulation configuration
- IBM Quantum execution
- Results collection
- Analysis integration

### 2. Analysis Framework

The analysis framework processes execution results:

- **Analysis**: Base class for all analyzers with core functionality for processing results
- **SuccessRate**: Calculates and analyzes success rates for circuit executions based on custom criteria

### 3. Implementation Details

#### BaseValidator

The `BaseValidator` extends Qiskit's `QuantumCircuit` and adds:

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
5. **Visualization**: Analysis results are plotted and interpreted

## Technical Guidelines

When extending Qiskit Qward with custom validators:

1. Inherit from BaseValidator
2. Initialize with appropriate number of qubits and classical bits
3. Define custom success criteria for your algorithm
4. Add appropriate analyzers
5. Implement your circuit-building logic
6. Use the built-in methods for simulation, execution, and analysis

## API Flow and Usage Patterns

### Pattern 1: Using Existing Validators

```python
# Import a validator
from qiskit_qward.examples.flip_coin.validator import FlipCoinValidator

# Create validator instance
validator = FlipCoinValidator(use_barriers=True)

# Run simulation
results = validator.run_simulation(show_histogram=True, num_jobs=100, shots_per_job=1024)

# Access analysis results
analysis = results["analysis"]["analyzer_0"]
print(f"Mean success rate: {analysis['mean_success_rate']:.2%}")

# Visualize results
validator.plot_analysis()
```

### Pattern 2: Creating Custom Validators

```python
from qiskit_qward.validators.base_validator import BaseValidator
from qiskit_qward.analysis.success_rate import SuccessRate

class MyCustomValidator(BaseValidator):
    def __init__(self):
        super().__init__(num_qubits=2, num_clbits=2, use_barriers=True, name="my_validator")
        
        # Define success criteria
        success_analyzer = SuccessRate()
        success_analyzer.set_success_criteria(lambda state: state == "00")
        self.add_analyzer(success_analyzer)
        
        # Build your circuit
        self._build_circuit()
    
    def _build_circuit(self):
        # Implement your circuit logic here
        self.h(0)
        self.cx(0, 1)
        self.measure([0, 1], [0, 1])
```

## Technical Details

### Validator Operation

1. Circuit initialization with configurable parameters
2. Validation algorithm implementation
3. Execution preparation
4. Backend selection and configuration
5. Results collection and primary processing
6. Metrics collection

### Analysis Pipeline

1. Raw results acceptance from validators
2. Processing based on algorithm expectations
3. Metrics calculation
4. Result formatting

## Development Guidelines

When extending Qward with custom validators:

1. Inherit from BaseValidator
2. Implement the required abstract methods:
   - `validate()`: Your circuit construction logic
3. Add appropriate metrics collection
4. Ensure compatibility with the analysis framework

## Technical Roadmap

Future technical enhancements include:

1. Parameter correlation analysis
2. Fixed parameter testing
3. Dynamic parameter testing
4. Depth analysis
5. Target performance testing
6. Advanced analysis capabilities
7. Visualization tools
8. Complete data management
9. Integration with the broader Qiskit ecosystem
