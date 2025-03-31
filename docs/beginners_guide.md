# Qiskit Qward Beginner's Guide

This guide provides a comprehensive introduction to Qiskit Qward, a framework for analyzing and validating quantum code execution quality on quantum processing units (QPUs).

## What is Qiskit Qward?

Qiskit Qward is a framework built on top of Qiskit that helps quantum developers understand how their quantum algorithms perform on both simulators and real quantum hardware. It provides tools to:

1. Create and run quantum circuits
2. Collect execution metrics
3. Analyze circuit performance
4. Validate algorithm correctness
5. Compare results across different backends

## Key Concepts

### Validators

In Qiskit Qward, validators are the main components you'll work with. They are quantum circuits with added functionality for:
- Setting up experiments
- Defining success criteria
- Running simulations with multiple jobs/shots
- Executing on IBM Quantum hardware
- Analyzing results

### Analysis

Qiskit Qward provides analysis tools to help you understand your quantum algorithm's performance:
- Success rate analysis
- Statistical aggregation
- Visualization tools

## Getting Started

### Installation

To install Qiskit Qward:

```bash
# Clone the repository
git clone https://github.com/your-org/qiskit-qward.git
cd qiskit-qward

# Install dependencies
pip install -r requirements.txt
```

For IBM Quantum access, create a `.env` file with your credentials:
```
IBM_QUANTUM_CHANNEL=ibm_quantum
IBM_QUANTUM_TOKEN=your_token_here
```

### First Steps: The Quantum Coin Flip

Let's start with a simple example - the quantum coin flip. This uses a single qubit in superposition to simulate a fair coin toss.

```python
from qiskit_qward.examples.flip_coin.validator import FlipCoinValidator

# Create the validator
validator = FlipCoinValidator(use_barriers=True)

# Show the circuit
print("Quantum Coin Flip Circuit:")
circuit_fig = validator.draw()
display(circuit_fig)

# Run simulation
print("\nRunning quantum simulation jobs...")
results = validator.run_simulation(
    show_histogram=True,
    num_jobs=100,
    shots_per_job=1024
)

# Check results
analysis = results["analysis"]["analyzer_0"]
print(f"\nSuccess rate (tails): {analysis['mean_success_rate']:.2%}")
print(f"Standard deviation: {analysis['std_success_rate']:.2%}")
```

This circuit:
1. Applies a Hadamard gate to put a qubit in superposition (50% |0⟩, 50% |1⟩)
2. Measures the qubit
3. Runs multiple jobs to collect statistics
4. Analyzes the results to verify the coin is fair

### Understanding the Circuit

The quantum coin flip circuit is simple but demonstrates important quantum principles:

```
┌───┐┌─┐
┤ H ├┤M├
└───┘└╥┘
     ┌╨┐
     │0│
     └─┘
```

1. **H gate**: Creates an equal superposition of |0⟩ and |1⟩
2. **Measurement**: Collapses the superposition into either 0 (heads) or 1 (tails)

The results should show approximately 50% heads and 50% tails, demonstrating quantum randomness.

## Going Further: The Two Doors Enigma

For a more complex example, try the Two Doors Enigma validator. This implements a quantum solution to a classic puzzle involving truth-tellers and liars.

```python
from qiskit_qward.examples.two_doors_enigma.quantum_enigma import QuantumEnigmaValidator

# Create the validator
validator = QuantumEnigmaValidator()

# Run simulation
results = validator.run_simulation(show_histogram=True)

# Check analysis results
analysis = validator.run_analysis()["analyzer_0"]
print(f"Success rate: {analysis['mean_success_rate']:.2%}")
```

This example demonstrates more advanced quantum concepts:
- Multiple qubits and classical bits
- Entanglement with CNOT gates
- Complex quantum logic

## Creating Your Own Validator

Once you're comfortable with the existing examples, you can create your own validator:

```python
from qiskit_qward.validators.base_validator import BaseValidator
from qiskit_qward.analysis.success_rate import SuccessRate

class MyFirstValidator(BaseValidator):
    def __init__(self, use_barriers=True):
        # Initialize with one qubit and one classical bit
        super().__init__(num_qubits=1, num_clbits=1, use_barriers=use_barriers, name="my_first")
        
        # Define what "success" means for this circuit
        # In this case, measuring |1⟩ is considered success
        def success_criteria(state):
            return state == "1"
        
        # Add success rate analyzer
        success_analyzer = SuccessRate()
        success_analyzer.set_success_criteria(success_criteria)
        self.add_analyzer(success_analyzer)
        
        # Build your circuit
        self._build_circuit()
    
    def _build_circuit(self):
        # Apply X gate to flip from |0⟩ to |1⟩
        self.x(0)
        
        if self.use_barriers:
            self.barrier()
            
        # Measure the qubit
        self.measure(0, 0)
```

## Next Steps

After getting familiar with the basics, you can:

1. Explore the [Quantum Coin Flip example](examples/flip_coin/notebook_demo.ipynb) in detail
2. Study the [Two Doors Enigma example](examples/two_doors_enigma/notebook_demo.ipynb)
3. Check the [Technical Documentation](technical_docs.md) for more details
4. Learn how to customize success criteria and analyze different metrics
5. Create validators for your own quantum algorithms

Remember, Qiskit Qward is about understanding how quantum algorithms perform in practice, helping you bridge the gap between theoretical quantum computing and practical implementation on real hardware.
