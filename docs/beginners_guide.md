# Qiskit Qward Beginner's Guide

This guide provides a comprehensive introduction to Qiskit Qward, a framework for analyzing and validating quantum code execution quality on quantum processing units (QPUs).

## What is Qiskit Qward?

Qiskit Qward is a framework built on top of Qiskit that helps quantum developers understand how their quantum algorithms perform on both simulators and real quantum hardware. It provides tools to:

1. Create and run quantum circuits
2. Collect execution metrics
3. Analyze circuit performance
4. Validate algorithm correctness
5. Compare results across different backends
6. Calculate circuit complexity metrics
7. Estimate quantum volume

## Key Concepts

### Validators

In Qiskit Qward, validators are the main components you'll work with. They are quantum circuits with added functionality for:
- Setting up experiments
- Defining success criteria
- Running simulations with multiple jobs/shots
- Executing on IBM Quantum hardware
- Analyzing results
- Calculating circuit complexity
- Estimating quantum volume

### Analysis

Qiskit Qward provides analysis tools to help you understand your quantum algorithm's performance:
- Success rate analysis
- Statistical aggregation
- Visualization tools
- Circuit complexity metrics
- Quantum volume estimation

## Getting Started

### Installation

You can set up Qiskit Qward in two ways:

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

Let's start with a simple example - the quantum coin flip. This uses a single qubit in superposition to simulate a fair coin toss.

```python
# If running in a notebook, ensure paths are set up correctly
from qiskit_qward.examples.flip_coin.scanner import ScanningQuantumFlipCoin

# Create the validator
scanner = ScanningQuantumFlipCoin(use_barriers=True)

# Show the circuit
print("Quantum Coin Flip Circuit:")
circuit_fig = scanner.draw()
display(circuit_fig)

# Run simulation
print("\nRunning quantum simulation jobs...")
results = scanner.run_simulation(
    show_histogram=True,
    num_jobs=100,
    shots_per_job=1024
)

# Check results
analysis = results["analysis"]["analyzer_0"]
print(f"\nSuccess rate (tails): {analysis['mean_success_rate']:.2%}")
print(f"Standard deviation: {analysis['std_success_rate']:.2%}")

# Examine complexity metrics
complexity = results["complexity_metrics"]
print(f"\nGate count: {complexity['gate_based_metrics']['gate_count']}")
print(f"Circuit depth: {complexity['gate_based_metrics']['circuit_depth']}")
print(f"Circuit volume: {complexity['standardized_metrics']['circuit_volume']}")

# Check quantum volume
qv = results["quantum_volume"]
print(f"\nQuantum Volume: {qv['standard_quantum_volume']}")
print(f"Enhanced Quantum Volume: {qv['enhanced_quantum_volume']}")
```

This circuit:
1. Applies a Hadamard gate to put a qubit in superposition (50% |0⟩, 50% |1⟩)
2. Measures the qubit
3. Runs multiple jobs to collect statistics
4. Analyzes the results to verify the coin is fair
5. Calculates circuit complexity metrics
6. Estimates quantum volume

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
from qiskit_qward.examples.two_doors_enigma.scanner import ScanningQuantumEnigma

# Create the validator
scanner = ScanningQuantumEnigma()

# Run simulation
results = scanner.run_simulation(show_histogram=True)

# Check analysis results
analysis = scanner.run_analysis()["analyzer_0"]
print(f"Success rate: {analysis['mean_success_rate']:.2%}")

# Examine complexity metrics
complexity = results["complexity_metrics"]
print(f"Circuit depth: {complexity['gate_based_metrics']['circuit_depth']}")
print(f"CNOT count: {complexity['gate_based_metrics']['cnot_count']}")
print(f"Entangling gate density: {complexity['entanglement_metrics']['entangling_gate_density']}")

# Check quantum volume
qv = results["quantum_volume"]
print(f"Quantum Volume: {qv['standard_quantum_volume']}")
print(f"Enhanced Quantum Volume: {qv['enhanced_quantum_volume']}")
```

This example demonstrates more advanced quantum concepts:
- Multiple qubits and classical bits
- Entanglement with CNOT gates
- Complex quantum logic
- Higher circuit complexity
- Larger quantum volume

## Understanding Circuit Complexity

Qiskit Qward provides comprehensive circuit complexity analysis through the `calculate_complexity_metrics()` method. This helps you understand your circuit's resource requirements and algorithmic complexity.

Key metrics include:

1. **Gate-based Metrics**:
   - Gate count, circuit depth, T-count, CNOT count
   - Two-qubit gate count and multi-qubit operation ratio

2. **Entanglement Metrics**:
   - Entangling gate density
   - Entangling width (maximum qubits that could be entangled)

3. **Standardized Metrics**:
   - Circuit volume (depth × width)
   - Gate density
   - Clifford vs non-Clifford gate ratios

4. **Advanced Metrics**:
   - Parallelism factors
   - Circuit efficiency 
   - Quantum resource utilization

You can access these metrics directly:

```python
# Create a scanner
scanner = ScanningQuantumFlipCoin()

# Calculate complexity metrics
metrics = scanner.calculate_complexity_metrics()

# Print selected metrics
print(f"Gate count: {metrics['gate_based_metrics']['gate_count']}")
print(f"Circuit depth: {metrics['gate_based_metrics']['circuit_depth']}")
print(f"Circuit volume: {metrics['standardized_metrics']['circuit_volume']}")
```

## Quantum Volume Estimation

Quantum Volume is an important metric for understanding a circuit's computational capacity. Qiskit Qward extends the standard quantum volume calculation with an enhanced metric that considers additional circuit characteristics.

The `estimate_quantum_volume()` method returns:

1. **Standard Quantum Volume**: Calculated as 2^n where n is the effective depth
2. **Enhanced Quantum Volume**: Factors in square ratio, density, and gate complexity
3. **Contributing Factors**: Details on what affects the quantum volume

Example usage:

```python
# Create a scanner
scanner = ScanningQuantumEnigma()

# Calculate quantum volume
qv = scanner.estimate_quantum_volume()

# Access quantum volume metrics
print(f"Standard QV: {qv['standard_quantum_volume']}")
print(f"Enhanced QV: {qv['enhanced_quantum_volume']}")
print(f"Effective depth: {qv['effective_depth']}")

# Examine contributing factors
for factor, value in qv['factors'].items():
    print(f"{factor}: {value}")
```

## Creating Your Own Validator

Once you're comfortable with the existing examples, you can create your own validator:

```python
from qiskit_qward.scanning_quantum_circuit import ScanningQuantumCircuit
from qiskit_qward.analysis.success_rate import SuccessRate

class MyFirstValidator(ScanningQuantumCircuit):
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

Your custom validator will automatically inherit all the advanced features like complexity metrics calculation and quantum volume estimation.

## Using Jupyter Notebooks

The easiest way to work with Qiskit Qward is using Jupyter notebooks. We provide example notebooks in the repository:

1. **Tutorials**: For detailed examples with explanations
   - Located in `docs/tutorials/example_tutorial.ipynb`

2. **How-to Guides**: For specific tasks
   - Located in `docs/how_tos/example_how_to.ipynb`

3. **Example Implementations**:
   - Flip Coin: `qiskit_qward/examples/flip_coin/notebook_demo.ipynb`
   - Two Doors Enigma: `qiskit_qward/examples/two_doors_enigma/notebook_demo.ipynb`

When using Docker with `./start.sh`, these notebooks will be accessible directly from the Jupyter Lab interface.

## Next Steps

After getting familiar with the basics, you can:

1. Explore the tutorials in the `docs/tutorials` directory
2. Check out the how-to guides in `docs/how_tos`
3. Review the [Technical Documentation](technical_docs.md) for more details
4. Learn how to customize success criteria and analyze different metrics
5. Understand circuit complexity and quantum volume metrics
6. Create validators for your own quantum algorithms

Remember, Qiskit Qward is about understanding how quantum algorithms perform in practice, helping you bridge the gap between theoretical quantum computing and practical implementation on real hardware.
