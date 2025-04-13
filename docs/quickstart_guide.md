# Quickstart Guide

Qward is a framework for analyzing and validating quantum code execution quality on quantum processing units (QPUs). This guide will help you quickly get started with using Qward.

## Installation

### Option 1: Local Installation

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

### Option 2: Using Docker

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

## Usage

### Core Components

Qward provides two main ways to use the framework:

1. Using existing scanning circuits
2. Creating your own custom scanning circuits

### Using Existing Scanning Circuits

#### Example 1: Quantum Coin Flip

The Quantum Coin Flip scanner demonstrates a simple quantum circuit that simulates a fair coin toss:

```python
from qward.examples.flip_coin.scanner import ScanningQuantumFlipCoin

# Create a scanner
scanner = ScanningQuantumFlipCoin(use_barriers=True)

# Run simulation with multiple jobs
results = scanner.run_simulation(
    show_histogram=True,   # Display histogram of results
    num_jobs=1000,         # Run 1000 independent jobs 
    shots_per_job=1024     # With 1024 shots each
)

# Access analysis results
analysis = results["analysis"]["analyzer_0"]
print(f"Mean success rate: {analysis['mean_success_rate']:.2%}")
print(f"Standard deviation: {analysis['std_success_rate']:.2%}")
print(f"Average heads count: {analysis['average_counts']['heads']:.2f}")
print(f"Average tails count: {analysis['average_counts']['tails']:.2f}")

# Access complexity metrics
complexity = results["complexity_metrics"]
print(f"\nCircuit complexity metrics:")
print(f"Gate count: {complexity['gate_based_metrics']['gate_count']}")
print(f"Circuit depth: {complexity['gate_based_metrics']['circuit_depth']}")
print(f"Circuit volume: {complexity['standardized_metrics']['circuit_volume']}")

# Access quantum volume
qv = results["quantum_volume"]
print(f"\nQuantum Volume: {qv['standard_quantum_volume']}")
print(f"Enhanced Quantum Volume: {qv['enhanced_quantum_volume']}")

# Plot results
scanner.plot_analysis(ideal_rate=0.5)

# Run on IBM Quantum hardware (if configured)
ibm_results = scanner.run_on_ibm()
```

#### Example 2: Two Doors Enigma

For a more complex example, try the Two Doors Enigma scanner:

```python
from qward.examples.two_doors_enigma.scanner import ScanningQuantumEnigma

# Create the scanner
scanner = ScanningQuantumEnigma()

# Run simulation
results = scanner.run_simulation(show_histogram=True)

# Access analysis results
analysis = scanner.run_analysis()["analyzer_0"]
print(f"Mean success rate: {analysis['mean_success_rate']:.2%}")

# Access complexity metrics
complexity = results["complexity_metrics"]
print(f"\nCircuit complexity metrics:")
print(f"Gate count: {complexity['gate_based_metrics']['gate_count']}")
print(f"CNOT count: {complexity['gate_based_metrics']['cnot_count']}")
print(f"Entangling gate density: {complexity['entanglement_metrics']['entangling_gate_density']}")

# Access quantum volume
qv = results["quantum_volume"]
print(f"\nQuantum Volume: {qv['standard_quantum_volume']}")
print(f"Enhanced Quantum Volume: {qv['enhanced_quantum_volume']}")

# Plot analysis results
scanner.plot_analysis(ideal_rate=1.0)
```

### Creating Custom Scanning Circuits

To create your own scanner, extend the ScanningQuantumCircuit class:

```python
from qward.scanning_quantum_circuit import ScanningQuantumCircuit
from qward.analysis.success_rate import SuccessRate

class MyCustomScanner(ScanningQuantumCircuit):
    def __init__(self, use_barriers=True):
        # Initialize with desired qubits and classical bits
        super().__init__(num_qubits=2, num_clbits=2, use_barriers=use_barriers, name="my_circuit")
        
        # Define success criteria
        def success_criteria(state):
            # For example, consider "00" a success
            return state == "00"
        
        # Add success rate analyzer with custom criteria
        success_analyzer = SuccessRate()
        success_analyzer.set_success_criteria(success_criteria)
        self.add_analyzer(success_analyzer)
        
        # Build the circuit
        self._build_circuit()
    
    def _build_circuit(self):
        # Implement your quantum circuit here
        self.h(0)              # Apply Hadamard to qubit 0
        self.cx(0, 1)          # CNOT with control qubit 0, target qubit 1
        
        if self.use_barriers:
            self.barrier()
            
        self.measure([0, 1], [0, 1])  # Measure both qubits
```

After running your custom scanner, you can access complexity metrics and quantum volume estimates:

```python
# Create your scanner
my_scanner = MyCustomScanner()

# Run simulation
results = my_scanner.run_simulation()

# Calculate complexity metrics directly
complexity_metrics = my_scanner.calculate_complexity_metrics()
print(f"Circuit depth: {complexity_metrics['gate_based_metrics']['circuit_depth']}")
print(f"Gate density: {complexity_metrics['standardized_metrics']['gate_density']}")

# Estimate quantum volume
qv_estimate = my_scanner.estimate_quantum_volume()
print(f"Quantum Volume: {qv_estimate['enhanced_quantum_volume']}")
```

## Circuit Complexity and Quantum Volume

Qward provides two key methods for analyzing circuit properties:

### 1. Calculate Complexity Metrics

The `calculate_complexity_metrics()` method returns detailed circuit complexity metrics:

```python
# Get metrics for any circuit
metrics = scanner.calculate_complexity_metrics()
```

Key metric categories include:
- Gate-based metrics (count, depth, T-count, CNOT count)
- Entanglement metrics (gate density, entangling width)
- Standardized metrics (volume, density, Clifford ratios)
- Advanced metrics (parallelism, efficiency)
- Derived metrics (weighted complexity)

### 2. Estimate Quantum Volume

The `estimate_quantum_volume()` method provides quantum volume metrics:

```python
# Get quantum volume for any circuit
qv = scanner.estimate_quantum_volume()
```

This returns both standard quantum volume (2^n) and an enhanced volume that considers:
- Square ratio (how close depth is to width)
- Circuit density (operations per time-step)
- Multi-qubit operation ratio
- Connectivity factors

## Using Jupyter Notebooks

The easiest way to work with Qward is using Jupyter notebooks. When using the Docker setup with `./start.sh`, you'll have access to:

1. **Tutorials**: `docs/tutorials/example_tutorial.ipynb`
2. **How-to Guides**: `docs/how_tos/example_how_to.ipynb`
3. **Example Notebooks**: 
   - `qward/examples/flip_coin/notebook_demo.ipynb`
   - `qward/examples/two_doors_enigma/notebook_demo.ipynb`

## IBM Quantum Execution

To run on real quantum hardware, you need an IBM Quantum account:

1. Register at [IBM Quantum Experience](https://quantum-computing.ibm.com/)
2. Get your API token from your account settings
3. Add it to your `.env` file:
   ```
   IBM_QUANTUM_CHANNEL=ibm_quantum
   IBM_QUANTUM_TOKEN=your_token_here
   ```

## Next Steps

- Explore the [Tutorials](tutorials/index.rst) for more detailed examples
- Check the [Technical Documentation](technical_docs.md) for advanced usage
- Read the [API Documentation](apidocs/index.rst) for a complete reference
