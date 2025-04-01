# Quickstart Guide

Qiskit Qward is a framework for analyzing and validating quantum code execution quality on quantum processing units (QPUs). This guide will help you quickly get started with using Qiskit Qward.

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

Qiskit Qward provides two main ways to use the framework:

1. Using existing validators
2. Creating your own custom validators

### Using Existing Validators

#### Example 1: Quantum Coin Flip

The Quantum Coin Flip validator demonstrates a simple quantum circuit that simulates a fair coin toss:

```python
from qiskit_qward.examples.flip_coin.validator import FlipCoinValidator

# Create a validator
validator = FlipCoinValidator(use_barriers=True)

# Run simulation with multiple jobs
results = validator.run_simulation(
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

# Plot results
validator.plot_analysis(ideal_rate=0.5)

# Run on IBM Quantum hardware (if configured)
ibm_results = validator.run_on_ibm()
```

#### Example 2: Two Doors Enigma

For a more complex example, try the Two Doors Enigma validator:

```python
from qiskit_qward.examples.two_doors_enigma.validator import QuantumEnigmaValidator

# Create the validator
validator = QuantumEnigmaValidator()

# Run simulation
results = validator.run_simulation(show_histogram=True)

# Access analysis results
analysis = validator.run_analysis()["analyzer_0"]
print(f"Mean success rate: {analysis['mean_success_rate']:.2%}")

# Plot analysis results
validator.plot_analysis(ideal_rate=1.0)
```

### Creating Custom Validators

To create your own validator, extend the ScanningQuantumCircuit class:

```python
from qiskit_qward.scanning_quantum_circuit import ScanningQuantumCircuit
from qiskit_qward.analysis.success_rate import SuccessRate

class MyCustomValidator(ScanningQuantumCircuit):
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

## Using Jupyter Notebooks

The easiest way to work with Qiskit Qward is using Jupyter notebooks. When using the Docker setup with `./start.sh`, you'll have access to:

1. **Tutorials**: `docs/tutorials/example_tutorial.ipynb`
2. **How-to Guides**: `docs/how_tos/example_how_to.ipynb`
3. **Example Notebooks**: 
   - `qiskit_qward/examples/flip_coin/notebook_demo.ipynb`
   - `qiskit_qward/examples/two_doors_enigma/notebook_demo.ipynb`

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
