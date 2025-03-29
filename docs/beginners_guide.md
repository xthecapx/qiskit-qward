# Qward Beginner's Guide

This guide provides a comprehensive introduction to Qward, a framework for analyzing and validating quantum code execution quality on quantum processing units (QPUs).

## About the Project

Qward is designed to help quantum developers and researchers understand how their quantum algorithms perform on real hardware. The framework provides tools to execute quantum circuits on QPUs, collect comprehensive execution metrics, analyze circuit performance, validate algorithm correctness, generate insights about QPU behavior, and compare results across different backends.

## Installation

To install Qward, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/your-org/qiskit-qward.git
cd qiskit-qward
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your IBM Quantum credentials in a `.env` file:
```
IBM_QUANTUM_CHANNEL=ibm_quantum
IBM_QUANTUM_TOKEN=your_token_here
```

## Usage

Qward is built around validators that extend Qiskit's QuantumCircuit functionality. Here's how to use one of the built-in validators:

```python
from qward.validators.teleportation_validator import TeleportationValidator

# Create a validator
validator = TeleportationValidator(
    payload_size=3,
    gates=["h", "x"],
    use_barriers=True
)

# Run simulation
results = validator.run_simulation(show_histogram=True)

# Access results
print(f"Circuit depth: {results['circuit_metrics']['depth']}")
print(f"Circuit width: {results['circuit_metrics']['width']}")
print(f"Operation count: {results['circuit_metrics']['count_ops']}")

# Run on IBM hardware (if configured)
ibm_results = validator.run_on_ibm()
```

## Example Problem

Let's explore a simple use case: validating a quantum teleportation circuit.

1. **Problem**: You want to assess how well a quantum teleportation algorithm performs on different backends.

2. **Solution with Qward**:
```python
from qward.validators.teleportation_validator import TeleportationValidator
from qward.analysis.success_rate import SuccessRate

# Create a teleportation validator
validator = TeleportationValidator(
    payload_size=1,  # Single qubit teleportation
    gates=["h", "x"],  # Gates to prepare the payload
    use_barriers=True  # Add barriers for readability
)

# Run on simulator
sim_results = validator.run_simulation()

# Run on IBM hardware (if configured)
ibm_results = validator.run_on_ibm()

# Analyze success rate
analyzer = SuccessRate()
sim_rate = analyzer.analyze(sim_results)
ibm_rate = analyzer.analyze(ibm_results)

print(f"Simulator success rate: {sim_rate}")
print(f"IBM hardware success rate: {ibm_rate}")
```

3. **Interpretation**: The difference between simulator and hardware success rates gives you insights into the real-world performance limitations of the quantum hardware.

## Conclusion

Qward provides a structured way to evaluate quantum algorithm performance across different backends. By using validators and analyzers, you can gain insights into how your quantum code executes on real hardware and identify opportunities for optimization. As you become more familiar with the framework, you can create custom validators for your specific quantum algorithms.
