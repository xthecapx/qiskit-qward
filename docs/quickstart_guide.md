# Quickstart Guide

Qward is a framework for analyzing and validating quantum code execution quality on quantum processing units (QPUs). This guide will help you quickly get started with using Qward.

## Installation

To install Qward and its dependencies:

```bash
# Clone the repository
git clone https://github.com/your-org/qiskit-qward.git
cd qiskit-qward

# Install dependencies
pip install -r requirements.txt
```

For IBM Quantum backend access, create a `.env` file with your credentials:
```
IBM_QUANTUM_CHANNEL=ibm_quantum
IBM_QUANTUM_TOKEN=your_token_here
```

## Usage

### Core Components

#### 1. Validators

Validators are the main building blocks of Qward. They extend Qiskit's QuantumCircuit and provide validation functionality:

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

# Run on IBM hardware
ibm_results = validator.run_on_ibm()
```

#### 2. Analysis Tools

Analyze the results of your quantum circuit executions:

```python
from qward.analysis.success_rate import SuccessRate

# Create analyzer
analyzer = SuccessRate()

# Analyze results
success_rate = analyzer.analyze(results)
print(f"Success rate: {success_rate}")
```

#### 3. Creating Custom Validators

Extend the BaseValidator to create custom validators for your algorithms:

```python
from qward.validators.base_validator import BaseValidator

class YourAlgorithmValidator(BaseValidator):
    def __init__(self, num_qubits=1, num_clbits=1, use_barriers=True, name=None):
        super().__init__(num_qubits, num_clbits, use_barriers, name)
        
    def validate(self):
        # Your validation logic
        pass
```
