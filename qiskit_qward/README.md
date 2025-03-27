# Qward

A framework for analyzing and validating quantum code execution quality on quantum processing units (QPUs). Qward helps developers and researchers understand how their quantum algorithms perform on real hardware, providing insights into QPU behavior and code quality metrics.

## Overview

Qward provides tools to:
- Execute quantum circuits on QPUs
- Collect comprehensive execution metrics
- Analyze circuit performance
- Validate algorithm correctness
- Generate insights about QPU behavior
- Compare results across different backends

## Current Implementation Status

This project is under active development. Here's the current status:

### ✅ Implemented Features
- Base validator system (extends Qiskit's QuantumCircuit)
- Algorithm validators (Teleportation, FlipCoin)
- Circuit execution on simulators and IBM Quantum hardware
- Basic analysis framework with success rate validator
- Circuit metrics collection (depth, width, size, etc.)
- Execution metrics (basic success rates)

### 🚧 In Progress / Coming Soon
- Experiments framework
- Parameter correlation analysis
- Fixed parameter testing
- Dynamic parameter testing
- Depth analysis
- Target performance testing
- Advanced analysis capabilities
- Visualization tools
- Complete data management
- Integration with Qiskit ecosystem

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. Set up your IBM Quantum credentials in a `.env` file:
```
IBM_QUANTUM_CHANNEL=ibm_quantum
IBM_QUANTUM_TOKEN=your_token_here
```

2. Run the example (note: Experiments framework is under development):
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

## Core Components

### 1. Base Validator System (✅ Implemented)

The framework provides a base validator system that can be extended for different quantum algorithms:

```python
from qward.validators.base_validator import BaseValidator

class YourAlgorithmValidator(BaseValidator):
    def __init__(self, num_qubits=1, num_clbits=1, use_barriers=True, name=None):
        super().__init__(num_qubits, num_clbits, use_barriers, name)
        
    def validate(self):
        # Your validation logic
        pass
```

### 2. Experiments Framework (🚧 Coming Soon)

A flexible framework for running and analyzing quantum experiments. This component is currently under development and will include:

```python
from qward.experiments.experiments import Experiments  # Coming soon

experiments = Experiments()
results = experiments.run_parameter_correlation(
    validator=your_validator,
    param_ranges=param_ranges
)
```

## Features

1. **Multiple Experiment Types** (🚧 Coming Soon)
   - Parameter correlation analysis
   - Fixed parameter testing
   - Dynamic parameter testing
   - Depth analysis
   - Target performance testing

2. **Comprehensive Metrics**
   - ✅ Circuit metrics (depth, width, size)
   - ✅ Basic execution metrics (success rates)
   - 🚧 Advanced hardware metrics (job duration, quantum duration)

3. **Analysis Capabilities**
   - ✅ Basic success rate analysis
   - 🚧 Advanced statistical analysis
   - 🚧 Error distribution analysis
   - 🚧 Performance correlation
   - 🚧 Enhanced visualization tools

4. **Data Management**
   - ✅ Basic CSV export
   - 🚧 Advanced results serialization
   - 🚧 Data preprocessing tools

## Development Roadmap

In the upcoming releases, we plan to implement:

1. **Q2 2025**
   - Complete Experiments framework
   - Add visualization tools
   - Improve metrics collection

2. **Q3 2025**
   - Add more validator examples
   - Enhance analysis capabilities
   - Improve documentation

3. **Q4 2025**
   - Integration with Qiskit ecosystem
   - Add support for more quantum backends
   - Release v1.0

## Contributing

We welcome contributions! To add your own validator:

1. Create a new validator class extending BaseValidator
2. Implement required methods
3. Add tests
4. Update documentation
5. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
