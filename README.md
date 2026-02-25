# QWARD - Quantum Circuit Analysis and Runtime Development

![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-informational)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-informational)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-%E2%89%A5%201.0.0-6133BD)](https://github.com/Qiskit/qiskit)
[![Code style: Black](https://img.shields.io/badge/Code%20style-Black-000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18773713.svg)](https://doi.org/10.5281/zenodo.18773713)

QWARD is a comprehensive framework for analyzing quantum circuits and validating quantum code execution quality on quantum processing units (QPUs). It provides tools to analyze circuit complexity, measure performance metrics, and visualize quantum algorithm behavior.

## 🚀 Quick Start

```python
from qiskit import QuantumCircuit
from qward import Scanner

# Create a quantum circuit
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)

# Analyze with all metrics in one line
Scanner(circuit).scan().summary()
```

### Full Control

```python
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics
from qward.visualization import Visualizer

# Build scanner and visualizer explicitly
scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics])
metrics = scanner.calculate_metrics()
visualizer = Visualizer(scanner=scanner)
dashboards = visualizer.create_dashboard(save=True)
```

## 📚 Documentation

### For Users
- **[Project Overview](docs/project_overview.md)** - Learn about QWARD's goals and capabilities
- **[Installation Guide](INSTALL.md)** - Get QWARD up and running
- **[Beginner's Guide](docs/beginners_guide.md)** - Start here if you're new to QWARD
- **[Quickstart Guide](docs/quickstart_guide.md)** - Jump right into using QWARD
- **[Architecture Overview](docs/architecture.md)** - Understand QWARD's design and components
- **[Visualization Guide](docs/visualization_guide.md)** - Learn about QWARD's visualization capabilities

### For Developers
- **[Developer Guide](docs/developer_guide.md)** - Development setup and code quality standards
- **[Technical Documentation](docs/technical_docs.md)** - Deep dive into QWARD's implementation
- **[Contribution Guidelines](CONTRIBUTING.md)** - How to contribute to QWARD
- **[API Documentation](docs/apidocs/index.rst)** - Complete API reference

## 🎯 Key Features

- **Circuit Analysis**: Comprehensive metrics for quantum circuit complexity and structure
- **Performance Monitoring**: Track success rates, fidelity, and execution statistics
- **Visualization**: Rich, interactive plots and dashboards for metric analysis
- **Schema Validation**: Type-safe metrics with Pydantic-based validation
- **Extensible Architecture**: Plugin-based system for custom metrics and visualizations
- **Multi-Backend Support**: Works with Qiskit Aer, IBM Quantum, and other providers

## 🛠️ Installation

```bash
# Install from PyPI (when available)
pip install qward

# Or install from source
git clone https://github.com/your-org/qiskit-qward.git
cd qiskit-qward
pip install -e .
```

## 📖 Examples

Explore comprehensive examples in the [`qward/examples/`](qward/examples/) directory:

- **[Basic Usage](qward/examples/example_visualizer.py)** - Complete workflow examples
- **[Circuit Performance](qward/examples/circuit_performance_demo.py)** - Performance analysis
- **[Visualization Demo](qward/examples/visualization_demo.py)** - Visualization capabilities
- **[Aer Integration](qward/examples/aer.py)** - Using QWARD with Qiskit Aer

## 🧪 Development & Linting

### Quick Local Check

Use `verify.sh` for a fast local validation against your active Python environment:

```bash
./verify.sh
```

### Replicating CI Exactly

CI runs `tox -elint` with **Python 3.11** in an isolated environment. To replicate this locally:

```bash
# Requires Python 3.11 available via pyenv
PYENV_VERSION=3.11 pyenv exec tox -elint
```

This creates the same isolated environment as CI (same Python version, same pinned tool versions, no extra packages like IPython) and runs:
1. `black --check .` - Code formatting
2. `pylint -rn --disable=C,R --ignore-paths=qward/examples qward tests` - Linting
3. `mypy --exclude qward/examples qward` - Type checking

### Running Tests

```bash
# Quick local test run
python -m pytest tests/ -v

# Full CI-equivalent test run with tox
PYENV_VERSION=3.11 pyenv exec tox -epy311
```

## 🤝 Contributing

We welcome contributions! Please see our [Contribution Guidelines](CONTRIBUTING.md) for details on:

- Setting up the development environment
- Code style and quality standards
- Testing requirements
- Submitting pull requests

## 📝 How to Cite

If you use QWARD in your research, please cite it as follows:

> Márquez, Cristian and Sierra-Sosa, Daniel and Garcés, Kelly. (2026). xthecapx/qiskit-qward (v0.18.0). Zenodo. https://doi.org/10.5281/zenodo.18773713

**BibTeX:**

```bibtex
@software{qward2026,
  author       = {Márquez, Cristian and Sierra-Sosa, Daniel and Garcés, Kelly},
  title        = {xthecapx/qiskit-qward},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v0.18.0},
  doi          = {10.5281/zenodo.18773713},
  url          = {https://doi.org/10.5281/zenodo.18773713}
}
```

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE.txt).

## 🔗 Links

- **Documentation**: [docs/](docs/)
- **Examples**: [qward/examples/](qward/examples/)
- **Issues**: [GitHub Issues](https://github.com/your-org/qiskit-qward/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/qiskit-qward/discussions)

---

*QWARD is designed to help quantum developers and researchers understand and optimize their quantum algorithms through comprehensive analysis and visualization tools.*
