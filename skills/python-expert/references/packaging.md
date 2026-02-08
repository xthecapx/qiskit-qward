# Python Packaging Reference

## pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "qiskit-qward"
version = "0.9.0"
description = "Quantum circuit metrics and analysis library for Qiskit"
readme = "README.md"
license = {text = "Apache-2.0"}
requires-python = ">=3.9"
authors = [
    {name = "Your Name", email = "you@example.com"},
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Physics",
]

dependencies = [
    "qiskit>=1.0",
    "pydantic>=2.0",
    "pandas>=1.5",
    "numpy>=1.23",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "ruff",
    "mypy",
]
viz = [
    "matplotlib>=3.5",
    "seaborn>=0.12",
]
all = ["qiskit-qward[dev,viz]"]

[tool.setuptools.packages.find]
include = ["qward*"]

[tool.ruff]
line-length = 100
target-version = "py39"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow",
    "integration: marks integration tests",
]

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
```

## Dependency Management

### Pin versions for reproducibility
```
# requirements.txt (pinned for CI)
qiskit==1.3.0
pydantic==2.5.0
pandas==2.1.0
numpy==1.26.0
```

### Use ranges for library dependencies
```toml
# In pyproject.toml
dependencies = [
    "qiskit>=1.0,<2.0",
    "pydantic>=2.0",
]
```

## Package Structure

```
qiskit-qward/
├── pyproject.toml
├── README.md
├── LICENSE
├── qward/
│   ├── __init__.py          # Public API exports
│   ├── scanner.py
│   ├── metrics/
│   │   ├── __init__.py      # Export metric classes
│   │   └── ...
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── ...
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── ...
│   └── utils/
│       ├── __init__.py
│       └── ...
├── tests/
│   ├── conftest.py
│   └── ...
├── docs/
└── examples/
```

## __init__.py Best Practices

```python
# qward/__init__.py
"""QWARD: Quantum circuit metrics and analysis for Qiskit."""

from qward.scanner import Scanner

__version__ = "0.9.0"
__all__ = ["Scanner"]

# qward/metrics/__init__.py
from qward.metrics.qiskit_metrics import QiskitMetrics
from qward.metrics.complexity_metrics import ComplexityMetrics
from qward.metrics.circuit_performance import CircuitPerformanceMetrics
from qward.metrics.behavioral_metrics import BehavioralMetrics
from qward.metrics.structural_metrics import StructuralMetrics
from qward.metrics.element_metrics import ElementMetrics
from qward.metrics.quantum_specific_metrics import QuantumSpecificMetrics

__all__ = [
    "QiskitMetrics",
    "ComplexityMetrics",
    "CircuitPerformanceMetrics",
    "BehavioralMetrics",
    "StructuralMetrics",
    "ElementMetrics",
    "QuantumSpecificMetrics",
]
```

## Versioning (SemVer)

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

```python
# Single source of truth
__version__ = "0.9.0"  # in __init__.py
```

## Development Workflow

```bash
# Install in editable mode
pip install -e ".[dev,viz]"

# Run linter
ruff check .
ruff format .

# Run type checker
mypy qward/

# Run tests
pytest --cov=qward

# Build package
python -m build

# Upload to PyPI
twine upload dist/*
```

## Entry Points (CLI)

```toml
[project.scripts]
qward-scan = "qward.cli:main"
```

```python
# qward/cli.py
def main():
    import argparse
    parser = argparse.ArgumentParser(description="QWARD circuit scanner")
    parser.add_argument("circuit_file", help="QASM file to analyze")
    args = parser.parse_args()
    # ... analyze circuit
```
