---
name: python-expert
description: Expert Python development skill for scientific library design and best practices. Use when writing Python code, designing APIs, creating Pydantic schemas, building abstract base classes, implementing design patterns (Strategy, Factory, Observer), writing type-safe code with type hints, structuring Python packages, creating tests with pytest, managing dependencies, optimizing performance, or following PEP conventions. Tailored for the QWARD Qiskit extension library.
---

# Python Expert

## Overview

Expert-level Python development guidance for building and maintaining the QWARD quantum computing metrics library. Covers library design patterns, type safety, testing, packaging, and performance optimization.

## QWARD Architecture Quick Reference

```
qward/
├── scanner.py              # Scanner (Context in Strategy pattern)
├── metrics/
│   ├── base_metric.py      # MetricCalculator (Strategy Interface)
│   ├── types.py            # MetricsType, MetricsId enums
│   ├── qiskit_metrics.py   # Concrete Strategy
│   ├── complexity_metrics.py
│   ├── circuit_performance.py
│   ├── behavioral_metrics.py
│   ├── structural_metrics.py
│   ├── element_metrics.py
│   └── quantum_specific_metrics.py
├── schemas/                # Pydantic validation
├── visualization/          # Strategy-based visualizers
└── utils/
```

## Core Patterns Used in QWARD

### Strategy Pattern (Primary)

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel

class MetricCalculator(ABC):
    """Strategy interface - all metrics implement this."""

    def __init__(self, circuit):
        self._circuit = circuit

    @abstractmethod
    def get_metrics(self) -> BaseModel:
        """Return validated Pydantic schema object."""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        ...
```

### Pydantic Schema Validation

```python
from pydantic import BaseModel, Field, field_validator

class ComplexityMetricsSchema(BaseModel):
    gate_count: int = Field(ge=0, description="Total gate count")
    gate_density: float = Field(ge=0.0, le=1.0)
    depth: int = Field(ge=0)

    def to_flat_dict(self) -> dict:
        """Convert nested schema to flat dict for DataFrame."""
        return {k: v for k, v in self.model_dump().items()}

    @field_validator('gate_density')
    @classmethod
    def validate_density(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('gate_density must be between 0 and 1')
        return v
```

## Reference Documentation

Load these as needed based on your task:

- **`references/design-patterns.md`** - Strategy, Factory, Observer, Decorator patterns with QWARD examples
- **`references/type-safety.md`** - Type hints, Pydantic, generics, Protocol, runtime validation
- **`references/testing.md`** - pytest fixtures, parametrize, mocking, coverage, property-based testing
- **`references/packaging.md`** - pyproject.toml, dependencies, versioning, publishing, CLI entry points

## Python Best Practices for QWARD

### Type Hints (Always Use)

```python
from typing import Optional, Union
from qiskit import QuantumCircuit
from pydantic import BaseModel

def analyze_circuit(
    circuit: QuantumCircuit,
    shots: int = 1024,
    success_criteria: Optional[callable] = None,
) -> dict[str, pd.DataFrame]:
    ...
```

### Abstract Base Classes

```python
from abc import ABC, abstractmethod
from qward.metrics.types import MetricsType, MetricsId

class MetricCalculator(ABC):
    @abstractmethod
    def _get_metric_type(self) -> MetricsType: ...

    @abstractmethod
    def _get_metric_id(self) -> MetricsId: ...

    @abstractmethod
    def is_ready(self) -> bool: ...

    @abstractmethod
    def get_metrics(self) -> BaseModel: ...
```

### Properties Over Direct Attribute Access

```python
class Scanner:
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._strategies: list[MetricCalculator] = []

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    @circuit.setter
    def circuit(self, value: QuantumCircuit) -> None:
        self._circuit = value
```

### Enums for Constants

```python
from enum import Enum

class MetricsType(Enum):
    PRE_RUNTIME = "pre_runtime"
    POST_RUNTIME = "post_runtime"

class MetricsId(Enum):
    QISKIT = "qiskit"
    COMPLEXITY = "complexity"
    CIRCUIT_PERFORMANCE = "circuit_performance"
```

### Context Managers

```python
from contextlib import contextmanager

@contextmanager
def timer(label: str):
    import time
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.3f}s")

# Usage
with timer("Metric calculation"):
    results = scanner.calculate_metrics()
```

### Dataclasses for Configuration

```python
from dataclasses import dataclass, field

@dataclass
class PlotConfig:
    figsize: tuple[int, int] = (10, 6)
    dpi: int = 300
    style: str = "default"
    color_palette: list[str] = field(default_factory=lambda: ["#1f77b4", "#ff7f0e"])
    save_format: str = "png"
    grid: bool = True
    alpha: float = 0.7
```

## Testing Patterns

### pytest Fixtures

```python
import pytest
from qiskit import QuantumCircuit

@pytest.fixture
def bell_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

@pytest.fixture
def ghz_circuit():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return qc

def test_qiskit_metrics(bell_circuit):
    from qward.metrics import QiskitMetrics
    metrics = QiskitMetrics(bell_circuit)
    result = metrics.get_metrics()
    assert result.basic_metrics.num_qubits == 2
    assert result.basic_metrics.depth > 0
```

### Parametrized Tests

```python
@pytest.mark.parametrize("num_qubits,expected_depth", [
    (2, 2),
    (3, 3),
    (5, 5),
])
def test_ghz_depth(num_qubits, expected_depth):
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    assert qc.depth() == expected_depth
```

## Error Handling

```python
class QWARDError(Exception):
    """Base exception for QWARD library."""
    pass

class MetricNotReadyError(QWARDError):
    """Raised when metric prerequisites are not met."""
    pass

class SchemaValidationError(QWARDError):
    """Raised when metric data fails schema validation."""
    pass

# Usage
def get_metrics(self) -> BaseModel:
    if not self.is_ready():
        raise MetricNotReadyError(
            f"{self.__class__.__name__} requires a valid circuit"
        )
    ...
```

## Performance Tips

1. **Use `__slots__`** for classes with many instances
2. **Cache expensive computations** with `functools.lru_cache`
3. **Use generators** for large data pipelines
4. **Profile before optimizing**: `cProfile`, `line_profiler`
5. **NumPy vectorization** over Python loops for numeric work
6. **Avoid repeated `transpile()`** calls -- cache transpiled circuits

## Code Style

- Follow PEP 8, enforced by `ruff` or `black`
- Docstrings: Google style or NumPy style (be consistent)
- Max line length: 88 (black default) or 100
- Import order: stdlib, third-party, local (enforced by `isort`)
- Use `f-strings` over `.format()` or `%`
- Prefer `pathlib.Path` over `os.path`
