# Design Patterns Reference (QWARD Context)

## Strategy Pattern (Core of QWARD)

The Strategy pattern defines a family of algorithms, encapsulates each one, and makes them interchangeable. QWARD uses this for metric calculators.

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel
from qiskit import QuantumCircuit

class MetricCalculator(ABC):
    """Strategy interface."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    @abstractmethod
    def get_metrics(self) -> BaseModel: ...

    @abstractmethod
    def is_ready(self) -> bool: ...

class Scanner:
    """Context: uses strategies interchangeably."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._strategies: list[MetricCalculator] = []

    def add_strategy(self, strategy: MetricCalculator) -> None:
        self._strategies.append(strategy)

    def calculate_metrics(self) -> dict[str, pd.DataFrame]:
        results = {}
        for strategy in self._strategies:
            if strategy.is_ready():
                schema = strategy.get_metrics()
                flat = schema.to_flat_dict()
                results[strategy.name] = pd.DataFrame([flat])
        return results
```

## Factory Pattern

Create objects without specifying exact classes.

```python
class MetricFactory:
    _registry: dict[str, type[MetricCalculator]] = {}

    @classmethod
    def register(cls, name: str, metric_class: type[MetricCalculator]):
        cls._registry[name] = metric_class

    @classmethod
    def create(cls, name: str, circuit: QuantumCircuit, **kwargs) -> MetricCalculator:
        if name not in cls._registry:
            raise ValueError(f"Unknown metric: {name}")
        return cls._registry[name](circuit, **kwargs)

# Registration
MetricFactory.register("qiskit", QiskitMetrics)
MetricFactory.register("complexity", ComplexityMetrics)

# Usage
metric = MetricFactory.create("complexity", circuit)
```

## Observer Pattern

Notify dependents when state changes.

```python
from typing import Protocol, Callable

class MetricObserver(Protocol):
    def on_metric_calculated(self, metric_name: str, result: BaseModel) -> None: ...

class ObservableScanner:
    def __init__(self):
        self._observers: list[MetricObserver] = []

    def add_observer(self, observer: MetricObserver) -> None:
        self._observers.append(observer)

    def _notify(self, metric_name: str, result: BaseModel) -> None:
        for observer in self._observers:
            observer.on_metric_calculated(metric_name, result)
```

## Decorator Pattern

Add behavior to objects dynamically.

```python
import functools
import time

def timed(func):
    """Measure execution time of metric calculations."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__qualname__}: {elapsed:.3f}s")
        return result
    return wrapper

# Usage
class ComplexityMetrics(MetricCalculator):
    @timed
    def get_metrics(self) -> ComplexityMetricsSchema:
        ...
```

## Builder Pattern

Construct complex objects step by step.

```python
class ScannerBuilder:
    def __init__(self, circuit: QuantumCircuit):
        self._scanner = Scanner(circuit=circuit)

    def with_qiskit_metrics(self) -> 'ScannerBuilder':
        self._scanner.add_strategy(QiskitMetrics(self._scanner.circuit))
        return self

    def with_complexity_metrics(self) -> 'ScannerBuilder':
        self._scanner.add_strategy(ComplexityMetrics(self._scanner.circuit))
        return self

    def with_performance_metrics(self, job=None) -> 'ScannerBuilder':
        self._scanner.add_strategy(
            CircuitPerformanceMetrics(self._scanner.circuit, job=job)
        )
        return self

    def build(self) -> Scanner:
        return self._scanner

# Usage
scanner = (ScannerBuilder(circuit)
    .with_qiskit_metrics()
    .with_complexity_metrics()
    .build())
```

## Template Method Pattern

Define algorithm skeleton, defer steps to subclasses.

```python
class MetricCalculator(ABC):
    def calculate_and_validate(self) -> BaseModel:
        """Template method."""
        self._pre_check()
        raw = self._compute_raw_metrics()
        validated = self._validate(raw)
        self._post_process(validated)
        return validated

    def _pre_check(self):
        if not self.is_ready():
            raise MetricNotReadyError()

    @abstractmethod
    def _compute_raw_metrics(self) -> dict: ...

    def _validate(self, raw: dict) -> BaseModel:
        return self._schema_class.model_validate(raw)

    def _post_process(self, result: BaseModel):
        pass  # Optional hook for subclasses
```
