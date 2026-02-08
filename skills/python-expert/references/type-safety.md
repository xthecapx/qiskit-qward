# Type Safety Reference

## Type Hints (Always Use)

```python
from typing import Optional, Union, Any
from collections.abc import Callable, Sequence

# Basic types
def process(name: str, count: int, ratio: float, active: bool) -> str: ...

# Collections (Python 3.9+)
def analyze(circuits: list[QuantumCircuit]) -> dict[str, pd.DataFrame]: ...

# Optional
def run(circuit: QuantumCircuit, shots: int = 1024,
        criteria: Optional[Callable[[str], bool]] = None) -> dict: ...

# Union (Python 3.10+ use |)
JobType = Union[AerJob, RuntimeJob]
# or: JobType = AerJob | RuntimeJob
```

## Pydantic Models

```python
from pydantic import BaseModel, Field, field_validator, model_validator

class GateMetricsSchema(BaseModel):
    gate_count: int = Field(ge=0, description="Total number of gates")
    two_qubit_gate_count: int = Field(ge=0)
    t_gate_count: int = Field(ge=0)

    @field_validator('gate_count')
    @classmethod
    def validate_gate_count(cls, v):
        if v < 0:
            raise ValueError('gate_count must be non-negative')
        return v

class ComplexityMetricsSchema(BaseModel):
    gate_based: GateMetricsSchema
    entanglement: EntanglementMetricsSchema
    standardized: StandardizedMetricsSchema

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten nested schema for DataFrame conversion."""
        result = {}
        for field_name, field_value in self:
            if isinstance(field_value, BaseModel):
                for k, v in field_value.model_dump().items():
                    result[f"{field_name}_{k}"] = v
            else:
                result[field_name] = field_value
        return result

    @model_validator(mode='after')
    def validate_consistency(self):
        """Cross-field validation."""
        if self.gate_based.gate_count == 0 and self.entanglement.entanglement_ratio > 0:
            raise ValueError('Cannot have entanglement with zero gates')
        return self
```

## Protocol (Structural Subtyping)

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Measurable(Protocol):
    def get_metrics(self) -> BaseModel: ...
    def is_ready(self) -> bool: ...

# Any class implementing these methods satisfies the Protocol
# No inheritance needed
def process_metric(m: Measurable) -> dict:
    if m.is_ready():
        return m.get_metrics().model_dump()
```

## Generics

```python
from typing import TypeVar, Generic

T = TypeVar('T', bound=BaseModel)

class MetricResult(Generic[T]):
    def __init__(self, schema: T, metadata: dict):
        self.schema = schema
        self.metadata = metadata

    def to_dict(self) -> dict:
        return self.schema.model_dump()

# Usage
result: MetricResult[ComplexityMetricsSchema] = metric.get_result()
```

## Enums

```python
from enum import Enum, auto

class MetricsType(Enum):
    PRE_RUNTIME = "pre_runtime"
    POST_RUNTIME = "post_runtime"

class MetricsId(Enum):
    QISKIT = auto()
    COMPLEXITY = auto()
    CIRCUIT_PERFORMANCE = auto()
    BEHAVIORAL = auto()
    STRUCTURAL = auto()
    ELEMENT = auto()
    QUANTUM_SPECIFIC = auto()
```

## TypedDict (for dict-like structures)

```python
from typing import TypedDict, NotRequired

class MetricConfig(TypedDict):
    name: str
    enabled: bool
    options: NotRequired[dict[str, Any]]
```

## Type Narrowing

```python
from typing import TypeGuard

def is_valid_circuit(obj: Any) -> TypeGuard[QuantumCircuit]:
    return isinstance(obj, QuantumCircuit) and obj.num_qubits > 0

# After check, type is narrowed
if is_valid_circuit(circuit):
    circuit.h(0)  # IDE knows this is QuantumCircuit
```

## Best Practices

1. **Always annotate** function signatures and class attributes
2. **Use `from __future__ import annotations`** for forward references
3. **Prefer Pydantic** over dataclass for validated data
4. **Use Protocol** over ABC when structural typing suffices
5. **Run mypy** or pyright for static type checking
6. **Use `Final`** for constants: `MAX_QUBITS: Final = 127`
