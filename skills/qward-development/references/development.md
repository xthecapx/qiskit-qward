# Development Reference

Guidelines for contributing to QWARD.

## Setup

```bash
# Clone repository
git clone https://github.com/your-org/qiskit-qward.git
cd qiskit-qward

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

## Code Quality Standards

QWARD maintains strict standards:

| Check | Requirement |
|-------|-------------|
| Black | Code formatted |
| Pylint | 10.00/10 score |
| MyPy | No type errors |
| Tests | All passing |

## Pre-Push Verification

### Quick Script
```bash
./verify.sh
```

### Manual Commands

```bash
# 1. Formatting
black --check .
black .  # Fix if needed

# 2. Linting
pylint -rn qward tests

# 3. Type checking
mypy qward tests

# 4. Tests
python -m pytest tests/ -v

# All at once
black --check . && pylint -rn qward tests && mypy qward tests && python -m pytest tests/ -v
```

### Using Tox

```bash
# Full lint suite
tox -e lint

# Tests with coverage
tox -e coverage

# Specific Python version
tox -e py310  # or py311, py312
```

## Project Structure

```
qward/
├── __init__.py              # Main exports
├── scanner.py               # Scanner class
├── version.py               # Version info
├── metrics/
│   ├── __init__.py          # Metric exports
│   ├── base_metric.py       # MetricCalculator base
│   ├── types.py             # MetricsType, MetricsId
│   ├── defaults.py          # Default strategies
│   ├── qiskit_metrics.py
│   ├── complexity_metrics.py
│   ├── circuit_performance.py
│   ├── element_metrics.py
│   ├── structural_metrics.py
│   ├── behavioral_metrics.py
│   ├── quantum_specific_metrics.py
│   └── differential_success_rate.py
├── schemas/
│   ├── qiskit_metrics_schema.py
│   ├── complexity_metrics_schema.py
│   └── ...
├── visualization/
│   ├── __init__.py
│   ├── base.py              # VisualizationStrategy
│   ├── visualizer.py        # Unified Visualizer
│   ├── constants.py         # Metrics, Plots constants
│   └── *_visualizer.py      # Specific visualizers
├── algorithms/
│   ├── __init__.py
│   ├── executor.py          # QuantumCircuitExecutor
│   ├── noise_generator.py
│   ├── experiment.py        # BaseExperimentRunner
│   ├── grover.py
│   ├── qft.py
│   ├── phase_estimation.py
│   └── ...
├── utils/
│   └── ...
└── examples/
    └── ...
```

## Adding a New Metric

1. **Create schema** in `schemas/`:
```python
# schemas/my_metrics_schema.py
from pydantic import BaseModel

class MyMetricsSchema(BaseModel):
    metric_a: float
    metric_b: int

    def to_flat_dict(self):
        return self.model_dump()
```

2. **Create metric** in `metrics/`:
```python
# metrics/my_metrics.py
from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsType, MetricsId
from qward.schemas.my_metrics_schema import MyMetricsSchema

class MyMetrics(MetricCalculator):
    def _get_metric_type(self) -> MetricsType:
        return MetricsType.PRE_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        return MetricsId.QISKIT

    def is_ready(self) -> bool:
        return self.circuit is not None

    def get_metrics(self) -> MyMetricsSchema:
        return MyMetricsSchema(
            metric_a=1.0,
            metric_b=2
        )
```

3. **Export** in `metrics/__init__.py`:
```python
from qward.metrics.my_metrics import MyMetrics
__all__ = [..., "MyMetrics"]
```

4. **Add tests** in `tests/`:
```python
# tests/test_my_metrics.py
def test_my_metrics():
    circuit = QuantumCircuit(2)
    metric = MyMetrics(circuit)
    result = metric.get_metrics()
    assert isinstance(result.metric_a, float)
```

## Adding a New Visualizer

1. **Create visualizer** in `visualization/`:
```python
from qward.visualization.base import VisualizationStrategy, PlotMetadata, PlotType

class MyVisualizer(VisualizationStrategy):
    PLOT_REGISTRY = {
        "my_plot": PlotMetadata(
            name="my_plot",
            method_name="_plot_my_plot",
            description="My custom plot",
            plot_type=PlotType.BAR,
            filename="my_plot.png",
            dependencies=["metric_a"],
            category="custom"
        )
    }

    def get_available_plots(self):
        return list(self.PLOT_REGISTRY.keys())

    # ... implement methods
```

2. **Register** in unified Visualizer if needed

## Testing Guidelines

```python
# Test structure matches source
tests/
├── test_scanner.py
├── test_qiskit_metrics.py
├── test_complexity_metrics.py
└── ...

# Run specific test
pytest tests/test_scanner.py::TestScanner::test_scan -v

# Run with coverage
pytest tests/ --cov=qward --cov-report=html
```

## Documentation

- Docstrings: NumPy or Google style
- Type hints: Complete for all public APIs
- Examples: In docstrings and `examples/` folder

```python
def calculate_metric(circuit: QuantumCircuit, *, option: bool = True) -> float:
    """
    Calculate a custom metric.

    Args:
        circuit: The quantum circuit to analyze
        option: Enable optional feature

    Returns:
        The calculated metric value

    Raises:
        ValueError: If circuit is empty

    Examples:
        >>> circuit = QuantumCircuit(2)
        >>> result = calculate_metric(circuit)
    """
```

## Configuration Files

| File | Purpose |
|------|---------|
| `.pylintrc` | Pylint configuration |
| `mypy.ini` | MyPy settings |
| `pyproject.toml` | Package config, black settings |
| `tox.ini` | Tox environments |
| `requirements.qward.txt` | Runtime dependencies |

## Common Issues

### Pylint score < 10.00
- Check output for specific issues
- Examples folder is excluded (configured)
- Use appropriate disable comments for false positives

### MyPy errors
- Add missing type hints
- Use `# type: ignore` for legitimate edge cases

### Test failures
```bash
pytest tests/ -v -s  # Verbose with stdout
pytest tests/test_specific.py -v  # Single file
```
