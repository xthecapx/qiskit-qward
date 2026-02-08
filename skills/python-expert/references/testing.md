# Testing Reference (pytest)

## Project Structure

```
tests/
├── conftest.py          # Shared fixtures
├── test_scanner.py
├── test_metrics/
│   ├── test_qiskit_metrics.py
│   ├── test_complexity_metrics.py
│   └── test_circuit_performance.py
├── test_schemas/
│   └── test_schema_validation.py
└── test_visualization/
```

## Fixtures

```python
# conftest.py
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

@pytest.fixture
def measured_bell(bell_circuit):
    bell_circuit.measure_all()
    return bell_circuit

@pytest.fixture
def scanner(bell_circuit):
    from qward import Scanner
    return Scanner(circuit=bell_circuit)
```

## Parametrized Tests

```python
@pytest.mark.parametrize("num_qubits", [2, 3, 5, 10])
def test_qiskit_metrics_any_size(num_qubits):
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    metrics = QiskitMetrics(qc)
    result = metrics.get_metrics()
    assert result.basic_metrics.num_qubits == num_qubits

@pytest.mark.parametrize("metric_class,expected_type", [
    (QiskitMetrics, MetricsType.PRE_RUNTIME),
    (ComplexityMetrics, MetricsType.PRE_RUNTIME),
    (CircuitPerformanceMetrics, MetricsType.POST_RUNTIME),
])
def test_metric_types(bell_circuit, metric_class, expected_type):
    metric = metric_class(bell_circuit)
    assert metric.metric_type == expected_type
```

## Testing Schemas

```python
def test_schema_validation():
    from qward.schemas.complexity_metrics_schema import ComplexityMetricsSchema

    # Valid data
    data = {"gate_count": 5, "depth": 3, "gate_density": 0.6}
    schema = ComplexityMetricsSchema(**data)
    assert schema.gate_count == 5

    # Invalid data should raise
    with pytest.raises(ValidationError):
        ComplexityMetricsSchema(gate_count=-1, depth=3, gate_density=0.6)

def test_to_flat_dict(bell_circuit):
    metrics = ComplexityMetrics(bell_circuit)
    result = metrics.get_metrics()
    flat = result.to_flat_dict()

    assert isinstance(flat, dict)
    assert all(isinstance(v, (int, float, str, bool, type(None))) for v in flat.values())
```

## Mocking

```python
from unittest.mock import Mock, patch, MagicMock

def test_scanner_with_mock_metric(bell_circuit):
    mock_metric = Mock(spec=MetricCalculator)
    mock_metric.is_ready.return_value = True
    mock_metric.get_metrics.return_value = Mock(to_flat_dict=lambda: {"value": 42})
    mock_metric.name = "mock"

    scanner = Scanner(circuit=bell_circuit)
    scanner.add_strategy(mock_metric)
    results = scanner.calculate_metrics()

    mock_metric.get_metrics.assert_called_once()
    assert "mock" in results

@patch('qward.metrics.circuit_performance.execute')
def test_performance_with_mock_job(mock_execute, bell_circuit):
    mock_execute.return_value = {"00": 500, "11": 524}
    # test circuit performance with mocked execution
```

## Markers

```python
@pytest.mark.slow
def test_large_circuit_analysis():
    """Test with 20+ qubit circuit (slow)."""
    ...

@pytest.mark.integration
def test_full_scanner_pipeline():
    """End-to-end test with all metrics."""
    ...

# Run specific markers:
# pytest -m "not slow"
# pytest -m integration
```

## Approximate Comparisons

```python
def test_metric_values(bell_circuit):
    metrics = ComplexityMetrics(bell_circuit)
    result = metrics.get_metrics()

    # Use pytest.approx for floating point
    assert result.standardized.gate_density == pytest.approx(0.75, abs=0.01)
    assert result.advanced.parallelism == pytest.approx(0.5, rel=0.1)
```

## Exception Testing

```python
def test_not_ready_raises():
    qc = QuantumCircuit(0)  # empty circuit
    metric = QiskitMetrics(qc)

    with pytest.raises(MetricNotReadyError, match="requires a valid circuit"):
        metric.get_metrics()

def test_invalid_schema():
    with pytest.raises(ValidationError) as exc_info:
        ComplexityMetricsSchema(gate_count=-1)
    assert "gate_count" in str(exc_info.value)
```

## Coverage

```bash
# Run with coverage
pytest --cov=qward --cov-report=html --cov-report=term-missing

# Minimum coverage threshold
pytest --cov=qward --cov-fail-under=80
```

## Running Tests

```bash
# All tests
pytest

# Verbose
pytest -v

# Specific file
pytest tests/test_scanner.py

# Specific test
pytest tests/test_scanner.py::test_calculate_metrics

# Stop on first failure
pytest -x

# Show print output
pytest -s

# Parallel (install pytest-xdist)
pytest -n auto
```
