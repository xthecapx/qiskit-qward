# Custom Metrics Reference

Create custom metric strategies by extending QWARD's base classes.

## Basic Custom Metric

```python
from qiskit import QuantumCircuit
from pydantic import BaseModel
from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsType, MetricsId

# 1. Define your schema
class MyMetricsSchema(BaseModel):
    custom_depth_score: float
    gate_efficiency: float
    circuit_signature: str

    def to_flat_dict(self):
        return self.model_dump()

# 2. Implement the metric calculator
class MyCustomMetric(MetricCalculator):
    def __init__(self, circuit: QuantumCircuit):
        super().__init__(circuit)

    def _get_metric_type(self) -> MetricsType:
        return MetricsType.PRE_RUNTIME  # or POST_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        return MetricsId.QISKIT  # Or define a new ID

    def is_ready(self) -> bool:
        return self.circuit is not None

    def get_metrics(self) -> MyMetricsSchema:
        depth = self.circuit.depth()
        num_qubits = self.circuit.num_qubits
        size = self.circuit.size()

        return MyMetricsSchema(
            custom_depth_score=depth / max(num_qubits, 1),
            gate_efficiency=size / max(depth * num_qubits, 1),
            circuit_signature=f"{num_qubits}q_{depth}d_{size}g"
        )

# 3. Use with Scanner
from qward import Scanner

scanner = Scanner(circuit)
scanner.add_strategy(MyCustomMetric(circuit))
results = scanner.calculate_metrics()
print(results["MyCustomMetric"])
```

## Schema with Validation

```python
from pydantic import BaseModel, Field, field_validator

class AdvancedMetricsSchema(BaseModel):
    efficiency: float = Field(ge=0.0, le=1.0, description="Circuit efficiency (0-1)")
    complexity_score: int = Field(ge=0, description="Non-negative complexity")
    qubit_utilization: float

    @field_validator('qubit_utilization')
    @classmethod
    def validate_utilization(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Qubit utilization must be between 0 and 1')
        return v

    def to_flat_dict(self):
        return self.model_dump()
```

## Nested Schema Structure

```python
from pydantic import BaseModel

class GateAnalysisSchema(BaseModel):
    total_gates: int
    single_qubit_gates: int
    two_qubit_gates: int
    gate_ratio: float

class DepthAnalysisSchema(BaseModel):
    raw_depth: int
    normalized_depth: float
    critical_path: int

class ComprehensiveMetricsSchema(BaseModel):
    gate_analysis: GateAnalysisSchema
    depth_analysis: DepthAnalysisSchema
    overall_score: float

    def to_flat_dict(self):
        """Flatten nested structure for DataFrame compatibility."""
        result = {}
        for field_name, field_value in self:
            if hasattr(field_value, 'model_dump'):
                for sub_key, sub_value in field_value.model_dump().items():
                    result[f"{field_name}.{sub_key}"] = sub_value
            else:
                result[field_name] = field_value
        return result

class ComprehensiveMetric(MetricCalculator):
    def get_metrics(self) -> ComprehensiveMetricsSchema:
        circuit = self.circuit

        # Count gates by type
        single_q = sum(1 for inst in circuit.data if len(inst.qubits) == 1)
        two_q = sum(1 for inst in circuit.data if len(inst.qubits) == 2)
        total = circuit.size()

        return ComprehensiveMetricsSchema(
            gate_analysis=GateAnalysisSchema(
                total_gates=total,
                single_qubit_gates=single_q,
                two_qubit_gates=two_q,
                gate_ratio=two_q / max(total, 1)
            ),
            depth_analysis=DepthAnalysisSchema(
                raw_depth=circuit.depth(),
                normalized_depth=circuit.depth() / max(circuit.num_qubits, 1),
                critical_path=circuit.depth()
            ),
            overall_score=0.5  # Your calculation
        )
```

## Post-Runtime Metric

```python
from typing import Optional, Callable
from qiskit.providers.job import Job

class PostRuntimeMetricsSchema(BaseModel):
    success_rate: float
    error_rate: float
    fidelity_estimate: float

    def to_flat_dict(self):
        return self.model_dump()

class MyPostRuntimeMetric(MetricCalculator):
    def __init__(
        self,
        circuit: QuantumCircuit,
        job: Optional[Job] = None,
        success_criteria: Optional[Callable[[str], bool]] = None
    ):
        super().__init__(circuit)
        self._job = job
        self._success_criteria = success_criteria or self._default_criteria

    def _default_criteria(self, result: str) -> bool:
        return True  # Override with your logic

    def _get_metric_type(self) -> MetricsType:
        return MetricsType.POST_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        return MetricsId.CIRCUIT_PERFORMANCE

    def is_ready(self) -> bool:
        return self.circuit is not None and self._job is not None

    def get_metrics(self) -> PostRuntimeMetricsSchema:
        if not self.is_ready():
            raise ValueError("Job required for post-runtime metrics")

        result = self._job.result()
        counts = result.get_counts()
        total = sum(counts.values())

        successful = sum(
            count for bitstring, count in counts.items()
            if self._success_criteria(bitstring)
        )

        success_rate = successful / total

        return PostRuntimeMetricsSchema(
            success_rate=success_rate,
            error_rate=1.0 - success_rate,
            fidelity_estimate=success_rate ** 0.5  # Simplified
        )
```

## Registering with Visualization

```python
from qward.visualization.base import VisualizationStrategy, PlotMetadata, PlotType
import matplotlib.pyplot as plt

class MyMetricVisualizer(VisualizationStrategy):
    PLOT_REGISTRY = {
        "custom_overview": PlotMetadata(
            name="custom_overview",
            method_name="_plot_custom_overview",
            description="Custom metric overview visualization",
            plot_type=PlotType.BAR,
            filename="custom_overview.png",
            dependencies=["custom_depth_score", "gate_efficiency"],
            category="custom"
        )
    }

    def get_available_plots(self):
        return list(self.PLOT_REGISTRY.keys())

    def get_plot_metadata(self, plot_name):
        return self.PLOT_REGISTRY.get(plot_name)

    def generate_plot(self, plot_name, save=False, show=True):
        metadata = self.PLOT_REGISTRY[plot_name]
        method = getattr(self, metadata.method_name)
        return method(save=save, show=show)

    def _plot_custom_overview(self, save=False, show=True):
        fig, ax = plt.subplots(figsize=self.config.figsize)
        # Your plotting logic
        if save:
            self.save_plot(fig, "custom_overview.png")
        if show:
            self.show_plot(fig)
        return fig

    def create_dashboard(self, save=False, show=True):
        return self.generate_all_plots(save=save, show=show)
```

## Best Practices

1. **Always use Pydantic schemas** - Provides validation and IDE support
2. **Implement `to_flat_dict()`** - Required for Scanner DataFrame conversion
3. **Use descriptive field names** - Makes DataFrame columns clear
4. **Add Field constraints** - Catch errors early with validation
5. **Document your metrics** - Use Field descriptions
6. **Test with Scanner** - Ensure integration works
7. **Consider metric type** - PRE_RUNTIME vs POST_RUNTIME
