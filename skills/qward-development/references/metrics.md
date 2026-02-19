# Metrics Reference

QWARD provides 8 metric types, all following the Strategy pattern with Pydantic schema validation.

## Common Interface

All metrics inherit from `MetricCalculator`:

```python
from qward.metrics import MetricCalculator

class MyMetric(MetricCalculator):
    def get_metrics(self) -> BaseModel:
        """Returns validated Pydantic schema object"""
        pass

    def is_ready(self) -> bool:
        """Check if metric can be calculated"""
        pass
```

## QiskitMetrics

Extracts Qiskit-native circuit properties.

```python
from qward.metrics import QiskitMetrics

qiskit = QiskitMetrics(circuit)
metrics = qiskit.get_metrics()  # QiskitMetricsSchema

# Basic metrics
metrics.basic_metrics.depth          # int
metrics.basic_metrics.width          # int
metrics.basic_metrics.size           # int (total gates)
metrics.basic_metrics.num_qubits     # int
metrics.basic_metrics.num_clbits     # int

# Instruction metrics
metrics.instruction_metrics.num_connected_components  # int
metrics.instruction_metrics.num_nonlocal_gates       # int
metrics.instruction_metrics.num_tensor_factors       # int

# Scheduling metrics (if available)
metrics.scheduling_metrics.depth      # Optional[int]
metrics.scheduling_metrics.duration   # Optional[float]
```

## ComplexityMetrics

Comprehensive circuit complexity analysis.

```python
from qward.metrics import ComplexityMetrics

complexity = ComplexityMetrics(circuit)
metrics = complexity.get_metrics()  # ComplexityMetricsSchema

# Gate-based metrics
metrics.gate_based_metrics.gate_count        # int
metrics.gate_based_metrics.depth             # int
metrics.gate_based_metrics.t_count           # int
metrics.gate_based_metrics.cnot_count        # int
metrics.gate_based_metrics.single_qubit_gates  # int
metrics.gate_based_metrics.two_qubit_gates     # int

# Entanglement metrics
metrics.entanglement_metrics.entangling_gate_count  # int
metrics.entanglement_metrics.entanglement_ratio     # float

# Standardized metrics
metrics.standardized_metrics.gate_density    # float
metrics.standardized_metrics.circuit_volume  # int

# Advanced metrics
metrics.advanced_metrics.parallelism_factor      # float
metrics.advanced_metrics.parallelism_efficiency  # float
metrics.advanced_metrics.circuit_efficiency      # float
metrics.advanced_metrics.critical_path_length    # int

# Derived metrics
metrics.derived_metrics.weighted_complexity      # float
metrics.derived_metrics.loschmidt_echo_bound    # float
```

## CircuitPerformanceMetrics

Post-runtime execution performance analysis.

```python
from qward.metrics import CircuitPerformanceMetrics

# With custom success criteria
def bell_success(result: str) -> bool:
    return result.replace(" ", "") in ["00", "11"]

perf = CircuitPerformanceMetrics(
    circuit=circuit,
    job=job,
    success_criteria=bell_success
)
metrics = perf.get_metrics()  # CircuitPerformanceSchema

# Success metrics
metrics.success_metrics.success_rate       # float (0.0-1.0)
metrics.success_metrics.error_rate         # float (validated: 1 - success_rate)
metrics.success_metrics.successful_shots   # int
metrics.success_metrics.failed_shots       # int
metrics.success_metrics.total_shots        # int

# Statistical metrics
metrics.statistical_metrics.entropy        # float
metrics.statistical_metrics.uniformity     # float
metrics.statistical_metrics.mode_count     # int

# For multiple jobs (aggregate)
metrics.success_metrics.mean_success_rate  # float
metrics.success_metrics.std_success_rate   # float
metrics.success_metrics.total_trials       # int
```

### Multiple Jobs

```python
perf = CircuitPerformanceMetrics(circuit=circuit, success_criteria=bell_success)
perf.add_job(job1)
perf.add_job(job2)
perf.add_job(job3)

# Scanner produces two DataFrames:
# "CircuitPerformance.individual_jobs" - per-job metrics
# "CircuitPerformance.aggregate" - mean, std, min, max across jobs
```

## ElementMetrics

Fundamental circuit building blocks analysis.

```python
from qward.metrics import ElementMetrics

element = ElementMetrics(circuit)
metrics = element.get_metrics()  # ElementMetricsSchema

# Gate usage analysis
metrics.gate_metrics.gate_counts           # Dict[str, int]
metrics.gate_metrics.unique_gates          # int
metrics.gate_metrics.total_gates           # int

# Operator analysis
metrics.operator_metrics.operator_frequencies  # Dict[str, int]
metrics.operator_metrics.parameterized_count   # int

# Operand analysis
metrics.operand_metrics.qubit_usage           # Dict[int, int]
metrics.operand_metrics.most_used_qubit       # int
```

## StructuralMetrics

Circuit architecture and topology analysis.

```python
from qward.metrics import StructuralMetrics

structural = StructuralMetrics(circuit)
metrics = structural.get_metrics()  # StructuralMetricsSchema

# Dimension metrics
metrics.dimension_metrics.depth          # int
metrics.dimension_metrics.width          # int
metrics.dimension_metrics.aspect_ratio   # float

# Layer analysis
metrics.layer_metrics.layer_count        # int
metrics.layer_metrics.avg_layer_size     # float
metrics.layer_metrics.max_layer_size     # int

# Connectivity metrics
metrics.connectivity_metrics.edge_count           # int
metrics.connectivity_metrics.avg_degree           # float
metrics.connectivity_metrics.connected_components # int
```

## BehavioralMetrics

Circuit execution behavior analysis.

```python
from qward.metrics import BehavioralMetrics

behavioral = BehavioralMetrics(circuit)
metrics = behavioral.get_metrics()  # BehavioralMetricsSchema

# State evolution metrics
metrics.evolution_metrics.state_changes      # int
metrics.evolution_metrics.superposition_ops  # int

# Interference metrics
metrics.interference_metrics.interference_points  # int
metrics.interference_metrics.destructive_ratio    # float

# Probability metrics
metrics.probability_metrics.output_distribution  # Dict[str, float]
```

## QuantumSpecificMetrics

Uniquely quantum properties analysis.

```python
from qward.metrics import QuantumSpecificMetrics

quantum = QuantumSpecificMetrics(circuit)
metrics = quantum.get_metrics()  # QuantumSpecificMetricsSchema

# Entanglement metrics
metrics.entanglement_metrics.max_entanglement_depth  # int
metrics.entanglement_metrics.entangling_gate_ratio   # float

# Non-classicality metrics
metrics.nonclassicality_metrics.non_clifford_count   # int
metrics.nonclassicality_metrics.magic_state_count    # int

# Resource metrics
metrics.resource_metrics.t_gate_cost       # int
metrics.resource_metrics.clifford_cost     # int
```

## Differential Success Rate (DSR)

Compare ideal vs noisy performance.

```python
from qward.metrics import compute_dsr, compute_dsr_percent, compute_dsr_with_flags

# Basic DSR
dsr = compute_dsr(ideal_success_rate=0.95, noisy_success_rate=0.72)
# Returns: 0.23

# Percentage DSR
dsr_pct = compute_dsr_percent(ideal=0.95, noisy=0.72)
# Returns: 24.21 (percentage)

# DSR with flags
result = compute_dsr_with_flags(ideal=0.95, noisy=0.72)
# Returns: DSRResult with severity classification
```

## Schema Validation

All metrics use Pydantic schemas providing:

```python
# Type safety with IDE autocomplete
metrics.basic_metrics.depth  # IDE knows this is int

# Automatic validation
# Raises ValidationError if constraints violated

# JSON schema generation
from qward.schemas import ComplexityMetricsSchema
schema = ComplexityMetricsSchema.model_json_schema()

# Flat dict conversion for DataFrames
flat = metrics.to_flat_dict()
# {"basic_metrics.depth": 5, "basic_metrics.width": 2, ...}
```

## Getting Default Strategies

```python
from qward.metrics import get_all_pre_runtime_strategies, get_default_strategies

# All pre-runtime metrics (no job required)
pre_runtime = get_all_pre_runtime_strategies()
# [QiskitMetrics, ComplexityMetrics, ElementMetrics, ...]

# Default set
defaults = get_default_strategies()
```
