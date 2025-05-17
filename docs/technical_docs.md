# Technical Documentation

This document provides detailed technical information about the Qward library's architecture, components, and advanced usage patterns. For a structural overview and class diagrams, please refer to `docs/architecture.md`.

## Core Architecture

Qward is designed around a `Scanner` that applies various `Metric` objects to Qiskit `QuantumCircuit`s and their execution results.

### 1. `Scanner` (`qward.Scanner`)

The `Scanner` is the central class for orchestrating analysis. 
-   **Initialization**: `Scanner(circuit: Optional[QuantumCircuit], job: Optional[Union[AerJob, QiskitJob]], result: Optional[qward.Result], metrics: Optional[list])`
    -   It takes an optional Qiskit `QuantumCircuit`.
    -   Optionally, a Qiskit `Job` (from Aer or a provider) and/or a `qward.Result` object (which wraps execution counts and metadata).
    -   Optionally, a list of metric *classes* or *instances* can be provided at initialization.
-   **Adding Metrics**: Metrics are added using `scanner.add_metric(metric_instance: Metric)`.
-   **Calculating Metrics**: `scanner.calculate_metrics() -> Dict[str, pd.DataFrame]`
    -   This method iterates through all added metrics, calls their `get_metrics()` method, and compiles the results into a dictionary. The keys are typically the metric class names (or a modified name for `SuccessRate` with multiple jobs), and values are pandas DataFrames containing the metric data.
    -   For `SuccessRate` with multiple jobs, two DataFrames are produced: `SuccessRate.individual_jobs` and `SuccessRate.aggregate`.

### 2. `Metric` System (`qward.metrics`)

Metrics are classes responsible for specific calculations or data extraction.

-   **Base Class**: `qward.metrics.base_metric.Metric`
    -   All metrics inherit from this abstract base class.
    -   Requires implementation of:
        -   `_get_metric_type() -> MetricsType`: Returns `MetricsType.PRE_RUNTIME` (if only circuit is needed) or `MetricsType.POST_RUNTIME` (if job/result data is needed).
        -   `_get_metric_id() -> MetricsId`: Returns a unique `MetricsId` enum value.
        -   `is_ready() -> bool`: Checks if the metric has sufficient data to be calculated.
        -   `get_metrics() -> Dict[str, Any]`: Performs the calculation and returns a dictionary of results.
    -   The constructor `Metric(circuit: QuantumCircuit)` stores the circuit and initializes `_metric_type` and `_id` by calling the abstract getter methods.

-   **Built-in Metrics**:
    -   **`QiskitMetrics` (`qward.metrics.qiskit_metrics.QiskitMetrics`)**
        -   Type: `PRE_RUNTIME`
        -   ID: `MetricsId.QISKIT`
        -   Extracts data directly from the `QuantumCircuit` object (e.g., depth, width, operations count, instruction details, basic scheduling info if available).
        -   `get_metrics()` returns a flattened dictionary of these properties.

    -   **`ComplexityMetrics` (`qward.metrics.complexity_metrics.ComplexityMetrics`)**
        -   Type: `PRE_RUNTIME`
        -   ID: `MetricsId.COMPLEXITY`
        -   Calculates a comprehensive set of complexity metrics based on D. Shami's "Character Complexity" paper, including gate-based, entanglement, standardized, advanced, and derived metrics.
        -   Also includes Quantum Volume estimation (`standard_quantum_volume`, `enhanced_quantum_volume`, and contributing factors).
        -   `get_metrics()` returns a dictionary where top-level keys correspond to these categories (e.g., "gate_based_metrics", "quantum_volume"), and values are sub-dictionaries of specific metrics.

    -   **`SuccessRate` (`qward.metrics.success_rate.SuccessRate`)**
        -   Type: `POST_RUNTIME`
        -   ID: `MetricsId.SUCCESS_RATE`
        -   Constructor: `SuccessRate(circuit: QuantumCircuit, job: Optional[Job]=None, jobs: Optional[List[Job]]=None, result: Optional[Dict]=None, success_criteria: Optional[Callable]=None)`
            -   Requires a `QuantumCircuit`.
            -   Needs execution data, provided via a single `job`, a list of `jobs`, or a `result` dictionary (though `job`/`jobs` are preferred for direct access to Qiskit's result objects).
            -   `success_criteria` is a callable `(bitstring: str) -> bool` that defines a successful outcome. Defaults to `bitstring == "0"`.
        -   `add_job(job_or_jobs)`: Allows adding more jobs after instantiation.
        -   `get_metrics()`: Calculates success rate, error rate, fidelity, total shots, successful shots. If multiple jobs are provided, it returns individual job stats and aggregate stats.

-   **Metric Types (`qward.metrics.types`)**
    -   `MetricsId(Enum)`: Defines unique IDs for metric types (e.g., `QISKIT`, `COMPLEXITY`, `SUCCESS_RATE`).
    -   `MetricsType(Enum)`: Defines when a metric is calculated (`PRE_RUNTIME`, `POST_RUNTIME`).

-   **Default Metrics (`qward.metrics.defaults`)**
    -   `get_default_metrics() -> List[Type[Metric]]`: Returns a list of default metric classes (`[QiskitMetrics, ComplexityMetrics, SuccessRate]`).

### 3. `Result` (`qward.Result`)

A utility class for encapsulating job execution data.
-   Constructor: `Result(job: Optional[Job]=None, counts: Optional[Dict]=None, metadata: Optional[Dict]=None)`.
-   Primarily stores Qiskit `Job` object, `counts` (measurement outcomes), and `metadata`.
-   Provides `save()` and `load()` methods for serialization to/from JSON.
-   `update_from_job()`: If a job is stored, this method (re-)populates counts and basic metadata from `job.result()`.

### 4. `QiskitRuntimeService` (`qward.runtime.qiskit_runtime.QiskitRuntimeService`)

Extends Qiskit's base `QiskitRuntimeService`.
-   Constructor: `QiskitRuntimeService(circuit: QuantumCircuit, backend: Union[Backend, str], **kwargs)`
    -   Takes a `QuantumCircuit` and a Qiskit `Backend` object or backend name string.
    -   Passes `**kwargs` to the base `QiskitRuntimeService` constructor (e.g., for channel, token).
-   Key Methods:
    -   `run()`: Executes the circuit using a `SamplerV2` on the specified backend.
    -   `run_and_watch()`: Runs the circuit and polls the job status until completion or failure, then returns a `qward.Result` object.
    -   `get_results()`: Retrieves results from the completed job and returns a `qward.Result`.

## Circuit Complexity Metrics (via `ComplexityMetrics`)

The `ComplexityMetrics` class, when used with the `Scanner`, provides a detailed breakdown of circuit complexity. The output DataFrame from `scanner.calculate_metrics()["ComplexityMetrics"]` will have columns corresponding to flattened metric names (e.g., `gate_based_metrics.gate_count`, `entanglement_metrics.entangling_gate_density`).

Refer to the `qward/metrics/complexity_metrics.py` source or its docstrings for a full list of all calculated metrics. The categories include:
-   Gate-based Metrics
-   Entanglement Metrics
-   Standardized Metrics
-   Advanced Metrics
-   Derived Metrics

Example access:
```python
# results = scanner.calculate_metrics()
# complexity_df = results.get("ComplexityMetrics")
# if complexity_df is not None:
#     gate_count = complexity_df['gate_based_metrics.gate_count'].iloc[0]
#     t_count = complexity_df['gate_based_metrics.t_count'].iloc[0]
#     # ... and so on
```

## Quantum Volume Estimation (via `ComplexityMetrics`)

The `ComplexityMetrics` class also performs Quantum Volume (QV) estimation based on the circuit's structure.

-   **Standard Quantum Volume**: `quantum_volume.standard_quantum_volume` (2^effective_depth)
-   **Enhanced Quantum Volume**: `quantum_volume.enhanced_quantum_volume` (adjusts standard QV by factors like square ratio, density, multi-qubit ratio, connectivity).
-   **Effective Depth**: `quantum_volume.effective_depth`
-   **Contributing Factors**: `quantum_volume.factors.square_ratio`, etc.

Example access from `ComplexityMetrics` DataFrame:
```python
# enhanced_qv = complexity_df['quantum_volume.enhanced_quantum_volume'].iloc[0]
```

## Technical Guidelines for Custom Metrics

When creating a new metric by inheriting from `qward.metrics.base_metric.Metric`:
1.  **Implement Abstract Methods**: `_get_metric_type`, `_get_metric_id`, `is_ready`, `get_metrics`.
2.  **Metric ID**: If your metric is conceptually new, consider if `MetricsId` enum in `qward.metrics.types` needs to be extended. For purely external or highly specific custom metrics, you might need a strategy if modifying the core enum is not desired (though the base class expects a `MetricsId` enum member).
3.  **Data Requirements**: Clearly define if your metric is `PRE_RUNTIME` (only needs circuit) or `POST_RUNTIME` (needs job/results). If `POST_RUNTIME`, your `__init__` should accept job(s) or result data, and `is_ready` should check for their presence.
4.  **Return Format**: `get_metrics()` must return a dictionary where keys are string metric names and values are the calculated metric values.

## API Flow and Usage Patterns

### Pattern 1: Basic Circuit Analysis (No Execution Results)
```python
from qiskit import QuantumCircuit
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics

qc = QuantumCircuit(2); qc.h(0); qc.cx(0,1)

scanner = Scanner(circuit=qc)
scanner.add_metric(QiskitMetrics(qc))
scanner.add_metric(ComplexityMetrics(qc))

results = scanner.calculate_metrics()
# print(results["QiskitMetrics"])
# print(results["ComplexityMetrics"])
```

### Pattern 2: Analysis with Execution Results
```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner, Result
from qward.metrics import QiskitMetrics, ComplexityMetrics, SuccessRate

qc = QuantumCircuit(2,2); qc.h(0); qc.cx(0,1); qc.measure_all()

sim = AerSimulator()
job = sim.run(qc).result()
counts = job.get_counts()

# For SuccessRate, we need the Qiskit Job object (not just its result)
qiskit_job_obj = sim.run(qc) # Re-run to get a job object to pass to SuccessRate

qward_res = Result(job=qiskit_job_obj, counts=counts)

scanner = Scanner(circuit=qc, job=qiskit_job_obj, result=qward_res)
scanner.add_metric(QiskitMetrics(qc))
scanner.add_metric(ComplexityMetrics(qc))

def criteria(s): return s == '00' or s == '11' # GHZ state
scanner.add_metric(SuccessRate(circuit=qc, job=qiskit_job_obj, success_criteria=criteria))

results = scanner.calculate_metrics()
# print(results["SuccessRate.aggregate"])
```

### Pattern 3: Using `QiskitRuntimeService`
```python
# from qiskit import QuantumCircuit
# from qward.runtime import QiskitRuntimeService as QwardRTService # alias for clarity
# from qward import Scanner
# from qward.metrics import SuccessRate

# qc = QuantumCircuit(2,2); qc.h(0); qc.cx(0,1); qc.measure_all()

# Note: Ensure IBMQ credentials are set up if not using a simulator backend
# qward_runtime = QwardRTService(circuit=qc, backend_name='ibmq_qasm_simulator') 
# result_from_runtime = qward_runtime.run_and_watch() # This is a qward.Result object

# scanner = Scanner(circuit=qc, job=result_from_runtime.job, result=result_from_runtime)
# def criteria(s): return s == '00' or s == '11'
# scanner.add_metric(SuccessRate(circuit=qc, job=result_from_runtime.job, success_criteria=criteria))
# metrics = scanner.calculate_metrics()
# print(metrics["SuccessRate.aggregate"])
```

## Conclusion

Qward provides a structured and extensible approach to quantum circuit analysis. By understanding the `Scanner`, the `Metric` system, and associated helper classes like `Result` and `QiskitRuntimeService`, developers can gain significant insights into their quantum circuits and execution outcomes.
