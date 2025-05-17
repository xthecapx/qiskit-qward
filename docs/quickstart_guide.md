# Quickstart Guide

Qward is a Python library for analyzing quantum circuits and their execution quality on quantum processing units (QPUs) or simulators. This guide will help you quickly get started.

## Installation

### Option 1: Local Installation

```bash
# Clone the repository
# git clone https://github.com/your-org/qiskit-qward.git # Replace with actual URL
# cd qiskit-qward

# Install in development mode (assuming you are in the project root)
pip install -e .

# Set up IBM Quantum credentials (optional, for running on IBM hardware)
# cp .env.example .env
# Edit .env with your IBM Quantum token
```

### Option 2: Using Docker

(Assuming Docker setup is configured in the project)
```bash
# Clone the repository
# git clone https://github.com/your-org/qiskit-qward.git # Replace with actual URL
# cd qiskit-qward

# Copy and edit .env file (optional, for IBM hardware)
# cp .env.example .env
# Edit .env with your IBM Quantum token

# Start Docker container with Jupyter Lab (example script name)
# chmod +x start.sh
# ./start.sh
```

## Usage

Qward revolves around the `Scanner` class, which uses various `Metric` objects to analyze Qiskit `QuantumCircuit` objects and their execution results.

### Core Workflow

1.  **Create/Load a `QuantumCircuit`**: Use Qiskit to define your circuit.
2.  **(Optional) Execute the Circuit**: Run your circuit on a simulator or quantum hardware to get a Qiskit `Job` and its `Result` (containing counts).
3.  **Instantiate `qward.Scanner`**: Provide the circuit, and optionally the Qiskit `Job` and `qward.Result` (which wraps Qiskit's job result/counts).
4.  **Add `Metric` Objects**: Instantiate and add desired metric classes from `qward.metrics` (e.g., `QiskitMetrics`, `ComplexityMetrics`, `SuccessRate`) to the scanner.
5.  **Calculate Metrics**: Call `scanner.calculate_metrics()`.
6.  **Interpret Results**: The result is a dictionary of pandas DataFrames, one for each metric type.

### Example: Analyzing a Simple Circuit

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner, Result # QWARD classes
from qward.metrics import QiskitMetrics, ComplexityMetrics, SuccessRate # QWARD metrics
from qward.examples.utils import get_display # For pretty printing in notebooks

display = get_display()

# 1. Create a Quantum Circuit (e.g., 2-qubit Bell state)
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0,1], [0,1])

print("Quantum Circuit:")
display(qc.draw(output='mpl'))

# 2. Execute the Circuit (e.g., using AerSimulator)
simulator = AerSimulator()
job = simulator.run(qc, shots=2048).result() # Get Qiskit's job result directly
counts = job.get_counts(qc)

# It's good practice to also have the Qiskit Job object if available
# For this example, we use AerSimulator().run().result(), so the job object itself is ephemeral.
# If using QiskitRuntimeService, you'd have a job object.
qiskit_job_instance = simulator.run(qc, shots=2048) # Re-run to get a job object for demonstration

# 3. Instantiate QWARD Scanner and Result
# QWARD's Result can wrap Qiskit's counts and job metadata
qward_result = Result(job=qiskit_job_instance, counts=counts, metadata=job.to_dict()) 

scanner = Scanner(circuit=qc, job=qiskit_job_instance, result=qward_result)

# 4. Add Metrics
scanner.add_metric(QiskitMetrics(circuit=qc))
scanner.add_metric(ComplexityMetrics(circuit=qc))

# For SuccessRate, define what a "successful" measurement is
def success_if_00(bitstring):
    return bitstring == "00"

# SuccessRate needs a job to get counts from, or you can pass counts directly if job not available at init
# Here, we pass the qiskit_job_instance that the qward_result is also based on.
scanner.add_metric(SuccessRate(circuit=qc, job=qiskit_job_instance, success_criteria=success_if_00))

# 5. Calculate Metrics
all_metric_data = scanner.calculate_metrics()

# 6. Interpret Results
print("\n--- All Calculated Metrics ---")
for metric_name, df in all_metric_data.items():
    print(f"\nMetric: {metric_name}")
    display(df)

# Example: Accessing specific data from ComplexityMetrics output
if "ComplexityMetrics" in all_metric_data:
    complexity_df = all_metric_data["ComplexityMetrics"]
    print("\nSelected Complexity Data:")
    print(f"  Gate Count: {complexity_df['gate_based_metrics.gate_count'].iloc[0]}")
    print(f"  Circuit Depth: {complexity_df['gate_based_metrics.circuit_depth'].iloc[0]}")
    print(f"  Enhanced QV Estimate: {complexity_df['quantum_volume.enhanced_quantum_volume'].iloc[0]}")

# Example: Accessing specific data from SuccessRate output
if "SuccessRate.aggregate" in all_metric_data: # For single job, aggregate is similar to individual
    success_df = all_metric_data["SuccessRate.aggregate"]
    print("\nSuccess Rate Data (for '00'):")
    print(f"  Mean Success Rate: {success_df['mean_success_rate'].iloc[0]:.2%}")
    print(f"  Total Shots: {success_df['total_trials'].iloc[0]}")
```

### Creating Custom Metrics

To create your own metric, inherit from `qward.metrics.base_metric.Metric` and implement the required abstract methods (`_get_metric_type`, `_get_metric_id`, `is_ready`, `get_metrics`).

```python
from qward.metrics.base_metric import Metric
from qward.metrics.types import MetricsType, MetricsId
from qiskit import QuantumCircuit

class MySimpleCustomMetric(Metric):
    def __init__(self, circuit: QuantumCircuit):
        super().__init__(circuit)

    def _get_metric_type(self) -> MetricsType:
        return MetricsType.PRE_RUNTIME # Only needs the circuit

    def _get_metric_id(self) -> MetricsId:
        # Ideally, add a new ID to MetricsId enum. For now, reuse for example.
        return MetricsId.QISKIT # Placeholder

    def is_ready(self) -> bool:
        return self.circuit is not None

    def get_metrics(self) -> dict:
        return {"custom_depth_plus_width": self.circuit.depth() + self.circuit.width()}

# Usage:
# custom_metric = MySimpleCustomMetric(qc)
# scanner.add_metric(custom_metric)
# results = scanner.calculate_metrics() 
# print(results['MySimpleCustomMetric'])
```

## Key Metrics Provided

Qward, through its built-in metric classes, offers insights into:

### 1. Circuit Structure (`QiskitMetrics`)
   - Basic properties like depth, width, number of qubits/clbits, operations count.

### 2. Circuit Complexity (`ComplexityMetrics`)
   - **Gate-based metrics**: Gate counts, T-count, CNOT count, etc.
   - **Entanglement metrics**: Entangling gate density, entangling width.
   - **Standardized metrics**: Circuit volume, gate density.
   - **Advanced & Derived metrics**: Parallelism, efficiency, weighted complexity.
   - All these are reported as columns in the DataFrame produced by `ComplexityMetrics`.

### 3. Quantum Volume Estimation (within `ComplexityMetrics`)
   - `ComplexityMetrics` also estimates Quantum Volume based on the circuit's structure.
   - Provides "standard" (2^eff_depth) and "enhanced" (factoring in density, squareness) QV estimates.
   - Reported as columns like `quantum_volume.standard_quantum_volume` in its DataFrame.

### 4. Execution Success (`SuccessRate`)
   - Calculates success rate, error rate, fidelity based on execution counts and your defined success criteria.
   - Handles single or multiple jobs (providing aggregate and per-job statistics).

## Using Jupyter Notebooks

The easiest way to work with Qward is often using Jupyter notebooks. If you use the Docker setup (if available and configured with `./start.sh`), or a local Python environment with Jupyter installed, you can explore the examples in `qward/examples/` such as `run_on_aer.ipynb`.

## IBM Quantum Execution

To run on real quantum hardware via IBM Quantum, you need an IBM Quantum account and your API token.

1.  Register at [IBM Quantum](https://quantum.ibm.com/).
2.  Get your API token from your account settings.
3.  You can set environment variables `IBM_QUANTUM_TOKEN` and `IBM_QUANTUM_CHANNEL` (e.g. `ibm_quantum`), or manage credentials as per Qiskit's documentation for `QiskitRuntimeService`.

Qward provides `qward.runtime.qiskit_runtime.QiskitRuntimeService` which can be used to run circuits on IBM backends. It offers a `run_and_watch()` method for convenience.

```python
# from qward.runtime import QiskitRuntimeService
# from qiskit import QuantumCircuit

# qc = QuantumCircuit(2)
# qc.h(0)
# qc.cx(0,1)

# For IBM Quantum:
# service = QiskitRuntimeService(channel='ibm_quantum') # Ensure token is available
# runtime_service_qward = QiskitRuntimeServiceQward( # This class name is from docs/architecture.md, ensure correct import
#     circuit=qc, 
#     backend_name='ibmq_qasm_simulator', # or a real backend name
#     service=service # Pass the base service
# )
# qward_result = runtime_service_qward.run_and_watch()
# print(qward_result.counts)

# Note: The above QiskitRuntimeServiceQward usage needs to be aligned with actual class name and constructor.
# The example from architecture.md is:
# runtime_service = QiskitRuntimeService(circuit=circuit, backend="ibmq_qasm_simulator")
# result = runtime_service.run_and_watch()
# This implies QiskitRuntimeService in qward.runtime directly takes circuit and backend.
```

## Next Steps

-   Explore the examples in the `qward/examples/` directory, especially `aer.py` and `run_on_aer.ipynb`.
-   Refer to the [Beginner's Guide](beginners_guide.md) for a more narrative introduction.
-   Consult the [API Documentation](apidocs/index.rst) (once generated/updated) for detailed class and method descriptions.
-   Review `docs/architecture.md` for an overview of the library structure.
