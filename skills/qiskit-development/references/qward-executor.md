# QWARD Executor: Simulate, IBM QPU, and Rigetti (qBraid + AWS Braket)

The QWARD project supports multiple execution backends for quantum circuits.
The `QuantumCircuitExecutor` in `qward.algorithms.executor` provides a unified interface
for local simulators, IBM Quantum hardware, and Rigetti devices via qBraid. Additionally,
circuits can be run directly on AWS Braket using `qiskit-braket-provider` for native
access to Rigetti Ankaa-3 and other AWS-hosted QPUs.

## Installation

```bash
# Core (always needed)
pip install qiskit qiskit-aer

# For IBM Quantum hardware
pip install qiskit-ibm-runtime

# For Rigetti via qBraid (Method A - recommended)
pip install qbraid

# For Rigetti via AWS Braket directly (Method B)
pip install qiskit-braket-provider
# Also requires: pip install amazon-braket-sdk boto3
# And AWS credentials configured (aws configure)
```

## Quick Start

```python
from qiskit import QuantumCircuit
from qward.algorithms import QuantumCircuitExecutor

executor = QuantumCircuitExecutor(shots=1024)

# Build circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Run locally
result = executor.simulate(qc)
print(result["counts"])
print(result["qward_metrics"])  # Automatic QWARD metrics
```

---

## Pattern 1: Local Aer Simulation

### Ideal simulation (no noise)

```python
executor = QuantumCircuitExecutor(shots=1024)
result = executor.simulate(circuit)

# Result contains:
# - result["counts"]: measurement counts dict
# - result["qward_metrics"]: dict of DataFrames (QiskitMetrics, ComplexityMetrics, etc.)
```

### Simulation with noise models

```python
# Built-in noise types
result = executor.simulate(circuit, noise_model="depolarizing", noise_level=0.05)
result = executor.simulate(circuit, noise_model="pauli", noise_level=0.03)
result = executor.simulate(circuit, noise_model="mixed", noise_level=0.05)
result = executor.simulate(circuit, noise_model="readout", noise_level=0.02)
result = executor.simulate(circuit, noise_model="combined", noise_level=0.05)

# Custom NoiseModel
from qiskit_aer.noise import NoiseModel
custom_noise = NoiseModel()
# ... configure custom noise ...
result = executor.simulate(circuit, noise_model=custom_noise)
```

### With success criteria and visualization

```python
def bell_success(bitstring: str) -> bool:
    clean = bitstring.replace(" ", "")
    return clean in ["00", "11"]

result = executor.simulate(
    circuit,
    success_criteria=bell_success,
    expected_outcomes=["00", "11"],
    show_results=True,  # Display QWARD visualizations
)
```

### With statevector saving

```python
executor = QuantumCircuitExecutor(save_statevector=True, shots=1024)
result = executor.simulate(circuit)
# result may contain statevector data for intermediate analysis
```

---

## Pattern 2: IBM Quantum Hardware (Batch Mode)

### One-time account setup

```python
QuantumCircuitExecutor.configure_ibm_account(
    token="YOUR_IBM_QUANTUM_TOKEN",
    channel="ibm_quantum",
)
```

### Run on IBM QPU

```python
executor = QuantumCircuitExecutor(shots=1024)

result = executor.run_ibm(
    circuit,
    backend_name="ibm_brisbane",         # Optional: specific backend
    optimization_levels=[0, 2, 3],        # Test multiple transpilation levels
    success_criteria=bell_success,        # Optional success function
    expected_outcomes=["00", "11"],       # Optional for DSR metrics
    timeout=600,                          # Max wait time (seconds)
    poll_interval=10,                     # Status check interval
    show_progress=True,                   # Print progress messages
)

# result is IBMBatchResult
print(f"Batch ID: {result.batch_id}")
print(f"Backend: {result.backend_name}")
print(f"Status: {result.status}")

# Iterate over individual job results (one per optimization level)
for job in result.jobs:
    print(f"Opt level {job.optimization_level}:")
    print(f"  Transpiled depth: {job.circuit_depth}")
    print(f"  Success rate: {job.success_rate:.2%}")
    print(f"  Top counts: {sorted(job.counts.items(), key=lambda x: -x[1])[:5]}")

# QWARD metrics for the original circuit
print(result.qward_metrics)
```

### With inline credentials (no saved account)

```python
result = executor.run_ibm(
    circuit,
    channel="ibm_quantum",
    token="YOUR_TOKEN",
    instance="ibm-q/open/main",
    optimization_levels=[2, 3],
)
```

### Auto-select least busy backend

```python
result = executor.run_ibm(circuit)  # backend_name=None => least_busy(simulator=False)
```

### IBM Batch internals

The executor uses `qiskit_ibm_runtime.Batch` with `SamplerV2`:

```python
# What happens inside run_ibm():
# 1. Connect to QiskitRuntimeService
# 2. Get backend (named or least_busy)
# 3. Create Batch session
# 4. For each optimization_level:
#    - generate_preset_pass_manager(opt_level, backend)
#    - Transpile circuit
#    - Submit via SamplerV2
# 5. Poll until all jobs complete or timeout
# 6. Extract counts from BitArray results
# 7. Calculate QWARD metrics
# 8. Return IBMBatchResult
```

### IBMBatchResult structure

```python
@dataclass
class IBMJobResult:
    job_id: str
    optimization_level: int
    status: str
    counts: Optional[Dict[str, int]]
    circuit_depth: int
    success_rate: Optional[float]
    error: Optional[str]
    raw_result: Any

@dataclass
class IBMBatchResult:
    batch_id: str
    backend_name: str
    status: str                              # "completed", "timeout", "error"
    jobs: List[IBMJobResult]
    original_circuit_depth: int
    qward_metrics: Optional[Dict[str, DataFrame]]
    error: Optional[str]
```

---

## Pattern 3: Rigetti via qBraid (Method A)

Uses the qBraid SDK which handles Qiskit-to-Braket transpilation automatically.
This is the method integrated into `QuantumCircuitExecutor`.

### Available qBraid devices

```python
from enum import Enum

class QbraidDevice(Enum):
    IONQ = "aws_ionq"
    QIR = "qbraid_qir_simulator"
    LUCY = "aws_oqc_lucy"              # Oxford Quantum Circuits
    RIGETTI = "rigetti_ankaa_3"         # Rigetti Ankaa-3 (84 qubits)
    IBM_SANTIAGO = "ibm_q_santiago"
    IBM_SIMULATOR = "ibm_simulator"
```

### Run on Rigetti hardware

```python
executor = QuantumCircuitExecutor(shots=1024, timeout=300)

result = executor.run_qbraid(
    circuit,
    device_id="rigetti_ankaa_3",         # Rigetti Ankaa-3
    success_criteria=bell_success,
)

# result dict contains:
# - result["status"]: "completed", "failed", "timeout", "error"
# - result["job_id"]: qBraid job ID
# - result["device"]: device used
# - result["transpiled_circuit"]: braket-transpiled circuit
# - result["qward_metrics"]: QWARD metrics DataFrames
```

### How qBraid integration works

```python
# Inside run_qbraid():
# 1. Create QbraidProvider()
# 2. Get device: provider.get_device(device_id="rigetti_ankaa_3")
# 3. Transpile: qbraid_transpile(circuit, "braket")  # Qiskit -> Braket
# 4. Submit: device.run(transpiled_circuit, shots=self.shots)
# 5. Poll job.status() until COMPLETED, FAILED, or timeout
# 6. Get QWARD metrics for original circuit
# 7. On error: fallback to AerSimulator
```

### Key notes on qBraid/Rigetti
- qBraid handles Qiskit-to-Braket circuit transpilation automatically
- Default device: `rigetti_ankaa_3` (Rigetti Ankaa-3, 84 qubits)
- The executor includes automatic fallback to local simulator on error
- Install qBraid: `pip install qbraid`
- Auth: configure via `qbraid configure` CLI or environment variables

### Loading and retrieving qBraid jobs

```python
from qbraid.runtime import QbraidProvider, load_job

# Retrieve a previously submitted job by ID
job = load_job("your-qbraid-job-id", provider="qbraid")

# For AWS Braket jobs submitted through qBraid
job = load_job("arn:aws:braket:us-west-1:...", provider="aws")

# Check status and get results
status = job.status()
if str(status).split('.')[-1] in ['COMPLETED', 'DONE']:
    result = job.result()
    counts = result.data.get_counts()
```

---

## Pattern 4: Rigetti via AWS Braket Directly (Method B)

Uses `qiskit-braket-provider` for native AWS Braket access without the qBraid layer.
This method was used for the teleportation experiments on Rigetti hardware.

### Setup and authentication

```python
import os
from qiskit_braket_provider import BraketProvider

# Set AWS region (Rigetti Ankaa-3 is in us-west-1)
os.environ['AWS_DEFAULT_REGION'] = 'us-west-1'

# Also requires AWS credentials:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# Or: aws configure (CLI)

provider = BraketProvider()
```

### Submit a circuit to Rigetti Ankaa-3

```python
from qiskit import QuantumCircuit
from qiskit_braket_provider import BraketProvider

provider = BraketProvider()
backend = provider.get_backend("Ankaa-3")

# Build circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# IMPORTANT: Remove barriers before submitting (AWS Braket incompatible)
from qiskit.circuit.library import Barrier
circuit_clean = qc.copy()
circuit_clean.data = [
    (gate, qubits, clbits) for gate, qubits, clbits in qc.data
    if not isinstance(gate, Barrier)
]

# Submit job (non-blocking, returns immediately)
job = backend.run(circuit_clean, shots=10)
print(f"Job ID: {job.job_id()}")
# Job ID format: arn:aws:braket:us-west-1:ACCOUNT_ID:quantum-task/UUID
```

### Retrieve results from a submitted job

```python
from qiskit_braket_provider import BraketProvider

provider = BraketProvider()
backend = provider.get_backend("Ankaa-3")

# Retrieve using the ARN job ID
job_arn = "arn:aws:braket:us-west-1:265556963705:quantum-task/511e89fb-..."
job = backend.retrieve_job(job_arn)

status = job.status()
print(f"Job status: {status}")

# Extract counts from AWS result
aws_result = job._tasks[0].result()
measured_entry = aws_result.entries[0].entries[0]
raw_counts = dict(measured_entry.counts)  # Counter -> dict

# IMPORTANT: Convert from big-endian to little-endian (Qiskit convention)
counts = {k[::-1]: v for k, v in raw_counts.items()}
print(f"Counts: {counts}")

# Calculate success rate
measured_qubits = len(list(counts.keys())[0])
success_pattern = '0' * measured_qubits
success_rate = counts.get(success_pattern, 0) / sum(counts.values())
```

### Circuit preparation for AWS Braket

AWS Braket has specific requirements that differ from Qiskit:

```python
# 1. Remove barriers (not supported by Braket)
from qiskit.circuit.library import Barrier
circuit_clean = circuit.copy()
circuit_clean.data = [
    (gate, qubits, clbits) for gate, qubits, clbits in circuit.data
    if not isinstance(gate, Barrier)
]

# 2. If circuit has custom measurements, normalize them
circuit_clean.remove_final_measurements()
circuit_clean.measure_all()

# 3. Submit
job = backend.run(circuit_clean, shots=shots)
```

### Batch CSV workflow for AWS experiments

The QWARD project uses a CSV-based async workflow for large experiment campaigns:

```python
# Step 1: Submit jobs (non-blocking, stores job ARNs in CSV)
experiments = Experiments()
experiments.run_dynamic_payload_gates(
    payload_range=(1, 4),
    gates_range=(500, 505),
    use_aws=True,
    aws_device_id="Ankaa-3",
)
# Creates: experiment_results_dynamic_1-4_500-505_TIMESTAMP.csv

# Step 2: Later, retrieve completed results by job ARN
validator = TeleportationValidator(payload_size=1)
validator.complete_aws_results_from_csv(
    csv_file_path="experiment_results_dynamic_1-4_500-505.csv",
    output_csv_path="experiment_results_updated.csv",
    region="us-west-1",
)

# Step 3: Enrich with full job metadata via qBraid SDK
experiments = Experiments()
experiments.update_table_with_job_info(
    input_csv="experiment_results_updated.csv",
    output_csv="experiment_results_final.csv",
)
# Adds: vendor, device_id, cost, shots, timestamps, measurement_counts_sdk
```

### Provider detection logic

```python
def determine_provider(job_id: str) -> str:
    """Detect whether a job was submitted via AWS or qBraid."""
    return 'aws' if job_id.startswith('arn:aws:braket:') else 'qbraid'
```

### Key differences: qBraid vs direct AWS Braket

| Feature | qBraid (Method A) | AWS Braket (Method B) |
|---------|-------------------|----------------------|
| Install | `pip install qbraid` | `pip install qiskit-braket-provider amazon-braket-sdk` |
| Auth | `qbraid configure` | AWS credentials (`aws configure`) |
| Transpilation | Automatic (`qbraid_transpile`) | Manual (remove barriers, normalize measurements) |
| Job ID format | qBraid job ID | `arn:aws:braket:REGION:ACCOUNT:quantum-task/UUID` |
| Endianness | Handled by qBraid | Manual conversion (big-endian -> little-endian) |
| QWARD integration | Built into executor | Manual metrics extraction |
| Fallback | Auto-fallback to simulator | Manual fallback handling |
| Device names | `rigetti_ankaa_3` | `Ankaa-3` |
| Cost tracking | Via `load_job` metadata | Via `job._tasks[0].result()` |

---

## Research-Justified Noise Models

The `NoiseModelGenerator` in `qward.algorithms.noise_generator` provides hardware-calibrated
noise presets based on published calibration data (January 2026).

### IBM Heron Processors

| QPU | Type | 2Q Error (median) | Readout Error |
|-----|------|-------------------|---------------|
| ibm_boston | Heron R3 | 0.113% (1.13e-3) | 0.46% |
| ibm_pittsburgh | Heron R3 | 0.155% (1.55e-3) | 0.42% |
| ibm_kingston | Heron R2 | 0.195% (1.95e-3) | 0.95% |
| ibm_torino | Heron R1 | 0.247% (2.47e-3) | 2.93% |
| ibm_marrakesh | Heron R2 | 0.263% (2.63e-3) | 1.27% |

### Rigetti Ankaa Processors

| QPU | Qubits | 2Q Fidelity | 2Q Error |
|-----|--------|-------------|----------|
| Ankaa-3 | 84 | 99.5% median | 0.5% |

### Using preset noise configs

```python
from qward.algorithms import NoiseModelGenerator, get_preset_noise_config, list_preset_noise_configs

# List all available presets
presets = list_preset_noise_configs()
print(presets)  # IBM-HERON-R1, IBM-HERON-R2, IBM-HERON-R3, RIGETTI-ANKAA3, ...

# Get a preset config
config = get_preset_noise_config("IBM-HERON-R2")
noise_model = NoiseModelGenerator.create_from_config(config)

# Use with executor
result = executor.simulate(circuit, noise_model=noise_model)

# Or create by type with custom level
noise = NoiseModelGenerator.create_by_type("depolarizing", noise_level=0.03)
noise = NoiseModelGenerator.create_by_type("mixed", noise_level=0.05)
```

---

## Experiment Framework

For systematic experiments across multiple configurations and noise models,
use the `BaseExperimentRunner` from `qward.algorithms.experiment`.

### Campaign workflow

```python
from qward.algorithms import BaseExperimentRunner

class MyRunner(BaseExperimentRunner):
    def create_circuit(self, config):
        # Build circuit from config
        return QuantumCircuit(config.num_qubits)

    def calculate_success(self, counts, config, circuit_gen):
        # Define success measurement
        return success_count / total_shots

    def create_result(self, ...):
        return MyExperimentResult(...)

runner = MyRunner()
results = runner.run_campaign(
    config_ids=["S2-1", "S3-1", "S4-1"],
    noise_ids=["IDEAL", "IBM-HERON-R2", "RIGETTI-ANKAA3"],
    num_runs=10,
)
```

### Experiment result data

Each result includes:
- Circuit properties: `num_qubits`, `circuit_depth`, `total_gates`
- Execution data: `counts`, `success_rate`, `execution_time_ms`
- QWARD metrics: full `qward_metrics` dict with pre-runtime metrics
- Metadata: `config_id`, `noise_model`, `run_number`, `timestamp`

### QPU config generator

After simulator experiments, use `qpu_config_generator.py` to identify
which configurations are worth running on real QPU hardware:

```bash
python qward/examples/papers/qpu_config_generator.py \
    --algorithm grover \
    --data-dir grover/data/simulator/raw \
    --noise-models IBM-HERON-R2 RIGETTI-ANKAA3 \
    --reference-noise IBM-HERON-R2
```

This classifies configs into three regions:
- **Region 1**: Algorithm works well (>50% success, >2x random chance) - run on QPU
- **Region 2**: Marginal success (>2x random chance) - boundary exploration
- **Region 3**: Algorithm fails - document limits only

---

## Real QPU Experiment Data in QWARD

The QWARD project has run extensive experiments on real quantum hardware:

### Grover's Algorithm
- **Simulator campaigns** with IBM Heron R1/R2/R3 and Rigetti Ankaa-3 noise models
- **Configs**: Scalability (S2-S8), Marked count (M3/M4), Hamming weight (H3/H4), Symmetry (SYM/ASYM)
- **Data**: `qward/examples/papers/grover/data/simulator/raw/*_RIGETTI-ANKAA3.json`
- **QPU guide**: `qward/examples/papers/grover/grover_qpu.md` (Three Regions Framework)

### QFT (Quantum Fourier Transform)
- **Modes**: Roundtrip verification (QFT -> QFT^-1) and period detection
- **Simulator campaigns** with IBM and Rigetti noise models
- **Data**: `qward/examples/papers/qft/data/raw/*_RIGETTI-ANKAA3.json`
- **QPU guide**: `qward/examples/papers/qft/qft_qpu.md`

### Teleportation Protocol
- **IBM hardware**: `qward/examples/papers/teleportation/ibm/` (1-5 qubits, 200-20000 shots)
- **AWS Braket (Rigetti)**: `qward/examples/papers/teleportation/aws/` (1-4 qubits, 500-7000 shots)
  - Uses direct AWS Braket ARNs (`arn:aws:braket:us-west-1:...`)
  - Dynamic payload gates protocol
  - Results include gate counts (h, x, y, z, s, sdg, cx, cz) and success rates

### Three Regions Framework
Used for QPU experiment selection:
- **Region 1** (Signal Dominant): Success > 50% AND > 2x random chance -> run on QPU
- **Region 2** (Signal + Noise): Success > 2x random chance -> boundary exploration
- **Region 3** (Noise Dominant): Success <= 2x random chance -> document limits only

---

## Extracting QWARD Metrics (Standalone)

```python
# Without running execution, just get metrics
metrics = executor.get_circuit_metrics(
    circuit,
    job=job,                          # Optional: for CircuitPerformanceMetrics
    success_criteria=bell_success,    # Optional
    expected_outcomes=["00", "11"],   # Optional for DSR
)

# metrics is Dict[str, DataFrame]:
# - "qiskit": QiskitMetrics (depth, width, gate counts...)
# - "complexity": ComplexityMetrics (gate density, entanglement ratio...)
# - "circuit_performance": CircuitPerformanceMetrics (success rate, error rate...)
```

---

## Complete Example: Run on All Backends

```python
import os
from qiskit import QuantumCircuit
from qward.algorithms import QuantumCircuitExecutor

# Build a Bell state circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

def bell_success(bs: str) -> bool:
    return bs.replace(" ", "") in ["00", "11"]

executor = QuantumCircuitExecutor(shots=1024)

# 1. Local simulation (ideal)
sim_result = executor.simulate(qc, success_criteria=bell_success)
print(f"Simulator: {sim_result['counts']}")

# 2. Local simulation (IBM Heron R2 noise)
from qward.algorithms import get_preset_noise_config, NoiseModelGenerator
noise = NoiseModelGenerator.create_from_config(get_preset_noise_config("IBM-HERON-R2"))
noisy_result = executor.simulate(qc, noise_model=noise, success_criteria=bell_success)
print(f"Noisy sim: {noisy_result['counts']}")

# 3. IBM Quantum hardware
ibm_result = executor.run_ibm(
    qc, optimization_levels=[0, 2, 3], success_criteria=bell_success
)
for job in ibm_result.jobs:
    print(f"IBM opt={job.optimization_level}: {job.success_rate:.2%}")

# 4. Rigetti via qBraid (Method A)
rigetti_result = executor.run_qbraid(qc, success_criteria=bell_success)
print(f"Rigetti (qBraid): {rigetti_result['status']}")

# 5. Rigetti via AWS Braket directly (Method B)
from qiskit_braket_provider import BraketProvider
from qiskit.circuit.library import Barrier

os.environ['AWS_DEFAULT_REGION'] = 'us-west-1'
aws_provider = BraketProvider()
aws_backend = aws_provider.get_backend("Ankaa-3")

# Remove barriers for AWS compatibility
qc_clean = qc.copy()
qc_clean.data = [
    (g, q, c) for g, q, c in qc.data if not isinstance(g, Barrier)
]
aws_job = aws_backend.run(qc_clean, shots=10)
print(f"AWS Braket job ARN: {aws_job.job_id()}")
# Retrieve later: aws_backend.retrieve_job(aws_job.job_id())
```
