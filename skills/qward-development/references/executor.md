# Executor & Algorithms Reference

QWARD provides a unified executor for running circuits and built-in algorithm implementations.

## QuantumCircuitExecutor

Unified interface for simulators and hardware.

### Basic Usage

```python
from qward.algorithms import QuantumCircuitExecutor

executor = QuantumCircuitExecutor(shots=1024, timeout=300)
```

### simulate() - Local Simulation

```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Ideal simulation
result = executor.simulate(qc, show_results=True)
print(result["counts"])
print(result["qward_metrics"])

# With noise model (string preset)
result = executor.simulate(qc, noise_model="depolarizing", noise_level=0.05)

# With noise model (hardware preset)
from qward.algorithms import get_preset_noise_config, NoiseModelGenerator

config = get_preset_noise_config("IBM-HERON-R2")
noise = NoiseModelGenerator.create_from_config(config)
result = executor.simulate(qc, noise_model=noise)
```

### run_ibm() - IBM Quantum Hardware

```python
# One-time account setup
# QuantumCircuitExecutor.configure_ibm_account(token="YOUR_TOKEN")

result = executor.run_ibm(
    qc,
    optimization_levels=[0, 2, 3],
    success_criteria=lambda bs: bs.replace(" ", "") in ["00", "11"],
)

for job in result.jobs:
    print(f"Opt {job.optimization_level}: depth={job.circuit_depth}, success={job.success_rate:.2%}")
```

### run_qbraid() - Rigetti via qBraid

```python
result = executor.run_qbraid(
    qc,
    device_id="rigetti_aspen_m_3",
    success_criteria=lambda bs: bs.replace(" ", "") in ["00", "11"],
)

print(f"Status: {result['status']}")
print(f"Counts: {result['counts']}")
print(f"QWARD metrics: {result['qward_metrics']}")
```

### AWS Braket Direct

```python
import os
from qiskit_braket_provider import BraketProvider

os.environ['AWS_DEFAULT_REGION'] = 'us-west-1'
provider = BraketProvider()
backend = provider.get_backend("Ankaa-3")

# Remove barriers (AWS incompatible)
from qiskit.circuit.library import Barrier
circuit_clean = qc.copy()
circuit_clean.data = [
    (g, q, c) for g, q, c in qc.data if not isinstance(g, Barrier)
]

job = backend.run(circuit_clean, shots=10)
print(f"Job ARN: {job.job_id()}")

# Later: retrieve results
job = backend.retrieve_job(job.job_id())
raw_counts = dict(job._tasks[0].result().entries[0].entries[0].counts)
counts = {k[::-1]: v for k, v in raw_counts.items()}  # Big-endian to little-endian
```

## Noise Model Generator

Hardware-calibrated noise presets.

```python
from qward.algorithms import (
    NoiseModelGenerator,
    NoiseConfig,
    get_preset_noise_config,
    list_preset_noise_configs,
    PRESET_NOISE_CONFIGS
)

# List available presets
presets = list_preset_noise_configs()
# ["IBM-HERON-R1", "IBM-HERON-R2", "IBM-HERON-R3", "RIGETTI-ANKAA3"]

# Get preset config
config = get_preset_noise_config("IBM-HERON-R2")

# Create noise model
noise = NoiseModelGenerator.create_from_config(config)

# Use in simulation
result = executor.simulate(circuit, noise_model=noise)
```

### Custom Noise Config

```python
from qward.algorithms import NoiseConfig, NoiseModelGenerator

custom_config = NoiseConfig(
    name="custom",
    single_qubit_error=0.001,
    two_qubit_error=0.01,
    readout_error=0.02,
    t1_time=100e-6,
    t2_time=50e-6,
)

noise = NoiseModelGenerator.create_from_config(custom_config)
```

## Experiment Framework

Systematic experiment campaigns.

### BaseExperimentRunner

```python
from qward.algorithms import BaseExperimentRunner, BaseExperimentResult

class MyExperimentRunner(BaseExperimentRunner):
    def _create_circuit(self, config_id: str) -> QuantumCircuit:
        """Create circuit for given configuration."""
        # Your circuit generation logic
        pass

    def _get_success_criteria(self, config_id: str):
        """Define success criteria for configuration."""
        return lambda bs: bs == "expected_result"

    def _run_single(self, config_id: str, noise_id: str) -> BaseExperimentResult:
        """Run single experiment."""
        pass

# Run campaign
runner = MyExperimentRunner()
results = runner.run_campaign(
    config_ids=["config-1", "config-2", "config-3"],
    noise_ids=["IDEAL", "IBM-HERON-R2", "RIGETTI-ANKAA3"],
    num_runs=10,
)
```

### Experiment Analysis

```python
from qward.algorithms import (
    compute_descriptive_stats,
    test_normality,
    analyze_noise_impact,
    compare_noise_models,
    generate_campaign_report,
)

# Statistical analysis
stats = compute_descriptive_stats(success_rates)

# Normality testing
normality = test_normality(data, method="shapiro")

# Noise impact analysis
impact = analyze_noise_impact(ideal_results, noisy_results)

# Compare noise models
comparison = compare_noise_models(results_dict)

# Generate report
report = generate_campaign_report(campaign_results)
```

## Built-in Algorithms

### Grover's Algorithm

```python
from qward.algorithms import Grover, GroverOracle, GroverCircuitGenerator

# Create oracle for marked states
oracle = GroverOracle(num_qubits=3, marked_states=["101", "110"])

# Generate Grover circuit
generator = GroverCircuitGenerator(oracle)
circuit = generator.create_circuit(iterations=2)

# Or use high-level API
grover = Grover(num_qubits=3, marked_states=["101"])
result = grover.run(shots=1024)
```

### Quantum Fourier Transform (QFT)

```python
from qward.algorithms import QFT, QFTCircuitGenerator

# Generate QFT circuit
generator = QFTCircuitGenerator(num_qubits=4)
circuit = generator.create_circuit()

# Or use high-level API
qft = QFT(num_qubits=4)
circuit = qft.get_circuit()
```

### Phase Estimation

```python
from qward.algorithms import PhaseEstimation, PhaseEstimationCircuitGenerator

# Create phase estimation circuit
generator = PhaseEstimationCircuitGenerator(
    unitary=unitary_matrix,
    num_counting_qubits=4
)
circuit = generator.create_circuit()

# High-level API
pe = PhaseEstimation(unitary=unitary, precision_qubits=4)
result = pe.run(shots=1024)
```

### Teleportation Protocols

```python
from qward.algorithms import (
    StandardTeleportationProtocol,
    VariationTeleportationProtocol,
    TeleportationCircuitGenerator,
)

# Standard teleportation
protocol = StandardTeleportationProtocol()
circuit = protocol.create_circuit()

# Variation protocol
variation = VariationTeleportationProtocol(variation_params=params)
circuit = variation.create_circuit()
```

### Matrix Product Verification

```python
from qward.algorithms import (
    MatrixProductVerification,
    QuantumFreivaldsVerification,
    BuhrmanSpalekVerification,
)

# Quantum Freivalds verification
verifier = QuantumFreivaldsVerification(matrix_a, matrix_b, matrix_c)
result = verifier.verify(num_trials=10)

print(f"Is correct: {result.is_correct}")
print(f"Confidence: {result.confidence:.2%}")
```

## Experiment Utilities

```python
from qward.algorithms import (
    calculate_qward_metrics,
    serialize_metrics_dict,
    ExperimentDefaults,
    DEFAULT_EXPERIMENT_PARAMS,
)

# Calculate QWARD metrics for a circuit
metrics = calculate_qward_metrics(circuit)

# Serialize for JSON export
serialized = serialize_metrics_dict(metrics)

# Default parameters
defaults = ExperimentDefaults()
print(defaults.shots)
print(defaults.optimization_level)
```
