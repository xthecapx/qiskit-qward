---
name: qiskit-development
description: IBM Qiskit quantum computing framework for circuit design, transpilation, execution, and analysis. Use when building quantum circuits, running on IBM Quantum hardware or simulators, running on Rigetti hardware via qBraid or AWS Braket directly, working with Qiskit Runtime primitives (Sampler/Estimator), optimizing transpilation, implementing quantum algorithms (VQE, QAOA, Grover), using QWARD's QuantumCircuitExecutor for simulate/run_ibm/run_qbraid workflows, direct AWS Braket submission with qiskit-braket-provider, noise model generation (IBM Heron R1-R3, Rigetti Ankaa-3), experiment campaigns with async job retrieval, or integrating with the QWARD metrics library. Covers Qiskit v2 primitives, session/batch execution modes, error mitigation, qBraid transpilation, AWS Braket integration, and visualization.
---

# Qiskit Development

## Overview

Qiskit is the world's most popular open-source quantum computing framework (13M+ downloads). Build quantum circuits, optimize for hardware, execute on simulators or real quantum computers, and analyze results with QWARD metrics.

**Key capabilities:**
- Backend-agnostic execution (local simulators, IBM Quantum cloud, or Rigetti via qBraid)
- V2 primitives: StatevectorSampler, StatevectorEstimator
- 83x faster transpilation, 29% fewer two-qubit gates
- Algorithm libraries for optimization, chemistry, and ML
- QWARD's QuantumCircuitExecutor: unified `simulate()`, `run_ibm()`, `run_qbraid()` interface
- Research-justified noise presets: IBM Heron R1/R2/R3, Rigetti Ankaa-3

## Quick Start

```python
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

sampler = StatevectorSampler()
result = sampler.run([qc], shots=1024).result()
counts = result[0].data.meas.get_counts()
print(counts)  # {'00': ~512, '11': ~512}
```

## Integration with QWARD

```python
from qiskit import QuantumCircuit
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)

scanner = Scanner(circuit=circuit)
scanner.add_strategy(QiskitMetrics(circuit))
scanner.add_strategy(ComplexityMetrics(circuit))
results = scanner.calculate_metrics()
```

## Reference Documentation

Load these as needed based on your task:

- **`references/setup.md`** - Installation, IBM Quantum account, authentication
- **`references/circuits.md`** - QuantumCircuit building, gates, measurements, composition
- **`references/primitives.md`** - Sampler and Estimator (V2), parameter binding, sessions
- **`references/transpilation.md`** - Optimization levels 0-3, layout, routing, basis gates
- **`references/visualization.md`** - Circuit drawings, histograms, Bloch spheres, state plots
- **`references/backends.md`** - IBM Quantum, IonQ, Aer, Rigetti via qBraid, Rigetti via AWS Braket direct, error mitigation
- **`references/patterns.md`** - Map/Optimize/Execute/Post-process workflow
- **`references/algorithms.md`** - VQE, QAOA, Grover, quantum chemistry, ML, optimization
- **`references/qward-executor.md`** - QWARD's QuantumCircuitExecutor: simulate, run_ibm, run_qbraid, direct AWS Braket, async job retrieval, noise presets, experiment framework

## Workflow Decision Guide

- Install Qiskit or set up IBM Quantum account -> `references/setup.md`
- Build a new quantum circuit -> `references/circuits.md`
- Run circuits and get measurements -> `references/primitives.md`
- Optimize circuits for hardware -> `references/transpilation.md`
- Visualize circuits or results -> `references/visualization.md`
- Execute on IBM Quantum hardware -> `references/backends.md`
- Execute on Rigetti via qBraid -> `references/qward-executor.md` (Pattern 3)
- Execute on Rigetti via AWS Braket direct -> `references/qward-executor.md` (Pattern 4)
- Run experiments with QWARD executor (simulate, IBM, Rigetti) -> `references/qward-executor.md`
- Retrieve async AWS Braket jobs, CSV batch workflows -> `references/qward-executor.md`
- Configure noise models (IBM Heron, Rigetti Ankaa) -> `references/qward-executor.md`
- Implement end-to-end quantum workflow -> `references/patterns.md`
- Build specific algorithm (VQE, QAOA, etc.) -> `references/algorithms.md`

## Common Patterns

### Pattern 1: QWARD Executor - Local Simulation

```python
from qiskit import QuantumCircuit
from qward.algorithms import QuantumCircuitExecutor

executor = QuantumCircuitExecutor(shots=1024)
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Ideal simulation with automatic QWARD metrics
result = executor.simulate(qc, show_results=True)
print(result["counts"])
print(result["qward_metrics"])

# With noise model
result = executor.simulate(qc, noise_model="depolarizing", noise_level=0.05)
```

### Pattern 2: QWARD Executor - IBM Quantum Hardware (Batch Mode)

```python
from qward.algorithms import QuantumCircuitExecutor

executor = QuantumCircuitExecutor(shots=1024)

# One-time setup:
# QuantumCircuitExecutor.configure_ibm_account(token="YOUR_TOKEN")

result = executor.run_ibm(
    qc,
    optimization_levels=[0, 2, 3],
    success_criteria=lambda bs: bs.replace(" ", "") in ["00", "11"],
)

for job in result.jobs:
    print(f"Opt {job.optimization_level}: depth={job.circuit_depth}, success={job.success_rate:.2%}")
```

### Pattern 3: QWARD Executor - Rigetti via qBraid

```python
from qward.algorithms import QuantumCircuitExecutor

executor = QuantumCircuitExecutor(shots=1024, timeout=300)

# Run on Rigetti Aspen-M3 through qBraid
result = executor.run_qbraid(
    qc,
    device_id="rigetti_aspen_m_3",
    success_criteria=lambda bs: bs.replace(" ", "") in ["00", "11"],
)
print(f"Status: {result['status']}")
print(f"QWARD metrics: {result['qward_metrics']}")
```

### Pattern 4: Rigetti via AWS Braket Direct (qiskit-braket-provider)

```python
import os
from qiskit_braket_provider import BraketProvider

os.environ['AWS_DEFAULT_REGION'] = 'us-west-1'
provider = BraketProvider()
backend = provider.get_backend("Ankaa-3")

# Remove barriers (AWS incompatible) and submit
from qiskit.circuit.library import Barrier
circuit_clean = qc.copy()
circuit_clean.data = [
    (g, q, c) for g, q, c in qc.data if not isinstance(g, Barrier)
]
job = backend.run(circuit_clean, shots=10)
print(f"Job ARN: {job.job_id()}")

# Later: retrieve results (big-endian -> little-endian conversion)
job = backend.retrieve_job(job.job_id())
raw_counts = dict(job._tasks[0].result().entries[0].entries[0].counts)
counts = {k[::-1]: v for k, v in raw_counts.items()}
```

### Pattern 5: Research-Justified Noise Models

```python
from qward.algorithms import NoiseModelGenerator, get_preset_noise_config

# Use hardware-calibrated presets (IBM Heron R1-R3, Rigetti Ankaa-3)
noise = NoiseModelGenerator.create_from_config(get_preset_noise_config("IBM-HERON-R2"))
result = executor.simulate(qc, noise_model=noise)

noise_rigetti = NoiseModelGenerator.create_from_config(get_preset_noise_config("RIGETTI-ANKAA3"))
result = executor.simulate(qc, noise_model=noise_rigetti)
```

### Pattern 6: Variational Algorithm (VQE)

```python
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from scipy.optimize import minimize

with Session(backend=backend) as session:
    estimator = Estimator(session=session)

    def cost_function(params):
        bound_qc = ansatz.assign_parameters(params)
        qc_isa = transpile(bound_qc, backend=backend)
        result = estimator.run([(qc_isa, hamiltonian)]).result()
        return result[0].data.evs

    result = minimize(cost_function, initial_params, method='COBYLA')
```

### Pattern 7: Experiment Campaign (Systematic Multi-Config)

```python
from qward.algorithms import BaseExperimentRunner

# Subclass for your algorithm, then run campaign across
# multiple circuit configs and noise models:
runner = MyAlgorithmRunner()
results = runner.run_campaign(
    config_ids=["S2-1", "S3-1", "S4-1"],
    noise_ids=["IDEAL", "IBM-HERON-R2", "RIGETTI-ANKAA3"],
    num_runs=10,
)
```

## Best Practices

1. **Start with simulators**: Use `executor.simulate()` before hardware
2. **Always transpile**: Use `optimization_level=3` for production
3. **Use QWARD executor**: Unified interface for simulate, IBM QPU, Rigetti
4. **Use appropriate primitives**: Sampler for bitstrings, Estimator for expectation values
5. **Choose execution mode**: Session for iterative (VQE/QAOA), Batch for parallel, qBraid for Rigetti
6. **Use hardware-calibrated noise**: Preset noise models for IBM Heron and Rigetti Ankaa
7. **Minimize two-qubit gates**: Major error source on hardware
8. **Save job IDs**: For later retrieval of hardware results
9. **Apply error mitigation**: Use `resilience_level` in runtime options
10. **Run experiment campaigns**: Systematic comparison across configs and noise models
