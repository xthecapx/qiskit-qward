"""Run Estimator on IBM QPU — GHZ-4 with 6 observables.

Submits a single Estimator job to validate QWARD's Estimator metrics
end-to-end on real hardware.

Usage:
    uv run qward/examples/estimator_ibm_experiment.py

After completion, run:
    uv run qward/examples/estimator_ibm_verify.py
"""

import json
import os
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService, Batch


def build_ghz4_circuit() -> QuantumCircuit:
    """Build a 4-qubit GHZ state: |0000> + |1111> / sqrt(2)."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    return qc


def main():
    circuit = build_ghz4_circuit()

    observable_labels = ["ZZZZ", "XXXX", "ZZII", "IIZZ", "XZXZ", "YYYY"]
    obs_list = [SparsePauliOp.from_list([(lbl, 1.0)]) for lbl in observable_labels]

    # GHZ-4 ideal expectation values:
    # ZZZZ=+1, XXXX=+1, ZZII=+1, IIZZ=+1, XZXZ=0, YYYY=-1
    ideal_values = np.array([1.0, 1.0, 1.0, 1.0, 0.0, -1.0])

    load_dotenv()
    kwargs = {}
    channel = os.getenv("IBM_QUANTUM_CHANNEL")
    token = os.getenv("IBM_QUANTUM_TOKEN")
    instance = os.getenv("IBM_QUANTUM_INSTANCE")
    if channel:
        kwargs["channel"] = channel
    if token:
        kwargs["token"] = token
    if instance:
        kwargs["instance"] = instance

    service = QiskitRuntimeService(**kwargs)
    backend = service.least_busy(min_num_qubits=4, operational=True)
    print(f"Backend: {backend.name}")

    pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
    isa_circuit = pm.run(circuit)
    print(f"ISA depth: {isa_circuit.depth()}, gates: {isa_circuit.size()}")

    isa_obs_list = [obs.apply_layout(isa_circuit.layout) for obs in obs_list]

    with Batch(backend=backend) as batch:
        estimator = EstimatorV2()
        job = estimator.run([(isa_circuit, isa_obs_list)])

    job_id = job.job_id()
    print(f"Job ID: {job_id}")
    print(f"Status: {job.status()}")

    result_data = {
        "job_id": job_id,
        "backend": backend.name,
        "circuit": "GHZ-4",
        "num_qubits": 4,
        "observables": observable_labels,
        "ideal_values": ideal_values.tolist(),
        "optimization_level": 2,
        "isa_depth": isa_circuit.depth(),
        "timestamp": datetime.now().isoformat(),
    }

    output_path = "qward/examples/estimator_ibm_result.json"
    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"\nSaved metadata to: {output_path}")
    print("\nAfter job completes, run:")
    print("  uv run qward/examples/estimator_ibm_verify.py")


if __name__ == "__main__":
    main()
