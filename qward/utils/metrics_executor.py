import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward.metrics.circuit_performance import CircuitPerformanceMetrics
from qward.scanner import Scanner

# from qward.metrics.behavioral_metrics import BehavioralMetrics
# from qward.metrics.quantum_specific_metrics import QuantumSpecificMetrics
# from qward.metrics.structural_metrics import StructuralMetrics
# from qward.metrics.element_metrics import ElementMetrics
from qward.algorithms.pvqd import ising_evolved_state, create_pvqd_ansatz
from qiskit.quantum_info import state_fidelity


def test_with_noise(circuit: QuantumCircuit, simulator_config: dict = None, n_shots: int = 1024):
    if simulator_config is None:
        simulator_config = {"method": "statevector"}

    # Import noise model components
    from qiskit_aer.noise import (
        NoiseModel,
        ReadoutError,
        pauli_error,
        depolarizing_error,
    )

    # Create an Aer simulator with default settings (no noise)
    simulator = AerSimulator(method="statevector")

    # Run the circuit multiple times with different noise models
    jobs = []

    rng = np.random.default_rng(42)
    num_params = circuit.num_parameters
    random_params = rng.uniform(0, 2 * np.pi, num_params)
    bound_qc = circuit.assign_parameters(random_params)

    # bound_qc = circuit
    bound_qc.save_statevector()
    # Run with default noise model (no noise)
    job1 = simulator.run(bound_qc, shots=n_shots)
    jobs.append(job1)

    # Create a noise model with depolarizing errors
    noise_model1 = NoiseModel()

    # Add depolarizing error to all single qubit gates
    depol_error = depolarizing_error(0.05, 1)  # 5% depolarizing error
    noise_model1.add_all_qubit_quantum_error(depol_error, ["u1", "u2", "u3"])

    # Add depolarizing error to all two qubit gates
    depol_error_2q = depolarizing_error(0.1, 2)  # 10% depolarizing error
    noise_model1.add_all_qubit_quantum_error(depol_error_2q, ["cx"])

    # Add readout error
    readout_error = ReadoutError([[0.9, 0.1], [0.1, 0.9]])  # 10% readout error
    noise_model1.add_all_qubit_readout_error(readout_error)

    # Create a simulator with the first noise model
    noisy_simulator1 = AerSimulator(noise_model=noise_model1, method="statevector")
    job2 = noisy_simulator1.run(bound_qc, shots=1024)
    jobs.append(job2)

    # Create a noise model with Pauli errors
    noise_model2 = NoiseModel()

    # Add Pauli error to all single qubit gates
    pauli_error_1q = pauli_error([("X", 0.05), ("Y", 0.05), ("Z", 0.05), ("I", 0.85)])
    noise_model2.add_all_qubit_quantum_error(pauli_error_1q, ["u1", "u2", "u3"])

    # Add Pauli error to all two qubit gates
    pauli_error_2q = pauli_error([("XX", 0.05), ("YY", 0.05), ("ZZ", 0.05), ("II", 0.85)])
    noise_model2.add_all_qubit_quantum_error(pauli_error_2q, ["cx"])

    # Add readout error
    readout_error = ReadoutError([[0.95, 0.05], [0.05, 0.95]])  # 5% readout error
    noise_model2.add_all_qubit_readout_error(readout_error)

    # Create a simulator with the second noise model
    noisy_simulator2 = AerSimulator(noise_model=noise_model2, method="statevector")

    job3 = noisy_simulator2.run(bound_qc, shots=1024)
    jobs.append(job3)

    print(job3)
    # Wait for all jobs to complete
    # for job in jobs:
    #     job.result()

    return jobs


def calculate_metrics(circuit: QuantumCircuit, jobs: list):

    # Add CircuitPerformance strategy with multiple jobs

    # Create a scanner with the circuit

    scanner = Scanner(circuit=circuit)

    def success_criteria(counts):
        print(counts)
        output_state = counts.result().get_statevector()
        target_value = ising_evolved_state(4)
        # Calculamos la fidelidad.
        fidelity = state_fidelity(target_value, output_state)
        return fidelity

    circuit_performance_strategy = CircuitPerformanceMetrics(
        circuit=circuit, success_criteria=success_criteria
    )
    # Add jobs one by one to demonstrate the new functionality
    circuit_performance_strategy.add_job(jobs[0])  # Add first job

    circuit_performance_strategy.add_job(jobs[1:])  # Add remaining jobs as a list
    scanner.add_strategy(circuit_performance_strategy)

    # Calculate metrics
    metrics_dict = scanner.calculate_metrics()
    return metrics_dict


def main():
    circuit = create_pvqd_ansatz(num_qubits=4, reps=3)
    jobs = test_with_noise(circuit)
    metris = calculate_metrics(circuit, jobs)
    return metris
