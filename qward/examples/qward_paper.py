# Create circuit
from qiskit import QuantumCircuit


def create_circuit():
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    print(f">>> Circuit: {circuit}")

    circuit.draw(output="mpl")

    return circuit


# Backend simulator

from qiskit_aer import AerSimulator, AerJob
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    pauli_error,
    depolarizing_error,
)


def run_simulators(circuit):
    # ****** no noise ******
    no_noise = AerSimulator()

    # ****** noise model with depolarizing errors ******
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
    depolarizing_errors = AerSimulator(noise_model=noise_model1)

    # ****** noise model with Pauli errors ******
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
    pauli_errors = AerSimulator(noise_model=noise_model2)

    jobs = [
        no_noise.run(circuit, shots=1024),
        depolarizing_errors.run(circuit, shots=1024),
        pauli_errors.run(circuit, shots=1024),
    ]

    return {
        "jobs": jobs,
        "circuits": [circuit, circuit, circuit],
        "backends": [no_noise, depolarizing_errors, pauli_errors],
    }


# IBM Backend

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, Batch
from qiskit.transpiler import generate_preset_pass_manager
import time

IBM_QUANTUM_CHANNEL = ""
IBM_QUANTUM_TOKEN = ""

QiskitRuntimeService.save_account(
    channel=IBM_QUANTUM_CHANNEL, token=IBM_QUANTUM_TOKEN, overwrite=True
)


def run_ibm(circuit):
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)
    print(f">>> Backend name: {backend.configuration().backend_name}")
    print(f">>> Backend status: {backend.status().pending_jobs}")

    batch = Batch(backend=backend)
    print(f">>> Batch ID: {batch.session_id}")

    pass_manager_0 = generate_preset_pass_manager(
        optimization_level=0,
        backend=backend,
    )
    pass_manager_1 = generate_preset_pass_manager(
        optimization_level=2,
        backend=backend,
        layout_method="sabre",
        routing_method="sabre",
    )
    pass_manager_2 = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend,
        layout_method="dense",
        routing_method="lookahead",
    )

    with batch:
        sampler = Sampler()
        isa_circuit_0 = pass_manager_0.run(circuit)
        isa_circuit_1 = pass_manager_1.run(circuit)
        isa_circuit_2 = pass_manager_2.run(circuit)
        job_0 = sampler.run([isa_circuit_0])
        job_1 = sampler.run([isa_circuit_1])
        job_2 = sampler.run([isa_circuit_2])

    start_time = time.time()

    while time.time() - start_time < 600:  # 10 minutes
        status = batch.status()
        print(f">>> Batch Status: {status}")

        # Print individual job statuses
        print(f">>> Job 0 (Opt Level 0) Status: {job_0.status()}")
        print(f">>> Job 1 (Opt Level 2) Status: {job_1.status()}")
        print(f">>> Job 2 (Opt Level 3) Status: {job_2.status()}")

        # Check if all jobs are completed
        if all(job.status() in ["DONE", "CANCELLED", "ERROR"] for job in [job_0, job_1, job_2]):
            print(">>> All jobs completed!")
            break

        time.sleep(10)  # Wait 10 seconds before checking again

    # Close the batch
    batch.close()
    print(f">>> Batch {batch.session_id} closed")

    # Print job results
    try:
        print("\n>>> Job Results:")
        print(f"Job 0 (Optimization Level 0): {job_0.result()}")
        print(f"Job 1 (Optimization Level 2): {job_1.result()}")
        print(f"Job 2 (Optimization Level 3): {job_2.result()}")

        # Print circuit depths for comparison
        print(f"\n>>> Circuit Depths:")
        print(f"Original circuit depth: {circuit.depth()}")
        print(f"Optimized circuit 0 depth: {isa_circuit_0.depth()}")
        print(f"Optimized circuit 1 depth: {isa_circuit_1.depth()}")
        print(f"Optimized circuit 2 depth: {isa_circuit_2.depth()}")

    except Exception as e:
        print(f">>> Error retrieving results: {e}")

    return {
        "batch_id": batch.session_id,
        "jobs": [job_0, job_1, job_2],
        "circuits": [isa_circuit_0, isa_circuit_1, isa_circuit_2],
    }


# Get jobs

circuit = create_circuit()
simulators_results = run_simulators(circuit)
ibm_results = run_ibm(circuit)

# qWard circuit data

from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics


# define metrics
def bell_success_criteria(result: str) -> bool:
    # Remove spaces to get clean bit string
    clean_result = result.replace(" ", "")
    # For Bell states, we expect either all 0s or all 1s
    return clean_result in ["00", "11"]


def expected_distribution():
    # For Bell states, we expect either all 0s or all 1s
    return {"00": 0.5, "11": 0.5}


bell_success = CircuitPerformanceMetrics(
    circuit,
    jobs=simulators_results["jobs"],
    success_criteria=bell_success_criteria,
    expected_distribution=expected_distribution(),
)

# Create a scanner with the circuit
scanner = Scanner(
    circuit=circuit,
    strategies=[
        bell_success,
        QiskitMetrics,
        ComplexityMetrics,
    ],
)

metrics_dict = scanner.calculate_metrics()

# Display summary using Scanner method
scanner.display_summary(metrics_dict)

# Display raw DataFrames for detailed analysis
print("\n" + "=" * 80)
print("📊 DETAILED METRICS DATA")
print("=" * 80)
for metric_name, metric_df in metrics_dict.items():
    print(f"\n=== {metric_name} ===")
    display(metric_df)

# Visualize metrics

from qward.visualization.constants import Metrics, Plots
from qward.visualization import PlotConfig
from qward import Visualizer

visualizer = Visualizer(scanner=scanner, output_dir="img/qward_paper")

error_plot = visualizer.generate_plot(
    metric_name=Metrics.CIRCUIT_PERFORMANCE,
    plot_name=Plots.CircuitPerformance.SUCCESS_ERROR_COMPARISON,
    save=True,
    show=True,
)

error_plot.show()

# Dashboard

dashboard = visualizer.create_dashboard(save=True, show=True)

### IBM on qWard


# define metrics
def bell_success_criteria(result: str) -> bool:
    # Remove spaces to get clean bit string
    clean_result = result.replace(" ", "")
    # For Bell states, we expect either all 0s or all 1s
    return clean_result in ["00", "11"]


def expected_distribution():
    # For Bell states, we expect either all 0s or all 1s
    return {"00": 0.5, "11": 0.5}


bell_success = CircuitPerformanceMetrics(
    circuit,
    jobs=ibm_results["jobs"],
    success_criteria=bell_success_criteria,
    expected_distribution=expected_distribution(),
)

# Create a scanner with the circuit
scanner = Scanner(
    circuit=circuit,
    strategies=[
        bell_success,
        QiskitMetrics,
        ComplexityMetrics,
    ],
)

metrics_dict = scanner.calculate_metrics()

# Display summary using Scanner method
scanner.display_summary(metrics_dict)

# Display raw DataFrames for detailed analysis
print("\n" + "=" * 80)
print("📊 DETAILED METRICS DATA")
print("=" * 80)
for metric_name, metric_df in metrics_dict.items():
    print(f"\n=== {metric_name} ===")
    display(metric_df)


from qward.visualization.constants import Metrics, Plots
from qward.visualization import PlotConfig
from qward import Visualizer

visualizer = Visualizer(scanner=scanner, output_dir="img/qward_paper")

error_plot = visualizer.generate_plot(
    metric_name=Metrics.CIRCUIT_PERFORMANCE,
    plot_name=Plots.CircuitPerformance.SUCCESS_ERROR_COMPARISON,
    save=True,
    show=True,
)

error_plot.show()

dashboard = visualizer.create_dashboard(save=True, show=True)


# # get ibm jobs from id

# # 87bc380e-026f-429a-9a51-5ad61b5d3b9e
# def get_ibm_from_id(session_id=None):
#     # Get job results from the service
#     service = QiskitRuntimeService(
#         channel=IBM_QUANTUM_CHANNEL,
#         instance='thecap',
#         token=IBM_QUANTUM_TOKEN
#     )

#     if session_id:
#         # Option 1: Get a specific batch by session ID
#         try:
#             batch = Batch.from_id(session_id, service)
#             print(f">>> Batch ID: {batch.session_id}")
#             print(f">>> Batch Status: {batch.status()}")
#             print(f">>> Batch Details: {batch.details()}")
#             return batch
#         except Exception as e:
#             print(f">>> Error retrieving batch: {e}")

#     # Option 2: Get all jobs and filter by session_id if provided
#     if session_id:
#         jobs = service.jobs(session_id=session_id, limit=100)
#         print(f">>> Found {len(jobs)} jobs in session {session_id}")
#     else:
#         # Get all recent jobs
#         jobs = service.jobs(limit=100)
#         print(f">>> Found {len(jobs)} total jobs")

#         # Group jobs by session_id to see which batches/sessions exist
#         sessions = {}
#         for job in jobs:
#             sid = getattr(job, 'session_id', None)
#             if sid:
#                 if sid not in sessions:
#                     sessions[sid] = []
#                 sessions[sid].append(job)

#         print(f">>> Found {len(sessions)} unique sessions/batches:")
#         for sid, session_jobs in sessions.items():
#             print(f"    Session {sid}: {len(session_jobs)} jobs")

#     return jobs

# def get_jobs_from_batch(batch):
#     """
#     Extract jobs from a batch object in different ways
#     """
#     print(f">>> Working with Batch ID: {batch.session_id}")
#     print(f">>> Batch Status: {batch.status()}")

#     # Method 1: Get jobs directly from the service using session_id
#     service = QiskitRuntimeService(
#         channel=IBM_QUANTUM_CHANNEL,
#         instance='thecap',
#         token=IBM_QUANTUM_TOKEN
#     )

#     # Get all jobs from this batch/session
#     batch_jobs = service.jobs(session_id=batch.session_id, limit=100)
#     print(f">>> Found {len(batch_jobs)} jobs in this batch")

#     # Method 2: If you have the batch object from run_ibm(),
#     # the jobs are already stored in the returned dictionary
#     # (This would be ibm_results['jobs'] from your run_ibm function)

#     # Display job information
#     for i, job in enumerate(batch_jobs):
#         print(f"    Job {i}: {job.job_id()}")
#         print(f"        Status: {job.status()}")
#         print(f"        Creation Date: {job.creation_date}")

#         # Get results if job is completed
#         if job.status() == 'DONE':
#             try:
#                 result = job.result()
#                 print(f"        Result: {result}")
#             except Exception as e:
#                 print(f"        Error getting result: {e}")
#         print()

#     return batch_jobs

# def extract_sampler_results(primitive_result):
#     """
#     Extract measurement results from PrimitiveResult (SamplerV2 format)

#     Args:
#         primitive_result: PrimitiveResult object from SamplerV2

#     Returns:
#         dict: Dictionary with counts and other useful information
#     """
#     print(">>> Extracting results from PrimitiveResult...")

#     # Get the first (and usually only) pub result
#     pub_result = primitive_result[0]  # SamplerPubResult

#     # Extract the BitArray from the data
#     bit_array = pub_result.data.c  # 'c' is the classical register name

#     # Get basic information
#     num_shots = bit_array.num_shots
#     num_bits = bit_array.num_bits
#     shape = bit_array.shape

#     print(f">>> Number of shots: {num_shots}")
#     print(f">>> Number of bits: {num_bits}")
#     print(f">>> Shape: {shape}")

#     # Convert BitArray to counts (like the old format)
#     # Method 1: Get all bit strings
#     bit_strings = []
#     for shot in range(num_shots):
#         # Get the bit string for this shot
#         bits = bit_array[shot] if shape == () else bit_array[shot, :]
#         bit_string = ''.join(str(int(b)) for b in bits)
#         bit_strings.append(bit_string)

#     # Method 2: Count occurrences to create counts dictionary
#     from collections import Counter
#     counts = Counter(bit_strings)

#     print(f">>> Counts: {dict(counts)}")

#     # Method 3: Get execution metadata
#     execution_metadata = primitive_result.metadata.get('execution', {})
#     execution_spans = execution_metadata.get('execution_spans', [])

#     if execution_spans:
#         span = execution_spans[0]
#         print(f">>> Execution time: {span.start} to {span.stop}")
#         print(f">>> Total shots in span: {span.size}")

#     # Return structured data
#     return {
#         'counts': dict(counts),
#         'bit_strings': bit_strings,
#         'num_shots': num_shots,
#         'num_bits': num_bits,
#         'raw_bit_array': bit_array,
#         'metadata': primitive_result.metadata
#     }

# # Example usage:
# # result = job.result()  # This gives you PrimitiveResult
# # extracted_data = extract_sampler_results(result)
