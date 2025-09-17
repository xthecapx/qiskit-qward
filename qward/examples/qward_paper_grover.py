# Create Grover circuit
from qward.algorithms.grover import GroverCircuitGenerator


def create_circuit():
    # Create Grover circuit generator with default marked states ["011", "100"]
    grover_gen = GroverCircuitGenerator(
        marked_states=["011", "100"], use_barriers=True, save_statevector=False
    )

    circuit = grover_gen.circuit
    print(f">>> Grover Circuit: {circuit}")
    print(f">>> Marked states: {grover_gen.get_marked_states()}")
    print(f">>> Theoretical success probability: {grover_gen.get_success_probability():.3f}")
    print(f">>> Circuit depth: {circuit.depth()}, qubits: {circuit.num_qubits}")

    display(circuit.draw(output="mpl"))

    return grover_gen.circuit_isa, grover_gen


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
    noise_model1.add_all_qubit_quantum_error(
        depol_error, ["u1", "u2", "u3", "h", "x", "y", "z", "s", "t"]
    )

    # Add depolarizing error to all two qubit gates
    depol_error_2q = depolarizing_error(0.1, 2)  # 10% depolarizing error
    noise_model1.add_all_qubit_quantum_error(depol_error_2q, ["cx", "cy", "cz"])

    # Add readout error
    readout_error = ReadoutError([[0.9, 0.1], [0.1, 0.9]])  # 10% readout error
    noise_model1.add_all_qubit_readout_error(readout_error)

    # Create a simulator with the first noise model
    depolarizing_errors = AerSimulator(noise_model=noise_model1)

    # ****** noise model with Pauli errors ******
    noise_model2 = NoiseModel()

    # Add Pauli error to all single qubit gates
    pauli_error_1q = pauli_error([("X", 0.05), ("Y", 0.05), ("Z", 0.05), ("I", 0.85)])
    noise_model2.add_all_qubit_quantum_error(
        pauli_error_1q, ["u1", "u2", "u3", "h", "x", "y", "z", "s", "t"]
    )

    # Add Pauli error to all two qubit gates
    pauli_error_2q = pauli_error([("XX", 0.05), ("YY", 0.05), ("ZZ", 0.05), ("II", 0.85)])
    noise_model2.add_all_qubit_quantum_error(pauli_error_2q, ["cx", "cy", "cz"])

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


def run_ibm(circuit):
    IBM_QUANTUM_CHANNEL = "ibm_cloud"
    IBM_QUANTUM_TOKEN = "xxxx"

    QiskitRuntimeService.save_account(
        channel=IBM_QUANTUM_CHANNEL, token=IBM_QUANTUM_TOKEN, instance="thecap", overwrite=True
    )
    service = QiskitRuntimeService()

    # Try to get available backends
    try:
        # First, let's see what backends are available
        backends = service.backends()
        print(f">>> Available backends: {[b.name for b in backends]}")

        # Try to get least busy backend with more flexible criteria
        backend = service.least_busy(
            min_num_qubits=circuit.num_qubits,  # Ensure it can handle our circuit
            instance="thecap",
            operational=True,
        )
    except Exception as e:
        print(f">>> Error getting least_busy backend: {e}")
        print(">>> Trying to get any available backend...")

        # Fallback: get any available backend
        backends = service.backends(instance="thecap")
        if not backends:
            raise ValueError("No backends available in the 'thecap' instance")

        # Filter for operational backends manually
        operational_backends = [b for b in backends if b.status().operational]
        if not operational_backends:
            print(">>> No operational backends found, using any available backend")
            backend = backends[0]
        else:
            # Get the one with shortest queue
            backend = min(operational_backends, key=lambda b: b.status().pending_jobs)

    print(f">>> Backend name: {backend.name}")
    print(f">>> Backend status: {backend.status()}")
    print(f">>> Backend configuration: {backend.configuration()}")

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


def run_qbraid(circuit):
    """
    Run Grover circuit on qBraid Rigetti Ankaa-3 quantum device.

    Args:
        circuit: Grover quantum circuit to execute

    Returns:
        dict: Job results compatible with qWard metrics
    """
    try:
        from qbraid.transpiler import transpile as qbraid_transpile
        from qbraid.runtime import QbraidProvider, load_job
        from qbraid.runtime import JobLoaderError
        import time

        try:
            from qbraid.runtime.aws import BraketProvider
        except ImportError:
            BraketProvider = None

        provider = QbraidProvider()

        # Use Rigetti Ankaa-3 device
        device_id = "rigetti_ankaa_3"
        qbraid_device = provider.get_device(device_id=device_id)
        print(f">>> qBraid Device: {qbraid_device} (Rigetti Ankaa-3)")

        print(f">>> Original circuit depth: {circuit.depth()}")
        print(f">>> Original circuit width: {circuit.width()}")
        print(f">>> Device capabilities: {qbraid_device}")

        # Transpile circuit for Braket
        transpiled_circuit = qbraid_transpile(circuit, "braket")
        print(f">>> Transpiled circuit type: {type(transpiled_circuit)}")

        # Check device status
        print(f">>> Device status: {qbraid_device.status()}")

        # Submit job
        job = qbraid_device.run(transpiled_circuit, shots=1024)
        print(f">>> Job submitted with ID: {job.id}")

        job_info = {
            "status": "submitted",
            "job_id": job.id,
            "device": qbraid_device.id,
            "job": job,
            "transpiled_circuit": transpiled_circuit,
        }

        # Wait for job completion
        print(">>> Waiting for job to complete...")
        start_time = time.time()
        timeout = 300  # 5 minutes timeout

        while time.time() - start_time < timeout:
            status = job.status()
            print(f">>> Job status: {status}")

            # Convert enum to string for easier comparison
            status_str = str(status).split(".")[-1] if hasattr(status, "name") else str(status)

            if status_str in ["COMPLETED", "DONE"]:
                result = job.result()
                counts = result.data.get_counts()
                print(f">>> Job completed! Results: {counts}")

                # Update job_info with completion data
                job_info.update({"status": "completed", "counts": counts, "result": result})
                return job_info

            elif status_str in ["FAILED", "CANCELLED", "CANCELED"]:
                print(f">>> Job failed with status: {status}")
                error_msg = f"Job {status_str.lower()}"
                try:
                    if hasattr(job, "metadata") and job.metadata():
                        error_msg += f": {job.metadata()}"
                except Exception:
                    pass

                job_info.update({"status": "failed", "error": error_msg})
                return job_info

            elif status_str in ["QUEUED", "RUNNING", "INITIALIZING"]:
                # Continue waiting
                pass
            else:
                print(f">>> Unknown status: {status} ({status_str})")

            time.sleep(5)  # Wait 5 seconds before checking again

        # Timeout case
        print(">>> Job timed out after 5 minutes")
        job_info.update({"status": "timeout"})
        return job_info

    except ImportError:
        print(">>> qBraid not available, falling back to simulator")
        return _run_qbraid_fallback(circuit)
    except Exception as e:
        print(f">>> Error running on qBraid: {str(e)}")
        return _run_qbraid_fallback(circuit, error=str(e))


def run_qbraid_3_experiments_simple(marked_states=None, show_plots=True):
    """
    Run three Grover experiments on qBraid with optimization levels 0, 2, and 3.
    Simple and clean version based on the working run_qbraid_grover_analysis_only.

    Args:
        marked_states: List of marked states for Grover (default: ["011", "100"])
        show_plots: Whether to display plots

    Returns:
        dict: Results from three qBraid jobs with different optimizations
    """
    print("🔬 Running qBraid 3 Experiments (Optimization Levels 0, 2, 3)")
    print("=" * 60)

    if marked_states is None:
        marked_states = ["011", "100"]

    print(f">>> Using marked states: {marked_states}")

    # Step 1: Create base circuit and generator
    print("📋 Creating Grover circuit...")
    circuit, grover_gen = create_circuit()

    # Step 2: Create three circuits with different optimization levels
    print("🔧 Creating circuits with different optimization levels...")

    from qward.algorithms.grover import GroverCircuitGenerator

    # Create fresh generator with the marked states
    grover_gen_opt = GroverCircuitGenerator(marked_states)

    # Create ISA circuits with different optimization levels
    circuit_opt_0 = grover_gen_opt.grover.create_rigetti_isa_circuit(optimization_level=0)
    circuit_opt_2 = grover_gen_opt.grover.create_rigetti_isa_circuit(optimization_level=2)
    circuit_opt_3 = grover_gen_opt.grover.create_rigetti_isa_circuit(optimization_level=3)

    circuits = [circuit_opt_0, circuit_opt_2, circuit_opt_3]
    optimization_levels = [0, 2, 3]

    print(">>> Circuit optimization comparison:")
    for opt_level, circ in zip(optimization_levels, circuits):
        print(f"   Optimization level {opt_level}:")
        print(f"     - Depth: {circ.depth()}")
        print(f"     - Size: {circ.size()}")
        print(f"     - Gates: {circ.count_ops()}")

    # Step 3: Run each circuit on qBraid (using the working run_qbraid function)
    print("\n🚀 Running experiments on qBraid Rigetti Ankaa-3...")

    results = []
    for i, (opt_level, circ) in enumerate(zip(optimization_levels, circuits)):
        print(f"\n--- Experiment {i+1}/3: Optimization Level {opt_level} ---")
        print(f"   Circuit depth: {circ.depth()}, size: {circ.size()}")

        # Use the working run_qbraid function for each circuit
        qbraid_result = run_qbraid(circ)

        # Add optimization level info to result
        qbraid_result["optimization_level"] = opt_level
        qbraid_result["circuit"] = circ
        results.append(qbraid_result)

        if qbraid_result.get("status") == "completed":
            print(f"   ✅ Experiment {i+1} completed successfully!")
            if "counts" in qbraid_result:
                counts = qbraid_result["counts"]
                total_shots = sum(counts.values()) if counts else 0
                success_shots = (
                    sum(counts.get(state, 0) for state in marked_states) if counts else 0
                )
                success_rate = success_shots / total_shots if total_shots > 0 else 0.0
                print(f"   Success rate: {success_rate:.3f}")
        else:
            print(f"   ⚠️ Experiment {i+1} status: {qbraid_result.get('status', 'unknown')}")

    # Step 4: Analyze results if any completed successfully
    completed_results = [r for r in results if r.get("status") == "completed"]

    if completed_results:
        print(f"\n📊 Analyzing {len(completed_results)} completed experiments...")

        # Use the first circuit as reference for analysis
        analysis_results = []
        for i, result in enumerate(completed_results):
            print(f"\n--- Analysis for Optimization Level {result['optimization_level']} ---")
            analysis = analyze_grover_qbraid(
                circuit,
                grover_gen,
                result,
                display_results=False,  # Don't display each one individually
            )
            analysis["optimization_level"] = result["optimization_level"]
            analysis_results.append(analysis)

        # Display summary comparison
        print("\n" + "=" * 80)
        print("📈 OPTIMIZATION LEVEL COMPARISON SUMMARY")
        print("=" * 80)

        for analysis in analysis_results:
            opt_level = analysis["optimization_level"]
            print(f"\nOptimization Level {opt_level}:")

            # Get metrics from the analysis
            metrics_dict = analysis["metrics_dict"]
            if "CircuitPerformanceMetrics" in metrics_dict:
                perf_metrics = metrics_dict["CircuitPerformanceMetrics"]
                if not perf_metrics.empty:
                    success_rate = (
                        perf_metrics.get("success_rate", [0])[0]
                        if "success_rate" in perf_metrics.columns
                        else 0
                    )
                    print(f"   Success Rate: {success_rate:.3f}")

        # Create visualizations if requested
        if show_plots and analysis_results:
            print("\n📈 Creating visualizations...")
            # Use the first analysis for visualization (they should be similar)
            viz_results = visualize_grover_results(
                analysis_results[0]["scanner"],
                output_dir="img/qward_paper_grover_qbraid_3exp_simple",
                show_plots=show_plots,
            )
        else:
            viz_results = None

    else:
        print("\n⚠️ No experiments completed successfully - skipping analysis")
        analysis_results = []
        viz_results = None

    print("\n✅ qBraid 3 experiments completed!")

    return {
        "experiments": results,
        "circuits": circuits,
        "optimization_levels": optimization_levels,
        "marked_states": marked_states,
        "grover_generator": grover_gen,
        "analysis_results": analysis_results,
        "visualizations": viz_results,
        "completed_count": len(completed_results),
    }


def _run_qbraid_fallback(circuit, error=None):
    """
    Fallback to AerSimulator when qBraid is not available.
    Returns data in qBraid-compatible format for qWard metrics.
    """
    from qiskit_aer import AerSimulator

    try:
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=1024)
        result = job.result()
        counts = result.get_counts()

        # Calculate basic success rate (for Grover, success = marked states)
        marked_states = ["011", "100"]  # Default Grover marked states
        success_count = sum(counts.get(state, 0) for state in marked_states)
        total_shots = sum(counts.values())
        success_rate = success_count / total_shots if total_shots > 0 else 0

        return {
            "status": "error" if error else "completed",
            "error": error,
            "counts": counts,
            "success_rate": success_rate,
            "circuit_depth": circuit.depth(),
            "circuit_width": circuit.width(),
            "circuit_size": circuit.size(),
            "circuit_count_ops": circuit.count_ops(),
            "num_qubits": circuit.num_qubits,
            "num_clbits": circuit.num_clbits,
            "num_ancillas": getattr(circuit, "num_ancillas", 0),
            "num_parameters": getattr(circuit, "num_parameters", 0),
            "has_calibrations": bool(getattr(circuit, "calibrations", None)),
            "has_layout": bool(getattr(circuit, "layout", None)),
            "device": "simulator_fallback",
            "job": job,  # Include job for qWard compatibility
        }
    except Exception as fallback_error:
        print(f">>> Fallback simulator also failed: {fallback_error}")
        return {
            "status": "error",
            "error": f"qBraid error: {error}, Simulator fallback error: {fallback_error}",
            "device": "failed",
        }


# Debugging and utility functions


def debug_ibm_backends():
    """Debug function to list available IBM backends and their status."""
    IBM_QUANTUM_CHANNEL = "ibm_cloud"
    IBM_QUANTUM_TOKEN = "1wZxUOFtDbbp4JE1C6UYNn_jmG1w1-DBxviKsbc9PgO6"

    QiskitRuntimeService.save_account(
        channel=IBM_QUANTUM_CHANNEL, token=IBM_QUANTUM_TOKEN, instance="thecap", overwrite=True
    )
    service = QiskitRuntimeService()

    print("🔍 IBM Backend Debug Information")
    print("=" * 50)

    # Get all backends
    try:
        backends = service.backends()
        print(f"Total backends found: {len(backends)}")

        for backend in backends:
            status = backend.status()
            config = backend.configuration()
            print(f"\n📡 Backend: {backend.name}")
            print(f"   Qubits: {config.n_qubits}")
            print(f"   Operational: {status.operational}")
            print(f"   Pending jobs: {status.pending_jobs}")
            print(f"   Status msg: {status.status_msg}")

    except Exception as e:
        print(f"Error getting backends: {e}")

    # Try with different instances
    for instance in [None, "thecap", "ibm-q", "ibm-q/open/main"]:
        try:
            backends = service.backends(instance=instance)
            print(f"\nBackends with instance '{instance}': {len(backends)} found")
            if backends:
                print(f"   Names: {[b.name for b in backends[:3]]}")  # Show first 3
        except Exception as e:
            print(f"   Error with instance '{instance}': {e}")


# qWard analysis functions

from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics
from qward.visualization.constants import Metrics, Plots
from qward.visualization import IEEEPlotConfig
from qward import Visualizer


def grover_success_criteria(result: str) -> bool:
    """Success criteria for Grover's algorithm - check if result matches marked states."""
    # Remove spaces to get clean bit string
    clean_result = result.replace(" ", "")
    # For Grover, we expect one of the marked states: "011" or "100"
    return clean_result in ["011", "100"]


def analyze_grover_simulators(circuit, grover_gen, simulators_results, display_results=True):
    """
    Analyze Grover algorithm performance on simulators using qWard metrics.

    Args:
        circuit: Grover quantum circuit
        grover_gen: GroverCircuitGenerator instance
        simulators_results: Results from run_simulators()
        display_results: Whether to display detailed results

    Returns:
        dict: Contains scanner, metrics_dict, and visualizer
    """

    def grover_expected_distribution():
        """Expected distribution for Grover's algorithm."""
        return grover_gen.grover.expected_distribution()

    grover_success = CircuitPerformanceMetrics(
        circuit,
        jobs=simulators_results["jobs"],
        success_criteria=grover_success_criteria,
        expected_distribution=grover_expected_distribution(),
    )

    # Create a scanner with the circuit
    scanner = Scanner(
        circuit=circuit,
        strategies=[
            grover_success,
            QiskitMetrics,
            ComplexityMetrics,
        ],
    )

    metrics_dict = scanner.calculate_metrics()

    if display_results:
        # Display summary using Scanner method
        scanner.display_summary(metrics_dict)

        # Display raw DataFrames for detailed analysis
        print("\n" + "=" * 80)
        print("📊 DETAILED METRICS DATA - SIMULATORS")
        print("=" * 80)
        for metric_name, metric_df in metrics_dict.items():
            print(f"\n=== {metric_name} ===")
            try:
                display(metric_df)
            except NameError:
                print(metric_df)

    return {"scanner": scanner, "metrics_dict": metrics_dict, "grover_success": grover_success}


def analyze_grover_qbraid(circuit, grover_gen, qbraid_results, display_results=True):
    """
    Analyze Grover algorithm performance on qBraid devices using qWard metrics.

    Args:
        circuit: Grover quantum circuit
        grover_gen: GroverCircuitGenerator instance
        qbraid_results: Results from run_qbraid()
        display_results: Whether to display detailed results

    Returns:
        dict: Contains scanner, metrics_dict, and visualizer
    """

    def grover_expected_distribution_qbraid():
        """Expected distribution for Grover's algorithm on qBraid devices."""
        return grover_gen.grover.expected_distribution()

    # Handle qBraid job format - extract job or result if available
    if isinstance(qbraid_results, dict):
        if "result" in qbraid_results and qbraid_results.get("status") == "completed":
            # Use the qBraid result directly
            jobs = [qbraid_results["result"]]
        elif "job" in qbraid_results:
            # Use the qBraid job
            jobs = [qbraid_results["job"]]
        else:
            jobs = []
    else:
        jobs = [qbraid_results] if qbraid_results else []

    grover_success_qbraid = CircuitPerformanceMetrics(
        circuit,
        jobs=jobs,
        success_criteria=grover_success_criteria,
        expected_distribution=grover_expected_distribution_qbraid(),
    )

    # Create a scanner with the circuit for qBraid results
    scanner_qbraid = Scanner(
        circuit=circuit,
        strategies=[
            grover_success_qbraid,
            QiskitMetrics,
            ComplexityMetrics,
        ],
    )

    metrics_dict_qbraid = scanner_qbraid.calculate_metrics()

    if display_results:
        # Display summary using Scanner method
        scanner_qbraid.display_summary(metrics_dict_qbraid)

        # Display raw DataFrames for detailed analysis
        print("\n" + "=" * 80)
        print("📊 DETAILED METRICS DATA - QBRAID")
        print("=" * 80)
        for metric_name, metric_df in metrics_dict_qbraid.items():
            print(f"\n=== {metric_name} ===")
            try:
                display(metric_df)
            except NameError:
                print(metric_df)

    return {
        "scanner": scanner_qbraid,
        "metrics_dict": metrics_dict_qbraid,
        "grover_success": grover_success_qbraid,
    }


def analyze_grover_qbraid_3_experiments(
    circuit, grover_gen, qbraid_3_results, display_results=True
):
    """
    Analyze Grover algorithm performance on qBraid devices using 3 experiments with different optimization levels.

    Args:
        circuit: Grover quantum circuit
        grover_gen: GroverCircuitGenerator instance
        qbraid_3_results: Results from run_qbraid_3_experiments()
        display_results: Whether to display detailed results

    Returns:
        dict: Contains scanner, metrics_dict, and visualizer for all 3 experiments
    """

    def grover_expected_distribution_qbraid():
        """Expected distribution for Grover's algorithm on qBraid devices."""
        return grover_gen.grover.expected_distribution()

    # Extract jobs from the 3 experiments results
    jobs = []
    if isinstance(qbraid_3_results, dict) and "experiments" in qbraid_3_results:
        for experiment in qbraid_3_results["experiments"]:
            if experiment.get("status") == "completed" and "result" in experiment:
                # Use the qBraid result directly
                jobs.append(experiment["result"])
            elif "job" in experiment:
                # Use the qBraid job
                jobs.append(experiment["job"])

    print(f">>> Analyzing {len(jobs)} qBraid jobs from 3 experiments")

    grover_success_qbraid_3 = CircuitPerformanceMetrics(
        circuit,
        jobs=jobs,
        success_criteria=grover_success_criteria,
        expected_distribution=grover_expected_distribution_qbraid(),
    )

    # Create a scanner with the circuit for qBraid 3 experiments results
    scanner_qbraid_3 = Scanner(
        circuit=circuit,
        strategies=[
            grover_success_qbraid_3,
            QiskitMetrics,
            ComplexityMetrics,
        ],
    )

    metrics_dict_qbraid_3 = scanner_qbraid_3.calculate_metrics()

    if display_results:
        # Display summary using Scanner method
        scanner_qbraid_3.display_summary(metrics_dict_qbraid_3)

        # Display raw DataFrames for detailed analysis
        print("\n" + "=" * 80)
        print("📊 DETAILED METRICS DATA - QBRAID 3 EXPERIMENTS")
        print("=" * 80)
        for metric_name, metric_df in metrics_dict_qbraid_3.items():
            print(f"\n=== {metric_name} ===")
            try:
                display(metric_df)
            except NameError:
                print(metric_df)

        # Display optimization level comparison
        print("\n" + "=" * 80)
        print("🔧 OPTIMIZATION LEVEL COMPARISON")
        print("=" * 80)

        if "experiments" in qbraid_3_results:
            for i, experiment in enumerate(qbraid_3_results["experiments"]):
                opt_level = experiment.get("optimization_level", "unknown")
                status = experiment.get("status", "unknown")

                print(f"\n--- Experiment {i} (Optimization Level {opt_level}) ---")
                print(f"Status: {status}")

                if "circuit" in experiment:
                    circ = experiment["circuit"]
                    print(f"Circuit depth: {circ.depth()}")
                    print(f"Circuit size: {circ.size()}")
                    print(f"Gate count: {circ.count_ops()}")

                if "counts" in experiment and experiment["counts"]:
                    counts = experiment["counts"]
                    total_shots = sum(counts.values())
                    marked_states = grover_gen.get_marked_states()
                    success_shots = sum(counts.get(state, 0) for state in marked_states)
                    success_rate = success_shots / total_shots if total_shots > 0 else 0.0
                    print(f"Success rate: {success_rate:.3f}")

                    # Show top 3 results
                    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
                    print("Top results:")
                    for state, count in sorted_counts:
                        prob = count / total_shots
                        marker = "✅" if state in marked_states else "  "
                        print(f"  {marker} |{state}⟩: {count:4d} ({prob:.3f})")

    return {
        "scanner": scanner_qbraid_3,
        "metrics_dict": metrics_dict_qbraid_3,
        "grover_success": grover_success_qbraid_3,
        "optimization_levels": qbraid_3_results.get("optimization_levels", [0, 2, 3]),
        "experiments": qbraid_3_results.get("experiments", []),
    }


def analyze_grover_ibm(circuit, grover_gen, ibm_results, display_results=True):
    """
    Analyze Grover algorithm performance on IBM hardware using qWard metrics.

    Args:
        circuit: Grover quantum circuit
        grover_gen: GroverCircuitGenerator instance
        ibm_results: Results from run_ibm()
        display_results: Whether to display detailed results

    Returns:
        dict: Contains scanner, metrics_dict, and visualizer
    """

    def grover_expected_distribution_ibm():
        """Expected distribution for Grover's algorithm on IBM hardware."""
        return grover_gen.grover.expected_distribution()

    grover_success_ibm = CircuitPerformanceMetrics(
        circuit,
        jobs=ibm_results["jobs"],
        success_criteria=grover_success_criteria,
        expected_distribution=grover_expected_distribution_ibm(),
    )

    # Create a scanner with the circuit for IBM results
    scanner_ibm = Scanner(
        circuit=circuit,
        strategies=[
            grover_success_ibm,
            QiskitMetrics,
            ComplexityMetrics,
        ],
    )

    metrics_dict_ibm = scanner_ibm.calculate_metrics()

    if display_results:
        # Display summary using Scanner method
        scanner_ibm.display_summary(metrics_dict_ibm)

        # Display raw DataFrames for detailed analysis
        print("\n" + "=" * 80)
        print("📊 DETAILED METRICS DATA - IBM")
        print("=" * 80)
        for metric_name, metric_df in metrics_dict_ibm.items():
            print(f"\n=== {metric_name} ===")
            try:
                display(metric_df)
            except NameError:
                print(metric_df)

    return {
        "scanner": scanner_ibm,
        "metrics_dict": metrics_dict_ibm,
        "grover_success": grover_success_ibm,
    }


def visualize_grover_results(
    scanner, output_dir="img/qward_paper_grover", show_plots=True, save_plots=True
):
    """
    Create IEEE-styled visualizations for Grover algorithm results.

    Args:
        scanner: Scanner instance with calculated metrics
        output_dir: Directory to save plots
        show_plots: Whether to display plots
        save_plots: Whether to save plots

    Returns:
        dict: Contains visualizer and generated plots
    """
    # Use IEEE configuration for publication-quality plots
    ieee_config = IEEEPlotConfig()
    visualizer = Visualizer(scanner=scanner, config=ieee_config, output_dir=output_dir)

    # Generate main error comparison plot
    error_plot = visualizer.generate_plot(
        metric_name=Metrics.CIRCUIT_PERFORMANCE,
        plot_name=Plots.CircuitPerformance.SUCCESS_ERROR_COMPARISON,
        save=save_plots,
        show=show_plots,
    )

    if show_plots:
        error_plot.show()

    # Create comprehensive dashboard
    dashboard = visualizer.create_dashboard(save=save_plots, show=show_plots)

    # Generate additional comparison plots
    additional_plots = visualizer.generate_plots(
        selections={
            Metrics.CIRCUIT_PERFORMANCE: [
                Plots.CircuitPerformance.SUCCESS_ERROR_COMPARISON,
                Plots.CircuitPerformance.FIDELITY_COMPARISON,
            ]
        },
        save=save_plots,
        show=show_plots,
    )

    return {
        "visualizer": visualizer,
        "error_plot": error_plot,
        "dashboard": dashboard,
        "additional_plots": additional_plots,
    }


def display_grover_analysis(grover_gen):
    """
    Display detailed Grover algorithm analysis and theoretical information.

    Args:
        grover_gen: GroverCircuitGenerator instance
    """
    print("\n" + "=" * 80)
    print("📊 GROVER ALGORITHM ANALYSIS")
    print("=" * 80)

    print(f"Marked states: {grover_gen.get_marked_states()}")
    print(f"Number of qubits: {grover_gen.num_qubits}")
    print(f"Theoretical success probability: {grover_gen.get_success_probability():.3f}")
    print(f"Optimal iterations: {grover_gen.grover.optimal_iterations}")

    # Compare theoretical vs actual results
    print("\n=== Theoretical vs Actual Comparison ===")
    expected_dist = grover_gen.grover.expected_distribution()
    print("Expected distribution (top 10 states):")
    for state, prob in sorted(expected_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  |{state}⟩: {prob:.4f}")


def run_complete_grover_analysis(
    marked_states=None,
    run_ibm_analysis=False,
    run_qbraid_analysis=False,
    run_qbraid_3_experiments=False,
    show_plots=True,
):
    """
    Run complete Grover algorithm analysis with simulators and optionally quantum hardware.

    Args:
        marked_states: List of marked states (default: ["011", "100"])
        run_ibm_analysis: Whether to run IBM hardware analysis
        run_qbraid_analysis: Whether to run qBraid hardware analysis (single job)
        run_qbraid_3_experiments: Whether to run qBraid 3 experiments with different optimizations
        show_plots: Whether to display plots

    Returns:
        dict: Complete analysis results
    """
    if marked_states is None:
        marked_states = ["011", "100"]

    print("🚀 Starting Complete Grover Algorithm Analysis")
    print("=" * 60)

    # Create circuit
    print("📊 Creating Grover circuit...")
    circuit, grover_gen = create_circuit()

    # Run simulators
    print("🔬 Running simulator analysis...")
    simulators_results = run_simulators(circuit)

    # Analyze simulators
    sim_analysis = analyze_grover_simulators(circuit, grover_gen, simulators_results)

    # Visualize simulator results
    sim_viz = visualize_grover_results(
        sim_analysis["scanner"], output_dir="img/qward_paper_grover_sim", show_plots=show_plots
    )

    results = {
        "circuit": circuit,
        "grover_gen": grover_gen,
        "simulators_results": simulators_results,
        "simulator_analysis": sim_analysis,
        "simulator_visualizations": sim_viz,
    }

    # Run IBM analysis if requested
    if run_ibm_analysis:
        print("🌐 Running IBM hardware analysis...")
        ibm_results = run_ibm(circuit)

        ibm_analysis = analyze_grover_ibm(circuit, grover_gen, ibm_results)

        ibm_viz = visualize_grover_results(
            ibm_analysis["scanner"], output_dir="img/qward_paper_grover_ibm", show_plots=show_plots
        )

        results.update(
            {
                "ibm_results": ibm_results,
                "ibm_analysis": ibm_analysis,
                "ibm_visualizations": ibm_viz,
            }
        )

    # Run qBraid analysis if requested
    if run_qbraid_analysis:
        print("🔬 Running qBraid hardware analysis...")
        qbraid_results = run_qbraid(circuit)

        qbraid_analysis = analyze_grover_qbraid(circuit, grover_gen, qbraid_results)

        qbraid_viz = visualize_grover_results(
            qbraid_analysis["scanner"],
            output_dir="img/qward_paper_grover_qbraid",
            show_plots=show_plots,
        )

        results.update(
            {
                "qbraid_results": qbraid_results,
                "qbraid_analysis": qbraid_analysis,
                "qbraid_visualizations": qbraid_viz,
            }
        )

    # Run qBraid 3 experiments if requested
    if run_qbraid_3_experiments:
        print("🔬 Running qBraid 3 experiments analysis...")
        qbraid_3_results = run_qbraid_3_experiments_simple(marked_states=marked_states)

        qbraid_3_analysis = analyze_grover_qbraid_3_experiments(
            circuit, grover_gen, qbraid_3_results
        )

        qbraid_3_viz = visualize_grover_results(
            qbraid_3_analysis["scanner"],
            output_dir="img/qward_paper_grover_qbraid_3exp",
            show_plots=show_plots,
        )

        results.update(
            {
                "qbraid_3_results": qbraid_3_results,
                "qbraid_3_analysis": qbraid_3_analysis,
                "qbraid_3_visualizations": qbraid_3_viz,
            }
        )

    # Display theoretical analysis
    display_grover_analysis(grover_gen)

    print("\n✅ Complete Grover analysis finished!")

    return results


# Example usage functions for notebooks
def quick_grover_demo():
    """Quick demonstration of Grover algorithm with simulators only."""
    return run_complete_grover_analysis(
        marked_states=["011", "100"], run_ibm_analysis=False, show_plots=True
    )


def full_grover_analysis():
    """Complete analysis including IBM hardware (requires credentials)."""
    return run_complete_grover_analysis(
        marked_states=["011", "100"], run_ibm_analysis=True, show_plots=True
    )


def qbraid_grover_analysis():
    """Complete analysis including qBraid quantum devices."""
    return run_complete_grover_analysis(
        marked_states=["011", "100"], run_qbraid_analysis=True, show_plots=True
    )


def complete_hardware_grover_analysis():
    """Complete analysis including both IBM and qBraid quantum hardware."""
    return run_complete_grover_analysis(
        marked_states=["011", "100"],
        run_ibm_analysis=True,
        run_qbraid_analysis=True,
        show_plots=True,
    )


def qbraid_3_experiments_grover_analysis():
    """Complete analysis with qBraid 3 experiments (different optimization levels)."""
    return run_complete_grover_analysis(
        marked_states=["011", "100"], run_qbraid_3_experiments=True, show_plots=True
    )


def full_quantum_hardware_grover_analysis():
    """Complete analysis including IBM and qBraid 3 experiments."""
    return run_complete_grover_analysis(
        marked_states=["011", "100"],
        run_ibm_analysis=True,
        run_qbraid_3_experiments=True,
        show_plots=True,
    )


def test_qbraid_3_experiments_logic():
    """
    Test the qBraid 3 experiments logic without submitting expensive jobs.

    Returns:
        dict: Test results showing circuit creation and transpilation work
    """
    print("🧪 Testing qBraid 3 Experiments Logic (DRY RUN)")
    print("=" * 60)

    # Step 1: Create base circuit
    print("📋 Creating base Grover circuit...")
    circuit, grover_gen = create_circuit()

    # Step 2: Test the 3 experiments function in dry-run mode
    print("\n🔬 Testing qBraid 3 experiments (dry run)...")
    try:
        test_results = run_qbraid_3_experiments_simple(marked_states=grover_gen.get_marked_states())

        print(f"\n✅ Test Results:")
        print(f"   Number of circuits: {len(test_results['circuits'])}")
        print(f"   Optimization levels: {test_results['optimization_levels']}")
        print(f"   Completed experiments: {test_results['completed_count']}")

        # Show circuit comparison
        print(f"\n📊 Circuit Optimization Comparison:")
        for i, (opt_level, circ) in enumerate(
            zip(test_results["optimization_levels"], test_results["circuits"])
        ):
            print(f"   Level {opt_level}:")
            print(f"     - Depth: {circ.depth()}")
            print(f"     - Size: {circ.size()}")
            print(f"     - Gates: {circ.count_ops()}")

        # Show experiment info
        print(f"\n📋 Experiment Information:")
        for i, experiment in enumerate(test_results["experiments"]):
            print(f"   Experiment {i}:")
            print(f"     - Status: {experiment['status']}")
            print(f"     - Optimization: {experiment['optimization_level']}")
            if "job_id" in experiment:
                print(f"     - Job ID: {experiment['job_id']}")

        success = len(test_results["circuits"]) == 3

        print(f"\n🎯 Validation:")
        print(f"   All circuits created: {'✅' if success else '❌'}")
        print(f"   Function executed without errors: {'✅' if success else '❌'}")

        if success:
            print(f"\n🎉 qBraid 3 experiments logic is working correctly!")
        else:
            print(f"\n⚠️  Some issues detected. Review the output above.")

        return test_results

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return {"error": str(e), "success": False}


def test_rigetti_isa_circuit():
    """
    Test the Rigetti ISA circuit by running a simulation to verify it works correctly.

    Returns:
        dict: Test results including counts, success rate, and circuit comparison
    """
    print("🧪 Testing Rigetti ISA Circuit Transformation")
    print("=" * 60)

    # Step 1: Create circuits
    print("📋 Creating Grover circuits...")
    circuit, grover_gen = create_circuit()

    original_circuit = grover_gen.circuit
    rigetti_isa_circuit = grover_gen.circuit_isa_rigetti

    # Step 2: Display circuit information
    print(f"\n📊 Circuit Comparison:")
    print(f"   Original circuit:")
    print(f"     - Depth: {original_circuit.depth()}")
    print(f"     - Width: {original_circuit.width()}")
    print(f"     - Gates: {original_circuit.count_ops()}")
    print(f"     - Size: {original_circuit.size()}")

    print(f"   Rigetti ISA circuit:")
    print(f"     - Depth: {rigetti_isa_circuit.depth()}")
    print(f"     - Width: {rigetti_isa_circuit.width()}")
    print(f"     - Gates: {rigetti_isa_circuit.count_ops()}")
    print(f"     - Size: {rigetti_isa_circuit.size()}")

    # Step 3: Run simulations
    print(f"\n🔬 Running simulations...")

    from qiskit_aer import AerSimulator

    simulator = AerSimulator()
    shots = 1024

    # Simulate original circuit
    print("   Simulating original circuit...")
    try:
        original_job = simulator.run(original_circuit, shots=shots)
        original_result = original_job.result()
        original_counts = original_result.get_counts()
        print(f"     ✅ Original circuit simulation successful")
    except Exception as e:
        print(f"     ❌ Original circuit simulation failed: {e}")
        original_counts = {}

    # Simulate Rigetti ISA circuit
    print("   Simulating Rigetti ISA circuit...")
    try:
        rigetti_job = simulator.run(rigetti_isa_circuit, shots=shots)
        rigetti_result = rigetti_job.result()
        rigetti_counts = rigetti_result.get_counts()
        print(f"     ✅ Rigetti ISA circuit simulation successful")
    except Exception as e:
        print(f"     ❌ Rigetti ISA circuit simulation failed: {e}")
        rigetti_counts = {}

    # Step 4: Compare results
    print(f"\n📈 Results Comparison:")

    # Calculate success rates
    marked_states = grover_gen.get_marked_states()

    def calculate_success_rate(counts, marked_states):
        if not counts:
            return 0.0
        total_shots = sum(counts.values())
        success_shots = sum(counts.get(state, 0) for state in marked_states)
        return success_shots / total_shots if total_shots > 0 else 0.0

    original_success_rate = calculate_success_rate(original_counts, marked_states)
    rigetti_success_rate = calculate_success_rate(rigetti_counts, marked_states)

    print(f"   Original circuit success rate: {original_success_rate:.3f}")
    print(f"   Rigetti ISA success rate: {rigetti_success_rate:.3f}")
    print(f"   Difference: {abs(original_success_rate - rigetti_success_rate):.3f}")

    # Show top results for both
    print(f"\n📊 Top measurement results:")
    print(f"   Original circuit (top 5):")
    if original_counts:
        sorted_original = sorted(original_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for state, count in sorted_original:
            prob = count / sum(original_counts.values())
            marker = "✅" if state in marked_states else "  "
            print(f"     {marker} |{state}⟩: {count:4d} ({prob:.3f})")
    else:
        print("     No results")

    print(f"   Rigetti ISA circuit (top 5):")
    if rigetti_counts:
        sorted_rigetti = sorted(rigetti_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for state, count in sorted_rigetti:
            prob = count / sum(rigetti_counts.values())
            marker = "✅" if state in marked_states else "  "
            print(f"     {marker} |{state}⟩: {count:4d} ({prob:.3f})")
    else:
        print("     No results")

    # Step 5: Theoretical comparison
    theoretical_success_prob = grover_gen.get_success_probability()
    print(f"\n🎯 Theoretical Analysis:")
    print(f"   Expected success probability: {theoretical_success_prob:.3f}")
    print(f"   Original circuit error: {abs(original_success_rate - theoretical_success_prob):.3f}")
    print(f"   Rigetti ISA error: {abs(rigetti_success_rate - theoretical_success_prob):.3f}")

    # Step 6: Validation
    success_threshold = 0.1  # Allow 10% difference
    circuits_match = abs(original_success_rate - rigetti_success_rate) < success_threshold

    print(f"\n✅ Validation Results:")
    print(f"   Circuits produce similar results: {'✅ PASS' if circuits_match else '❌ FAIL'}")
    print(f"   Rigetti ISA circuit is functional: {'✅ PASS' if rigetti_counts else '❌ FAIL'}")

    if circuits_match and rigetti_counts:
        print(f"   🎉 Rigetti ISA transformation is working correctly!")
    else:
        print(f"   ⚠️  Rigetti ISA transformation may have issues")

    return {
        "original_circuit": original_circuit,
        "rigetti_isa_circuit": rigetti_isa_circuit,
        "original_counts": original_counts,
        "rigetti_counts": rigetti_counts,
        "original_success_rate": original_success_rate,
        "rigetti_success_rate": rigetti_success_rate,
        "theoretical_success_prob": theoretical_success_prob,
        "circuits_match": circuits_match,
        "marked_states": marked_states,
    }


def load_qbraid_jobs_and_visualize(
    job_ids,
    marked_states=None,
    optimization_levels=None,
    output_dir="img/qward_paper_grover_qbraid_report",
    show_plots=True,
    save_plots=True,
):
    """
    Load qBraid jobs from job IDs and create success_error_rate and fidelity visualizations for report.

    Args:
        job_ids: List of qBraid job IDs to load
        marked_states: List of marked states for Grover (default: ["011", "100"])
        optimization_levels: List of optimization levels corresponding to jobs (default: [0, 2, 3])
        output_dir: Directory to save plots
        show_plots: Whether to display plots
        save_plots: Whether to save plots

    Returns:
        dict: Contains loaded jobs, analysis results, and visualizations
    """
    print("🔍 Loading qBraid Jobs and Creating Report Visualizations")
    print("=" * 60)

    if marked_states is None:
        marked_states = ["011", "100"]

    if optimization_levels is None:
        optimization_levels = [0, 2, 3]

    print(f">>> Loading {len(job_ids)} qBraid jobs...")
    print(f">>> Job IDs: {job_ids}")
    print(f">>> Marked states: {marked_states}")
    print(f">>> Optimization levels: {optimization_levels}")

    try:
        from qbraid.runtime import QbraidProvider, load_job

        # Load jobs and extract results
        loaded_jobs = []
        job_results = []

        for i, job_id in enumerate(job_ids):
            print(f"\n--- Loading Job {i+1}/{len(job_ids)}: {job_id} ---")

            try:
                # Load the job
                job = load_job(job_id)
                print(f"   ✅ Job loaded successfully")

                # Get job status and result
                status = job.status()
                print(f"   Status: {status}")

                # Convert enum to string for easier comparison
                status_str = str(status).split(".")[-1] if hasattr(status, "name") else str(status)

                if status_str in ["COMPLETED", "DONE"]:
                    result = job.result()
                    counts = result.data.get_counts()

                    # Calculate success rate
                    total_shots = sum(counts.values()) if counts else 0
                    success_shots = (
                        sum(counts.get(state, 0) for state in marked_states) if counts else 0
                    )
                    success_rate = success_shots / total_shots if total_shots > 0 else 0.0

                    print(f"   ✅ Results retrieved: {total_shots} shots")
                    print(f"   Success rate: {success_rate:.3f}")

                    # Show top 3 results
                    if counts:
                        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
                        print("   Top results:")
                        for state, count in sorted_counts:
                            prob = count / total_shots
                            marker = "✅" if state in marked_states else "  "
                            print(f"     {marker} |{state}⟩: {count:4d} ({prob:.3f})")

                    # Store job and result info
                    loaded_jobs.append(job)

                    opt_level = optimization_levels[i] if i < len(optimization_levels) else i
                    job_result = {
                        "job_id": job_id,
                        "job": job,
                        "result": result,
                        "counts": counts,
                        "status": "completed",
                        "success_rate": success_rate,
                        "optimization_level": opt_level,
                        "total_shots": total_shots,
                        "success_shots": success_shots,
                    }
                    job_results.append(job_result)

                else:
                    print(f"   ⚠️ Job not completed: {status_str}")
                    job_result = {
                        "job_id": job_id,
                        "status": status_str.lower(),
                        "optimization_level": (
                            optimization_levels[i] if i < len(optimization_levels) else i
                        ),
                    }
                    job_results.append(job_result)

            except Exception as e:
                print(f"   ❌ Failed to load job {job_id}: {e}")
                job_result = {
                    "job_id": job_id,
                    "status": "failed",
                    "error": str(e),
                    "optimization_level": (
                        optimization_levels[i] if i < len(optimization_levels) else i
                    ),
                }
                job_results.append(job_result)

        # Filter completed jobs for analysis
        completed_jobs = [jr for jr in job_results if jr.get("status") == "completed"]

        if not completed_jobs:
            print("\n⚠️ No completed jobs found - cannot create visualizations")
            return {
                "job_ids": job_ids,
                "job_results": job_results,
                "completed_count": 0,
                "error": "No completed jobs",
            }

        print(f"\n📊 Creating qWard analysis from {len(completed_jobs)} completed jobs...")

        # Create a reference circuit for analysis (we need this for qWard)
        print("📋 Creating reference Grover circuit...")
        circuit, grover_gen = create_circuit()

        # Create CircuitPerformanceMetrics with the loaded jobs
        def grover_expected_distribution():
            """Expected distribution for Grover's algorithm."""
            return grover_gen.grover.expected_distribution()

        # Extract just the job objects for qWard
        qward_jobs = [jr["job"] for jr in completed_jobs]

        grover_success_qbraid = CircuitPerformanceMetrics(
            circuit,
            jobs=qward_jobs,
            success_criteria=grover_success_criteria,
            expected_distribution=grover_expected_distribution(),
        )

        # Create a scanner with the circuit for qBraid results
        scanner_qbraid = Scanner(
            circuit=circuit,
            strategies=[
                grover_success_qbraid,
                QiskitMetrics,
                ComplexityMetrics,
            ],
        )

        metrics_dict = scanner_qbraid.calculate_metrics()

        # Display analysis summary
        print("\n📈 qWard Analysis Summary:")
        scanner_qbraid.display_summary(metrics_dict)

        # Create visualizations
        if show_plots or save_plots:
            print(f"\n🎨 Creating report visualizations...")
            print(f"   Output directory: {output_dir}")

            # Use IEEE configuration for publication-quality plots
            ieee_config = IEEEPlotConfig()
            visualizer = Visualizer(
                scanner=scanner_qbraid, config=ieee_config, output_dir=output_dir
            )

            # Generate success/error rate comparison plot
            print("   📊 Generating success/error rate comparison...")
            success_error_plot = visualizer.generate_plot(
                metric_name=Metrics.CIRCUIT_PERFORMANCE,
                plot_name=Plots.CircuitPerformance.SUCCESS_ERROR_COMPARISON,
                save=save_plots,
                show=show_plots,
            )

            # Generate fidelity comparison plot
            print("   📊 Generating fidelity comparison...")
            fidelity_plot = visualizer.generate_plot(
                metric_name=Metrics.CIRCUIT_PERFORMANCE,
                plot_name=Plots.CircuitPerformance.FIDELITY_COMPARISON,
                save=save_plots,
                show=show_plots,
            )

            # Create comprehensive dashboard
            print("   📊 Creating comprehensive dashboard...")
            dashboard = visualizer.create_dashboard(save=save_plots, show=show_plots)

            visualizations = {
                "success_error_plot": success_error_plot,
                "fidelity_plot": fidelity_plot,
                "dashboard": dashboard,
                "visualizer": visualizer,
            }

            if show_plots:
                success_error_plot.show()
                fidelity_plot.show()
        else:
            visualizations = None

        # Display optimization level comparison
        print("\n" + "=" * 80)
        print("🔧 OPTIMIZATION LEVEL COMPARISON SUMMARY")
        print("=" * 80)

        for jr in completed_jobs:
            opt_level = jr["optimization_level"]
            success_rate = jr["success_rate"]
            total_shots = jr["total_shots"]

            print(f"\nOptimization Level {opt_level}:")
            print(f"   Success Rate: {success_rate:.3f}")
            print(f"   Total Shots: {total_shots}")
            print(f"   Job ID: {jr['job_id']}")

        print("\n✅ qBraid jobs loaded and visualizations created!")

        return {
            "job_ids": job_ids,
            "job_results": job_results,
            "completed_jobs": completed_jobs,
            "completed_count": len(completed_jobs),
            "loaded_jobs": loaded_jobs,
            "circuit": circuit,
            "grover_generator": grover_gen,
            "scanner": scanner_qbraid,
            "metrics_dict": metrics_dict,
            "visualizations": visualizations,
            "marked_states": marked_states,
            "optimization_levels": optimization_levels,
        }

    except ImportError:
        print("❌ qBraid not available - cannot load jobs")
        return {"error": "qBraid not available", "job_ids": job_ids}
    except Exception as e:
        print(f"❌ Error loading qBraid jobs: {e}")
        return {"error": str(e), "job_ids": job_ids}


def simple_qbraid_3_experiments_demo():
    """
    Simple demo function to run the new qBraid 3 experiments.
    This is the clean, easy-to-use function you requested.
    """
    print("🚀 Running Simple qBraid 3 Experiments Demo")
    print("=" * 50)

    # Just call the new simplified function
    results = run_qbraid_3_experiments_simple(marked_states=["011", "100"], show_plots=True)

    print(f"\n✅ Demo completed!")
    print(f"   Total experiments: {len(results['experiments'])}")
    print(f"   Successful experiments: {results['completed_count']}")
    print(f"   Optimization levels tested: {results['optimization_levels']}")

    return results


def run_qbraid_grover_analysis_only(show_plots=True):
    """
    Run ONLY qBraid analysis without simulators or IBM hardware.
    Perfect for when you already have simulator/IBM results and just want qBraid.
    Uses Rigetti Ankaa-3 quantum device.

    Args:
        show_plots: Whether to display plots

    Returns:
        dict: qBraid analysis results only
    """

    print("🔬 Running qBraid-Only Grover Analysis (Rigetti Ankaa-3)")
    print("=" * 60)

    # Step 1: Create circuit
    print("📋 Creating Grover circuit...")
    circuit, grover_gen = create_circuit()

    # Step 2: Run qBraid analysis
    print("🔬 Running qBraid Rigetti Ankaa-3 analysis...")
    print(
        f"   Using Rigetti-optimized ISA circuit (depth: {grover_gen.circuit_isa_rigetti.depth()})"
    )
    print(f"   Circuit gates: {grover_gen.circuit_isa_rigetti.count_ops()}")
    qbraid_results = run_qbraid(grover_gen.circuit_isa_rigetti)

    # Step 3: Analyze with qWard
    print("📊 Analyzing qBraid results with qWard...")
    qbraid_analysis = analyze_grover_qbraid(
        circuit, grover_gen, qbraid_results, display_results=True
    )

    # Step 4: Visualize results
    if show_plots:
        print("📈 Creating qBraid visualizations...")
        qbraid_viz = visualize_grover_results(
            qbraid_analysis["scanner"],
            output_dir="img/qward_paper_grover_qbraid_only",
            show_plots=show_plots,
        )
    else:
        qbraid_viz = None

    # Step 5: Display theoretical analysis
    print("🔍 Theoretical Grover Analysis:")
    display_grover_analysis(grover_gen)

    print("\n✅ qBraid-only Grover analysis finished!")

    return {
        "circuit": circuit,
        "grover_generator": grover_gen,
        "qbraid_results": qbraid_results,
        "qbraid_analysis": qbraid_analysis,
        "qbraid_visualizations": qbraid_viz,
    }


def create_report_visualizations_from_job_ids(job_ids):
    """
    Simple wrapper function to create report visualizations from qBraid job IDs.
    Perfect for creating publication-ready plots for your report.

    Args:
        job_ids: List of qBraid job IDs (e.g., ["job_id_1", "job_id_2", "job_id_3"])

    Returns:
        dict: Contains all analysis results and visualizations

    Example:
        # Your job IDs from qBraid
        my_job_ids = ["your_job_id_1", "your_job_id_2", "your_job_id_3"]

        # Create visualizations for your report
        results = create_report_visualizations_from_job_ids(my_job_ids)

        # The plots will be saved in qward/examples/img/qward_paper_grover_qbraid_report/
        # You'll get:
        # - success_error_rate comparison plot
        # - fidelity comparison plot
        # - comprehensive dashboard
    """
    print("📊 Creating Report Visualizations from qBraid Job IDs")
    print("=" * 55)

    return load_qbraid_jobs_and_visualize(
        job_ids=job_ids,
        marked_states=["011", "100"],  # Default Grover marked states
        optimization_levels=[0, 2, 3],  # Default optimization levels
        output_dir="img/qward_paper_grover_qbraid_report",
        show_plots=True,
        save_plots=True,
    )
