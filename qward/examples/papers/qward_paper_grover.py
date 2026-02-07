"""
QWARD Grover Algorithm Paper Example

This script demonstrates Grover's algorithm analysis using QWARD metrics
on various backends: simulators, IBM Quantum, and qBraid (Rigetti).

Simplified and consolidated version for paper experiments.
"""

import os
import time

from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    pauli_error,
    depolarizing_error,
)

from qward import Scanner, Visualizer
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics
from qward.visualization.constants import Metrics, Plots
from qward.visualization import IEEEPlotConfig
from qward.algorithms.grover import GroverCircuitGenerator

# Import display with fallback for non-notebook environments
try:
    from IPython.display import display
except ImportError:
    display = print


# =============================================================================
# Configuration
# =============================================================================

# Default marked states for Grover's algorithm
DEFAULT_MARKED_STATES = ["011", "100"]

# IBM credentials from environment variables (set these in your environment)
IBM_QUANTUM_CHANNEL = os.getenv("IBM_QUANTUM_CHANNEL", "ibm_cloud")
IBM_QUANTUM_TOKEN = os.getenv("IBM_QUANTUM_TOKEN", "")
IBM_QUANTUM_INSTANCE = os.getenv("IBM_QUANTUM_INSTANCE", "")


# =============================================================================
# Circuit Creation
# =============================================================================


def create_circuit(marked_states=None):
    """
    Create a Grover circuit with the specified marked states.

    Args:
        marked_states: List of marked states (default: ["011", "100"])

    Returns:
        tuple: (isa_circuit, grover_generator)
    """
    if marked_states is None:
        marked_states = DEFAULT_MARKED_STATES

    grover_gen = GroverCircuitGenerator(
        marked_states=marked_states,
        use_barriers=True,
    )

    circuit = grover_gen.circuit
    print(f">>> Grover Circuit created")
    print(f">>> Marked states: {grover_gen.get_marked_states()}")
    print(f">>> Theoretical success probability: {grover_gen.get_success_probability():.3f}")
    print(f">>> Circuit depth: {circuit.depth()}, qubits: {circuit.num_qubits}")

    display(circuit.draw(output="mpl"))

    return grover_gen.circuit_isa, grover_gen


def grover_success_criteria(result: str) -> bool:
    """Success criteria for Grover's algorithm."""
    clean_result = result.replace(" ", "")
    return clean_result in DEFAULT_MARKED_STATES


# =============================================================================
# Backend Runners
# =============================================================================


def run_simulators(circuit, shots=1024):
    """
    Run Grover circuit on simulators with various noise models.

    Args:
        circuit: Quantum circuit to run
        shots: Number of shots per simulation

    Returns:
        dict: Contains jobs, circuits, and backends
    """
    print("🔬 Running simulator experiments...")

    # No noise simulator
    no_noise = AerSimulator()

    # Depolarizing noise model
    noise_model1 = NoiseModel()
    depol_error = depolarizing_error(0.05, 1)
    noise_model1.add_all_qubit_quantum_error(
        depol_error, ["u1", "u2", "u3", "h", "x", "y", "z", "s", "t"]
    )
    depol_error_2q = depolarizing_error(0.1, 2)
    noise_model1.add_all_qubit_quantum_error(depol_error_2q, ["cx", "cy", "cz"])
    readout_error = ReadoutError([[0.9, 0.1], [0.1, 0.9]])
    noise_model1.add_all_qubit_readout_error(readout_error)
    depolarizing_sim = AerSimulator(noise_model=noise_model1)

    # Pauli noise model
    noise_model2 = NoiseModel()
    pauli_error_1q = pauli_error([("X", 0.05), ("Y", 0.05), ("Z", 0.05), ("I", 0.85)])
    noise_model2.add_all_qubit_quantum_error(
        pauli_error_1q, ["u1", "u2", "u3", "h", "x", "y", "z", "s", "t"]
    )
    pauli_error_2q = pauli_error([("XX", 0.05), ("YY", 0.05), ("ZZ", 0.05), ("II", 0.85)])
    noise_model2.add_all_qubit_quantum_error(pauli_error_2q, ["cx", "cy", "cz"])
    readout_error_5 = ReadoutError([[0.95, 0.05], [0.05, 0.95]])
    noise_model2.add_all_qubit_readout_error(readout_error_5)
    pauli_sim = AerSimulator(noise_model=noise_model2)

    jobs = [
        no_noise.run(circuit, shots=shots),
        depolarizing_sim.run(circuit, shots=shots),
        pauli_sim.run(circuit, shots=shots),
    ]

    print(f"   ✓ Submitted {len(jobs)} simulator jobs")

    return {
        "jobs": jobs,
        "circuits": [circuit] * 3,
        "backends": [no_noise, depolarizing_sim, pauli_sim],
        "backend_names": ["ideal", "depolarizing", "pauli"],
    }


def run_ibm(circuit):
    """
    Run Grover circuit on IBM Quantum hardware.

    Requires IBM_QUANTUM_TOKEN and IBM_QUANTUM_INSTANCE environment variables.

    Args:
        circuit: Quantum circuit to run

    Returns:
        dict: Contains batch_id, jobs, and circuits
    """
    if not IBM_QUANTUM_TOKEN:
        raise ValueError(
            "IBM_QUANTUM_TOKEN environment variable not set. "
            "Set it with: export IBM_QUANTUM_TOKEN='your_token'"
        )

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, Batch
    from qiskit.transpiler import generate_preset_pass_manager

    print("🌐 Running IBM Quantum experiments...")

    QiskitRuntimeService.save_account(
        channel=IBM_QUANTUM_CHANNEL,
        token=IBM_QUANTUM_TOKEN,
        instance=IBM_QUANTUM_INSTANCE,
        overwrite=True,
    )
    service = QiskitRuntimeService()

    # Get backend
    try:
        backend = service.least_busy(
            min_num_qubits=circuit.num_qubits,
            instance=IBM_QUANTUM_INSTANCE,
            operational=True,
        )
    except Exception as e:
        print(f">>> Error getting least_busy backend: {e}")
        backends = service.backends(instance=IBM_QUANTUM_INSTANCE)
        if not backends:
            raise ValueError(f"No backends available in instance '{IBM_QUANTUM_INSTANCE}'")
        operational = [b for b in backends if b.status().operational]
        backend = (
            min(operational, key=lambda b: b.status().pending_jobs) if operational else backends[0]
        )

    print(f">>> Backend: {backend.name}")
    print(f">>> Pending jobs: {backend.status().pending_jobs}")

    batch = Batch(backend=backend)
    print(f">>> Batch ID: {batch.session_id}")

    # Create pass managers for different optimization levels
    pm_0 = generate_preset_pass_manager(optimization_level=0, backend=backend)
    pm_2 = generate_preset_pass_manager(
        optimization_level=2, backend=backend, layout_method="sabre", routing_method="sabre"
    )
    pm_3 = generate_preset_pass_manager(
        optimization_level=3, backend=backend, layout_method="dense", routing_method="lookahead"
    )

    with batch:
        sampler = Sampler()
        isa_0 = pm_0.run(circuit)
        isa_2 = pm_2.run(circuit)
        isa_3 = pm_3.run(circuit)
        job_0 = sampler.run([isa_0])
        job_1 = sampler.run([isa_2])
        job_2 = sampler.run([isa_3])

    # Wait for completion
    print(">>> Waiting for jobs to complete...")
    start_time = time.time()
    while time.time() - start_time < 600:
        if all(j.status() in ["DONE", "CANCELLED", "ERROR"] for j in [job_0, job_1, job_2]):
            print(">>> All jobs completed!")
            break
        time.sleep(10)

    batch.close()

    return {
        "batch_id": batch.session_id,
        "jobs": [job_0, job_1, job_2],
        "circuits": [isa_0, isa_2, isa_3],
        "optimization_levels": [0, 2, 3],
    }


def run_qbraid(circuit, shots=1024):
    """
    Run Grover circuit on qBraid Rigetti Ankaa-3 device.

    Args:
        circuit: Quantum circuit to run
        shots: Number of shots

    Returns:
        dict: Job results
    """
    print("🔬 Running qBraid Rigetti experiment...")

    try:
        from qbraid.transpiler import transpile as qbraid_transpile
        from qbraid.runtime import QbraidProvider

        provider = QbraidProvider()
        device = provider.get_device(device_id="rigetti_ankaa_3")
        print(f">>> Device: {device} (Rigetti Ankaa-3)")

        # Transpile for Braket
        transpiled = qbraid_transpile(circuit, "braket")
        print(f">>> Circuit transpiled for Braket")

        # Submit job
        job = device.run(transpiled, shots=shots)
        print(f">>> Job submitted: {job.id}")

        # Wait for completion
        start_time = time.time()
        while time.time() - start_time < 300:
            status = job.status()
            status_str = str(status).split(".")[-1] if hasattr(status, "name") else str(status)

            if status_str in ["COMPLETED", "DONE"]:
                result = job.result()
                counts = result.data.get_counts()
                print(f">>> Job completed! Total shots: {sum(counts.values())}")
                return {
                    "status": "completed",
                    "job_id": job.id,
                    "job": job,
                    "result": result,
                    "counts": counts,
                    "device": device.id,
                }
            elif status_str in ["FAILED", "CANCELLED", "CANCELED"]:
                return {"status": "failed", "job_id": job.id, "error": f"Job {status_str}"}

            time.sleep(5)

        return {"status": "timeout", "job_id": job.id}

    except ImportError:
        print(">>> qBraid not available, using simulator fallback")
        return _run_simulator_fallback(circuit, shots)
    except Exception as e:
        print(f">>> qBraid error: {e}, using simulator fallback")
        return _run_simulator_fallback(circuit, shots, error=str(e))


def run_qbraid_3_experiments(marked_states=None, shots=1024):
    """
    Run three qBraid experiments with different optimization levels.

    Args:
        marked_states: List of marked states
        shots: Number of shots per experiment

    Returns:
        dict: Results from all three experiments
    """
    if marked_states is None:
        marked_states = DEFAULT_MARKED_STATES

    print("🔬 Running qBraid 3 Experiments (Optimization Levels 0, 2, 3)")
    print("=" * 60)

    # Create circuits
    _, grover_gen = create_circuit(marked_states)

    circuits = [
        grover_gen.grover.create_rigetti_isa_circuit(optimization_level=0),
        grover_gen.grover.create_rigetti_isa_circuit(optimization_level=2),
        grover_gen.grover.create_rigetti_isa_circuit(optimization_level=3),
    ]
    optimization_levels = [0, 2, 3]

    print(">>> Circuit comparison:")
    for opt, circ in zip(optimization_levels, circuits):
        print(f"   Level {opt}: depth={circ.depth()}, size={circ.size()}")

    # Run experiments
    results = []
    for i, (opt, circ) in enumerate(zip(optimization_levels, circuits)):
        print(f"\n--- Experiment {i+1}/3: Optimization Level {opt} ---")
        result = run_qbraid(circ, shots=shots)
        result["optimization_level"] = opt
        result["circuit"] = circ
        results.append(result)

        if result.get("status") == "completed" and "counts" in result:
            counts = result["counts"]
            total = sum(counts.values())
            success = sum(counts.get(s, 0) for s in marked_states)
            print(f"   Success rate: {success/total:.3f}")

    completed = [r for r in results if r.get("status") == "completed"]
    print(f"\n✅ Completed {len(completed)}/{len(results)} experiments")

    return {
        "experiments": results,
        "circuits": circuits,
        "optimization_levels": optimization_levels,
        "marked_states": marked_states,
        "grover_generator": grover_gen,
        "completed_count": len(completed),
    }


def _run_simulator_fallback(circuit, shots=1024, error=None):
    """Fallback to AerSimulator when hardware is unavailable."""
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()

    success_count = sum(counts.get(s, 0) for s in DEFAULT_MARKED_STATES)
    total = sum(counts.values())

    return {
        "status": "completed" if not error else "fallback",
        "error": error,
        "counts": counts,
        "success_rate": success_count / total if total > 0 else 0,
        "device": "simulator_fallback",
        "job": job,
    }


# =============================================================================
# Unified Analysis Function
# =============================================================================


def analyze_grover_results(
    circuit, grover_gen, results, backend_type="simulator", display_results=True
):
    """
    Unified analysis function for Grover results from any backend.

    Args:
        circuit: The Grover quantum circuit
        grover_gen: GroverCircuitGenerator instance
        results: Results dict from any backend runner
        backend_type: One of "simulator", "ibm", "qbraid", "qbraid_3exp"
        display_results: Whether to display detailed results

    Returns:
        dict: Contains scanner, metrics_dict, and analysis data
    """
    print(f"\n📊 Analyzing {backend_type.upper()} results...")

    # Extract jobs based on backend type
    if backend_type == "qbraid_3exp":
        jobs = []
        for exp in results.get("experiments", []):
            if exp.get("status") == "completed":
                if "result" in exp:
                    jobs.append(exp["result"])
                elif "job" in exp:
                    jobs.append(exp["job"])
    elif backend_type == "qbraid":
        if results.get("status") == "completed" and "result" in results:
            jobs = [results["result"]]
        elif "job" in results:
            jobs = [results["job"]]
        else:
            jobs = []
    else:
        jobs = results.get("jobs", [])

    if not jobs:
        print("   ⚠️ No jobs to analyze")
        return {"scanner": None, "metrics_dict": {}, "error": "No jobs"}

    # Create CircuitPerformance metrics
    circuit_performance = CircuitPerformanceMetrics(
        circuit,
        jobs=jobs,
        success_criteria=grover_success_criteria,
    )

    # Create scanner
    scanner = Scanner(
        circuit=circuit,
        strategies=[circuit_performance, QiskitMetrics, ComplexityMetrics],
    )

    metrics_dict = scanner.calculate_metrics()

    if display_results:
        scanner.display_summary(metrics_dict)

        print(f"\n{'='*80}")
        print(f"📊 DETAILED METRICS - {backend_type.upper()}")
        print("=" * 80)
        for name, df in metrics_dict.items():
            print(f"\n=== {name} ===")
            display(df)

    return {
        "scanner": scanner,
        "metrics_dict": metrics_dict,
        "circuit_performance": circuit_performance,
        "backend_type": backend_type,
    }


# =============================================================================
# Visualization
# =============================================================================


def visualize_grover_results(
    scanner, output_dir="img/qward_paper_grover", show_plots=True, save_plots=True
):
    """
    Create IEEE-styled visualizations for Grover results.

    Args:
        scanner: Scanner instance with calculated metrics
        output_dir: Directory to save plots
        show_plots: Whether to display plots
        save_plots: Whether to save plots

    Returns:
        dict: Contains visualizer and generated plots
    """
    print(f"\n🎨 Creating visualizations...")
    print(f"   Output: {output_dir}")

    ieee_config = IEEEPlotConfig()
    visualizer = Visualizer(scanner=scanner, config=ieee_config, output_dir=output_dir)

    # Success/error comparison
    error_plot = visualizer.generate_plot(
        metric_name=Metrics.CIRCUIT_PERFORMANCE,
        plot_name=Plots.CircuitPerformance.SUCCESS_ERROR_COMPARISON,
        save=save_plots,
        show=show_plots,
    )

    # Dashboard
    dashboard = visualizer.create_dashboard(save=save_plots, show=show_plots)

    print(f"   ✓ Visualizations created")

    return {
        "visualizer": visualizer,
        "error_plot": error_plot,
        "dashboard": dashboard,
    }


def display_grover_analysis(grover_gen):
    """Display theoretical Grover algorithm analysis."""
    print("\n" + "=" * 80)
    print("📊 GROVER ALGORITHM THEORETICAL ANALYSIS")
    print("=" * 80)

    print(f"Marked states: {grover_gen.get_marked_states()}")
    print(f"Number of qubits: {grover_gen.num_qubits}")
    print(f"Theoretical success probability: {grover_gen.get_success_probability():.3f}")
    print(f"Optimal iterations: {grover_gen.grover.optimal_iterations}")

    print("\nExpected distribution (top 5 states):")
    expected = grover_gen.grover.expected_distribution()
    for state, prob in sorted(expected.items(), key=lambda x: x[1], reverse=True)[:5]:
        marker = "✅" if state in grover_gen.get_marked_states() else "  "
        print(f"  {marker} |{state}⟩: {prob:.4f}")


# =============================================================================
# Main Entry Point - Unified Runner
# =============================================================================


def run_grover_experiment(
    backend="simulator",
    marked_states=None,
    show_plots=True,
    save_plots=True,
    output_dir=None,
):
    """
    Unified entry point for running Grover experiments.

    Args:
        backend: One of "simulator", "ibm", "qbraid", "qbraid_3exp", "all"
        marked_states: List of marked states (default: ["011", "100"])
        show_plots: Whether to display plots
        save_plots: Whether to save plots
        output_dir: Custom output directory (auto-generated if None)

    Returns:
        dict: Complete experiment results

    Examples:
        # Quick simulator test
        results = run_grover_experiment(backend="simulator")

        # qBraid single experiment
        results = run_grover_experiment(backend="qbraid")

        # qBraid 3 optimization levels
        results = run_grover_experiment(backend="qbraid_3exp")

        # IBM Quantum
        results = run_grover_experiment(backend="ibm")

        # All backends
        results = run_grover_experiment(backend="all")
    """
    if marked_states is None:
        marked_states = DEFAULT_MARKED_STATES

    if output_dir is None:
        output_dir = f"img/qward_paper_grover_{backend}"

    print("🚀 QWARD Grover Experiment")
    print("=" * 60)
    print(f"Backend: {backend}")
    print(f"Marked states: {marked_states}")
    print("=" * 60)

    # Create circuit
    circuit, grover_gen = create_circuit(marked_states)

    results = {
        "circuit": circuit,
        "grover_gen": grover_gen,
        "marked_states": marked_states,
        "backend": backend,
    }

    # Run experiments based on backend selection
    backends_to_run = []
    if backend == "all":
        backends_to_run = ["simulator", "qbraid", "ibm"]
    elif backend in ["simulator", "ibm", "qbraid", "qbraid_3exp"]:
        backends_to_run = [backend]
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Use: simulator, ibm, qbraid, qbraid_3exp, all"
        )

    for b in backends_to_run:
        try:
            if b == "simulator":
                print("\n" + "=" * 60)
                sim_results = run_simulators(circuit)
                analysis = analyze_grover_results(circuit, grover_gen, sim_results, "simulator")
                if analysis["scanner"]:
                    viz = visualize_grover_results(
                        analysis["scanner"], f"{output_dir}/simulator", show_plots, save_plots
                    )
                    results["simulator"] = {
                        "results": sim_results,
                        "analysis": analysis,
                        "viz": viz,
                    }

            elif b == "ibm":
                print("\n" + "=" * 60)
                ibm_results = run_ibm(circuit)
                analysis = analyze_grover_results(circuit, grover_gen, ibm_results, "ibm")
                if analysis["scanner"]:
                    viz = visualize_grover_results(
                        analysis["scanner"], f"{output_dir}/ibm", show_plots, save_plots
                    )
                    results["ibm"] = {"results": ibm_results, "analysis": analysis, "viz": viz}

            elif b == "qbraid":
                print("\n" + "=" * 60)
                # Use Rigetti-optimized circuit
                qbraid_results = run_qbraid(grover_gen.circuit_isa_rigetti)
                analysis = analyze_grover_results(circuit, grover_gen, qbraid_results, "qbraid")
                if analysis["scanner"]:
                    viz = visualize_grover_results(
                        analysis["scanner"], f"{output_dir}/qbraid", show_plots, save_plots
                    )
                    results["qbraid"] = {
                        "results": qbraid_results,
                        "analysis": analysis,
                        "viz": viz,
                    }

            elif b == "qbraid_3exp":
                print("\n" + "=" * 60)
                qbraid_3_results = run_qbraid_3_experiments(marked_states)
                analysis = analyze_grover_results(
                    circuit, grover_gen, qbraid_3_results, "qbraid_3exp"
                )
                if analysis["scanner"]:
                    viz = visualize_grover_results(
                        analysis["scanner"], f"{output_dir}/qbraid_3exp", show_plots, save_plots
                    )
                    results["qbraid_3exp"] = {
                        "results": qbraid_3_results,
                        "analysis": analysis,
                        "viz": viz,
                    }

        except Exception as e:
            print(f"   ❌ Error running {b}: {e}")
            results[b] = {"error": str(e)}

    # Display theoretical analysis
    display_grover_analysis(grover_gen)

    print("\n" + "=" * 60)
    print("✅ Grover experiment completed!")
    print("=" * 60)

    return results


# =============================================================================
# Utility Functions
# =============================================================================


def load_qbraid_jobs(
    job_ids, marked_states=None, optimization_levels=None, output_dir=None, show_plots=True
):
    """
    Load existing qBraid jobs and create visualizations.

    Args:
        job_ids: List of qBraid job IDs
        marked_states: List of marked states
        optimization_levels: List of optimization levels for each job
        output_dir: Output directory for plots
        show_plots: Whether to display plots

    Returns:
        dict: Analysis results and visualizations
    """
    if marked_states is None:
        marked_states = DEFAULT_MARKED_STATES
    if optimization_levels is None:
        optimization_levels = [0, 2, 3]
    if output_dir is None:
        output_dir = "img/qward_paper_grover_loaded"

    print("🔍 Loading qBraid Jobs")
    print("=" * 60)
    print(f"Job IDs: {job_ids}")

    try:
        from qbraid.runtime import load_job

        completed_jobs = []
        for i, job_id in enumerate(job_ids):
            print(f"\n--- Loading Job {i+1}/{len(job_ids)}: {job_id} ---")
            try:
                job = load_job(job_id)
                status = str(job.status()).split(".")[-1]

                if status in ["COMPLETED", "DONE"]:
                    result = job.result()
                    counts = result.data.get_counts()
                    total = sum(counts.values())
                    success = sum(counts.get(s, 0) for s in marked_states)

                    print(f"   ✅ Loaded: {total} shots, success rate: {success/total:.3f}")
                    completed_jobs.append(
                        {
                            "job_id": job_id,
                            "job": job,
                            "result": result,
                            "counts": counts,
                            "optimization_level": (
                                optimization_levels[i] if i < len(optimization_levels) else i
                            ),
                        }
                    )
                else:
                    print(f"   ⚠️ Job status: {status}")
            except Exception as e:
                print(f"   ❌ Failed: {e}")

        if not completed_jobs:
            return {"error": "No completed jobs", "job_ids": job_ids}

        # Create analysis
        circuit, grover_gen = create_circuit(marked_states)
        jobs_for_analysis = [j["job"] for j in completed_jobs]

        circuit_performance = CircuitPerformanceMetrics(
            circuit,
            jobs=jobs_for_analysis,
            success_criteria=grover_success_criteria,
        )

        scanner = Scanner(
            circuit=circuit,
            strategies=[circuit_performance, QiskitMetrics, ComplexityMetrics],
        )

        metrics_dict = scanner.calculate_metrics()
        scanner.display_summary(metrics_dict)

        # Visualize
        viz = visualize_grover_results(scanner, output_dir, show_plots, save_plots=True)

        return {
            "job_ids": job_ids,
            "completed_jobs": completed_jobs,
            "scanner": scanner,
            "metrics_dict": metrics_dict,
            "visualizations": viz,
        }

    except ImportError:
        return {"error": "qBraid not available", "job_ids": job_ids}


# =============================================================================
# Test Functions
# =============================================================================


def test_grover_setup():
    """
    Quick test to verify Grover setup is working.

    Returns:
        bool: True if test passes
    """
    print("🧪 Testing Grover Setup")
    print("=" * 40)

    try:
        # Test circuit creation
        circuit, grover_gen = create_circuit()
        print(f"   ✓ Circuit created: {circuit.num_qubits} qubits, depth {circuit.depth()}")

        # Test Rigetti ISA circuit
        rigetti_circuit = grover_gen.circuit_isa_rigetti
        print(f"   ✓ Rigetti ISA: depth {rigetti_circuit.depth()}")

        # Test simulator
        sim = AerSimulator()
        job = sim.run(circuit, shots=100)
        counts = job.result().get_counts()
        success = sum(counts.get(s, 0) for s in DEFAULT_MARKED_STATES)
        print(f"   ✓ Simulator: success rate {success/100:.2f}")

        # Test analysis
        results = {"jobs": [job]}
        analysis = analyze_grover_results(
            circuit, grover_gen, results, "simulator", display_results=False
        )
        print(f"   ✓ Analysis: {len(analysis['metrics_dict'])} metric types")

        print("\n✅ All tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


def test_rigetti_isa_circuit():
    """
    Test Rigetti ISA circuit transformation.

    Returns:
        dict: Test results with circuit comparison
    """
    print("🧪 Testing Rigetti ISA Circuit")
    print("=" * 40)

    _, grover_gen = create_circuit()

    original = grover_gen.circuit
    rigetti = grover_gen.circuit_isa_rigetti

    print(f"Original: depth={original.depth()}, size={original.size()}")
    print(f"Rigetti:  depth={rigetti.depth()}, size={rigetti.size()}")

    # Simulate both
    sim = AerSimulator()
    shots = 1024

    orig_counts = sim.run(original, shots=shots).result().get_counts()
    rig_counts = sim.run(rigetti, shots=shots).result().get_counts()

    orig_success = sum(orig_counts.get(s, 0) for s in DEFAULT_MARKED_STATES) / shots
    rig_success = sum(rig_counts.get(s, 0) for s in DEFAULT_MARKED_STATES) / shots

    print(f"\nOriginal success rate: {orig_success:.3f}")
    print(f"Rigetti success rate:  {rig_success:.3f}")
    print(f"Difference: {abs(orig_success - rig_success):.3f}")

    match = abs(orig_success - rig_success) < 0.1
    print(f"\n{'✅ PASS' if match else '❌ FAIL'}: Circuits produce similar results")

    return {
        "original_success": orig_success,
        "rigetti_success": rig_success,
        "match": match,
    }


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    # Quick test
    test_grover_setup()
