"""
Quantum Circuit Executor

This module provides a reusable executor class for running quantum circuits
on different backends including local simulators, IBM Quantum QPUs, qBraid
cloud devices, and AWS Braket devices. It integrates with qward's metrics
system for comprehensive circuit analysis.
"""

# pylint: disable=too-many-lines

import os
import time
import importlib.util
from typing import Dict, Any, Optional, Union, List, TYPE_CHECKING, Callable
from dataclasses import dataclass, field

import pandas as pd
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from qward import Scanner, Visualizer
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics
from qward.metrics.differential_success_rate import (
    compute_dsr,
    compute_dsr_ratio,
    compute_dsr_log_ratio,
    compute_dsr_normalized_margin,
    compute_dsr_with_flags,
)
from .noise_generator import NoiseConfig, NoiseModelGenerator

if TYPE_CHECKING:
    try:
        from qbraid import QbraidJob
    except ImportError:
        pass

# qBraid imports (optional, will handle import errors gracefully)
try:
    from qbraid import QbraidProvider, qbraid_transpile

    QBRAID_AVAILABLE = True
except ImportError:
    QBRAID_AVAILABLE = False

# Transpiler (used by both IBM and AWS optimization paths)
from qiskit.transpiler import generate_preset_pass_manager

# IBM Quantum imports (optional, will handle import errors gracefully)
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, Batch

    IBM_QUANTUM_AVAILABLE = True
except ImportError:
    IBM_QUANTUM_AVAILABLE = False

# AWS Braket dependency check (avoid importing provider at module import time).
AWS_BRAKET_AVAILABLE = importlib.util.find_spec("qiskit_braket_provider") is not None


def _get_braket_provider_class():
    """Load BraketProvider lazily to avoid heavy import-time dependencies."""
    from qiskit_braket_provider import BraketProvider

    return BraketProvider


@dataclass
class IBMJobResult:
    """Result from an IBM Quantum job execution."""

    job_id: str
    optimization_level: int
    status: str
    counts: Optional[Dict[str, int]] = None
    circuit_depth: int = 0
    success_rate: Optional[float] = None
    error: Optional[str] = None
    raw_result: Any = None


@dataclass
class IBMBatchResult:
    """Result from an IBM Quantum batch execution."""

    batch_id: str
    backend_name: str
    status: str
    jobs: List[IBMJobResult] = field(default_factory=list)
    original_circuit_depth: int = 0
    qward_metrics: Optional[Dict[str, pd.DataFrame]] = None
    error: Optional[str] = None


@dataclass
class AWSJobResult:
    """Result from an AWS Braket job execution.

    The primary quality metric is the Differential Success Rate (DSR), computed
    from the observed counts and the caller-supplied ``expected_outcomes``.  All
    four DSR variants (Michelson, ratio, log-ratio, normalized margin) are stored
    so downstream analyses can pick the contrast measure that best fits their
    needs.  A ``peak_mismatch`` flag indicates whether the highest-count outcome
    was *not* among the expected outcomes.
    """

    job_id: str
    device_name: str
    status: str
    counts: Optional[Dict[str, int]] = None
    circuit_depth: int = 0
    original_circuit_depth: int = 0
    # DSR metrics (primary quality indicator)
    dsr_michelson: Optional[float] = None
    dsr_ratio: Optional[float] = None
    dsr_log_ratio: Optional[float] = None
    dsr_normalized_margin: Optional[float] = None
    peak_mismatch: Optional[bool] = None
    expected_outcomes: Optional[List[str]] = None
    error: Optional[str] = None
    raw_result: Any = None
    qward_metrics: Optional[Dict[str, pd.DataFrame]] = None
    region: str = "us-west-1"


class QuantumCircuitExecutor:
    """Reusable class for executing quantum circuits on simulators, IBM Quantum,
    qBraid, and AWS Braket devices.

    This class provides a unified interface for running quantum circuits on different
    backends, including local simulators, qBraid cloud devices, IBM Quantum hardware,
    and AWS Braket devices (via qiskit-braket-provider). It integrates with qward's
    comprehensive metrics and visualization system to provide detailed analysis
    and visual feedback of circuit execution results.

    Features:
    - Local Aer simulator execution with comprehensive qward metrics
    - qBraid cloud device execution (when available)
    - IBM Quantum hardware execution with batch mode
    - AWS Braket execution via qiskit-braket-provider (Rigetti, IonQ, etc.)
    - Automatic generation of QiskitMetrics, ComplexityMetrics, and CircuitPerformanceMetrics
    - Optional visualization display using qward's Visualizer
    - Structured results with pandas DataFrames
    - Support for various noise models (depolarizing, Pauli, readout errors)

    Args:
        save_statevector: Whether to save intermediate statevectors during simulation
        timeout: Timeout in seconds for job completion (default: 300)
        shots: Number of shots for quantum execution (default: 1024)
    """

    def __init__(self, save_statevector: bool = False, timeout: int = 300, shots: int = 1024):
        self.save_statevector = save_statevector
        self.timeout = timeout
        self.shots = shots

    @staticmethod
    def _configure_aws_environment(
        region: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ) -> None:
        """Configure AWS credentials and region for Braket provider calls.

        The explicit ``region`` argument takes precedence over pre-existing
        environment settings so callers can deterministically target a region.
        """
        if aws_access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        if aws_secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

        # Keep both variables aligned for boto3/braket compatibility.
        os.environ["AWS_DEFAULT_REGION"] = region
        os.environ["AWS_REGION"] = region

    def _get_noise_model(
        self, noise_model_config: Union[str, NoiseModel, None], noise_level: float = 0.05
    ) -> Optional[NoiseModel]:
        """Get the appropriate noise model based on configuration.

        Args:
            noise_model_config: Noise model configuration
            noise_level: Noise strength (0.0 to 1.0)

        Returns:
            NoiseModel instance or None if no noise is configured
        """
        if noise_model_config is None:
            return None
        elif isinstance(noise_model_config, NoiseModel):
            return noise_model_config
        elif isinstance(noise_model_config, NoiseConfig):
            return NoiseModelGenerator.create_from_config(noise_model_config)
        elif noise_model_config == "depolarizing":
            return NoiseModelGenerator.create_by_type("depolarizing", noise_level)
        elif noise_model_config == "pauli":
            return NoiseModelGenerator.create_by_type("pauli", noise_level)
        elif noise_model_config == "mixed":
            return NoiseModelGenerator.create_by_type("mixed", noise_level)
        elif noise_model_config in {"readout", "combined"}:
            return NoiseModelGenerator.create_by_type(noise_model_config, noise_level)
        else:
            raise ValueError(f"Unknown noise model type: {noise_model_config}")

    def simulate(
        self,
        circuit: QuantumCircuit,
        *,
        success_criteria=None,
        expected_outcomes: Optional[List[str]] = None,
        show_results: bool = False,
        noise_model: Union[str, NoiseModel, None] = None,
        noise_level: float = 0.05,
    ) -> Dict[str, Any]:
        """Run circuit on local Aer simulator with comprehensive qward metrics.

        Args:
            circuit: The quantum circuit to simulate
            success_criteria: Optional function to define success criteria for CircuitPerformanceMetrics
            expected_outcomes: Optional expected bitstrings for DSR metric calculations
            show_results: If True, display visualizations of the results using qward
            noise_model: Noise model configuration. Can be:
                - None: No noise (default)
                - 'depolarizing': Depolarizing error model
                - 'pauli': Pauli error model
                - 'mixed': Depolarizing (1Q) + Pauli (2Q) + readout
                - 'readout': Readout-only error model
                - 'combined': Depolarizing + readout error model
                - NoiseModel instance: Custom noise model
            noise_level: Noise strength (0.0 to 1.0, default: 0.05 for 5% error rate)

        Returns:
            Dictionary containing simulation results, qward metrics, and metadata
        """
        # Get noise model if configured
        noise_model_instance = self._get_noise_model(noise_model, noise_level)

        # Create simulator with appropriate configuration
        if self.save_statevector:
            simulator = AerSimulator(method="statevector", noise_model=noise_model_instance)
        else:
            simulator = AerSimulator(noise_model=noise_model_instance)

        # Run simulation
        job = simulator.run(circuit, shots=self.shots)
        result = job.result()
        counts = result.get_counts()

        # Basic simulation data
        data = {"counts": counts}

        if self.save_statevector:
            # Extract statevector data if available
            result_data = result.data()
            for key in result_data:
                if "statevector" in key or key.endswith("_payload") or key.endswith("_validation"):
                    data[key] = result_data[key]

        # Get comprehensive metrics using qward
        scanner = self._create_scanner(
            circuit,
            job=job,
            success_criteria=success_criteria,
            expected_outcomes=expected_outcomes,
        )
        metrics_data = scanner.calculate_metrics()
        data["qward_metrics"] = metrics_data

        # Show visualizations if requested
        if show_results:
            self._display_results(scanner, data, noise_model, noise_level)

        return data

    def get_circuit_metrics(
        self,
        circuit: QuantumCircuit,
        job=None,
        success_criteria=None,
        expected_outcomes: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Extract comprehensive metrics from a quantum circuit using qward.

        Args:
            circuit: The quantum circuit to analyze
            job: Optional job result for performance metrics
            success_criteria: Optional function to define success criteria
            expected_outcomes: Optional expected bitstrings for DSR metric calculations

        Returns:
            Dictionary containing qward metric DataFrames
        """
        scanner = self._create_scanner(
            circuit,
            job=job,
            success_criteria=success_criteria,
            expected_outcomes=expected_outcomes,
        )
        return scanner.calculate_metrics()

    def _create_scanner(
        self,
        circuit: QuantumCircuit,
        job=None,
        success_criteria=None,
        expected_outcomes: Optional[List[str]] = None,
    ) -> Scanner:
        """Create a Scanner with appropriate metrics strategies.

        Args:
            circuit: The quantum circuit to analyze
            job: Optional job result for performance metrics
            success_criteria: Optional function to define success criteria
            expected_outcomes: Optional expected bitstrings for DSR metric calculations

        Returns:
            Configured Scanner instance
        """
        # Create scanner with circuit
        scanner = Scanner(circuit=circuit)

        # Add basic metrics strategies
        scanner.add_strategy(QiskitMetrics(circuit))
        scanner.add_strategy(ComplexityMetrics(circuit))

        # Add performance metrics if job is available
        if job is not None:
            circuit_performance = CircuitPerformanceMetrics(
                circuit=circuit,
                job=job,
                success_criteria=success_criteria,
                expected_outcomes=expected_outcomes,
            )
            scanner.add_strategy(circuit_performance)

        return scanner

    def _display_results(
        self,
        scanner: Scanner,
        data: Dict[str, Any],
        noise_model: Union[str, NoiseModel, None] = None,
        noise_level: float = 0.05,
    ) -> None:
        """Display comprehensive results using qward's visualization system.

        Args:
            scanner: Configured Scanner instance with calculated metrics
            data: Simulation data including counts and metrics
            noise_model: Noise model configuration used in simulation
            noise_level: Noise level used in simulation
        """
        print("🚀 QUANTUM CIRCUIT EXECUTION RESULTS")
        print("=" * 60)

        # Display basic execution results
        print("\n📊 EXECUTION SUMMARY")
        print("-" * 40)
        counts = data["counts"]
        total_shots = sum(counts.values())
        print(f"Total shots: {total_shots}")
        print(f"Unique outcomes: {len(counts)}")
        print(
            f"Most frequent outcome: {max(counts, key=counts.get)} ({counts[max(counts, key=counts.get)]} shots)"
        )

        # Display noise model information
        if noise_model is not None:
            print(f"Noise model: {noise_model} (level: {noise_level})")
        else:
            print("Noise model: None (ideal simulation)")

        # Display top outcomes
        print("\nTop 5 outcomes:")
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for i, (outcome, count) in enumerate(sorted_counts[:5]):
            percentage = (count / total_shots) * 100
            print(f"  {i+1}. |{outcome}⟩: {count} shots ({percentage:.1f}%)")

        # Display qward metrics summary
        print("\n📋 QWARD METRICS SUMMARY")
        print("-" * 40)
        scanner.display_summary(data["qward_metrics"])

        # Create and display visualizations using qward's Visualizer
        print("\n🎨 GENERATING VISUALIZATIONS")
        print("-" * 40)

        try:
            # Create visualizer from scanner (following qward examples)
            visualizer = Visualizer(scanner=scanner, output_dir="img")

            # Print available metrics (like in the examples)
            visualizer.print_available_metrics()

            # Create dashboard for all available metrics (simple approach)
            print("\nCreating comprehensive dashboard...")
            dashboards = visualizer.create_dashboard(save=False, show=True)

            print(f"✅ Created {len(dashboards)} dashboards")
            for metric_name in dashboards:
                print(f"  - {metric_name} dashboard displayed")

        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            print("Continuing without visualizations...")

        print("\n✅ RESULTS DISPLAY COMPLETE")
        print("=" * 60)

    def run_qbraid(
        self, circuit: QuantumCircuit, device_id: Optional[str] = None, success_criteria=None
    ) -> Dict[str, Any]:
        """Run circuit on qBraid cloud device with comprehensive qward metrics.

        Args:
            circuit: The quantum circuit to execute
            device_id: Optional device ID (defaults to Rigetti device)
            success_criteria: Optional function to define success criteria for CircuitPerformanceMetrics

        Returns:
            Dictionary containing execution results, qward metrics, and metadata
        """
        if not QBRAID_AVAILABLE:
            raise ImportError("qBraid is not available. Install with: pip install qbraid")

        try:
            provider = QbraidProvider()
            device_id = device_id or "rigetti_aspen_m_3"  # Default Rigetti device
            qbraid_device = provider.get_device(device_id=device_id)

            print(f"Using device: {qbraid_device}")
            print(f"Original circuit depth: {circuit.depth()}")
            print(f"Original circuit width: {circuit.width()}")
            print(f"Device capabilities: {qbraid_device}")

            # Transpile circuit for the target device
            transpiled_circuit = qbraid_transpile(circuit, "braket")
            print(f"Transpiled circuit type: {type(transpiled_circuit)}")

            # Check device status
            print(f"Device status: {qbraid_device.status()}")

            # Submit job
            job_result = qbraid_device.run(transpiled_circuit, shots=self.shots)

            # Handle potential list return (qBraid type annotation issue)
            if isinstance(job_result, list):
                job = job_result[0]  # Take first job if list is returned
            else:
                job = job_result

            print(f"Job submitted with ID: {job.id}")

            job_info: Dict[str, Any] = {
                "status": "submitted",
                "job_id": job.id,
                "device": qbraid_device.id,
                "job": job,
            }

            # Wait for job completion
            print("Waiting for job to complete...")
            start_time = time.time()

            while time.time() - start_time < self.timeout:
                status = job.status()
                print(f"Job status: {status}")

                # Convert enum to string for easier comparison
                status_str = (
                    str(status).rsplit(".", maxsplit=1)[-1]
                    if hasattr(status, "name")
                    else str(status)
                )

                if status_str in ["COMPLETED", "DONE"]:
                    # Get qward metrics for the original circuit
                    metrics_data = self.get_circuit_metrics(
                        circuit, success_criteria=success_criteria
                    )

                    job_info.update(
                        {
                            "status": "completed",
                            "transpiled_circuit": transpiled_circuit,
                            "qward_metrics": metrics_data,
                        }
                    )
                    return job_info

                elif status_str in ["FAILED", "CANCELLED", "CANCELED"]:
                    print(f"Job failed with status: {status}")
                    error_msg = f"Job {status_str.lower()}"
                    try:
                        if hasattr(job, "metadata") and job.metadata():
                            error_msg += f": {job.metadata()}"
                    except Exception:  # pylint: disable=broad-except
                        pass

                    job_info.update({"status": "failed", "error": error_msg})
                    return job_info

                elif status_str in ["QUEUED", "RUNNING", "INITIALIZING"]:
                    # Continue waiting for these status values
                    pass
                else:
                    print(f"Unknown status: {status} ({status_str})")

                time.sleep(5)  # Wait 5 seconds before checking again

            # Timeout case
            print(f"Job timed out after {self.timeout} seconds")
            job_info.update({"status": "timeout"})
            return job_info

        except Exception as e:
            print(f"Error running on qBraid: {str(e)}")
            # Fallback to simulator
            job = AerSimulator().run(circuit, shots=self.shots)

            # Get qward metrics for fallback
            metrics_data = self.get_circuit_metrics(
                circuit, job=job, success_criteria=success_criteria
            )

            return {
                "status": "error",
                "error": str(e),
                "qward_metrics": metrics_data,
                "device": "simulator_fallback",
            }

    @staticmethod
    def configure_ibm_account(
        token: str,
        channel: str = "ibm_quantum",
        instance: Optional[str] = None,
        overwrite: bool = True,
    ) -> None:
        """Configure and save IBM Quantum account credentials.

        This method saves IBM Quantum credentials for future use. Call this once
        before using run_ibm() to set up authentication.

        Args:
            token: IBM Quantum API token
            channel: IBM Quantum channel ('ibm_quantum' or 'ibm_cloud')
            instance: Optional instance name (e.g., 'ibm-q/open/main')
            overwrite: Whether to overwrite existing saved credentials

        Example:
            >>> QuantumCircuitExecutor.configure_ibm_account(
            ...     token="your-api-token",
            ...     channel="ibm_quantum"
            ... )
        """
        if not IBM_QUANTUM_AVAILABLE:
            raise ImportError(
                "IBM Quantum Runtime is not available. "
                "Install with: pip install qiskit-ibm-runtime"
            )

        save_kwargs = {
            "token": token,
            "channel": channel,
            "overwrite": overwrite,
        }
        if instance:
            save_kwargs["instance"] = instance

        QiskitRuntimeService.save_account(**save_kwargs)
        print(f">>> IBM Quantum account configured (channel: {channel})")

    def run_ibm(
        self,
        circuit: QuantumCircuit,
        *,
        backend_name: Optional[str] = None,
        optimization_levels: Optional[List[int]] = None,
        success_criteria: Optional[Callable[[str], bool]] = None,
        expected_outcomes: Optional[List[str]] = None,
        timeout: int = 600,
        poll_interval: int = 10,
        show_progress: bool = True,
        channel: Optional[str] = None,
        token: Optional[str] = None,
        instance: Optional[str] = None,
    ) -> IBMBatchResult:
        # pylint: disable=too-many-branches
        """Run circuit on IBM Quantum hardware using Batch mode.

        This method executes a quantum circuit on real IBM Quantum hardware using
        the IBM Qiskit Runtime service. It supports multiple optimization levels
        for circuit transpilation and provides comprehensive metrics.

        Args:
            circuit: The quantum circuit to execute
            backend_name: Optional IBM backend name. If None, uses the least busy
                operational backend
            optimization_levels: List of optimization levels to test (default: [0, 2, 3])
            success_criteria: Optional function to define success criteria for metrics
            expected_outcomes: Optional expected bitstrings for DSR metric calculations
            timeout: Maximum time to wait for job completion in seconds (default: 600)
            poll_interval: Time between status checks in seconds (default: 10)
            show_progress: If True, print progress messages (default: True)
            channel: IBM Quantum channel ('ibm_quantum' or 'ibm_cloud'). If None,
                uses saved account
            token: IBM Quantum API token. If None, uses saved account
            instance: IBM Quantum instance (e.g., 'ibm-q/open/main')

        Returns:
            IBMBatchResult containing execution results, metrics, and metadata

        Raises:
            ImportError: If qiskit-ibm-runtime is not installed
            RuntimeError: If IBM Quantum service is not configured

        Example:
            >>> # Option 1: Use saved credentials
            >>> executor = QuantumCircuitExecutor(shots=1024)
            >>> result = executor.run_ibm(circuit, optimization_levels=[0, 2])

            >>> # Option 2: Pass credentials directly
            >>> result = executor.run_ibm(
            ...     circuit,
            ...     channel="ibm_quantum",
            ...     token="your-api-token",
            ... )

            >>> print(f"Batch ID: {result.batch_id}")
            >>> for job in result.jobs:
            ...     print(f"Opt {job.optimization_level}: {job.success_rate:.2%}")
        """
        if not IBM_QUANTUM_AVAILABLE:
            raise ImportError(
                "IBM Quantum Runtime is not available. "
                "Install with: pip install qiskit-ibm-runtime"
            )

        if optimization_levels is None:
            optimization_levels = [0, 2, 3]

        try:
            # Connect to IBM Quantum service
            service_kwargs = {}
            if channel:
                service_kwargs["channel"] = channel
            if token:
                service_kwargs["token"] = token
            if instance:
                service_kwargs["instance"] = instance

            service = QiskitRuntimeService(**service_kwargs)

            # Get backend
            if backend_name:
                backend = service.backend(backend_name)
            else:
                backend = service.least_busy(simulator=False, operational=True)

            if show_progress:
                print(f">>> Backend: {backend.name}")
                print(f">>> Pending jobs: {backend.status().pending_jobs}")
                print(f">>> Circuit qubits: {circuit.num_qubits}")
                print(f">>> Original depth: {circuit.depth()}")

            # Create batch session
            batch = Batch(backend=backend)

            if show_progress:
                print(f">>> Batch ID: {batch.session_id}")

            # Create pass managers for each optimization level
            pass_managers = {}
            # Use preset optimization levels with IBM's recommended defaults
            # Level 0: TrivialLayout + SabreSwap (no optimization)
            # Level 1: VF2LayoutPostLayout + SabreSwap (light optimization)
            # Level 2: SabreLayout + SabreSwap with deeper search (medium optimization)
            # Level 3: SabreLayout + KAK decomposition + unitarity passes (heavy optimization)
            for opt_level in optimization_levels:
                pass_managers[opt_level] = generate_preset_pass_manager(
                    optimization_level=min(opt_level, 3),  # Cap at level 3
                    backend=backend,
                )

            # Transpile circuits and submit jobs
            jobs = {}
            isa_circuits = {}

            with batch:
                sampler = Sampler()
                for opt_level in optimization_levels:
                    isa_circuits[opt_level] = pass_managers[opt_level].run(circuit)
                    jobs[opt_level] = sampler.run([isa_circuits[opt_level]], shots=self.shots)

                    if show_progress:
                        print(
                            f">>> Submitted job for opt_level={opt_level}, "
                            f"transpiled depth={isa_circuits[opt_level].depth()}"
                        )

            # Wait for job completion
            if show_progress:
                print(f">>> Waiting for jobs to complete (timeout: {timeout}s)...")

            start_time = time.time()
            all_completed = False

            while time.time() - start_time < timeout:
                # Check batch status
                batch_status = batch.status()

                if show_progress:
                    elapsed = int(time.time() - start_time)
                    print(f">>> [{elapsed}s] Batch status: {batch_status}")

                # Check individual job statuses
                job_statuses = []
                for opt_level, job in jobs.items():
                    status = str(job.status())
                    job_statuses.append(status)
                    if show_progress:
                        print(f"    - Opt level {opt_level}: {status}")

                # Check if all jobs are completed
                completed_states = {"DONE", "CANCELLED", "ERROR"}
                if all(
                    any(state in status for state in completed_states) for status in job_statuses
                ):
                    all_completed = True
                    if show_progress:
                        print(">>> All jobs completed!")
                    break

                time.sleep(poll_interval)

            # Close the batch
            batch.close()

            if show_progress:
                print(f">>> Batch {batch.session_id} closed")

            # Collect results
            job_results = []
            for opt_level in optimization_levels:
                job = jobs[opt_level]
                isa_circuit = isa_circuits[opt_level]

                job_result = IBMJobResult(
                    job_id=job.job_id(),
                    optimization_level=opt_level,
                    status=str(job.status()),
                    circuit_depth=isa_circuit.depth(),
                )

                try:
                    if "DONE" in str(job.status()):
                        result = job.result()
                        counts = self._extract_counts_from_sampler_result(result)
                        job_result.counts = counts
                        job_result.raw_result = result

                        # Calculate success rate if criteria provided
                        if success_criteria and counts:
                            total = sum(counts.values())
                            success = sum(c for k, c in counts.items() if success_criteria(k))
                            job_result.success_rate = success / total if total > 0 else 0.0

                except Exception as e:
                    job_result.error = str(e)

                job_results.append(job_result)

            # Get qward metrics for the original circuit
            qward_metrics = None
            try:
                # Use first successful job for metrics
                first_counts = next((jr.counts for jr in job_results if jr.counts), None)
                if first_counts:
                    qward_metrics = self.get_circuit_metrics(
                        circuit,
                        success_criteria=success_criteria,
                        expected_outcomes=expected_outcomes,
                    )
            except Exception as e:
                if show_progress:
                    print(f">>> Warning: Could not calculate qward metrics: {e}")

            # Build result
            result = IBMBatchResult(
                batch_id=batch.session_id,
                backend_name=backend.name,
                status="completed" if all_completed else "timeout",
                jobs=job_results,
                original_circuit_depth=circuit.depth(),
                qward_metrics=qward_metrics,
            )

            if show_progress:
                self._display_ibm_results(result)

            return result

        except Exception as e:
            error_msg = str(e)
            if show_progress:
                print(f">>> Error running on IBM Quantum: {error_msg}")

            return IBMBatchResult(
                batch_id="",
                backend_name=backend_name or "unknown",
                status="error",
                error=error_msg,
            )

    def run_aws(
        self,
        circuit: QuantumCircuit,
        *,
        device_id: str = "Ankaa-3",
        region: str = "us-west-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        expected_outcomes: Optional[List[str]] = None,
        timeout: int = 600,
        poll_interval: int = 10,
        show_progress: bool = True,
        wait_for_results: bool = True,
        max_local_gate_count: int = 3_333,
        optimization_level: Optional[int] = None,
    ) -> AWSJobResult:
        # pylint: disable=too-many-branches,too-many-statements
        """Run circuit on AWS Braket hardware via qiskit-braket-provider.

        This method executes a quantum circuit on AWS Braket devices (e.g. Rigetti
        Ankaa-3, IonQ, etc.) using the qiskit-braket-provider package. It handles
        barrier removal, endianness conversion, job polling, and integrates with
        QWARD metrics.

        The primary quality indicator is the **Differential Success Rate (DSR)**,
        computed from the observed hardware counts and the caller-supplied
        ``expected_outcomes``.  All four DSR variants (Michelson, ratio, log-ratio,
        normalized-margin) are returned in the result so downstream analyses can
        pick the contrast measure that best fits their needs.

        AWS credentials can be provided directly via parameters, read from
        environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY), or
        pre-configured via ``aws configure``.

        Args:
            circuit: The quantum circuit to execute
            device_id: AWS Braket device name (default: "Ankaa-3" for Rigetti)
            region: AWS region where the device is hosted (default: "us-west-1")
            aws_access_key_id: Optional AWS access key ID. If None, uses
                environment variable or pre-configured credentials
            aws_secret_access_key: Optional AWS secret access key. If None, uses
                environment variable or pre-configured credentials
            expected_outcomes: Expected bitstrings (marked states) for DSR
                calculation.  For example ``["000"]`` for a 3-qubit circuit whose
                correct answer is all-zeros.
            timeout: Maximum time to wait for job completion in seconds (default: 600)
            poll_interval: Time between status checks in seconds (default: 10)
            show_progress: If True, print progress messages (default: True)
            wait_for_results: If True, poll until job completes. If False, submit
                and return immediately with status "submitted" (default: True)
            max_local_gate_count: Maximum allowed *local* gate count for the
                prepared circuit (before Braket device compilation). Braket's
                on-device compiler typically inflates gate counts by 3-6x, so
                the default of 3,333 maps to ~20,000 device gates which is
                Ankaa-3's hard limit. Empirically, S6-1 (2,748 local gates)
                passes while S7-1 (5,742) fails at 36,442 device gates. Set
                to 0 to disable the check.
            optimization_level: If set (e.g. 3), transpile the circuit with
                the Braket backend using this Qiskit optimization level before
                submission. When None (default), only decomposition and barrier
                removal are applied so Braket's compiler does all optimization.
                Use 3 to mirror IBM behaviour where optimization level 3 helps.

        Returns:
            AWSJobResult containing execution results, DSR metrics, and metadata

        Raises:
            ImportError: If qiskit-braket-provider is not installed

        Example:
            >>> executor = QuantumCircuitExecutor(shots=1024)
            >>> result = executor.run_aws(
            ...     circuit,
            ...     device_id="Ankaa-3",
            ...     expected_outcomes=["000"],
            ... )
            >>> print(f"Job ID: {result.job_id}")
            >>> print(f"Counts: {result.counts}")
            >>> print(f"DSR (Michelson): {result.dsr_michelson:.4f}")

            >>> # Fire-and-forget (non-blocking)
            >>> result = executor.run_aws(circuit, wait_for_results=False)
            >>> print(f"Job ARN: {result.job_id}")  # retrieve later
        """
        if not AWS_BRAKET_AVAILABLE:
            raise ImportError(
                "qiskit-braket-provider is not available. "
                "Install with: pip install qiskit-braket-provider"
            )

        if show_progress:
            print(f">>> Circuit qubits: {circuit.num_qubits}")
            print(f">>> Original depth: {circuit.depth()}")

        try:
            # Configure AWS credentials/region for this execution.
            self._configure_aws_environment(
                region=region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )

            if show_progress:
                print(f">>> Setting up AWS Braket provider (region: {region})...")

            braket_provider_class = _get_braket_provider_class()
            provider = braket_provider_class()

            if show_progress:
                print(f">>> Getting device: {device_id}")

            aws_backend = provider.get_backend(device_id)

            # Prepare circuit: with optional Qiskit optimization or decompose-only
            if optimization_level is not None:
                if show_progress:
                    print(f">>> Preparing circuit with optimization_level={optimization_level}")
                circuit_clean = self._prepare_circuit_for_aws_with_optimization(
                    circuit, aws_backend, optimization_level
                )
            else:
                circuit_clean = self._prepare_circuit_for_aws(circuit)

            local_gate_count = circuit_clean.size()
            if show_progress:
                print(f">>> Prepared circuit depth: {circuit_clean.depth()}")
                print(f">>> Prepared circuit gates: {local_gate_count}")

            if local_gate_count > max_local_gate_count > 0:
                error_msg = (
                    f"Circuit has {local_gate_count} local gates which exceeds the "
                    f"pre-submission limit of {max_local_gate_count}. "
                    f"Braket device compilation typically inflates gate counts by "
                    f"3-6x, so ~{local_gate_count * 6} device gates are expected."
                )
                if show_progress:
                    print(f">>> BLOCKED: {error_msg}")
                return AWSJobResult(
                    job_id="",
                    device_name=device_id,
                    status="error",
                    error=error_msg,
                    original_circuit_depth=circuit.depth(),
                    circuit_depth=circuit_clean.depth(),
                    expected_outcomes=expected_outcomes,
                    region=region,
                )

            if show_progress:
                print(f">>> Device obtained: {aws_backend}")
                print(f">>> Submitting to {device_id} with {self.shots} shots...")

            # Submit job
            job = aws_backend.run(circuit_clean, shots=self.shots)
            job_arn = job.job_id()

            if show_progress:
                print(">>> Job submitted successfully!")
                print(f">>> Job ARN: {job_arn}")

            # If fire-and-forget mode, return immediately
            if not wait_for_results:
                # Still get pre-execution qward metrics
                qward_metrics = None
                try:
                    qward_metrics = self.get_circuit_metrics(
                        circuit,
                        expected_outcomes=expected_outcomes,
                    )
                except Exception as e:
                    if show_progress:
                        print(f">>> Warning: Could not calculate qward metrics: {e}")

                return AWSJobResult(
                    job_id=job_arn,
                    device_name=device_id,
                    status="submitted",
                    original_circuit_depth=circuit.depth(),
                    circuit_depth=circuit_clean.depth(),
                    expected_outcomes=expected_outcomes,
                    qward_metrics=qward_metrics,
                    region=region,
                )

            # Wait for job completion
            if show_progress:
                print(f">>> Waiting for job to complete (timeout: {timeout}s)...")

            start_time = time.time()

            while time.time() - start_time < timeout:
                status = job.status()
                status_str = str(status).rsplit(".", maxsplit=1)[-1].upper()

                if show_progress:
                    elapsed = int(time.time() - start_time)
                    print(f">>> [{elapsed}s] Job status: {status}")

                if status_str in ["DONE", "COMPLETED"]:
                    if show_progress:
                        print(">>> Job completed!")

                    # Extract counts (Braket big-endian → Qiskit little-endian)
                    counts = self._extract_counts_from_aws_result(job)

                    # Compute DSR from hardware counts
                    dsr_fields = self._compute_dsr_from_counts(
                        counts, expected_outcomes, show_progress
                    )

                    # Get qward metrics for the original circuit
                    qward_metrics = None
                    try:
                        qward_metrics = self.get_circuit_metrics(
                            circuit,
                            expected_outcomes=expected_outcomes,
                        )
                    except Exception as e:
                        if show_progress:
                            print(f">>> Warning: Could not calculate qward metrics: {e}")

                    result = AWSJobResult(
                        job_id=job_arn,
                        device_name=device_id,
                        status="completed",
                        counts=counts,
                        circuit_depth=circuit_clean.depth(),
                        original_circuit_depth=circuit.depth(),
                        expected_outcomes=expected_outcomes,
                        qward_metrics=qward_metrics,
                        region=region,
                        **dsr_fields,
                    )

                    if show_progress:
                        self._display_aws_results(result)

                    return result

                elif status_str in ["FAILED", "CANCELLED", "CANCELED"]:
                    error_msg = f"Job {status_str.lower()}"
                    if show_progress:
                        print(f">>> Job failed: {error_msg}")

                    return AWSJobResult(
                        job_id=job_arn,
                        device_name=device_id,
                        status="failed",
                        original_circuit_depth=circuit.depth(),
                        circuit_depth=circuit_clean.depth(),
                        expected_outcomes=expected_outcomes,
                        error=error_msg,
                        region=region,
                    )

                elif status_str in ["QUEUED", "RUNNING", "INITIALIZING", "CREATED"]:
                    pass  # Continue waiting
                else:
                    if show_progress:
                        print(f">>> Unknown status: {status} ({status_str})")

                time.sleep(poll_interval)

            # Timeout case
            if show_progress:
                print(f">>> Job timed out after {timeout} seconds")

            return AWSJobResult(
                job_id=job_arn,
                device_name=device_id,
                status="timeout",
                original_circuit_depth=circuit.depth(),
                circuit_depth=circuit_clean.depth(),
                expected_outcomes=expected_outcomes,
                region=region,
            )

        except Exception as e:
            error_msg = str(e)
            if show_progress:
                print(f">>> Error running on AWS Braket: {error_msg}")

            return AWSJobResult(
                job_id="",
                device_name=device_id,
                status="error",
                original_circuit_depth=circuit.depth(),
                error=error_msg,
                region=region,
            )

    def retrieve_aws_job(
        self,
        job_id: str,
        *,
        device_id: str = "Ankaa-3",
        region: str = "us-west-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        expected_outcomes: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> AWSJobResult:
        """Retrieve results from a previously submitted AWS Braket job.

        Use this method to get results from jobs submitted with
        ``wait_for_results=False``, or to re-fetch results using a saved job ARN.
        If ``expected_outcomes`` is provided, DSR is computed from the retrieved
        counts.

        Args:
            job_id: AWS Braket job ARN
                (e.g. "arn:aws:braket:us-west-1:ACCOUNT:quantum-task/UUID")
            device_id: Device name used for the original submission (default: "Ankaa-3")
            region: AWS region (default: "us-west-1")
            aws_access_key_id: Optional AWS access key ID
            aws_secret_access_key: Optional AWS secret access key
            expected_outcomes: Expected bitstrings (marked states) for DSR
                calculation
            show_progress: If True, print progress messages (default: True)

        Returns:
            AWSJobResult with counts, DSR metrics, and status

        Example:
            >>> result = executor.retrieve_aws_job(
            ...     "arn:aws:braket:us-west-1:123456:quantum-task/abc-def",
            ...     expected_outcomes=["000"],
            ... )
            >>> print(f"Counts: {result.counts}")
            >>> print(f"DSR: {result.dsr_michelson:.4f}")
        """
        if not AWS_BRAKET_AVAILABLE:
            raise ImportError(
                "qiskit-braket-provider is not available. "
                "Install with: pip install qiskit-braket-provider"
            )

        try:
            # Configure AWS credentials/region for retrieval.
            self._configure_aws_environment(
                region=region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )

            if show_progress:
                print(f">>> Retrieving AWS job: {job_id}")

            braket_provider_class = _get_braket_provider_class()
            provider = braket_provider_class()
            aws_backend = provider.get_backend(device_id)
            job = aws_backend.retrieve_job(job_id)

            status = job.status()
            status_str = str(status).rsplit(".", maxsplit=1)[-1].upper()

            if show_progress:
                print(f">>> Job status: {status}")

            if status_str in ["DONE", "COMPLETED"]:
                counts = self._extract_counts_from_aws_result(job)

                # Compute DSR from counts
                dsr_fields = self._compute_dsr_from_counts(counts, expected_outcomes, show_progress)

                result = AWSJobResult(
                    job_id=job_id,
                    device_name=device_id,
                    status="completed",
                    counts=counts,
                    expected_outcomes=expected_outcomes,
                    region=region,
                    **dsr_fields,
                )

                if show_progress:
                    self._display_aws_results(result)

                return result

            else:
                return AWSJobResult(
                    job_id=job_id,
                    device_name=device_id,
                    status=status_str.lower(),
                    expected_outcomes=expected_outcomes,
                    region=region,
                )

        except Exception as e:
            error_msg = str(e)
            if show_progress:
                print(f">>> Error retrieving AWS job: {error_msg}")

            return AWSJobResult(
                job_id=job_id,
                device_name=device_id,
                status="error",
                error=error_msg,
                region=region,
            )

    @staticmethod
    def _compute_dsr_from_counts(
        counts: Dict[str, int],
        expected_outcomes: Optional[List[str]],
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Compute all DSR variants from raw counts and expected outcomes.

        Returns a dict suitable for unpacking into an ``AWSJobResult`` constructor
        (keys match the DSR field names of the dataclass).

        Args:
            counts: Measurement outcome counts (little-endian / Qiskit convention)
            expected_outcomes: Expected bitstrings (marked states)
            show_progress: Whether to print warnings on failure

        Returns:
            Dict with keys ``dsr_michelson``, ``dsr_ratio``, ``dsr_log_ratio``,
            ``dsr_normalized_margin``, and ``peak_mismatch``.  All values are
            ``None`` when ``expected_outcomes`` is not provided or counts are empty.
        """
        empty: Dict[str, Any] = {
            "dsr_michelson": None,
            "dsr_ratio": None,
            "dsr_log_ratio": None,
            "dsr_normalized_margin": None,
            "peak_mismatch": None,
        }

        if not expected_outcomes or not counts:
            return empty

        try:
            dsr_michelson_val, peak_mismatch_val = compute_dsr_with_flags(counts, expected_outcomes)
            return {
                "dsr_michelson": float(dsr_michelson_val),
                "dsr_ratio": float(compute_dsr_ratio(counts, expected_outcomes)),
                "dsr_log_ratio": float(compute_dsr_log_ratio(counts, expected_outcomes)),
                "dsr_normalized_margin": float(
                    compute_dsr_normalized_margin(counts, expected_outcomes)
                ),
                "peak_mismatch": bool(peak_mismatch_val),
            }
        except Exception as e:
            if show_progress:
                print(f">>> Warning: Could not compute DSR: {e}")
            return empty

    @staticmethod
    def _remove_barriers(circuit: QuantumCircuit) -> QuantumCircuit:
        """Return a copy of the circuit with all Barrier instructions removed."""
        from qiskit.circuit import Barrier

        out = circuit.copy()
        out.data = [
            (gate, qubits, clbits)
            for gate, qubits, clbits in circuit.data
            if not isinstance(gate, Barrier)
        ]
        return out

    def _prepare_circuit_for_aws_with_optimization(
        self,
        circuit: QuantumCircuit,
        backend: Any,
        optimization_level: int,
    ) -> QuantumCircuit:
        """Prepare circuit for AWS with Qiskit transpiler optimization.

        Decomposes the circuit, runs the preset pass manager at the given
        optimization level targeting the Braket backend, then removes barriers.
        Use this to mirror IBM behaviour where e.g. optimization_level=3
        improves success on hardware.
        """
        circuit_decomposed = circuit.decompose(reps=10)
        circuit_decomposed.remove_final_measurements()
        circuit_decomposed.measure_all()

        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=min(optimization_level, 3),
        )
        circuit_opt = pm.run(circuit_decomposed)
        return self._remove_barriers(circuit_opt)

    @staticmethod
    def _prepare_circuit_for_aws(circuit: QuantumCircuit) -> QuantumCircuit:
        """Prepare a Qiskit circuit for AWS Braket submission.

        This method:
        1. **Decomposes** opaque/composed operators (e.g. ``grover_op.power(n)``,
           library meta-gates) into standard gates that the
           ``qiskit-braket-provider`` adapter can translate directly.
           Without this step the adapter converts opaque gates to a unitary
           matrix, which degrades results on real hardware.
        2. Removes barrier instructions (unsupported by AWS Braket).
        3. Normalises final measurements.

        We intentionally do **not** run a full transpiler pass (no layout,
        routing, or optimisation) so that qubit ordering is preserved and
        Braket's own device compiler can perform hardware-aware compilation.

        Args:
            circuit: The original Qiskit quantum circuit

        Returns:
            A new QuantumCircuit ready for AWS Braket submission
        """
        from qiskit.circuit import Barrier

        # Step 1: Recursively decompose opaque/composite gates until only
        # standard gates remain.  ``reps=10`` is generous enough for deeply
        # nested library gates (e.g. power -> grover_operator -> MCMT -> CCX).
        circuit_decomposed = circuit.decompose(reps=10)

        # Step 2: Normalise measurements
        circuit_decomposed.remove_final_measurements()
        circuit_decomposed.measure_all()

        # Step 3: Remove barriers (AWS Braket incompatible)
        circuit_clean = circuit_decomposed.copy()
        circuit_clean.data = [
            (gate, qubits, clbits)
            for gate, qubits, clbits in circuit_decomposed.data
            if not isinstance(gate, Barrier)
        ]

        return circuit_clean

    @staticmethod
    def _extract_counts_from_aws_result(job) -> Dict[str, int]:
        """Extract measurement counts from an AWS Braket job result.

        Fetches the raw Braket ``measurement_counts`` and reverses bitstrings
        from Braket convention (big-endian, leftmost = q0) to Qiskit
        convention (little-endian, rightmost = q0).

        Args:
            job: A BraketQuantumTask / qiskit-braket-provider job object.

        Returns:
            Dictionary of measurement outcomes in Qiskit (little-endian)
            convention and their counts.
        """
        try:
            # pylint: disable=protected-access
            braket_result = job._tasks[0].result()
            raw_counts = dict(braket_result.measurement_counts)
            # Calibration note (2026-02-17, Ankaa-3):
            # 3-qubit test circuit X(q0) produced dominant "001" only when
            # reversing Braket counts, so this conversion is required.
            # Reverse: Braket big-endian -> Qiskit little-endian
            return {k[::-1]: v for k, v in raw_counts.items()}
        except Exception:
            try:
                return dict(job.result().get_counts())
            except Exception as exc:
                print(f">>> Warning: Could not extract counts: {exc}")
                return {}

    def _display_aws_results(self, result: AWSJobResult) -> None:
        """Display AWS Braket execution results.

        Args:
            result: AWSJobResult to display
        """
        print("\n" + "=" * 60)
        print("AWS BRAKET EXECUTION RESULTS")
        print("=" * 60)
        print(f"Job ARN: {result.job_id}")
        print(f"Device: {result.device_name}")
        print(f"Region: {result.region}")
        print(f"Status: {result.status}")

        if result.original_circuit_depth:
            print(f"Original circuit depth: {result.original_circuit_depth}")
        if result.circuit_depth:
            print(f"Submitted circuit depth: {result.circuit_depth}")

        if result.counts:
            total = sum(result.counts.values())
            print(f"\nTotal shots: {total}")
            print(f"Unique outcomes: {len(result.counts)}")

            # Show top 5 outcomes
            sorted_counts = sorted(result.counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print("Top outcomes:")
            for outcome, count in sorted_counts:
                pct = (count / total) * 100
                print(f"  |{outcome}⟩: {count} ({pct:.1f}%)")

        # Display DSR metrics
        if result.dsr_michelson is not None:
            print(f"\n--- DSR Metrics (expected: {result.expected_outcomes}) ---")
            print(f"  DSR Michelson:         {result.dsr_michelson:.4f}")
            print(f"  DSR Ratio:             {result.dsr_ratio:.4f}")
            print(f"  DSR Log-Ratio:         {result.dsr_log_ratio:.4f}")
            print(f"  DSR Normalized Margin: {result.dsr_normalized_margin:.4f}")
            print(f"  Peak Mismatch:         {result.peak_mismatch}")

        if result.error:
            print(f"\nError: {result.error}")

        print("=" * 60)

    def _extract_counts_from_sampler_result(self, primitive_result) -> Dict[str, int]:
        """Extract measurement counts from SamplerV2 PrimitiveResult.

        Args:
            primitive_result: PrimitiveResult object from SamplerV2

        Returns:
            Dictionary of measurement outcomes and their counts
        """
        try:
            # Get the first pub result
            pub_result = primitive_result[0]

            # Try to find the classical register data
            # Common names are 'c', 'meas', or the measurement register name
            bit_array = None
            for attr in ["c", "meas", "cr"]:
                if hasattr(pub_result.data, attr):
                    bit_array = getattr(pub_result.data, attr)
                    break

            if bit_array is None:
                # Try to get the first available data attribute
                data_attrs = [a for a in dir(pub_result.data) if not a.startswith("_")]
                # Filter to only BitArray-like attributes
                for attr in data_attrs:
                    obj = getattr(pub_result.data, attr)
                    if hasattr(obj, "get_counts") or hasattr(obj, "num_shots"):
                        bit_array = obj
                        break

            if bit_array is None:
                return {}

            # Method 1: Use get_counts() if available (preferred)
            if hasattr(bit_array, "get_counts"):
                return dict(bit_array.get_counts())

            # Method 2: Use get_bitstrings() if available
            if hasattr(bit_array, "get_bitstrings"):
                from collections import Counter

                bitstrings = bit_array.get_bitstrings()
                return dict(Counter(bitstrings))

            # Method 3: Manual extraction from array
            from collections import Counter

            num_shots = bit_array.num_shots
            bit_strings = []

            # Try array-based access
            if hasattr(bit_array, "array"):
                arr = bit_array.array
                num_bits = bit_array.num_bits
                for row in arr:
                    # Convert numpy array row to bitstring
                    bit_string = "".join(str(int(b)) for b in row[:num_bits])
                    bit_strings.append(bit_string)
            else:
                # Fallback to index-based access
                for shot in range(num_shots):
                    try:
                        bits = bit_array[shot]
                        if hasattr(bits, "__iter__") and not isinstance(bits, str):
                            bit_string = "".join(str(int(b)) for b in bits)
                        else:
                            bit_string = str(bits)
                        bit_strings.append(bit_string)
                    except Exception:
                        continue

            return dict(Counter(bit_strings))

        except Exception as e:
            print(f">>> Warning: Could not extract counts: {e}")
            return {}

    def _display_ibm_results(self, result: IBMBatchResult) -> None:
        """Display IBM Quantum execution results.

        Args:
            result: IBMBatchResult to display
        """
        print("\n" + "=" * 60)
        print("IBM QUANTUM EXECUTION RESULTS")
        print("=" * 60)
        print(f"Batch ID: {result.batch_id}")
        print(f"Backend: {result.backend_name}")
        print(f"Status: {result.status}")
        print(f"Original circuit depth: {result.original_circuit_depth}")

        print("\n--- Job Results ---")
        for job in result.jobs:
            print(f"\nOptimization Level {job.optimization_level}:")
            print(f"  Job ID: {job.job_id}")
            print(f"  Status: {job.status}")
            print(f"  Transpiled depth: {job.circuit_depth}")

            if job.counts:
                total = sum(job.counts.values())
                print(f"  Total shots: {total}")

                # Show top 5 outcomes
                sorted_counts = sorted(job.counts.items(), key=lambda x: x[1], reverse=True)[:5]
                print("  Top outcomes:")
                for outcome, count in sorted_counts:
                    pct = (count / total) * 100
                    print(f"    |{outcome}⟩: {count} ({pct:.1f}%)")

                if job.success_rate is not None:
                    print(f"  Success rate: {job.success_rate:.2%}")

            if job.error:
                print(f"  Error: {job.error}")

        print("\n" + "=" * 60)
