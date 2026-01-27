"""
Quantum Circuit Executor

This module provides a reusable executor class for running quantum circuits
on different backends including local simulators, IBM Quantum QPUs, and qBraid
cloud devices. It integrates with qward's metrics system for comprehensive
circuit analysis.
"""

import time
from typing import Dict, Any, Optional, Union, List, TYPE_CHECKING, Callable
from dataclasses import dataclass, field

import pandas as pd
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from qward import Scanner, Visualizer
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics
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

# IBM Quantum imports (optional, will handle import errors gracefully)
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, Batch
    from qiskit.transpiler import generate_preset_pass_manager

    IBM_QUANTUM_AVAILABLE = True
except ImportError:
    IBM_QUANTUM_AVAILABLE = False


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


class QuantumCircuitExecutor:
    """Reusable class for executing quantum circuits on simulators and qBraid devices.

    This class provides a unified interface for running quantum circuits on different
    backends, including local simulators and qBraid cloud devices. It integrates with
    qward's comprehensive metrics and visualization system to provide detailed analysis
    and visual feedback of circuit execution results.

    Features:
    - Local Aer simulator execution with comprehensive qward metrics
    - qBraid cloud device execution (when available)
    - Automatic generation of QiskitMetrics, ComplexityMetrics, and CircuitPerformanceMetrics
    - Optional visualization display using qward's Visualizer
    - Structured results with pandas DataFrames
    - Support for various noise models (depolarizing, Pauli, readout errors)

    Args:
        save_statevector: Whether to save intermediate statevectors during simulation
        timeout: Timeout in seconds for qBraid job completion (default: 300)
        shots: Number of shots for quantum execution (default: 1024)
    """

    def __init__(self, save_statevector: bool = False, timeout: int = 300, shots: int = 1024):
        self.save_statevector = save_statevector
        self.timeout = timeout
        self.shots = shots

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
        expected_distribution=None,
        show_results: bool = False,
        noise_model: Union[str, NoiseModel, None] = None,
        noise_level: float = 0.05,
    ) -> Dict[str, Any]:
        """Run circuit on local Aer simulator with comprehensive qward metrics.

        Args:
            circuit: The quantum circuit to simulate
            success_criteria: Optional function to define success criteria for CircuitPerformanceMetrics
            expected_distribution: Optional expected probability distribution for fidelity calculation
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
            expected_distribution=expected_distribution,
        )
        metrics_data = scanner.calculate_metrics()
        data["qward_metrics"] = metrics_data

        # Show visualizations if requested
        if show_results:
            self._display_results(scanner, data, noise_model, noise_level)

        return data

    def get_circuit_metrics(
        self, circuit: QuantumCircuit, job=None, success_criteria=None, expected_distribution=None
    ) -> Dict[str, pd.DataFrame]:
        """Extract comprehensive metrics from a quantum circuit using qward.

        Args:
            circuit: The quantum circuit to analyze
            job: Optional job result for performance metrics
            success_criteria: Optional function to define success criteria
            expected_distribution: Optional expected probability distribution for fidelity calculation

        Returns:
            Dictionary containing qward metric DataFrames
        """
        scanner = self._create_scanner(
            circuit,
            job=job,
            success_criteria=success_criteria,
            expected_distribution=expected_distribution,
        )
        return scanner.calculate_metrics()

    def _create_scanner(
        self, circuit: QuantumCircuit, job=None, success_criteria=None, expected_distribution=None
    ) -> Scanner:
        """Create a Scanner with appropriate metrics strategies.

        Args:
            circuit: The quantum circuit to analyze
            job: Optional job result for performance metrics
            success_criteria: Optional function to define success criteria
            expected_distribution: Optional expected probability distribution for fidelity calculation

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
                expected_distribution=expected_distribution,
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
        expected_distribution: Optional[Dict[str, float]] = None,
        timeout: int = 600,
        poll_interval: int = 10,
        show_progress: bool = True,
        channel: Optional[str] = None,
        token: Optional[str] = None,
        instance: Optional[str] = None,
    ) -> IBMBatchResult:
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
            expected_distribution: Optional expected probability distribution for fidelity
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
                        expected_distribution=expected_distribution,
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
