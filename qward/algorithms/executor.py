"""
Quantum Circuit Executor

This module provides a reusable executor class for running quantum circuits
on different backends including local simulators and qBraid cloud devices.
It integrates with qward's metrics system for comprehensive circuit analysis.
"""

import time
from typing import Dict, Any, Optional, Union, TYPE_CHECKING, cast

import pandas as pd
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    pauli_error,
    depolarizing_error,
)

from qward import Scanner, Visualizer
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics

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

    def _create_depolarizing_noise_model(self, noise_level: float) -> NoiseModel:
        """Create a noise model with depolarizing errors.

        Args:
            noise_level: Noise strength (0.0 to 1.0)

        Returns:
            NoiseModel with depolarizing errors on single and two-qubit gates
        """
        noise_model = NoiseModel()

        # Add depolarizing error to all single qubit gates
        depol_error_1q = depolarizing_error(noise_level, 1)
        noise_model.add_all_qubit_quantum_error(
            depol_error_1q, ["u1", "u2", "u3", "h", "x", "y", "z", "s", "t"]
        )

        # Add depolarizing error to all two qubit gates (higher error rate)
        depol_error_2q = depolarizing_error(noise_level * 2, 2)
        noise_model.add_all_qubit_quantum_error(depol_error_2q, ["cx", "cy", "cz"])

        # Add readout error
        readout_error = ReadoutError(
            [[1 - noise_level, noise_level], [noise_level, 1 - noise_level]]
        )
        noise_model.add_all_qubit_readout_error(readout_error)

        return noise_model

    def _create_pauli_noise_model(self, noise_level: float) -> NoiseModel:
        """Create a noise model with Pauli errors.

        Args:
            noise_level: Noise strength (0.0 to 1.0)

        Returns:
            NoiseModel with Pauli errors on single and two-qubit gates
        """
        noise_model = NoiseModel()

        # Add Pauli error to all single qubit gates
        # Distribute error equally among X, Y, Z with remaining probability for I
        pauli_prob = noise_level / 3
        identity_prob = 1 - noise_level
        pauli_error_1q = pauli_error(
            [("X", pauli_prob), ("Y", pauli_prob), ("Z", pauli_prob), ("I", identity_prob)]
        )
        noise_model.add_all_qubit_quantum_error(
            pauli_error_1q, ["u1", "u2", "u3", "h", "x", "y", "z", "s", "t"]
        )

        # Add Pauli error to all two qubit gates
        pauli_prob_2q = noise_level / 3
        identity_prob_2q = 1 - noise_level
        pauli_error_2q = pauli_error(
            [
                ("XX", pauli_prob_2q),
                ("YY", pauli_prob_2q),
                ("ZZ", pauli_prob_2q),
                ("II", identity_prob_2q),
            ]
        )
        noise_model.add_all_qubit_quantum_error(pauli_error_2q, ["cx", "cy", "cz"])

        # Add readout error (lower than gate errors)
        readout_prob = noise_level / 2
        readout_error = ReadoutError(
            [[1 - readout_prob, readout_prob], [readout_prob, 1 - readout_prob]]
        )
        noise_model.add_all_qubit_readout_error(readout_error)

        return noise_model

    def _create_mixed_noise_model(self, noise_level: float) -> NoiseModel:
        """Create a noise model combining depolarizing and Pauli errors.

        Args:
            noise_level: Noise strength (0.0 to 1.0)

        Returns:
            NoiseModel with mixed error types
        """
        noise_model = NoiseModel()

        # Use depolarizing errors for single qubit gates
        depol_error_1q = depolarizing_error(noise_level, 1)
        noise_model.add_all_qubit_quantum_error(
            depol_error_1q, ["u1", "u2", "u3", "h", "x", "y", "z", "s", "t"]
        )

        # Use Pauli errors for two qubit gates
        pauli_prob_2q = noise_level / 3
        identity_prob_2q = 1 - noise_level
        pauli_error_2q = pauli_error(
            [
                ("XX", pauli_prob_2q),
                ("YY", pauli_prob_2q),
                ("ZZ", pauli_prob_2q),
                ("II", identity_prob_2q),
            ]
        )
        noise_model.add_all_qubit_quantum_error(pauli_error_2q, ["cx", "cy", "cz"])

        # Add readout error
        readout_prob = noise_level
        readout_error = ReadoutError(
            [[1 - readout_prob, readout_prob], [readout_prob, 1 - readout_prob]]
        )
        noise_model.add_all_qubit_readout_error(readout_error)

        return noise_model

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
        elif noise_model_config == "depolarizing":
            return self._create_depolarizing_noise_model(noise_level)
        elif noise_model_config == "pauli":
            return self._create_pauli_noise_model(noise_level)
        elif noise_model_config == "mixed":
            return self._create_mixed_noise_model(noise_level)
        else:
            raise ValueError(f"Unknown noise model type: {noise_model_config}")

    def simulate(
        self,
        circuit: QuantumCircuit,
        *,
        success_criteria=None,
        show_results: bool = False,
        noise_model: Union[str, NoiseModel, None] = None,
        noise_level: float = 0.05,
    ) -> Dict[str, Any]:
        """Run circuit on local Aer simulator with comprehensive qward metrics.

        Args:
            circuit: The quantum circuit to simulate
            success_criteria: Optional function to define success criteria for CircuitPerformanceMetrics
            show_results: If True, display visualizations of the results using qward
            noise_model: Noise model configuration. Can be:
                - None: No noise (default)
                - 'depolarizing': Depolarizing error model
                - 'pauli': Pauli error model
                - 'mixed': Combined depolarizing and Pauli errors
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
        scanner = self._create_scanner(circuit, job=job, success_criteria=success_criteria)
        metrics_data = scanner.calculate_metrics()
        data["qward_metrics"] = metrics_data

        # Show visualizations if requested
        if show_results:
            self._display_results(scanner, data, noise_model, noise_level)

        return data

    def get_circuit_metrics(
        self, circuit: QuantumCircuit, job=None, success_criteria=None
    ) -> Dict[str, pd.DataFrame]:
        """Extract comprehensive metrics from a quantum circuit using qward.

        Args:
            circuit: The quantum circuit to analyze
            job: Optional job result for performance metrics
            success_criteria: Optional function to define success criteria

        Returns:
            Dictionary containing qward metric DataFrames
        """
        scanner = self._create_scanner(circuit, job=job, success_criteria=success_criteria)
        return scanner.calculate_metrics()

    def _create_scanner(self, circuit: QuantumCircuit, job=None, success_criteria=None) -> Scanner:
        """Create a Scanner with appropriate metrics strategies.

        Args:
            circuit: The quantum circuit to analyze
            job: Optional job result for performance metrics
            success_criteria: Optional function to define success criteria

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
                circuit=circuit, job=job, success_criteria=success_criteria
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
            visualizer = Visualizer(scanner=scanner, output_dir="qward/examples/img")

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
