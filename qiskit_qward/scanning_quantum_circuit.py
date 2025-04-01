from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
import os
from typing import List
from dotenv import load_dotenv
from .analysis import Analysis
import time

# For visualization support in notebooks - wrap in try/except for CI environments
try:
    from IPython.display import display
except ImportError:
    # Define a no-op display function for environments without IPython
    def display(*args, **kwargs):
        """Stub for IPython.display when not available."""
        pass


# Load environment variables
load_dotenv()

# Get IBM Quantum credentials from environment variables
IBM_QUANTUM_CHANNEL = os.getenv("IBM_QUANTUM_CHANNEL", "ibm_quantum")
IBM_QUANTUM_TOKEN = os.getenv("IBM_QUANTUM_TOKEN")


class ScanningQuantumCircuit(QuantumCircuit):
    """
    Base class for quantum circuit scanning and validation in the Qiskit ecosystem.
    Extends Qiskit's QuantumCircuit to provide standardized analysis and validation capabilities.
    """

    def __init__(
        self, num_qubits: int = 1, num_clbits: int = 1, use_barriers: bool = True, name: str = None
    ):
        """
        Initialize the scanning quantum circuit.

        Args:
            num_qubits (int): Number of qubits in the circuit
            num_clbits (int): Number of classical bits for measurement
            use_barriers (bool): Whether to include barriers in the circuit
            name (str): Name of the circuit
        """
        # Create quantum and classical registers
        qr = QuantumRegister(num_qubits, "q")
        cr = ClassicalRegister(num_clbits, "c")

        # Initialize the quantum circuit
        super().__init__(qr, cr, name=name)

        # Store additional attributes
        self.use_barriers = use_barriers
        self.analyzers: List[Analysis] = []

    def add_analyzer(self, analyzer: Analysis):
        """
        Add an analyzer to the scanner.

        Args:
            analyzer (Analysis): The analyzer to add
        """
        self.analyzers.append(analyzer)

    def _simulate(self, shots: int = 1024, show_histogram: bool = False):
        """
        Simulate the circuit using Qiskit's Aer simulator.

        Args:
            shots (int): Number of shots per simulation
            show_histogram (bool): Whether to display the measurement histogram

        Returns:
            dict: Dictionary containing simulation results
        """
        simulator = AerSimulator()
        result = simulator.run(self, shots=shots).result()
        counts = result.get_counts()

        if show_histogram:
            display(plot_histogram(counts))

        return {"counts": counts, "shots": shots}

    def run_simulation(
        self, show_histogram: bool = False, num_jobs: int = 1000, shots_per_job: int = 1024
    ):
        """
        Run a simulation of the circuit and return comprehensive results.

        Args:
            show_histogram (bool): Whether to display the measurement histogram
            num_jobs (int): Number of jobs to run (each job is an independent experiment)
            shots_per_job (int): Number of shots per job

        Returns:
            dict: Dictionary containing simulation results and circuit metrics
        """
        # Start timing
        

        start_time = time.time()
        start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))

        # Get first simulation results
        sim_results = self._simulate(shots=shots_per_job, show_histogram=show_histogram)

        # Basic circuit metrics
        circuit_metrics = {
            "depth": self.depth(),
            "width": self.width(),
            "size": self.size(),
            "count_ops": self.count_ops(),
            "num_qubits": self.num_qubits,
            "num_clbits": self.num_clbits,
            "num_ancillas": self.num_ancillas,
            "num_parameters": self.num_parameters,
            "has_calibrations": bool(self.calibrations),
            "has_layout": bool(self.layout),
        }

        # Calculate advanced complexity metrics
        complexity_metrics = self.calculate_complexity_metrics()
        
        # Calculate quantum volume estimate
        qv_estimate = self.estimate_quantum_volume()

        # Add first job results to analyzers
        for analyzer in self.analyzers:
            analyzer.add_results(
                {"counts": sim_results["counts"], "shots": shots_per_job, "job_id": 0}
            )

        # Run additional jobs if requested
        if num_jobs > 1:
            for job_id in range(1, num_jobs):
                job_result = self._simulate(shots=shots_per_job, show_histogram=False)
                # Add each job's results separately to analyzers
                for analyzer in self.analyzers:
                    analyzer.add_results(
                        {"counts": job_result["counts"], "shots": shots_per_job, "job_id": job_id}
                    )
                # Aggregate counts for final results
                sim_results["counts"] = {
                    k: sim_results["counts"].get(k, 0) + job_result["counts"].get(k, 0)
                    for k in set(sim_results["counts"]) | set(job_result["counts"])
                }

        # End timing
        end_time = time.time()
        end_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
        execution_time = end_time - start_time

        # Add timing information to results
        timing_info = {
            "start_time": start_timestamp,
            "end_time": end_timestamp,
            "execution_time": execution_time,
            "num_jobs": num_jobs,
            "shots_per_job": shots_per_job,
            "total_shots": num_jobs * shots_per_job,
            "shots_per_second": (
                (num_jobs * shots_per_job) / execution_time if execution_time > 0 else 0
            ),
        }

        return {
            "results_metrics": sim_results,
            "circuit_metrics": circuit_metrics,
            "complexity_metrics": complexity_metrics,
            "quantum_volume": qv_estimate,
            "timing_info": timing_info,
        }

    def run_on_ibm(self, channel: str = None, token: str = None):
        """
        Run the circuit on IBM Quantum hardware.

        Args:
            channel (str, optional): IBM Quantum channel
            token (str, optional): IBM Quantum token

        Returns:
            dict: Dictionary containing execution results and job information
        """
        try:
            overall_start_time = time.time()
            overall_start_timestamp = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(overall_start_time)
            )

            # Use environment variables if no credentials are provided
            channel = channel or IBM_QUANTUM_CHANNEL
            token = token or IBM_QUANTUM_TOKEN

            if not token:
                raise ValueError(
                    "No IBM Quantum token provided. Please set IBM_QUANTUM_TOKEN in .env file or provide it directly."
                )

            QiskitRuntimeService.save_account(channel=channel, token=token, overwrite=True)

            service = QiskitRuntimeService()
            backend = service.least_busy(simulator=False, operational=True)
            print(f"Using backend: {backend.configuration().backend_name}")
            print(f"Pending jobs: {backend.status().pending_jobs}")

            # Time the compilation and job submission
            compile_start_time = time.time()
            pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
            sampler = Sampler(backend)
            isa_circuit = pm.run(self)
            job = sampler.run([isa_circuit])
            job_id = job.job_id()
            compile_end_time = time.time()
            compile_time = compile_end_time - compile_start_time

            print(f">>> Job ID: {job_id}")
            print(f">>> Job Status: {job.status()}")
            print(f">>> Compilation time: {compile_time:.3f} seconds")

            # Time the job execution
            execution_start_time = time.time()
            execution_start_timestamp = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(execution_start_time)
            )

            timeout = 600  # 10 minutes timeout
            polling_interval = 5  # Check status every 5 seconds
            polling_count = 0

            while time.time() - execution_start_time < timeout:
                status = job.status()
                polling_count += 1
                print(f">>> Job Status: {status} (Poll #{polling_count})")

                if status == "DONE":
                    execution_end_time = time.time()
                    execution_end_timestamp = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(execution_end_time)
                    )
                    execution_time = execution_end_time - execution_start_time

                    result = job.result()
                    counts = result[0].data.c.get_counts()
                    num_shots = sum(counts.values())

                    print(f">>> Execution completed in {execution_time:.3f} seconds")
                    print(counts)
                    display(plot_histogram(counts))

                    # Update analyzers with raw IBM results
                    for analyzer in self.analyzers:
                        analyzer.add_results({"counts": counts})

                    # Calculate advanced complexity metrics
                    complexity_metrics = self.calculate_complexity_metrics()
                    
                    # Calculate quantum volume estimate
                    qv_estimate = self.estimate_quantum_volume()

                    # Calculate overall timing
                    overall_end_time = time.time()
                    overall_end_timestamp = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(overall_end_time)
                    )
                    overall_time = overall_end_time - overall_start_time

                    # Create timing information
                    timing_info = {
                        "overall_start_time": overall_start_timestamp,
                        "overall_end_time": overall_end_timestamp,
                        "overall_time": overall_time,
                        "compile_time": compile_time,
                        "queue_time": execution_start_time - compile_end_time,
                        "execution_time": execution_time,
                        "execution_start_time": execution_start_timestamp,
                        "execution_end_time": execution_end_timestamp,
                        "polling_count": polling_count,
                        "shots": num_shots,
                        "shots_per_second": num_shots / execution_time if execution_time > 0 else 0,
                    }

                    return {
                        "status": "completed",
                        "job_id": job_id,
                        "counts": counts,
                        "backend": backend.configuration().backend_name,
                        "timing_info": timing_info,
                        "complexity_metrics": complexity_metrics,
                        "quantum_volume": qv_estimate,
                    }
                elif status != "RUNNING" and status != "QUEUED":
                    # Calculate timing even for errors
                    execution_end_time = time.time()
                    overall_end_time = time.time()

                    # Calculate complexity metrics even for failures
                    complexity_metrics = self.calculate_complexity_metrics()
                    qv_estimate = self.estimate_quantum_volume()

                    timing_info = {
                        "overall_start_time": overall_start_timestamp,
                        "overall_end_time": time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime(overall_end_time)
                        ),
                        "overall_time": overall_end_time - overall_start_time,
                        "compile_time": compile_time,
                        "error_time": execution_end_time - execution_start_time,
                        "polling_count": polling_count,
                    }

                    print(f"Job ended with status: {status}")
                    return {
                        "status": "error",
                        "job_id": job_id,
                        "error": f"Job ended with status: {status}",
                        "backend": backend.configuration().backend_name,
                        "timing_info": timing_info,
                        "complexity_metrics": complexity_metrics,
                        "quantum_volume": qv_estimate,
                    }

                time.sleep(polling_interval)

            # Handle timeout
            overall_end_time = time.time()
            
            # Calculate complexity metrics even for timeouts
            complexity_metrics = self.calculate_complexity_metrics()
            qv_estimate = self.estimate_quantum_volume()
            
            timing_info = {
                "overall_start_time": overall_start_timestamp,
                "overall_end_time": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(overall_end_time)
                ),
                "overall_time": overall_end_time - overall_start_time,
                "compile_time": compile_time,
                "timeout_after": timeout,
                "polling_count": polling_count,
            }

            print("Job timed out after 10 minutes")
            return {
                "status": "pending",
                "job_id": job_id,
                "backend": backend.configuration().backend_name,
                "timing_info": timing_info,
                "complexity_metrics": complexity_metrics,
                "quantum_volume": qv_estimate,
            }
        except Exception as e:
            # Calculate timing even for exceptions
            overall_end_time = time.time()
            overall_time = overall_end_time - overall_start_time
            
            # Try to calculate complexity metrics if possible
            try:
                complexity_metrics = self.calculate_complexity_metrics()
                qv_estimate = self.estimate_quantum_volume()
                include_metrics = True
            except Exception:
                include_metrics = False

            timing_info = {
                "overall_start_time": overall_start_timestamp,
                "overall_end_time": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(overall_end_time)
                ),
                "overall_time": overall_time,
                "error_occurred_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            print(f"An error occurred: {e}")
            result = {"status": "error", "error": str(e), "timing_info": timing_info}
            
            # Add metrics if they were successfully calculated
            if include_metrics:
                result["complexity_metrics"] = complexity_metrics
                result["quantum_volume"] = qv_estimate
                
            return result

    def draw(self):
        """
        Draw the quantum circuit.

        Returns:
            matplotlib.figure.Figure: The circuit diagram
        """
        return super().draw(output="mpl")

    def run_analysis(self):
        """
        Run analysis on all analyzers and return their results.

        Returns:
            dict: Dictionary containing analysis results from each analyzer
        """
        analysis_results = {}
        for i, analyzer in enumerate(self.analyzers):
            analysis_results[f"analyzer_{i}"] = analyzer.analyze()
        return analysis_results

    def plot_analysis(self, ideal_rate: float = 0.5):
        """
        Generate plots for all analyzers.

        Args:
            ideal_rate (float): The ideal success rate to mark on the plots
        """
        for i, analyzer in enumerate(self.analyzers):
            print(f"\nPlotting analysis for analyzer {i}:")
            analyzer.plot(ideal_rate=ideal_rate)

    def estimate_quantum_volume(self):
        """
        Estimate the quantum volume of the current circuit.
        
        This is a circuit complexity metric based on the existing circuit's
        characteristics rather than the formal IBM Quantum Volume protocol.
        
        The estimate considers:
        - Circuit depth (temporal complexity)
        - Circuit width (spatial complexity)
        - Operation counts and types
        - Connectivity patterns
        
        Returns:
            dict: Quantum volume estimate and related metrics
        """
        # Get circuit metrics
        depth = self.depth()
        width = self.width()
        num_qubits = self.num_qubits
        size = self.size()
        op_counts = self.count_ops()
        
        # Start with baseline QV calculation based on effective square size
        effective_depth = min(depth, num_qubits)
        
        # Calculate standard QV base as 2^n where n is effective depth
        standard_qv = 2**effective_depth
        
        # Calculate complexity factors
        
        # 1. Square circuit factor - how close is it to a square circuit?
        # Perfect square circuit has depth = width
        square_ratio = min(depth, width) / max(depth, width) if max(depth, width) > 0 else 1.0
        
        # 2. Circuit density - how many operations per qubit-timestep?
        max_possible_ops = depth * width
        density = size / max_possible_ops if max_possible_ops > 0 else 0.0
        
        # 3. Gate complexity - multi-qubit operations are more complex
        # Count 2+ qubit gates (like cx, swaps) vs single-qubit gates
        multi_qubit_ops = sum(count for gate, count in op_counts.items() 
                            if gate not in ['barrier', 'measure', 'id', 'u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'h', 'x', 'y', 'z', 's', 't'])
        multi_qubit_ratio = multi_qubit_ops / size if size > 0 else 0.0
        
        # 4. Connectivity factor - for current circuit
        # This is a simplified approximation based on the presence of entangling operations
        connectivity_factor = 0.5 + 0.5 * (multi_qubit_ratio > 0)
        
        # Calculate the enhanced quantum volume
        # Use factors to adjust the standard QV
        enhanced_factor = (
            0.4 * square_ratio +     # Square circuits are foundational to QV
            0.3 * density +          # Dense circuits are more complex
            0.2 * multi_qubit_ratio + # Multi-qubit operations increase complexity
            0.1 * connectivity_factor # Connectivity affects feasibility
        )
        
        # Enhanced QV: apply enhancement factor to standard QV
        enhanced_qv = standard_qv * (1 + enhanced_factor)
        
        # Round to significant figures for clarity
        enhanced_qv_rounded = round(enhanced_qv, 2)
        
        return {
            "standard_quantum_volume": standard_qv,
            "enhanced_quantum_volume": enhanced_qv_rounded,
            "effective_depth": effective_depth,
            "factors": {
                "square_ratio": round(square_ratio, 2),
                "circuit_density": round(density, 2),
                "multi_qubit_ratio": round(multi_qubit_ratio, 2),
                "connectivity_factor": round(connectivity_factor, 2),
                "enhancement_factor": round(enhanced_factor, 2)
            },
            "circuit_metrics": {
                "depth": depth,
                "width": width,
                "size": size,
                "num_qubits": num_qubits,
                "operation_counts": op_counts
            }
        }
        
    def calculate_complexity_metrics(self):
        """
        Calculate various quantum circuit complexity measures as described in
        "Character Complexity: A Novel Measure for Quantum Circuit Analysis" by Daksh Shami.
        
        This includes traditional gate-based metrics, entanglement-based metrics,
        and approximation-based metrics.
        
        Returns:
            dict: Dictionary containing various complexity metrics
        """
        # Basic circuit properties
        depth = self.depth()
        width = self.num_qubits
        size = self.size()
        op_counts = self.count_ops()
        
        # 1. Gate-based metrics
        
        # 1.1 Gate count (total number of gates)
        gate_count = size
        
        # 1.2 Circuit depth (longest path through the circuit)
        circuit_depth = depth
        
        # 1.3 T-count (number of T gates, often considered costly in fault-tolerant implementations)
        t_count = op_counts.get('t', 0) + op_counts.get('tdg', 0)
        
        # 1.4 CNOT count (number of CNOT gates, important for entanglement)
        cnot_count = op_counts.get('cx', 0)
        
        # 1.5 Two-qubit gate count
        two_qubit_gates = ['cx', 'cz', 'swap', 'iswap', 'cp', 'cu', 'rxx', 'ryy', 'rzz', 'crx', 'cry', 'crz']
        two_qubit_count = sum(op_counts.get(gate, 0) for gate in two_qubit_gates)
        
        # 1.6 Multi-qubit gate ratio
        single_qubit_gates = ['id', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'rx', 'ry', 'rz', 'u1', 'u2', 'u3', 'p']
        single_qubit_count = sum(op_counts.get(gate, 0) for gate in single_qubit_gates)
        multi_qubit_count = gate_count - single_qubit_count - op_counts.get('barrier', 0) - op_counts.get('measure', 0)
        multi_qubit_ratio = multi_qubit_count / gate_count if gate_count > 0 else 0
        
        # 2. Entanglement-based metrics
        
        # 2.1 Entangling gate density
        # Ratio of entangling gates to total gates
        entangling_gate_density = two_qubit_count / gate_count if gate_count > 0 else 0
        
        # 2.2 Entangling width
        # Approximation: maximum number of qubits that could be entangled
        # This is an upper bound based on connectivity through CNOT gates
        entangling_width = min(width, two_qubit_count + 1) if two_qubit_count > 0 else 1
        
        # 3. Standardized metrics
        
        # 3.1 Circuit volume (depth × width)
        circuit_volume = depth * width
        
        # 3.2 Gate density (gates per qubit-time-step)
        gate_density = gate_count / circuit_volume if circuit_volume > 0 else 0
        
        # 3.3 Clifford vs non-Clifford ratio
        # Clifford gates: h, s, sdg, cx, cz, x, y, z
        clifford_gates = ['h', 's', 'sdg', 'cx', 'cz', 'x', 'y', 'z']
        clifford_count = sum(op_counts.get(gate, 0) for gate in clifford_gates)
        non_clifford_count = gate_count - clifford_count - op_counts.get('barrier', 0) - op_counts.get('measure', 0)
        clifford_ratio = clifford_count / gate_count if gate_count > 0 else 0
        non_clifford_ratio = non_clifford_count / gate_count if gate_count > 0 else 0
        
        # 4. Advanced metrics based on circuit structure
        
        # 4.1 Parallelism factor
        # How many gates can be executed in parallel on average
        parallelism_factor = gate_count / depth if depth > 0 else 0
        max_parallelism = width  # Maximum gates that could be executed in parallel (width of circuit)
        parallelism_efficiency = parallelism_factor / max_parallelism if max_parallelism > 0 else 0
        
        # 4.2 Circuit efficiency 
        # How efficiently the circuit uses the available qubits
        circuit_efficiency = gate_count / (width * depth) if (width * depth) > 0 else 0
        
        # 4.3 Quantum resource utilization
        # Combination of space (qubits) and time (depth) efficiency
        quantum_resource_utilization = 0.5 * (gate_count / (width * width) if width > 0 else 0) + 0.5 * (gate_count / (depth * depth) if depth > 0 else 0)
        
        # 5. Derived complexity metrics
        
        # 5.1 Square circuit factor (from Quantum Volume calculation)
        square_ratio = min(depth, width) / max(depth, width) if max(depth, width) > 0 else 1.0
        
        # 5.2 Weighted gate complexity
        # Assigning weights to different gate types based on their complexity
        gate_weights = {
            # Single-qubit gates
            'id': 1, 'x': 1, 'y': 1, 'z': 1, 'h': 1, 's': 1, 'sdg': 1,
            # More complex single-qubit gates
            't': 2, 'tdg': 2, 'rx': 2, 'ry': 2, 'rz': 2, 'p': 2,
            'u1': 2, 'u2': 3, 'u3': 4,
            # Two-qubit gates
            'cx': 10, 'cz': 10, 'swap': 12, 'cp': 12, 
            # Multi-qubit gates
            'ccx': 30, 'cswap': 32, 'mcx': 40,
            # Others default to 5
        }
        
        weighted_complexity = sum(count * gate_weights.get(gate, 5) for gate, count in op_counts.items())
        
        # 5.3 Normalized weighted complexity (per qubit)
        normalized_weighted_complexity = weighted_complexity / width if width > 0 else 0
        
        # Combine all metrics
        return {
            "gate_based_metrics": {
                "gate_count": gate_count,
                "circuit_depth": circuit_depth,
                "t_count": t_count,
                "cnot_count": cnot_count,
                "two_qubit_count": two_qubit_count,
                "multi_qubit_ratio": round(multi_qubit_ratio, 3)
            },
            "entanglement_metrics": {
                "entangling_gate_density": round(entangling_gate_density, 3),
                "entangling_width": entangling_width
            },
            "standardized_metrics": {
                "circuit_volume": circuit_volume,
                "gate_density": round(gate_density, 3),
                "clifford_ratio": round(clifford_ratio, 3),
                "non_clifford_ratio": round(non_clifford_ratio, 3)
            },
            "advanced_metrics": {
                "parallelism_factor": round(parallelism_factor, 3),
                "parallelism_efficiency": round(parallelism_efficiency, 3),
                "circuit_efficiency": round(circuit_efficiency, 3),
                "quantum_resource_utilization": round(quantum_resource_utilization, 3)
            },
            "derived_metrics": {
                "square_ratio": round(square_ratio, 3),
                "weighted_complexity": weighted_complexity,
                "normalized_weighted_complexity": round(normalized_weighted_complexity, 3)
            },
            "basic_properties": {
                "num_qubits": width,
                "depth": depth,
                "size": size,
                "operation_counts": op_counts
            }
        }

