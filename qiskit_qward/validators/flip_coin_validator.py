from qiskit import QuantumCircuit
from .base_validator import BaseValidator
from ..analysis.success_rate import SuccessRate


class FlipCoinValidator(BaseValidator):
    """
    Validator for quantum coin flip circuit.
    """

    def __init__(self, use_barriers: bool = True):
        """
        Initialize the quantum coin flip validator.

        Args:
            use_barriers (bool): Whether to include barriers in the circuit
        """
        super().__init__(num_qubits=1, num_clbits=1, use_barriers=use_barriers, name="flip_coin")

        # Define success criteria: tails (1) is considered success
        def success_criteria(state):
            return state == "1"

        # Add success rate analyzer with the criteria
        success_analyzer = SuccessRate()
        success_analyzer.set_success_criteria(success_criteria)
        self.add_analyzer(success_analyzer)

        # Build the circuit
        self._build_circuit()

    def _build_circuit(self):
        """
        Build the quantum coin flip circuit.
        """
        # Apply Hadamard gate to create superposition
        self.h(0)

        if self.use_barriers:
            self.barrier()

        # Measure the qubit
        self.measure(0, 0)

    def run_simulation(
        self, show_histogram: bool = False, num_jobs: int = 1000, shots_per_job: int = 1024
    ):
        """
        Run simulation with success rate analysis.

        Args:
            show_histogram (bool): Whether to display the measurement histogram
            num_jobs (int): Number of independent jobs to run
            shots_per_job (int): Number of shots per job

        Returns:
            dict: Simulation results including success rate analysis
        """
        results = super().run_simulation(
            show_histogram=show_histogram, num_jobs=num_jobs, shots_per_job=shots_per_job
        )

        # Add analysis results
        results["analysis"] = self.run_analysis()

        return results
