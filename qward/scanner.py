"""
Scanner class for QWARD.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd

from qiskit import QuantumCircuit
from qiskit_aer import AerJob
from qiskit.providers.job import Job as QiskitJob

from qward.metrics.base_metric import MetricCalculator
from qward.result import Result


class Scanner:
    """
    Class for analyzing quantum circuits.

    This class provides methods for analyzing quantum circuits using various metrics.
    """

    def __init__(
        self,
        circuit: Optional[QuantumCircuit] = None,
        job: Optional[Union[AerJob, QiskitJob]] = None,
        result: Optional[Result] = None,
        calculators: Optional[list] = None,
        metrics: Optional[list] = None,  # Backward compatibility
    ):
        """
        Initialize a Scanner object.

        Args:
            circuit: The quantum circuit to analyze
            job: The job that executed the circuit
            result: The result of the job execution
            calculators: Optional list of metric calculator classes or instances. If a class is provided, it will be instantiated with the circuit. If an instance is provided, its circuit must match the Scanner's circuit if it has one.
            metrics: Deprecated. Use 'calculators' instead. Maintained for backward compatibility.
        """
        self._circuit = circuit
        self._job = job
        self._result = result
        self._calculators: List[MetricCalculator] = []

        # Handle backward compatibility
        metric_list = calculators if calculators is not None else metrics

        if metric_list is not None:
            for calculator in metric_list:
                # If calculator is a class (not instance), instantiate with circuit
                if isinstance(calculator, type):
                    self._calculators.append(calculator(circuit))
                else:
                    # If calculator is an instance, check for .circuit or ._circuit attribute
                    calculator_circuit = getattr(calculator, "circuit", None)
                    if calculator_circuit is None:
                        # Try protected attribute (for base class)
                        calculator_circuit = getattr(calculator, "_circuit", None)
                    if calculator_circuit is not None:
                        if calculator_circuit is not circuit:
                            raise ValueError(
                                f"Calculator instance {calculator.__class__.__name__} was initialized with a different circuit than the Scanner."
                            )
                    self._calculators.append(calculator)
        # If no calculators provided, user can add them later with add_calculator

    @property
    def circuit(self) -> Optional[QuantumCircuit]:
        """
        Get the quantum circuit.

        Returns:
            Optional[QuantumCircuit]: The quantum circuit
        """
        return self._circuit

    @property
    def job(self) -> Optional[Union[AerJob, QiskitJob]]:
        """
        Get the job that executed the circuit.

        Returns:
            Optional[Union[AerJob, QiskitJob]]: The job that executed the circuit
        """
        return self._job

    @property
    def result(self) -> Optional[Result]:
        """
        Get the result of the job execution.

        Returns:
            Optional[Result]: The result of the job execution
        """
        return self._result

    @property
    def calculators(self) -> List[MetricCalculator]:
        """
        Get the metric calculators.

        Returns:
            List[MetricCalculator]: The metric calculators
        """
        return self._calculators

    @property
    def metrics(self) -> List[MetricCalculator]:
        """
        Get the metrics (backward compatibility alias for calculators).

        Returns:
            List[MetricCalculator]: The metric calculators
        """
        return self._calculators

    def add_calculator(self, calculator: MetricCalculator) -> None:
        """
        Add a metric calculator to the scanner.

        Args:
            calculator: The metric calculator to add
        """
        self._calculators.append(calculator)

    def add_metric(self, metric: MetricCalculator) -> None:
        """
        Add a metric to the scanner (backward compatibility alias for add_calculator).

        Args:
            metric: The metric calculator to add
        """
        self.add_calculator(metric)

    def calculate_metrics(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate metrics for all jobs.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing DataFrames for each metric type.
            For SuccessRate metrics, returns two DataFrames:
            - "SuccessRate.individual_jobs": DataFrame containing metrics for each job
            - "SuccessRate.aggregate": DataFrame containing aggregate metrics across all jobs
        """
        # Initialize a dictionary to store DataFrames for each metric type
        metric_dataframes = {}

        # Calculate metrics for each calculator
        for _, calculator in enumerate(self.calculators):
            # Get the metrics from the calculator
            metric_results = calculator.get_metrics()

            # Create a DataFrame for this metric type
            metric_name = calculator.__class__.__name__

            # Special handling for SuccessRate metrics
            if metric_name == "SuccessRate":
                if "individual_jobs" in metric_results:
                    # Multiple jobs case - already formatted correctly
                    individual_jobs_df = pd.DataFrame(metric_results["individual_jobs"])
                    metric_dataframes[f"{metric_name}.individual_jobs"] = individual_jobs_df

                    # Create DataFrame for aggregate metrics
                    aggregate_metrics = metric_results["aggregate"]
                    aggregate_df = pd.DataFrame([aggregate_metrics])
                    metric_dataframes[f"{metric_name}.aggregate"] = aggregate_df
                else:
                    # Single job case - format it like multiple jobs case
                    # Create individual jobs DataFrame
                    individual_jobs_df = pd.DataFrame([metric_results])
                    metric_dataframes[f"{metric_name}.individual_jobs"] = individual_jobs_df

                    # Create aggregate metrics
                    aggregate_metrics = {
                        "mean_success_rate": metric_results["success_rate"],
                        "std_success_rate": 0.0,  # No std dev for single job
                        "min_success_rate": metric_results["success_rate"],
                        "max_success_rate": metric_results["success_rate"],
                        "total_trials": metric_results["total_shots"],
                        "fidelity": metric_results["fidelity"],
                        "error_rate": metric_results["error_rate"],
                    }
                    aggregate_df = pd.DataFrame([aggregate_metrics])
                    metric_dataframes[f"{metric_name}.aggregate"] = aggregate_df
            else:
                # Handle other metrics normally
                flattened_metrics = {}
                for key, value in metric_results.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            flattened_metrics[f"{key}.{subkey}"] = subvalue
                    else:
                        flattened_metrics[key] = value

                # Create DataFrame for this metric type
                df = pd.DataFrame([flattened_metrics])
                metric_dataframes[metric_name] = df

        return metric_dataframes

    def _flatten_dict(
        self, d: Dict[str, Any], parent_key: str = "", sep: str = "_"
    ) -> Dict[str, Any]:
        """
        Flatten a nested dictionary.

        Args:
            d: The dictionary to flatten
            parent_key: The parent key for the current level
            sep: The separator to use between keys

        Returns:
            Dict[str, Any]: The flattened dictionary
        """
        items: List[Tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def set_circuit(self, circuit: QuantumCircuit) -> None:
        """
        Set the quantum circuit.

        Args:
            circuit: The quantum circuit
        """
        self._circuit = circuit

    def set_job(self, job: Union[AerJob, QiskitJob]) -> None:
        """
        Set the job that executed the circuit.

        Args:
            job: The job that executed the circuit
        """
        self._job = job

    def set_result(self, result: Result) -> None:
        """
        Set the result of the job execution.

        Args:
            result: The result of the job execution
        """
        self._result = result
