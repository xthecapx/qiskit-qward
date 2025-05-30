"""
Scanner class for QWARD.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd

from qiskit import QuantumCircuit
from qiskit_aer import AerJob
from qiskit.providers.job import Job as QiskitJob

from qward.metrics.base_metric import MetricCalculator
from qward.result import Result


class Scanner:
    """
    Class for analyzing quantum circuits using the Strategy pattern.

    This class provides methods for analyzing quantum circuits using various metric strategies.
    """

    def __init__(
        self,
        circuit: Optional[QuantumCircuit] = None,
        *,
        job: Optional[Union[AerJob, QiskitJob]] = None,
        result: Optional[Result] = None,
        strategies: Optional[list] = None,
    ):
        """
        Initialize a Scanner object.

        Args:
            circuit: The quantum circuit to analyze
            job: The job that executed the circuit
            result: The result of the job execution
            strategies: Optional list of metric strategy classes or instances.
                       If a class is provided, it will be instantiated with the circuit.
                       If an instance is provided, its circuit must match the Scanner's circuit.
        """
        self._circuit = circuit
        self._job = job
        self._result = result
        self._strategies: List[MetricCalculator] = []

        if strategies is not None:
            for strategy in strategies:
                # If strategy is a class (not instance), instantiate with circuit
                if isinstance(strategy, type):
                    self._strategies.append(strategy(circuit))
                else:
                    # If strategy is an instance, check for .circuit or ._circuit attribute
                    strategy_circuit = getattr(strategy, "circuit", None)
                    if strategy_circuit is None:
                        # Try protected attribute (for base class)
                        strategy_circuit = getattr(strategy, "_circuit", None)
                    if strategy_circuit is not None:
                        if strategy_circuit is not circuit:
                            raise ValueError(
                                f"Strategy instance {strategy.__class__.__name__} was initialized with a different circuit than the Scanner."
                            )
                    self._strategies.append(strategy)

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
    def strategies(self) -> List[MetricCalculator]:
        """
        Get the metric strategies.

        Returns:
            List[MetricCalculator]: The metric strategies
        """
        return self._strategies

    def add_strategy(self, strategy: MetricCalculator) -> None:
        """
        Add a metric strategy to the scanner.

        Args:
            strategy: The metric strategy to add
        """
        self._strategies.append(strategy)

    def calculate_metrics(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate metrics using all registered strategies.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing DataFrames for each metric type.
            For CircuitPerformance metrics, returns two DataFrames:
            - "CircuitPerformance.individual_jobs": DataFrame containing metrics for each job
            - "CircuitPerformance.aggregate": DataFrame containing aggregate metrics across all jobs
        """
        # Initialize a dictionary to store DataFrames for each metric type
        metric_dataframes = {}

        # Calculate metrics for each strategy
        for strategy in self.strategies:
            # Get the metrics from the strategy
            metric_results = strategy.get_metrics()

            # Create a DataFrame for this metric type
            metric_name = strategy.__class__.__name__

            # Special handling for CircuitPerformance metrics
            if metric_name == "CircuitPerformance":
                # Check if this is the new API structure or legacy structure
                if hasattr(strategy, "get_single_job_metrics") and hasattr(
                    strategy, "get_multiple_jobs_metrics"
                ):
                    # Use legacy methods for backward compatibility with visualization
                    if len(getattr(strategy, "_jobs", [])) > 1:
                        # Multiple jobs case
                        legacy_metrics = strategy.get_multiple_jobs_metrics()
                        individual_jobs_df = pd.DataFrame(legacy_metrics["individual_jobs"])
                        aggregate_df = pd.DataFrame([legacy_metrics["aggregate"]])

                        metric_dataframes[f"{metric_name}.individual_jobs"] = individual_jobs_df
                        metric_dataframes[f"{metric_name}.aggregate"] = aggregate_df
                    else:
                        # Single job case
                        single_metrics = strategy.get_single_job_metrics()
                        single_job_df = pd.DataFrame([single_metrics])
                        metric_dataframes[f"{metric_name}.individual_jobs"] = single_job_df
                elif "individual_jobs" in metric_results:
                    # Legacy structure - multiple jobs case
                    individual_jobs_df = pd.DataFrame(metric_results["individual_jobs"])
                    aggregate_df = pd.DataFrame([metric_results["aggregate"]])

                    metric_dataframes[f"{metric_name}.individual_jobs"] = individual_jobs_df
                    metric_dataframes[f"{metric_name}.aggregate"] = aggregate_df
                else:
                    # New API structure - flatten it for DataFrame compatibility
                    flattened_metrics = {}
                    for category, category_metrics in metric_results.items():
                        if isinstance(category_metrics, dict):
                            for key, value in category_metrics.items():
                                flattened_metrics[f"{category}.{key}"] = value
                        else:
                            flattened_metrics[category] = category_metrics

                    single_job_df = pd.DataFrame([flattened_metrics])
                    metric_dataframes[f"{metric_name}.individual_jobs"] = single_job_df
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
