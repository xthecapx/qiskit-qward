"""
Scanner class for QWARD.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd

from qiskit import QuantumCircuit
from qiskit_aer import AerJob
from qiskit.providers.job import Job as QiskitJob

from qward.metrics.base_metric import MetricCalculator


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
        strategies: Optional[list] = None,
    ):
        """
        Initialize a Scanner object.

        Args:
            circuit: The quantum circuit to analyze
            job: The job that executed the circuit
            strategies: Optional list of metric strategy classes or instances.
                       If a class is provided, it will be instantiated with the circuit.
                       If an instance is provided, its circuit must match the Scanner's circuit.
        """
        self._circuit = circuit
        self._job = job
        self._strategies: List[MetricCalculator] = []

        if strategies is not None:
            for strategy in strategies:
                self._add_strategy_to_list(strategy)

    def _add_strategy_to_list(self, strategy):
        """Helper method to add a strategy to the list with validation."""
        # If strategy is a class (not instance), instantiate with circuit
        if isinstance(strategy, type):
            self._strategies.append(strategy(self._circuit))
            return

        # Strategy is an instance - validate circuit compatibility
        strategy_circuit = getattr(strategy, "circuit", None) or getattr(strategy, "_circuit", None)

        if strategy_circuit is not None and strategy_circuit is not self._circuit:
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
        metric_dataframes: Dict[str, pd.DataFrame] = {}

        for strategy in self.strategies:
            metric_results = strategy.get_metrics()
            metric_name = strategy.__class__.__name__

            if metric_name == "CircuitPerformanceMetrics":
                self._process_circuit_performance_metrics(
                    strategy, metric_results, metric_dataframes
                )
            else:
                self._process_standard_metrics(metric_name, metric_results, metric_dataframes)

        return metric_dataframes

    def _process_circuit_performance_metrics(self, strategy, metric_results, metric_dataframes):
        """Process CircuitPerformanceMetrics with backward compatibility."""
        display_name = "CircuitPerformance"

        # Check for legacy API methods
        if self._has_legacy_api(strategy):
            self._process_legacy_circuit_performance(strategy, display_name, metric_dataframes)
            return

        # Check for new schema-based API
        if hasattr(metric_results, "to_flat_dict"):
            flattened_metrics = metric_results.to_flat_dict()
            single_job_df = pd.DataFrame([flattened_metrics])
            metric_dataframes[f"{display_name}.individual_jobs"] = single_job_df
            return

        # Check for legacy structure with individual_jobs
        if isinstance(metric_results, dict) and "individual_jobs" in metric_results:
            individual_jobs_df = pd.DataFrame(metric_results["individual_jobs"])
            aggregate_df = pd.DataFrame([metric_results["aggregate"]])
            metric_dataframes[f"{display_name}.individual_jobs"] = individual_jobs_df
            metric_dataframes[f"{display_name}.aggregate"] = aggregate_df
            return

        # Fallback - flatten manually
        flattened_metrics = self._flatten_metrics(metric_results)
        single_job_df = pd.DataFrame([flattened_metrics])
        metric_dataframes[f"{display_name}.individual_jobs"] = single_job_df

    def _has_legacy_api(self, strategy):
        """Check if strategy has legacy API methods."""
        return hasattr(strategy, "get_single_job_metrics") and hasattr(
            strategy, "get_multiple_jobs_metrics"
        )

    def _process_legacy_circuit_performance(self, strategy, display_name, metric_dataframes):
        """Process legacy CircuitPerformance metrics."""
        jobs_count = len(getattr(strategy, "_jobs", []))

        if jobs_count > 1:
            legacy_metrics = strategy.get_multiple_jobs_metrics()
            individual_jobs_df = pd.DataFrame(legacy_metrics["individual_jobs"])
            aggregate_df = pd.DataFrame([legacy_metrics["aggregate"]])
            metric_dataframes[f"{display_name}.individual_jobs"] = individual_jobs_df
            metric_dataframes[f"{display_name}.aggregate"] = aggregate_df
        else:
            single_metrics = strategy.get_single_job_metrics()
            single_job_df = pd.DataFrame([single_metrics])
            metric_dataframes[f"{display_name}.individual_jobs"] = single_job_df

    def _process_standard_metrics(self, metric_name, metric_results, metric_dataframes):
        """Process standard (non-CircuitPerformance) metrics."""
        # Check if it's a schema object with to_flat_dict method
        if hasattr(metric_results, "to_flat_dict"):
            flattened_metrics = metric_results.to_flat_dict()
        else:
            flattened_metrics = self._flatten_metrics(metric_results)

        df = pd.DataFrame([flattened_metrics])
        metric_dataframes[metric_name] = df

    def _flatten_metrics(self, metric_results):
        """Flatten metric results into a single-level dictionary."""
        flattened_metrics = {}

        if isinstance(metric_results, dict):
            for key, value in metric_results.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flattened_metrics[f"{key}.{subkey}"] = subvalue
                else:
                    flattened_metrics[key] = value
        elif hasattr(metric_results, "model_dump"):
            # Schema object without to_flat_dict - try model_dump
            metric_dict = metric_results.model_dump()
            for key, value in metric_dict.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flattened_metrics[f"{key}.{subkey}"] = subvalue
                else:
                    flattened_metrics[key] = value
        else:
            # Fallback - treat as single value
            flattened_metrics = {"value": metric_results}

        return flattened_metrics

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
