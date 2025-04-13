"""
Scanner class for QWARD.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd

from qiskit import QuantumCircuit
from qiskit_aer import AerJob
from qiskit.providers.job import Job as QiskitJob

from qward.metrics.base_metric import Metric
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
    ):
        """
        Initialize a Scanner object.

        Args:
            circuit: The quantum circuit to analyze
            job: The job that executed the circuit
            result: The result of the job execution
        """
        self._circuit = circuit
        self._job = job
        self._result = result
        self._metrics: List[Metric] = []

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
    def metrics(self) -> List[Metric]:
        """
        Get the metrics.

        Returns:
            List[Metric]: The metrics
        """
        return self._metrics

    def add_metric(self, metric: Metric) -> None:
        """
        Add a metric to the scanner.

        Args:
            metric: The metric to add
        """
        self._metrics.append(metric)

    def calculate_metrics(self) -> pd.DataFrame:
        """
        Calculate metrics for all jobs.

        Returns:
            pd.DataFrame: DataFrame containing the metrics
        """
        # Initialize an empty dictionary to store the metrics
        metrics_dict = {}

        # Calculate metrics for each metric class
        for i, metric_class in enumerate(self.metrics):
            # Get the metrics from the metric class
            metric_results = metric_class.get_metrics()

            # Add the metrics to the dictionary with a unique key
            metric_name = f"{metric_class.__class__.__name__}_{i}"
            metrics_dict[metric_name] = metric_results

        # Create a DataFrame from the metrics dictionary
        metrics_df = pd.DataFrame(metrics_dict)

        # Sort the columns alphabetically
        metrics_df = metrics_df.reindex(sorted(metrics_df.columns), axis=1)

        return metrics_df

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
        items = []
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
