"""
Success rate metrics implementation for QWARD.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerJob
from qiskit.providers.job import JobV1 as QiskitJob

from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsType, MetricsId


class SuccessRate(MetricCalculator):
    """
    Class for calculating success rate metrics for quantum circuits.

    This class provides methods for analyzing the success rate of quantum circuits,
    including metrics such as the probability of success, fidelity, and more.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        job: Optional[Union[AerJob, QiskitJob]] = None,
        jobs: Optional[List[Union[AerJob, QiskitJob]]] = None,
        result: Optional[Dict] = None,
        success_criteria: Optional[Callable[[str], bool]] = None,
    ):
        """
        Initialize a SuccessRate object.

        Args:
            circuit: The quantum circuit to analyze
            job: A single job that executed the circuit
            jobs: A list of jobs that executed the circuit (for multiple runs)
            result: The result of the job execution
            success_criteria: Function that determines if a measurement result is successful
        """
        super().__init__(circuit)
        self._job = job
        self._jobs = jobs or []
        if job and not self._jobs:
            self._jobs = [job]
        self._result = result
        self.success_criteria = success_criteria or self._default_success_criteria()
        self.runtime_job = self._job
        self.runtime_jobs = self._jobs

    def _default_success_criteria(self) -> Callable[[str], bool]:
        """
        Define the default success criteria for the circuit.
        By default, considers all zeros as success.

        Returns:
            Callable[[str], bool]: Function that takes a measurement result and returns True if successful
        """
        return lambda result: result == "0"

    def _get_metric_type(self) -> MetricsType:
        """
        Get the type of this metric.

        Returns:
            MetricsType: The type of this metric
        """
        return MetricsType.POST_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        """
        Get the ID of this metric.

        Returns:
            MetricsId: The ID of this metric
        """
        return MetricsId.SUCCESS_RATE

    def is_ready(self) -> bool:
        """
        Check if the metric is ready to be calculated.

        Returns:
            bool: True if the metric is ready to be calculated, False otherwise
        """
        return self.circuit is not None and (
            self._result is not None or self.runtime_job is not None or len(self.runtime_jobs) > 0
        )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the metrics.

        Returns:
            Dict[str, Any]: Dictionary containing the metrics
        """
        # If we have multiple jobs, use the multiple job metrics
        if len(self.runtime_jobs) > 1:
            return self.get_multiple_jobs_metrics()
        # Otherwise, use the single job metrics
        elif len(self.runtime_jobs) == 1:
            return self.get_single_job_metrics(self.runtime_jobs[0])
        else:
            raise ValueError("No jobs available to calculate metrics")

    def get_single_job_metrics(self, job: Optional[QiskitJob] = None) -> Dict[str, Any]:
        """
        Calculate success rate metrics from a single job result.

        Args:
            job: Optional job to use for metrics calculation. If None, uses self.runtime_job.

        Returns:
            dict: Success rate metrics for a single job
        """
        if self.runtime_job is None and job is None:
            raise ValueError("We need a runtime job to calculate success rate")

        job_to_use = job or self.runtime_job
        result = job_to_use.result()

        # Get counts from the result
        counts = result.get_counts()

        if not counts:
            # Use job_id from job object if available
            job_id = job_to_use.job_id() if hasattr(job_to_use, "job_id") else "0"
            return {
                "job_id": job_id,
                "success_rate": 0.0,
                "error_rate": 1.0,
                "fidelity": 0.0,
                "total_shots": 0,
                "successful_shots": 0,
                "average_counts": counts,
            }

        # Calculate success rate using the custom success criteria
        total_shots = sum(counts.values())
        successful_shots = 0

        for state, count in counts.items():
            if self.success_criteria(state):
                successful_shots += count

        success_rate = successful_shots / total_shots if total_shots > 0 else 0.0

        # Calculate fidelity as the maximum probability
        max_count = max(counts.values())
        fidelity = max_count / total_shots if total_shots > 0 else 0.0

        # Calculate error rate as 1 - success_rate
        error_rate = 1.0 - success_rate

        # Extract job_id from the job object if available
        job_id = job_to_use.job_id() if hasattr(job_to_use, "job_id") else "0"

        metrics = {
            "job_id": job_id,
            "success_rate": float(success_rate),
            "error_rate": float(error_rate),
            "fidelity": float(fidelity),
            "total_shots": total_shots,
            "successful_shots": successful_shots,
            "average_counts": counts,
        }

        return metrics

    def get_multiple_jobs_metrics(self) -> Dict[str, Any]:
        """
        Calculate success rate metrics from multiple job results.

        Returns:
            dict: Success rate metrics for multiple jobs, including individual job metrics
                  and aggregate metrics across all jobs
        """
        if not self.runtime_jobs:
            raise ValueError("We need multiple runtime jobs to calculate multiple job metrics")

        # Calculate metrics for each job
        success_rates: List[float] = []
        fidelities: List[float] = []
        total_shots_list: List[int] = []
        successful_shots_list: List[int] = []
        job_metrics: List[Dict[str, Any]] = []

        for i, job in enumerate(self.runtime_jobs):
            # Temporarily set the current job to calculate single job metrics
            original_job = self.runtime_job
            self.runtime_job = job

            # Reuse single job metrics calculation
            single_job_metrics = self.get_single_job_metrics()

            # Restore the original job
            self.runtime_job = original_job

            # Extract metrics we need for aggregation
            success_rates.append(single_job_metrics["success_rate"])
            fidelities.append(single_job_metrics["fidelity"])
            total_shots_list.append(single_job_metrics["total_shots"])
            successful_shots_list.append(single_job_metrics["successful_shots"])

            # Add to job metrics list with actual job_id from the job object
            job_metrics.append(
                {
                    "job_id": job.job_id() if hasattr(job, "job_id") else str(i),
                    "success_rate": single_job_metrics["success_rate"],
                    "error_rate": single_job_metrics["error_rate"],
                    "fidelity": single_job_metrics["fidelity"],
                    "total_shots": single_job_metrics["total_shots"],
                    "successful_shots": single_job_metrics["successful_shots"],
                }
            )

        if not success_rates:
            return {
                "individual_jobs": [],
                "aggregate": {
                    "mean_success_rate": 0.0,
                    "std_success_rate": 0.0,
                    "min_success_rate": 0.0,
                    "max_success_rate": 0.0,
                    "total_trials": 0,
                    "fidelity": 0.0,
                    "error_rate": 1.0,
                },
            }

        # Calculate aggregate metrics
        success_rates_array = np.array(success_rates)
        mean_success_rate = float(np.mean(success_rates_array))
        std_success_rate = (
            float(np.std(success_rates_array)) if len(success_rates_array) > 1 else 0.0
        )
        min_success_rate = float(np.min(success_rates_array))
        max_success_rate = float(np.max(success_rates_array))
        total_trials = sum(total_shots_list)

        # Calculate average fidelity
        avg_fidelity = float(np.mean(fidelities))

        # Calculate error rate as 1 - mean_success_rate
        error_rate = 1.0 - mean_success_rate

        # Return both individual job metrics and aggregate metrics
        metrics = {
            "individual_jobs": job_metrics,
            "aggregate": {
                "mean_success_rate": mean_success_rate,
                "std_success_rate": std_success_rate,
                "min_success_rate": min_success_rate,
                "max_success_rate": max_success_rate,
                "total_trials": total_trials,
                "fidelity": avg_fidelity,
                "error_rate": error_rate,
            },
        }

        return metrics

    def add_job(self, job: Union[AerJob, QiskitJob, List[Union[AerJob, QiskitJob]]]) -> None:
        """
        Add one or more jobs to the list of jobs for multiple job metrics.

        Args:
            job: A single job or a list of jobs to add
        """
        if isinstance(job, list):
            for single_job in job:
                if single_job not in self.runtime_jobs:
                    self.runtime_jobs.append(single_job)
                    # If this is the first job, also set it as the single job
                    if not self.runtime_job:
                        self.runtime_job = single_job
        else:
            if job not in self.runtime_jobs:
                self.runtime_jobs.append(job)
                # If this is the first job, also set it as the single job
                if not self.runtime_job:
                    self.runtime_job = job
