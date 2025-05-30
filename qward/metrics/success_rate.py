"""
Success rate metrics implementation for QWARD.

This module provides the SuccessRate class for analyzing the success rate of
quantum circuits based on job execution results. It supports both single job
and multiple job analysis with customizable success criteria.

The success rate metrics help evaluate the performance and reliability of
quantum circuit executions.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerJob
from qiskit.providers.job import JobV1 as QiskitJob

from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsType, MetricsId

# Import schemas for structured data validation
try:
    from qward.metrics.schemas import (
        SuccessRateJobSchema,
        SuccessRateAggregateSchema,
    )

    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False

# Type alias for job types
JobType = Union[AerJob, QiskitJob]


class SuccessRate(MetricCalculator):
    """
    Calculate success rate metrics for quantum circuits.

    This class provides methods for analyzing the success rate of quantum circuits
    based on job execution results. It supports both single job and multiple job
    analysis with customizable success criteria.

    The success rate metrics include probability of success, fidelity, error rates,
    and statistical analysis across multiple runs.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        job: Optional[JobType] = None,
        jobs: Optional[List[JobType]] = None,
        result: Optional[Dict] = None,
        success_criteria: Optional[Callable[[str], bool]] = None,
    ):
        """
        Initialize a SuccessRate object.

        Args:
            circuit: The quantum circuit to analyze
            job: A single job that executed the circuit
            jobs: A list of jobs that executed the circuit (for multiple runs)
            result: The result of the job execution (deprecated, use job instead)
            success_criteria: Function that determines if a measurement result is successful
        """
        super().__init__(circuit)
        self._job = job
        self._jobs = jobs or []
        if job and not self._jobs:
            self._jobs = [job]
        self._result = result  # Keep for backward compatibility
        self.success_criteria = success_criteria or self._default_success_criteria()

        # Maintain backward compatibility with runtime_job/runtime_jobs
        self.runtime_job = self._job
        self.runtime_jobs = self._jobs

    def _get_metric_type(self) -> MetricsType:
        """Get the type of this metric."""
        return MetricsType.POST_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        """Get the ID of this metric."""
        return MetricsId.SUCCESS_RATE

    def is_ready(self) -> bool:
        """Check if the metric is ready to be calculated."""
        return self.circuit is not None and (
            self._result is not None or self.runtime_job is not None or len(self.runtime_jobs) > 0
        )

    def _ensure_schemas_available(self) -> None:
        """Ensure Pydantic schemas are available, raise ImportError if not."""
        if not SCHEMAS_AVAILABLE:
            raise ImportError(
                "Pydantic schemas are not available. Install pydantic to use structured metrics."
            )

    def _default_success_criteria(self) -> Callable[[str], bool]:
        """
        Define the default success criteria for the circuit.

        By default, considers the ground state (all zeros) as success.
        Handles measurement results with spaces and classical bit information.

        Returns:
            Callable[[str], bool]: Function that takes a measurement result and returns True if successful
        """

        def is_ground_state(result: str) -> bool:
            # Remove all spaces to get clean bit string
            clean_result = result.replace(" ", "")

            # Check if all bits are zero
            return all(bit == "0" for bit in clean_result)

        return is_ground_state

    def _extract_job_id(self, job: JobType) -> str:
        """Extract job ID from a job object."""
        return job.job_id() if hasattr(job, "job_id") else "unknown"

    def _calculate_success_metrics(self, counts: Dict[str, int], job_id: str) -> Dict[str, Any]:
        """
        Calculate success metrics from measurement counts.

        Args:
            counts: Dictionary of measurement outcomes and their counts
            job_id: Identifier for the job

        Returns:
            Dict[str, Any]: Success rate metrics
        """
        if not counts:
            return {
                "job_id": job_id,
                "success_rate": 0.0,
                "error_rate": 1.0,
                "fidelity": 0.0,
                "total_shots": 0,
                "successful_shots": 0,
                "average_counts": counts,
            }

        # Calculate basic statistics
        total_shots = sum(counts.values())
        successful_shots = sum(
            count for state, count in counts.items() if self.success_criteria(state)
        )

        # Calculate rates
        success_rate = successful_shots / total_shots if total_shots > 0 else 0.0
        error_rate = 1.0 - success_rate

        # Calculate fidelity as the maximum probability
        max_count = max(counts.values())
        fidelity = max_count / total_shots if total_shots > 0 else 0.0

        return {
            "job_id": job_id,
            "success_rate": float(success_rate),
            "error_rate": float(error_rate),
            "fidelity": float(fidelity),
            "total_shots": total_shots,
            "successful_shots": successful_shots,
            "average_counts": counts,
        }

    # =============================================================================
    # Main API Methods
    # =============================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get success rate metrics based on available jobs.

        Returns:
            Dict[str, Any]: Dictionary containing success rate metrics
        """
        if len(self.runtime_jobs) > 1:
            return self.get_multiple_jobs_metrics()
        elif len(self.runtime_jobs) == 1:
            return self.get_single_job_metrics(self.runtime_jobs[0])
        else:
            raise ValueError("No jobs available to calculate metrics")

    def get_structured_metrics(self) -> Union["SuccessRateJobSchema", "SuccessRateAggregateSchema"]:
        """
        Get success rate metrics as structured, validated schema objects.

        Returns:
            Union[SuccessRateJobSchema, SuccessRateAggregateSchema]: Validated metrics schema

        Raises:
            ImportError: If Pydantic schemas are not available
            ValidationError: If metrics data doesn't match schema constraints
        """
        self._ensure_schemas_available()

        if len(self.runtime_jobs) > 1:
            return self.get_structured_multiple_jobs_metrics()
        elif len(self.runtime_jobs) == 1:
            return self.get_structured_single_job_metrics(self.runtime_jobs[0])
        else:
            raise ValueError("No jobs available to calculate metrics")

    # =============================================================================
    # Single Job Metrics
    # =============================================================================

    def get_single_job_metrics(self, job: Optional[JobType] = None) -> Dict[str, Any]:
        """
        Calculate success rate metrics from a single job result.

        Args:
            job: Optional job to use for metrics calculation. If None, uses self.runtime_job.

        Returns:
            Dict[str, Any]: Success rate metrics for a single job
        """
        if self.runtime_job is None and job is None:
            raise ValueError("A runtime job is required to calculate success rate")

        job_to_use = job or self.runtime_job
        result = job_to_use.result()
        counts = result.get_counts()
        job_id = self._extract_job_id(job_to_use)

        return self._calculate_success_metrics(counts, job_id)

    def get_structured_single_job_metrics(
        self, job: Optional[JobType] = None
    ) -> "SuccessRateJobSchema":
        """
        Get single job metrics as a validated schema object.

        Args:
            job: Optional job to use for metrics calculation

        Returns:
            SuccessRateJobSchema: Validated single job metrics
        """
        self._ensure_schemas_available()
        metrics = self.get_single_job_metrics(job)

        # Remove fields not in schema
        schema_data = {k: v for k, v in metrics.items() if k != "average_counts"}
        return SuccessRateJobSchema(**schema_data)

    # =============================================================================
    # Multiple Jobs Metrics
    # =============================================================================

    def get_multiple_jobs_metrics(self) -> Dict[str, Any]:
        """
        Calculate success rate metrics from multiple job results.

        Returns:
            Dict[str, Any]: Success rate metrics for multiple jobs with individual and aggregate data
        """
        if not self.runtime_jobs:
            raise ValueError("Multiple runtime jobs are required to calculate multiple job metrics")

        # Calculate metrics for each job
        job_metrics = []
        success_rates = []
        fidelities = []
        total_shots_list = []

        for job in self.runtime_jobs:
            single_metrics = self.get_single_job_metrics(job)

            # Extract data for aggregation
            success_rates.append(single_metrics["success_rate"])
            fidelities.append(single_metrics["fidelity"])
            total_shots_list.append(single_metrics["total_shots"])

            # Add to individual job metrics (exclude average_counts for cleaner output)
            job_data = {k: v for k, v in single_metrics.items() if k != "average_counts"}
            job_metrics.append(job_data)

        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(
            success_rates, fidelities, total_shots_list
        )

        return {
            "individual_jobs": job_metrics,
            "aggregate": aggregate_metrics,
        }

    def get_structured_multiple_jobs_metrics(self) -> "SuccessRateAggregateSchema":
        """
        Get multiple jobs metrics as a validated schema object.

        Returns:
            SuccessRateAggregateSchema: Validated aggregate metrics
        """
        self._ensure_schemas_available()
        metrics = self.get_multiple_jobs_metrics()
        return SuccessRateAggregateSchema(**metrics["aggregate"])

    def _calculate_aggregate_metrics(
        self, success_rates: List[float], fidelities: List[float], total_shots_list: List[int]
    ) -> Dict[str, Any]:
        """
        Calculate aggregate statistics from multiple job results.

        Args:
            success_rates: List of success rates from individual jobs
            fidelities: List of fidelities from individual jobs
            total_shots_list: List of total shots from individual jobs

        Returns:
            Dict[str, Any]: Aggregate metrics
        """
        if not success_rates:
            return {
                "mean_success_rate": 0.0,
                "std_success_rate": 0.0,
                "min_success_rate": 0.0,
                "max_success_rate": 0.0,
                "total_trials": 0,
                "fidelity": 0.0,
                "error_rate": 1.0,
            }

        # Convert to numpy arrays for efficient calculation
        success_rates_array = np.array(success_rates)
        fidelities_array = np.array(fidelities)

        # Calculate statistics
        mean_success_rate = float(np.mean(success_rates_array))
        std_success_rate = (
            float(np.std(success_rates_array)) if len(success_rates_array) > 1 else 0.0
        )
        min_success_rate = float(np.min(success_rates_array))
        max_success_rate = float(np.max(success_rates_array))
        total_trials = sum(total_shots_list)
        avg_fidelity = float(np.mean(fidelities_array))
        error_rate = 1.0 - mean_success_rate

        return {
            "mean_success_rate": mean_success_rate,
            "std_success_rate": std_success_rate,
            "min_success_rate": min_success_rate,
            "max_success_rate": max_success_rate,
            "total_trials": total_trials,
            "fidelity": avg_fidelity,
            "error_rate": error_rate,
        }

    # =============================================================================
    # Job Management Methods
    # =============================================================================

    def add_job(self, job: Union[JobType, List[JobType]]) -> None:
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
