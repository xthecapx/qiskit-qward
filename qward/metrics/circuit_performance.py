"""
Circuit performance metrics implementation for QWARD.

This module provides the CircuitPerformanceMetrics class for analyzing the performance of
quantum circuits based on job execution results. It supports both single job
and multiple job analysis with customizable success criteria.

Supports multiple job types:
- AerJob: Qiskit Aer simulator jobs
- QiskitJob (JobV1): Traditional Qiskit jobs
- RuntimeJobV2: IBM Quantum Runtime V2 primitive jobs (SamplerV2, EstimatorV2)

The performance metrics help evaluate the success rate, fidelity, and statistical
properties of quantum circuit executions.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerJob
from qiskit_ibm_runtime import RuntimeJobV2

from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsType, MetricsId

# Import schemas for structured data validation
try:
    from qward.metrics.schemas import (
        CircuitPerformanceSchema,
        SuccessMetricsSchema,
        FidelityMetricsSchema,
        StatisticalMetricsSchema,
    )

    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False

# Type alias for job types - include RuntimeJobV2
JobType = Union[AerJob, RuntimeJobV2]


class CircuitPerformanceMetrics(MetricCalculator):
    """
    Calculate circuit performance metrics for quantum circuits.

    This class provides methods for analyzing the performance of quantum circuits
    based on job execution results. It supports both single job and multiple job
    analysis with customizable success criteria.

    Supports multiple job types:
    - AerJob: Qiskit Aer simulator jobs
    - QiskitJob (JobV1): Traditional Qiskit jobs
    - RuntimeJobV2: IBM Quantum Runtime V2 primitive jobs (SamplerV2, EstimatorV2)

    The performance metrics include:
    - Success metrics: Success rate, error rate, successful shots analysis
    - Fidelity metrics: Quantum fidelity between measured and expected distributions
    - Statistical metrics: Entropy, uniformity, concentration analysis

    Fidelity Calculation:
    - If expected_distribution is provided: Classical fidelity F = Σᵢ √(pᵢ × qᵢ)
      where pᵢ is measured probability and qᵢ is expected probability
    - If no expected distribution: Probability of most frequent successful outcome

    Example:
        # Basic usage with default success criteria (ground state)
        performance = CircuitPerformanceMetrics(circuit, job=job)
        metrics = performance.get_metrics()

        # With RuntimeJobV2 from SamplerV2
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        sampler = Sampler(backend)
        job = sampler.run([circuit])  # Returns RuntimeJobV2
        performance = CircuitPerformanceMetrics(circuit, job=job)
        success_metrics = performance.get_success_metrics()

        # With custom success criteria and expected distribution
        def custom_success(state): return state == "101"
        expected = {"000": 0.5, "101": 0.5}  # Bell state expectation

        performance = CircuitPerformanceMetrics(
            circuit,
            job=job,
            success_criteria=custom_success,
            expected_distribution=expected
        )
        success_metrics = performance.get_success_metrics()
        fidelity_metrics = performance.get_fidelity_metrics()
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        job: Optional[JobType] = None,
        jobs: Optional[List[JobType]] = None,
        success_criteria: Optional[Callable[[str], bool]] = None,
        expected_distribution: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize a CircuitPerformanceMetrics object.

        Args:
            circuit: The quantum circuit to analyze
            job: A single job that executed the circuit
            jobs: A list of jobs that executed the circuit (for multiple runs)
            success_criteria: Function that determines if a measurement result is successful
            expected_distribution: Expected probability distribution for fidelity calculation
        """
        super().__init__(circuit)
        self._job = job
        self._jobs = jobs or []
        if job and not self._jobs:
            self._jobs = [job]
        self.success_criteria = success_criteria or self._default_success_criteria()
        self.expected_distribution = expected_distribution

    def _get_metric_type(self) -> MetricsType:
        """Get the type of this metric."""
        return MetricsType.POST_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        """Get the ID of this metric."""
        return MetricsId.CIRCUIT_PERFORMANCE

    def is_ready(self) -> bool:
        """Check if the metric is ready to be calculated."""
        return self.circuit is not None and (self._job is not None or len(self._jobs) > 0)

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

    def _extract_counts(self, job: JobType) -> Dict[str, int]:
        """
        Extract counts from a job result, handling different job types.

        Args:
            job: The job object to extract counts from

        Returns:
            Dict[str, int]: Dictionary of measurement outcomes and their counts

        Raises:
            ValueError: If counts cannot be extracted from the job result
        """
        try:
            if isinstance(job, RuntimeJobV2):
                # Handle RuntimeJobV2 (V2 primitives)
                result = job.result()
                # V2 primitives return a list of PubResult objects
                if len(result) > 0:
                    pub_result = result[0]  # Get first (and typically only) PUB result

                    # Try common classical register names
                    common_names = ["meas", "c", "cr", "classical"]
                    for name in common_names:
                        if hasattr(pub_result.data, name):
                            register_data = getattr(pub_result.data, name)
                            if hasattr(register_data, "get_counts"):
                                return register_data.get_counts()

                    # If common names don't work, try to find any classical register data
                    data_attrs = [attr for attr in dir(pub_result.data) if not attr.startswith("_")]

                    for attr in data_attrs:
                        try:
                            register_data = getattr(pub_result.data, attr)
                            if hasattr(register_data, "get_counts"):
                                return register_data.get_counts()
                        except (AttributeError, TypeError):
                            continue

                    # If no classical register found, return empty dict
                    return {}
                else:
                    return {}
            else:
                # Handle traditional job types (V1 primitives, AerJob, etc.)
                result = job.result()
                if hasattr(result, "get_counts"):
                    return result.get_counts()
                else:
                    raise ValueError(
                        f"Job result of type {type(result)} does not have get_counts method"
                    )

        except Exception as e:
            raise ValueError(
                f"Failed to extract counts from job {self._extract_job_id(job)}: {str(e)}"
            ) from e

    def _calculate_fidelity(self, counts: Dict[str, int]) -> float:
        """
        Calculate quantum fidelity between measured and expected distributions.

        If no expected distribution is provided, returns the probability of the most
        successful outcome according to success_criteria.

        Args:
            counts: Dictionary of measurement outcomes and their counts

        Returns:
            float: Fidelity value between 0.0 and 1.0
        """
        if not counts:
            return 0.0

        total_shots = sum(counts.values())
        if total_shots == 0:
            return 0.0

        # Convert counts to probabilities
        measured_probs = {state: count / total_shots for state, count in counts.items()}

        if self.expected_distribution is not None:
            # Calculate classical fidelity: F = Σᵢ √(pᵢ × qᵢ)
            fidelity = 0.0
            all_states = set(measured_probs.keys()) | set(self.expected_distribution.keys())

            for state in all_states:
                p_measured = measured_probs.get(state, 0.0)
                p_expected = self.expected_distribution.get(state, 0.0)
                fidelity += np.sqrt(p_measured * p_expected)

            return float(fidelity)
        else:
            # Fallback: return probability of most successful state
            successful_states = [state for state in counts.keys() if self.success_criteria(state)]
            if not successful_states:
                return 0.0

            # Find the most frequent successful state
            max_successful_count = max(counts[state] for state in successful_states)
            return float(max_successful_count / total_shots)

    def set_expected_distribution(self, distribution: Dict[str, float]) -> None:
        """
        Set the expected probability distribution for fidelity calculation.

        Args:
            distribution: Dictionary mapping measurement outcomes to expected probabilities
                         (should sum to 1.0)
        """
        # Validate that probabilities sum to approximately 1.0
        total_prob = sum(distribution.values())
        if not np.isclose(total_prob, 1.0, atol=1e-6):
            raise ValueError(
                f"Expected distribution probabilities must sum to 1.0, got {total_prob}"
            )

        self.expected_distribution = distribution.copy()

    # =============================================================================
    # Main API Methods
    # =============================================================================

    def get_metrics(self) -> "CircuitPerformanceSchema":
        """
        Get all performance metrics as a structured, validated schema object.

        Returns:
            CircuitPerformanceSchema: Complete validated performance metrics schema

        Raises:
            ImportError: If Pydantic schemas are not available
            ValidationError: If metrics data doesn't match schema constraints
        """
        self._ensure_schemas_available()

        return CircuitPerformanceSchema(
            success_metrics=self.get_success_metrics(),
            fidelity_metrics=self.get_fidelity_metrics(),
            statistical_metrics=self.get_statistical_metrics(),
        )

    # =============================================================================
    # Success Metrics
    # =============================================================================

    def get_success_metrics(self) -> "SuccessMetricsSchema":
        """
        Get success rate analysis metrics as a validated schema object.

        Returns:
            SuccessMetricsSchema: Validated success metrics schema

        Raises:
            ImportError: If Pydantic schemas are not available
            ValidationError: If metrics data doesn't match schema constraints
        """
        self._ensure_schemas_available()
        success_dict = self.get_success_metrics_dict()
        return SuccessMetricsSchema(**success_dict)

    def get_success_metrics_dict(self) -> Dict[str, Any]:
        """
        Get success rate analysis metrics as a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing success rate, error rate, and shot analysis
        """
        if len(self._jobs) > 1:
            return self._get_multiple_jobs_success_metrics()
        elif len(self._jobs) == 1:
            return self._get_single_job_success_metrics(self._jobs[0])
        else:
            raise ValueError("No jobs available to calculate success metrics")

    # =============================================================================
    # Fidelity Metrics
    # =============================================================================

    def get_fidelity_metrics(self) -> "FidelityMetricsSchema":
        """
        Get quantum fidelity analysis metrics as a validated schema object.

        Returns:
            FidelityMetricsSchema: Validated fidelity metrics schema

        Raises:
            ImportError: If Pydantic schemas are not available
            ValidationError: If metrics data doesn't match schema constraints
        """
        self._ensure_schemas_available()
        fidelity_dict = self.get_fidelity_metrics_dict()
        return FidelityMetricsSchema(**fidelity_dict)

    def get_fidelity_metrics_dict(self) -> Dict[str, Any]:
        """
        Get quantum fidelity analysis metrics as a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing fidelity measurements and analysis
        """
        if len(self._jobs) > 1:
            return self._get_multiple_jobs_fidelity_metrics()
        elif len(self._jobs) == 1:
            return self._get_single_job_fidelity_metrics(self._jobs[0])
        else:
            raise ValueError("No jobs available to calculate fidelity metrics")

    # =============================================================================
    # Statistical Metrics
    # =============================================================================

    def get_statistical_metrics(self) -> "StatisticalMetricsSchema":
        """
        Get statistical analysis metrics as a validated schema object.

        Returns:
            StatisticalMetricsSchema: Validated statistical metrics schema

        Raises:
            ImportError: If Pydantic schemas are not available
            ValidationError: If metrics data doesn't match schema constraints
        """
        self._ensure_schemas_available()
        statistical_dict = self.get_statistical_metrics_dict()
        return StatisticalMetricsSchema(**statistical_dict)

    def get_statistical_metrics_dict(self) -> Dict[str, Any]:
        """
        Get statistical analysis metrics as a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing entropy, uniformity, and concentration metrics
        """
        if len(self._jobs) > 1:
            return self._get_multiple_jobs_statistical_metrics()
        elif len(self._jobs) == 1:
            return self._get_single_job_statistical_metrics(self._jobs[0])
        else:
            raise ValueError("No jobs available to calculate statistical metrics")

    # =============================================================================
    # Single Job Metrics
    # =============================================================================

    def get_single_job_metrics(self, job: Optional[JobType] = None) -> Dict[str, Any]:
        """
        Calculate circuit performance metrics from a single job result.

        DEPRECATED: Use get_success_metrics(), get_fidelity_metrics(), or get_statistical_metrics() instead.

        Args:
            job: Optional job to use for metrics calculation. If None, uses self._job.

        Returns:
            Dict[str, Any]: Circuit performance metrics for a single job
        """
        if self._job is None and job is None:
            raise ValueError("A runtime job is required to calculate circuit performance metrics")

        job_to_use = job or self._job

        # Combine all metrics for backward compatibility
        success_metrics = self._get_single_job_success_metrics(job_to_use)
        fidelity_metrics = self._get_single_job_fidelity_metrics(job_to_use)
        statistical_metrics = self._get_single_job_statistical_metrics(job_to_use)

        # Merge all metrics into a single dictionary
        combined_metrics = {**success_metrics, **fidelity_metrics, **statistical_metrics}

        return combined_metrics

    def get_structured_single_job_metrics(
        self, job: Optional[JobType] = None
    ) -> "CircuitPerformanceSchema":
        """
        Get single job metrics as a validated schema object.

        DEPRECATED: Use get_structured_metrics() instead.

        Returns:
            CircuitPerformanceSchema: Validated single job metrics
        """
        self._ensure_schemas_available()
        metrics = self.get_single_job_metrics(job)

        # Remove fields not in schema
        schema_data = {k: v for k, v in metrics.items() if k != "average_counts"}
        return CircuitPerformanceSchema(**schema_data)

    # =============================================================================
    # Multiple Jobs Metrics
    # =============================================================================

    def get_multiple_jobs_metrics(self) -> Dict[str, Any]:
        """
        Calculate circuit performance metrics from multiple job results.

        DEPRECATED: Use get_success_metrics(), get_fidelity_metrics(), or get_statistical_metrics() instead.

        Returns:
            Dict[str, Any]: Circuit performance metrics for multiple jobs with individual and aggregate data
        """
        if not self._jobs:
            raise ValueError("Multiple runtime jobs are required to calculate multiple job metrics")

        # Get metrics for each category
        success_metrics = self._get_multiple_jobs_success_metrics()
        fidelity_metrics = self._get_multiple_jobs_fidelity_metrics()
        statistical_metrics = self._get_multiple_jobs_statistical_metrics()

        # Combine individual job metrics (flatten all categories for each job)
        combined_individual_jobs = []
        for i in range(len(self._jobs)):
            combined_job_metrics = {}

            # Add success metrics for this job
            if i < len(success_metrics.get("individual_jobs", [])):
                combined_job_metrics.update(success_metrics["individual_jobs"][i])

            # Add fidelity metrics for this job
            if i < len(fidelity_metrics.get("individual_jobs", [])):
                fidelity_job_metrics = fidelity_metrics["individual_jobs"][i]
                # Remove duplicate job_id to avoid conflicts
                fidelity_job_metrics = {
                    k: v for k, v in fidelity_job_metrics.items() if k != "job_id"
                }
                combined_job_metrics.update(fidelity_job_metrics)

            # Add statistical metrics for this job
            if i < len(statistical_metrics.get("individual_jobs", [])):
                statistical_job_metrics = statistical_metrics["individual_jobs"][i]
                # Remove duplicate job_id to avoid conflicts
                statistical_job_metrics = {
                    k: v for k, v in statistical_job_metrics.items() if k != "job_id"
                }
                combined_job_metrics.update(statistical_job_metrics)

            combined_individual_jobs.append(combined_job_metrics)

        # Combine aggregate metrics
        aggregate_metrics = {}
        for metrics_dict in [success_metrics, fidelity_metrics, statistical_metrics]:
            for key, value in metrics_dict.items():
                if key != "individual_jobs":
                    aggregate_metrics[key] = value

        return {
            "aggregate": aggregate_metrics,
            "individual_jobs": combined_individual_jobs,
        }

    def get_structured_multiple_jobs_metrics(self) -> "CircuitPerformanceSchema":
        """
        Get multiple jobs metrics as a validated schema object.

        DEPRECATED: Use get_structured_metrics() instead.

        Returns:
            CircuitPerformanceSchema: Validated aggregate metrics
        """
        self._ensure_schemas_available()
        metrics = self.get_multiple_jobs_metrics()
        return CircuitPerformanceSchema(**metrics["aggregate"])

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
                if single_job not in self._jobs:
                    self._jobs.append(single_job)
                    # If this is the first job, also set it as the single job
                    if not self._job:
                        self._job = single_job
        else:
            if job not in self._jobs:
                self._jobs.append(job)
                # If this is the first job, also set it as the single job
                if not self._job:
                    self._job = job

    @staticmethod
    def create_uniform_distribution(num_qubits: int) -> Dict[str, float]:
        """
        Create a uniform distribution over all possible measurement outcomes.

        Args:
            num_qubits: Number of qubits in the circuit

        Returns:
            Dict[str, float]: Uniform probability distribution
        """
        num_states = 2**num_qubits
        prob = 1.0 / num_states
        return {format(i, f"0{num_qubits}b"): prob for i in range(num_states)}

    @staticmethod
    def create_ground_state_distribution(num_qubits: int) -> Dict[str, float]:
        """
        Create a distribution where only the ground state (|000...⟩) has probability 1.

        Args:
            num_qubits: Number of qubits in the circuit

        Returns:
            Dict[str, float]: Ground state distribution
        """
        ground_state = "0" * num_qubits
        return {ground_state: 1.0}

    @staticmethod
    def create_bell_state_distribution() -> Dict[str, float]:
        """
        Create the expected distribution for a Bell state (|00⟩ + |11⟩)/√2.

        Returns:
            Dict[str, float]: Bell state distribution
        """
        return {"00": 0.5, "11": 0.5}

    # =============================================================================
    # Helper Methods for Success Metrics
    # =============================================================================

    def _get_single_job_success_metrics(self, job: JobType) -> Dict[str, Any]:
        """Calculate success metrics from a single job result."""
        counts = self._extract_counts(job)
        job_id = self._extract_job_id(job)

        if not counts:
            return {
                "job_id": job_id,
                "success_rate": 0.0,
                "error_rate": 1.0,
                "total_shots": 0,
                "successful_shots": 0,
            }

        # Calculate basic statistics
        total_shots = sum(counts.values())
        successful_shots = sum(
            count for state, count in counts.items() if self.success_criteria(state)
        )

        # Calculate rates
        success_rate = successful_shots / total_shots if total_shots > 0 else 0.0
        error_rate = 1.0 - success_rate

        return {
            "job_id": job_id,
            "success_rate": float(success_rate),
            "error_rate": float(error_rate),
            "total_shots": total_shots,
            "successful_shots": successful_shots,
        }

    def _get_multiple_jobs_success_metrics(self) -> Dict[str, Any]:
        """Calculate success metrics from multiple job results."""
        if not self._jobs:
            raise ValueError("Multiple runtime jobs are required to calculate success metrics")

        # Calculate metrics for each job
        job_metrics = []
        success_rates = []
        total_shots_list = []

        for job in self._jobs:
            single_metrics = self._get_single_job_success_metrics(job)
            success_rates.append(single_metrics["success_rate"])
            total_shots_list.append(single_metrics["total_shots"])
            job_metrics.append(single_metrics)

        # Calculate aggregate metrics
        if not success_rates:
            return {
                "mean_success_rate": 0.0,
                "std_success_rate": 0.0,
                "min_success_rate": 0.0,
                "max_success_rate": 0.0,
                "total_trials": 0,
                "error_rate": 1.0,
                "individual_jobs": job_metrics,
            }

        success_rates_array = np.array(success_rates)
        mean_success_rate = float(np.mean(success_rates_array))
        std_success_rate = (
            float(np.std(success_rates_array)) if len(success_rates_array) > 1 else 0.0
        )
        min_success_rate = float(np.min(success_rates_array))
        max_success_rate = float(np.max(success_rates_array))
        total_trials = sum(total_shots_list)
        error_rate = 1.0 - mean_success_rate

        return {
            "mean_success_rate": mean_success_rate,
            "std_success_rate": std_success_rate,
            "min_success_rate": min_success_rate,
            "max_success_rate": max_success_rate,
            "total_trials": total_trials,
            "error_rate": error_rate,
            "individual_jobs": job_metrics,
        }

    # =============================================================================
    # Helper Methods for Fidelity Metrics
    # =============================================================================

    def _get_single_job_fidelity_metrics(self, job: JobType) -> Dict[str, Any]:
        """Calculate fidelity metrics from a single job result."""
        counts = self._extract_counts(job)
        job_id = self._extract_job_id(job)

        fidelity = self._calculate_fidelity(counts)
        method = "theoretical_comparison" if self.expected_distribution else "success_based"
        confidence = "high" if self.expected_distribution else "medium"

        return {
            "job_id": job_id,
            "fidelity": float(fidelity),
            "method": method,
            "confidence": confidence,
            "has_expected_distribution": self.expected_distribution is not None,
        }

    def _get_multiple_jobs_fidelity_metrics(self) -> Dict[str, Any]:
        """Calculate fidelity metrics from multiple job results."""
        if not self._jobs:
            raise ValueError("Multiple runtime jobs are required to calculate fidelity metrics")

        # Calculate metrics for each job
        job_metrics = []
        fidelities = []

        for job in self._jobs:
            single_metrics = self._get_single_job_fidelity_metrics(job)
            fidelities.append(single_metrics["fidelity"])
            job_metrics.append(single_metrics)

        # Calculate aggregate metrics
        if not fidelities:
            return {
                "mean_fidelity": 0.0,
                "std_fidelity": 0.0,
                "min_fidelity": 0.0,
                "max_fidelity": 0.0,
                "method": "unknown",
                "confidence": "low",
                "individual_jobs": job_metrics,
            }

        fidelities_array = np.array(fidelities)
        method = job_metrics[0]["method"] if job_metrics else "unknown"
        confidence = job_metrics[0]["confidence"] if job_metrics else "low"

        return {
            "mean_fidelity": float(np.mean(fidelities_array)),
            "std_fidelity": float(np.std(fidelities_array)) if len(fidelities_array) > 1 else 0.0,
            "min_fidelity": float(np.min(fidelities_array)),
            "max_fidelity": float(np.max(fidelities_array)),
            "method": method,
            "confidence": confidence,
            "individual_jobs": job_metrics,
        }

    # =============================================================================
    # Helper Methods for Statistical Metrics
    # =============================================================================

    def _get_single_job_statistical_metrics(self, job: JobType) -> Dict[str, Any]:
        """Calculate statistical metrics from a single job result."""
        counts = self._extract_counts(job)
        job_id = self._extract_job_id(job)

        if not counts:
            return {
                "job_id": job_id,
                "entropy": 0.0,
                "uniformity": 0.0,
                "concentration": 1.0,
                "dominant_outcome_probability": 0.0,
                "num_unique_outcomes": 0,
            }

        total_shots = sum(counts.values())
        probabilities = [count / total_shots for count in counts.values()]

        # Calculate entropy
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)

        # Calculate uniformity (how close to uniform distribution)
        num_outcomes = len(counts)
        max_entropy = np.log2(num_outcomes) if num_outcomes > 1 else 0
        uniformity = entropy / max_entropy if max_entropy > 0 else 1.0

        # Calculate concentration (opposite of uniformity)
        concentration = 1.0 - uniformity

        # Dominant outcome probability
        dominant_outcome_probability = max(probabilities) if probabilities else 0.0

        return {
            "job_id": job_id,
            "entropy": float(entropy),
            "uniformity": float(uniformity),
            "concentration": float(concentration),
            "dominant_outcome_probability": float(dominant_outcome_probability),
            "num_unique_outcomes": num_outcomes,
        }

    def _get_multiple_jobs_statistical_metrics(self) -> Dict[str, Any]:
        """Calculate statistical metrics from multiple job results."""
        if not self._jobs:
            raise ValueError("Multiple runtime jobs are required to calculate statistical metrics")

        # Calculate metrics for each job
        job_metrics = []
        entropies = []
        uniformities = []
        concentrations = []
        dominant_probs = []

        for job in self._jobs:
            single_metrics = self._get_single_job_statistical_metrics(job)
            entropies.append(single_metrics["entropy"])
            uniformities.append(single_metrics["uniformity"])
            concentrations.append(single_metrics["concentration"])
            dominant_probs.append(single_metrics["dominant_outcome_probability"])
            job_metrics.append(single_metrics)

        # Calculate aggregate metrics
        if not entropies:
            return {
                "mean_entropy": 0.0,
                "mean_uniformity": 0.0,
                "mean_concentration": 1.0,
                "mean_dominant_probability": 0.0,
                "std_entropy": 0.0,
                "individual_jobs": job_metrics,
            }

        return {
            "mean_entropy": float(np.mean(entropies)),
            "mean_uniformity": float(np.mean(uniformities)),
            "mean_concentration": float(np.mean(concentrations)),
            "mean_dominant_probability": float(np.mean(dominant_probs)),
            "std_entropy": float(np.std(entropies)) if len(entropies) > 1 else 0.0,
            "std_uniformity": float(np.std(uniformities)) if len(uniformities) > 1 else 0.0,
            "individual_jobs": job_metrics,
        }
