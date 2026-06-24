"""Post-runtime metrics for Estimator primitive results.

Computes success probabilities, observable fidelity, SNR, and
depolarization estimates from expectation values of quantum observables.
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerJob
from qiskit_ibm_runtime import RuntimeJobV2

from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsId, MetricsType
from qward.schemas.estimator_schema import EstimatorSchema

JobType = Union[AerJob, RuntimeJobV2]

EPSILON = 1e-10


class EstimatorMetrics(MetricCalculator):
    """Post-runtime metrics for Estimator primitive results.

    Supports two usage modes:
    - With raw numpy arrays (credential-free, for local simulation)
    - With a job/result object (from StatevectorEstimator or IBM Runtime)

    Example:
        # From raw values (credential-free)
        em = EstimatorMetrics(
            circuit,
            expectation_values=np.array([0.95, -0.02]),
            standard_deviations=np.array([0.01, 0.03]),
            ideal_expectation_values=np.array([1.0, 0.0]),
        )
        schema = em.get_metrics()

        # From job result
        em = EstimatorMetrics(circuit, job=estimator_job)
        schema = em.get_metrics()
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        job: Optional[Any] = None,
        expectation_values: Optional[np.ndarray] = None,
        standard_deviations: Optional[np.ndarray] = None,
        ideal_expectation_values: Optional[np.ndarray] = None,
        observable_labels: Optional[List[str]] = None,
    ):
        if expectation_values is not None and job is not None:
            raise ValueError("Provide either 'job' or 'expectation_values', not both.")

        self._job = job
        self._evs = (
            np.atleast_1d(np.asarray(expectation_values, dtype=float))
            if expectation_values is not None
            else None
        )
        self._stds = (
            np.atleast_1d(np.asarray(standard_deviations, dtype=float))
            if standard_deviations is not None
            else None
        )
        self._ideal = (
            np.atleast_1d(np.asarray(ideal_expectation_values, dtype=float))
            if ideal_expectation_values is not None
            else None
        )
        self._observable_labels = observable_labels
        super().__init__(circuit)

    def _get_metric_type(self) -> MetricsType:
        return MetricsType.POST_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        return MetricsId.ESTIMATOR

    def is_ready(self) -> bool:
        return self.circuit is not None and (self._evs is not None or self._job is not None)

    def get_metrics(self) -> EstimatorSchema:
        if not self.is_ready():
            return EstimatorSchema()

        evs, stds = self._resolve_expectation_values()
        n = len(evs)

        success_probs = self._compute_success_probabilities(evs)

        schema_data = {
            "num_observables": n,
            "expectation_values": evs.tolist(),
            "success_probabilities": success_probs.tolist(),
            "mean_expectation_value": float(np.mean(evs)),
            "min_expectation_value": float(np.min(evs)),
            "max_expectation_value": float(np.max(evs)),
            "mean_success_probability": float(np.mean(success_probs)),
        }

        if stds is not None:
            schema_data["standard_deviations"] = stds.tolist()
            snr = self._compute_snr(evs, stds)
            schema_data["signal_to_noise_ratios"] = snr.tolist()
            schema_data["mean_snr"] = (
                float(np.mean(snr[np.isfinite(snr)])) if np.any(np.isfinite(snr)) else None
            )

        if self._observable_labels is not None:
            schema_data["observable_labels"] = self._observable_labels

        if self._ideal is not None:
            ideal = self._ideal
            schema_data["ideal_expectation_values"] = ideal.tolist()
            fidelities = self._compute_observable_fidelity(evs, ideal)
            rel_errors = self._compute_relative_errors(evs, ideal)
            depol = self._compute_depolarization_factor(evs, ideal)
            schema_data["observable_fidelities"] = fidelities.tolist()
            schema_data["mean_observable_fidelity"] = float(np.mean(fidelities))
            schema_data["relative_errors"] = rel_errors.tolist()
            schema_data["mean_relative_error"] = float(np.mean(rel_errors))
            if depol is not None:
                schema_data["depolarization_factor"] = depol

        return EstimatorSchema(**schema_data)

    def _resolve_expectation_values(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self._evs is not None:
            return self._evs, self._stds
        return self._extract_estimator_results(self._job)

    def _extract_estimator_results(self, job: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        result = job.result() if hasattr(job, "result") else job

        if hasattr(result, "__getitem__"):
            pub_result = result[0]
        else:
            pub_result = result

        if not hasattr(pub_result, "data") or not hasattr(pub_result.data, "evs"):
            raise ValueError(
                "Job does not contain Estimator results. "
                "Expected 'evs' in result data. "
                "For Sampler results, use FidelityMetrics instead."
            )

        evs = np.atleast_1d(np.asarray(pub_result.data.evs, dtype=float))
        stds = None
        if hasattr(pub_result.data, "stds"):
            stds = np.atleast_1d(np.asarray(pub_result.data.stds, dtype=float))

        return evs, stds

    @staticmethod
    def _compute_success_probabilities(evs: np.ndarray) -> np.ndarray:
        return np.clip((1 + evs) / 2, 0.0, 1.0)

    @staticmethod
    def _compute_observable_fidelity(evs: np.ndarray, ideal: np.ndarray) -> np.ndarray:
        return np.clip(1.0 - np.abs(evs - ideal) / 2.0, 0.0, 1.0)

    @staticmethod
    def _compute_relative_errors(evs: np.ndarray, ideal: np.ndarray) -> np.ndarray:
        denom = np.maximum(np.abs(ideal), EPSILON)
        return np.abs(evs - ideal) / denom

    @staticmethod
    def _compute_snr(evs: np.ndarray, stds: np.ndarray) -> np.ndarray:
        safe_stds = np.where(stds > EPSILON, stds, np.inf)
        return np.abs(evs) / safe_stds

    @staticmethod
    def _compute_depolarization_factor(evs: np.ndarray, ideal: np.ndarray) -> Optional[float]:
        mask = np.abs(ideal) > EPSILON
        if not np.any(mask):
            return None
        ratios = evs[mask] / ideal[mask]
        factor = float(np.mean(ratios))
        return max(0.0, min(1.0, factor))
