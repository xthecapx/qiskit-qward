"""Fidelity metrics for quantum circuit output validation.

Computes DSR (Michelson), Hellinger fidelity, TVD, and success rate
from execution results. Accepts either a job object or raw counts dictionary.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerJob
from qiskit_ibm_runtime import RuntimeJobV2

from qward.metrics.base_metric import MetricCalculator
from qward.metrics.differential_success_rate import compute_dsr_with_flags
from qward.metrics.types import MetricsId, MetricsType
from qward.schemas.fidelity_schema import FidelitySchema

JobType = Union[AerJob, RuntimeJobV2]


class FidelityMetrics(MetricCalculator):
    """Post-runtime fidelity metrics for quantum circuit outputs.

    Supports two usage modes:
    - With a job object (AerJob or RuntimeJobV2)
    - With raw counts dictionary (no credentials needed)

    Example:
        # From counts (credential-free)
        fm = FidelityMetrics(circuit, counts={"00": 900, "11": 100}, target_state="00")
        schema = fm.get_metrics()

        # From job
        fm = FidelityMetrics(circuit, job=aer_job, expected_outcomes=["00"])
        schema = fm.get_metrics()
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        job: Optional[JobType] = None,
        jobs: Optional[List[JobType]] = None,
        counts: Optional[Dict[str, int]] = None,
        expected_outcomes: Optional[List[str]] = None,
        target_histogram: Optional[Dict[str, float]] = None,
        target_state: Optional[str] = None,
        success_criteria: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(circuit)
        if counts is not None and (job is not None or jobs is not None):
            raise ValueError("Provide either 'job'/'jobs' or 'counts', not both.")

        self._counts = counts
        self._success_criteria = success_criteria
        self._jobs: List[JobType] = list(jobs) if jobs else []
        if job is not None and job not in self._jobs:
            self._jobs.append(job)
        self._job = self._jobs[0] if self._jobs else None

        if target_state is not None:
            if expected_outcomes is None:
                expected_outcomes = [target_state]
            if target_histogram is None:
                target_histogram = {target_state: 1.0}

        self._expected_outcomes = expected_outcomes
        self._target_histogram = target_histogram

    def _get_metric_type(self) -> MetricsType:
        return MetricsType.POST_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        return MetricsId.FIDELITY

    def is_ready(self) -> bool:
        return self.circuit is not None and (len(self._jobs) > 0 or self._counts is not None)

    def get_metrics(self) -> FidelitySchema:
        """Compute all fidelity metrics and return validated schema."""
        counts = self._resolve_counts()
        if not counts:
            return FidelitySchema()
        return self._compute_schema(counts)

    def get_metrics_all(self) -> List[FidelitySchema]:
        """Compute fidelity metrics for each job. Returns one schema per job.

        When using counts (no jobs), returns a single-element list.
        """
        if self._counts is not None or len(self._jobs) <= 1:
            return [self.get_metrics()]

        schemas = []
        for job in self._jobs:
            counts = self._normalize_keys(self._extract_counts(job))
            if not counts:
                schemas.append(FidelitySchema())
                continue
            schemas.append(self._compute_schema(counts))
        return schemas

    def add_job(self, job: Union[JobType, List[JobType]]) -> None:
        """Add one or more jobs for multi-job analysis."""
        if isinstance(job, list):
            for j in job:
                if j not in self._jobs:
                    self._jobs.append(j)
        elif job not in self._jobs:
            self._jobs.append(job)
        if not self._job and self._jobs:
            self._job = self._jobs[0]

    def _compute_schema(self, counts: Dict[str, int]) -> FidelitySchema:
        """Compute fidelity schema from already-normalized counts."""
        total = sum(counts.values())
        unique = len(counts)

        result: Dict[str, Any] = {
            "shots": total,
            "unique_outcomes": unique,
        }

        if self._expected_outcomes:
            result["expected_outcomes"] = self._expected_outcomes
            dsr, peak_mismatch = compute_dsr_with_flags(counts, self._expected_outcomes)
            result["dsr"] = round(dsr, 6)
            result["peak_mismatch"] = peak_mismatch

            success_count = sum(counts.get(o, 0) for o in self._expected_outcomes)
            result["success_rate"] = round(success_count / total, 6) if total > 0 else None

        if self._success_criteria and not self._expected_outcomes:
            success_count = sum(c for state, c in counts.items() if self._success_criteria(state))
            result["success_rate"] = round(success_count / total, 6) if total > 0 else None

        if self._target_histogram:
            observed = self._normalize_counts(counts)
            hf = self._hellinger_fidelity(observed, self._target_histogram)
            tvd = self._total_variation_distance(observed, self._target_histogram)
            result["hellinger_fidelity"] = round(hf, 6)
            result["tvd"] = round(tvd, 6)
            result["tvd_fidelity"] = round(1.0 - tvd, 6)

        return FidelitySchema(**result)

    def _resolve_counts(self) -> Dict[str, int]:
        """Get counts from either direct input or job extraction.

        Normalizes bitstring keys by stripping spaces (Aer uses "11 00" format
        when measure_all() adds a second classical register).
        """
        if self._counts is not None:
            raw = self._counts
        elif self._job is not None:
            raw = self._extract_counts(self._job)
        else:
            return {}

        return self._normalize_keys(raw)

    @staticmethod
    def _normalize_keys(counts: Dict[str, int]) -> Dict[str, int]:
        """Strip spaces from bitstring keys and merge duplicates."""
        normalized: Dict[str, int] = {}
        for key, val in counts.items():
            clean = key.replace(" ", "")
            normalized[clean] = normalized.get(clean, 0) + val
        return normalized

    def _extract_counts(self, job: JobType) -> Dict[str, int]:
        """Extract counts from job, handling different job types."""
        try:
            if hasattr(job, "data") and hasattr(job.data, "get_counts"):
                return job.data.get_counts()

            if hasattr(job, "result"):
                result = job.result()
                if hasattr(result, "data") and hasattr(result.data, "get_counts"):
                    return result.data.get_counts()

            if isinstance(job, RuntimeJobV2):
                return self._extract_runtime_v2_counts(job)

            result = job.result()
            if hasattr(result, "get_counts"):
                return result.get_counts()

            return {}
        except Exception:
            return {}

    def _extract_runtime_v2_counts(self, job: RuntimeJobV2) -> Dict[str, int]:
        """Extract counts from RuntimeJobV2."""
        result = job.result()
        if len(result) == 0:
            return {}

        pub_result = result[0]
        for name in ["meas", "c", "cr", "classical"]:
            if hasattr(pub_result.data, name):
                register_data = getattr(pub_result.data, name)
                if hasattr(register_data, "get_counts"):
                    return register_data.get_counts()

        for attr in dir(pub_result.data):
            if attr.startswith("_"):
                continue
            try:
                register_data = getattr(pub_result.data, attr)
                if hasattr(register_data, "get_counts"):
                    return register_data.get_counts()
            except (AttributeError, TypeError):
                continue
        return {}

    @staticmethod
    def _normalize_counts(counts: Dict[str, int]) -> Dict[str, float]:
        """Convert raw counts to probability distribution."""
        total = sum(counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in counts.items()}

    @staticmethod
    def _hellinger_fidelity(p: Dict[str, float], q: Dict[str, float]) -> float:
        """HF = (sum_i sqrt(p_i * q_i))^2."""
        all_keys = set(p.keys()) | set(q.keys())
        bc = sum(np.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in all_keys)
        return float(min(1.0, bc**2))

    @staticmethod
    def _total_variation_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
        """TVD = 0.5 * sum_i |p_i - q_i|."""
        all_keys = set(p.keys()) | set(q.keys())
        return float(0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in all_keys))
