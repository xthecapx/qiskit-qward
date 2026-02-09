"""
Grover Experiment Success Metrics

This module implements various success metrics for evaluating Grover's algorithm
at different levels: per-shot, per-job, and per-batch.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np
from scipy import stats

# =============================================================================
# Level 1: Per-Shot Success
# =============================================================================


def success_per_shot(outcome: str, marked_states: List[str]) -> bool:
    """
    Binary success: did this single measurement hit a marked state?

    Args:
        outcome: Single measurement outcome string (e.g., "011")
        marked_states: List of target states

    Returns:
        bool: True if outcome is a marked state
    """
    # Clean the outcome string (remove spaces if any)
    clean_outcome = outcome.replace(" ", "")
    return clean_outcome in marked_states


def success_rate(counts: Dict[str, int], marked_states: List[str]) -> float:
    """
    Calculate the fraction of shots that hit marked states.

    Args:
        counts: Dictionary of measurement outcomes and their counts
        marked_states: List of target states

    Returns:
        float: Success rate (0.0 to 1.0)
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0

    success_count = sum(counts.get(state, 0) for state in marked_states)
    return success_count / total


def success_count(counts: Dict[str, int], marked_states: List[str]) -> int:
    """
    Count the number of successful shots.

    Args:
        counts: Dictionary of measurement outcomes and their counts
        marked_states: List of target states

    Returns:
        int: Number of successful shots
    """
    return sum(counts.get(state, 0) for state in marked_states)


# =============================================================================
# Level 2: Per-Job Success
# =============================================================================


def job_success_threshold(
    counts: Dict[str, int], marked_states: List[str], threshold: float = 0.5
) -> bool:
    """
    Job succeeds if success rate exceeds threshold.

    Args:
        counts: Dictionary of measurement outcomes and their counts
        marked_states: List of target states
        threshold: Minimum success rate required (default: 0.5)

    Returns:
        bool: True if success rate >= threshold
    """
    rate = success_rate(counts, marked_states)
    return rate >= threshold


def job_success_statistical(
    counts: Dict[str, int], marked_states: List[str], num_qubits: int, confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Job succeeds if success rate is statistically above random chance.

    Uses binomial test to determine if observed success is significantly
    better than what would be expected from random measurement.

    Args:
        counts: Dictionary of measurement outcomes and their counts
        marked_states: List of target states
        num_qubits: Number of qubits in the circuit
        confidence: Confidence level for the test (default: 0.95)

    Returns:
        dict: Contains success flag, p-value, and interpretation
    """
    total = sum(counts.values())
    success = sum(counts.get(state, 0) for state in marked_states)

    # Random chance baseline
    random_prob = len(marked_states) / (2**num_qubits)

    # Binomial test: is observed success significantly better than random?
    result = stats.binomtest(success, total, random_prob, alternative="greater")

    is_success = result.pvalue < (1 - confidence)

    return {
        "success": is_success,
        "p_value": result.pvalue,
        "observed_rate": success / total if total > 0 else 0,
        "random_prob": random_prob,
        "confidence": confidence,
        "interpretation": (
            f"Success rate significantly above random (p={result.pvalue:.4f})"
            if is_success
            else f"Success rate not significantly above random (p={result.pvalue:.4f})"
        ),
    }


def job_success_quantum_advantage(
    counts: Dict[str, int], marked_states: List[str], num_qubits: int, advantage_factor: float = 2.0
) -> Dict[str, Any]:
    """
    Job succeeds if algorithm outperforms classical random search by factor X.

    Args:
        counts: Dictionary of measurement outcomes and their counts
        marked_states: List of target states
        num_qubits: Number of qubits in the circuit
        advantage_factor: Required improvement over classical (default: 2.0)

    Returns:
        dict: Contains success flag, advantage ratio, and interpretation
    """
    rate = success_rate(counts, marked_states)

    # Classical random search expected success
    classical_expected = len(marked_states) / (2**num_qubits)

    # Calculate advantage ratio
    advantage_ratio = rate / classical_expected if classical_expected > 0 else float("inf")

    is_success = rate > (classical_expected * advantage_factor)

    return {
        "success": is_success,
        "observed_rate": rate,
        "classical_expected": classical_expected,
        "advantage_ratio": advantage_ratio,
        "required_factor": advantage_factor,
        "interpretation": (
            f"Quantum advantage achieved: {advantage_ratio:.2f}x classical"
            if is_success
            else f"No quantum advantage: {advantage_ratio:.2f}x classical (need {advantage_factor}x)"
        ),
    }


def job_success_theoretical(
    counts: Dict[str, int],
    marked_states: List[str],
    theoretical_prob: float,
    tolerance: float = 0.1,
) -> Dict[str, Any]:
    """
    Job succeeds if observed rate is within tolerance of theoretical probability.

    Args:
        counts: Dictionary of measurement outcomes and their counts
        marked_states: List of target states
        theoretical_prob: Expected theoretical success probability
        tolerance: Allowed deviation from theoretical (default: 0.1 = 10%)

    Returns:
        dict: Contains success flag, deviation, and interpretation
    """
    rate = success_rate(counts, marked_states)

    deviation = abs(rate - theoretical_prob)
    relative_deviation = deviation / theoretical_prob if theoretical_prob > 0 else float("inf")

    is_success = deviation <= tolerance

    return {
        "success": is_success,
        "observed_rate": rate,
        "theoretical_prob": theoretical_prob,
        "absolute_deviation": deviation,
        "relative_deviation": relative_deviation,
        "tolerance": tolerance,
        "interpretation": (
            f"Within tolerance: {rate:.3f} vs {theoretical_prob:.3f} (±{deviation:.3f})"
            if is_success
            else f"Outside tolerance: {rate:.3f} vs {theoretical_prob:.3f} (±{deviation:.3f})"
        ),
    }


# =============================================================================
# Level 3: Per-Batch Success
# =============================================================================


def batch_success_mean(
    job_results: List[Dict[str, int]], marked_states: List[str], threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Batch succeeds if mean success rate across jobs exceeds threshold.

    Args:
        job_results: List of counts dicts from multiple job executions
        marked_states: List of target states
        threshold: Minimum mean success rate required

    Returns:
        dict: Contains success flag, mean rate, and statistics
    """
    rates = [success_rate(job, marked_states) for job in job_results]

    mean_rate = np.mean(rates)
    is_success = mean_rate >= threshold

    return {
        "success": is_success,
        "mean_rate": mean_rate,
        "std_rate": np.std(rates, ddof=1) if len(rates) > 1 else 0,
        "min_rate": np.min(rates),
        "max_rate": np.max(rates),
        "threshold": threshold,
        "num_jobs": len(job_results),
        "interpretation": (
            f"Batch passed: mean={mean_rate:.3f} >= {threshold}"
            if is_success
            else f"Batch failed: mean={mean_rate:.3f} < {threshold}"
        ),
    }


def batch_success_min(
    job_results: List[Dict[str, int]], marked_states: List[str], threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Batch succeeds if ALL jobs meet the threshold (worst-case guarantee).

    Args:
        job_results: List of counts dicts from multiple job executions
        marked_states: List of target states
        threshold: Minimum success rate required for EVERY job

    Returns:
        dict: Contains success flag, min rate, and statistics
    """
    rates = [success_rate(job, marked_states) for job in job_results]

    min_rate = np.min(rates)
    is_success = min_rate >= threshold

    return {
        "success": is_success,
        "min_rate": min_rate,
        "mean_rate": np.mean(rates),
        "jobs_passing": sum(1 for r in rates if r >= threshold),
        "total_jobs": len(job_results),
        "threshold": threshold,
        "interpretation": (
            f"All jobs passed: min={min_rate:.3f} >= {threshold}"
            if is_success
            else f"Some jobs failed: min={min_rate:.3f} < {threshold}"
        ),
    }


def batch_success_median(
    job_results: List[Dict[str, int]], marked_states: List[str], threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Batch succeeds if median success rate exceeds threshold (robust to outliers).

    Args:
        job_results: List of counts dicts from multiple job executions
        marked_states: List of target states
        threshold: Minimum median success rate required

    Returns:
        dict: Contains success flag, median rate, and statistics
    """
    rates = [success_rate(job, marked_states) for job in job_results]

    median_rate = np.median(rates)
    is_success = median_rate >= threshold

    return {
        "success": is_success,
        "median_rate": median_rate,
        "mean_rate": np.mean(rates),
        "q1": np.percentile(rates, 25),
        "q3": np.percentile(rates, 75),
        "threshold": threshold,
        "interpretation": (
            f"Batch passed (median): {median_rate:.3f} >= {threshold}"
            if is_success
            else f"Batch failed (median): {median_rate:.3f} < {threshold}"
        ),
    }


def batch_success_consistency(
    job_results: List[Dict[str, int]], marked_states: List[str], max_std: float = 0.1
) -> Dict[str, Any]:
    """
    Batch succeeds if results are consistent (low variance).

    Args:
        job_results: List of counts dicts from multiple job executions
        marked_states: List of target states
        max_std: Maximum allowed standard deviation

    Returns:
        dict: Contains success flag, std, and statistics
    """
    rates = [success_rate(job, marked_states) for job in job_results]

    std_rate = np.std(rates, ddof=1) if len(rates) > 1 else 0
    cv = std_rate / np.mean(rates) if np.mean(rates) > 0 else float("inf")

    is_success = std_rate <= max_std

    return {
        "success": is_success,
        "std_rate": std_rate,
        "cv": cv,  # Coefficient of variation
        "mean_rate": np.mean(rates),
        "range": np.max(rates) - np.min(rates),
        "max_std": max_std,
        "interpretation": (
            f"Consistent results: std={std_rate:.3f} <= {max_std}"
            if is_success
            else f"Inconsistent results: std={std_rate:.3f} > {max_std}"
        ),
    }


# =============================================================================
# Comprehensive Evaluation
# =============================================================================


@dataclass
class SuccessEvaluation:
    """Complete success evaluation across all levels and metrics."""

    # Configuration
    config_id: str
    marked_states: List[str]
    num_qubits: int
    theoretical_prob: float

    # Per-Shot
    success_rate: float
    success_count: int
    total_shots: int

    # Per-Job (all thresholds)
    threshold_30: bool
    threshold_50: bool
    threshold_70: bool
    threshold_90: bool
    threshold_theoretical: bool

    # Per-Job (statistical)
    statistical_success: bool
    statistical_pvalue: float

    # Per-Job (quantum advantage)
    quantum_advantage_success: bool
    advantage_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for data storage."""
        return {
            "config_id": self.config_id,
            "marked_states": self.marked_states,
            "num_qubits": self.num_qubits,
            "theoretical_prob": self.theoretical_prob,
            "success_rate": self.success_rate,
            "success_count": self.success_count,
            "total_shots": self.total_shots,
            "threshold_30": self.threshold_30,
            "threshold_50": self.threshold_50,
            "threshold_70": self.threshold_70,
            "threshold_90": self.threshold_90,
            "threshold_theoretical": self.threshold_theoretical,
            "statistical_success": self.statistical_success,
            "statistical_pvalue": self.statistical_pvalue,
            "quantum_advantage_success": self.quantum_advantage_success,
            "advantage_ratio": self.advantage_ratio,
        }


def evaluate_job(
    counts: Dict[str, int],
    marked_states: List[str],
    num_qubits: int,
    theoretical_prob: float,
    config_id: str = "",
) -> SuccessEvaluation:
    """
    Comprehensive evaluation of a single job using all success metrics.

    Args:
        counts: Dictionary of measurement outcomes and their counts
        marked_states: List of target states
        num_qubits: Number of qubits in the circuit
        theoretical_prob: Expected theoretical success probability
        config_id: Configuration identifier

    Returns:
        SuccessEvaluation: Complete evaluation across all metrics
    """
    rate = success_rate(counts, marked_states)
    s_count = success_count(counts, marked_states)
    total = sum(counts.values())

    # Statistical evaluation
    stat_result = job_success_statistical(counts, marked_states, num_qubits)

    # Quantum advantage evaluation
    qa_result = job_success_quantum_advantage(counts, marked_states, num_qubits)

    return SuccessEvaluation(
        config_id=config_id,
        marked_states=marked_states,
        num_qubits=num_qubits,
        theoretical_prob=theoretical_prob,
        success_rate=rate,
        success_count=s_count,
        total_shots=total,
        threshold_30=job_success_threshold(counts, marked_states, 0.3),
        threshold_50=job_success_threshold(counts, marked_states, 0.5),
        threshold_70=job_success_threshold(counts, marked_states, 0.7),
        threshold_90=job_success_threshold(counts, marked_states, 0.9),
        threshold_theoretical=job_success_threshold(counts, marked_states, theoretical_prob * 0.9),
        statistical_success=stat_result["success"],
        statistical_pvalue=stat_result["p_value"],
        quantum_advantage_success=qa_result["success"],
        advantage_ratio=qa_result["advantage_ratio"],
    )


def evaluate_batch(
    job_results: List[Dict[str, int]],
    marked_states: List[str],
    num_qubits: int,
    theoretical_prob: float,
    config_id: str = "",
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a batch of jobs.

    Args:
        job_results: List of counts dicts from multiple job executions
        marked_states: List of target states
        num_qubits: Number of qubits in the circuit
        theoretical_prob: Expected theoretical success probability
        config_id: Configuration identifier

    Returns:
        dict: Complete batch evaluation with all metrics
    """
    # Evaluate each job
    job_evaluations = [
        evaluate_job(counts, marked_states, num_qubits, theoretical_prob, config_id)
        for counts in job_results
    ]

    rates = [e.success_rate for e in job_evaluations]

    # Batch-level evaluations
    mean_result = batch_success_mean(job_results, marked_states)
    min_result = batch_success_min(job_results, marked_states)
    median_result = batch_success_median(job_results, marked_states)
    consistency_result = batch_success_consistency(job_results, marked_states)

    return {
        "config_id": config_id,
        "num_jobs": len(job_results),
        "num_qubits": num_qubits,
        "theoretical_prob": theoretical_prob,
        # Aggregate statistics
        "mean_success_rate": np.mean(rates),
        "std_success_rate": np.std(rates, ddof=1) if len(rates) > 1 else 0,
        "min_success_rate": np.min(rates),
        "max_success_rate": np.max(rates),
        "median_success_rate": np.median(rates),
        # Batch success metrics
        "batch_mean_50": mean_result["success"],
        "batch_mean_70": batch_success_mean(job_results, marked_states, 0.7)["success"],
        "batch_min_50": min_result["success"],
        "batch_median_50": median_result["success"],
        "batch_consistency": consistency_result["success"],
        # Job-level aggregates
        "jobs_passing_50": sum(1 for e in job_evaluations if e.threshold_50),
        "jobs_passing_70": sum(1 for e in job_evaluations if e.threshold_70),
        "jobs_statistical_success": sum(1 for e in job_evaluations if e.statistical_success),
        "jobs_quantum_advantage": sum(1 for e in job_evaluations if e.quantum_advantage_success),
        # Individual job evaluations
        "job_evaluations": [e.to_dict() for e in job_evaluations],
    }


# =============================================================================
# Utility Functions
# =============================================================================


def print_evaluation_summary(evaluation: Dict[str, Any]) -> None:
    """Print a formatted summary of batch evaluation results."""
    print("=" * 60)
    print(f"BATCH EVALUATION: {evaluation['config_id']}")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Jobs: {evaluation['num_jobs']}")
    print(f"  Qubits: {evaluation['num_qubits']}")
    print(f"  Theoretical P(success): {evaluation['theoretical_prob']:.3f}")

    print(f"\nSuccess Rate Statistics:")
    print(f"  Mean:   {evaluation['mean_success_rate']:.4f}")
    print(f"  Std:    {evaluation['std_success_rate']:.4f}")
    print(f"  Min:    {evaluation['min_success_rate']:.4f}")
    print(f"  Max:    {evaluation['max_success_rate']:.4f}")
    print(f"  Median: {evaluation['median_success_rate']:.4f}")

    print(f"\nBatch-Level Success:")
    print(f"  Mean >= 50%:      {'✓' if evaluation['batch_mean_50'] else '✗'}")
    print(f"  Mean >= 70%:      {'✓' if evaluation['batch_mean_70'] else '✗'}")
    print(f"  Min >= 50%:       {'✓' if evaluation['batch_min_50'] else '✗'}")
    print(f"  Median >= 50%:    {'✓' if evaluation['batch_median_50'] else '✗'}")
    print(f"  Consistent:       {'✓' if evaluation['batch_consistency'] else '✗'}")

    print(f"\nJob-Level Summary:")
    print(f"  Jobs passing 50%: {evaluation['jobs_passing_50']}/{evaluation['num_jobs']}")
    print(f"  Jobs passing 70%: {evaluation['jobs_passing_70']}/{evaluation['num_jobs']}")
    print(
        f"  Statistical success: {evaluation['jobs_statistical_success']}/{evaluation['num_jobs']}"
    )
    print(f"  Quantum advantage: {evaluation['jobs_quantum_advantage']}/{evaluation['num_jobs']}")

    print("=" * 60)
