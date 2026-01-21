"""
Grover Experiment Runner

This module provides the experiment runner for systematic evaluation
of Grover's algorithm under various configurations and noise models.

Uses the BaseExperimentRunner framework for consistent workflow with
incremental saving, resume support, and QWARD metrics integration.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from qiskit import QuantumCircuit

# Local imports
from .grover_configs import (
    ExperimentConfig,
    NoiseConfig,
    get_config,
    get_noise_config,
    get_configs_by_type,
    ALL_EXPERIMENT_CONFIGS,
    NOISE_CONFIGS,
    CONFIGS_BY_ID,
    NOISE_BY_ID,
)
from .grover_success_metrics import (
    success_rate,
    success_count,
    evaluate_job,
)
from .grover_statistical_analysis import (
    analyze_config_results,
    ConfigAnalysis,
    print_analysis_summary,
)

# QWARD imports
from qward.algorithms import GroverCircuitGenerator
from qward.algorithms.experiment import (
    BaseExperimentResult,
    BaseBatchResult,
    BaseExperimentRunner,
)


# =============================================================================
# Experiment Parameters
# =============================================================================

SHOTS = 1024
NUM_RUNS = 10
OPTIMIZATION_LEVEL = 0


# =============================================================================
# Grover-Specific Result Classes
# =============================================================================


@dataclass
class GroverExperimentResult(BaseExperimentResult):
    """Result from a single Grover experiment run."""

    # Grover-specific properties
    marked_states: List[str] = None
    num_marked: int = 0
    theoretical_success: float = 0.0
    grover_iterations: int = 0

    # Success metrics
    threshold_30: bool = False
    threshold_50: bool = False
    threshold_70: bool = False
    threshold_90: bool = False
    statistical_success: bool = False
    statistical_pvalue: float = 1.0
    quantum_advantage: bool = False
    advantage_ratio: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroverExperimentResult":
        """Reconstruct from dictionary."""
        return cls(
            experiment_id=data.get("experiment_id", ""),
            config_id=data.get("config_id", ""),
            noise_model=data.get("noise_model", ""),
            run_number=data.get("run_number", 0),
            timestamp=data.get("timestamp", ""),
            backend_type=data.get("backend_type", "simulator"),
            backend_name=data.get("backend_name", "AerSimulator"),
            num_qubits=data.get("num_qubits", 0),
            circuit_depth=data.get("circuit_depth", 0),
            total_gates=data.get("total_gates", 0),
            qward_metrics=data.get("qward_metrics"),
            shots=data.get("shots", SHOTS),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            counts=data.get("counts", {}),
            success_rate=data.get("success_rate", 0.0),
            success_count=data.get("success_count", 0),
            marked_states=data.get("marked_states", []),
            num_marked=data.get("num_marked", 0),
            theoretical_success=data.get("theoretical_success", 0.0),
            grover_iterations=data.get("grover_iterations", 0),
            threshold_30=data.get("threshold_30", False),
            threshold_50=data.get("threshold_50", False),
            threshold_70=data.get("threshold_70", False),
            threshold_90=data.get("threshold_90", False),
            statistical_success=data.get("statistical_success", False),
            statistical_pvalue=data.get("statistical_pvalue", 1.0),
            quantum_advantage=data.get("quantum_advantage", False),
            advantage_ratio=data.get("advantage_ratio", 0.0),
        )


@dataclass
class GroverBatchResult(BaseBatchResult[GroverExperimentResult, ConfigAnalysis]):
    """Result from running multiple Grover experiment runs."""

    pass  # Uses base implementation


# =============================================================================
# Grover Experiment Runner
# =============================================================================


class GroverExperimentRunner(BaseExperimentRunner[
    ExperimentConfig, GroverExperimentResult, GroverBatchResult, ConfigAnalysis
]):
    """
    Experiment runner for Grover's algorithm.
    
    Provides systematic evaluation of Grover under various configurations
    and noise models with incremental saving and resume support.
    """

    def __init__(
        self,
        shots: int = SHOTS,
        num_runs: int = NUM_RUNS,
        optimization_level: int = OPTIMIZATION_LEVEL,
        output_dir: str = "data/simulator",
        backend_type: str = "simulator",
        backend_name: str = "AerSimulator",
    ):
        # Resolve output_dir relative to this file's location
        base_dir = Path(__file__).parent / output_dir
        super().__init__(
            shots=shots,
            num_runs=num_runs,
            optimization_level=optimization_level,
            output_dir=str(base_dir),
            backend_type=backend_type,
            backend_name=backend_name,
        )

    @property
    def algorithm_name(self) -> str:
        return "GROVER"

    def create_circuit(
        self, config: ExperimentConfig
    ) -> Tuple[QuantumCircuit, GroverCircuitGenerator]:
        """Create Grover circuit based on configuration."""
        grover_gen = GroverCircuitGenerator(
            marked_states=config.marked_states,
            use_barriers=True,
        )
        return grover_gen.circuit, grover_gen

    def calculate_success(
        self,
        counts: Dict[str, int],
        config: ExperimentConfig,
        circuit_metadata: GroverCircuitGenerator,
    ) -> Tuple[float, int]:
        """Calculate Grover success metrics."""
        rate = success_rate(counts, config.marked_states)
        s_count = success_count(counts, config.marked_states)
        return rate, s_count

    def create_result(
        self,
        config: ExperimentConfig,
        noise_config: NoiseConfig,
        run_number: int,
        transpiled_circuit: QuantumCircuit,
        counts: Dict[str, int],
        execution_time_ms: float,
        success_rate: float,
        success_count: int,
        qward_metrics: Optional[Dict[str, Any]],
        circuit_metadata: GroverCircuitGenerator,
        backend_type: str,
        backend_name: str,
    ) -> GroverExperimentResult:
        """Create Grover experiment result."""
        experiment_id = f"{config.config_id}_{noise_config.noise_id}_{run_number:03d}"

        # Evaluate with all success metrics
        evaluation = evaluate_job(
            counts=counts,
            marked_states=config.marked_states,
            num_qubits=config.num_qubits,
            theoretical_prob=config.theoretical_success,
            config_id=config.config_id,
        )

        return GroverExperimentResult(
            experiment_id=experiment_id,
            config_id=config.config_id,
            noise_model=noise_config.noise_id,
            run_number=run_number,
            timestamp=datetime.now().isoformat(),
            backend_type=backend_type,
            backend_name=backend_name,
            num_qubits=config.num_qubits,
            circuit_depth=transpiled_circuit.depth(),
            total_gates=sum(transpiled_circuit.count_ops().values()),
            qward_metrics=qward_metrics,
            shots=self.shots,
            execution_time_ms=execution_time_ms,
            counts=counts,
            success_rate=success_rate,
            success_count=success_count,
            marked_states=config.marked_states,
            num_marked=config.num_marked,
            theoretical_success=config.theoretical_success,
            grover_iterations=config.theoretical_iterations,
            threshold_30=evaluation.threshold_30,
            threshold_50=evaluation.threshold_50,
            threshold_70=evaluation.threshold_70,
            threshold_90=evaluation.threshold_90,
            statistical_success=evaluation.statistical_success,
            statistical_pvalue=evaluation.statistical_pvalue,
            quantum_advantage=evaluation.quantum_advantage_success,
            advantage_ratio=evaluation.advantage_ratio,
        )

    def get_config(self, config_id: str) -> ExperimentConfig:
        """Get experiment configuration by ID."""
        return get_config(config_id)

    def get_noise_config(self, noise_id: str) -> NoiseConfig:
        """Get noise configuration by ID."""
        return get_noise_config(noise_id)

    def get_all_config_ids(self) -> List[str]:
        """Get all available configuration IDs."""
        return list(CONFIGS_BY_ID.keys())

    def get_all_noise_ids(self) -> List[str]:
        """Get all available noise model IDs."""
        return list(NOISE_BY_ID.keys())

    def get_config_description(self, config: ExperimentConfig) -> str:
        """Get description for config in verbose output."""
        return f"Qubits: {config.num_qubits}, Marked: {config.num_marked}"

    def analyze_batch(
        self,
        success_rates: List[float],
        config_id: str,
        noise_model: str,
        ideal_rates: Optional[List[float]] = None,
    ) -> Optional[ConfigAnalysis]:
        """Perform statistical analysis on Grover batch results."""
        return analyze_config_results(
            success_rates=success_rates,
            config_id=config_id,
            noise_model=noise_model,
            ideal_rates=ideal_rates,
        )

    def print_batch_analysis(self, analysis: Optional[ConfigAnalysis]) -> None:
        """Print Grover batch analysis summary."""
        if analysis is not None:
            print_analysis_summary(analysis)

    def load_result_from_dict(self, data: Dict[str, Any]) -> GroverExperimentResult:
        """Reconstruct Grover result from dictionary."""
        return GroverExperimentResult.from_dict(data)

    def load_analysis_from_dict(
        self, data: Optional[Dict[str, Any]]
    ) -> Optional[ConfigAnalysis]:
        """Reconstruct ConfigAnalysis from dictionary."""
        if data is None:
            return None
        return ConfigAnalysis(
            config_id=data.get("config_id", ""),
            noise_model=data.get("noise_model", ""),
            num_runs=data.get("num_runs", 0),
            mean=data.get("mean", 0.0),
            std=data.get("std", 0.0),
            median=data.get("median", 0.0),
            min_val=data.get("min", data.get("min_val", 0.0)),
            max_val=data.get("max", data.get("max_val", 0.0)),
            ci_lower=data.get("ci_lower", 0.0),
            ci_upper=data.get("ci_upper", 0.0),
            skewness=data.get("skewness", 0.0),
            kurtosis=data.get("kurtosis", 0.0),
            is_normal=data.get("is_normal", False),
            normality_pvalue=data.get("normality_pvalue"),
            degradation_from_ideal=data.get("degradation_from_ideal", 0.0),
            cohens_d_vs_ideal=data.get("cohens_d_vs_ideal", 0.0),
            effect_size_vs_ideal=data.get("effect_size_vs_ideal", ""),
        )


# =============================================================================
# Module-Level Convenience Functions
# =============================================================================

# Default runner instance
_default_runner: Optional[GroverExperimentRunner] = None


def _get_runner() -> GroverExperimentRunner:
    """Get or create the default runner instance."""
    global _default_runner
    if _default_runner is None:
        _default_runner = GroverExperimentRunner()
    return _default_runner


def run_single_experiment(
    exp_config: ExperimentConfig,
    noise_config: NoiseConfig,
    run_number: int,
    shots: int = SHOTS,
    calculate_qward: bool = True,
) -> GroverExperimentResult:
    """
    Run a single experiment with given configuration.
    
    Convenience function for backward compatibility.
    """
    runner = _get_runner()
    runner.shots = shots
    return runner.run_single(exp_config, noise_config, run_number, shots, calculate_qward)


def run_batch(
    config_id: str,
    noise_id: str,
    num_runs: int = NUM_RUNS,
    shots: int = SHOTS,
    ideal_rates: Optional[List[float]] = None,
    verbose: bool = True,
) -> GroverBatchResult:
    """
    Run multiple experiments with the same configuration.
    
    Convenience function for backward compatibility.
    """
    runner = _get_runner()
    runner.shots = shots
    runner.num_runs = num_runs
    return runner.run_batch(config_id, noise_id, num_runs, shots, ideal_rates, verbose)


def run_experiment_campaign(
    config_ids: Optional[List[str]] = None,
    noise_ids: Optional[List[str]] = None,
    num_runs: int = NUM_RUNS,
    shots: int = SHOTS,
    save_results: bool = True,
    output_dir: str = "data/simulator",
    verbose: bool = True,
    incremental_save: bool = True,
    session_id: Optional[str] = None,
    skip_existing: bool = True,
) -> Dict[Tuple[str, str], GroverBatchResult]:
    """
    Run a full experiment campaign across configurations and noise models.
    
    Convenience function that uses the GroverExperimentRunner.
    """
    # Create runner with correct output_dir
    runner = GroverExperimentRunner(
        shots=shots,
        num_runs=num_runs,
        output_dir=output_dir,
    )
    
    return runner.run_campaign(
        config_ids=config_ids,
        noise_ids=noise_ids,
        num_runs=num_runs,
        shots=shots,
        save_results=save_results,
        verbose=verbose,
        incremental_save=incremental_save,
        session_id=session_id,
        skip_existing=skip_existing,
    )


def aggregate_session_results(
    session_id: str,
    output_dir: str = "data/simulator",
    verbose: bool = True,
) -> Dict[Tuple[str, str], GroverBatchResult]:
    """
    Aggregate all results from a session directory.
    """
    runner = GroverExperimentRunner(output_dir=output_dir)
    return runner.aggregate_session(session_id, verbose=verbose)


# =============================================================================
# Quick Test Functions
# =============================================================================


def test_single_config(
    config_id: str = "S3-1",
    noise_id: str = "IDEAL",
    num_runs: int = 3,
) -> GroverBatchResult:
    """Quick test with a single configuration."""
    return run_batch(config_id, noise_id, num_runs=num_runs, verbose=True)


def test_pilot_study(verbose: bool = True) -> Dict[Tuple[str, str], GroverBatchResult]:
    """
    Run a pilot study with 3-qubit configs and two noise models.

    Good for validating the infrastructure before full campaign.
    """
    pilot_configs = ["S3-1", "M3-1", "H3-0", "H3-3"]
    pilot_noise = ["IDEAL", "DEP-MED"]

    return run_experiment_campaign(
        config_ids=pilot_configs,
        noise_ids=pilot_noise,
        num_runs=5,
        save_results=False,
        verbose=verbose,
    )


def test_grover_base_case(verbose: bool = True) -> GroverExperimentResult:
    """
    Run a single base case to understand Grover's algorithm behavior.

    Base Case:
    - 3 qubits (search space = 8)
    - 1 marked state: |011⟩
    - Optimal iterations: 2
    - Theoretical success: ~94.5%
    """
    if verbose:
        print("\n" + "=" * 60)
        print("GROVER BASE CASE")
        print("=" * 60)
        print("\nAlgorithm explanation:")
        print("  1. Initialize all qubits to |+⟩ (uniform superposition)")
        print("  2. Apply Grover operator (oracle + diffusion) 2 times")
        print("  3. Measure - should peak at |011⟩")
        print("\nExpected:")
        print("  - Search space: 8 states")
        print("  - Marked state: |011⟩")
        print("  - Theoretical success: ~94.5%")
        print("=" * 60)

    config = get_config("S3-1")
    result = run_single_experiment(
        exp_config=config,
        noise_config=get_noise_config("IDEAL"),
        run_number=1,
        shots=1024,
        calculate_qward=True,
    )

    if verbose:
        print(f"\nResults:")
        print(f"  Success rate: {result.success_rate:.4f}")
        print(f"  Theoretical: {config.theoretical_success:.4f}")
        print(f"  Successful shots: {result.success_count}/{result.shots}")
        print(f"  Circuit depth: {result.circuit_depth}")
        print(f"  Total gates: {result.total_gates}")
        print(f"  Grover iterations: {result.grover_iterations}")
        print(f"\nTop outcomes:")
        sorted_counts = sorted(result.counts.items(), key=lambda x: x[1], reverse=True)
        for outcome, count in sorted_counts[:5]:
            is_marked = outcome in config.marked_states
            marker = "✓" if is_marked else " "
            print(f"  {marker} |{outcome}⟩: {count} shots ({count/result.shots*100:.1f}%)")

    return result


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point for running experiments."""
    print("Grover Experiment Runner")
    print("=" * 60)
    print("\nAvailable functions:")
    print("  - test_grover_base_case()        # Understand Grover algorithm")
    print("  - test_single_config(config_id, noise_id, num_runs)")
    print("  - test_pilot_study()")
    print("  - run_batch(config_id, noise_id, num_runs, shots)")
    print("  - run_experiment_campaign(...)")
    print("  - aggregate_session_results(session_id)  # Resume partial campaign")
    print("\nKey features:")
    print("  - incremental_save=True  # Save after each batch")
    print("  - skip_existing=True     # Don't re-run completed experiments")
    print("  - session_id='...'       # Resume a specific session")
    print("\nExample:")
    print("  from grover_experiment import run_experiment_campaign")
    print("  results = run_experiment_campaign(")
    print("      config_ids=['S3-1', 'S4-1'],")
    print("      noise_ids=['IDEAL', 'DEP-MED'],")
    print("      session_id='my_experiment'")
    print("  )")


if __name__ == "__main__":
    main()
