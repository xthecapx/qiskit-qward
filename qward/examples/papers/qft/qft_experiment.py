"""
QFT Experiment Runner

This module provides the experiment runner for systematic evaluation
of Quantum Fourier Transform under various configurations and noise models.

Uses the BaseExperimentRunner framework for consistent workflow with
incremental saving, resume support, and QWARD metrics integration.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from qiskit import QuantumCircuit

# Local imports
from .qft_configs import (
    QFTExperimentConfig,
    NoiseConfig,
    get_config,
    get_noise_config,
    get_configs_by_type,
    ALL_EXPERIMENT_CONFIGS,
    NOISE_CONFIGS,
    CONFIGS_BY_ID,
    NOISE_BY_ID,
)

# QWARD imports
from qward.algorithms import QFTCircuitGenerator
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
# Success Rate Calculation
# =============================================================================


def calculate_success_rate(
    counts: Dict[str, int],
    success_criteria_func,
) -> float:
    """
    Calculate success rate using the circuit generator's success criteria.

    Args:
        counts: Measurement counts from experiment
        success_criteria_func: Function that takes outcome string and returns bool

    Returns:
        Success rate as float in [0, 1]
    """
    total_shots = sum(counts.values())
    successful_shots = sum(
        count for outcome, count in counts.items() if success_criteria_func(outcome)
    )
    return successful_shots / total_shots if total_shots > 0 else 0.0


def count_successful_shots(
    counts: Dict[str, int],
    success_criteria_func,
) -> int:
    """Count number of successful shots."""
    return sum(
        count for outcome, count in counts.items() if success_criteria_func(outcome)
    )


# =============================================================================
# QFT-Specific Result Classes
# =============================================================================


@dataclass
class QFTExperimentResult(BaseExperimentResult):
    """Result from a single QFT experiment run."""

    # QFT-specific properties
    test_mode: str = ""
    input_state: Optional[str] = None
    period: Optional[int] = None

    # QFT-specific thresholds
    threshold_90: bool = False
    threshold_95: bool = False
    threshold_99: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QFTExperimentResult":
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
            test_mode=data.get("test_mode", ""),
            input_state=data.get("input_state"),
            period=data.get("period"),
            threshold_90=data.get("threshold_90", False),
            threshold_95=data.get("threshold_95", False),
            threshold_99=data.get("threshold_99", False),
        )


@dataclass
class QFTBatchResult(BaseBatchResult[QFTExperimentResult, None]):
    """Result from running multiple QFT experiment runs."""

    pass  # Uses base implementation


# =============================================================================
# QFT Experiment Runner
# =============================================================================


class QFTExperimentRunner(BaseExperimentRunner[
    QFTExperimentConfig, QFTExperimentResult, QFTBatchResult, None
]):
    """
    Experiment runner for QFT algorithm.
    
    Provides systematic evaluation of QFT under various configurations
    and noise models with incremental saving and resume support.
    """

    def __init__(
        self,
        shots: int = SHOTS,
        num_runs: int = NUM_RUNS,
        optimization_level: int = OPTIMIZATION_LEVEL,
        output_dir: str = "data",
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
        return "QFT"

    def create_circuit(
        self, config: QFTExperimentConfig
    ) -> Tuple[QuantumCircuit, QFTCircuitGenerator]:
        """Create QFT circuit based on configuration."""
        if config.test_mode == "roundtrip":
            qft_gen = QFTCircuitGenerator(
                num_qubits=config.num_qubits,
                test_mode="roundtrip",
                input_state=config.input_state,
                use_barriers=True,
            )
        else:  # period_detection
            qft_gen = QFTCircuitGenerator(
                num_qubits=config.num_qubits,
                test_mode="period_detection",
                period=config.period,
                use_barriers=True,
            )
        return qft_gen.circuit, qft_gen

    def calculate_success(
        self,
        counts: Dict[str, int],
        config: QFTExperimentConfig,
        circuit_metadata: QFTCircuitGenerator,
    ) -> Tuple[float, int]:
        """Calculate QFT success metrics."""
        rate = calculate_success_rate(counts, circuit_metadata.success_criteria)
        s_count = count_successful_shots(counts, circuit_metadata.success_criteria)
        return rate, s_count

    def create_result(
        self,
        config: QFTExperimentConfig,
        noise_config: NoiseConfig,
        run_number: int,
        transpiled_circuit: QuantumCircuit,
        counts: Dict[str, int],
        execution_time_ms: float,
        success_rate: float,
        success_count: int,
        qward_metrics: Optional[Dict[str, Any]],
        circuit_metadata: QFTCircuitGenerator,
        backend_type: str,
        backend_name: str,
    ) -> QFTExperimentResult:
        """Create QFT experiment result."""
        experiment_id = f"{config.config_id}_{noise_config.noise_id}_{run_number:03d}"

        return QFTExperimentResult(
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
            test_mode=config.test_mode,
            input_state=config.input_state,
            period=config.period,
            threshold_90=success_rate >= 0.90,
            threshold_95=success_rate >= 0.95,
            threshold_99=success_rate >= 0.99,
        )

    def get_config(self, config_id: str) -> QFTExperimentConfig:
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

    def get_config_description(self, config: QFTExperimentConfig) -> str:
        """Get description for config in verbose output."""
        return f"Mode: {config.test_mode}, Qubits: {config.num_qubits}"

    def load_result_from_dict(self, data: Dict[str, Any]) -> QFTExperimentResult:
        """Reconstruct QFT result from dictionary."""
        return QFTExperimentResult.from_dict(data)


# =============================================================================
# Module-Level Convenience Functions
# =============================================================================

# Default runner instance
_default_runner: Optional[QFTExperimentRunner] = None


def _get_runner() -> QFTExperimentRunner:
    """Get or create the default runner instance."""
    global _default_runner
    if _default_runner is None:
        _default_runner = QFTExperimentRunner()
    return _default_runner


def run_single_experiment(
    exp_config: QFTExperimentConfig,
    noise_config: NoiseConfig,
    run_number: int,
    shots: int = SHOTS,
    calculate_qward: bool = True,
) -> QFTExperimentResult:
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
    verbose: bool = True,
) -> QFTBatchResult:
    """
    Run multiple experiments with the same configuration.
    
    Convenience function for backward compatibility.
    """
    runner = _get_runner()
    runner.shots = shots
    runner.num_runs = num_runs
    return runner.run_batch(config_id, noise_id, num_runs, shots, verbose=verbose)


def run_experiment_campaign(
    config_ids: Optional[List[str]] = None,
    noise_ids: Optional[List[str]] = None,
    num_runs: int = NUM_RUNS,
    shots: int = SHOTS,
    save_results: bool = True,
    output_dir: str = "data",
    verbose: bool = True,
    incremental_save: bool = True,
    session_id: Optional[str] = None,
    skip_existing: bool = True,
) -> Dict[Tuple[str, str], QFTBatchResult]:
    """
    Run a full experiment campaign across configurations and noise models.
    
    Convenience function that uses the QFTExperimentRunner.
    """
    # Create runner with correct output_dir
    runner = QFTExperimentRunner(
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
    output_dir: str = "data",
    verbose: bool = True,
) -> Dict[Tuple[str, str], QFTBatchResult]:
    """
    Aggregate all results from a session directory.
    """
    runner = QFTExperimentRunner(output_dir=output_dir)
    return runner.aggregate_session(session_id, verbose=verbose)


# =============================================================================
# Quick Test Functions
# =============================================================================


def test_single_config(
    config_id: str = "SR3",
    noise_id: str = "IDEAL",
    num_runs: int = 3,
) -> QFTBatchResult:
    """Quick test with a single configuration."""
    return run_batch(config_id, noise_id, num_runs=num_runs, verbose=True)


def test_pilot_study(verbose: bool = True) -> Dict[Tuple[str, str], QFTBatchResult]:
    """
    Run a pilot study with small configs and two noise models.

    Good for validating the infrastructure before full campaign.
    """
    pilot_configs = ["SR3", "SR4", "SP4-P4"]
    pilot_noise = ["IDEAL", "DEP-MED"]

    return run_experiment_campaign(
        config_ids=pilot_configs,
        noise_ids=pilot_noise,
        num_runs=5,
        save_results=False,
        verbose=verbose,
    )


def test_roundtrip_base_case(verbose: bool = True) -> QFTExperimentResult:
    """
    Run a single base case for roundtrip mode to understand the algorithm.

    Base Case:
    - 3 qubits
    - Input state: |101⟩
    - Expected: After QFT→QFT⁻¹, should return |101⟩
    - Success: measurement == "101"
    """
    if verbose:
        print("\n" + "=" * 60)
        print("QFT ROUND-TRIP BASE CASE")
        print("=" * 60)
        print("\nAlgorithm explanation:")
        print("  1. Prepare input state |101⟩ (using X gates)")
        print("  2. Apply QFT (Hadamards + controlled phase rotations)")
        print("  3. Apply inverse QFT")
        print("  4. Measure - should return to |101⟩")
        print("\nSuccess criteria: measurement == '101'")
        print("=" * 60)

    result = run_single_experiment(
        exp_config=get_config("SR3"),
        noise_config=get_noise_config("IDEAL"),
        run_number=1,
        shots=1024,
        calculate_qward=True,
    )

    if verbose:
        print(f"\nResults:")
        print(f"  Success rate: {result.success_rate:.4f}")
        print(f"  Successful shots: {result.success_count}/{result.shots}")
        print(f"  Circuit depth: {result.circuit_depth}")
        print(f"  Total gates: {result.total_gates}")
        print(f"\nTop 5 outcomes:")
        sorted_counts = sorted(result.counts.items(), key=lambda x: x[1], reverse=True)
        for outcome, count in sorted_counts[:5]:
            is_success = outcome == "101"
            marker = "✓" if is_success else " "
            print(f"  {marker} |{outcome}⟩: {count} shots ({count/result.shots*100:.1f}%)")

    return result


def test_period_detection_base_case(verbose: bool = True) -> QFTExperimentResult:
    """
    Run a single base case for period detection mode to understand the algorithm.

    Base Case:
    - 4 qubits (N = 16)
    - Period = 4
    - Expected peaks: 16/4 = 4, so peaks at 0, 4, 8, 12
    - In binary: 0000, 0100, 1000, 1100
    - Success: measurement is within ±1 of a peak
    """
    if verbose:
        print("\n" + "=" * 60)
        print("QFT PERIOD DETECTION BASE CASE")
        print("=" * 60)
        print("\nAlgorithm explanation:")
        print("  1. Prepare ancilla in |1⟩")
        print("  2. Put counting qubits (4) in superposition")
        print("  3. Apply controlled phase rotations to encode period=4")
        print("  4. Apply inverse QFT to counting register")
        print("  5. Measure - should peak at multiples of 16/4=4")
        print("\nExpected peaks: 0, 4, 8, 12 (binary: 0000, 0100, 1000, 1100)")
        print("Success criteria: measurement at or near these peaks")
        print("=" * 60)

    result = run_single_experiment(
        exp_config=get_config("SP4-P4"),
        noise_config=get_noise_config("IDEAL"),
        run_number=1,
        shots=1024,
        calculate_qward=True,
    )

    if verbose:
        print(f"\nResults:")
        print(f"  Success rate: {result.success_rate:.4f}")
        print(f"  Successful shots: {result.success_count}/{result.shots}")
        print(f"  Circuit depth: {result.circuit_depth}")
        print(f"  Total gates: {result.total_gates}")
        print(f"\nAll outcomes (should peak at 0, 4, 8, 12):")
        sorted_counts = sorted(result.counts.items(), key=lambda x: x[1], reverse=True)
        for outcome, count in sorted_counts:
            decimal = int(outcome, 2)
            is_peak = decimal in [0, 4, 8, 12]
            marker = "✓" if is_peak else " "
            print(f"  {marker} |{outcome}⟩ (={decimal:2d}): {count} shots ({count/result.shots*100:.1f}%)")

    return result


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point for running experiments."""
    print("QFT Experiment Runner")
    print("=" * 60)
    print("\nAvailable functions:")
    print("  - test_roundtrip_base_case()     # Understand roundtrip mode")
    print("  - test_period_detection_base_case()  # Understand period detection")
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
    print("  from qft_experiment import run_experiment_campaign")
    print("  results = run_experiment_campaign(")
    print("      config_ids=['SR3', 'SR4'],")
    print("      noise_ids=['IDEAL', 'DEP-MED'],")
    print("      session_id='my_experiment'")
    print("  )")


if __name__ == "__main__":
    main()
