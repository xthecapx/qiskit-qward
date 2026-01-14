"""
QFT Experiment Runner

This module provides the main experiment runner for systematic evaluation
of Quantum Fourier Transform under various configurations and noise models.

Integrates with QWARD for pre-runtime metrics analysis.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import numpy as np

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, pauli_error
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

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
from qward import Scanner
from qward.metrics import (
    QiskitMetrics,
    ComplexityMetrics,
    StructuralMetrics,
    QuantumSpecificMetrics,
)


# =============================================================================
# Experiment Parameters
# =============================================================================

SHOTS = 1024
NUM_RUNS = 10
OPTIMIZATION_LEVEL = 0


# =============================================================================
# QWARD Metrics Calculator
# =============================================================================


def calculate_qward_metrics(circuit) -> Dict[str, Any]:
    """
    Calculate pre-runtime QWARD metrics for a circuit using Scanner.

    These metrics can be used to analyze correlations with:
    - Success rate
    - Execution time
    - QPU price (in real hardware)

    Args:
        circuit: The quantum circuit to analyze

    Returns:
        Dictionary with all QWARD metrics (converted from DataFrames for JSON serialization)
        Returns empty dict if metrics calculation fails.
    """
    try:
        # Create scanner with pre-runtime metric strategies
        scanner = Scanner(
            circuit=circuit,
            strategies=[QiskitMetrics, ComplexityMetrics, StructuralMetrics, QuantumSpecificMetrics],
        )

        # Calculate all metrics
        metrics_dict = scanner.calculate_metrics()

        # Convert DataFrames to flat dictionaries for JSON serialization
        result = {}
        for metric_name, df in metrics_dict.items():
            if df is not None and not df.empty:
                row = df.iloc[0]
                result[metric_name] = {col: _serialize_value(val) for col, val in row.items()}

        return result
    except Exception as e:
        print(f"    Warning: QWARD metrics failed: {e}")
        return {"error": str(e)}


def _serialize_value(value):
    """Convert a value to JSON-serializable format."""
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    else:
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)


# =============================================================================
# Noise Model Factory
# =============================================================================


def create_noise_model(noise_config: NoiseConfig) -> Optional[NoiseModel]:
    """
    Create a Qiskit noise model from configuration.

    Args:
        noise_config: Noise configuration

    Returns:
        NoiseModel or None for ideal simulation
    """
    if noise_config.noise_type == "none":
        return None

    noise_model = NoiseModel()
    params = noise_config.parameters

    if noise_config.noise_type == "depolarizing":
        p1 = params.get("p1", 0.01)
        p2 = params.get("p2", 0.05)

        # Single-qubit depolarizing
        depol_1q = depolarizing_error(p1, 1)
        noise_model.add_all_qubit_quantum_error(
            depol_1q, ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "p"]
        )

        # Two-qubit depolarizing (important for controlled phase gates in QFT)
        depol_2q = depolarizing_error(p2, 2)
        noise_model.add_all_qubit_quantum_error(depol_2q, ["cx", "cz", "cp", "swap"])

    elif noise_config.noise_type == "readout":
        p01 = params.get("p01", 0.02)
        p10 = params.get("p10", 0.02)

        readout_err = ReadoutError([[1 - p01, p01], [p10, 1 - p10]])
        noise_model.add_all_qubit_readout_error(readout_err)

    elif noise_config.noise_type == "combined":
        p1 = params.get("p1", 0.01)
        p2 = params.get("p2", 0.05)
        p_readout = params.get("p_readout", 0.02)

        # Depolarizing
        depol_1q = depolarizing_error(p1, 1)
        noise_model.add_all_qubit_quantum_error(
            depol_1q, ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "p"]
        )
        depol_2q = depolarizing_error(p2, 2)
        noise_model.add_all_qubit_quantum_error(depol_2q, ["cx", "cz", "cp", "swap"])

        # Readout
        readout_err = ReadoutError(
            [[1 - p_readout, p_readout], [p_readout, 1 - p_readout]]
        )
        noise_model.add_all_qubit_readout_error(readout_err)

    return noise_model


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
# Single Experiment Run
# =============================================================================


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""

    # Identification
    experiment_id: str
    config_id: str
    noise_model: str
    run_number: int
    timestamp: str

    # Circuit properties
    num_qubits: int
    test_mode: str
    input_state: Optional[str]
    period: Optional[int]

    # Circuit metrics (basic)
    circuit_depth: int
    total_gates: int

    # QWARD Pre-runtime Metrics
    qward_metrics: Optional[Dict[str, Any]] = None

    # Execution
    shots: int = SHOTS
    execution_time_ms: float = 0.0

    # Results
    counts: Dict[str, int] = None
    success_rate: float = 0.0
    success_count: int = 0

    # Thresholds
    threshold_90: bool = False
    threshold_95: bool = False
    threshold_99: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def run_single_experiment(
    exp_config: QFTExperimentConfig,
    noise_config: NoiseConfig,
    run_number: int,
    shots: int = SHOTS,
    calculate_qward: bool = True,
) -> ExperimentResult:
    """
    Run a single experiment with given configuration.

    Args:
        exp_config: Experiment configuration
        noise_config: Noise model configuration
        run_number: Run number (1-indexed)
        shots: Number of shots
        calculate_qward: Whether to calculate QWARD pre-runtime metrics

    Returns:
        ExperimentResult with QWARD metrics for correlation analysis
    """
    # Create QFT circuit generator based on test mode
    if exp_config.test_mode == "roundtrip":
        qft_gen = QFTCircuitGenerator(
            num_qubits=exp_config.num_qubits,
            test_mode="roundtrip",
            input_state=exp_config.input_state,
            use_barriers=True,
        )
    else:  # period_detection
        qft_gen = QFTCircuitGenerator(
            num_qubits=exp_config.num_qubits,
            test_mode="period_detection",
            period=exp_config.period,
            use_barriers=True,
        )

    circuit = qft_gen.circuit

    # Create noise model
    noise_model = create_noise_model(noise_config)

    # Create simulator
    simulator = AerSimulator(noise_model=noise_model)

    # Transpile circuit
    pm = generate_preset_pass_manager(
        target=simulator.target,
        optimization_level=OPTIMIZATION_LEVEL,
    )
    transpiled_circuit = pm.run(circuit)

    # Calculate QWARD pre-runtime metrics on transpiled circuit
    qward_metrics = None
    if calculate_qward:
        qward_metrics = calculate_qward_metrics(transpiled_circuit)

    # Run experiment
    start_time = time.time()
    job = simulator.run(transpiled_circuit, shots=shots)
    result = job.result()
    execution_time_ms = (time.time() - start_time) * 1000

    counts = result.get_counts()

    # Calculate success metrics using generator's success criteria
    rate = calculate_success_rate(counts, qft_gen.success_criteria)
    s_count = count_successful_shots(counts, qft_gen.success_criteria)

    # Create experiment ID
    experiment_id = f"{exp_config.config_id}_{noise_config.noise_id}_{run_number:03d}"

    return ExperimentResult(
        experiment_id=experiment_id,
        config_id=exp_config.config_id,
        noise_model=noise_config.noise_id,
        run_number=run_number,
        timestamp=datetime.now().isoformat(),
        num_qubits=exp_config.num_qubits,
        test_mode=exp_config.test_mode,
        input_state=exp_config.input_state,
        period=exp_config.period,
        circuit_depth=transpiled_circuit.depth(),
        total_gates=sum(transpiled_circuit.count_ops().values()),
        qward_metrics=qward_metrics,
        shots=shots,
        execution_time_ms=execution_time_ms,
        counts=counts,
        success_rate=rate,
        success_count=s_count,
        threshold_90=rate >= 0.90,
        threshold_95=rate >= 0.95,
        threshold_99=rate >= 0.99,
    )


# =============================================================================
# Batch Experiment Runner
# =============================================================================


@dataclass
class BatchResult:
    """Result from running multiple runs of the same configuration."""

    config_id: str
    noise_model: str
    num_runs: int
    shots_per_run: int

    # Aggregate statistics
    mean_success_rate: float
    std_success_rate: float
    min_success_rate: float
    max_success_rate: float
    median_success_rate: float

    # Individual results
    results: List[ExperimentResult]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without individual results for summary)."""
        return {
            "config_id": self.config_id,
            "noise_model": self.noise_model,
            "num_runs": self.num_runs,
            "shots_per_run": self.shots_per_run,
            "mean_success_rate": self.mean_success_rate,
            "std_success_rate": self.std_success_rate,
            "min_success_rate": self.min_success_rate,
            "max_success_rate": self.max_success_rate,
            "median_success_rate": self.median_success_rate,
        }


def run_batch(
    config_id: str,
    noise_id: str,
    num_runs: int = NUM_RUNS,
    shots: int = SHOTS,
    verbose: bool = True,
) -> BatchResult:
    """
    Run multiple experiments with the same configuration.

    Args:
        config_id: Configuration ID
        noise_id: Noise model ID
        num_runs: Number of runs
        shots: Shots per run
        verbose: Print progress

    Returns:
        BatchResult
    """
    exp_config = get_config(config_id)
    noise_config = get_noise_config(noise_id)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Running {config_id} with {noise_id} ({num_runs} runs)")
        print(f"  Mode: {exp_config.test_mode}, Qubits: {exp_config.num_qubits}")
        print(f"{'=' * 60}")

    results = []
    for run in range(1, num_runs + 1):
        if verbose:
            print(f"  Run {run}/{num_runs}...", end=" ", flush=True)

        result = run_single_experiment(exp_config, noise_config, run, shots)
        results.append(result)

        if verbose:
            print(f"Success rate: {result.success_rate:.4f}")

    # Calculate aggregate statistics
    rates = [r.success_rate for r in results]

    if verbose:
        print(f"\n  Summary: mean={np.mean(rates):.4f}, std={np.std(rates):.4f}")

    return BatchResult(
        config_id=config_id,
        noise_model=noise_id,
        num_runs=num_runs,
        shots_per_run=shots,
        mean_success_rate=float(np.mean(rates)),
        std_success_rate=float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
        min_success_rate=float(np.min(rates)),
        max_success_rate=float(np.max(rates)),
        median_success_rate=float(np.median(rates)),
        results=results,
    )


# =============================================================================
# Full Experiment Campaign
# =============================================================================


def run_experiment_campaign(
    config_ids: Optional[List[str]] = None,
    noise_ids: Optional[List[str]] = None,
    num_runs: int = NUM_RUNS,
    shots: int = SHOTS,
    save_results: bool = True,
    output_dir: str = "data",
    verbose: bool = True,
) -> Dict[str, BatchResult]:
    """
    Run a full experiment campaign across configurations and noise models.

    Args:
        config_ids: List of configuration IDs (None = all)
        noise_ids: List of noise model IDs (None = all)
        num_runs: Number of runs per configuration
        shots: Shots per run
        save_results: Whether to save results to disk
        output_dir: Output directory for results
        verbose: Print progress

    Returns:
        Dictionary mapping (config_id, noise_id) to BatchResult
    """
    # Default to all configurations
    if config_ids is None:
        config_ids = list(CONFIGS_BY_ID.keys())
    if noise_ids is None:
        noise_ids = list(NOISE_BY_ID.keys())

    # Ensure IDEAL is first for baseline comparison
    if "IDEAL" in noise_ids:
        noise_ids = ["IDEAL"] + [n for n in noise_ids if n != "IDEAL"]

    total_batches = len(config_ids) * len(noise_ids)

    if verbose:
        print(f"\n{'#' * 70}")
        print("QFT EXPERIMENT CAMPAIGN")
        print(f"{'#' * 70}")
        print(f"Configurations: {len(config_ids)}")
        print(f"Noise models: {len(noise_ids)}")
        print(f"Runs per batch: {num_runs}")
        print(f"Shots per run: {shots}")
        print(f"Total batches: {total_batches}")
        print(f"{'#' * 70}")

    all_results = {}

    batch_num = 0
    for config_id in config_ids:
        for noise_id in noise_ids:
            batch_num += 1

            if verbose:
                print(f"\n[{batch_num}/{total_batches}] ", end="")

            batch_result = run_batch(
                config_id=config_id,
                noise_id=noise_id,
                num_runs=num_runs,
                shots=shots,
                verbose=verbose,
            )

            all_results[(config_id, noise_id)] = batch_result

    # Save results if requested
    if save_results:
        save_campaign_results(all_results, output_dir, verbose)

    if verbose:
        print(f"\n{'#' * 70}")
        print("CAMPAIGN COMPLETE")
        print(f"{'#' * 70}")

    return all_results


# =============================================================================
# Data Persistence
# =============================================================================


def save_campaign_results(
    results: Dict[tuple, BatchResult],
    output_dir: str,
    verbose: bool = True,
) -> None:
    """Save campaign results to disk."""
    base_path = Path(__file__).parent / output_dir
    raw_path = base_path / "raw"
    agg_path = base_path / "aggregated"

    raw_path.mkdir(parents=True, exist_ok=True)
    agg_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save individual results (raw)
    for (config_id, noise_id), batch in results.items():
        filename = f"{config_id}_{noise_id}_{timestamp}.json"
        filepath = raw_path / filename

        data = {
            "batch_summary": batch.to_dict(),
            "individual_results": [r.to_dict() for r in batch.results],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # Save aggregated summary
    summary = []
    for (config_id, noise_id), batch in results.items():
        summary.append(
            {
                "config_id": config_id,
                "noise_model": noise_id,
                **batch.to_dict(),
            }
        )

    summary_file = agg_path / f"campaign_summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if verbose:
        print(f"\nResults saved to {base_path}")


# =============================================================================
# Quick Test Functions
# =============================================================================


def test_single_config(
    config_id: str = "SR3",
    noise_id: str = "IDEAL",
    num_runs: int = 3,
) -> BatchResult:
    """Quick test with a single configuration."""
    return run_batch(config_id, noise_id, num_runs=num_runs, verbose=True)


def test_pilot_study(verbose: bool = True) -> Dict[str, BatchResult]:
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


def test_roundtrip_base_case(verbose: bool = True) -> ExperimentResult:
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


def test_period_detection_base_case(verbose: bool = True) -> ExperimentResult:
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
    print("\nExample:")
    print("  from qft_experiment import test_roundtrip_base_case")
    print("  result = test_roundtrip_base_case()")


if __name__ == "__main__":
    main()
