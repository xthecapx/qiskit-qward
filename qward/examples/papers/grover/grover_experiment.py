"""
Grover Experiment Runner

This module provides the main experiment runner for systematic evaluation
of Grover's algorithm under various configurations and noise models.
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
from .grover_configs import (
    ExperimentConfig, NoiseConfig,
    get_config, get_noise_config, get_configs_by_type,
    ALL_EXPERIMENT_CONFIGS, NOISE_CONFIGS,
    CONFIGS_BY_ID, NOISE_BY_ID,
)
from .grover_success_metrics import (
    success_rate, success_count, evaluate_job, evaluate_batch,
)
from .grover_statistical_analysis import (
    compute_descriptive_stats, test_normality, analyze_noise_impact,
    analyze_config_results, ConfigAnalysis, print_analysis_summary,
)

# QWARD imports
from qward.algorithms import GroverCircuitGenerator
from qward import Scanner
from qward.metrics import (
    QiskitMetrics,
    ComplexityMetrics,
    StructuralMetrics,
    QuantumSpecificMetrics,
    ElementMetrics,
    CircuitPerformanceMetrics,
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
            strategies=[QiskitMetrics, ComplexityMetrics, StructuralMetrics, QuantumSpecificMetrics]
        )
        
        # Calculate all metrics
        metrics_dict = scanner.calculate_metrics()
        
        # Convert DataFrames to flat dictionaries for JSON serialization
        # Scanner returns DataFrames with one row where columns are metric names
        result = {}
        for metric_name, df in metrics_dict.items():
            if df is not None and not df.empty:
                # DataFrame has columns as metric names, single row with values
                # Convert to {column_name: value} dict
                row = df.iloc[0]
                result[metric_name] = {col: _serialize_value(val) for col, val in row.items()}
        
        return result
    except Exception as e:
        # Log warning but don't fail the experiment
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
        # Convert numpy types or other objects to Python natives
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)


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
            depol_1q, ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz"]
        )
        
        # Two-qubit depolarizing
        depol_2q = depolarizing_error(p2, 2)
        noise_model.add_all_qubit_quantum_error(depol_2q, ["cx", "cz"])
        
    elif noise_config.noise_type == "pauli":
        px = params.get("pX", 0.01)
        py = params.get("pY", 0.01)
        pz = params.get("pZ", 0.01)
        pi = 1 - px - py - pz
        
        pauli_err = pauli_error([("X", px), ("Y", py), ("Z", pz), ("I", pi)])
        noise_model.add_all_qubit_quantum_error(
            pauli_err, ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz"]
        )
        
    elif noise_config.noise_type == "readout":
        p01 = params.get("p01", 0.02)
        p10 = params.get("p10", 0.02)
        
        readout_err = ReadoutError([[1 - p01, p01], [p10, 1 - p10]])
        noise_model.add_all_qubit_readout_error(readout_err)
        
    elif noise_config.noise_type == "combined":
        # Combination of depolarizing and readout
        p1 = params.get("p1", 0.01)
        p2 = params.get("p2", 0.05)
        p_readout = params.get("p_readout", 0.02)
        
        # Depolarizing
        depol_1q = depolarizing_error(p1, 1)
        noise_model.add_all_qubit_quantum_error(
            depol_1q, ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz"]
        )
        depol_2q = depolarizing_error(p2, 2)
        noise_model.add_all_qubit_quantum_error(depol_2q, ["cx", "cz"])
        
        # Readout
        readout_err = ReadoutError([[1 - p_readout, p_readout], [p_readout, 1 - p_readout]])
        noise_model.add_all_qubit_readout_error(readout_err)
        
    elif noise_config.noise_type == "thermal":
        # Simplified thermal noise (T1/T2 would need more complex modeling)
        # For now, approximate with depolarizing
        t1 = params.get("T1", 50e-6)
        t2 = params.get("T2", 70e-6)
        # Rough approximation: error rate proportional to gate time / T1
        gate_time = 50e-9  # 50 ns typical gate time
        p_thermal = gate_time / t1
        
        depol_thermal = depolarizing_error(p_thermal, 1)
        noise_model.add_all_qubit_quantum_error(
            depol_thermal, ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz"]
        )
    
    return noise_model


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
    marked_states: List[str]
    num_marked: int
    theoretical_success: float
    grover_iterations: int
    
    # Circuit metrics (basic)
    circuit_depth: int
    total_gates: int
    
    # QWARD Pre-runtime Metrics (from Scanner)
    # Contains: QiskitMetrics, ComplexityMetrics, StructuralMetrics, QuantumSpecificMetrics
    qward_metrics: Optional[Dict[str, Any]] = None
    
    # Execution
    shots: int = SHOTS
    execution_time_ms: float = 0.0
    
    # Results
    counts: Dict[str, int] = None
    success_rate: float = 0.0
    success_count: int = 0
    
    # Success metrics
    threshold_30: bool = False
    threshold_50: bool = False
    threshold_70: bool = False
    threshold_90: bool = False
    statistical_success: bool = False
    statistical_pvalue: float = 1.0
    quantum_advantage: bool = False
    advantage_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def run_single_experiment(
    exp_config: ExperimentConfig,
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
    # Create Grover circuit
    grover_gen = GroverCircuitGenerator(
        marked_states=exp_config.marked_states,
        use_barriers=True,
    )
    circuit = grover_gen.circuit
    
    # Create noise model
    noise_model = create_noise_model(noise_config)
    
    # Create simulator
    simulator = AerSimulator(noise_model=noise_model)
    
    # Transpile circuit for the simulator (required for complex gates)
    pm = generate_preset_pass_manager(
        target=simulator.target,
        optimization_level=OPTIMIZATION_LEVEL,
    )
    transpiled_circuit = pm.run(circuit)
    
    # Calculate QWARD pre-runtime metrics on transpiled circuit
    # These can be used to find correlations with success/time/cost
    qward_metrics = None
    if calculate_qward:
        qward_metrics = calculate_qward_metrics(transpiled_circuit)
    
    # Run experiment
    start_time = time.time()
    job = simulator.run(transpiled_circuit, shots=shots)
    result = job.result()
    execution_time_ms = (time.time() - start_time) * 1000
    
    counts = result.get_counts()
    
    # Calculate success metrics
    rate = success_rate(counts, exp_config.marked_states)
    s_count = success_count(counts, exp_config.marked_states)
    
    # Evaluate with all success metrics
    evaluation = evaluate_job(
        counts=counts,
        marked_states=exp_config.marked_states,
        num_qubits=exp_config.num_qubits,
        theoretical_prob=exp_config.theoretical_success,
        config_id=exp_config.config_id,
    )
    
    # Create experiment ID
    experiment_id = f"{exp_config.config_id}_{noise_config.noise_id}_{run_number:03d}"
    
    return ExperimentResult(
        experiment_id=experiment_id,
        config_id=exp_config.config_id,
        noise_model=noise_config.noise_id,
        run_number=run_number,
        timestamp=datetime.now().isoformat(),
        num_qubits=exp_config.num_qubits,
        marked_states=exp_config.marked_states,
        num_marked=exp_config.num_marked,
        theoretical_success=exp_config.theoretical_success,
        grover_iterations=exp_config.theoretical_iterations,
        circuit_depth=transpiled_circuit.depth(),
        total_gates=sum(transpiled_circuit.count_ops().values()),
        qward_metrics=qward_metrics,
        shots=shots,
        execution_time_ms=execution_time_ms,
        counts=counts,
        success_rate=rate,
        success_count=s_count,
        threshold_30=evaluation.threshold_30,
        threshold_50=evaluation.threshold_50,
        threshold_70=evaluation.threshold_70,
        threshold_90=evaluation.threshold_90,
        statistical_success=evaluation.statistical_success,
        statistical_pvalue=evaluation.statistical_pvalue,
        quantum_advantage=evaluation.quantum_advantage_success,
        advantage_ratio=evaluation.advantage_ratio,
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
    
    # Statistical analysis
    analysis: Optional[ConfigAnalysis]
    
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
            "analysis": self.analysis.to_dict() if self.analysis else None,
        }


def run_batch(
    config_id: str,
    noise_id: str,
    num_runs: int = NUM_RUNS,
    shots: int = SHOTS,
    ideal_rates: Optional[List[float]] = None,
    verbose: bool = True,
) -> BatchResult:
    """
    Run multiple experiments with the same configuration.
    
    Args:
        config_id: Configuration ID
        noise_id: Noise model ID
        num_runs: Number of runs
        shots: Shots per run
        ideal_rates: Optional ideal rates for comparison
        verbose: Print progress
        
    Returns:
        BatchResult
    """
    exp_config = get_config(config_id)
    noise_config = get_noise_config(noise_id)
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Running {config_id} with {noise_id} ({num_runs} runs)")
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
    
    # Statistical analysis
    analysis = analyze_config_results(
        success_rates=rates,
        config_id=config_id,
        noise_model=noise_id,
        ideal_rates=ideal_rates,
    )
    
    if verbose:
        print_analysis_summary(analysis)
    
    return BatchResult(
        config_id=config_id,
        noise_model=noise_id,
        num_runs=num_runs,
        shots_per_run=shots,
        mean_success_rate=np.mean(rates),
        std_success_rate=np.std(rates, ddof=1) if len(rates) > 1 else 0,
        min_success_rate=np.min(rates),
        max_success_rate=np.max(rates),
        median_success_rate=np.median(rates),
        analysis=analysis,
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
    output_dir: str = "data/simulator",
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
        print(f"GROVER EXPERIMENT CAMPAIGN")
        print(f"{'#' * 70}")
        print(f"Configurations: {len(config_ids)}")
        print(f"Noise models: {len(noise_ids)}")
        print(f"Runs per batch: {num_runs}")
        print(f"Shots per run: {shots}")
        print(f"Total batches: {total_batches}")
        print(f"{'#' * 70}")
    
    all_results = {}
    ideal_rates_cache = {}  # Cache ideal rates for comparison
    
    batch_num = 0
    for config_id in config_ids:
        for noise_id in noise_ids:
            batch_num += 1
            
            if verbose:
                print(f"\n[{batch_num}/{total_batches}] ", end="")
            
            # Get ideal rates for comparison (if available)
            ideal_rates = ideal_rates_cache.get(config_id)
            
            # Run batch
            batch_result = run_batch(
                config_id=config_id,
                noise_id=noise_id,
                num_runs=num_runs,
                shots=shots,
                ideal_rates=ideal_rates,
                verbose=verbose,
            )
            
            # Cache ideal rates
            if noise_id == "IDEAL":
                ideal_rates_cache[config_id] = [r.success_rate for r in batch_result.results]
            
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
    """
    Save campaign results to disk.
    
    Args:
        results: Dictionary of batch results
        output_dir: Output directory
        verbose: Print progress
    """
    base_path = Path(__file__).parent / output_dir
    raw_path = base_path / "raw"
    agg_path = base_path / "aggregated"
    
    # Create directories
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
        summary.append({
            "config_id": config_id,
            "noise_model": noise_id,
            **batch.to_dict(),
        })
    
    summary_file = agg_path / f"campaign_summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    if verbose:
        print(f"\nResults saved to {base_path}")
        print(f"  Raw: {raw_path}")
        print(f"  Aggregated: {agg_path}")


def load_campaign_results(filepath: str) -> List[Dict[str, Any]]:
    """Load campaign results from disk."""
    with open(filepath, "r") as f:
        return json.load(f)


# =============================================================================
# Quick Test Functions
# =============================================================================

def test_single_config(
    config_id: str = "S3-1",
    noise_id: str = "IDEAL",
    num_runs: int = 3,
) -> BatchResult:
    """Quick test with a single configuration."""
    return run_batch(config_id, noise_id, num_runs=num_runs, verbose=True)


def test_pilot_study(verbose: bool = True) -> Dict[str, BatchResult]:
    """
    Run a pilot study with 3-qubit configs and two noise models.
    
    Good for validating the infrastructure before full campaign.
    """
    # 3-qubit configurations only
    pilot_configs = ["S3-1", "M3-1", "H3-0", "H3-3"]
    pilot_noise = ["IDEAL", "DEP-MED"]
    
    return run_experiment_campaign(
        config_ids=pilot_configs,
        noise_ids=pilot_noise,
        num_runs=5,  # Fewer runs for quick test
        save_results=False,
        verbose=verbose,
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for running experiments."""
    print("Grover Experiment Runner")
    print("=" * 60)
    print("\nAvailable functions:")
    print("  - test_single_config(config_id, noise_id, num_runs)")
    print("  - test_pilot_study()")
    print("  - run_batch(config_id, noise_id, num_runs, shots)")
    print("  - run_experiment_campaign(config_ids, noise_ids, ...)")
    print("\nExample:")
    print("  from grover_experiment import test_pilot_study")
    print("  results = test_pilot_study()")


if __name__ == "__main__":
    main()

