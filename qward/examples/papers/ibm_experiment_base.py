#!/usr/bin/env python3
"""
IBM QPU Experiment Base Module

This module provides reusable infrastructure for running quantum algorithms
on IBM Quantum hardware with rich data output similar to simulator experiments.

QWARD metrics are calculated on the original circuit BEFORE transpilation/execution,
as the transpiled circuit will be different from what we designed.

Subclass `IBMExperimentBase` to create algorithm-specific implementations
(e.g., Grover, QFT, Phase Estimation).
"""

import argparse
import json
import statistics
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, TypeVar, Generic

from qiskit import QuantumCircuit

from qward.algorithms import QuantumCircuitExecutor, IBMBatchResult
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics


# Type variable for config classes
ConfigT = TypeVar("ConfigT")


def calculate_qward_metrics(circuit: QuantumCircuit) -> Dict[str, Any]:
    """
    Calculate QWARD metrics for a circuit BEFORE execution.

    This should be called on the original circuit, not the transpiled one,
    as we want to measure the circuit we designed, not the hardware-specific
    version.

    Args:
        circuit: The original quantum circuit to analyze

    Returns:
        Dictionary with QWARD metrics, or error dict if calculation fails
    """
    try:
        scanner = Scanner(circuit=circuit)
        scanner.add_strategy(QiskitMetrics(circuit))
        scanner.add_strategy(ComplexityMetrics(circuit))

        metrics_dict = scanner.calculate_metrics()

        # Convert DataFrames to dictionaries
        result = {}
        for metric_name, df in metrics_dict.items():
            if df is not None and hasattr(df, "empty") and not df.empty:
                row = df.iloc[0]
                result[metric_name] = {}
                for col in df.columns:
                    val = row[col]
                    # Convert to JSON-serializable format
                    if isinstance(val, (int, float, str, bool, type(None))):
                        result[metric_name][col] = val
                    elif hasattr(val, "item"):  # numpy types
                        result[metric_name][col] = val.item()
                    elif isinstance(val, (list, tuple)):
                        result[metric_name][col] = [str(v) for v in val]
                    else:
                        result[metric_name][col] = str(val)

        return result

    except Exception as e:
        return {"error": str(e)}


def calculate_statistical_analysis(
    success_rates: List[float], config_id: str, noise_model: str
) -> Dict[str, Any]:
    """
    Calculate statistical analysis for batch results.

    Similar to simulator analysis including CI, skewness, kurtosis, normality tests.

    Args:
        success_rates: List of success rates from individual runs
        config_id: Configuration ID
        noise_model: Noise model identifier

    Returns:
        Dictionary with statistical analysis
    """
    n = len(success_rates)
    if n < 2:
        return {}

    mean = statistics.mean(success_rates)
    std = statistics.stdev(success_rates)

    # Confidence interval (95%)
    se = std / math.sqrt(n)
    t_value = 1.96  # Approximate for large samples
    ci_lower = mean - t_value * se
    ci_upper = mean + t_value * se

    # Skewness
    if std > 0:
        skewness = sum((x - mean) ** 3 for x in success_rates) / (n * std**3)
    else:
        skewness = 0.0

    # Kurtosis (excess kurtosis)
    if std > 0:
        kurtosis = sum((x - mean) ** 4 for x in success_rates) / (n * std**4) - 3
    else:
        kurtosis = 0.0

    # Simple normality check (Shapiro-Wilk would be better but requires scipy)
    # For now, just check if skewness and kurtosis are within normal range
    is_normal = abs(skewness) < 2 and abs(kurtosis) < 7

    return {
        "config_id": config_id,
        "noise_model": noise_model,
        "num_runs": n,
        "mean": mean,
        "std": std,
        "median": statistics.median(success_rates),
        "min": min(success_rates),
        "max": max(success_rates),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "is_normal": is_normal,
    }


@dataclass
class IBMExperimentConfig:
    """Base configuration for IBM experiments."""

    config_id: str
    num_qubits: int
    expected_success: float
    circuit_depth: int
    region: str = "Region 1"


class IBMExperimentBase(ABC, Generic[ConfigT]):
    """Base class for IBM QPU experiments.

    Provides common infrastructure for:
    - Circuit execution on IBM QPU
    - QWARD metrics calculation (on original circuit before transpilation)
    - Rich result building with statistical analysis
    - Result saving
    - Command-line interface

    Subclass and implement the abstract methods for your specific algorithm.
    """

    def __init__(
        self,
        shots: int = 1024,
        timeout: int = 600,
        output_subdir: str = "data/qpu/raw",
    ):
        """Initialize the experiment base.

        Args:
            shots: Number of shots per circuit execution
            timeout: Maximum wait time for job completion (seconds)
            output_subdir: Subdirectory for saving results (relative to algorithm folder)
        """
        self.shots = shots
        self.timeout = timeout
        self.output_subdir = output_subdir
        self.executor = QuantumCircuitExecutor(shots=shots)

    # =========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # =========================================================================

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Return the algorithm name (e.g., 'GROVER', 'QFT')."""
        pass

    @abstractmethod
    def get_config(self, config_id: str) -> ConfigT:
        """Get the configuration for the given config_id."""
        pass

    @abstractmethod
    def get_all_config_ids(self) -> List[str]:
        """Get all available configuration IDs."""
        pass

    @abstractmethod
    def create_circuit(self, config: ConfigT) -> QuantumCircuit:
        """Create the quantum circuit for the given configuration."""
        pass

    @abstractmethod
    def create_success_criteria(self, config: ConfigT) -> Callable[[str], bool]:
        """Create a success criteria function for the configuration."""
        pass

    @abstractmethod
    def get_random_chance(self, config: ConfigT) -> float:
        """Get the classical random chance for the configuration."""
        pass

    @abstractmethod
    def get_config_description(self, config: ConfigT) -> Dict[str, Any]:
        """Get a dictionary describing the configuration for saving."""
        pass

    # =========================================================================
    # Optional Methods - Can be overridden by subclasses
    # =========================================================================

    def evaluate_result(
        self,
        counts: Dict[str, int],
        config: ConfigT,
        total_shots: int,
    ) -> Dict[str, Any]:
        """Evaluate the result and return algorithm-specific metrics.

        Override this method to add algorithm-specific evaluation.
        """
        success_criteria = self.create_success_criteria(config)
        s_count = sum(c for k, c in counts.items() if success_criteria(k))
        s_rate = s_count / total_shots if total_shots > 0 else 0.0

        random_chance = self.get_random_chance(config)
        advantage_ratio = s_rate / random_chance if random_chance > 0 else 0.0

        return {
            "success_rate": s_rate,
            "success_count": s_count,
            "random_chance": random_chance,
            "advantage_ratio": advantage_ratio,
            "quantum_advantage": advantage_ratio > 2.0,
            "threshold_30": s_rate >= 0.30,
            "threshold_50": s_rate >= 0.50,
            "threshold_70": s_rate >= 0.70,
            "threshold_90": s_rate >= 0.90,
        }

    def get_priority_configs(self) -> List[Dict[str, Any]]:
        """Get prioritized configurations for QPU execution."""
        return []

    def get_output_dir(self) -> Path:
        """Get the output directory for saving results."""
        return Path(self.output_subdir)

    # =========================================================================
    # Core Execution Methods
    # =========================================================================

    def run(
        self,
        config_id: str,
        backend_name: Optional[str] = None,
        optimization_levels: Optional[List[int]] = None,
        save_results: bool = True,
        channel: Optional[str] = None,
        token: Optional[str] = None,
        instance: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the algorithm on IBM Quantum hardware.

        QWARD metrics are calculated on the original circuit BEFORE execution.
        """
        # Get configuration
        config = self.get_config(config_id)

        # Print header
        self._print_header(config)

        # Create circuit
        print("\nCreating circuit...")
        circuit = self.create_circuit(config)
        original_depth = circuit.depth()
        original_gates = circuit.size()
        print(f"Circuit depth: {original_depth}")
        print(f"Circuit gates: {original_gates}")

        # Calculate QWARD metrics on ORIGINAL circuit (before transpilation)
        print("\nCalculating QWARD metrics on original circuit...")
        qward_metrics = calculate_qward_metrics(circuit)
        if "error" in qward_metrics:
            print(f"  Warning: QWARD metrics error: {qward_metrics['error']}")
        else:
            print(f"  QWARD metrics calculated successfully")

        # Create success criteria
        success_criteria = self.create_success_criteria(config)

        # Set default optimization levels (all 4 levels: 0, 1, 2, 3)
        if optimization_levels is None:
            optimization_levels = [0, 1, 2, 3]

        # Execute on IBM QPU
        print("\nSubmitting to IBM Quantum...")
        ibm_result = self.executor.run_ibm(
            circuit,
            backend_name=backend_name,
            optimization_levels=optimization_levels,
            success_criteria=success_criteria,
            timeout=self.timeout,
            poll_interval=10,
            show_progress=True,
            channel=channel,
            token=token,
            instance=instance,
        )

        # Build rich result (pass pre-calculated QWARD metrics)
        result = self._build_rich_result(
            ibm_result, config, circuit, qward_metrics, original_depth, original_gates
        )

        # Print analysis
        self._print_analysis(result, config)

        # Save results
        if save_results and ibm_result.status == "completed":
            save_path = self._save_results(result, config, ibm_result.backend_name)
            print(f"\nResults saved to: {save_path}")

        return result

    def _print_header(self, config: ConfigT) -> None:
        """Print execution header."""
        config_desc = self.get_config_description(config)

        print("=" * 70)
        print(f"{self.algorithm_name} IBM QPU EXECUTION")
        print("=" * 70)
        print(f"Config ID: {config_desc.get('config_id', 'unknown')}")

        for key, value in config_desc.items():
            if key != "config_id":
                print(f"{key}: {value}")

        print("=" * 70)

    def _build_rich_result(
        self,
        ibm_result: IBMBatchResult,
        config: ConfigT,
        circuit: QuantumCircuit,
        qward_metrics: Dict[str, Any],
        original_depth: int,
        original_gates: int,
    ) -> Dict[str, Any]:
        """Build rich result structure with pre-calculated QWARD metrics."""
        individual_results = []
        success_rates = []

        config_desc = self.get_config_description(config)

        for job in ibm_result.jobs:
            # Get counts and evaluate
            counts = job.counts or {}
            total_shots = sum(counts.values()) if counts else self.shots

            # Evaluate result (algorithm-specific)
            evaluation = self.evaluate_result(counts, config, total_shots)
            success_rates.append(evaluation.get("success_rate", 0.0))

            # Build individual result
            run_result = {
                "experiment_id": f"{config_desc.get('config_id', 'unknown')}_IBM-QPU_{job.optimization_level:03d}",
                "config_id": config_desc.get("config_id", "unknown"),
                "noise_model": "IBM-QPU",
                "optimization_level": job.optimization_level,
                "job_id": job.job_id,
                "timestamp": datetime.now().isoformat(),
                "num_qubits": config_desc.get("num_qubits", circuit.num_qubits),
                # Original circuit metrics (what we designed)
                "circuit_depth": original_depth,
                "total_gates": original_gates,
                # Transpiled circuit metrics (what actually ran)
                "transpiled_depth": job.circuit_depth,
                # QWARD metrics (from original circuit)
                "qward_metrics": qward_metrics,
                "backend_type": "qpu",
                "backend_name": ibm_result.backend_name,
                "shots": total_shots,
                "counts": counts,
                "status": job.status,
                "error": job.error,
            }

            # Add evaluation metrics
            run_result.update(evaluation)

            individual_results.append(run_result)

        # Build batch summary with statistical analysis
        batch_summary = {
            "config_id": config_desc.get("config_id", "unknown"),
            "noise_model": "IBM-QPU",
            "backend_name": ibm_result.backend_name,
            "num_runs": len(individual_results),
            "shots_per_run": self.shots,
            "backend_type": "qpu",
        }

        if success_rates:
            batch_summary.update(
                {
                    "mean_success_rate": statistics.mean(success_rates),
                    "std_success_rate": (
                        statistics.stdev(success_rates) if len(success_rates) > 1 else 0.0
                    ),
                    "min_success_rate": min(success_rates),
                    "max_success_rate": max(success_rates),
                    "median_success_rate": statistics.median(success_rates),
                }
            )

            random_chance = self.get_random_chance(config)
            if random_chance > 0:
                mean_advantage = batch_summary["mean_success_rate"] / random_chance
                batch_summary["mean_quantum_advantage_ratio"] = mean_advantage
                batch_summary["quantum_advantage_demonstrated"] = mean_advantage > 2.0

            # Add statistical analysis
            if len(success_rates) >= 2:
                analysis = calculate_statistical_analysis(
                    success_rates, config_desc.get("config_id", "unknown"), "IBM-QPU"
                )
                batch_summary["analysis"] = analysis

        # Build full result
        result = {
            "config_id": config_desc.get("config_id", "unknown"),
            "noise_id": "IBM-QPU",
            "algorithm": self.algorithm_name,
            "execution_type": "IBM_QPU",
            "saved_at": datetime.now().isoformat(),
            "batch_id": ibm_result.batch_id,
            "backend_name": ibm_result.backend_name,
            "status": ibm_result.status,
            "config": config_desc,
            "batch_summary": batch_summary,
            "individual_results": individual_results,
        }

        return result

    def _print_analysis(self, result: Dict[str, Any], config: ConfigT) -> None:
        """Print quantum advantage analysis."""
        print("\n" + "=" * 70)
        print("QUANTUM ADVANTAGE ANALYSIS")
        print("=" * 70)

        random_chance = self.get_random_chance(config)

        for run in result.get("individual_results", []):
            s_rate = run.get("success_rate", 0)
            opt_level = run.get("optimization_level", "?")
            advantage_ratio = run.get("advantage_ratio", 0)

            print(f"\nOptimization Level {opt_level}:")
            print(f"  Success rate: {s_rate:.2%}")
            print(f"  Random chance: {random_chance:.2%}")
            print(f"  Quantum advantage ratio: {advantage_ratio:.2f}x")
            print(
                f"  Original depth: {run.get('circuit_depth', '?')}, Transpiled: {run.get('transpiled_depth', '?')}"
            )

            thresholds = []
            for t in [30, 50, 70, 90]:
                if run.get(f"threshold_{t}", False):
                    thresholds.append(f"{t}%")
            print(f"  Thresholds passed: {', '.join(thresholds) if thresholds else 'None'}")

            if advantage_ratio > 2:
                print("  Status: QUANTUM ADVANTAGE DEMONSTRATED")
            elif advantage_ratio > 1:
                print("  Status: Better than random (marginal)")
            else:
                print("  Status: No advantage")

        # Summary
        summary = result.get("batch_summary", {})
        if summary.get("mean_success_rate") is not None:
            print("\n--- SUMMARY ---")
            print(f"Mean success rate: {summary['mean_success_rate']:.2%}")
            print(f"Mean quantum advantage: {summary.get('mean_quantum_advantage_ratio', 0):.2f}x")
            if summary.get("quantum_advantage_demonstrated"):
                print("Result: QUANTUM ADVANTAGE DEMONSTRATED")
            else:
                print("Result: No consistent quantum advantage")

    def _save_results(
        self,
        result: Dict[str, Any],
        config: ConfigT,
        backend_name: str,
    ) -> Path:
        """Save results to JSON file."""
        output_dir = self.get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        config_desc = self.get_config_description(config)
        config_id = config_desc.get("config_id", "unknown")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{config_id}_IBM_{backend_name}_{timestamp}.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(result, f, indent=2, default=str)

        return filepath

    # =========================================================================
    # CLI Support
    # =========================================================================

    def create_argument_parser(self) -> argparse.ArgumentParser:
        """Create argument parser for CLI usage."""
        parser = argparse.ArgumentParser(
            description=f"Run {self.algorithm_name} on IBM Quantum hardware"
        )
        parser.add_argument("--config", "-c", default=None, help="Configuration ID to run")
        parser.add_argument(
            "--backend",
            "-b",
            default=None,
            help="IBM backend name (default: auto-select least busy)",
        )
        parser.add_argument(
            "--opt-levels",
            "-o",
            nargs="+",
            type=int,
            default=[0, 1, 2, 3],
            help="Optimization levels to test (default: 0 1 2 3)",
        )
        parser.add_argument(
            "--shots", "-s", type=int, default=1024, help="Number of shots (default: 1024)"
        )
        parser.add_argument(
            "--timeout", "-t", type=int, default=600, help="Timeout in seconds (default: 600)"
        )
        parser.add_argument(
            "--list", "-l", action="store_true", help="List available configurations"
        )
        parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
        parser.add_argument(
            "--channel", default=None, help="IBM Quantum channel ('ibm_quantum' or 'ibm_cloud')"
        )
        parser.add_argument("--token", default=None, help="IBM Quantum API token")
        parser.add_argument(
            "--instance", default=None, help="IBM Quantum instance (e.g., 'ibm-q/open/main')"
        )
        return parser

    def list_configs(self) -> None:
        """Print available configurations."""
        print("=" * 70)
        print(f"AVAILABLE {self.algorithm_name} CONFIGURATIONS FOR IBM QPU")
        print("=" * 70)

        priority_configs = self.get_priority_configs()

        if priority_configs:
            print("\nREGION 1 (Recommended - High Expected Success):")
            print("-" * 70)
            print(f"{'Config':<12} {'Qubits':<8} {'Depth':<8} {'Expected':<12} {'Description'}")
            print("-" * 70)

            for cfg in priority_configs:
                desc = cfg.get("description", "")[:30]
                print(
                    f"{cfg['config_id']:<12} {cfg['qubits']:<8} {cfg['depth']:<8} "
                    f"{cfg['expected_success']:.1%}        {desc}"
                )
        else:
            print("\nAvailable configurations:")
            for config_id in self.get_all_config_ids():
                print(f"  - {config_id}")

        print("\n" + "=" * 70)
        print(f"Usage: python <script>.py --config <CONFIG_ID>")
        print("=" * 70)

    def run_cli(self, args: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Run from command line arguments."""
        parser = self.create_argument_parser()
        parsed = parser.parse_args(args)

        if parsed.list:
            self.list_configs()
            return None

        if parsed.config is None:
            priority = self.get_priority_configs()
            if priority:
                parsed.config = priority[0]["config_id"]
            else:
                configs = self.get_all_config_ids()
                if configs:
                    parsed.config = configs[0]
                else:
                    print("Error: No configurations available")
                    return None

        if parsed.config not in self.get_all_config_ids():
            print(f"Error: Unknown config '{parsed.config}'")
            print(f"Available configs: {self.get_all_config_ids()}")
            return None

        self.shots = parsed.shots
        self.timeout = parsed.timeout
        self.executor = QuantumCircuitExecutor(shots=parsed.shots)

        result = self.run(
            config_id=parsed.config,
            backend_name=parsed.backend,
            optimization_levels=parsed.opt_levels,
            save_results=not parsed.no_save,
            channel=parsed.channel,
            token=parsed.token,
            instance=parsed.instance,
        )

        print("\n" + "=" * 70)
        if result.get("status") == "completed":
            print("EXECUTION COMPLETE")
            summary = result.get("batch_summary", {})
            if summary.get("mean_success_rate") is not None:
                print(f"Mean success rate: {summary['mean_success_rate']:.2%}")
                print(f"Quantum advantage: {summary.get('mean_quantum_advantage_ratio', 0):.2f}x")
        else:
            print(f"EXECUTION STATUS: {result.get('status', 'unknown')}")
        print("=" * 70)

        return result
