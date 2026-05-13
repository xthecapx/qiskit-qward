#!/usr/bin/env python3
"""
Coin-Toss (Ry rotation) AWS Braket Execution Script.

Runs the coin-toss experiment on AWS Braket hardware (Rigetti Ankaa-3 by
default) using the ``AWSExperimentBase`` framework.

Usage:
    python coin_toss_aws.py                        # Run default config (CT1)
    python coin_toss_aws.py --config CT3           # Run specific config
    python coin_toss_aws.py --list                 # List available configs
    python coin_toss_aws.py --characterize         # Run small-scale Rigetti batch

Rigetti uses optimization_level=3 to mirror the choice made in QFT/Grover
AWS scripts.

Example:
    >>> from qward.examples.papers.coin_toss.coin_toss_aws import CoinTossAWSExperiment
    >>> experiment = CoinTossAWSExperiment()
    >>> result = experiment.run("CT3", device_id="Ankaa-3")
"""

import argparse
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from qiskit import QuantumCircuit

from qward.algorithms import AWSJobResult, CoinTossCircuitGenerator
from qward.examples.papers.aws_experiment_base import AWSExperimentBase
from qward.examples.papers.coin_toss.coin_toss_configs import (
    CONFIGS_BY_ID,
    CoinTossExperimentConfig,
    get_config,
)
from qward.examples.papers.coin_toss.coin_toss_ibm import REGION1_PRIORITY


# Use opt_level=3 for Rigetti, mirroring QFT/Grover.
RIGETTI_OPTIMIZATION_LEVEL = 3
# Small-scale Rigetti characterization batch.
RIGETTI_COIN_TOSS_CONFIGS = ["CT1", "CT2", "CT3", "CT3-B25", "CT3-B75"]


class CoinTossAWSExperiment(AWSExperimentBase[CoinTossExperimentConfig]):
    """Coin-toss algorithm experiment runner for AWS Braket."""

    @property
    def algorithm_name(self) -> str:
        return "COIN_TOSS"

    def run(
        self,
        config_id: str,
        device_id: str = "Ankaa-3",
        region: str = "us-west-1",
        save_results: bool = True,
        wait_for_results: bool = True,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        optimization_level: Optional[int] = RIGETTI_OPTIMIZATION_LEVEL,
    ) -> Dict[str, Any]:
        """Run on AWS with optimization_level=3 by default for Rigetti."""
        return super().run(
            config_id=config_id,
            device_id=device_id,
            region=region,
            save_results=save_results,
            wait_for_results=wait_for_results,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            optimization_level=optimization_level,
        )

    def get_config(self, config_id: str) -> CoinTossExperimentConfig:
        """Get coin-toss experiment configuration."""
        return get_config(config_id)

    def get_all_config_ids(self) -> List[str]:
        """Get all available configuration IDs."""
        return list(CONFIGS_BY_ID.keys())

    def create_circuit(self, config: CoinTossExperimentConfig) -> QuantumCircuit:
        """Create the coin-toss test circuit for the given configuration."""
        if config.test_mode == "fair":
            generator = CoinTossCircuitGenerator(
                num_qubits=config.num_qubits,
                test_mode="fair",
                use_barriers=True,
            )
        else:
            generator = CoinTossCircuitGenerator(
                num_qubits=config.num_qubits,
                test_mode="biased",
                theta=config.resolved_theta(),
                use_barriers=True,
            )

        self._current_generator = generator
        return generator.circuit

    def create_success_criteria(
        self, config: CoinTossExperimentConfig
    ) -> Callable[[str], bool]:
        """Create success criteria for the coin-toss experiment."""
        if getattr(self, "_current_generator", None) is not None:
            return self._current_generator.success_criteria

        if config.test_mode == "fair":
            fallback = CoinTossCircuitGenerator(
                num_qubits=config.num_qubits,
                test_mode="fair",
            )
        else:
            fallback = CoinTossCircuitGenerator(
                num_qubits=config.num_qubits,
                test_mode="biased",
                theta=config.resolved_theta(),
            )
        return fallback.success_criteria

    def get_expected_outcomes(self, config: CoinTossExperimentConfig) -> List[str]:
        """Return expected outcomes used for DSR calculations."""
        # Use a fresh generator so we don't depend on _current_generator state.
        if config.test_mode == "fair":
            generator = CoinTossCircuitGenerator(
                num_qubits=config.num_qubits,
                test_mode="fair",
            )
        else:
            generator = CoinTossCircuitGenerator(
                num_qubits=config.num_qubits,
                test_mode="biased",
                theta=config.resolved_theta(),
            )
        return generator.expected_outcomes()

    def get_random_chance(self, config: CoinTossExperimentConfig) -> float:
        """Classical random-guess baseline matching success_criteria."""
        if config.test_mode == "fair":
            return 1.0
        if config.target_p_one is not None and abs(config.target_p_one - 0.5) < 1e-12:
            return 1.0
        return 1.0 / config.search_space

    def get_config_description(self, config: CoinTossExperimentConfig) -> Dict[str, Any]:
        """Get configuration description for saving."""
        desc: Dict[str, Any] = {
            "config_id": config.config_id,
            "experiment_type": config.experiment_type,
            "num_qubits": config.num_qubits,
            "test_mode": config.test_mode,
            "theta": config.resolved_theta(),
            "theta_degrees": math.degrees(config.resolved_theta()),
            "search_space": config.search_space,
            "expected_num_peaks": config.expected_num_peaks,
            "theoretical_gate_count": config.theoretical_gate_count,
            "description": config.description,
        }
        if config.test_mode == "biased":
            desc["target_p_one"] = config.target_p_one
        return desc

    def evaluate_result(
        self,
        counts: Dict[str, int],
        config: CoinTossExperimentConfig,
        total_shots: int,
        aws_result: Optional[AWSJobResult] = None,
    ) -> Dict[str, Any]:
        """Evaluate coin-toss result with algorithm-specific metrics."""
        success_criteria = self.create_success_criteria(config)
        s_count = sum(c for k, c in counts.items() if success_criteria(k))
        s_rate = s_count / total_shots if total_shots > 0 else 0.0

        random_chance = self.get_random_chance(config)
        advantage_ratio = s_rate / random_chance if random_chance > 0 else 0.0

        result: Dict[str, Any] = {
            "success_rate": s_rate,
            "success_count": s_count,
            "test_mode": config.test_mode,
            "random_chance": random_chance,
            "advantage_ratio": advantage_ratio,
            "quantum_advantage": advantage_ratio > 2.0,
            "threshold_30": s_rate >= 0.30,
            "threshold_50": s_rate >= 0.50,
            "threshold_70": s_rate >= 0.70,
            "threshold_90": s_rate >= 0.90,
            "threshold_95": s_rate >= 0.95,
            "threshold_99": s_rate >= 0.99,
        }

        if config.test_mode == "biased":
            result["target_p_one"] = config.target_p_one
            result["theta"] = config.resolved_theta()

        if aws_result is not None:
            result.update(
                {
                    "dsr_michelson": aws_result.dsr_michelson,
                    "dsr_ratio": aws_result.dsr_ratio,
                    "dsr_log_ratio": aws_result.dsr_log_ratio,
                    "dsr_normalized_margin": aws_result.dsr_normalized_margin,
                    "peak_mismatch": aws_result.peak_mismatch,
                }
            )

        return result

    def get_priority_configs(self) -> List[Dict[str, Any]]:
        """Get prioritized configurations for QPU execution."""
        return REGION1_PRIORITY

    def create_argument_parser(self) -> argparse.ArgumentParser:
        """Add --characterize for the Rigetti small-scale coin-toss batch."""
        parser = super().create_argument_parser()
        parser.add_argument(
            "--characterize",
            "-C",
            action="store_true",
            help=(
                "Run Rigetti small-scale batch "
                "(CT1, CT2, CT3, CT3-B25, CT3-B75), one job per config"
            ),
        )
        return parser

    def run_cli(self, args: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Handle --characterize then delegate to base CLI."""
        parser = self.create_argument_parser()
        parsed = parser.parse_args(args)
        if getattr(parsed, "characterize", False):
            return self.run_batch(
                config_ids=RIGETTI_COIN_TOSS_CONFIGS,
                device_id=parsed.device or "Ankaa-3",
                region=parsed.region or "us-west-1",
                save_results=not getattr(parsed, "no_save", False),
                batch_timeout=600,
                aws_access_key_id=parsed.aws_access_key_id,
                aws_secret_access_key=parsed.aws_secret_access_key,
            )
        return super().run_cli(args)

    def get_output_dir(self) -> Path:
        """Get output directory for coin-toss AWS results."""
        return Path(__file__).parent / "data" / "qpu" / "aws"


# =============================================================================
# Convenience Functions
# =============================================================================


def run_coin_toss_on_aws(
    config_id: str = "CT1",
    device_id: str = "Ankaa-3",
    region: str = "us-west-1",
    shots: int = 1024,
    timeout: int = 600,
    save_results: bool = True,
    wait_for_results: bool = True,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Run coin-toss experiment on AWS Braket hardware."""
    experiment = CoinTossAWSExperiment(shots=shots, timeout=timeout)
    return experiment.run(
        config_id=config_id,
        device_id=device_id,
        region=region,
        save_results=save_results,
        wait_for_results=wait_for_results,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


def list_configs() -> None:
    """List available coin-toss configurations."""
    experiment = CoinTossAWSExperiment()
    experiment.list_configs()


# =============================================================================
# Main entry point
# =============================================================================


def main() -> None:
    """Run coin-toss AWS experiment from command line."""
    experiment = CoinTossAWSExperiment()
    experiment.run_cli()


if __name__ == "__main__":
    main()
