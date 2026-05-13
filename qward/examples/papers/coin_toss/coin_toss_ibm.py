#!/usr/bin/env python3
"""
Coin-Toss (Ry rotation) IBM QPU Execution Script.

Runs the coin-toss experiment on IBM Quantum hardware using the
``IBMExperimentBase`` framework.

Usage:
    python coin_toss_ibm.py                        # Run default config (CT1)
    python coin_toss_ibm.py --config CT3           # Run specific config
    python coin_toss_ibm.py --list                 # List available configs

Example:
    >>> from qward.examples.papers.coin_toss.coin_toss_ibm import CoinTossIBMExperiment
    >>> experiment = CoinTossIBMExperiment()
    >>> result = experiment.run("CT3")
"""

import math
from pathlib import Path
from typing import Any, Callable, Dict, List

from qiskit import QuantumCircuit

from qward.algorithms import CoinTossCircuitGenerator
from qward.examples.papers.coin_toss.coin_toss_configs import (
    CONFIGS_BY_ID,
    CoinTossExperimentConfig,
    get_config,
)
from qward.examples.papers.ibm_experiment_base import IBMExperimentBase


# =============================================================================
# Region 1 / priority configurations for QPU execution
# =============================================================================
# Coin-toss circuits are extremely shallow (depth 1 + measurements), so all
# fair and biased configs are reasonable on hardware.  We list them here in
# scaling order so --batch picks the smallest first.

REGION1_PRIORITY: List[Dict[str, Any]] = [
    {
        "config_id": "CT1",
        "expected_success": 1.000,
        "qubits": 1,
        "depth": 1,
        "description": "fair coin (1 qubit)",
    },
    {
        "config_id": "CT1-B25",
        "expected_success": 0.750,
        "qubits": 1,
        "depth": 1,
        "description": "biased P(1)=0.25 (1 qubit)",
    },
    {
        "config_id": "CT1-B75",
        "expected_success": 0.750,
        "qubits": 1,
        "depth": 1,
        "description": "biased P(1)=0.75 (1 qubit)",
    },
    {
        "config_id": "CT2",
        "expected_success": 1.000,
        "qubits": 2,
        "depth": 1,
        "description": "fair (2 qubits)",
    },
    {
        "config_id": "CT3",
        "expected_success": 1.000,
        "qubits": 3,
        "depth": 1,
        "description": "fair (3 qubits)",
    },
    {
        "config_id": "CT3-B25",
        "expected_success": 0.421875,
        "qubits": 3,
        "depth": 1,
        "description": "biased P(1)=0.25 (3 qubits, mode |000>)",
    },
    {
        "config_id": "CT3-B75",
        "expected_success": 0.421875,
        "qubits": 3,
        "depth": 1,
        "description": "biased P(1)=0.75 (3 qubits, mode |111>)",
    },
    {
        "config_id": "CT4",
        "expected_success": 1.000,
        "qubits": 4,
        "depth": 1,
        "description": "fair (4 qubits)",
    },
    {
        "config_id": "CT5",
        "expected_success": 1.000,
        "qubits": 5,
        "depth": 1,
        "description": "fair (5 qubits)",
    },
    {
        "config_id": "CT5-B25",
        "expected_success": 0.2373046875,
        "qubits": 5,
        "depth": 1,
        "description": "biased P(1)=0.25 (5 qubits, mode |00000>)",
    },
    {
        "config_id": "CT5-B75",
        "expected_success": 0.2373046875,
        "qubits": 5,
        "depth": 1,
        "description": "biased P(1)=0.75 (5 qubits, mode |11111>)",
    },
]


class CoinTossIBMExperiment(IBMExperimentBase[CoinTossExperimentConfig]):
    """Coin-toss algorithm experiment runner for IBM QPU."""

    @property
    def algorithm_name(self) -> str:
        return "COIN_TOSS"

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

        # Fallback: rebuild a generator just for the criterion (no circuit reuse).
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

    def get_random_chance(self, config: CoinTossExperimentConfig) -> float:
        """Classical random-guess baseline matching success_criteria."""
        if config.test_mode == "fair":
            # Every outcome is "valid" -> baseline is 1.0.
            return 1.0
        # Biased symmetric (single theta): one mode unless P=0.5 exactly.
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

        return result

    def get_priority_configs(self) -> List[Dict[str, Any]]:
        """Get prioritized configurations for QPU execution."""
        return REGION1_PRIORITY

    def get_output_dir(self) -> Path:
        """Get output directory for coin-toss IBM results."""
        return Path(__file__).parent / "data" / "qpu" / "raw"


# =============================================================================
# Convenience Functions
# =============================================================================


def run_coin_toss_on_ibm(
    config_id: str = "CT1",
    backend_name: str = None,
    optimization_levels: List[int] = None,
    shots: int = 1024,
    timeout: int = 600,
    save_results: bool = True,
    channel: str = None,
    token: str = None,
    instance: str = None,
) -> Dict[str, Any]:
    """Run coin-toss experiment on IBM Quantum hardware."""
    experiment = CoinTossIBMExperiment(shots=shots, timeout=timeout)
    return experiment.run(
        config_id=config_id,
        backend_name=backend_name,
        optimization_levels=optimization_levels,
        save_results=save_results,
        channel=channel,
        token=token,
        instance=instance,
    )


def list_configs() -> None:
    """List available coin-toss configurations."""
    experiment = CoinTossIBMExperiment()
    experiment.list_configs()


# =============================================================================
# Main entry point
# =============================================================================


def main() -> None:
    """Run coin-toss IBM experiment from command line."""
    experiment = CoinTossIBMExperiment()
    experiment.run_cli()


if __name__ == "__main__":
    main()
