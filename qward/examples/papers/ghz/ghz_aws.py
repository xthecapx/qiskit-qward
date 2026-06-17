#!/usr/bin/env python3
"""
GHZ State AWS Braket Execution Script

Usage:
    uv run python qward/examples/papers/ghz/ghz_aws.py --config GHZ3
    uv run python qward/examples/papers/ghz/ghz_aws.py --list
"""

from pathlib import Path
from typing import Any, Dict, List, Callable, Optional

from qiskit import QuantumCircuit

from qward.algorithms import GHZ
from qward.examples.papers.aws_experiment_base import AWSExperimentBase
from qward.examples.papers.ghz.ghz_configs import (
    get_config,
    GHZExperimentConfig,
    CONFIGS_BY_ID,
)

RIGETTI_OPTIMIZATION_LEVEL = 3


class GHZAWSExperiment(AWSExperimentBase[GHZExperimentConfig]):
    """GHZ state experiment runner for AWS Braket."""

    def __init__(self, shots: int = 1024, timeout: int = 600):
        super().__init__(
            shots=shots,
            timeout=timeout,
            output_subdir=str(Path(__file__).resolve().parent / "data" / "qpu" / "aws"),
        )

    @property
    def algorithm_name(self) -> str:
        return "GHZ"

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

    def get_config(self, config_id: str) -> GHZExperimentConfig:
        return get_config(config_id)

    def get_all_config_ids(self) -> List[str]:
        return list(CONFIGS_BY_ID.keys())

    def create_circuit(self, config: GHZExperimentConfig) -> QuantumCircuit:
        ghz = GHZ(num_qubits=config.num_qubits, use_barriers=True)
        return ghz.circuit

    def create_success_criteria(self, config: GHZExperimentConfig) -> Callable[[str], bool]:
        expected = config.expected_outcomes

        def is_success(result: str) -> bool:
            return result.replace(" ", "").strip() in expected

        return is_success

    def get_expected_outcomes(self, config: GHZExperimentConfig) -> List[str]:
        return config.expected_outcomes

    def get_random_chance(self, config: GHZExperimentConfig) -> float:
        return config.classical_random_prob

    def get_config_description(self, config: GHZExperimentConfig) -> Dict[str, Any]:
        return config.to_dict()

    def evaluate_result(
        self,
        counts: Dict[str, int],
        config: GHZExperimentConfig,
        total_shots: int,
    ) -> Dict[str, Any]:
        success_criteria = self.create_success_criteria(config)
        s_count = sum(c for k, c in counts.items() if success_criteria(k))
        s_rate = s_count / total_shots if total_shots > 0 else 0.0

        random_chance = self.get_random_chance(config)
        advantage_ratio = s_rate / random_chance if random_chance > 0 else 0.0

        all_zeros = "0" * config.num_qubits
        all_ones = "1" * config.num_qubits
        p_zeros = counts.get(all_zeros, 0) / total_shots
        p_ones = counts.get(all_ones, 0) / total_shots

        return {
            "success_rate": s_rate,
            "success_count": s_count,
            "p_all_zeros": p_zeros,
            "p_all_ones": p_ones,
            "balance_ratio": (
                min(p_zeros, p_ones) / max(p_zeros, p_ones) if max(p_zeros, p_ones) > 0 else 0
            ),
            "random_chance": random_chance,
            "advantage_ratio": advantage_ratio,
            "quantum_advantage": advantage_ratio > 2.0,
            "threshold_30": s_rate >= 0.30,
            "threshold_50": s_rate >= 0.50,
            "threshold_70": s_rate >= 0.70,
            "threshold_90": s_rate >= 0.90,
        }


if __name__ == "__main__":
    experiment = GHZAWSExperiment()
    experiment.run_cli()
