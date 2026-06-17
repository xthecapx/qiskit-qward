#!/usr/bin/env python3
"""
Random Volumetric IBM QPU Execution Script

Mirror circuits serve as hardware control group — no algorithm-specific
structure, pure noise characterization at given width x depth.

Usage:
    uv run python qward/examples/papers/random_volumetric/rv_ibm.py --config RV4-D8
    uv run python qward/examples/papers/random_volumetric/rv_ibm.py --list
"""

from pathlib import Path
from typing import Dict, Any, List, Callable

from qiskit import QuantumCircuit

from qward.algorithms.random_volumetric import RandomVolumetric
from qward.examples.papers.ibm_experiment_base import IBMExperimentBase
from qward.examples.papers.random_volumetric.rv_configs import (
    get_config,
    RVExperimentConfig,
    CONFIGS_BY_ID,
)


class RVIBMExperiment(IBMExperimentBase[RVExperimentConfig]):
    """Random Volumetric experiment runner for IBM QPU."""

    def __init__(self, shots: int = 1024, timeout: int = 600):
        super().__init__(
            shots=shots,
            timeout=timeout,
            output_subdir=str(Path(__file__).resolve().parent / "data" / "qpu" / "raw"),
        )

    @property
    def algorithm_name(self) -> str:
        return "RANDOM-VOLUMETRIC"

    def get_config(self, config_id: str) -> RVExperimentConfig:
        return get_config(config_id)

    def get_all_config_ids(self) -> List[str]:
        return list(CONFIGS_BY_ID.keys())

    def create_circuit(self, config: RVExperimentConfig) -> QuantumCircuit:
        rv = RandomVolumetric(
            num_qubits=config.num_qubits,
            depth=config.depth,
            seed=config.seed,
            use_barriers=True,
        )
        return rv.circuit

    def create_success_criteria(self, config: RVExperimentConfig) -> Callable[[str], bool]:
        expected = config.expected_outcome

        def is_success(result: str) -> bool:
            return result.replace(" ", "").strip() == expected

        return is_success

    def get_random_chance(self, config: RVExperimentConfig) -> float:
        return config.classical_random_prob

    def get_config_description(self, config: RVExperimentConfig) -> Dict[str, Any]:
        return config.to_dict()

    def evaluate_result(
        self,
        counts: Dict[str, int],
        config: RVExperimentConfig,
        total_shots: int,
    ) -> Dict[str, Any]:
        success_criteria = self.create_success_criteria(config)
        s_count = sum(c for k, c in counts.items() if success_criteria(k))
        s_rate = s_count / total_shots if total_shots > 0 else 0.0

        random_chance = self.get_random_chance(config)
        advantage_ratio = s_rate / random_chance if random_chance > 0 else 0.0

        return {
            "success_rate": s_rate,
            "success_count": s_count,
            "num_qubits": config.num_qubits,
            "depth": config.depth,
            "depth_multiplier": config.depth_multiplier,
            "random_chance": random_chance,
            "advantage_ratio": advantage_ratio,
            "quantum_advantage": advantage_ratio > 2.0,
            "threshold_30": s_rate >= 0.30,
            "threshold_50": s_rate >= 0.50,
            "threshold_70": s_rate >= 0.70,
            "threshold_90": s_rate >= 0.90,
        }

    def get_priority_configs(self) -> List[Dict[str, Any]]:
        priorities = []
        for n in range(2, 9):
            for mult in [1, 2]:
                cid = f"RV{n}-D{n*mult}"
                if cid in CONFIGS_BY_ID:
                    cfg = CONFIGS_BY_ID[cid]
                    priorities.append(
                        {
                            "config_id": cid,
                            "num_qubits": cfg.num_qubits,
                            "depth": cfg.depth,
                        }
                    )
        return priorities


if __name__ == "__main__":
    experiment = RVIBMExperiment()
    experiment.run_cli()
