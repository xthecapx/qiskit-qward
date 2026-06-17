#!/usr/bin/env python3
"""
Bernstein-Vazirani IBM QPU Execution Script

BV has O(1) oracle depth — expected to maintain near-perfect success rates
even at high qubit counts, unlike Grover/QFT which degrade with depth.

Usage:
    uv run python qward/examples/papers/bv/bv_ibm.py --config BV3-ONES
    uv run python qward/examples/papers/bv/bv_ibm.py --list
"""

from pathlib import Path
from typing import Dict, Any, List, Callable

from qiskit import QuantumCircuit

from qward.algorithms import BernsteinVazirani
from qward.examples.papers.ibm_experiment_base import IBMExperimentBase
from qward.examples.papers.bv.bv_configs import (
    get_config,
    BVExperimentConfig,
    CONFIGS_BY_ID,
)


class BVIBMExperiment(IBMExperimentBase[BVExperimentConfig]):
    """Bernstein-Vazirani experiment runner for IBM QPU."""

    def __init__(self, shots: int = 1024, timeout: int = 600):
        super().__init__(
            shots=shots,
            timeout=timeout,
            output_subdir=str(Path(__file__).resolve().parent / "data" / "qpu" / "raw"),
        )

    @property
    def algorithm_name(self) -> str:
        return "BERNSTEIN-VAZIRANI"

    def get_config(self, config_id: str) -> BVExperimentConfig:
        return get_config(config_id)

    def get_all_config_ids(self) -> List[str]:
        return list(CONFIGS_BY_ID.keys())

    def create_circuit(self, config: BVExperimentConfig) -> QuantumCircuit:
        bv = BernsteinVazirani(secret_string=config.secret_string, use_barriers=True)
        return bv.circuit

    def create_success_criteria(self, config: BVExperimentConfig) -> Callable[[str], bool]:
        expected = config.expected_outcome

        def is_success(result: str) -> bool:
            clean = result.replace(" ", "").strip()
            # BV measures n qubits (ignoring ancilla)
            return clean == expected

        return is_success

    def get_random_chance(self, config: BVExperimentConfig) -> float:
        return config.classical_random_prob

    def get_config_description(self, config: BVExperimentConfig) -> Dict[str, Any]:
        return config.to_dict()

    def evaluate_result(
        self,
        counts: Dict[str, int],
        config: BVExperimentConfig,
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
            "secret_string": config.secret_string,
            "expected_outcome": config.expected_outcome,
            "random_chance": random_chance,
            "advantage_ratio": advantage_ratio,
            "quantum_advantage": advantage_ratio > 2.0,
            "threshold_30": s_rate >= 0.30,
            "threshold_50": s_rate >= 0.50,
            "threshold_70": s_rate >= 0.70,
            "threshold_90": s_rate >= 0.90,
        }

    def get_priority_configs(self) -> List[Dict[str, Any]]:
        """BV should maintain high success at all qubit counts."""
        return [
            {"config_id": f"BV{n}-ONES", "qubits": n, "expected_success": 1.0} for n in range(2, 15)
        ]


if __name__ == "__main__":
    experiment = BVIBMExperiment()
    experiment.run_cli()
