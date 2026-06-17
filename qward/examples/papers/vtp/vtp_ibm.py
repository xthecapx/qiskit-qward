#!/usr/bin/env python3
"""
Variational Teleportation Protocol IBM QPU Execution Script

Standard protocol uses dynamic circuits (mid-circuit measurement + if_test).
Variation protocol uses fixed CX/CZ corrections.

Usage:
    uv run python qward/examples/papers/vtp/vtp_ibm.py --config VTP-S3-TRIPLE
    uv run python qward/examples/papers/vtp/vtp_ibm.py --list
"""

from pathlib import Path
from typing import Dict, Any, List, Callable

from qiskit import QuantumCircuit

from qward.algorithms import TeleportationCircuitGenerator
from qward.examples.papers.ibm_experiment_base import IBMExperimentBase
from qward.examples.papers.vtp.vtp_configs import (
    get_config,
    VTPExperimentConfig,
    CONFIGS_BY_ID,
)


class VTPIBMExperiment(IBMExperimentBase[VTPExperimentConfig]):
    """vTP experiment runner for IBM QPU."""

    def __init__(self, shots: int = 1024, timeout: int = 600):
        super().__init__(
            shots=shots,
            timeout=timeout,
            output_subdir=str(Path(__file__).resolve().parent / "data" / "qpu" / "raw"),
        )

    @property
    def algorithm_name(self) -> str:
        return "VTP"

    def get_config(self, config_id: str) -> VTPExperimentConfig:
        return get_config(config_id)

    def get_all_config_ids(self) -> List[str]:
        return list(CONFIGS_BY_ID.keys())

    def create_circuit(self, config: VTPExperimentConfig) -> QuantumCircuit:
        gen = TeleportationCircuitGenerator(
            payload_size=config.payload_size,
            gates=config.gates,
            use_barriers=True,
            protocol_type=config.protocol_type,
        )
        return gen.circuit

    def create_success_criteria(self, config: VTPExperimentConfig) -> Callable[[str], bool]:
        expected = config.expected_outcome
        n = config.payload_size

        def is_success(result: str) -> bool:
            parts = result.strip().split(" ")
            if len(parts) > 1:
                test_bits = parts[0]
            else:
                test_bits = result.strip()[:n]
            return test_bits == expected

        return is_success

    def get_random_chance(self, config: VTPExperimentConfig) -> float:
        return config.classical_random_prob

    def get_config_description(self, config: VTPExperimentConfig) -> Dict[str, Any]:
        return config.to_dict()

    def evaluate_result(
        self,
        counts: Dict[str, int],
        config: VTPExperimentConfig,
        total_shots: int,
    ) -> Dict[str, Any]:
        success_criteria = self.create_success_criteria(config)
        s_count = sum(c for k, c in counts.items() if success_criteria(k))
        s_rate = s_count / total_shots if total_shots > 0 else 0.0

        expected = config.expected_outcome
        p_exact = counts.get(expected, 0) / total_shots

        random_chance = self.get_random_chance(config)
        advantage_ratio = s_rate / random_chance if random_chance > 0 else 0.0

        return {
            "success_rate": s_rate,
            "success_count": s_count,
            "p_all_zeros": p_exact,
            "payload_size": config.payload_size,
            "gates_per_qubit": config.gates_per_qubit,
            "protocol_type": config.protocol_type,
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
        for cid, cfg in CONFIGS_BY_ID.items():
            if cfg.protocol_type == "standard" and cfg.payload_size <= 3:
                priorities.append(
                    {
                        "config_id": cid,
                        "total_qubits": cfg.total_qubits,
                        "protocol": cfg.protocol_type,
                        "gates_per_qubit": cfg.gates_per_qubit,
                    }
                )
        return priorities


if __name__ == "__main__":
    experiment = VTPIBMExperiment()
    experiment.run_cli()
