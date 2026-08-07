"""Shor experiment configurations for simulator and IBM QPU campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for a Shor order-finding experiment."""

    config_id: str
    N: int
    a: int
    num_control: int
    strategy: str = "permutation"
    true_order: int = 0
    description: str = ""
    for_qpu: bool = False

    @property
    def num_target(self) -> int:
        import math

        return math.floor(math.log2(self.N - 1)) + 1

    @property
    def num_qubits(self) -> int:
        return self.num_control + self.num_target

    @property
    def classical_random_prob(self) -> float:
        """Exact fraction of bitstrings whose CF denominator equals true_order."""
        from qward.algorithms.shor import bitstring_to_phase, phase_to_order

        m = self.num_control
        dim = 2**m
        hits = 0
        for i in range(dim):
            bits = format(i, f"0{m}b")
            phase = bitstring_to_phase(bits, m)
            r, _ = phase_to_order(phase, self.N)
            if r == self.true_order:
                hits += 1
        return hits / dim if dim else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "N": self.N,
            "a": self.a,
            "num_control": self.num_control,
            "num_target": self.num_target,
            "num_qubits": self.num_qubits,
            "strategy": self.strategy,
            "true_order": self.true_order,
            "description": self.description,
            "for_qpu": self.for_qpu,
            "classical_random_prob": self.classical_random_prob,
        }


# QPU m-sweep: N=15, a=7, swap_network (M_7 / M_4)
QPU_CONFIGS: List[ExperimentConfig] = [
    ExperimentConfig(
        "SHOR-N15-M3", 15, 7, 3, "swap_network", 4,
        "N=15 a=7 m=3 shallow safety net (1/2 peak → r=2 nuance)", True,
    ),
    ExperimentConfig(
        "SHOR-N15-M4", 15, 7, 4, "swap_network", 4,
        "N=15 a=7 m=4", True,
    ),
    ExperimentConfig(
        "SHOR-N15-M6", 15, 7, 6, "swap_network", 4,
        "N=15 a=7 m=6", True,
    ),
    ExperimentConfig(
        "SHOR-N15-M8", 15, 7, 8, "swap_network", 4,
        "N=15 a=7 m=8 textbook control size", True,
    ),
]

# Simulator-only extras
SIM_CONFIGS: List[ExperimentConfig] = [
    ExperimentConfig("SHOR-N15-A2-SIM", 15, 2, 8, "permutation", 4, "sim N=15 a=2"),
    ExperimentConfig("SHOR-N15-A7-SIM", 15, 7, 8, "permutation", 4, "sim N=15 a=7"),
    ExperimentConfig("SHOR-N21-A2-SIM", 21, 2, 10, "permutation", 6, "sim N=21 a=2"),
    ExperimentConfig(
        "SHOR-N21-A5-SIM", 21, 5, 10, "permutation", 6,
        "sim N=21 a=5 — even order but trivial x≡-1 failure case",
    ),
]

ALL_EXPERIMENT_CONFIGS = QPU_CONFIGS + SIM_CONFIGS
CONFIGS_BY_ID = {c.config_id: c for c in ALL_EXPERIMENT_CONFIGS}


def get_config(config_id: str) -> ExperimentConfig:
    if config_id not in CONFIGS_BY_ID:
        raise ValueError(
            f"Unknown config ID: {config_id}. Available: {list(CONFIGS_BY_ID)}"
        )
    return CONFIGS_BY_ID[config_id]
