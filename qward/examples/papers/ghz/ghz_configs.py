"""
GHZ State Experiment Configurations

GHZ has O(n) depth (linear in qubit count via CNOT chain).
Expected outcomes: |0...0> and |1...1> each with 50% probability.
"""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class GHZExperimentConfig:
    """Configuration for a single GHZ experiment."""

    config_id: str
    num_qubits: int
    description: str = ""

    @property
    def search_space(self) -> int:
        return 2**self.num_qubits

    @property
    def classical_random_prob(self) -> float:
        return 2.0 / self.search_space

    @property
    def theoretical_success(self) -> float:
        return 1.0

    @property
    def expected_outcomes(self) -> List[str]:
        """GHZ produces |0...0> and |1...1> each with ~50%."""
        return ["0" * self.num_qubits, "1" * self.num_qubits]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "num_qubits": self.num_qubits,
            "expected_outcomes": self.expected_outcomes,
            "search_space": self.search_space,
            "classical_random_prob": self.classical_random_prob,
            "theoretical_success": self.theoretical_success,
            "description": self.description,
        }


def _generate_configs() -> List[GHZExperimentConfig]:
    """Generate GHZ scaling configurations from 2 to 20 qubits."""
    configs = []
    for n in range(2, 21):
        configs.append(
            GHZExperimentConfig(
                config_id=f"GHZ{n}",
                num_qubits=n,
                description=f"{n}-qubit GHZ state",
            )
        )
    return configs


ALL_CONFIGS = _generate_configs()
CONFIGS_BY_ID = {c.config_id: c for c in ALL_CONFIGS}


def get_config(config_id: str) -> GHZExperimentConfig:
    """Get config by ID."""
    if config_id not in CONFIGS_BY_ID:
        available = list(CONFIGS_BY_ID.keys())[:10]
        raise ValueError(f"Unknown config_id: {config_id}. Available: {available}...")
    return CONFIGS_BY_ID[config_id]
