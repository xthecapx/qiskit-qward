"""
Random Volumetric (Mirror Circuit) Experiment Configurations

Mirror circuits apply random 2-qubit layers then their inverse.
Expected output: always |0...0> (identity operation).
Serves as hardware control group — pure noise characterization.

Scaling axes:
- num_qubits: 2-14
- depth_multiplier: 1x, 2x, 3x (depth = n * multiplier)
"""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class RVExperimentConfig:
    """Configuration for a single Random Volumetric experiment."""

    config_id: str
    num_qubits: int
    depth: int
    seed: int = 42
    description: str = ""

    @property
    def search_space(self) -> int:
        return 2**self.num_qubits

    @property
    def classical_random_prob(self) -> float:
        return 1.0 / self.search_space

    @property
    def theoretical_success(self) -> float:
        return 1.0

    @property
    def expected_outcome(self) -> str:
        return "0" * self.num_qubits

    @property
    def depth_multiplier(self) -> float:
        return self.depth / self.num_qubits if self.num_qubits > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "num_qubits": self.num_qubits,
            "depth": self.depth,
            "depth_multiplier": self.depth_multiplier,
            "seed": self.seed,
            "search_space": self.search_space,
            "classical_random_prob": self.classical_random_prob,
            "theoretical_success": self.theoretical_success,
            "expected_outcome": self.expected_outcome,
            "description": self.description,
        }


def _generate_configs() -> List[RVExperimentConfig]:
    """Generate RV configs: qubits x depth multiplier grid."""
    configs = []
    for n in range(2, 15):
        for mult in [1, 2, 3]:
            d = n * mult
            config_id = f"RV{n}-D{d}"
            configs.append(
                RVExperimentConfig(
                    config_id=config_id,
                    num_qubits=n,
                    depth=d,
                    seed=42,
                    description=f"{n} qubits, depth {d} ({mult}x n)",
                )
            )
    return configs


ALL_CONFIGS = _generate_configs()
CONFIGS_BY_ID = {c.config_id: c for c in ALL_CONFIGS}


def get_config(config_id: str) -> RVExperimentConfig:
    """Get config by ID."""
    if config_id not in CONFIGS_BY_ID:
        available = list(CONFIGS_BY_ID.keys())[:10]
        raise ValueError(f"Unknown config_id: {config_id}. Available: {available}...")
    return CONFIGS_BY_ID[config_id]
