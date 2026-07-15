"""
Bernstein-Vazirani Experiment Configurations

BV has O(1) depth (independent of n), making it highly resilient to noise.
Expected to maintain high success rates even at large qubit counts.
"""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class BVExperimentConfig:
    """Configuration for a single BV experiment."""

    config_id: str
    num_qubits: int
    secret_string: str
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
        """Expected measurement (Qiskit little-endian: reversed)."""
        return self.secret_string[::-1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "num_qubits": self.num_qubits,
            "secret_string": self.secret_string,
            "expected_outcome": self.expected_outcome,
            "search_space": self.search_space,
            "classical_random_prob": self.classical_random_prob,
            "theoretical_success": self.theoretical_success,
            "description": self.description,
        }


def _generate_configs() -> List[BVExperimentConfig]:
    """Generate BV experiment configurations.

    Includes:
      * Scalability ladder for the combined DSR figure: n = 2..14
      * Beyond-wall sizes for the HF/TVDF-infeasibility section: n = 29, 30, 31
        (total qubits = n+1 = 30, 31, 32 — past the classical statevector wall)
    """
    configs = []
    # Combined-plot ladder + mid sizes; wall demonstration at 29..31.
    qubit_sizes = list(range(2, 15)) + [29, 30, 31]
    for n in qubit_sizes:
        # All-ones
        configs.append(
            BVExperimentConfig(
                config_id=f"BV{n}-ONES",
                num_qubits=n,
                secret_string="1" * n,
                description=f"{n} qubits, all-ones secret",
            )
        )
        # Alternating (default pattern for paper figures)
        alt = "".join("1" if i % 2 == 0 else "0" for i in range(n))
        configs.append(
            BVExperimentConfig(
                config_id=f"BV{n}-ALT",
                num_qubits=n,
                secret_string=alt,
                description=f"{n} qubits, alternating secret",
            )
        )
        # Single-bit (last position)
        single = "0" * (n - 1) + "1"
        configs.append(
            BVExperimentConfig(
                config_id=f"BV{n}-SINGLE",
                num_qubits=n,
                secret_string=single,
                description=f"{n} qubits, single-bit secret",
            )
        )
    return configs


ALL_CONFIGS = _generate_configs()
CONFIGS_BY_ID = {c.config_id: c for c in ALL_CONFIGS}


def get_config(config_id: str) -> BVExperimentConfig:
    """Get config by ID."""
    if config_id not in CONFIGS_BY_ID:
        available = list(CONFIGS_BY_ID.keys())[:10]
        raise ValueError(f"Unknown config_id: {config_id}. Available: {available}...")
    return CONFIGS_BY_ID[config_id]
