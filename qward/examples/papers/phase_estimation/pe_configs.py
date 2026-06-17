"""
Phase Estimation Experiment Configurations

QPE circuit depth scales with 2^(counting_qubits) controlled-U operations.
Total qubits = counting_qubits + 1 (eigenstate).

Test cases with known phases:
- T gate: phase = 1/8 (exactly representable with 3+ counting qubits)
- S gate: phase = 1/4 (exactly representable with 2+ counting qubits)
- Z gate: phase = 1/2 (exactly representable with 1+ counting qubit)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class PEExperimentConfig:
    """Configuration for a single Phase Estimation experiment."""

    config_id: str
    test_case: str
    num_counting_qubits: int
    expected_phase: float
    description: str = ""

    @property
    def total_qubits(self) -> int:
        return self.num_counting_qubits + 1

    @property
    def search_space(self) -> int:
        return 2**self.num_counting_qubits

    @property
    def classical_random_prob(self) -> float:
        return 1.0 / self.search_space

    @property
    def theoretical_success(self) -> float:
        exact_value = self.expected_phase * self.search_space
        if abs(exact_value - round(exact_value)) < 1e-10:
            return 1.0
        n = self.num_counting_qubits
        delta = self.expected_phase - round(exact_value) / self.search_space
        angle = math.pi * delta
        if abs(angle) < 1e-12:
            return 1.0
        num = math.sin(math.pi * self.search_space * delta)
        den = math.sin(angle)
        return (num / den) ** 2 / (self.search_space**2)

    @property
    def expected_outcome(self) -> str:
        measured_value = round(self.expected_phase * self.search_space) % self.search_space
        return format(measured_value, f"0{self.num_counting_qubits}b")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "test_case": self.test_case,
            "num_counting_qubits": self.num_counting_qubits,
            "total_qubits": self.total_qubits,
            "expected_phase": self.expected_phase,
            "expected_outcome": self.expected_outcome,
            "search_space": self.search_space,
            "classical_random_prob": self.classical_random_prob,
            "theoretical_success": self.theoretical_success,
            "description": self.description,
        }


TEST_CASES = {
    "t_gate": {"phase": 1 / 8, "label": "T"},
    "s_gate": {"phase": 1 / 4, "label": "S"},
    "z_gate": {"phase": 1 / 2, "label": "Z"},
}


def _generate_configs() -> List[PEExperimentConfig]:
    """Generate PE scaling configs: 2-10 counting qubits x 3 test cases."""
    configs = []
    for case_name, case_info in TEST_CASES.items():
        label = case_info["label"]
        phase = case_info["phase"]
        for n in range(2, 11):
            config_id = f"PE-{label}{n}"
            configs.append(
                PEExperimentConfig(
                    config_id=config_id,
                    test_case=case_name,
                    num_counting_qubits=n,
                    expected_phase=phase,
                    description=f"{case_name} with {n} counting qubits (total={n+1}q)",
                )
            )
    return configs


ALL_CONFIGS = _generate_configs()
CONFIGS_BY_ID = {c.config_id: c for c in ALL_CONFIGS}


def get_config(config_id: str) -> PEExperimentConfig:
    """Get config by ID."""
    if config_id not in CONFIGS_BY_ID:
        available = list(CONFIGS_BY_ID.keys())[:10]
        raise ValueError(f"Unknown config_id: {config_id}. Available: {available}...")
    return CONFIGS_BY_ID[config_id]
