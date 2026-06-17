"""
Variational Teleportation Protocol (vTP) Experiment Configurations

Two protocol types:
- "standard": Uses dynamic circuits (mid-circuit measurement + conditional ops)
- "variation": Fixed CX/CZ corrections (no dynamic circuits needed)

Scaling axes:
- payload_size: 1-5 auxiliary qubits (total = payload + 3)
- gates_per_qubit: 1, 3, 5 (circuit depth per qubit)

Expected outcome: all zeros in measurement register (successful teleportation).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List

GATE_SETS = {
    "SINGLE": ["h"],
    "TRIPLE": ["h", "s", "x"],
    "FIVE": ["h", "s", "x", "z", "h"],
}


@dataclass
class VTPExperimentConfig:
    """Configuration for a single vTP experiment."""

    config_id: str
    payload_size: int
    gates: List[str]
    protocol_type: str
    description: str = ""

    @property
    def total_qubits(self) -> int:
        return self.payload_size + 3

    @property
    def gates_per_qubit(self) -> int:
        return len(self.gates) // self.payload_size if self.payload_size > 0 else 0

    @property
    def search_space(self) -> int:
        return 2**self.payload_size

    @property
    def classical_random_prob(self) -> float:
        return 1.0 / self.search_space

    @property
    def theoretical_success(self) -> float:
        return 1.0

    @property
    def expected_outcome(self) -> str:
        return "0" * self.payload_size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "payload_size": self.payload_size,
            "total_qubits": self.total_qubits,
            "gates": self.gates,
            "gates_per_qubit": self.gates_per_qubit,
            "protocol_type": self.protocol_type,
            "search_space": self.search_space,
            "classical_random_prob": self.classical_random_prob,
            "theoretical_success": self.theoretical_success,
            "expected_outcome": self.expected_outcome,
            "description": self.description,
        }


def _generate_configs() -> List[VTPExperimentConfig]:
    """Generate vTP configs: payload x depth x protocol."""
    configs = []
    for protocol in ["standard", "variation"]:
        proto_label = "S" if protocol == "standard" else "V"
        for payload in range(1, 6):
            for gate_label, gate_set in GATE_SETS.items():
                total_gates = gate_set * payload
                config_id = f"VTP-{proto_label}{payload}-{gate_label}"
                depth_desc = f"{len(gate_set)} gates/qubit"
                configs.append(
                    VTPExperimentConfig(
                        config_id=config_id,
                        payload_size=payload,
                        gates=total_gates,
                        protocol_type=protocol,
                        description=f"{protocol} protocol, {payload} payload qubits, {depth_desc}",
                    )
                )
    return configs


ALL_CONFIGS = _generate_configs()
CONFIGS_BY_ID = {c.config_id: c for c in ALL_CONFIGS}


def get_config(config_id: str) -> VTPExperimentConfig:
    """Get config by ID."""
    if config_id not in CONFIGS_BY_ID:
        available = list(CONFIGS_BY_ID.keys())[:10]
        raise ValueError(f"Unknown config_id: {config_id}. Available: {available}...")
    return CONFIGS_BY_ID[config_id]
