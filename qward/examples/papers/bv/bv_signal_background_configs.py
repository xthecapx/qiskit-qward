"""IBM campaign configurations for the BV signal-plus-background experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from qward.examples.papers.bv.bv_signal_background import (
    CAMPAIGN_TOTAL_QUBITS,
    DEFAULT_SEED,
    DEFAULT_TARGET_MASS,
    SignalBackgroundSpec,
    make_spec,
)


@dataclass(frozen=True)
class BVSignalBackgroundConfig:
    """Configuration for one IBM signal-plus-background campaign."""

    config_id: str
    num_total_qubits: int
    target_mass: float = DEFAULT_TARGET_MASS
    seed: int = DEFAULT_SEED

    @property
    def spec(self) -> SignalBackgroundSpec:
        """Return the circuit-level configuration."""
        return make_spec(
            self.num_total_qubits,
            target_mass=self.target_mass,
            seed=self.seed,
        )

    @property
    def num_qubits(self) -> int:
        """Alias used by the shared IBM experiment schema."""
        return self.num_total_qubits

    @property
    def expected_outcomes(self) -> tuple[str, str]:
        """Return the two analytically known measured targets."""
        return self.spec.targets

    @property
    def random_chance(self) -> float:
        """Return the chance of randomly drawing either target."""
        return len(self.expected_outcomes) / (2**self.spec.num_measured_qubits)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the campaign configuration."""
        spec = self.spec
        return {
            "config_id": self.config_id,
            "num_qubits": self.num_total_qubits,
            "num_total_qubits": self.num_total_qubits,
            "num_data_qubits": spec.num_data_qubits,
            "num_measured_qubits": spec.num_measured_qubits,
            "target_mass": self.target_mass,
            "background_mass": spec.background_mass,
            "secret_string": spec.secret_string,
            "flip_qubits": list(spec.flip_qubits),
            "expected_outcomes": list(spec.targets),
            "random_chance": self.random_chance,
            "seed": self.seed,
            "ideal_histogram_method": "dense_statevector",
            "ideal_histogram_status": "not_computed_beyond_local_wall",
        }


CONFIGS = {
    f"BVSB{total_qubits}": BVSignalBackgroundConfig(
        config_id=f"BVSB{total_qubits}",
        num_total_qubits=total_qubits,
    )
    for total_qubits in CAMPAIGN_TOTAL_QUBITS
}


def get_config(config_id: str) -> BVSignalBackgroundConfig:
    """Return a campaign configuration by identifier."""
    try:
        return CONFIGS[config_id]
    except KeyError as error:
        raise ValueError(
            f"Unknown config_id {config_id!r}; available: {sorted(CONFIGS)}"
        ) from error
