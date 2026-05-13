"""
Coin-Toss (Ry rotation) Experiment Configurations.

Defines circuit configurations for the coin-toss experiment:

- Fair (CT<n>):       theta = pi/2 on every qubit  -> uniform distribution.
- Biased (CT<n>-B<P>): same theta on every qubit such that
                       P(|1>) = P/100  -> "cheated" coin.

The same NoiseConfig list used by QFT is reused so noise sweeps are
directly comparable across algorithms.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from qward.algorithms.noise_generator import NoiseConfig

# Default theta for fair coin: Ry(pi/2)|0> = (|0> + |1>) / sqrt(2)
FAIR_THETA = math.pi / 2


def theta_for_p1(p_one: float) -> float:
    """Compute the Ry angle that yields a target P(|1>).

    P(|1>) = sin^2(theta / 2)  =>  theta = 2 * arcsin(sqrt(P)).

    Args:
        p_one: Desired probability of measuring |1> per qubit, in [0, 1].

    Returns:
        Rotation angle theta in radians.
    """
    if not 0.0 <= p_one <= 1.0:
        raise ValueError(f"p_one must be in [0, 1], got {p_one}")
    return 2.0 * math.asin(math.sqrt(p_one))


@dataclass
class CoinTossExperimentConfig:
    """Configuration for a single coin-toss experiment.

    Attributes:
        config_id: Unique config identifier (e.g. "CT3", "CT3-B25").
        experiment_type: High-level grouping for analysis
            ("fair_scaling" or "biased_scaling").
        num_qubits: Number of independent Ry rotations (= coins).
        test_mode: "fair" or "biased".
        theta: Rotation angle in radians.  None means fair (pi/2).
        target_p_one: For biased configs, the target P(|1>) per qubit.
            Stored for reporting / metadata only.
        description: Human-readable description.
        hardware_only: If True, skip in simulator (very large search space).
    """

    config_id: str
    experiment_type: str
    num_qubits: int
    test_mode: str = "fair"
    theta: Optional[float] = None
    target_p_one: Optional[float] = None
    description: str = ""
    hardware_only: bool = False
    thetas_per_qubit: Optional[List[float]] = field(default=None, repr=False)

    @property
    def search_space(self) -> int:
        """Total state-space size (2^n)."""
        return 2**self.num_qubits

    @property
    def expected_num_peaks(self) -> int:
        """Number of theoretically dominant outcomes.

        - Fair: every outcome is equally likely -> count all 2^n as peaks.
        - Biased symmetric (single theta): one mode (all-zeros if P<0.5,
          all-ones if P>0.5; degenerate at exactly P=0.5).
        """
        if self.test_mode == "fair":
            return self.search_space
        if self.target_p_one is None:
            return 1
        if abs(self.target_p_one - 0.5) < 1e-12:
            return self.search_space
        return 1

    @property
    def theoretical_gate_count(self) -> int:
        """Theoretical gate count: one Ry per qubit (+ measurements)."""
        return self.num_qubits

    def resolved_theta(self) -> float:
        """Return the (scalar) rotation angle, defaulting to fair."""
        if self.theta is not None:
            return self.theta
        return FAIR_THETA

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for data storage."""
        return {
            "config_id": self.config_id,
            "experiment_type": self.experiment_type,
            "num_qubits": self.num_qubits,
            "test_mode": self.test_mode,
            "theta": self.resolved_theta(),
            "target_p_one": self.target_p_one,
            "search_space": self.search_space,
            "expected_num_peaks": self.expected_num_peaks,
            "theoretical_gate_count": self.theoretical_gate_count,
            "description": self.description,
            "hardware_only": self.hardware_only,
        }


# =============================================================================
# Experiment 1: Fair coin scaling (CT)
# =============================================================================
# theta = pi/2 on every qubit  ->  uniform 1/2^n distribution.
# Used to validate hardware sampling uniformity / randomness.

FAIR_CONFIGS: List[CoinTossExperimentConfig] = [
    CoinTossExperimentConfig(
        config_id="CT1",
        experiment_type="fair_scaling",
        num_qubits=1,
        test_mode="fair",
        theta=FAIR_THETA,
        description="1 qubit fair coin (single Ry(pi/2))",
    ),
    CoinTossExperimentConfig(
        config_id="CT2",
        experiment_type="fair_scaling",
        num_qubits=2,
        test_mode="fair",
        theta=FAIR_THETA,
        description="2 qubits fair (uniform over 4 outcomes)",
    ),
    CoinTossExperimentConfig(
        config_id="CT3",
        experiment_type="fair_scaling",
        num_qubits=3,
        test_mode="fair",
        theta=FAIR_THETA,
        description="3 qubits fair (uniform over 8 outcomes)",
    ),
    CoinTossExperimentConfig(
        config_id="CT4",
        experiment_type="fair_scaling",
        num_qubits=4,
        test_mode="fair",
        theta=FAIR_THETA,
        description="4 qubits fair (uniform over 16 outcomes)",
    ),
    CoinTossExperimentConfig(
        config_id="CT5",
        experiment_type="fair_scaling",
        num_qubits=5,
        test_mode="fair",
        theta=FAIR_THETA,
        description="5 qubits fair (uniform over 32 outcomes)",
    ),
]


# =============================================================================
# Experiment 2: Biased coin scaling (CT<n>-B<P>)
# =============================================================================
# Same theta on every qubit, chosen so P(|1>) = P/100 on each qubit.
# Allows the QPU to be tested against a non-uniform target distribution.

_BIASED_TARGETS = (
    # (qubits, p_one_percent, description_suffix)
    (1, 25, "P(1)=0.25 (single biased coin towards 0)"),
    (1, 75, "P(1)=0.75 (single biased coin towards 1)"),
    (3, 25, "P(1)=0.25 per qubit (mode |000>)"),
    (3, 75, "P(1)=0.75 per qubit (mode |111>)"),
    (5, 25, "P(1)=0.25 per qubit (mode |00000>)"),
    (5, 75, "P(1)=0.75 per qubit (mode |11111>)"),
)


def _build_biased_configs() -> List[CoinTossExperimentConfig]:
    configs: List[CoinTossExperimentConfig] = []
    for num_qubits, p_pct, suffix in _BIASED_TARGETS:
        p_one = p_pct / 100.0
        theta = theta_for_p1(p_one)
        configs.append(
            CoinTossExperimentConfig(
                config_id=f"CT{num_qubits}-B{p_pct}",
                experiment_type="biased_scaling",
                num_qubits=num_qubits,
                test_mode="biased",
                theta=theta,
                target_p_one=p_one,
                description=f"{num_qubits} qubits, {suffix}",
            )
        )
    return configs


BIASED_CONFIGS: List[CoinTossExperimentConfig] = _build_biased_configs()


# =============================================================================
# Noise Model Configurations (mirrors qft_configs for cross-algorithm parity)
# =============================================================================

NOISE_CONFIGS: List[NoiseConfig] = [
    NoiseConfig(
        noise_id="IDEAL",
        noise_type="none",
        parameters={},
        description="Ideal simulation (no noise)",
    ),
    NoiseConfig(
        noise_id="DEP-LOW",
        noise_type="depolarizing",
        parameters={"p1": 0.001, "p2": 0.005},
        description="Low depolarizing (p1=0.1%, p2=0.5%)",
    ),
    NoiseConfig(
        noise_id="DEP-MED",
        noise_type="depolarizing",
        parameters={"p1": 0.005, "p2": 0.02},
        description="Medium depolarizing (p1=0.5%, p2=2%)",
    ),
    NoiseConfig(
        noise_id="DEP-HIGH",
        noise_type="depolarizing",
        parameters={"p1": 0.01, "p2": 0.05},
        description="High depolarizing (p1=1%, p2=5%)",
    ),
    NoiseConfig(
        noise_id="IBM-HERON-R3",
        noise_type="combined",
        parameters={"p1": 0.0005, "p2": 0.00115, "p_readout": 0.0046},
        description="IBM Heron r3 (ibm_boston): 2Q=0.113%, readout=0.46%",
    ),
    NoiseConfig(
        noise_id="IBM-HERON-R2",
        noise_type="combined",
        parameters={"p1": 0.001, "p2": 0.0026, "p_readout": 0.0127},
        description="IBM Heron r2 (ibm_marrakesh): 2Q=0.26%, readout=1.27%",
    ),
    NoiseConfig(
        noise_id="IBM-HERON-R1",
        noise_type="combined",
        parameters={"p1": 0.001, "p2": 0.0025, "p_readout": 0.0293},
        description="IBM Heron r1 (ibm_torino): 2Q=0.25%, readout=2.93%",
    ),
    NoiseConfig(
        noise_id="RIGETTI-ANKAA3",
        noise_type="combined",
        parameters={"p1": 0.002, "p2": 0.005, "p_readout": 0.02},
        description="Rigetti Ankaa-3 (84q): 99.5% 2Q fidelity (0.5% error)",
    ),
    NoiseConfig(
        noise_id="READOUT",
        noise_type="readout",
        parameters={"p01": 0.02, "p10": 0.02},
        description="Readout errors only (2% symmetric flip)",
    ),
    NoiseConfig(
        noise_id="COMBINED",
        noise_type="combined",
        parameters={"p1": 0.005, "p2": 0.015, "p_readout": 0.02},
        description="Generic combined noise (realistic NISQ baseline)",
    ),
]


# =============================================================================
# Aggregated views
# =============================================================================

ALL_EXPERIMENT_CONFIGS: List[CoinTossExperimentConfig] = FAIR_CONFIGS + BIASED_CONFIGS

SIMULATOR_CONFIGS: List[CoinTossExperimentConfig] = [
    c for c in ALL_EXPERIMENT_CONFIGS if not c.hardware_only
]
HARDWARE_ONLY_CONFIGS: List[CoinTossExperimentConfig] = [
    c for c in ALL_EXPERIMENT_CONFIGS if c.hardware_only
]

CONFIGS_BY_ID: Dict[str, CoinTossExperimentConfig] = {
    c.config_id: c for c in ALL_EXPERIMENT_CONFIGS
}
NOISE_BY_ID: Dict[str, NoiseConfig] = {n.noise_id: n for n in NOISE_CONFIGS}


# =============================================================================
# Helpers
# =============================================================================


def get_config(config_id: str) -> CoinTossExperimentConfig:
    """Fetch a coin-toss config by id."""
    if config_id not in CONFIGS_BY_ID:
        raise ValueError(f"Unknown config_id: {config_id}. Available: {list(CONFIGS_BY_ID.keys())}")
    return CONFIGS_BY_ID[config_id]


def get_noise_config(noise_id: str) -> NoiseConfig:
    """Fetch a noise config by id."""
    if noise_id not in NOISE_BY_ID:
        raise ValueError(f"Unknown noise_id: {noise_id}. Available: {list(NOISE_BY_ID.keys())}")
    return NOISE_BY_ID[noise_id]


def get_configs_by_type(
    experiment_type: str, include_hardware_only: bool = True
) -> List[CoinTossExperimentConfig]:
    """All configs of a given experiment_type."""
    configs = [c for c in ALL_EXPERIMENT_CONFIGS if c.experiment_type == experiment_type]
    if not include_hardware_only:
        configs = [c for c in configs if not c.hardware_only]
    return configs


def get_simulator_configs() -> List[CoinTossExperimentConfig]:
    """Configs suitable for simulator execution (excludes hardware-only)."""
    return SIMULATOR_CONFIGS


def get_simulator_config_ids() -> List[str]:
    """Config ids suitable for simulator execution."""
    return [c.config_id for c in SIMULATOR_CONFIGS]


def get_hardware_only_configs() -> List[CoinTossExperimentConfig]:
    """Configs that should only run on real hardware."""
    return HARDWARE_ONLY_CONFIGS


def list_all_configs() -> None:
    """Print all available coin-toss configurations."""
    print("Coin-Toss Experiment Configurations")
    print("=" * 60)

    for exp_type in ("fair_scaling", "biased_scaling"):
        configs = get_configs_by_type(exp_type)
        print(f"\n{exp_type.upper()} ({len(configs)} configs):")
        for c in configs:
            hw_tag = " [HARDWARE ONLY]" if c.hardware_only else ""
            theta_deg = math.degrees(c.resolved_theta())
            extra = f"target P(1)={c.target_p_one:.2f}" if c.target_p_one is not None else "fair"
            print(
                f"  {c.config_id}: {c.num_qubits}q, theta={theta_deg:.1f} deg, " f"{extra}{hw_tag}"
            )

    print("\nSUMMARY:")
    print(f"  Total configs: {len(ALL_EXPERIMENT_CONFIGS)}")
    print(f"  Simulator-friendly: {len(SIMULATOR_CONFIGS)}")
    print(f"  Hardware-only: {len(HARDWARE_ONLY_CONFIGS)}")

    print(f"\nNOISE MODELS ({len(NOISE_CONFIGS)}):")
    for n in NOISE_CONFIGS:
        print(f"  {n.noise_id}: {n.description}")
