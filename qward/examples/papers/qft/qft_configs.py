"""
QFT Experiment Circuit Configurations

This module defines all circuit configurations for the QFT experiment,
including scalability studies for both round-trip and period detection modes.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class QFTExperimentConfig:
    """Configuration for a single QFT experiment."""

    config_id: str
    experiment_type: str  # "scalability_roundtrip", "scalability_period", "period_variation"
    num_qubits: int
    test_mode: str  # "roundtrip" or "period_detection"
    description: str = ""

    # Mode-specific parameters
    input_state: str = None  # For roundtrip mode
    period: int = None  # For period_detection mode

    @property
    def search_space(self) -> int:
        """Total state space size (2^n)."""
        return 2**self.num_qubits

    @property
    def expected_num_peaks(self) -> int:
        """For period detection: number of expected measurement peaks."""
        if self.test_mode == "period_detection" and self.period:
            return self.period
        return 1  # roundtrip expects single peak

    @property
    def theoretical_gate_count(self) -> int:
        """Theoretical gate count: n(n+1)/2 + floor(n/2) for QFT."""
        n = self.num_qubits
        # Note: This is for reference only - use QWARD metrics for actual analysis
        return (n * (n + 1)) // 2 + n // 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for data storage."""
        return {
            "config_id": self.config_id,
            "experiment_type": self.experiment_type,
            "num_qubits": self.num_qubits,
            "test_mode": self.test_mode,
            "input_state": self.input_state,
            "period": self.period,
            "search_space": self.search_space,
            "expected_num_peaks": self.expected_num_peaks,
            "theoretical_gate_count": self.theoretical_gate_count,
            "description": self.description,
        }


@dataclass
class NoiseConfig:
    """Noise model configuration."""

    noise_id: str
    noise_type: str  # "none", "depolarizing", "pauli", "readout", "combined"
    parameters: Dict[str, float]
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "noise_id": self.noise_id,
            "noise_type": self.noise_type,
            "parameters": self.parameters,
            "description": self.description,
        }


# =============================================================================
# Experiment 1: Scalability Study - Round-Trip Mode (SR)
# =============================================================================
# Testing how QFT→QFT⁻¹ success rate degrades as qubit count increases

SCALABILITY_ROUNDTRIP_CONFIGS = [
    QFTExperimentConfig(
        config_id="SR2",
        experiment_type="scalability_roundtrip",
        num_qubits=2,
        test_mode="roundtrip",
        input_state="01",
        description="2 qubits, round-trip, input |01⟩",
    ),
    QFTExperimentConfig(
        config_id="SR3",
        experiment_type="scalability_roundtrip",
        num_qubits=3,
        test_mode="roundtrip",
        input_state="101",
        description="3 qubits, round-trip, input |101⟩",
    ),
    QFTExperimentConfig(
        config_id="SR4",
        experiment_type="scalability_roundtrip",
        num_qubits=4,
        test_mode="roundtrip",
        input_state="0110",
        description="4 qubits, round-trip, input |0110⟩",
    ),
    QFTExperimentConfig(
        config_id="SR5",
        experiment_type="scalability_roundtrip",
        num_qubits=5,
        test_mode="roundtrip",
        input_state="10101",
        description="5 qubits, round-trip, input |10101⟩",
    ),
    QFTExperimentConfig(
        config_id="SR6",
        experiment_type="scalability_roundtrip",
        num_qubits=6,
        test_mode="roundtrip",
        input_state="011001",
        description="6 qubits, round-trip, input |011001⟩",
    ),
    QFTExperimentConfig(
        config_id="SR7",
        experiment_type="scalability_roundtrip",
        num_qubits=7,
        test_mode="roundtrip",
        input_state="1010101",
        description="7 qubits, round-trip, input |1010101⟩",
    ),
    QFTExperimentConfig(
        config_id="SR8",
        experiment_type="scalability_roundtrip",
        num_qubits=8,
        test_mode="roundtrip",
        input_state="01100110",
        description="8 qubits, round-trip, input |01100110⟩",
    ),
    QFTExperimentConfig(
        config_id="SR10",
        experiment_type="scalability_roundtrip",
        num_qubits=10,
        test_mode="roundtrip",
        input_state="0110011001",
        description="10 qubits, round-trip, input |0110011001⟩",
    ),
    QFTExperimentConfig(
        config_id="SR12",
        experiment_type="scalability_roundtrip",
        num_qubits=12,
        test_mode="roundtrip",
        input_state="011001100110",
        description="12 qubits, round-trip (stress test)",
    ),
]


# =============================================================================
# Experiment 2: Scalability Study - Period Detection Mode (SP)
# =============================================================================
# Testing how QFT period detection degrades as qubit count increases

SCALABILITY_PERIOD_CONFIGS = [
    QFTExperimentConfig(
        config_id="SP3-P2",
        experiment_type="scalability_period",
        num_qubits=3,
        test_mode="period_detection",
        period=2,
        description="3 qubits, period=2 (N/period=4, peaks at 0,4)",
    ),
    QFTExperimentConfig(
        config_id="SP4-P2",
        experiment_type="scalability_period",
        num_qubits=4,
        test_mode="period_detection",
        period=2,
        description="4 qubits, period=2 (N/period=8, peaks at 0,8)",
    ),
    QFTExperimentConfig(
        config_id="SP4-P4",
        experiment_type="scalability_period",
        num_qubits=4,
        test_mode="period_detection",
        period=4,
        description="4 qubits, period=4 (N/period=4, peaks at 0,4,8,12)",
    ),
    QFTExperimentConfig(
        config_id="SP5-P4",
        experiment_type="scalability_period",
        num_qubits=5,
        test_mode="period_detection",
        period=4,
        description="5 qubits, period=4",
    ),
    QFTExperimentConfig(
        config_id="SP6-P4",
        experiment_type="scalability_period",
        num_qubits=6,
        test_mode="period_detection",
        period=4,
        description="6 qubits, period=4",
    ),
    QFTExperimentConfig(
        config_id="SP6-P8",
        experiment_type="scalability_period",
        num_qubits=6,
        test_mode="period_detection",
        period=8,
        description="6 qubits, period=8",
    ),
    QFTExperimentConfig(
        config_id="SP8-P4",
        experiment_type="scalability_period",
        num_qubits=8,
        test_mode="period_detection",
        period=4,
        description="8 qubits, period=4",
    ),
    QFTExperimentConfig(
        config_id="SP8-P16",
        experiment_type="scalability_period",
        num_qubits=8,
        test_mode="period_detection",
        period=16,
        description="8 qubits, period=16",
    ),
]


# =============================================================================
# Experiment 3: Period Variation Study (PV)
# =============================================================================
# Testing how different periods affect detection accuracy at fixed qubit count

PERIOD_VARIATION_CONFIGS = [
    # 4 qubits, varying period
    QFTExperimentConfig(
        config_id="PV4-P2",
        experiment_type="period_variation",
        num_qubits=4,
        test_mode="period_detection",
        period=2,
        description="4 qubits, period=2 (easy - far apart peaks)",
    ),
    QFTExperimentConfig(
        config_id="PV4-P4",
        experiment_type="period_variation",
        num_qubits=4,
        test_mode="period_detection",
        period=4,
        description="4 qubits, period=4 (medium)",
    ),
    QFTExperimentConfig(
        config_id="PV4-P8",
        experiment_type="period_variation",
        num_qubits=4,
        test_mode="period_detection",
        period=8,
        description="4 qubits, period=8 (hard - peaks close together)",
    ),
    # 6 qubits, varying period
    QFTExperimentConfig(
        config_id="PV6-P2",
        experiment_type="period_variation",
        num_qubits=6,
        test_mode="period_detection",
        period=2,
        description="6 qubits, period=2",
    ),
    QFTExperimentConfig(
        config_id="PV6-P4",
        experiment_type="period_variation",
        num_qubits=6,
        test_mode="period_detection",
        period=4,
        description="6 qubits, period=4",
    ),
    QFTExperimentConfig(
        config_id="PV6-P8",
        experiment_type="period_variation",
        num_qubits=6,
        test_mode="period_detection",
        period=8,
        description="6 qubits, period=8",
    ),
    QFTExperimentConfig(
        config_id="PV6-P16",
        experiment_type="period_variation",
        num_qubits=6,
        test_mode="period_detection",
        period=16,
        description="6 qubits, period=16",
    ),
]


# =============================================================================
# Experiment 4: Input State Variation (roundtrip)
# =============================================================================
# Testing whether different input states affect roundtrip success

INPUT_VARIATION_CONFIGS = [
    # 4 qubits, different input states
    QFTExperimentConfig(
        config_id="IV4-0000",
        experiment_type="input_variation",
        num_qubits=4,
        test_mode="roundtrip",
        input_state="0000",
        description="4 qubits, all zeros input",
    ),
    QFTExperimentConfig(
        config_id="IV4-1111",
        experiment_type="input_variation",
        num_qubits=4,
        test_mode="roundtrip",
        input_state="1111",
        description="4 qubits, all ones input",
    ),
    QFTExperimentConfig(
        config_id="IV4-0101",
        experiment_type="input_variation",
        num_qubits=4,
        test_mode="roundtrip",
        input_state="0101",
        description="4 qubits, alternating input",
    ),
    QFTExperimentConfig(
        config_id="IV4-1010",
        experiment_type="input_variation",
        num_qubits=4,
        test_mode="roundtrip",
        input_state="1010",
        description="4 qubits, alternating input (opposite)",
    ),
    QFTExperimentConfig(
        config_id="IV4-0001",
        experiment_type="input_variation",
        num_qubits=4,
        test_mode="roundtrip",
        input_state="0001",
        description="4 qubits, single one",
    ),
]


# =============================================================================
# Noise Model Configurations
# =============================================================================

NOISE_CONFIGS = [
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
        description="Low depolarizing noise (0.1% 1Q, 0.5% 2Q)",
    ),
    NoiseConfig(
        noise_id="DEP-MED",
        noise_type="depolarizing",
        parameters={"p1": 0.005, "p2": 0.02},
        description="Medium depolarizing noise (0.5% 1Q, 2% 2Q)",
    ),
    NoiseConfig(
        noise_id="DEP-HIGH",
        noise_type="depolarizing",
        parameters={"p1": 0.01, "p2": 0.05},
        description="High depolarizing noise (1% 1Q, 5% 2Q)",
    ),
    NoiseConfig(
        noise_id="READOUT",
        noise_type="readout",
        parameters={"p01": 0.02, "p10": 0.02},
        description="Readout errors only (2% flip rate)",
    ),
    NoiseConfig(
        noise_id="COMBINED",
        noise_type="combined",
        parameters={"p1": 0.005, "p2": 0.02, "p_readout": 0.02},
        description="Combined depolarizing + readout",
    ),
]


# =============================================================================
# All Configurations
# =============================================================================

ALL_EXPERIMENT_CONFIGS = (
    SCALABILITY_ROUNDTRIP_CONFIGS
    + SCALABILITY_PERIOD_CONFIGS
    + PERIOD_VARIATION_CONFIGS
    + INPUT_VARIATION_CONFIGS
)

# Index by config_id for quick lookup
CONFIGS_BY_ID = {c.config_id: c for c in ALL_EXPERIMENT_CONFIGS}
NOISE_BY_ID = {n.noise_id: n for n in NOISE_CONFIGS}


# =============================================================================
# Helper Functions
# =============================================================================


def get_config(config_id: str) -> QFTExperimentConfig:
    """Get configuration by ID."""
    if config_id not in CONFIGS_BY_ID:
        raise ValueError(f"Unknown config_id: {config_id}. Available: {list(CONFIGS_BY_ID.keys())}")
    return CONFIGS_BY_ID[config_id]


def get_noise_config(noise_id: str) -> NoiseConfig:
    """Get noise configuration by ID."""
    if noise_id not in NOISE_BY_ID:
        raise ValueError(f"Unknown noise_id: {noise_id}. Available: {list(NOISE_BY_ID.keys())}")
    return NOISE_BY_ID[noise_id]


def get_configs_by_type(experiment_type: str) -> List[QFTExperimentConfig]:
    """Get all configurations of a given type."""
    return [c for c in ALL_EXPERIMENT_CONFIGS if c.experiment_type == experiment_type]


def list_all_configs() -> None:
    """Print all available configurations."""
    print("QFT Experiment Configurations")
    print("=" * 60)

    for exp_type in ["scalability_roundtrip", "scalability_period", "period_variation", "input_variation"]:
        configs = get_configs_by_type(exp_type)
        print(f"\n{exp_type.upper()} ({len(configs)} configs):")
        for c in configs:
            if c.test_mode == "roundtrip":
                print(f"  {c.config_id}: {c.num_qubits}q, input={c.input_state}")
            else:
                print(f"  {c.config_id}: {c.num_qubits}q, period={c.period}")

    print(f"\nNOISE MODELS ({len(NOISE_CONFIGS)}):")
    for n in NOISE_CONFIGS:
        print(f"  {n.noise_id}: {n.description}")
