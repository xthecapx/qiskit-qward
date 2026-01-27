"""
QFT Experiment Circuit Configurations

This module defines all circuit configurations for the QFT experiment,
including scalability studies for both round-trip and period detection modes.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

from qward.algorithms.noise_generator import NoiseConfig


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

    # Execution constraints
    hardware_only: bool = False  # If True, skip in simulator (too expensive)

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
            "hardware_only": self.hardware_only,
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
        description="10 qubits, round-trip (HARDWARE ONLY - too slow for simulator)",
        hardware_only=True,  # 2^10 = 1024 states, expensive to simulate
    ),
    QFTExperimentConfig(
        config_id="SR12",
        experiment_type="scalability_roundtrip",
        num_qubits=12,
        test_mode="roundtrip",
        input_state="011001100110",
        description="12 qubits, round-trip (HARDWARE ONLY - stress test)",
        hardware_only=True,  # 2^12 = 4096 states, very expensive to simulate
    ),
    QFTExperimentConfig(
        config_id="SR14",
        experiment_type="scalability_roundtrip",
        num_qubits=14,
        test_mode="roundtrip",
        input_state="01100110011001",
        description="14 qubits, round-trip (HARDWARE ONLY - extreme test)",
        hardware_only=True,  # 2^14 = 16384 states
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
        description="8 qubits, period=4 (HARDWARE ONLY - 9 total qubits)",
        hardware_only=True,  # 8 counting + 1 ancilla = 9 qubits
    ),
    QFTExperimentConfig(
        config_id="SP8-P16",
        experiment_type="scalability_period",
        num_qubits=8,
        test_mode="period_detection",
        period=16,
        description="8 qubits, period=16 (HARDWARE ONLY - 9 total qubits)",
        hardware_only=True,  # 8 counting + 1 ancilla = 9 qubits
    ),
    QFTExperimentConfig(
        config_id="SP10-P4",
        experiment_type="scalability_period",
        num_qubits=10,
        test_mode="period_detection",
        period=4,
        description="10 qubits, period=4 (HARDWARE ONLY - 11 total qubits)",
        hardware_only=True,
    ),
    QFTExperimentConfig(
        config_id="SP10-P16",
        experiment_type="scalability_period",
        num_qubits=10,
        test_mode="period_detection",
        period=16,
        description="10 qubits, period=16 (HARDWARE ONLY)",
        hardware_only=True,
    ),
    QFTExperimentConfig(
        config_id="SP12-P8",
        experiment_type="scalability_period",
        num_qubits=12,
        test_mode="period_detection",
        period=8,
        description="12 qubits, period=8 (HARDWARE ONLY - extreme test)",
        hardware_only=True,
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
# These noise configurations are based on empirical data from quantum hardware
# providers. See qward/algorithms/noise_generator.py for full documentation
# and literature references.
#
# Hardware data sources (January 2026):
# - IBM Quantum Platform: https://quantum.cloud.ibm.com/computers
# - IBM Heron characterization: https://research.ibm.com/publications/noise-characterization-and-error-mitigation-on-ibm-heron-processors-part-1--1
# - Rigetti Ankaa-3: https://thequantuminsider.com/2024/12/23/rigetti-computing-reports-84-qubit-ankaa-3-system-achieves-99-5-median-two-qubit-gate-fidelity-milestone/

NOISE_CONFIGS = [
    # =========================================================================
    # Ideal (baseline)
    # =========================================================================
    NoiseConfig(
        noise_id="IDEAL",
        noise_type="none",
        parameters={},
        description="Ideal simulation (no noise)",
    ),

    # =========================================================================
    # Generic Depolarizing Noise (for parameter sweeps)
    # =========================================================================
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

    # =========================================================================
    # Hardware-Specific: IBM Heron Processors
    # =========================================================================
    # Data from quantum.cloud.ibm.com/computers (Jan 2026)
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

    # =========================================================================
    # Hardware-Specific: Rigetti Ankaa Processors
    # =========================================================================
    # Data from Quantum Insider (Dec 2024) and qcs.rigetti.com
    NoiseConfig(
        noise_id="RIGETTI-ANKAA3",
        noise_type="combined",
        parameters={"p1": 0.002, "p2": 0.005, "p_readout": 0.02},
        description="Rigetti Ankaa-3 (84q): 99.5% 2Q fidelity (0.5% error)",
    ),

    # =========================================================================
    # Other Noise Types
    # =========================================================================
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
# All Configurations
# =============================================================================

ALL_EXPERIMENT_CONFIGS = (
    SCALABILITY_ROUNDTRIP_CONFIGS
    + SCALABILITY_PERIOD_CONFIGS
    + PERIOD_VARIATION_CONFIGS
    + INPUT_VARIATION_CONFIGS
)

# Separate configs by execution target
SIMULATOR_CONFIGS = [c for c in ALL_EXPERIMENT_CONFIGS if not c.hardware_only]
HARDWARE_ONLY_CONFIGS = [c for c in ALL_EXPERIMENT_CONFIGS if c.hardware_only]

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


def get_configs_by_type(experiment_type: str, include_hardware_only: bool = True) -> List[QFTExperimentConfig]:
    """Get all configurations of a given type."""
    configs = [c for c in ALL_EXPERIMENT_CONFIGS if c.experiment_type == experiment_type]
    if not include_hardware_only:
        configs = [c for c in configs if not c.hardware_only]
    return configs


def get_simulator_configs() -> List[QFTExperimentConfig]:
    """Get all configurations suitable for simulator execution (excludes hardware_only)."""
    return SIMULATOR_CONFIGS


def get_simulator_config_ids() -> List[str]:
    """Get config IDs suitable for simulator execution."""
    return [c.config_id for c in SIMULATOR_CONFIGS]


def get_hardware_only_configs() -> List[QFTExperimentConfig]:
    """Get configurations that should only run on real hardware."""
    return HARDWARE_ONLY_CONFIGS


def list_all_configs() -> None:
    """Print all available configurations."""
    print("QFT Experiment Configurations")
    print("=" * 60)

    for exp_type in ["scalability_roundtrip", "scalability_period", "period_variation", "input_variation"]:
        configs = get_configs_by_type(exp_type)
        print(f"\n{exp_type.upper()} ({len(configs)} configs):")
        for c in configs:
            hw_tag = " [HARDWARE ONLY]" if c.hardware_only else ""
            if c.test_mode == "roundtrip":
                print(f"  {c.config_id}: {c.num_qubits}q, input={c.input_state}{hw_tag}")
            else:
                print(f"  {c.config_id}: {c.num_qubits}q, period={c.period}{hw_tag}")

    print(f"\nSUMMARY:")
    print(f"  Total configs: {len(ALL_EXPERIMENT_CONFIGS)}")
    print(f"  Simulator-friendly: {len(SIMULATOR_CONFIGS)}")
    print(f"  Hardware-only: {len(HARDWARE_ONLY_CONFIGS)}")

    print(f"\nNOISE MODELS ({len(NOISE_CONFIGS)}):")
    for n in NOISE_CONFIGS:
        print(f"  {n.noise_id}: {n.description}")
