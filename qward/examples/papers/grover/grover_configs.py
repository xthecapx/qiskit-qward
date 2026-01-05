"""
Grover Experiment Circuit Configurations

This module defines all circuit configurations for the Grover experiment,
including scalability, marked state, Hamming weight, and symmetry studies.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import math


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    
    config_id: str
    experiment_type: str  # "scalability", "marked_count", "hamming", "symmetry"
    num_qubits: int
    marked_states: List[str]
    description: str = ""
    
    @property
    def num_marked(self) -> int:
        """Number of marked states."""
        return len(self.marked_states)
    
    @property
    def search_space(self) -> int:
        """Total search space size (2^n)."""
        return 2 ** self.num_qubits
    
    @property
    def theoretical_iterations(self) -> int:
        """Optimal number of Grover iterations."""
        return math.floor(
            math.pi / (4 * math.asin(math.sqrt(self.num_marked / self.search_space)))
        )
    
    @property
    def theoretical_success(self) -> float:
        """Theoretical success probability after optimal iterations."""
        theta = math.asin(math.sqrt(self.num_marked / self.search_space))
        return math.sin((2 * self.theoretical_iterations + 1) * theta) ** 2
    
    @property
    def classical_random_prob(self) -> float:
        """Classical random search success probability."""
        return self.num_marked / self.search_space
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for data storage."""
        return {
            "config_id": self.config_id,
            "experiment_type": self.experiment_type,
            "num_qubits": self.num_qubits,
            "marked_states": self.marked_states,
            "num_marked": self.num_marked,
            "search_space": self.search_space,
            "theoretical_iterations": self.theoretical_iterations,
            "theoretical_success": self.theoretical_success,
            "classical_random_prob": self.classical_random_prob,
            "description": self.description,
        }


# =============================================================================
# Experiment 1: Scalability Study (S)
# =============================================================================
# Testing how success rate degrades as qubit count increases

SCALABILITY_CONFIGS = [
    ExperimentConfig(
        config_id="S2-1",
        experiment_type="scalability",
        num_qubits=2,
        marked_states=["01"],
        description="2 qubits, single marked state"
    ),
    ExperimentConfig(
        config_id="S3-1",
        experiment_type="scalability",
        num_qubits=3,
        marked_states=["011"],
        description="3 qubits, single marked state"
    ),
    ExperimentConfig(
        config_id="S4-1",
        experiment_type="scalability",
        num_qubits=4,
        marked_states=["0110"],
        description="4 qubits, single marked state"
    ),
    ExperimentConfig(
        config_id="S5-1",
        experiment_type="scalability",
        num_qubits=5,
        marked_states=["01100"],
        description="5 qubits, single marked state"
    ),
    ExperimentConfig(
        config_id="S6-1",
        experiment_type="scalability",
        num_qubits=6,
        marked_states=["011001"],
        description="6 qubits, single marked state"
    ),
    ExperimentConfig(
        config_id="S7-1",
        experiment_type="scalability",
        num_qubits=7,
        marked_states=["0110011"],
        description="7 qubits, single marked state"
    ),
    ExperimentConfig(
        config_id="S8-1",
        experiment_type="scalability",
        num_qubits=8,
        marked_states=["01100110"],
        description="8 qubits, single marked state"
    ),
]


# =============================================================================
# Experiment 2A: Number of Marked States (M)
# =============================================================================
# Testing how the number of marked states affects performance

MARKED_COUNT_CONFIGS = [
    # 3 qubits
    ExperimentConfig(
        config_id="M3-1",
        experiment_type="marked_count",
        num_qubits=3,
        marked_states=["000"],
        description="3 qubits, 1 marked state"
    ),
    ExperimentConfig(
        config_id="M3-2",
        experiment_type="marked_count",
        num_qubits=3,
        marked_states=["000", "111"],
        description="3 qubits, 2 marked states (extremes)"
    ),
    ExperimentConfig(
        config_id="M3-4",
        experiment_type="marked_count",
        num_qubits=3,
        marked_states=["000", "001", "110", "111"],
        description="3 qubits, 4 marked states"
    ),
    # 4 qubits
    ExperimentConfig(
        config_id="M4-1",
        experiment_type="marked_count",
        num_qubits=4,
        marked_states=["0000"],
        description="4 qubits, 1 marked state"
    ),
    ExperimentConfig(
        config_id="M4-2",
        experiment_type="marked_count",
        num_qubits=4,
        marked_states=["0000", "1111"],
        description="4 qubits, 2 marked states (extremes)"
    ),
    ExperimentConfig(
        config_id="M4-4",
        experiment_type="marked_count",
        num_qubits=4,
        marked_states=["0000", "0011", "1100", "1111"],
        description="4 qubits, 4 marked states"
    ),
]


# =============================================================================
# Experiment 2B: Hamming Weight Study (H)
# =============================================================================
# Testing whether the Hamming weight of marked states affects performance

HAMMING_CONFIGS = [
    # 3 qubits - all Hamming weights
    ExperimentConfig(
        config_id="H3-0",
        experiment_type="hamming",
        num_qubits=3,
        marked_states=["000"],
        description="3 qubits, Hamming weight 0 (all zeros)"
    ),
    ExperimentConfig(
        config_id="H3-1",
        experiment_type="hamming",
        num_qubits=3,
        marked_states=["001"],
        description="3 qubits, Hamming weight 1"
    ),
    ExperimentConfig(
        config_id="H3-2",
        experiment_type="hamming",
        num_qubits=3,
        marked_states=["011"],
        description="3 qubits, Hamming weight 2"
    ),
    ExperimentConfig(
        config_id="H3-3",
        experiment_type="hamming",
        num_qubits=3,
        marked_states=["111"],
        description="3 qubits, Hamming weight 3 (all ones)"
    ),
    # 4 qubits - selected Hamming weights
    ExperimentConfig(
        config_id="H4-0",
        experiment_type="hamming",
        num_qubits=4,
        marked_states=["0000"],
        description="4 qubits, Hamming weight 0 (all zeros)"
    ),
    ExperimentConfig(
        config_id="H4-2",
        experiment_type="hamming",
        num_qubits=4,
        marked_states=["0011"],
        description="4 qubits, Hamming weight 2 (balanced)"
    ),
    ExperimentConfig(
        config_id="H4-4",
        experiment_type="hamming",
        num_qubits=4,
        marked_states=["1111"],
        description="4 qubits, Hamming weight 4 (all ones)"
    ),
]


# =============================================================================
# Experiment 2C: Symmetric vs Asymmetric Marked States (SYM/ASYM)
# =============================================================================
# Testing whether symmetric patterns affect interference/performance

SYMMETRY_CONFIGS = [
    ExperimentConfig(
        config_id="SYM-1",
        experiment_type="symmetry",
        num_qubits=3,
        marked_states=["000", "111"],
        description="Symmetric: complement pair (extremes)"
    ),
    ExperimentConfig(
        config_id="SYM-2",
        experiment_type="symmetry",
        num_qubits=3,
        marked_states=["001", "110"],
        description="Symmetric: complement pair (1-bit from extremes)"
    ),
    ExperimentConfig(
        config_id="ASYM-1",
        experiment_type="symmetry",
        num_qubits=3,
        marked_states=["000", "001"],
        description="Asymmetric: adjacent states (1-bit difference)"
    ),
    ExperimentConfig(
        config_id="ASYM-2",
        experiment_type="symmetry",
        num_qubits=3,
        marked_states=["000", "011"],
        description="Asymmetric: 2-bit difference"
    ),
]


# =============================================================================
# Noise Model Configurations
# =============================================================================

@dataclass
class NoiseConfig:
    """Configuration for a noise model."""
    
    noise_id: str
    noise_type: str  # "none", "depolarizing", "pauli", "thermal", "readout", "combined"
    parameters: Dict[str, float] = field(default_factory=dict)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for data storage."""
        return {
            "noise_id": self.noise_id,
            "noise_type": self.noise_type,
            "parameters": self.parameters,
            "description": self.description,
        }


NOISE_CONFIGS = [
    NoiseConfig(
        noise_id="IDEAL",
        noise_type="none",
        parameters={},
        description="Perfect execution baseline"
    ),
    NoiseConfig(
        noise_id="DEP-LOW",
        noise_type="depolarizing",
        parameters={"p1": 0.001, "p2": 0.01},
        description="Light depolarizing noise"
    ),
    NoiseConfig(
        noise_id="DEP-MED",
        noise_type="depolarizing",
        parameters={"p1": 0.01, "p2": 0.05},
        description="Medium depolarizing noise"
    ),
    NoiseConfig(
        noise_id="DEP-HIGH",
        noise_type="depolarizing",
        parameters={"p1": 0.05, "p2": 0.10},
        description="Heavy depolarizing noise"
    ),
    NoiseConfig(
        noise_id="PAULI",
        noise_type="pauli",
        parameters={"pX": 0.01, "pY": 0.01, "pZ": 0.01},
        description="Structured Pauli errors"
    ),
    NoiseConfig(
        noise_id="THERMAL",
        noise_type="thermal",
        parameters={"T1": 50e-6, "T2": 70e-6},  # in seconds
        description="T1/T2 relaxation"
    ),
    NoiseConfig(
        noise_id="READOUT",
        noise_type="readout",
        parameters={"p01": 0.02, "p10": 0.02},
        description="Measurement errors only"
    ),
    NoiseConfig(
        noise_id="COMBINED",
        noise_type="combined",
        parameters={"p1": 0.01, "p2": 0.05, "p_readout": 0.02},
        description="Realistic combined noise (DEP-MED + READOUT)"
    ),
]


# =============================================================================
# Configuration Registry
# =============================================================================

ALL_EXPERIMENT_CONFIGS = (
    SCALABILITY_CONFIGS + 
    MARKED_COUNT_CONFIGS + 
    HAMMING_CONFIGS + 
    SYMMETRY_CONFIGS
)

# Create lookup dictionaries
CONFIGS_BY_ID = {config.config_id: config for config in ALL_EXPERIMENT_CONFIGS}
NOISE_BY_ID = {noise.noise_id: noise for noise in NOISE_CONFIGS}

CONFIGS_BY_TYPE = {
    "scalability": SCALABILITY_CONFIGS,
    "marked_count": MARKED_COUNT_CONFIGS,
    "hamming": HAMMING_CONFIGS,
    "symmetry": SYMMETRY_CONFIGS,
}


def get_config(config_id: str) -> ExperimentConfig:
    """Get an experiment configuration by ID."""
    if config_id not in CONFIGS_BY_ID:
        raise ValueError(f"Unknown config ID: {config_id}. Available: {list(CONFIGS_BY_ID.keys())}")
    return CONFIGS_BY_ID[config_id]


def get_noise_config(noise_id: str) -> NoiseConfig:
    """Get a noise configuration by ID."""
    if noise_id not in NOISE_BY_ID:
        raise ValueError(f"Unknown noise ID: {noise_id}. Available: {list(NOISE_BY_ID.keys())}")
    return NOISE_BY_ID[noise_id]


def get_configs_by_type(experiment_type: str) -> List[ExperimentConfig]:
    """Get all configurations for a specific experiment type."""
    if experiment_type not in CONFIGS_BY_TYPE:
        raise ValueError(f"Unknown experiment type: {experiment_type}. Available: {list(CONFIGS_BY_TYPE.keys())}")
    return CONFIGS_BY_TYPE[experiment_type]


def list_all_configs() -> None:
    """Print summary of all configurations."""
    print("=" * 80)
    print("GROVER EXPERIMENT CONFIGURATIONS")
    print("=" * 80)
    
    for exp_type, configs in CONFIGS_BY_TYPE.items():
        print(f"\n{exp_type.upper()} ({len(configs)} configs)")
        print("-" * 40)
        for config in configs:
            print(f"  {config.config_id}: {config.num_qubits}q, {config.num_marked} marked, "
                  f"P(success)={config.theoretical_success:.3f}")
    
    print(f"\n{'=' * 80}")
    print(f"TOTAL: {len(ALL_EXPERIMENT_CONFIGS)} experiment configurations")
    print(f"NOISE MODELS: {len(NOISE_CONFIGS)}")
    print("=" * 80)


def list_noise_configs() -> None:
    """Print summary of all noise configurations."""
    print("=" * 60)
    print("NOISE MODEL CONFIGURATIONS")
    print("=" * 60)
    
    for noise in NOISE_CONFIGS:
        print(f"\n{noise.noise_id}: {noise.description}")
        if noise.parameters:
            print(f"  Parameters: {noise.parameters}")


# =============================================================================
# Quick validation
# =============================================================================

if __name__ == "__main__":
    list_all_configs()
    print()
    list_noise_configs()

