"""
Quantum algorithms module for qWard.

This module contains various quantum algorithm implementations and utilities.
"""

from .executor import QuantumCircuitExecutor
from .v_tp import (
    QuantumGate,
    BaseTeleportation,
    StandardTeleportationProtocol,
    VariationTeleportationProtocol,
    TeleportationCircuitGenerator,
)
from .grover import (
    Grover,
    GroverOracle,
    GroverCircuitGenerator,
)
from .qft import (
    QFT,
    QFTCircuitGenerator,
)
from .phase_estimation import (
    PhaseEstimation,
    PhaseEstimationCircuitGenerator,
)
from .noise_generator import (
    NoiseConfig,
    NoiseModelGenerator,
    PRESET_NOISE_CONFIGS,
    get_preset_noise_config,
    list_preset_noise_configs,
)
from .experiment_utils import (
    calculate_qward_metrics,
    serialize_value,
    serialize_metrics_dict,
    ExperimentDefaults,
    DEFAULT_EXPERIMENT_PARAMS,
)
from .experiment import (
    BaseExperimentResult,
    BaseBatchResult,
    BaseExperimentRunner,
    CampaignProgress,
)

__all__ = [
    # Executor
    "QuantumCircuitExecutor",
    # Teleportation
    "QuantumGate",
    "BaseTeleportation",
    "StandardTeleportationProtocol",
    "VariationTeleportationProtocol",
    "TeleportationCircuitGenerator",
    # Grover
    "Grover",
    "GroverOracle",
    "GroverCircuitGenerator",
    # QFT
    "QFT",
    "QFTCircuitGenerator",
    # Phase Estimation
    "PhaseEstimation",
    "PhaseEstimationCircuitGenerator",
    # Noise Generation
    "NoiseConfig",
    "NoiseModelGenerator",
    "PRESET_NOISE_CONFIGS",
    "get_preset_noise_config",
    "list_preset_noise_configs",
    # Experiment Utilities
    "calculate_qward_metrics",
    "serialize_value",
    "serialize_metrics_dict",
    "ExperimentDefaults",
    "DEFAULT_EXPERIMENT_PARAMS",
    # Experiment Framework
    "BaseExperimentResult",
    "BaseBatchResult",
    "BaseExperimentRunner",
    "CampaignProgress",
]
