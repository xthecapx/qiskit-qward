"""
Quantum algorithms module for qWard.

This module contains various quantum algorithm implementations and utilities.
"""

from .executor import QuantumCircuitExecutor, IBMJobResult, IBMBatchResult
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
from .experiment_analysis import (
    compute_descriptive_stats,
    NormalityTestResult,
    test_shapiro_wilk,
    test_dagostino_pearson,
    test_anderson_darling,
    test_kolmogorov_smirnov,
    test_normality,
    analyze_noise_impact,
    characterize_distribution,
    compare_noise_models,
    analyze_config_results_base,
    load_latest_batch_files,
    load_batch_results,
    extract_success_rates,
    build_results_by_config,
    build_noise_means,
    generate_campaign_report,
)
from .matrix_product_verification import (
    VerificationMethod,
    VerificationResult,
    MatrixProductVerificationBase,
    QuantumFreivaldsVerification,
    BuhrmanSpalekVerification,
    MatrixProductVerification,
)

__all__ = [
    # Executor
    "QuantumCircuitExecutor",
    "IBMJobResult",
    "IBMBatchResult",
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
    # Experiment Analysis
    "compute_descriptive_stats",
    "NormalityTestResult",
    "test_shapiro_wilk",
    "test_dagostino_pearson",
    "test_anderson_darling",
    "test_kolmogorov_smirnov",
    "test_normality",
    "analyze_noise_impact",
    "characterize_distribution",
    "compare_noise_models",
    "analyze_config_results_base",
    "load_latest_batch_files",
    "load_batch_results",
    "extract_success_rates",
    "build_results_by_config",
    "build_noise_means",
    "generate_campaign_report",
    # Matrix Product Verification
    "VerificationMethod",
    "VerificationResult",
    "MatrixProductVerificationBase",
    "QuantumFreivaldsVerification",
    "BuhrmanSpalekVerification",
    "MatrixProductVerification",
]
