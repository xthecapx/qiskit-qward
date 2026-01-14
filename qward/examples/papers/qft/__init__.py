"""
QFT Experiment Module

This module contains the experimental framework for evaluating
Quantum Fourier Transform performance using the QWARD library.
"""

from .qft_configs import (
    QFTExperimentConfig,
    NoiseConfig,
    get_config,
    get_noise_config,
    get_configs_by_type,
    list_all_configs,
    ALL_EXPERIMENT_CONFIGS,
    NOISE_CONFIGS,
    CONFIGS_BY_ID,
    NOISE_BY_ID,
    SCALABILITY_ROUNDTRIP_CONFIGS,
    SCALABILITY_PERIOD_CONFIGS,
    PERIOD_VARIATION_CONFIGS,
    INPUT_VARIATION_CONFIGS,
)

from .qft_experiment import (
    ExperimentResult,
    BatchResult,
    run_single_experiment,
    run_batch,
    run_experiment_campaign,
    test_single_config,
    test_pilot_study,
    test_roundtrip_base_case,
    test_period_detection_base_case,
    calculate_qward_metrics,
)

__all__ = [
    # Configurations
    "QFTExperimentConfig",
    "NoiseConfig",
    "get_config",
    "get_noise_config",
    "get_configs_by_type",
    "list_all_configs",
    "ALL_EXPERIMENT_CONFIGS",
    "NOISE_CONFIGS",
    "CONFIGS_BY_ID",
    "NOISE_BY_ID",
    "SCALABILITY_ROUNDTRIP_CONFIGS",
    "SCALABILITY_PERIOD_CONFIGS",
    "PERIOD_VARIATION_CONFIGS",
    "INPUT_VARIATION_CONFIGS",
    # Experiment runners
    "ExperimentResult",
    "BatchResult",
    "run_single_experiment",
    "run_batch",
    "run_experiment_campaign",
    "test_single_config",
    "test_pilot_study",
    "test_roundtrip_base_case",
    "test_period_detection_base_case",
    "calculate_qward_metrics",
]
