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
    get_simulator_configs,
    get_simulator_config_ids,
    get_hardware_only_configs,
    list_all_configs,
    ALL_EXPERIMENT_CONFIGS,
    SIMULATOR_CONFIGS,
    HARDWARE_ONLY_CONFIGS,
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
    save_batch_result,
    load_batch_result,
    save_campaign_summary,
    aggregate_session_results,
    test_single_config,
    test_pilot_study,
    test_roundtrip_base_case,
    test_period_detection_base_case,
    calculate_qward_metrics,
)

from .qft_statistical_analysis import (
    # Descriptive statistics
    compute_descriptive_stats,
    # Normality tests
    NormalityTestResult,
    test_shapiro_wilk,
    test_dagostino_pearson,
    test_anderson_darling,
    test_kolmogorov_smirnov,
    test_normality,
    # Noise impact analysis
    analyze_noise_impact,
    # Distribution characterization
    characterize_distribution,
    # Multi-noise comparison
    compare_noise_models,
    # QFT-specific analysis
    analyze_scalability,
    analyze_period_impact,
    compare_test_modes,
    # Config analysis
    QFTConfigAnalysis,
    analyze_qft_config_results,
    # Utility functions
    print_qft_analysis_summary,
    print_qft_comparison_table,
    generate_qft_statistical_report,
)

__all__ = [
    # Configurations
    "QFTExperimentConfig",
    "NoiseConfig",
    "get_config",
    "get_noise_config",
    "get_configs_by_type",
    "get_simulator_configs",
    "get_simulator_config_ids",
    "get_hardware_only_configs",
    "list_all_configs",
    "ALL_EXPERIMENT_CONFIGS",
    "SIMULATOR_CONFIGS",
    "HARDWARE_ONLY_CONFIGS",
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
    "save_batch_result",
    "load_batch_result",
    "save_campaign_summary",
    "aggregate_session_results",
    "test_single_config",
    "test_pilot_study",
    "test_roundtrip_base_case",
    "test_period_detection_base_case",
    "calculate_qward_metrics",
    # Statistical analysis
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
    "analyze_scalability",
    "analyze_period_impact",
    "compare_test_modes",
    "QFTConfigAnalysis",
    "analyze_qft_config_results",
    "print_qft_analysis_summary",
    "print_qft_comparison_table",
    "generate_qft_statistical_report",
]
