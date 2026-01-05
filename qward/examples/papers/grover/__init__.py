"""
Grover's Algorithm Experiment Framework

This module provides tools for systematic evaluation of Grover's algorithm
performance under various conditions including different noise models,
circuit configurations, and success metrics.

Quick Start:
    from qward.examples.papers.grover import test_single_config, test_pilot_study
    
    # Test a single configuration
    result = test_single_config("S3-1", "IDEAL", num_runs=3)
    
    # Run a pilot study
    results = test_pilot_study()

Full Experiment:
    from qward.examples.papers.grover import run_experiment_campaign
    
    # Run full campaign
    results = run_experiment_campaign(
        config_ids=["S3-1", "S4-1"],
        noise_ids=["IDEAL", "DEP-MED"],
        num_runs=10,
    )
"""

from .grover_configs import (
    ExperimentConfig,
    NoiseConfig,
    get_config,
    get_noise_config,
    get_configs_by_type,
    list_all_configs,
    list_noise_configs,
    ALL_EXPERIMENT_CONFIGS,
    NOISE_CONFIGS,
    SCALABILITY_CONFIGS,
    MARKED_COUNT_CONFIGS,
    HAMMING_CONFIGS,
    SYMMETRY_CONFIGS,
)

from .grover_success_metrics import (
    success_rate,
    success_count,
    success_per_shot,
    job_success_threshold,
    job_success_statistical,
    job_success_quantum_advantage,
    batch_success_mean,
    batch_success_min,
    batch_success_median,
    batch_success_consistency,
    evaluate_job,
    evaluate_batch,
)

from .grover_statistical_analysis import (
    compute_descriptive_stats,
    test_normality,
    analyze_noise_impact,
    characterize_distribution,
    compare_noise_models,
    analyze_config_results,
    ConfigAnalysis,
)

from .grover_experiment import (
    run_single_experiment,
    run_batch,
    run_experiment_campaign,
    test_single_config,
    test_pilot_study,
    calculate_qward_metrics,
    ExperimentResult,
    BatchResult,
    SHOTS,
    NUM_RUNS,
)

__all__ = [
    # Configs
    "ExperimentConfig",
    "NoiseConfig", 
    "get_config",
    "get_noise_config",
    "get_configs_by_type",
    "list_all_configs",
    "list_noise_configs",
    "ALL_EXPERIMENT_CONFIGS",
    "NOISE_CONFIGS",
    "SCALABILITY_CONFIGS",
    "MARKED_COUNT_CONFIGS",
    "HAMMING_CONFIGS",
    "SYMMETRY_CONFIGS",
    
    # Success Metrics
    "success_rate",
    "success_count",
    "success_per_shot",
    "job_success_threshold",
    "job_success_statistical",
    "job_success_quantum_advantage",
    "batch_success_mean",
    "batch_success_min",
    "batch_success_median",
    "batch_success_consistency",
    "evaluate_job",
    "evaluate_batch",
    
    # Statistical Analysis
    "compute_descriptive_stats",
    "test_normality",
    "analyze_noise_impact",
    "characterize_distribution",
    "compare_noise_models",
    "analyze_config_results",
    "ConfigAnalysis",
    
    # Experiment Runner
    "run_single_experiment",
    "run_batch",
    "run_experiment_campaign",
    "test_single_config",
    "test_pilot_study",
    "calculate_qward_metrics",
    "ExperimentResult",
    "BatchResult",
    "SHOTS",
    "NUM_RUNS",
]
