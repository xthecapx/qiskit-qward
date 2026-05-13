"""
Coin-Toss (Ry rotation) Experiment Module.

A simple, scalable experiment that exercises a circuit of independent
``Ry`` rotations (one per qubit), modelling fair or biased coin tosses.

Use it to validate hardware sampling uniformity (fair mode) or to
measure how the QPU recovers a known biased product distribution.
"""

from .coin_toss_configs import (
    ALL_EXPERIMENT_CONFIGS,
    BIASED_CONFIGS,
    CONFIGS_BY_ID,
    CoinTossExperimentConfig,
    FAIR_CONFIGS,
    FAIR_THETA,
    HARDWARE_ONLY_CONFIGS,
    NOISE_BY_ID,
    NOISE_CONFIGS,
    NoiseConfig,
    SIMULATOR_CONFIGS,
    get_config,
    get_configs_by_type,
    get_hardware_only_configs,
    get_noise_config,
    get_simulator_config_ids,
    get_simulator_configs,
    list_all_configs,
    theta_for_p1,
)
from .coin_toss_aws import (
    CoinTossAWSExperiment,
    list_configs as list_configs_aws,
    run_coin_toss_on_aws,
)
from .coin_toss_ibm import (
    CoinTossIBMExperiment,
    REGION1_PRIORITY,
    list_configs as list_configs_ibm,
    run_coin_toss_on_ibm,
)

__all__ = [
    # Config dataclass + helpers
    "CoinTossExperimentConfig",
    "NoiseConfig",
    "FAIR_THETA",
    "theta_for_p1",
    "get_config",
    "get_noise_config",
    "get_configs_by_type",
    "get_simulator_configs",
    "get_simulator_config_ids",
    "get_hardware_only_configs",
    "list_all_configs",
    # Config collections
    "FAIR_CONFIGS",
    "BIASED_CONFIGS",
    "ALL_EXPERIMENT_CONFIGS",
    "SIMULATOR_CONFIGS",
    "HARDWARE_ONLY_CONFIGS",
    "NOISE_CONFIGS",
    "CONFIGS_BY_ID",
    "NOISE_BY_ID",
    # IBM runner
    "CoinTossIBMExperiment",
    "run_coin_toss_on_ibm",
    "list_configs_ibm",
    "REGION1_PRIORITY",
    # AWS runner
    "CoinTossAWSExperiment",
    "run_coin_toss_on_aws",
    "list_configs_aws",
]
