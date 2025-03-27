"""
Qward - A framework for analyzing and validating quantum code execution quality.
"""

from importlib_metadata import version as metadata_version, PackageNotFoundError

from .validators.base_validator import BaseValidator
from .validators.teleportation_validator import TeleportationValidator
from .validators.flip_coin_validator import FlipCoinValidator
from .analysis.analysis import Analysis
from .analysis.success_rate import SuccessRate

try:
    __version__ = metadata_version("qiskit-qward")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "0.1.0"

__all__ = [
    "BaseValidator",
    "TeleportationValidator",
    "FlipCoinValidator",
    "Analysis",
    "SuccessRate",
]
