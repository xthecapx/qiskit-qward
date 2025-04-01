"""
Qward - A framework for analyzing and validating quantum code execution quality.
"""

from importlib_metadata import version as metadata_version, PackageNotFoundError

from .scanning_quantum_circuit import ScanningQuantumCircuit
from .analysis.analysis import Analysis
from .analysis.success_rate import SuccessRate

try:
    __version__ = metadata_version("qiskit_qward")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "0.0.0"

__all__ = [
    "ScanningQuantumCircuit",
    "Analysis",
    "SuccessRate",
]
