"""
Types for QWARD metrics.
"""

from enum import Enum


class MetricsId(Enum):
    """
    Enum for the different types of metrics available.
    """

    QISKIT = "QISKIT"
    COMPLEXITY = "COMPLEXITY"
    CIRCUIT_PERFORMANCE = "CIRCUIT_PERFORMANCE"
    ELEMENT = ("ELEMENT",)
    STRUCTURAL = "STRUCTURAL"
    BEHAVIORAL = "BEHAVIORAL"
    QUANTUM_SPECIFIC = "QUANTUM_SPECIFIC"
    FIDELITY = "FIDELITY"
    ESTIMATOR = "ESTIMATOR"
    BACKEND_CALIBRATION = "BACKEND_CALIBRATION"
    GATE_ERROR_CHARACTERIZATION = "GATE_ERROR_CHARACTERIZATION"


class MetricsType(Enum):
    """
    Enum for the different types of metrics available.
    """

    PRE_RUNTIME = "PRE_RUNTIME"
    POST_RUNTIME = "POST_RUNTIME"
    BACKEND_CONTEXT = "BACKEND_CONTEXT"
