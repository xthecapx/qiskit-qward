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


class MetricsType(Enum):
    """
    Enum for the different types of metrics available.
    """

    PRE_RUNTIME = "PRE_RUNTIME"
    POST_RUNTIME = "POST_RUNTIME"
