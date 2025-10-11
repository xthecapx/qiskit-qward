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
    LOC = "LOC"
    QUANTUM_HALSTEAD = "QUANTUM_HALSTEAD"
    ELEMENT = "ELEMENT",
    STRUCTURAL = "STRUCTURAL"
    BEHAVIORAL = "BEHAVIORAL"


class MetricsType(Enum):
    """
    Enum for the different types of metrics available.
    """

    PRE_RUNTIME = "PRE_RUNTIME"
    POST_RUNTIME = "POST_RUNTIME"
