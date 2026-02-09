"""Default metric strategies for QWARD."""

from typing import List, Type

from qward.metrics.base_metric import MetricCalculator
from qward.metrics.qiskit_metrics import QiskitMetrics
from qward.metrics.complexity_metrics import ComplexityMetrics


def get_default_strategies() -> List[Type[MetricCalculator]]:
    """
    Get the default list of metric strategies.

    Note: CircuitPerformance is not included in defaults since it requires
    job execution results and should be added manually when needed.

    Returns:
        List[Type[MetricCalculator]]: List of default metric strategy classes
    """
    return [QiskitMetrics, ComplexityMetrics]


def get_all_pre_runtime_strategies() -> List[Type[MetricCalculator]]:
    """
    Get all pre-runtime metric strategy classes.

    Returns:
        List[Type[MetricCalculator]]: List of all pre-runtime strategy classes
    """
    from qward.metrics.behavioral_metrics import BehavioralMetrics
    from qward.metrics.element_metrics import ElementMetrics
    from qward.metrics.quantum_specific_metrics import QuantumSpecificMetrics
    from qward.metrics.structural_metrics import StructuralMetrics

    return [
        QiskitMetrics,
        ComplexityMetrics,
        ElementMetrics,
        StructuralMetrics,
        BehavioralMetrics,
        QuantumSpecificMetrics,
    ]
