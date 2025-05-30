"""Default metric strategies for QWARD."""

from typing import List, Type

from qward.metrics.base_metric import MetricCalculator
from qward.metrics.qiskit_metrics import QiskitMetrics
from qward.metrics.complexity_metrics import ComplexityMetrics
from qward.metrics.circuit_performance import CircuitPerformance


def get_default_strategies() -> List[Type[MetricCalculator]]:
    """
    Get the default list of metric strategies.

    Returns:
        List[Type[MetricCalculator]]: List of default metric strategy classes
    """
    return [QiskitMetrics, ComplexityMetrics, CircuitPerformance]
