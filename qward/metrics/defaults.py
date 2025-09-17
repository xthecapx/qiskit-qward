"""Default metric strategies for QWARD."""

from typing import List, Type

from qward.metrics.base_metric import MetricCalculator
from qward.metrics.qiskit_metrics import QiskitMetrics
from qward.metrics.complexity_metrics import ComplexityMetrics
from qward.metrics.loc_metrics import LocMetrics


def get_default_strategies() -> List[Type[MetricCalculator]]:
    """
    Get the default list of metric strategies.

    Note: CircuitPerformance is not included in defaults since it requires
    job execution results and should be added manually when needed.

    Returns:
        List[Type[MetricCalculator]]: List of default metric strategy classes
    """
    return [QiskitMetrics, ComplexityMetrics]
