"""Default metrics for QWARD."""

from typing import List, Type

from qward.metrics.base_metric import Metric
from qward.metrics.qiskit_metrics import QiskitMetrics
from qward.metrics.complexity_metrics import ComplexityMetrics
from qward.metrics.success_rate import SuccessRate


def get_default_metrics() -> List[Type[Metric]]:
    """Get the default metric classes to use.

    Returns:
        List[Type[Metric]]: List of default metric classes
    """
    return [QiskitMetrics, ComplexityMetrics, SuccessRate]
