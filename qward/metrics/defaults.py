"""Default metric calculators for QWARD."""

from typing import List, Type

from qward.metrics.base_metric import MetricCalculator
from qward.metrics.qiskit_metrics import QiskitMetrics
from qward.metrics.complexity_metrics import ComplexityMetrics
from qward.metrics.success_rate import SuccessRate


def get_default_metrics_strategies() -> List[Type[MetricCalculator]]:
    """Get the default metric calculator classes to use.

    Returns:
        List[Type[MetricCalculator]]: List of default metric calculator classes
    """
    return [QiskitMetrics, ComplexityMetrics, SuccessRate]
