"""
Base metric calculator class for QWARD.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union, TYPE_CHECKING

from qiskit import QuantumCircuit

if TYPE_CHECKING:
    from qward.metrics.types import MetricsType, MetricsId


class MetricCalculator(ABC):
    """
    Base class for all metric calculators in QWARD.

    This class defines the interface that all metric calculators must implement,
    providing methods for metric calculation and type identification.
    """

    def __init__(self, circuit: QuantumCircuit):
        """
        Initialize a MetricCalculator object.

        Args:
            circuit: The quantum circuit to analyze
        """
        self._circuit = circuit
        self._metric_type = self._get_metric_type()
        self._id = self._get_metric_id()

    @property
    def metric_type(self) -> "MetricsType":
        """
        Get the type of this metric.

        Returns:
            MetricsType: The type of this metric
        """
        return self._metric_type

    @property
    def id(self) -> "MetricsId":
        """
        Get the ID of the metric.

        Returns:
            MetricsId: The ID of this metric
        """
        return self._id

    @property
    def name(self) -> str:
        """
        Get the name of the metric.

        Returns:
            str: The name of the metric class.
        """
        return self.__class__.__name__

    @property
    def circuit(self) -> QuantumCircuit:
        """
        Get the quantum circuit.

        Returns:
            QuantumCircuit: The quantum circuit
        """
        return self._circuit

    @abstractmethod
    def _get_metric_type(self) -> "MetricsType":
        """
        Get the type of this metric.

        Returns:
            MetricsType: The type of this metric
        """
        pass

    @abstractmethod
    def _get_metric_id(self) -> "MetricsId":
        """
        Get the ID of this metric.

        Returns:
            MetricsId: The ID of this metric
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Check if the metric is ready to be calculated.

        Returns:
            bool: True if the metric is ready to be calculated, False otherwise
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Union[Dict[str, Any], Any]:
        """
        Get the metrics for this circuit.

        Returns:
            Union[Dict[str, Any], Any]: Dictionary of metric names and values, or a schema object
        """
        pass
