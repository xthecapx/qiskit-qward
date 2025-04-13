"""
Base metric class for QWARD.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING

from qiskit import QuantumCircuit

if TYPE_CHECKING:
    from qward.metrics.types import MetricsType, MetricsId


class Metric(ABC):
    """
    Base class for all metrics in QWARD.

    This class defines the interface that all metrics must implement,
    providing methods for metric calculation and type identification.
    """

    def __init__(self, circuit: QuantumCircuit):
        """
        Initialize a Metric object.

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
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the metrics for this circuit.

        Returns:
            Dict[str, Any]: Dictionary of metric names and values
        """
        pass
