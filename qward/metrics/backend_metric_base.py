"""
Base class for backend-aware metric collectors in QWARD.

Unlike MetricCalculator (which operates on circuits only), BackendMetricCollector
operates on backend objects to extract hardware calibration and error data.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union


class BackendMetricCollector(ABC):
    """
    Base class for metrics that require a backend object.

    This is parallel to MetricCalculator but takes a backend (IBM BackendV2,
    AWS device, etc.) instead of a QuantumCircuit.
    """

    def __init__(self, backend):
        self._backend = backend

    @property
    def backend(self):
        return self._backend

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend provides required calibration data."""
        pass

    @abstractmethod
    def get_metrics(self) -> Union[Dict[str, Any], Any]:
        """Extract metrics from the backend. Returns dict or schema object."""
        pass
