"""
Shared utilities for experiment runners.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from qward import Scanner
from qward.metrics import (
    QiskitMetrics,
    ComplexityMetrics,
    StructuralMetrics,
    QuantumSpecificMetrics,
)


DEFAULT_METRIC_STRATEGIES = [
    QiskitMetrics,
    ComplexityMetrics,
    StructuralMetrics,
    QuantumSpecificMetrics,
]


def calculate_qward_metrics(circuit, strategies: Optional[List[Type]] = None) -> Dict[str, Any]:
    """
    Calculate pre-runtime QWARD metrics for a circuit using Scanner.

    Args:
        circuit: The quantum circuit to analyze
        strategies: Optional list of metric strategy classes

    Returns:
        Dictionary with all QWARD metrics (converted from DataFrames for JSON serialization)
        Returns error dict if metrics calculation fails.
    """
    try:
        scanner = Scanner(circuit=circuit, strategies=strategies or DEFAULT_METRIC_STRATEGIES)
        metrics_dict = scanner.calculate_metrics()
        return serialize_metrics_dict(metrics_dict)
    except Exception as e:
        print(f"    Warning: QWARD metrics failed: {e}")
        return {"error": str(e)}


def serialize_value(value: Any) -> Any:
    """Convert a value to JSON-serializable format."""
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def serialize_metrics_dict(metrics_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Convert QWARD metrics DataFrames to JSON-serializable dictionaries."""
    result = {}
    for metric_name, df in metrics_dict.items():
        if df is not None and not df.empty:
            row = df.iloc[0]
            result[metric_name] = {col: serialize_value(val) for col, val in row.items()}
    return result


@dataclass
class ExperimentDefaults:
    """Default experiment parameters."""

    shots: int = 1024
    num_runs: int = 10
    optimization_level: int = 0


DEFAULT_EXPERIMENT_PARAMS = ExperimentDefaults()


__all__ = [
    "calculate_qward_metrics",
    "serialize_value",
    "serialize_metrics_dict",
    "ExperimentDefaults",
    "DEFAULT_EXPERIMENT_PARAMS",
    "DEFAULT_METRIC_STRATEGIES",
]
