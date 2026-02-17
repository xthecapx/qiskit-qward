"""Shared helper functions for paper experiment runners."""

import math
import statistics
from typing import Any, Dict, List

from qiskit import QuantumCircuit

from qward import Scanner
from qward.metrics import ComplexityMetrics, QiskitMetrics


def calculate_qward_metrics(circuit: QuantumCircuit) -> Dict[str, Any]:
    """Calculate QWARD metrics for a circuit before execution."""
    try:
        scanner = Scanner(circuit=circuit)
        scanner.add_strategy(QiskitMetrics(circuit))
        scanner.add_strategy(ComplexityMetrics(circuit))

        metrics_dict = scanner.calculate_metrics()

        result: Dict[str, Any] = {}
        for metric_name, df in metrics_dict.items():
            if df is not None and hasattr(df, "empty") and not df.empty:
                row = df.iloc[0]
                result[metric_name] = {}
                for col in df.columns:
                    val = row[col]
                    if isinstance(val, (int, float, str, bool, type(None))):
                        result[metric_name][col] = val
                    elif hasattr(val, "item"):
                        result[metric_name][col] = val.item()
                    elif isinstance(val, (list, tuple)):
                        result[metric_name][col] = [str(v) for v in val]
                    else:
                        result[metric_name][col] = str(val)

        return result

    except Exception as exc:
        return {"error": str(exc)}


def calculate_statistical_analysis(
    success_rates: List[float], config_id: str, noise_model: str
) -> Dict[str, Any]:
    """Calculate descriptive statistics and simple normality checks."""
    n = len(success_rates)
    if n < 2:
        return {}

    mean = statistics.mean(success_rates)
    std = statistics.stdev(success_rates)

    se = std / math.sqrt(n)
    t_value = 1.96
    ci_lower = mean - t_value * se
    ci_upper = mean + t_value * se

    if std > 0:
        skewness = sum((x - mean) ** 3 for x in success_rates) / (n * std**3)
    else:
        skewness = 0.0

    if std > 0:
        kurtosis = sum((x - mean) ** 4 for x in success_rates) / (n * std**4) - 3
    else:
        kurtosis = 0.0

    is_normal = abs(skewness) < 2 and abs(kurtosis) < 7

    return {
        "config_id": config_id,
        "noise_model": noise_model,
        "num_runs": n,
        "mean": mean,
        "std": std,
        "median": statistics.median(success_rates),
        "min": min(success_rates),
        "max": max(success_rates),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "is_normal": is_normal,
    }
