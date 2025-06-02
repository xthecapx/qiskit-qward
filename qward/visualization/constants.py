"""
Constants for QWARD visualization system.

This module provides type-safe constants for metrics and plots to prevent typos
and enable IDE autocompletion. These constants can be shared with the metrics API
for consistency across the QWARD ecosystem.
"""


class Metrics:
    """Constants for metric names - shared with metrics API for consistency."""

    QISKIT = "QiskitMetrics"
    COMPLEXITY = "ComplexityMetrics"
    CIRCUIT_PERFORMANCE = "CircuitPerformance"


class Plots:
    """Constants for plot names organized by metric type."""

    class Qiskit:
        """Plot constants for QiskitMetrics visualizations."""

        CIRCUIT_STRUCTURE = "circuit_structure"
        GATE_DISTRIBUTION = "gate_distribution"
        INSTRUCTION_METRICS = "instruction_metrics"
        CIRCUIT_SUMMARY = "circuit_summary"

    class Complexity:
        """Plot constants for ComplexityMetrics visualizations."""

        GATE_BASED_METRICS = "gate_based_metrics"
        COMPLEXITY_RADAR = "complexity_radar"
        EFFICIENCY_METRICS = "efficiency_metrics"

    class CircuitPerformance:
        """Plot constants for CircuitPerformance visualizations."""

        SUCCESS_ERROR_COMPARISON = "success_error_comparison"
        FIDELITY_COMPARISON = "fidelity_comparison"
        SHOT_DISTRIBUTION = "shot_distribution"
        AGGREGATE_SUMMARY = "aggregate_summary"
