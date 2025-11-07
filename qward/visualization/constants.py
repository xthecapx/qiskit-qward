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
    ELEMENT = "ElementMetrics"
    STRUCTURAL = "StructuralMetrics"
    BEHAVIORAL = "BehavioralMetrics"
    QUANTUM_SPECIFIC = "QuantumSpecificMetrics"


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

    class Element:
        """Plot constants for ElementMetrics visualizations."""

        PAULI_GATES = "pauli_gates"
        SINGLE_QUBIT_BREAKDOWN = "single_qubit_breakdown"
        CONTROLLED_GATES = "controlled_gates"
        ORACLE_USAGE = "oracle_usage"
        ORACLE_RATIOS = "oracle_ratios"
        CNOT_TOFFOLI_STATS = "cnot_toffoli_stats"
        MEASUREMENT_ANCILLA = "measurement_ancilla"
        SUMMARY = "summary"

    class Structural:
        """Plot constants for StructuralMetrics visualizations."""

        LOC_BREAKDOWN = "loc_breakdown"
        HALSTEAD_BASIC = "halstead_basic"
        HALSTEAD_DERIVED = "halstead_derived"
        STRUCTURE_DIMENSIONS = "structure_dimensions"
        DENSITY_METRICS = "density_metrics"
        SUMMARY = "summary"

    class Behavioral:
        """Plot constants for BehavioralMetrics visualizations."""

        NORMALIZED_DEPTH = "normalized_depth"
        PROGRAM_COMMUNICATION = "program_communication"
        CRITICAL_DEPTH = "critical_depth"
        MEASUREMENT_LIVENESS = "measurement_liveness"
        PARALLELISM = "parallelism"
        BEHAVIORAL_RADAR = "behavioral_radar"
        SUMMARY = "summary"

    class QuantumSpecific:
        """Plot constants for QuantumSpecificMetrics visualizations."""

        ALL_METRICS_BAR = "all_metrics_bar"
        QUANTUM_RADAR = "quantum_radar"
        SUMMARY = "summary"
