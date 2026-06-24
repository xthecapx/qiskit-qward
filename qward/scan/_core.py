"""Core scan functions — no IBM credentials required."""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

from qward.metrics.fidelity_metrics import FidelityMetrics
from qward.scanner import Scanner

MAX_QUBITS_FOR_UNITARY = 20


def scan_pre(
    circuit: QuantumCircuit,
    *,
    include_quantum_specific: bool = True,
    max_qubits_for_unitary: int = MAX_QUBITS_FOR_UNITARY,
) -> Dict[str, pd.DataFrame]:
    """Compute all pre-runtime metrics from a circuit (no execution needed).

    Args:
        circuit: Quantum circuit to analyze.
        include_quantum_specific: Include QuantumSpecificMetrics (expensive for >20 qubits).
        max_qubits_for_unitary: Qubit limit for QuantumSpecificMetrics.

    Returns:
        Dict mapping metric names to DataFrames.
    """
    from qward.metrics import (
        BehavioralMetrics,
        ComplexityMetrics,
        ElementMetrics,
        QiskitMetrics,
        StructuralMetrics,
    )

    scanner = Scanner(circuit=circuit)
    scanner.add_strategy(QiskitMetrics(circuit))
    scanner.add_strategy(ComplexityMetrics(circuit))
    scanner.add_strategy(ElementMetrics(circuit))
    scanner.add_strategy(StructuralMetrics(circuit))
    scanner.add_strategy(BehavioralMetrics(circuit))

    if include_quantum_specific and circuit.num_qubits <= max_qubits_for_unitary:
        from qward.metrics import QuantumSpecificMetrics

        scanner.add_strategy(QuantumSpecificMetrics(circuit))

    return scanner.calculate_metrics()


def scan_post(
    circuit: QuantumCircuit,
    counts: Optional[Dict[str, int]] = None,
    *,
    expected_outcomes: Optional[List[str]] = None,
    target_histogram: Optional[Dict[str, float]] = None,
    target_state: Optional[str] = None,
    success_criteria: Optional[Callable[[str], bool]] = None,
    expectation_values: Optional[np.ndarray] = None,
    standard_deviations: Optional[np.ndarray] = None,
    ideal_expectation_values: Optional[np.ndarray] = None,
    observable_labels: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Compute post-runtime fidelity metrics — auto-detects primitive type.

    Provide **either** Sampler results (counts) or Estimator results
    (expectation_values). The function detects which primitive produced
    the data and computes appropriate metrics.

    Args:
        circuit: Quantum circuit that was executed.
        counts: Sampler measurement results as {bitstring: count}.
        expected_outcomes: Expected bitstrings for DSR/success_rate (Sampler).
        target_histogram: Ideal probability distribution for HF/TVD (Sampler).
        target_state: Shortcut — sets both expected_outcomes and target_histogram.
        success_criteria: Callable for custom success definition (Sampler).
        expectation_values: Estimator expectation values (numpy array).
        standard_deviations: Estimator standard deviations.
        ideal_expectation_values: Ideal values for fidelity (Estimator).
        observable_labels: Human-readable labels (Estimator).

    Returns:
        Dict with "FidelityMetrics" key mapping to DataFrame.
    """
    fm = FidelityMetrics(
        circuit,
        counts=counts,
        expected_outcomes=expected_outcomes,
        target_histogram=target_histogram,
        target_state=target_state,
        success_criteria=success_criteria,
        expectation_values=expectation_values,
        standard_deviations=standard_deviations,
        ideal_expectation_values=ideal_expectation_values,
        observable_labels=observable_labels,
    )
    scanner = Scanner(circuit=circuit)
    scanner.add_strategy(fm)
    return scanner.calculate_metrics()
