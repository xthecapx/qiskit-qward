"""
Ansatz circuit builders for VQE eigensolver.

Provides parameterized quantum circuits (ansatze) for variational eigenvalue
estimation. The selection logic follows Phase 2 theoretical design, Section 4:

- 1-qubit systems: RY + RZ parameterization (2 parameters, universal)
- 2-qubit systems: EfficientSU2 with reps=2 (12 parameters, universal)

References:
    - Phase 2: phase2_theoretical_design.md, Section 4
    - Kandala et al., Nature 549, 242-246 (2017) (hardware-efficient ansatze)
"""

import math

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import efficient_su2


def build_ansatz(
    num_qubits: int,
    reps: int = 2,
) -> QuantumCircuit:
    """Build a hardware-efficient ansatz for VQE.

    Args:
        num_qubits: Number of qubits (1 or 2 for this project).
        reps: Number of repetition layers for multi-qubit ansatze.

    Returns:
        Parameterized QuantumCircuit suitable for VQE optimization.
    """
    if num_qubits == 1:
        return _build_single_qubit_ansatz()
    return _build_multi_qubit_ansatz(num_qubits, reps)


def _build_single_qubit_ansatz() -> QuantumCircuit:
    """Build a universal single-qubit ansatz: RY(t0) RZ(t1).

    This parameterizes the full Bloch sphere, reaching any single-qubit
    state (equivalent to U3 up to global phase).

    Returns:
        QuantumCircuit with 2 parameters.
    """
    params = ParameterVector("t", 2)
    qc = QuantumCircuit(1)
    qc.ry(params[0], 0)
    qc.rz(params[1], 0)
    return qc


def _build_multi_qubit_ansatz(
    num_qubits: int,
    reps: int = 2,
) -> QuantumCircuit:
    """Build a multi-qubit EfficientSU2 ansatz.

    Uses Qiskit's EfficientSU2 library circuit with RY and RZ rotation
    gates and linear CX entanglement.

    Args:
        num_qubits: Number of qubits.
        reps: Number of entanglement layers.

    Returns:
        QuantumCircuit with 4 * (reps + 1) * num_qubits parameters.
    """
    return efficient_su2(
        num_qubits=num_qubits,
        reps=reps,
        entanglement="linear",
    )
