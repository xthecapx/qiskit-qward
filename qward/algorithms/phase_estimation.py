"""
Quantum Phase Estimation (QPE) implementation for qWard.

This module provides Phase Estimation circuits based on Kitaev's algorithm (1995).
Given a unitary operator U and one of its eigenvectors |ψ⟩, the algorithm
estimates the phase φ where U|ψ⟩ = e^(2πiφ)|ψ⟩.

The algorithm uses:
1. m counting qubits for precision (determines accuracy of φ)
2. n qubits for the eigenvector |ψ⟩
3. Controlled-U operations with phase kickback
4. Inverse QFT to extract the phase
"""

import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFTGate
from qiskit.circuit import Gate

# Import display with fallback for non-notebook environments
try:
    from IPython.display import display
except ImportError:
    display = print


class PhaseEstimation:
    """Quantum Phase Estimation implementation.

    Given a unitary operator U and one of its eigenvectors |ψ⟩,
    estimates the phase φ where U|ψ⟩ = e^(2πiφ)|ψ⟩.

    Based on Kitaev's algorithm (1995) as described in BasicQuantumAlgorithms.tex.

    How it works:
    -------------
    1. Prepare counting qubits in superposition |+⟩^⊗m
    2. Apply controlled-U^(2^j) operations (phase kickback encodes φ)
    3. Apply inverse QFT to extract φ from the phase information
    4. Measure counting qubits to get binary approximation of φ

    The measured value m gives φ ≈ m/2^n where n is num_counting_qubits.
    More counting qubits = higher precision.

    Args:
        unitary: The unitary operator U (as a Gate or QuantumCircuit)
        num_counting_qubits: Number of qubits for phase precision (default: 4)
        eigenvector_prep: Optional circuit to prepare the eigenvector |ψ⟩
        use_barriers: Whether to add barriers for visualization

    Attributes:
        circuit: The complete phase estimation circuit
        num_counting_qubits: Number of counting qubits
        num_unitary_qubits: Number of qubits for the unitary

    Example:
        >>> from qiskit.circuit.library import TGate
        >>> # T gate has eigenvalue e^(iπ/4) for |1⟩, so phase = 1/8
        >>> prep = QuantumCircuit(1); prep.x(0)  # Prepare |1⟩
        >>> pe = PhaseEstimation(TGate(), num_counting_qubits=4, eigenvector_prep=prep)
        >>> # Measurement should give binary approximation of 1/8 = 0.125
        >>> # With 4 qubits: 0.125 * 16 = 2, so expect "0010"
    """

    def __init__(
        self,
        unitary: Gate | QuantumCircuit,
        num_counting_qubits: int = 4,
        *,
        eigenvector_prep: QuantumCircuit = None,
        use_barriers: bool = True,
    ):
        if num_counting_qubits < 1:
            raise ValueError("num_counting_qubits must be at least 1")

        self.unitary = unitary
        self.num_counting_qubits = num_counting_qubits
        self.eigenvector_prep = eigenvector_prep
        self.use_barriers = use_barriers
        self.expected_phase = None  # Set via set_expected_phase()

        # Determine number of qubits for the unitary
        if isinstance(unitary, QuantumCircuit):
            self.num_unitary_qubits = unitary.num_qubits
        else:
            self.num_unitary_qubits = unitary.num_qubits

        # Total qubits = counting + unitary target
        self.num_qubits = num_counting_qubits + self.num_unitary_qubits

        # Build the circuit and decompose QFTGate for simulation
        self.circuit = self._build_phase_estimation_circuit().decompose()

    def _build_phase_estimation_circuit(self) -> QuantumCircuit:
        """Build the phase estimation circuit."""
        # Create registers
        counting_reg = QuantumRegister(self.num_counting_qubits, "counting")
        unitary_reg = QuantumRegister(self.num_unitary_qubits, "eigenstate")
        classical_reg = ClassicalRegister(self.num_counting_qubits, "result")

        qc = QuantumCircuit(counting_reg, unitary_reg, classical_reg)

        # Step 1: Prepare eigenvector if circuit provided
        if self.eigenvector_prep is not None:
            qc.compose(
                self.eigenvector_prep,
                qubits=range(self.num_counting_qubits, self.num_qubits),
                inplace=True,
            )

        if self.use_barriers:
            qc.barrier()

        # Step 2: Put counting qubits in superposition
        for i in range(self.num_counting_qubits):
            qc.h(counting_reg[i])

        if self.use_barriers:
            qc.barrier()

        # Step 3: Apply controlled-U^(2^j) operations
        # Each counting qubit j controls U^(2^j)
        for j in range(self.num_counting_qubits):
            power = 2**j
            controlled_u_power = self._create_controlled_power(power)

            # Apply with counting qubit j as control
            qc.compose(
                controlled_u_power,
                qubits=[counting_reg[j]] + list(unitary_reg),
                inplace=True,
            )

        if self.use_barriers:
            qc.barrier()

        # Step 4: Apply inverse QFT to counting register
        qft_inv_gate = QFTGate(self.num_counting_qubits).inverse()
        qc.append(qft_inv_gate, counting_reg)

        if self.use_barriers:
            qc.barrier()

        # Step 5: Measure counting register
        qc.measure(counting_reg, classical_reg)

        return qc

    def _create_controlled_power(self, power: int) -> QuantumCircuit:
        """Create a controlled U^power gate.

        Args:
            power: The power to raise U to

        Returns:
            QuantumCircuit implementing controlled-U^power
        """
        # Create circuit with 1 control + unitary qubits
        qc = QuantumCircuit(1 + self.num_unitary_qubits, name=f"c-U^{power}")

        if isinstance(self.unitary, QuantumCircuit):
            # Raise unitary to power by repeated composition
            u_power = QuantumCircuit(self.num_unitary_qubits)
            for _ in range(power):
                u_power.compose(self.unitary, inplace=True)
            controlled_gate = u_power.to_gate().control(1)
        else:
            # Gate case - use power method if available
            u_power = self.unitary.power(power)
            controlled_gate = u_power.control(1)

        qc.append(controlled_gate, range(1 + self.num_unitary_qubits))

        return qc

    def success_criteria(self, outcome: str) -> bool:
        """Determine if measurement matches expected phase.

        The measured value m should satisfy: m/2^n ≈ φ
        where n is num_counting_qubits.

        Args:
            outcome: Measurement result string

        Returns:
            True if measurement is within tolerance of expected phase
        """
        if self.expected_phase is None:
            return True  # No expected phase set, accept all

        clean_outcome = outcome.replace(" ", "")
        measured_value = int(clean_outcome, 2)
        measured_phase = measured_value / (2**self.num_counting_qubits)

        # Allow tolerance of 1/(2^n) for measurement error
        tolerance = 1 / (2**self.num_counting_qubits)
        phase_diff = abs(measured_phase - self.expected_phase)

        # Handle wraparound (e.g., 0.99 vs 0.01)
        phase_diff = min(phase_diff, 1.0 - phase_diff)

        return phase_diff <= tolerance

    def expected_distribution(self) -> dict:
        """Get expected probability distribution.

        Returns:
            Dictionary mapping outcome strings to expected probabilities
        """
        if self.expected_phase is None:
            # Return uniform distribution as fallback
            N = 2**self.num_counting_qubits
            return {format(i, f"0{self.num_counting_qubits}b"): 1 / N for i in range(N)}

        n = self.num_counting_qubits
        N = 2**n
        phi = self.expected_phase % 1.0

        # Ideal QPE distribution over all outcomes
        expected = {}
        total_prob = 0.0
        for m in range(N):
            delta = phi - (m / N)
            angle = math.pi * delta
            if abs(angle) < 1e-12:
                prob = 1.0
            else:
                numerator = math.sin(math.pi * (N * delta))
                denominator = math.sin(angle)
                prob = (numerator / denominator) ** 2 / (N**2)
            state = format(m, f"0{n}b")
            expected[state] = prob
            total_prob += prob

        if total_prob > 0:
            for state in expected:
                expected[state] /= total_prob

        return expected

    def set_expected_phase(self, phase: float) -> "PhaseEstimation":
        """Set the expected phase for success criteria.

        Args:
            phase: Expected phase value in [0, 1)

        Returns:
            Self for method chaining
        """
        self.expected_phase = phase % 1.0  # Normalize to [0, 1)
        return self

    def draw(self, **kwargs):
        """Draw the circuit.

        Args:
            **kwargs: Additional arguments passed to circuit.draw()

        Returns:
            Circuit visualization
        """
        output = kwargs.pop("output", "mpl")
        return self.circuit.draw(output=output, **kwargs)

    def get_phase_from_measurement(self, measurement: str) -> float:
        """Convert measurement outcome to phase estimate.

        Args:
            measurement: Measurement result string (e.g., "0010")

        Returns:
            Estimated phase as float in [0, 1)
        """
        clean_outcome = measurement.replace(" ", "")
        measured_value = int(clean_outcome, 2)
        return measured_value / (2**self.num_counting_qubits)


class PhaseEstimationCircuitGenerator:
    """Circuit generator for testing Phase Estimation with common unitaries.

    Provides pre-built test cases using known unitaries with known eigenvalues
    for validation and scalability testing.

    Test Cases:
    -----------
    - "t_gate": T gate on |1⟩ → eigenvalue e^(iπ/4), phase = 1/8 = 0.125
    - "s_gate": S gate on |1⟩ → eigenvalue e^(iπ/2) = i, phase = 1/4 = 0.25
    - "z_gate": Z gate on |1⟩ → eigenvalue e^(iπ) = -1, phase = 1/2 = 0.5
    - "custom": User-provided unitary and expected phase

    Understanding the phases:
    -------------------------
    For a gate G with eigenvalue λ = e^(2πiφ):
    - T gate: T|1⟩ = e^(iπ/4)|1⟩ = e^(2πi·1/8)|1⟩, so φ = 1/8
    - S gate: S|1⟩ = e^(iπ/2)|1⟩ = e^(2πi·1/4)|1⟩, so φ = 1/4
    - Z gate: Z|1⟩ = e^(iπ)|1⟩ = e^(2πi·1/2)|1⟩, so φ = 1/2

    Args:
        test_case: "t_gate", "s_gate", "z_gate", or "custom"
        num_counting_qubits: Number of qubits for phase precision (default: 4)
        custom_unitary: User-provided unitary (required for "custom")
        custom_phase: Expected phase for custom unitary (required for "custom")
        use_barriers: Whether to add barriers for visualization

    Example:
        >>> gen = PhaseEstimationCircuitGenerator(test_case="t_gate", num_counting_qubits=4)
        >>> # T gate phase = 1/8, with 4 qubits: 1/8 * 16 = 2
        >>> # Expected measurement: "0010" (binary for 2)
        >>> gen.success_criteria("0010")  # True
        >>> gen.get_expected_phase()  # 0.125
    """

    def __init__(
        self,
        test_case: str = "t_gate",
        num_counting_qubits: int = 4,
        *,
        custom_unitary: Gate | QuantumCircuit = None,
        custom_phase: float = None,
        custom_eigenvector_prep: QuantumCircuit = None,
        use_barriers: bool = True,
    ):
        self.test_case = test_case
        self.num_counting_qubits = num_counting_qubits
        self.use_barriers = use_barriers

        # Get unitary and expected phase based on test case
        self.unitary, self.expected_phase, self.eigenvector_prep = self._get_test_case_params(
            test_case, custom_unitary, custom_phase, custom_eigenvector_prep
        )

        # Create phase estimation instance
        self.phase_estimation = PhaseEstimation(
            unitary=self.unitary,
            num_counting_qubits=num_counting_qubits,
            eigenvector_prep=self.eigenvector_prep,
            use_barriers=use_barriers,
        )
        self.phase_estimation.set_expected_phase(self.expected_phase)

        # Direct access to circuit (already decomposed from PhaseEstimation)
        self.circuit = self.phase_estimation.circuit

    def _get_test_case_params(self, test_case, custom_unitary, custom_phase, custom_prep):
        """Get unitary, expected phase, and eigenvector prep for test case.

        Args:
            test_case: Test case name
            custom_unitary: User-provided unitary
            custom_phase: User-provided expected phase
            custom_prep: User-provided eigenvector preparation

        Returns:
            Tuple of (unitary, expected_phase, eigenvector_prep)
        """
        from qiskit.circuit.library import TGate, SGate, ZGate

        if test_case == "t_gate":
            # T gate: T|1⟩ = e^(iπ/4)|1⟩, phase = 1/8 = 0.125
            unitary = TGate()
            expected_phase = 1 / 8
            # Prepare |1⟩ eigenvector
            prep = QuantumCircuit(1)
            prep.x(0)
            return unitary, expected_phase, prep

        elif test_case == "s_gate":
            # S gate: S|1⟩ = e^(iπ/2)|1⟩ = i|1⟩, phase = 1/4 = 0.25
            unitary = SGate()
            expected_phase = 1 / 4
            # Prepare |1⟩ eigenvector
            prep = QuantumCircuit(1)
            prep.x(0)
            return unitary, expected_phase, prep

        elif test_case == "z_gate":
            # Z gate: Z|1⟩ = -|1⟩ = e^(iπ)|1⟩, phase = 1/2 = 0.5
            unitary = ZGate()
            expected_phase = 1 / 2
            # Prepare |1⟩ eigenvector
            prep = QuantumCircuit(1)
            prep.x(0)
            return unitary, expected_phase, prep

        elif test_case == "custom":
            if custom_unitary is None or custom_phase is None:
                raise ValueError("custom_unitary and custom_phase required for 'custom' test case")
            return custom_unitary, custom_phase, custom_prep

        else:
            raise ValueError(
                f"Unknown test_case: {test_case}. " "Use 't_gate', 's_gate', 'z_gate', or 'custom'"
            )

    def success_criteria(self, outcome: str) -> bool:
        """Delegate to phase estimation success criteria.

        Args:
            outcome: Measurement result string

        Returns:
            True if outcome matches expected phase
        """
        return self.phase_estimation.success_criteria(outcome)

    def expected_distribution(self) -> dict:
        """Delegate to phase estimation expected distribution.

        Returns:
            Dictionary mapping outcomes to probabilities
        """
        return self.phase_estimation.expected_distribution()

    def draw(self, **kwargs):
        """Draw the circuit.

        Args:
            **kwargs: Additional arguments passed to circuit.draw()

        Returns:
            Circuit visualization
        """
        output = kwargs.pop("output", "mpl")
        return self.circuit.draw(output=output, **kwargs)

    def get_expected_phase(self) -> float:
        """Get the expected phase for this test case.

        Returns:
            Expected phase value
        """
        return self.expected_phase

    def get_phase_estimation(self) -> PhaseEstimation:
        """Get the underlying PhaseEstimation instance.

        Returns:
            The PhaseEstimation instance
        """
        return self.phase_estimation
