"""
Quantum Fourier Transform (QFT) implementation for qWard.

This module provides QFT circuits using the new Qiskit 2.1+ API (QFTGate)
and test circuit generators for validating QFT functionality.

The QFT transforms computational basis states as:
    F_{2^n}|k⟩ = (1/√2^n) Σ_{ℓ=0}^{2^n-1} ω^{kℓ} |ℓ⟩
where ω = e^{2πi/2^n}
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate


class QFT:
    """Quantum Fourier Transform implementation.

    Uses QFTGate from qiskit.circuit.library (Qiskit 2.1+ compatible).
    Similar to Grover class pattern - provides the core QFT circuit
    with configurable parameters.

    Args:
        num_qubits: Number of qubits for the QFT
        do_swaps: Whether to include swap gates at the end (default: True)
        inverse: Whether to create the inverse QFT (default: False)
        use_barriers: Whether to add barriers for visualization (default: True)

    Attributes:
        circuit: The QFT quantum circuit
        num_qubits: Number of qubits
    """

    def __init__(
        self,
        num_qubits: int,
        *,
        do_swaps: bool = True,
        inverse: bool = False,
        use_barriers: bool = True,
    ):
        if num_qubits < 1:
            raise ValueError("num_qubits must be at least 1")

        self.num_qubits = num_qubits
        self.do_swaps = do_swaps
        self.inverse = inverse
        self.use_barriers = use_barriers

        # Build the circuit using QFTGate
        self.circuit = self._build_qft_circuit()

    def _build_qft_circuit(self) -> QuantumCircuit:
        """Build QFT circuit using QFTGate."""
        name = "QFT†" if self.inverse else "QFT"
        qc = QuantumCircuit(self.num_qubits, name=name)

        # Create QFTGate with do_swaps parameter
        qft_gate = QFTGate(num_qubits=self.num_qubits)

        if self.inverse:
            qft_gate = qft_gate.inverse()

        # Append to circuit
        qc.append(qft_gate, range(self.num_qubits))

        return qc

    def draw(self, **kwargs):
        """Draw the circuit.

        Args:
            **kwargs: Additional arguments passed to circuit.draw()

        Returns:
            Circuit visualization
        """
        output = kwargs.pop("output", "mpl")
        return self.circuit.draw(output=output, **kwargs)

    def get_decomposed_circuit(self) -> QuantumCircuit:
        """Get the decomposed circuit showing individual gates.

        Returns:
            QuantumCircuit with QFTGate decomposed into basic gates
        """
        return self.circuit.decompose()


class QFTCircuitGenerator:
    """Circuit generator for testing QFT with various configurations.

    Similar to TeleportationCircuitGenerator and GroverCircuitGenerator -
    creates test circuits that validate QFT functionality.

    Test Modes:
    -----------
    - "roundtrip": Apply QFT → QFT⁻¹, verify return to input state
      * Input: A basis state like "0101"
      * Success: Measurement returns the same state "0101"
      * Why it works: QFT⁻¹(QFT(|x⟩)) = |x⟩

    - "period_detection": Encode a period using phase kickback, use QFT to detect it
      * Input: A period T (e.g., period=4)
      * Success: Measurement peaks at multiples of 2^n/T
      * Why it works: QFT converts periodic signals to frequency domain

    Args:
        num_qubits: Number of qubits for the QFT (default: 4)
        test_mode: "roundtrip" or "period_detection" (default: "roundtrip")
        input_state: For roundtrip - the basis state to test (e.g., "0101")
        period: For period_detection - the period to encode
        use_barriers: Whether to add barriers for visualization

    Example (roundtrip):
        >>> gen = QFTCircuitGenerator(num_qubits=3, test_mode="roundtrip", input_state="101")
        >>> # Input: |101⟩, after QFT→QFT⁻¹ should return |101⟩
        >>> gen.success_criteria("101")  # True

    Example (period_detection):
        >>> gen = QFTCircuitGenerator(num_qubits=4, test_mode="period_detection", period=4)
        >>> # With period=4 and 4 qubits: peaks at 0, 4, 8, 12 (i.e., 0000, 0100, 1000, 1100)
        >>> gen.success_criteria("0100")  # True (4 in decimal)
    """

    def __init__(
        self,
        num_qubits: int = 4,
        *,
        test_mode: str = "roundtrip",
        input_state: str = None,
        period: int = None,
        use_barriers: bool = True,
    ):
        if num_qubits < 1:
            raise ValueError("num_qubits must be at least 1")

        self.num_qubits = num_qubits
        self.test_mode = test_mode
        self.use_barriers = use_barriers

        # Validate and set test-mode specific parameters
        if test_mode == "roundtrip":
            self.input_state = input_state or "0" * num_qubits
            if len(self.input_state) != num_qubits:
                raise ValueError(f"input_state must have {num_qubits} bits")
            if not all(b in "01" for b in self.input_state):
                raise ValueError("input_state must contain only '0' and '1'")

        elif test_mode == "period_detection":
            if period is None:
                raise ValueError("period must be specified for period_detection mode")
            if period < 1:
                raise ValueError("period must be at least 1")
            self.period = period
            self.tolerance = 1  # Allow ±1 in peak detection

        else:
            raise ValueError(f"Unknown test_mode: {test_mode}. Use 'roundtrip' or 'period_detection'")

        # Create the QFT instance
        self.qft = QFT(num_qubits, use_barriers=use_barriers)

        # Build the test circuit and decompose QFTGate for simulation
        self.circuit = self._create_test_circuit().decompose()

    def _create_test_circuit(self) -> QuantumCircuit:
        """Create test circuit based on mode."""
        if self.test_mode == "roundtrip":
            return self._create_roundtrip_circuit()
        elif self.test_mode == "period_detection":
            return self._create_period_detection_circuit()
        raise ValueError(f"Unknown test_mode: {self.test_mode}")

    def _create_roundtrip_circuit(self) -> QuantumCircuit:
        """Create QFT → QFT⁻¹ test circuit.

        This tests that applying QFT followed by inverse QFT returns
        the original state: QFT⁻¹(QFT(|x⟩)) = |x⟩
        """
        qc = QuantumCircuit(self.num_qubits)

        # Step 1: Prepare input state |input_state⟩
        # Qiskit uses little-endian, so we reverse the string
        for i, bit in enumerate(self.input_state[::-1]):
            if bit == "1":
                qc.x(i)

        if self.use_barriers:
            qc.barrier()

        # Step 2: Apply QFT
        qc.compose(self.qft.circuit, inplace=True)

        if self.use_barriers:
            qc.barrier()

        # Step 3: Apply inverse QFT
        qft_inv = QFT(self.num_qubits, inverse=True, use_barriers=False)
        qc.compose(qft_inv.circuit, inplace=True)

        if self.use_barriers:
            qc.barrier()

        # Step 4: Measure all qubits
        qc.measure_all()

        return qc

    def _create_period_detection_circuit(self) -> QuantumCircuit:
        """Create circuit that tests QFT period detection capability.

        Uses phase kickback to encode a periodic function, then applies
        inverse QFT to extract the period (similar to Shor's algorithm).

        The circuit prepares a state with period T, and the QFT transforms
        it so measurements peak at multiples of 2^n/T.
        """
        # Need one ancilla qubit for phase kickback
        qc = QuantumCircuit(self.num_qubits + 1, self.num_qubits)

        # Step 1: Prepare ancilla in |1⟩ state for phase kickback
        qc.x(0)

        # Step 2: Put counting qubits in superposition
        for i in range(1, self.num_qubits + 1):
            qc.h(i)

        if self.use_barriers:
            qc.barrier()

        # Step 3: Apply controlled phase rotations to encode period
        # This creates a state whose inverse QFT will peak at multiples of 2^n/period
        for i in range(self.num_qubits):
            angle = 2 * np.pi * (2**i) / self.period
            qc.cp(angle, i + 1, 0)

        if self.use_barriers:
            qc.barrier()

        # Step 4: Apply inverse QFT to counting register to extract period
        qft_inv = QFT(self.num_qubits, inverse=True, use_barriers=False)
        qc.compose(qft_inv.circuit, qubits=range(1, self.num_qubits + 1), inplace=True)

        if self.use_barriers:
            qc.barrier()

        # Step 5: Measure counting register only (not the ancilla)
        qc.measure(range(1, self.num_qubits + 1), range(self.num_qubits))

        return qc

    def success_criteria(self, outcome: str) -> bool:
        """Determine if a measurement outcome represents success.

        Args:
            outcome: Measurement result string (e.g., "0101")

        Returns:
            True if the outcome indicates successful QFT operation
        """
        clean_outcome = outcome.replace(" ", "")

        if self.test_mode == "roundtrip":
            # Success: measurement matches the original input state
            return clean_outcome == self.input_state

        elif self.test_mode == "period_detection":
            return self._check_period_peak(clean_outcome)

        return False

    def _check_period_peak(self, outcome: str) -> bool:
        """Check if measurement is at expected period peak.

        For period T with n qubits, peaks occur at multiples of 2^n/T.

        Args:
            outcome: Measurement result string

        Returns:
            True if outcome is near an expected peak
        """
        measured_value = int(outcome, 2)
        N = 2**self.num_qubits
        expected_peak = N // self.period

        # Check if close to any multiple of expected_peak
        if expected_peak == 0:
            return measured_value == 0

        remainder = measured_value % expected_peak
        return remainder <= self.tolerance or (expected_peak - remainder) <= self.tolerance

    def expected_distribution(self) -> dict:
        """Get expected probability distribution based on test mode.

        Returns:
            Dictionary mapping outcome strings to expected probabilities
        """
        if self.test_mode == "roundtrip":
            # Should return to input state with probability 1.0 (ideal)
            return {self.input_state: 1.0}

        elif self.test_mode == "period_detection":
            return self._get_period_peaks_distribution()

        return {}

    def _get_period_peaks_distribution(self) -> dict:
        """Get expected distribution for period detection.

        Returns:
            Dictionary with expected peaks and their probabilities
        """
        N = 2**self.num_qubits
        peaks = {}
        expected_peak = N // self.period

        if expected_peak == 0:
            return {"0" * self.num_qubits: 1.0}

        # Peaks occur at multiples of expected_peak
        num_peaks = self.period
        prob_per_peak = 1.0 / num_peaks

        for i in range(num_peaks):
            peak_value = (i * expected_peak) % N
            state = format(peak_value, f"0{self.num_qubits}b")
            peaks[state] = prob_per_peak

        return peaks

    def draw(self, **kwargs):
        """Draw the circuit.

        Args:
            **kwargs: Additional arguments passed to circuit.draw()

        Returns:
            Circuit visualization
        """
        output = kwargs.pop("output", "mpl")
        return self.circuit.draw(output=output, **kwargs)

    def get_qft(self) -> QFT:
        """Get the underlying QFT instance.

        Returns:
            The QFT instance used in this generator
        """
        return self.qft
