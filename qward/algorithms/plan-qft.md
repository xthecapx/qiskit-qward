# Plan: Quantum Fourier Transform (QFT) & Phase Estimation for qWard

## Executive Summary

This document outlines the implementation plan for QFT and Phase Estimation modules following the patterns established in `grover.py` and `v_tp.py`. The key challenge is defining meaningful **success functions** since QFT is a unitary transformation.

### ⚠️ Important: Qiskit 2.1 Deprecation Notice

```python
# DEPRECATED (will be removed in Qiskit 3.0):
from qiskit.circuit.library import QFT  # ❌ Don't use

# NEW API - Use these instead:
from qiskit.circuit.library import QFTGate  # ✅ For gate-based composition
from qiskit.synthesis.qft import synth_qft_full  # ✅ For synthesis
```

---

## 1. Feasibility Analysis

### ✅ Implementation is Feasible

**Qiskit Support (Updated for 2.1+):**
- Use `qiskit.circuit.library.QFTGate` for gate-based QFT
- Use `qiskit.synthesis.qft.synth_qft_full` for circuit synthesis
- Also available: `QFTSynthesisFull` and `QFTSynthesisLine` for hardware-optimized synthesis

**Mathematical Foundation (from BasicQuantumAlgorithms.tex):**

The QFT on $n$ qubits transforms computational basis states as:

$$F_{2^n}|k\rangle = \frac{1}{\sqrt{2^n}} \sum_{\ell=0}^{2^n-1} \omega^{k\ell} |\ell\rangle$$

where $\omega = e^{2\pi i / 2^n}$

**Gate Complexity:**
- Total gates: $\frac{n(n+1)}{2} + \lfloor n/2 \rfloor$
- Uses: Hadamard gates, controlled phase rotations $R_k$, and SWAP gates
- Complexity: $O(n^2)$ vs classical FFT's $O(n \cdot 2^n)$

---

## 2. Circuit Design

### 2.1 QFT Circuit Structure

From the research paper (BasicQuantumAlgorithms.tex, Section 4001+):

```
Circuit for n qubits:
┌───┐                                          
│ H │──■────■────■──────────────────────────────
└───┘┌─┴─┐  │    │                              
─────│R_2│──┼────┼──────────────────────────────
     └───┘┌─┴─┐  │    ┌───┐                     
──────────│R_3│──┼────│ H │──■────■─────────────
          └───┘┌─┴─┐  └───┘┌─┴─┐  │             
───────────────│R_4│───────│R_2│──┼──────┌───┐──
               └───┘       └───┘┌─┴─┐    │ H │──
                                │R_3│────└───┘  
                                └───┘    + SWAPs
```

**Key Components:**
1. **Block k** (for qubit k): Apply H, then CR_2, CR_3, ..., CR_{n-k+1}
2. **Final block**: Apply ⌊n/2⌋ SWAP gates to reverse qubit order

### 2.2 Controlled Rotation Gates

$$R_k = \begin{bmatrix} 1 & 0 \\ 0 & e^{2\pi i / 2^k} \end{bmatrix}$$

Special cases:
- $R_0 = I$ (identity)
- $R_1 = Z$ (Pauli-Z)
- $R_2 = S$ (Phase gate)
- $R_3 = T$ (π/8 gate)

---

## 3. Success Function Definition ⚠️ KEY CHALLENGE

Unlike Grover (marked states) or Teleportation (state transfer verification), QFT is a **unitary transformation**. We implement **two test modes**:

### ✅ Mode A: Round-Trip Verification

**Concept:** Apply QFT → Inverse QFT and verify return to original state.

```python
def success_criteria(self, outcome: str) -> bool:
    """Success if round-trip returns to original basis state."""
    return outcome == self.input_state
```

**Expected Distribution:** Concentrated on the prepared input state.

| Pros | Cons |
|------|------|
| Simple, verifiable | Only tests QFT⁻¹ ∘ QFT = I |
| Works with measurement | Doesn't test QFT individually |
| Hardware-friendly | - |

### ✅ Mode B: Period Detection Test

**Concept:** Prepare a periodic function state and verify QFT detects the period.

From PennyLane tutorial: Prepare state with period T:
$$|\psi\rangle = \frac{1}{\sqrt{2^n}} \sum_{x=0}^{2^n-1} e^{-2\pi i x / T} |x\rangle$$

After QFT, output should peak at $|2^n / T\rangle$

```python
def success_criteria(self, outcome: str) -> bool:
    """Success if measurement reveals the encoded period."""
    measured_value = int(outcome, 2)
    expected_peak = 2**self.num_qubits // self.period
    # Allow small tolerance for approximate peaks
    return abs(measured_value - expected_peak) <= self.tolerance
```

| Pros | Cons |
|------|------|
| Tests actual QFT functionality | More complex setup |
| Scalable | Requires period encoding |
| Used in real algorithms (Shor) | - |

### 📖 Theoretical Understanding Only: Statevector Fidelity

For understanding QFT behavior (not a circuit mode):
- QFT on basis state $|k\rangle$ produces uniform superposition
- Probabilities: $1/N$ for each state where $N = 2^n$
- Phase information encoded in amplitudes (not measurable directly)

---

## 4. Scalability Analysis

### 4.1 Circuit Growth

| Qubits (n) | Gates | Circuit Depth | Two-qubit Gates |
|------------|-------|---------------|-----------------|
| 2          | 4     | 3             | 1               |
| 3          | 8     | 5             | 3               |
| 4          | 13    | 7             | 6               |
| 5          | 19    | 9             | 10              |
| 8          | 42    | 15            | 28              |
| 10         | 65    | 19            | 45              |
| 16         | 152   | 31            | 120             |

Formula: Gates = $\frac{n(n+1)}{2} + \lfloor n/2 \rfloor$

### 4.2 Limiting Factors

1. **Gate Errors Accumulation:**
   - For noise level $\epsilon$ per gate, total error ~$\epsilon \cdot \text{gates}$
   - At n=10 with 1% gate error: ~65% total error accumulation

2. **Small Rotation Problem:**
   - $R_k$ for large k has phase $2\pi/2^k$
   - For k=10: rotation angle = $2\pi/1024 \approx 0.006$ radians
   - These become noise-dominated on real hardware

> **Note:** We implement **exact QFT only** (no approximation). The small rotation problem is documented for understanding hardware limits, not for mitigation in our implementation.

### 4.3 Scalability Test Plan

```python
SCALE_CONFIGS = [
    {"qubits": 2, "expected_success_rate": 0.99},   # Easy
    {"qubits": 4, "expected_success_rate": 0.95},   # Standard
    {"qubits": 6, "expected_success_rate": 0.90},   # Moderate
    {"qubits": 8, "expected_success_rate": 0.80},   # Challenging
    {"qubits": 10, "expected_success_rate": 0.60},  # Difficult
    {"qubits": 12, "expected_success_rate": 0.40},  # Limit
]
```

---

## 5. Implementation Plan

### 5.1 QFT Classes

```python
# qward/algorithms/qft.py

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate  # New API (Qiskit 2.1+)
import numpy as np


class QFT:
    """Quantum Fourier Transform implementation.
    
    Uses QFTGate from qiskit.circuit.library (Qiskit 2.1+ compatible).
    Similar to Grover class pattern - provides the core QFT circuit
    with configurable parameters.
    
    Note: This is a wrapper that uses the new Qiskit API. For the deprecated
    qiskit.circuit.library.QFT class, use QFTGate instead.
    """
    
    def __init__(
        self,
        num_qubits: int,
        *,
        do_swaps: bool = True,
        inverse: bool = False,
        use_barriers: bool = True,
    ):
        self.num_qubits = num_qubits
        self.do_swaps = do_swaps
        self.inverse = inverse
        self.use_barriers = use_barriers
        
        # Build the circuit using QFTGate
        self.circuit = self._build_qft_circuit()
    
    def _build_qft_circuit(self) -> QuantumCircuit:
        """Build QFT circuit using QFTGate."""
        qc = QuantumCircuit(self.num_qubits, name="QFT" if not self.inverse else "QFT†")
        
        # Create QFTGate
        qft_gate = QFTGate(num_qubits=self.num_qubits)
        
        if self.inverse:
            qft_gate = qft_gate.inverse()
        
        # Append to circuit
        qc.append(qft_gate, range(self.num_qubits))
        
        return qc
    
    def draw(self):
        """Draw the circuit."""
        return self.circuit.draw(output="mpl")
    
    def get_decomposed_circuit(self) -> QuantumCircuit:
        """Get the decomposed circuit showing individual gates."""
        return self.circuit.decompose()


class QFTCircuitGenerator:
    """Circuit generator for testing QFT with various configurations.
    
    Similar to TeleportationCircuitGenerator - creates test circuits
    that validate QFT functionality.
    
    Test Modes:
    - "roundtrip": Apply QFT → QFT⁻¹, verify return to input state
    - "period_detection": Encode period, use QFT to detect it
    """
    
    def __init__(
        self,
        num_qubits: int = 4,
        *,
        test_mode: str = "roundtrip",  # "roundtrip" or "period_detection"
        input_state: str = None,       # For roundtrip: e.g., "0101"
        period: int = None,            # For period_detection
        use_barriers: bool = True,
    ):
        self.num_qubits = num_qubits
        self.test_mode = test_mode
        self.use_barriers = use_barriers
        
        # Validate and set test-mode specific parameters
        if test_mode == "roundtrip":
            self.input_state = input_state or "0" * num_qubits
            if len(self.input_state) != num_qubits:
                raise ValueError(f"input_state must have {num_qubits} bits")
        elif test_mode == "period_detection":
            if period is None:
                raise ValueError("period must be specified for period_detection mode")
            self.period = period
            self.tolerance = 1  # Allow ±1 in peak detection
        else:
            raise ValueError(f"Unknown test_mode: {test_mode}")
        
        # Create the QFT instance
        self.qft = QFT(num_qubits, use_barriers=use_barriers)
        
        # Build the test circuit
        self.circuit = self._create_test_circuit()
    
    def _create_test_circuit(self) -> QuantumCircuit:
        """Create test circuit based on mode."""
        if self.test_mode == "roundtrip":
            return self._create_roundtrip_circuit()
        elif self.test_mode == "period_detection":
            return self._create_period_detection_circuit()
    
    def _create_roundtrip_circuit(self) -> QuantumCircuit:
        """Create QFT → QFT⁻¹ test circuit."""
        qc = QuantumCircuit(self.num_qubits)
        
        # Prepare input state
        for i, bit in enumerate(self.input_state[::-1]):
            if bit == '1':
                qc.x(i)
        
        if self.use_barriers:
            qc.barrier()
        
        # Apply QFT
        qc.compose(self.qft.circuit, inplace=True)
        
        if self.use_barriers:
            qc.barrier()
        
        # Apply inverse QFT
        qft_inv = QFT(self.num_qubits, inverse=True, use_barriers=False)
        qc.compose(qft_inv.circuit, inplace=True)
        
        if self.use_barriers:
            qc.barrier()
        
        # Measure
        qc.measure_all()
        
        return qc
    
    def _create_period_detection_circuit(self) -> QuantumCircuit:
        """Create circuit that tests QFT period detection capability.
        
        Uses phase kickback to encode a periodic function, then applies
        inverse QFT to extract the period (similar to Shor's algorithm).
        """
        # Need one ancilla qubit for phase kickback
        qc = QuantumCircuit(self.num_qubits + 1, self.num_qubits)
        
        # Prepare ancilla in |1⟩ state for phase kickback
        qc.x(0)
        
        # Put counting qubits in superposition
        for i in range(1, self.num_qubits + 1):
            qc.h(i)
        
        if self.use_barriers:
            qc.barrier()
        
        # Apply controlled phase rotations to encode period
        # This creates a state whose QFT will peak at multiples of 2^n/period
        for i in range(self.num_qubits):
            angle = 2 * np.pi * (2**i) / self.period
            qc.cp(angle, i + 1, 0)
        
        if self.use_barriers:
            qc.barrier()
        
        # Apply inverse QFT to counting register to extract period
        qft_inv = QFT(self.num_qubits, inverse=True, use_barriers=False)
        qc.compose(qft_inv.circuit, qubits=range(1, self.num_qubits + 1), inplace=True)
        
        if self.use_barriers:
            qc.barrier()
        
        # Measure counting register only
        qc.measure(range(1, self.num_qubits + 1), range(self.num_qubits))
        
        return qc
    
    def success_criteria(self, outcome: str) -> bool:
        """Define success based on test mode."""
        clean_outcome = outcome.replace(" ", "")
        
        if self.test_mode == "roundtrip":
            return clean_outcome == self.input_state
        elif self.test_mode == "period_detection":
            return self._check_period_peak(clean_outcome)
        return False
    
    def _check_period_peak(self, outcome: str) -> bool:
        """Check if measurement is at expected period peak."""
        measured_value = int(outcome, 2)
        N = 2**self.num_qubits
        expected_peak = N // self.period
        
        # Check if close to any multiple of expected_peak
        if expected_peak == 0:
            return measured_value == 0
        
        remainder = measured_value % expected_peak
        return remainder <= self.tolerance or (expected_peak - remainder) <= self.tolerance
    
    def expected_distribution(self) -> dict:
        """Get expected probability distribution based on test mode."""
        if self.test_mode == "roundtrip":
            # Should return to input state with high probability
            return {self.input_state: 1.0}
        elif self.test_mode == "period_detection":
            return self._get_period_peaks_distribution()
        return {}
    
    def _get_period_peaks_distribution(self) -> dict:
        """Get expected distribution for period detection."""
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
            state = format(peak_value, f'0{self.num_qubits}b')
            peaks[state] = prob_per_peak
        
        return peaks
    
    def draw(self):
        """Draw the circuit."""
        return self.circuit.draw(output="mpl")
```

---

## 6. Phase Estimation Implementation

### 6.1 Background (from BasicQuantumAlgorithms.tex, Section 5007+)

Phase Estimation (Kitaev, 1995) finds the eigenvalue $e^{2\pi i \phi}$ of a unitary operator $U$ given its eigenvector $|\psi\rangle$.

**Key insight:** $U|\psi\rangle = e^{2\pi i \phi}|\psi\rangle$

The algorithm uses:
1. **m counting qubits** for precision (determines accuracy of $\phi$)
2. **n qubits** for the eigenvector $|\psi\rangle$
3. **Controlled-U operations** with phase kickback
4. **Inverse QFT** to extract the phase

### 6.2 Phase Estimation Circuit

```
          ┌───┐                                    ┌─────────┐
|0⟩ ──────│ H │────■───────────────────────────────│         │── M
          └───┘    │                               │         │
          ┌───┐    │                               │         │
|0⟩ ──────│ H │────┼────■──────────────────────────│ QFT⁻¹   │── M
          └───┘    │    │                          │         │
          ┌───┐    │    │                          │         │
|0⟩ ──────│ H │────┼────┼────■─────────────────────│         │── M
          └───┘    │    │    │                     └─────────┘
                 ┌─┴─┐┌─┴──┐┌┴───┐
|ψ⟩ ─────────────│U^1││U^2 ││U^4 │─────────────────────────────
                 └───┘└────┘└────┘
```

### 6.3 PhaseEstimation Class Design

```python
# qward/algorithms/phase_estimation.py

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFTGate
from qiskit.circuit import Gate
import numpy as np


class PhaseEstimation:
    """Quantum Phase Estimation implementation.
    
    Given a unitary operator U and one of its eigenvectors |ψ⟩,
    estimates the phase φ where U|ψ⟩ = e^(2πiφ)|ψ⟩.
    
    Based on Kitaev's algorithm (1995) as described in BasicQuantumAlgorithms.tex.
    
    Args:
        unitary: The unitary operator U (as a Gate or QuantumCircuit)
        num_counting_qubits: Number of qubits for phase precision (m)
        eigenvector_prep: Optional circuit to prepare the eigenvector |ψ⟩
        use_barriers: Whether to add barriers for visualization
    """
    
    def __init__(
        self,
        unitary: Gate | QuantumCircuit,
        num_counting_qubits: int = 4,
        *,
        eigenvector_prep: QuantumCircuit = None,
        use_barriers: bool = True,
    ):
        self.unitary = unitary
        self.num_counting_qubits = num_counting_qubits
        self.eigenvector_prep = eigenvector_prep
        self.use_barriers = use_barriers
        
        # Determine number of qubits for the unitary
        if isinstance(unitary, QuantumCircuit):
            self.num_unitary_qubits = unitary.num_qubits
        else:
            self.num_unitary_qubits = unitary.num_qubits
        
        # Total qubits = counting + unitary target
        self.num_qubits = num_counting_qubits + self.num_unitary_qubits
        
        # Build the circuit
        self.circuit = self._build_phase_estimation_circuit()
    
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
                inplace=True
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
            # Create controlled version of U^(2^j)
            power = 2 ** j
            controlled_u_power = self._create_controlled_power(power)
            
            # Apply with counting qubit j as control
            qc.compose(
                controlled_u_power,
                qubits=[counting_reg[j]] + list(unitary_reg),
                inplace=True
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
        """Create a controlled U^power gate."""
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
        """Success if measurement matches expected phase.
        
        The measured value m should satisfy: m/2^n ≈ φ
        where n is num_counting_qubits.
        """
        if not hasattr(self, 'expected_phase'):
            return True  # No expected phase set
        
        clean_outcome = outcome.replace(" ", "")
        measured_value = int(clean_outcome, 2)
        measured_phase = measured_value / (2 ** self.num_counting_qubits)
        
        # Allow tolerance of 1/(2^n) for measurement error
        tolerance = 1 / (2 ** self.num_counting_qubits)
        return abs(measured_phase - self.expected_phase) <= tolerance
    
    def expected_distribution(self) -> dict:
        """Get expected probability distribution.
        
        If expected_phase is set, return distribution peaked at that phase.
        """
        if not hasattr(self, 'expected_phase') or self.expected_phase is None:
            # Return uniform distribution as fallback
            N = 2 ** self.num_counting_qubits
            return {format(i, f'0{self.num_counting_qubits}b'): 1/N for i in range(N)}
        
        # Expected measurement is closest integer to φ * 2^n
        expected_m = round(self.expected_phase * (2 ** self.num_counting_qubits))
        expected_m = expected_m % (2 ** self.num_counting_qubits)
        expected_state = format(expected_m, f'0{self.num_counting_qubits}b')
        
        return {expected_state: 1.0}
    
    def set_expected_phase(self, phase: float):
        """Set the expected phase for success criteria."""
        self.expected_phase = phase % 1.0  # Normalize to [0, 1)
        return self
    
    def draw(self):
        """Draw the circuit."""
        return self.circuit.draw(output="mpl")
    
    def get_phase_from_measurement(self, measurement: str) -> float:
        """Convert measurement outcome to phase estimate."""
        clean_outcome = measurement.replace(" ", "")
        measured_value = int(clean_outcome, 2)
        return measured_value / (2 ** self.num_counting_qubits)


class PhaseEstimationCircuitGenerator:
    """Circuit generator for testing Phase Estimation with common unitaries.
    
    Provides pre-built test cases using known unitaries with known eigenvalues
    for validation and scalability testing.
    
    Test Cases:
    - "t_gate": T gate with eigenvalue e^(iπ/4), phase = 1/8
    - "s_gate": S gate with eigenvalue e^(iπ/2), phase = 1/4  
    - "z_gate": Z gate with eigenvalue -1, phase = 1/2
    - "custom": User-provided unitary and expected phase
    """
    
    def __init__(
        self,
        test_case: str = "t_gate",
        num_counting_qubits: int = 4,
        *,
        custom_unitary: Gate | QuantumCircuit = None,
        custom_phase: float = None,
        use_barriers: bool = True,
    ):
        self.test_case = test_case
        self.num_counting_qubits = num_counting_qubits
        self.use_barriers = use_barriers
        
        # Get unitary and expected phase based on test case
        self.unitary, self.expected_phase, self.eigenvector_prep = \
            self._get_test_case_params(test_case, custom_unitary, custom_phase)
        
        # Create phase estimation instance
        self.phase_estimation = PhaseEstimation(
            unitary=self.unitary,
            num_counting_qubits=num_counting_qubits,
            eigenvector_prep=self.eigenvector_prep,
            use_barriers=use_barriers,
        )
        self.phase_estimation.set_expected_phase(self.expected_phase)
        
        # Direct access to circuit
        self.circuit = self.phase_estimation.circuit
    
    def _get_test_case_params(self, test_case, custom_unitary, custom_phase):
        """Get unitary, expected phase, and eigenvector prep for test case."""
        from qiskit.circuit.library import TGate, SGate, ZGate
        
        if test_case == "t_gate":
            # T gate: T|1⟩ = e^(iπ/4)|1⟩, phase = 1/8
            unitary = TGate()
            expected_phase = 1/8
            # Prepare |1⟩ eigenvector
            prep = QuantumCircuit(1)
            prep.x(0)
            return unitary, expected_phase, prep
        
        elif test_case == "s_gate":
            # S gate: S|1⟩ = e^(iπ/2)|1⟩ = i|1⟩, phase = 1/4
            unitary = SGate()
            expected_phase = 1/4
            # Prepare |1⟩ eigenvector
            prep = QuantumCircuit(1)
            prep.x(0)
            return unitary, expected_phase, prep
        
        elif test_case == "z_gate":
            # Z gate: Z|1⟩ = -|1⟩ = e^(iπ)|1⟩, phase = 1/2
            unitary = ZGate()
            expected_phase = 1/2
            # Prepare |1⟩ eigenvector
            prep = QuantumCircuit(1)
            prep.x(0)
            return unitary, expected_phase, prep
        
        elif test_case == "custom":
            if custom_unitary is None or custom_phase is None:
                raise ValueError("custom_unitary and custom_phase required for 'custom' test case")
            return custom_unitary, custom_phase, None
        
        else:
            raise ValueError(f"Unknown test_case: {test_case}")
    
    def success_criteria(self, outcome: str) -> bool:
        """Delegate to phase estimation success criteria."""
        return self.phase_estimation.success_criteria(outcome)
    
    def expected_distribution(self) -> dict:
        """Delegate to phase estimation expected distribution."""
        return self.phase_estimation.expected_distribution()
    
    def draw(self):
        """Draw the circuit."""
        return self.circuit.draw(output="mpl")
    
    def get_expected_phase(self) -> float:
        """Get the expected phase for this test case."""
        return self.expected_phase
```

---

## 7. Comparison with Existing Implementations

| Feature | Grover | Teleportation | QFT | Phase Estimation |
|---------|--------|---------------|-----|------------------|
| Success Criteria | Marked state found | State transferred | Round-trip / Period | Phase matches expected |
| Expected Dist. | High prob on marked | All zeros | Input state or peaks | Peak at φ×2^n |
| Scalability | O(√N) iterations | Fixed 3 qubits | O(n²) gates | O(n²) + O(U^2^n) |
| Key Parameter | marked_states | protocol_type | test_mode | num_counting_qubits |
| Hardware Limit | Noise amplification | Mid-circuit meas. | Small rotations | U^2^n depth |

---

## 8. Files to Create

1. **`qward/algorithms/qft.py`** - QFT and QFTCircuitGenerator classes
2. **`qward/algorithms/phase_estimation.py`** - PhaseEstimation and PhaseEstimationCircuitGenerator classes
3. **`tests/test_qft.py`** - Unit tests for QFT
4. **`tests/test_phase_estimation.py`** - Unit tests for Phase Estimation
5. **`qward/examples/qft_example.py`** - Example usage
6. **`qward/algorithms/qft.ipynb`** - Interactive notebook

### Integration Points

Update `qward/algorithms/__init__.py`:
```python
from .qft import (
    QFT,
    QFTCircuitGenerator,
)
from .phase_estimation import (
    PhaseEstimation,
    PhaseEstimationCircuitGenerator,
)

__all__ = [
    # ... existing exports ...
    "QFT",
    "QFTCircuitGenerator",
    "PhaseEstimation",
    "PhaseEstimationCircuitGenerator",
]
```

---

## 9. Testing Strategy

### 9.1 QFT Unit Tests
```python
def test_qft_roundtrip_identity():
    """QFT followed by inverse QFT should return to original state."""
    for n in [2, 3, 4, 5]:
        qft = QFTCircuitGenerator(num_qubits=n, test_mode="roundtrip")
        result = executor.simulate(qft.circuit, success_criteria=qft.success_criteria)
        assert result["success_rate"] > 0.95

def test_qft_period_detection():
    """QFT should detect encoded periods."""
    periods = [2, 4, 8]
    for period in periods:
        qft = QFTCircuitGenerator(num_qubits=6, test_mode="period_detection", period=period)
        result = executor.simulate(qft.circuit, success_criteria=qft.success_criteria)
        assert result["success_rate"] > 0.80
```

### 9.2 Phase Estimation Unit Tests
```python
def test_phase_estimation_t_gate():
    """Phase estimation should correctly identify T gate phase (1/8)."""
    pe = PhaseEstimationCircuitGenerator(test_case="t_gate", num_counting_qubits=4)
    result = executor.simulate(pe.circuit, success_criteria=pe.success_criteria)
    assert result["success_rate"] > 0.90

def test_phase_estimation_scalability():
    """Test phase estimation precision vs counting qubits."""
    for n in [3, 4, 5, 6]:
        pe = PhaseEstimationCircuitGenerator(test_case="t_gate", num_counting_qubits=n)
        # Expected precision: 1/2^n
        expected_precision = 1 / (2**n)
        # Verify circuit can achieve this precision
```

### 9.3 Noise Sensitivity Tests
```python
def test_qft_noise_sensitivity():
    """Test QFT success rate degradation with noise."""
    noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05]
    for noise in noise_levels:
        qft = QFTCircuitGenerator(num_qubits=4, test_mode="roundtrip")
        result = executor.simulate(
            qft.circuit,
            noise_model="depolarizing",
            noise_level=noise,
            success_criteria=qft.success_criteria
        )
        # Log success_rate vs noise for analysis
```

---

## 10. References

1. **BasicQuantumAlgorithms.tex** 
   - Section 4001-4150: QFT circuit derivation
   - Section 5007+: Phase Estimation algorithm (Kitaev)
2. **PennyLane Tutorial** - [Intro to QFT](https://pennylane.ai/qml/demos/tutorial_qft)
3. **Qiskit Documentation** 
   - `qiskit.circuit.library.QFTGate` (new API)
   - `qiskit.synthesis.qft.synth_qft_full`

---

## 11. Implementation Checklist

- [x] Implement `QFT` class using `QFTGate` ✅ `qward/algorithms/qft.py`
- [x] Implement `QFTCircuitGenerator` with roundtrip mode ✅ `qward/algorithms/qft.py`
- [x] Add period detection mode to `QFTCircuitGenerator` ✅ `qward/algorithms/qft.py`
- [x] Implement `PhaseEstimation` class ✅ `qward/algorithms/phase_estimation.py`
- [x] Implement `PhaseEstimationCircuitGenerator` with test cases ✅ `qward/algorithms/phase_estimation.py`
- [ ] Create unit tests for QFT (pending: `tests/test_qft.py`)
- [ ] Create unit tests for Phase Estimation (pending: `tests/test_phase_estimation.py`)
- [x] Create QFT experiment framework ✅ `qward/examples/papers/qft/`
- [ ] Run scalability analysis (use experiment framework)
- [ ] Create example notebook (skipped per user request)
- [x] Update `__init__.py` ✅ `qward/algorithms/__init__.py`

---

## 12. Decisions Made ✅

| Question | Decision |
|----------|----------|
| Approximate QFT? | **No** - Implement exact QFT only |
| Test modes | **Both** roundtrip and period_detection |
| Statevector fidelity circuit? | **No** - Theoretical understanding only |
| Qiskit API | Use `QFTGate` (not deprecated `QFT`) |
| Phase Estimation | **Yes** - Separate class |

---

*Document created: January 2026*
*Updated: January 2026*
*Author: qWard Development Team*
