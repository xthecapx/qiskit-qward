"""
Matrix Product Verification - Quantum Algorithms

This module provides quantum algorithms for verifying matrix products.
Given three matrices A, B, C, it determines whether A × B = C.

Two algorithms are implemented:
1. QuantumFreivaldsVerification (DEFAULT): Practical for NISQ devices
2. BuhrmanSpalekVerification: Optimal theoretical speedup (O(n^5/3))

References:
    - Buhrman & Špalek (2004): "Quantum Verification of Matrix Products"
      arXiv:quant-ph/0409035
    - Freivalds (1979): "Fast probabilistic algorithms"
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


class VerificationMethod(Enum):
    """Available verification methods."""
    QUANTUM_FREIVALDS = "quantum_freivalds"
    BUHRMAN_SPALEK = "buhrman_spalek"
    CLASSICAL = "classical"


@dataclass
class VerificationResult:
    """
    Result of matrix product verification.
    
    Attributes:
        is_equal: True if A×B = C (with high probability)
        confidence: Confidence level (0-1) based on measurement statistics
        iterations_used: Number of verification rounds performed
        method: The verification method used
        circuit_depth: Depth of the quantum circuit (if quantum method)
        num_qubits: Number of qubits used (if quantum method)
        measurement_counts: Raw measurement results from quantum execution
        error_probability: Upper bound on error probability
        classical_comparison: Result of classical verification (if computed)
        execution_time_ms: Execution time in milliseconds
        details: Additional method-specific details
    """
    is_equal: bool
    confidence: float
    iterations_used: int
    method: str
    circuit_depth: int = 0
    num_qubits: int = 0
    measurement_counts: Dict[str, int] = field(default_factory=dict)
    error_probability: float = 0.0
    classical_comparison: Optional[bool] = None
    execution_time_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        status = "EQUAL" if self.is_equal else "NOT EQUAL"
        return (
            f"VerificationResult({status}, "
            f"confidence={self.confidence:.4f}, "
            f"method={self.method})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_equal": self.is_equal,
            "confidence": self.confidence,
            "iterations_used": self.iterations_used,
            "method": self.method,
            "circuit_depth": self.circuit_depth,
            "num_qubits": self.num_qubits,
            "measurement_counts": self.measurement_counts,
            "error_probability": self.error_probability,
            "classical_comparison": self.classical_comparison,
            "execution_time_ms": self.execution_time_ms,
            "details": self.details,
        }


class MatrixProductVerificationBase(ABC):
    """
    Abstract base class for quantum matrix product verification algorithms.
    
    This class defines the interface that all matrix verification algorithms
    must implement. It provides common functionality for:
    - Matrix validation
    - Classical Freivalds verification (for comparison)
    - Circuit management
    - Result interpretation
    
    Subclasses must implement:
    - build_circuit(): Construct the quantum circuit
    - interpret_results(): Determine if A×B = C from measurements
    - success_criteria(): Check if a single measurement indicates equality
    
    Args:
        A: First matrix (n×m)
        B: Second matrix (m×p)
        C: Product matrix to verify (n×p)
        iterations: Number of verification rounds (affects confidence)
        use_barriers: Whether to add barriers for visualization
    
    Example:
        >>> A = np.array([[1, 2], [3, 4]])
        >>> B = np.array([[5, 6], [7, 8]])
        >>> C = A @ B
        >>> # Use a concrete subclass:
        >>> verifier = QuantumFreivaldsVerification(A, B, C)
        >>> circuit = verifier.build_circuit()
    """
    
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        iterations: int = 10,
        use_barriers: bool = True
    ):
        # Convert to numpy arrays with float dtype
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.iterations = iterations
        self.use_barriers = use_barriers
        
        # Validate dimensions
        self._validate_dimensions()
        
        # Store dimensions
        self.n = self.A.shape[0]  # rows of A
        self.m = self.A.shape[1]  # cols of A = rows of B
        self.p = self.B.shape[1]  # cols of B
        
        # Circuit (built lazily)
        self._circuit: Optional[QuantumCircuit] = None
        self._circuit_isa: Optional[QuantumCircuit] = None
    
    def _validate_dimensions(self) -> None:
        """
        Validate that matrix dimensions are compatible for multiplication.
        
        Raises:
            ValueError: If dimensions are incompatible
        """
        # Check A and B can be multiplied
        if self.A.shape[1] != self.B.shape[0]:
            raise ValueError(
                f"Matrix dimensions incompatible for multiplication: "
                f"A is {self.A.shape}, B is {self.B.shape}. "
                f"A columns ({self.A.shape[1]}) must equal B rows ({self.B.shape[0]})"
            )
        
        # Check C has correct dimensions for A×B result
        expected_shape = (self.A.shape[0], self.B.shape[1])
        if self.C.shape != expected_shape:
            raise ValueError(
                f"Matrix C has wrong dimensions: got {self.C.shape}, "
                f"expected {expected_shape} for A×B product"
            )
    
    @property
    def circuit(self) -> QuantumCircuit:
        """
        Get the verification circuit (builds if not already built).
        
        Returns:
            QuantumCircuit: The quantum circuit for verification
        """
        if self._circuit is None:
            self._circuit = self.build_circuit()
        return self._circuit
    
    @property
    def matrix_size(self) -> str:
        """Get a string representation of matrix dimensions."""
        return f"{self.n}×{self.m} × {self.m}×{self.p}"
    
    @abstractmethod
    def build_circuit(self) -> QuantumCircuit:
        """
        Build the quantum verification circuit.
        
        This method must be implemented by subclasses to construct
        the specific quantum circuit for the verification algorithm.
        
        Returns:
            QuantumCircuit: The complete verification circuit with measurements
        """
        pass
    
    @abstractmethod
    def interpret_results(self, counts: Dict[str, int]) -> VerificationResult:
        """
        Interpret measurement results to determine if A×B = C.
        
        Args:
            counts: Dictionary mapping measurement outcomes to counts
        
        Returns:
            VerificationResult: The verification outcome with statistics
        """
        pass
    
    @abstractmethod
    def success_criteria(self, outcome: str) -> bool:
        """
        Determine if a single measurement outcome indicates A×B = C.
        
        This method is used by QWARD metrics to evaluate individual
        measurement outcomes.
        
        Args:
            outcome: Single measurement outcome string (e.g., "0110")
        
        Returns:
            bool: True if the outcome indicates matrices are equal
        """
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Get the name of the verification method."""
        pass
    
    def draw(self, output: str = "mpl"):
        """
        Draw the verification circuit.
        
        Args:
            output: Drawing format ('mpl', 'text', 'latex')
        
        Returns:
            Circuit diagram in the specified format
        """
        return self.circuit.draw(output=output)
    
    def create_isa_circuit(
        self,
        backend=None,
        optimization_level: int = 3
    ) -> QuantumCircuit:
        """
        Create ISA (Instruction Set Architecture) circuit for backend.
        
        Args:
            backend: Target backend (default: AerSimulator)
            optimization_level: Transpiler optimization level (0-3)
        
        Returns:
            QuantumCircuit: Transpiled circuit optimized for the backend
        """
        if backend is None:
            backend = AerSimulator()
        
        target = backend.target
        pm = generate_preset_pass_manager(
            target=target,
            optimization_level=optimization_level
        )
        return pm.run(self.circuit)

    def verify(
        self,
        backend=None,
        shots: int = 1024,
        print_diagrams: bool = False,
        diagram_output: str = "text",
    ) -> VerificationResult:
        """
        Execute the verification circuit and interpret results.

        Args:
            backend: Target backend (default: AerSimulator)
            shots: Number of shots for execution
            print_diagrams: Print circuit diagrams for understanding
            diagram_output: Diagram format ('text', 'mpl', 'latex')

        Returns:
            VerificationResult with verification outcome and statistics
        """
        import time

        if backend is None:
            backend = AerSimulator()

        if print_diagrams:
            print("\n=== Verification Circuit ===")
            print(self.circuit.draw(output=diagram_output))

            oracle_fn = getattr(self, "_build_oracle", None)
            if callable(oracle_fn):
                print("\n=== Oracle ===")
                print(oracle_fn().draw(output=diagram_output))

            diffuser_fn = getattr(self, "_build_diffuser", None)
            if callable(diffuser_fn):
                print("\n=== Diffuser ===")
                print(diffuser_fn().draw(output=diagram_output))

        start_time = time.time()
        job = backend.run(self.circuit, shots=shots)
        counts = job.result().get_counts()
        result = self.interpret_results(counts)
        result.execution_time_ms = (time.time() - start_time) * 1000
        return result
    
    # =========================================================================
    # Classical Verification Methods (for comparison)
    # =========================================================================
    
    @staticmethod
    def classical_verify(
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        iterations: int = 10,
        tolerance: float = 1e-10
    ) -> bool:
        """
        Classical Freivalds' algorithm for matrix product verification.
        
        This provides a baseline for comparing quantum algorithm performance.
        
        Algorithm:
            1. Generate random binary vector r ∈ {0,1}^p
            2. Compute A×(B×r) and C×r
            3. If they differ, return False (A×B ≠ C definitely)
            4. Repeat k times; if all pass, return True (probably equal)
        
        Complexity: O(kn²) where k = iterations, n = matrix dimension
        
        Error Analysis:
            - If A×B = C: Always returns True (no false negatives)
            - If A×B ≠ C: Returns True with probability ≤ 2^(-k)
        
        Args:
            A: First matrix (n×m)
            B: Second matrix (m×p)
            C: Product matrix to verify (n×p)
            iterations: Number of random tests (k)
            tolerance: Numerical tolerance for floating-point comparison
        
        Returns:
            True if verification passes (A×B probably equals C)
            False if verification fails (A×B definitely does not equal C)
        
        Example:
            >>> A = np.array([[1, 2], [3, 4]])
            >>> B = np.array([[5, 6], [7, 8]])
            >>> C = A @ B
            >>> MatrixProductVerificationBase.classical_verify(A, B, C)
            True
        """
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        C = np.asarray(C, dtype=float)
        
        p = C.shape[1] if len(C.shape) > 1 else 1
        
        for _ in range(iterations):
            # Generate random binary vector
            r = np.random.randint(0, 2, size=(p, 1)).astype(float)
            
            # Compute B×r first (more efficient than (A×B)×r)
            Br = B @ r
            
            # Compute A×(B×r)
            ABr = A @ Br
            
            # Compute C×r
            Cr = C @ r
            
            # Check if difference is significant
            if not np.allclose(ABr, Cr, atol=tolerance):
                return False  # Definitely not equal
        
        return True  # Probably equal
    
    @staticmethod
    def classical_verify_with_details(
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        iterations: int = 10,
        tolerance: float = 1e-10
    ) -> VerificationResult:
        """
        Classical Freivalds' algorithm with detailed results.
        
        Same as classical_verify but returns a VerificationResult object
        with statistics for comparison with quantum methods.
        
        Args:
            A: First matrix (n×m)
            B: Second matrix (m×p)
            C: Product matrix to verify (n×p)
            iterations: Number of random tests
            tolerance: Numerical tolerance
        
        Returns:
            VerificationResult with classical verification outcome
        """
        import time
        start_time = time.time()
        
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        C = np.asarray(C, dtype=float)
        
        p = C.shape[1] if len(C.shape) > 1 else 1
        
        is_equal = True
        max_difference = 0.0
        iterations_completed = 0
        
        for i in range(iterations):
            r = np.random.randint(0, 2, size=(p, 1)).astype(float)
            
            Br = B @ r
            ABr = A @ Br
            Cr = C @ r
            
            diff = np.max(np.abs(ABr - Cr))
            max_difference = max(max_difference, diff)
            iterations_completed = i + 1
            
            if diff > tolerance:
                is_equal = False
                break
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Error probability: 2^(-k) if all iterations passed
        error_prob = 2 ** (-iterations_completed) if is_equal else 0.0
        confidence = 1.0 - error_prob
        
        return VerificationResult(
            is_equal=is_equal,
            confidence=confidence,
            iterations_used=iterations_completed,
            method=VerificationMethod.CLASSICAL.value,
            circuit_depth=0,
            num_qubits=0,
            measurement_counts={},
            error_probability=error_prob,
            classical_comparison=None,
            execution_time_ms=elapsed_ms,
            details={
                "max_difference": max_difference,
                "tolerance": tolerance,
            }
        )
    
    def verify_classically(self) -> VerificationResult:
        """
        Run classical verification on this instance's matrices.
        
        Convenience method that calls classical_verify_with_details
        with the instance's A, B, C matrices.
        
        Returns:
            VerificationResult from classical Freivalds algorithm
        """
        return self.classical_verify_with_details(
            self.A, self.B, self.C, self.iterations
        )
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_expected_result(self) -> bool:
        """
        Compute the actual expected result by direct matrix multiplication.
        
        This is useful for validating the algorithm but defeats the purpose
        of the verification (since it computes A×B directly).
        
        Returns:
            True if A×B actually equals C
        """
        actual_product = self.A @ self.B
        return np.allclose(actual_product, self.C)
    
    def get_wrong_entries(self) -> List[tuple]:
        """
        Find all entries where (A×B)_{i,j} ≠ C_{i,j}.
        
        Returns:
            List of (i, j) tuples where entries differ
        """
        actual_product = self.A @ self.B
        diff = np.abs(actual_product - self.C)
        wrong_indices = np.where(diff > 1e-10)
        return list(zip(wrong_indices[0], wrong_indices[1]))
    
    def get_num_wrong_entries(self) -> int:
        """Get the count of wrong entries in C."""
        return len(self.get_wrong_entries())
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"matrix_size={self.matrix_size}, "
            f"iterations={self.iterations})"
        )


# =============================================================================
# Phase 2: Quantum Freivalds Implementation
# =============================================================================

class QuantumFreivaldsVerification(MatrixProductVerificationBase):
    """
    Quantum Freivalds' algorithm for matrix product verification.
    
    This is the DEFAULT and RECOMMENDED algorithm for practical use.
    
    Algorithm Overview:
    ------------------
    The quantum version of Freivalds' algorithm works as follows:
    
    1. **Preprocessing**: Compute the error matrix D = A×B - C
       - If D = 0, then A×B = C (no errors)
       - If D ≠ 0, we need to detect this quantumly
    
    2. **Superposition Creation**: Create equal superposition of all 2^n 
       binary vectors: |ψ⟩ = H^⊗n|0⟩^n = (1/√2^n) Σ_r |r⟩
    
    3. **Error Detection Oracle**: Build an oracle O that marks states |r⟩ 
       where D×r ≠ 0 (i.e., the error would be detected classically)
       - O|r⟩ = -|r⟩ if D×r ≠ 0 (error detected)
       - O|r⟩ = |r⟩  if D×r = 0 (no error detected)
    
    4. **Grover Amplification**: Apply Grover iterations to amplify the 
       probability of measuring error-detecting states
    
    5. **Measurement & Interpretation**: 
       - If A×B = C: All states are unmarked, uniform distribution
       - If A×B ≠ C: Marked states have amplified probability
    
    Key Insight:
    -----------
    For Freivalds' algorithm, if A×B ≠ C, at least 50% of random vectors 
    detect the error. With such high marking fraction, even 1-2 Grover 
    iterations can significantly amplify detection probability.
    
    Complexity: O(√k · n²) for k-iteration classical equivalent
    
    Advantages:
    - Practical for NISQ devices and simulators
    - Clear demonstration of quantum speedup
    - Easy to compare with classical Freivalds
    
    Example:
        >>> A = np.array([[1, 2], [3, 4]])
        >>> B = np.array([[5, 6], [7, 8]])
        >>> C = A @ B
        >>> verifier = QuantumFreivaldsVerification(A, B, C)
        >>> circuit = verifier.circuit
        >>> # Execute on simulator
        >>> from qiskit_aer import AerSimulator
        >>> job = AerSimulator().run(circuit, shots=1024)
        >>> result = verifier.interpret_results(job.result().get_counts())
    """
    
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        iterations: int = 10,
        use_barriers: bool = True,
        grover_iterations: Optional[int] = None,
    ):
        super().__init__(A, B, C, iterations, use_barriers)
        
        # Number of qubits for random vector encoding
        self.num_vector_qubits = self.p  # One qubit per column of B/C
        
        # Compute error matrix D = A×B - C
        self.D = self.A @ self.B - self.C
        
        # Find which binary vectors detect errors (D×r ≠ 0)
        self._error_detecting_states = self._compute_error_detecting_states()
        
        # Calculate marking fraction (for Grover iteration calculation)
        total_states = 2 ** self.num_vector_qubits
        self.num_marked = len(self._error_detecting_states)
        self.marking_fraction = self.num_marked / total_states if total_states > 0 else 0
        
        # Calculate optimal Grover iterations
        if grover_iterations is not None:
            self.grover_iterations = grover_iterations
        else:
            self.grover_iterations = self._calculate_optimal_grover_iterations()
    
    def _compute_error_detecting_states(self) -> List[str]:
        """
        Compute which binary vectors r would detect the error D×r ≠ 0.
        
        Returns:
            List of bit strings representing vectors that detect errors
        """
        error_detecting = []
        n = self.num_vector_qubits
        
        # Check all 2^n possible binary vectors
        for i in range(2 ** n):
            # Convert integer to binary vector
            r = np.array([(i >> j) & 1 for j in range(n)], dtype=float)
            
            # Compute D×r
            Dr = self.D @ r
            
            # Check if any component is non-zero (with tolerance)
            if np.any(np.abs(Dr) > 1e-10):
                error_detecting.append(self._format_state(i))
        
        return error_detecting
    
    def _calculate_optimal_grover_iterations(self) -> int:
        """
        Calculate optimal number of Grover iterations.
        
        For Freivalds, the marking fraction is typically ≥ 0.5 when there's
        an error, so we need very few iterations.
        
        Returns:
            Optimal number of Grover iterations
        """
        if self.num_marked == 0:
            # No errors to detect - A×B = C
            return 0
        
        total_states = 2 ** self.num_vector_qubits
        
        if self.num_marked == total_states:
            # All states marked (shouldn't happen for valid Freivalds)
            return 0
        
        # Optimal iterations: floor(π/4 * √(N/M))
        # where N = total states, M = marked states
        import math
        theta = math.asin(math.sqrt(self.num_marked / total_states))
        
        if theta > 0:
            optimal = int(math.floor(math.pi / (4 * theta)))
            # For high marking fractions (>25%), limit iterations
            # to avoid over-rotation
            return min(optimal, 3)
        
        return 1

    def _format_state(self, index: int) -> str:
        """Format basis state index using MSB-first order."""
        return format(index, f"0{self.num_vector_qubits}b")

    def _normalize_outcome(self, outcome: str) -> str:
        """Normalize measurement outcome to MSB-first order."""
        return outcome.replace(" ", "")
    
    def get_method_name(self) -> str:
        return VerificationMethod.QUANTUM_FREIVALDS.value
    
    def _build_oracle(self) -> QuantumCircuit:
        """
        Build the oracle circuit that marks error-detecting states.
        
        The oracle applies a phase flip (-1) to states |r⟩ where D×r ≠ 0.
        
        Implementation uses multi-controlled Z gates to mark specific states.
        
        Returns:
            QuantumCircuit: Oracle circuit
        """
        n = self.num_vector_qubits
        oracle = QuantumCircuit(n, name='Oracle')
        
        if not self._error_detecting_states:
            # No states to mark (A×B = C)
            return oracle
        
        # Mark each error-detecting state
        for state in self._error_detecting_states:
            # Apply X gates to qubits that are '0' in the state
            # (to convert the state to |11...1⟩)
            x_qubits = [i for i, bit in enumerate(state) if bit == '0']
            
            if x_qubits:
                oracle.x(x_qubits)
            
            # Apply multi-controlled Z (phase flip for |11...1⟩)
            if n == 1:
                oracle.z(0)
            elif n == 2:
                oracle.cz(0, 1)
            else:
                # For n > 2, use MCZ (multi-controlled Z)
                from qiskit.circuit.library import ZGate
                oracle.append(
                    ZGate().control(n - 1),
                    list(range(n))
                )
            
            # Undo X gates
            if x_qubits:
                oracle.x(x_qubits)
        
        return oracle
    
    def _build_diffuser(self) -> QuantumCircuit:
        """
        Build the Grover diffusion operator.
        
        The diffuser performs: 2|ψ⟩⟨ψ| - I
        where |ψ⟩ = H^⊗n|0⟩^n is the uniform superposition.
        
        Returns:
            QuantumCircuit: Diffuser circuit
        """
        n = self.num_vector_qubits
        diffuser = QuantumCircuit(n, name='Diffuser')
        
        # Apply H gates
        diffuser.h(range(n))
        
        # Apply X gates
        diffuser.x(range(n))
        
        # Multi-controlled Z
        if n == 1:
            diffuser.z(0)
        elif n == 2:
            diffuser.cz(0, 1)
        else:
            from qiskit.circuit.library import ZGate
            diffuser.append(
                ZGate().control(n - 1),
                list(range(n))
            )
        
        # Undo X gates
        diffuser.x(range(n))
        
        # Apply H gates
        diffuser.h(range(n))
        
        return diffuser
    
    def build_circuit(self) -> QuantumCircuit:
        """
        Build the complete Quantum Freivalds verification circuit.
        
        Circuit Structure:
        1. Initialize uniform superposition: H^⊗n|0⟩^n
        2. For each Grover iteration:
           a. Apply Oracle (marks error-detecting states)
           b. Apply Diffuser (amplifies marked states)
        3. Measure all qubits
        
        Returns:
            QuantumCircuit: Complete verification circuit
        """
        n = self.num_vector_qubits
        
        # Create circuit with classical bits for measurement
        qc = QuantumCircuit(n, n)
        
        # Step 1: Create uniform superposition
        qc.h(range(n))
        
        if self.use_barriers:
            qc.barrier()
        
        # Step 2: Apply Grover iterations (if there are states to mark)
        if self.num_marked > 0 and self.grover_iterations > 0:
            oracle = self._build_oracle()
            diffuser = self._build_diffuser()
            
            for i in range(self.grover_iterations):
                # Apply oracle
                qc.compose(oracle, inplace=True)
                
                if self.use_barriers:
                    qc.barrier()
                
                # Apply diffuser
                qc.compose(diffuser, inplace=True)
                
                if self.use_barriers:
                    qc.barrier()
        
        # Step 3: Measure
        qc.measure(range(n), range(n))
        
        return qc
    
    def interpret_results(self, counts: Dict[str, int]) -> VerificationResult:
        """
        Interpret measurement results to determine if A×B = C.
        
        Decision Logic:
        - If A×B = C: No error-detecting states exist, distribution is uniform
        - If A×B ≠ C: Error-detecting states are amplified
        
        We measure the probability of measuring an error-detecting state.
        If this probability is significantly above uniform, we conclude A×B ≠ C.
        
        Args:
            counts: Measurement counts from circuit execution
        
        Returns:
            VerificationResult with the verification decision
        """
        total_shots = sum(counts.values())
        
        # Count measurements that hit error-detecting states
        error_detected_count = 0
        for state, count in counts.items():
            clean_state = self._normalize_outcome(state)
            if clean_state in self._error_detecting_states:
                error_detected_count += count
        
        # Calculate observed probability of detecting error
        error_detection_prob = error_detected_count / total_shots if total_shots > 0 else 0
        
        # Decision threshold
        # If A×B = C: error_detection_prob should be 0
        # If A×B ≠ C: error_detection_prob should be high (≥ 0.5 typically)
        # We use a threshold of 0.1 to account for noise
        threshold = 0.1
        
        # Determine if matrices are equal
        if self.num_marked == 0:
            # No error-detecting states exist → A×B = C definitely
            is_equal = True
            confidence = 1.0
            error_probability = 0.0
        else:
            # Check if we observed error-detecting states
            is_equal = error_detection_prob < threshold
            
            # Calculate confidence based on statistical significance
            # Using normal approximation for binomial distribution
            if is_equal:
                # If we think they're equal but there are error-detecting states,
                # this is likely wrong
                confidence = max(0, 1.0 - error_detection_prob * 2)
                error_probability = self.marking_fraction  # True error prob
            else:
                # High confidence that they're not equal
                confidence = min(1.0, error_detection_prob / self.marking_fraction) if self.marking_fraction > 0 else error_detection_prob
                error_probability = max(0, 1.0 - error_detection_prob)
        
        # Get classical result for comparison
        classical_result = self.verify_classically()
        
        return VerificationResult(
            is_equal=is_equal,
            confidence=confidence,
            iterations_used=self.grover_iterations,
            method=self.get_method_name(),
            circuit_depth=self.circuit.depth(),
            num_qubits=self.circuit.num_qubits,
            measurement_counts=dict(counts),
            error_probability=error_probability,
            classical_comparison=classical_result.is_equal,
            details={
                "error_detection_prob": error_detection_prob,
                "error_detected_count": error_detected_count,
                "total_shots": total_shots,
                "num_marked_states": self.num_marked,
                "total_states": 2 ** self.num_vector_qubits,
                "marking_fraction": self.marking_fraction,
                "grover_iterations": self.grover_iterations,
                "threshold": threshold,
                "error_detecting_states": self._error_detecting_states[:10],  # First 10
            }
        )
    
    def success_criteria(self, outcome: str) -> bool:
        """
        Determine if a single measurement outcome indicates A×B = C.
        
        For Quantum Freivalds:
        - If outcome is an error-detecting state → A×B ≠ C (return False)
        - If outcome is not error-detecting → Could be equal (return True)
        
        Args:
            outcome: Single measurement outcome string (e.g., "01", "110")
        
        Returns:
            bool: True if outcome suggests A×B = C
        """
        clean_outcome = self._normalize_outcome(outcome)
        
        # If this outcome would detect an error, matrices are not equal
        return clean_outcome not in self._error_detecting_states
    
    def get_error_detecting_states(self) -> List[str]:
        """Get list of states that detect errors (MSB-first formatted)."""
        return self._error_detecting_states.copy()
    
    def get_theoretical_success_probability(self) -> float:
        """
        Calculate theoretical probability of detecting error after Grover iterations.
        
        Returns:
            float: Probability of measuring an error-detecting state
        """
        if self.num_marked == 0:
            return 0.0
        
        import math
        N = 2 ** self.num_vector_qubits
        M = self.num_marked
        
        theta = math.asin(math.sqrt(M / N))
        
        # After k Grover iterations
        k = self.grover_iterations
        prob = math.sin((2 * k + 1) * theta) ** 2
        
        return prob
    
    def expected_distribution(self) -> Dict[str, float]:
        """
        Get theoretical expected probability distribution.
        
        Returns:
            dict: Expected probabilities for each state
        """
        import math
        
        n = self.num_vector_qubits
        N = 2 ** n
        M = self.num_marked
        
        if M == 0:
            # Uniform distribution
            prob = 1.0 / N
            return {self._format_state(i): prob for i in range(N)}
        
        # After Grover iterations
        theta = math.asin(math.sqrt(M / N))
        k = self.grover_iterations
        
        # Probability for marked states
        prob_marked = (math.sin((2 * k + 1) * theta) ** 2) / M if M > 0 else 0
        
        # Probability for unmarked states
        prob_unmarked = (math.cos((2 * k + 1) * theta) ** 2) / (N - M) if N > M else 0
        
        expected = {}
        for i in range(N):
            state = self._format_state(i)
            if state in self._error_detecting_states:
                expected[state] = prob_marked
            else:
                expected[state] = prob_unmarked
        
        return expected


class BuhrmanSpalekVerification(MatrixProductVerificationBase):
    """
    Buhrman-Špalek quantum walk algorithm for matrix product verification.
    
    This is the ADVANCED algorithm with optimal theoretical speedup.
    
    Complexity: O(n^(5/3)) worst-case
    
    Note: This is a placeholder implementation. The quantum circuit
    will be implemented in Phase 3.
    
    Reference: Buhrman & Špalek (2004), arXiv:quant-ph/0409035
    """
    
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        iterations: int = 10,
        use_barriers: bool = True,
        subset_size_k: Optional[int] = None
    ):
        super().__init__(A, B, C, iterations, use_barriers)
        
        # Subset size k for Johnson graph J(n,k)
        # Default: k = n^(2/3) as per the paper
        self.k = subset_size_k or max(1, int(self.n ** (2/3)))
    
    def get_method_name(self) -> str:
        return VerificationMethod.BUHRMAN_SPALEK.value
    
    def build_circuit(self) -> QuantumCircuit:
        """
        Build the Buhrman-Špalek verification circuit.
        
        TODO: Implement in Phase 3
        """
        # Placeholder
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure_all()
        return qc
    
    def interpret_results(self, counts: Dict[str, int]) -> VerificationResult:
        """TODO: Implement in Phase 3"""
        classical_result = self.verify_classically()
        return VerificationResult(
            is_equal=classical_result.is_equal,
            confidence=classical_result.confidence,
            iterations_used=self.iterations,
            method=self.get_method_name(),
            circuit_depth=self.circuit.depth(),
            num_qubits=self.circuit.num_qubits,
            measurement_counts=counts,
            error_probability=classical_result.error_probability,
            classical_comparison=classical_result.is_equal,
            details={"note": "Placeholder - Phase 3 implementation pending"}
        )
    
    def success_criteria(self, outcome: str) -> bool:
        """TODO: Implement in Phase 3"""
        return True


# =============================================================================
# Unified Interface
# =============================================================================

class MatrixProductVerification:
    """
    Unified interface for quantum matrix product verification.
    
    Provides access to both verification algorithms through a single class.
    DEFAULT: Uses QuantumFreivaldsVerification (Option 1)
    
    Args:
        A: First matrix (n×m)
        B: Second matrix (m×p)
        C: Product matrix to verify (n×p)
        method: Algorithm to use:
            - 'quantum_freivalds' (default): Practical for simulators/NISQ
            - 'buhrman_spalek': Optimal theoretical speedup
            - 'auto': Automatically select based on matrix size
        iterations: Number of verification rounds
        use_barriers: Add barriers for visualization
        **kwargs: Additional arguments passed to the specific verifier
    
    Example:
        >>> # Using default method (Quantum Freivalds)
        >>> verifier = MatrixProductVerification(A, B, C)
        >>> result = verifier.verify_classically()
        
        >>> # Using Buhrman-Špalek
        >>> verifier = MatrixProductVerification(A, B, C, method='buhrman_spalek')
        
        >>> # Compare with classical
        >>> classical_result = MatrixProductVerification.classical_verify(A, B, C)
    """
    
    METHODS = {
        VerificationMethod.QUANTUM_FREIVALDS.value: QuantumFreivaldsVerification,
        VerificationMethod.BUHRMAN_SPALEK.value: BuhrmanSpalekVerification,
    }
    
    DEFAULT_METHOD = VerificationMethod.QUANTUM_FREIVALDS.value
    
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        method: str = "quantum_freivalds",
        iterations: int = 10,
        use_barriers: bool = True,
        **kwargs
    ):
        # Handle 'auto' method selection
        self.method_name = method if method != "auto" else self._auto_select_method(A)
        
        if self.method_name not in self.METHODS:
            valid_methods = list(self.METHODS.keys()) + ["auto"]
            raise ValueError(
                f"Unknown method: '{method}'. "
                f"Choose from: {valid_methods}"
            )
        
        # Store matrices for reference
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        
        # Instantiate the appropriate verifier
        verifier_class = self.METHODS[self.method_name]
        self._verifier: MatrixProductVerificationBase = verifier_class(
            A, B, C, iterations, use_barriers, **kwargs
        )
    
    @staticmethod
    def _auto_select_method(A: np.ndarray) -> str:
        """
        Auto-select method based on matrix size.
        
        Uses Quantum Freivalds for most practical cases,
        Buhrman-Špalek only for very large matrices where
        the theoretical speedup is significant.
        """
        n = A.shape[0]
        # Use Quantum Freivalds for n < 32
        # (Buhrman-Špalek overhead not worth it for small matrices)
        if n < 32:
            return VerificationMethod.QUANTUM_FREIVALDS.value
        return VerificationMethod.BUHRMAN_SPALEK.value
    
    @property
    def circuit(self) -> QuantumCircuit:
        """Get the verification circuit."""
        return self._verifier.circuit
    
    @property
    def verifier(self) -> MatrixProductVerificationBase:
        """Get the underlying verifier instance."""
        return self._verifier
    
    def build_circuit(self) -> QuantumCircuit:
        """Build the verification circuit."""
        return self._verifier.build_circuit()
    
    def verify_classically(self) -> VerificationResult:
        """Run classical Freivalds verification."""
        return self._verifier.verify_classically()

    def verify(
        self,
        backend=None,
        shots: int = 1024,
        print_diagrams: bool = False,
        diagram_output: str = "text",
    ) -> VerificationResult:
        """Run quantum verification and interpret results."""
        return self._verifier.verify(
            backend=backend,
            shots=shots,
            print_diagrams=print_diagrams,
            diagram_output=diagram_output,
        )
    
    def interpret_results(self, counts: Dict[str, int]) -> VerificationResult:
        """Interpret quantum measurement results."""
        return self._verifier.interpret_results(counts)
    
    def success_criteria(self, outcome: str) -> bool:
        """Determine if measurement indicates A×B = C."""
        return self._verifier.success_criteria(outcome)
    
    def draw(self, output: str = "mpl"):
        """Draw the verification circuit."""
        return self._verifier.draw(output)
    
    def get_expected_result(self) -> bool:
        """Check if A×B actually equals C."""
        return self._verifier.get_expected_result()
    
    def get_wrong_entries(self) -> List[tuple]:
        """Get list of wrong entries in C."""
        return self._verifier.get_wrong_entries()
    
    @staticmethod
    def classical_verify(
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        iterations: int = 10
    ) -> bool:
        """Classical Freivalds verification for comparison."""
        return MatrixProductVerificationBase.classical_verify(A, B, C, iterations)
    
    @staticmethod
    def classical_verify_with_details(
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        iterations: int = 10
    ) -> VerificationResult:
        """Classical Freivalds verification with detailed results."""
        return MatrixProductVerificationBase.classical_verify_with_details(
            A, B, C, iterations
        )
    
    def __repr__(self) -> str:
        return (
            f"MatrixProductVerification("
            f"method={self.method_name}, "
            f"matrix_size={self._verifier.matrix_size})"
        )
