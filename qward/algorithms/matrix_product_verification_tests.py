"""
Matrix Product Verification - Test Cases and Classical Reference Implementation

This module provides:
1. Simple test cases for verifying the algorithm
2. Classical Freivalds' algorithm implementation for comparison
3. Utility functions for matrix verification

These test cases serve as the foundation for validating the quantum implementation.
"""

# Matrix verification examples intentionally use A, B, C notation.
# pylint: disable=invalid-name

import numpy as np
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass


@dataclass
class VerificationTestCase:
    """A test case for matrix product verification."""

    name: str
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    expected_result: bool  # True if A×B = C
    description: str


# =============================================================================
# Classical Freivalds' Algorithm
# =============================================================================


def freivalds_verify(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, iterations: int = 10, tolerance: float = 1e-10
) -> bool:
    """
    Freivalds' probabilistic algorithm for matrix product verification.

    Verifies if A × B = C in O(kn²) time where k is the number of iterations.

    Algorithm:
        1. Generate random binary vector r ∈ {0,1}^n
        2. Compute A×(B×r) and C×r
        3. If they differ, A×B ≠ C (definitely)
        4. Repeat k times; if all pass, A×B = C (with high probability)

    Error Analysis:
        - If A×B = C: Always returns True (no false negatives)
        - If A×B ≠ C: Returns True with probability ≤ 2^(-k)

    Args:
        A: First matrix (n×m)
        B: Second matrix (m×p)
        C: Product matrix to verify (n×p)
        iterations: Number of random tests (k)
        tolerance: Numerical tolerance for comparison

    Returns:
        True if verification passes (A×B probably equals C)
        False if verification fails (A×B definitely does not equal C)

    Example:
        >>> A = np.array([[1, 2], [3, 4]])
        >>> B = np.array([[5, 6], [7, 8]])
        >>> C = A @ B  # Correct product
        >>> freivalds_verify(A, B, C)
        True
    """
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


def freivalds_verify_with_details(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, iterations: int = 10
) -> Tuple[bool, Dict[str, Any]]:
    """
    Freivalds' algorithm with detailed statistics.

    Returns verification result and statistics about the verification process.
    """
    p = C.shape[1] if len(C.shape) > 1 else 1

    stats: Dict[str, Any] = {
        "iterations": iterations,
        "failures": 0,
        "max_difference": 0.0,
        "random_vectors_used": [],
    }

    for _ in range(iterations):
        r = np.random.randint(0, 2, size=(p, 1)).astype(float)
        stats["random_vectors_used"].append(r.flatten().tolist())

        Br = B @ r
        ABr = A @ Br
        Cr = C @ r

        diff = np.max(np.abs(ABr - Cr))
        stats["max_difference"] = max(stats["max_difference"], diff)

        if diff > 1e-10:
            stats["failures"] += 1
            return False, stats

    return True, stats


# =============================================================================
# Test Cases
# =============================================================================


def create_test_cases() -> List[VerificationTestCase]:
    """Create a comprehensive list of test cases."""

    test_cases = []

    # Test Case 1: Simple 2×2 correct product
    A1 = np.array([[1, 2], [3, 4]], dtype=float)
    B1 = np.array([[5, 6], [7, 8]], dtype=float)
    C1 = A1 @ B1  # [[19, 22], [43, 50]]

    test_cases.append(
        VerificationTestCase(
            name="2x2_correct_product",
            A=A1,
            B=B1,
            C=C1,
            expected_result=True,
            description="Simple 2×2 matrices with correct product",
        )
    )

    # Test Case 2: 2×2 with wrong product (one entry error)
    C2_wrong = np.array([[19, 22], [43, 51]], dtype=float)  # 51 should be 50

    test_cases.append(
        VerificationTestCase(
            name="2x2_wrong_product_single_error",
            A=A1,
            B=B1,
            C=C2_wrong,
            expected_result=False,
            description="2×2 matrices with one wrong entry (51 instead of 50)",
        )
    )

    # Test Case 3: Identity matrix multiplication
    A3 = np.array([[1, 0, 2], [0, 1, 0], [2, 0, 1]], dtype=float)
    I3 = np.eye(3)
    C3 = A3.copy()  # A × I = A

    test_cases.append(
        VerificationTestCase(
            name="3x3_identity_multiplication",
            A=A3,
            B=I3,
            C=C3,
            expected_result=True,
            description="3×3 matrix multiplied by identity",
        )
    )

    # Test Case 4: Zero matrix
    A4 = np.array([[1, 2], [3, 4]], dtype=float)
    B4 = np.zeros((2, 2))
    C4 = np.zeros((2, 2))

    test_cases.append(
        VerificationTestCase(
            name="2x2_zero_product",
            A=A4,
            B=B4,
            C=C4,
            expected_result=True,
            description="Matrix multiplied by zero matrix",
        )
    )

    # Test Case 5: Larger 4×4 correct product
    A5 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=float)
    B5 = np.array([[16, 15, 14, 13], [12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]], dtype=float)
    C5 = A5 @ B5

    test_cases.append(
        VerificationTestCase(
            name="4x4_correct_product",
            A=A5,
            B=B5,
            C=C5,
            expected_result=True,
            description="4×4 matrices with correct product",
        )
    )

    # Test Case 6: 4×4 wrong product (multiple errors)
    C6_wrong = C5.copy()
    C6_wrong[0, 0] += 1  # Add error at (0,0)
    C6_wrong[2, 3] -= 2  # Add error at (2,3)

    test_cases.append(
        VerificationTestCase(
            name="4x4_wrong_product_multiple_errors",
            A=A5,
            B=B5,
            C=C6_wrong,
            expected_result=False,
            description="4×4 matrices with two wrong entries",
        )
    )

    # Test Case 7: Random 8×8 matrices
    np.random.seed(42)  # For reproducibility
    A7 = np.random.randn(8, 8)
    B7 = np.random.randn(8, 8)
    C7 = A7 @ B7

    test_cases.append(
        VerificationTestCase(
            name="8x8_random_correct",
            A=A7,
            B=B7,
            C=C7,
            expected_result=True,
            description="Random 8×8 matrices with correct product",
        )
    )

    # Test Case 8: Non-square matrices (2×3) × (3×2) = (2×2)
    A8 = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    B8 = np.array([[7, 8], [9, 10], [11, 12]], dtype=float)
    C8 = A8 @ B8  # [[58, 64], [139, 154]]

    test_cases.append(
        VerificationTestCase(
            name="2x3_3x2_correct_product",
            A=A8,
            B=B8,
            C=C8,
            expected_result=True,
            description="Non-square matrices (2×3) × (3×2) with correct product",
        )
    )

    return test_cases


# =============================================================================
# Test Runner
# =============================================================================


def run_all_tests(verbose: bool = True) -> Dict[str, Any]:
    """
    Run all test cases and report results.

    Args:
        verbose: Print detailed output for each test

    Returns:
        Dictionary with test results summary
    """
    test_cases = create_test_cases()
    results: Dict[str, Any] = {"total": len(test_cases), "passed": 0, "failed": 0, "details": []}

    if verbose:
        print("=" * 60)
        print("Matrix Product Verification - Test Suite")
        print("=" * 60)
        print()

    for tc in test_cases:
        actual_result = freivalds_verify(tc.A, tc.B, tc.C, iterations=20)
        passed = actual_result == tc.expected_result

        if passed:
            results["passed"] += 1
            status = "✓ PASS"
        else:
            results["failed"] += 1
            status = "✗ FAIL"

        results["details"].append(
            {
                "name": tc.name,
                "passed": passed,
                "expected": tc.expected_result,
                "actual": actual_result,
            }
        )

        if verbose:
            print(f"{status} | {tc.name}")
            print(f"       Description: {tc.description}")
            print(f"       Expected: {'EQUAL' if tc.expected_result else 'NOT EQUAL'}")
            print(f"       Actual:   {'EQUAL' if actual_result else 'NOT EQUAL'}")
            print()

    if verbose:
        print("=" * 60)
        print(f"Results: {results['passed']}/{results['total']} tests passed")
        if results["failed"] > 0:
            print(f"WARNING: {results['failed']} tests failed!")
        print("=" * 60)

    return results


def demonstrate_freivalds_algorithm():
    """
    Interactive demonstration of Freivalds' algorithm.

    Shows step-by-step how the classical algorithm works.
    """
    print("\n" + "=" * 60)
    print("Demonstration: Freivalds' Algorithm Step-by-Step")
    print("=" * 60)

    # Simple example
    A = np.array([[1, 2], [3, 4]], dtype=float)
    B = np.array([[5, 6], [7, 8]], dtype=float)
    C_correct = A @ B
    C_wrong = np.array([[19, 22], [43, 51]], dtype=float)  # Wrong!

    print("\nMatrices:")
    print(f"A = \n{A}")
    print(f"\nB = \n{B}")
    print(f"\nCorrect C = A×B = \n{C_correct}")
    print(f"\nWrong C (for testing) = \n{C_wrong}")

    print("\n" + "-" * 40)
    print("Testing CORRECT product (A×B = C_correct):")
    print("-" * 40)

    for i in range(3):
        r = np.random.randint(0, 2, size=(2, 1)).astype(float)
        Br = B @ r
        ABr = A @ Br
        Cr = C_correct @ r

        print(f"\nIteration {i+1}:")
        print(f"  Random vector r = {r.flatten()}")
        print(f"  B×r = {Br.flatten()}")
        print(f"  A×(B×r) = {ABr.flatten()}")
        print(f"  C×r = {Cr.flatten()}")
        print(f"  Difference = {(ABr - Cr).flatten()}")
        print(f"  Match? {'YES ✓' if np.allclose(ABr, Cr) else 'NO ✗'}")

    print("\n" + "-" * 40)
    print("Testing WRONG product (A×B ≠ C_wrong):")
    print("-" * 40)

    detected = False
    for i in range(10):
        r = np.random.randint(0, 2, size=(2, 1)).astype(float)
        Br = B @ r
        ABr = A @ Br
        Cr = C_wrong @ r

        if not np.allclose(ABr, Cr):
            print(f"\nIteration {i+1}: ERROR DETECTED!")
            print(f"  Random vector r = {r.flatten()}")
            print(f"  A×(B×r) = {ABr.flatten()}")
            print(f"  C_wrong×r = {Cr.flatten()}")
            print(f"  Difference = {(ABr - Cr).flatten()} ≠ 0")
            detected = True
            break
        print(f"\nIteration {i+1}: No error detected yet (lucky vector)")

    if not detected:
        print("\nNote: Error not detected in 10 iterations (very rare!)")

    print("\n" + "=" * 60)


# =============================================================================
# Tests for New Class Structure (Phase 1 - Task 1.3)
# =============================================================================


def test_base_class_structure():
    """Test the MatrixProductVerificationBase class structure."""
    from qward.algorithms import (
        MatrixProductVerification,
        QuantumFreivaldsVerification,
        BuhrmanSpalekVerification,
        VerificationResult,
        VerificationMethod,
    )

    print("\n" + "=" * 60)
    print("Testing Base Class Structure (Phase 1 - Task 1.3)")
    print("=" * 60)

    # Test matrices
    A = np.array([[1, 2], [3, 4]], dtype=float)
    B = np.array([[5, 6], [7, 8]], dtype=float)
    C_correct = A @ B
    C_wrong = np.array([[19, 22], [43, 51]], dtype=float)

    tests_passed = 0
    tests_total = 0

    # Test 1: QuantumFreivaldsVerification instantiation
    tests_total += 1
    try:
        verifier = QuantumFreivaldsVerification(A, B, C_correct)
        assert verifier.n == 2
        assert verifier.m == 2
        assert verifier.p == 2
        assert verifier.get_method_name() == "quantum_freivalds"
        print("✓ QuantumFreivaldsVerification instantiation")
        tests_passed += 1
    except Exception as e:
        print(f"✗ QuantumFreivaldsVerification instantiation: {e}")

    # Test 2: BuhrmanSpalekVerification instantiation
    tests_total += 1
    try:
        verifier = BuhrmanSpalekVerification(A, B, C_correct)
        assert verifier.get_method_name() == "buhrman_spalek"
        print("✓ BuhrmanSpalekVerification instantiation")
        tests_passed += 1
    except Exception as e:
        print(f"✗ BuhrmanSpalekVerification instantiation: {e}")

    # Test 3: MatrixProductVerification unified interface (default method)
    tests_total += 1
    try:
        verifier = MatrixProductVerification(A, B, C_correct)
        assert verifier.method_name == "quantum_freivalds"
        print("✓ MatrixProductVerification default method is quantum_freivalds")
        tests_passed += 1
    except Exception as e:
        print(f"✗ MatrixProductVerification default method: {e}")

    # Test 4: MatrixProductVerification with explicit method
    tests_total += 1
    try:
        verifier = MatrixProductVerification(A, B, C_correct, method="buhrman_spalek")
        assert verifier.method_name == "buhrman_spalek"
        print("✓ MatrixProductVerification explicit method selection")
        tests_passed += 1
    except Exception as e:
        print(f"✗ MatrixProductVerification explicit method: {e}")

    # Test 5: Classical verification via base class
    tests_total += 1
    try:
        result = MatrixProductVerification.classical_verify(A, B, C_correct)
        assert result is True
        print("✓ Classical verification (correct product) = True")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Classical verification (correct): {e}")

    # Test 6: Classical verification with wrong product
    tests_total += 1
    try:
        result = MatrixProductVerification.classical_verify(A, B, C_wrong)
        assert result is False
        print("✓ Classical verification (wrong product) = False")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Classical verification (wrong): {e}")

    # Test 7: Classical verification with details
    tests_total += 1
    try:
        result = MatrixProductVerification.classical_verify_with_details(A, B, C_correct)
        assert isinstance(result, VerificationResult)
        assert result.is_equal is True
        assert result.method == "classical"
        assert result.confidence > 0.99
        print(f"✓ Classical verification with details: {result}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Classical verification with details: {e}")

    # Test 8: verify_classically method
    tests_total += 1
    try:
        verifier = MatrixProductVerification(A, B, C_correct)
        result = verifier.verify_classically()
        assert result.is_equal is True
        print(f"✓ verifier.verify_classically(): {result}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ verify_classically: {e}")

    # Test 9: get_expected_result
    tests_total += 1
    try:
        verifier = MatrixProductVerification(A, B, C_correct)
        assert verifier.get_expected_result() is True

        verifier_wrong = MatrixProductVerification(A, B, C_wrong)
        assert verifier_wrong.get_expected_result() is False
        print("✓ get_expected_result() works correctly")
        tests_passed += 1
    except Exception as e:
        print(f"✗ get_expected_result: {e}")

    # Test 10: get_wrong_entries
    tests_total += 1
    try:
        verifier = MatrixProductVerification(A, B, C_wrong)
        wrong_entries = verifier.get_wrong_entries()
        assert len(wrong_entries) == 1
        assert wrong_entries[0] == (1, 1)  # Entry at row 1, col 1 is wrong
        print(f"✓ get_wrong_entries(): {wrong_entries}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ get_wrong_entries: {e}")

    # Test 11: Dimension validation
    tests_total += 1
    try:
        # This should raise ValueError
        A_bad = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
        B_bad = np.array([[1, 2], [3, 4]])  # 2x2 - incompatible!
        C_bad = np.array([[1, 2], [3, 4]])

        try:
            MatrixProductVerification(A_bad, B_bad, C_bad)
            print("✗ Dimension validation: Should have raised ValueError")
        except ValueError as ve:
            print(f"✓ Dimension validation raises ValueError: {str(ve)[:50]}...")
            tests_passed += 1
    except Exception as e:
        print(f"✗ Dimension validation: {e}")

    # Test 12: Circuit building (placeholder)
    tests_total += 1
    try:
        verifier = MatrixProductVerification(A, B, C_correct)
        circuit = verifier.circuit
        assert circuit is not None
        assert circuit.num_qubits > 0
        print(f"✓ Circuit built: {circuit.num_qubits} qubits, depth {circuit.depth()}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Circuit building: {e}")

    # Test 13: VerificationResult dataclass
    tests_total += 1
    try:
        result = VerificationResult(
            is_equal=True, confidence=0.999, iterations_used=10, method="test"
        )
        result_dict = result.to_dict()
        assert "is_equal" in result_dict
        assert result_dict["confidence"] == 0.999
        print(f"✓ VerificationResult dataclass: {result}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ VerificationResult: {e}")

    # Test 14: Auto method selection
    tests_total += 1
    try:
        # Small matrix should select quantum_freivalds
        small_A = np.eye(4)
        small_B = np.eye(4)
        small_C = np.eye(4)
        verifier = MatrixProductVerification(small_A, small_B, small_C, method="auto")
        assert verifier.method_name == "quantum_freivalds"
        print(f"✓ Auto method selection (4x4): {verifier.method_name}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Auto method selection: {e}")

    print("\n" + "-" * 60)
    print(f"Phase 1 Class Tests: {tests_passed}/{tests_total} passed")
    print("-" * 60)

    return tests_passed == tests_total


def test_non_square_matrices():
    """Test with non-square matrices."""
    from qward.algorithms import MatrixProductVerification

    print("\n" + "=" * 60)
    print("Testing Non-Square Matrices")
    print("=" * 60)

    tests_passed = 0
    tests_total = 0

    # Test 1: 2x3 × 3x2 = 2x2
    tests_total += 1
    try:
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)  # 2x3
        B = np.array([[7, 8], [9, 10], [11, 12]], dtype=float)  # 3x2
        C = A @ B  # 2x2

        verifier = MatrixProductVerification(A, B, C)
        assert verifier._verifier.n == 2
        assert verifier._verifier.m == 3
        assert verifier._verifier.p == 2

        result = verifier.verify_classically()
        assert result.is_equal is True
        print(f"✓ 2×3 × 3×2 = 2×2: {result}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Non-square test 1: {e}")

    # Test 2: 3x2 × 2x4 = 3x4
    tests_total += 1
    try:
        A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)  # 3x2
        B = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)  # 2x4
        C = A @ B  # 3x4

        verifier = MatrixProductVerification(A, B, C)
        result = verifier.verify_classically()
        assert result.is_equal is True
        print(f"✓ 3×2 × 2×4 = 3×4: {result}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Non-square test 2: {e}")

    print(f"\nNon-square tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


# =============================================================================
# Phase 2: Quantum Freivalds Tests
# =============================================================================


def test_quantum_freivalds_circuit():
    """Test Quantum Freivalds circuit construction."""
    from qward.algorithms import QuantumFreivaldsVerification

    print("\n" + "=" * 60)
    print("Testing Quantum Freivalds Circuit Construction (Phase 2)")
    print("=" * 60)

    tests_passed = 0
    tests_total = 0

    # Test 1: Circuit for correct 2×2 product
    tests_total += 1
    try:
        A = np.array([[1, 2], [3, 4]], dtype=float)
        B = np.array([[5, 6], [7, 8]], dtype=float)
        C = A @ B  # Correct product

        verifier = QuantumFreivaldsVerification(A, B, C)
        circuit = verifier.circuit

        assert circuit is not None
        assert circuit.num_qubits == 2  # 2 qubits for 2-column matrix
        assert verifier.num_marked == 0  # No error-detecting states
        assert verifier.grover_iterations == 0  # No iterations needed
        print(f"✓ Correct 2×2: {circuit.num_qubits} qubits, {verifier.num_marked} marked states")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Correct 2×2 circuit: {e}")

    # Test 2: Circuit for wrong 2×2 product
    tests_total += 1
    try:
        A = np.array([[1, 2], [3, 4]], dtype=float)
        B = np.array([[5, 6], [7, 8]], dtype=float)
        C_wrong = np.array([[19, 22], [43, 51]], dtype=float)  # Wrong!

        verifier = QuantumFreivaldsVerification(A, B, C_wrong)
        circuit = verifier.circuit

        assert circuit is not None
        assert verifier.num_marked > 0  # Should have error-detecting states
        assert verifier.grover_iterations >= 0
        print(
            f"✓ Wrong 2×2: {verifier.num_marked} marked states, {verifier.grover_iterations} Grover iterations"
        )
        print(f"  Error-detecting states: {verifier.get_error_detecting_states()}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Wrong 2×2 circuit: {e}")

    # Test 3: Error detection states computation
    tests_total += 1
    try:
        A = np.array([[1, 0], [0, 1]], dtype=float)  # Identity
        B = np.array([[1, 0], [0, 1]], dtype=float)  # Identity
        C_wrong = np.array([[1, 1], [0, 1]], dtype=float)  # Wrong at (0,1)

        verifier = QuantumFreivaldsVerification(A, B, C_wrong)
        error_states = verifier.get_error_detecting_states()

        # D = I - C_wrong = [[0, -1], [0, 0]]
        # D×[0,1] = [-1, 0] ≠ 0 → state "01" detects error (MSB-first)
        # D×[1,1] = [-1, 0] ≠ 0 → state "11" detects error
        # D×[0,0] = [0, 0] → no error detection
        # D×[1,0] = [0, 0] → no error detection

        assert len(error_states) == 2  # States with r[1]=1 detect error
        print(f"✓ Error states: {error_states}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Error states computation: {e}")

    # Test 4: Circuit depth is reasonable
    tests_total += 1
    try:
        A = np.array([[1, 2], [3, 4]], dtype=float)
        B = np.array([[5, 6], [7, 8]], dtype=float)
        C_wrong = np.array([[19, 22], [43, 51]], dtype=float)

        verifier = QuantumFreivaldsVerification(A, B, C_wrong)
        circuit = verifier.circuit

        # Circuit should have reasonable depth
        assert circuit.depth() < 100
        print(f"✓ Circuit depth: {circuit.depth()}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Circuit depth: {e}")

    # Test 5: 3×3 matrix circuit
    tests_total += 1
    try:
        A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        C = A @ B

        verifier = QuantumFreivaldsVerification(A, B, C)
        circuit = verifier.circuit

        assert circuit.num_qubits == 3  # 3 columns
        assert verifier.num_marked == 0
        print(f"✓ Correct 3×3: {circuit.num_qubits} qubits, {verifier.num_marked} marked")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 3×3 circuit: {e}")

    print(f"\nCircuit tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def test_quantum_freivalds_simulation():
    """Test Quantum Freivalds with actual quantum simulation."""
    from qward.algorithms import QuantumFreivaldsVerification
    from qiskit_aer import AerSimulator

    print("\n" + "=" * 60)
    print("Testing Quantum Freivalds Simulation (Phase 2)")
    print("=" * 60)

    tests_passed = 0
    tests_total = 0
    simulator = AerSimulator()
    shots = 1024

    # Test 1: Correct product → should detect equality
    tests_total += 1
    try:
        A = np.array([[1, 2], [3, 4]], dtype=float)
        B = np.array([[5, 6], [7, 8]], dtype=float)
        C = A @ B

        verifier = QuantumFreivaldsVerification(A, B, C)
        circuit = verifier.circuit

        job = simulator.run(circuit, shots=shots)
        counts = job.result().get_counts()
        result = verifier.interpret_results(counts)

        assert result.is_equal is True
        assert result.method == "quantum_freivalds"
        print(f"✓ Correct product: {result}")
        print(f"  Counts: {dict(list(counts.items())[:4])}...")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Correct product simulation: {e}")

    # Test 2: Wrong product → should detect inequality
    tests_total += 1
    try:
        A = np.array([[1, 2], [3, 4]], dtype=float)
        B = np.array([[5, 6], [7, 8]], dtype=float)
        C_wrong = np.array([[19, 22], [43, 51]], dtype=float)

        verifier = QuantumFreivaldsVerification(A, B, C_wrong)
        circuit = verifier.circuit

        job = simulator.run(circuit, shots=shots)
        counts = job.result().get_counts()
        result = verifier.interpret_results(counts)

        assert result.is_equal is False
        print(f"✓ Wrong product: {result}")
        print(f"  Error detection prob: {result.details['error_detection_prob']:.3f}")
        print(f"  Marked states found: {result.details['error_detected_count']}/{shots}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Wrong product simulation: {e}")

    # Test 3: Verify classical comparison matches
    tests_total += 1
    try:
        A = np.array([[1, 0], [0, 1]], dtype=float)
        B = np.array([[2, 3], [4, 5]], dtype=float)
        C = A @ B

        verifier = QuantumFreivaldsVerification(A, B, C)
        circuit = verifier.circuit

        job = simulator.run(circuit, shots=shots)
        counts = job.result().get_counts()
        result = verifier.interpret_results(counts)

        assert result.is_equal == result.classical_comparison
        print(f"✓ Quantum/Classical match: both say {'EQUAL' if result.is_equal else 'NOT EQUAL'}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Classical comparison: {e}")

    # Test 4: Success criteria function
    tests_total += 1
    try:
        A = np.array([[1, 2], [3, 4]], dtype=float)
        B = np.array([[5, 6], [7, 8]], dtype=float)
        C_wrong = np.array([[19, 22], [43, 51]], dtype=float)

        verifier = QuantumFreivaldsVerification(A, B, C_wrong)
        error_states = verifier.get_error_detecting_states()

        # Error-detecting states should return False for success_criteria
        for state in error_states:
            assert verifier.success_criteria(state) is False

        # Non-error states should return True
        all_states = [format(i, "02b") for i in range(4)]
        non_error_states = [s for s in all_states if s not in error_states]
        for state in non_error_states:
            assert verifier.success_criteria(state) is True

        print(f"✓ Success criteria: error states={error_states}, non-error={non_error_states}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Success criteria: {e}")

    # Test 5: Theoretical success probability
    tests_total += 1
    try:
        A = np.array([[1, 2], [3, 4]], dtype=float)
        B = np.array([[5, 6], [7, 8]], dtype=float)
        C_wrong = np.array([[19, 22], [43, 51]], dtype=float)

        verifier = QuantumFreivaldsVerification(A, B, C_wrong)
        theoretical_prob = verifier.get_theoretical_success_probability()

        # Run simulation
        circuit = verifier.circuit
        job = simulator.run(circuit, shots=shots)
        counts = job.result().get_counts()

        # Compute observed probability
        error_count = sum(counts.get(s, 0) for s in verifier.get_error_detecting_states())
        observed_prob = error_count / shots

        # Should be reasonably close (within 0.2)
        assert abs(theoretical_prob - observed_prob) < 0.2
        print(f"✓ Theoretical prob: {theoretical_prob:.3f}, Observed: {observed_prob:.3f}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Theoretical probability: {e}")

    # Test 6: 3×3 simulation
    tests_total += 1
    try:
        A = np.array([[1, 2, 0], [0, 1, 2], [2, 0, 1]], dtype=float)
        B = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        C = A @ B  # C = A (since B is identity)

        verifier = QuantumFreivaldsVerification(A, B, C)
        circuit = verifier.circuit

        job = simulator.run(circuit, shots=shots)
        counts = job.result().get_counts()
        result = verifier.interpret_results(counts)

        assert result.is_equal is True
        print(f"✓ 3×3 simulation: {result}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 3×3 simulation: {e}")

    # Test 7: Multiple wrong entries
    tests_total += 1
    try:
        A = np.array([[1, 0], [0, 1]], dtype=float)
        B = np.array([[1, 0], [0, 1]], dtype=float)
        # Multiple errors
        C_wrong = np.array([[2, 1], [1, 2]], dtype=float)

        verifier = QuantumFreivaldsVerification(A, B, C_wrong)
        circuit = verifier.circuit

        job = simulator.run(circuit, shots=shots)
        counts = job.result().get_counts()
        result = verifier.interpret_results(counts)

        assert result.is_equal is False
        print(f"✓ Multiple errors: {result}")
        print(f"  Num marked states: {verifier.num_marked}/4")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Multiple errors: {e}")

    print(f"\nSimulation tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def test_bit_ordering_reference():
    """Validate bit ordering against Qiskit measurement strings."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qward.algorithms import QuantumFreivaldsVerification

    print("\n" + "=" * 60)
    print("Testing Bit Ordering Reference (MSB vs LSB)")
    print("=" * 60)

    simulator = AerSimulator()
    shots = 256

    # Prepare |10010> (qubits 4 and 1 set)
    qc = QuantumCircuit(5, 5)
    qc.x(4)
    qc.x(1)
    qc.measure(range(5), range(5))

    counts = simulator.run(qc, shots=shots).result().get_counts()
    top_outcome = max(counts.items(), key=lambda x: x[1])[0]

    msb_state = "10010"
    index = int(msb_state, 2)

    # Create verifiers to expose formatting behavior
    A = np.eye(5)
    B = np.eye(5)
    C = np.eye(5)
    verifier_msb = QuantumFreivaldsVerification(A, B, C)

    msb_formatted = verifier_msb._format_state(index)

    print(f"Measured counts: {counts}")
    print(f"Top outcome: {top_outcome}")
    print(f"MSB formatted index=1: {msb_formatted}")

    # Qiskit measurement strings are MSB-first
    assert top_outcome == msb_state
    assert msb_formatted == msb_state

    print("✓ Bit ordering reference confirms MSB-first measurement strings")
    return True


def test_bit_order_error_state_mapping():
    """Verify error-detecting state mapping for MSB."""
    from qward.algorithms import QuantumFreivaldsVerification

    print("\n" + "=" * 60)
    print("Testing Bit Order Error-State Mapping")
    print("=" * 60)

    # Construct a case with a known error pattern
    A = np.eye(2)
    B = np.eye(2)
    C_wrong = np.array([[1, 1], [0, 1]], dtype=float)  # Error at (0,1)

    # For this D, error-detecting vectors have r1=1 → indices 2 ("10"), 3 ("11")
    expected_msb = {"10", "11"}

    verifier_msb = QuantumFreivaldsVerification(A, B, C_wrong)

    msb_states = set(verifier_msb.get_error_detecting_states())

    print(f"MSB error states: {sorted(msb_states)}")

    assert msb_states == expected_msb

    # MSB success criteria sanity checks
    assert verifier_msb.success_criteria("10") is False
    assert verifier_msb.success_criteria("00") is True

    print("✓ Error-detecting state mapping matches expectations")
    return True


def test_quantum_vs_classical_comparison():
    """Compare quantum and classical Freivalds results."""
    from qward.algorithms import MatrixProductVerification
    from qiskit_aer import AerSimulator

    print("\n" + "=" * 60)
    print("Quantum vs Classical Comparison (Phase 2)")
    print("=" * 60)

    simulator = AerSimulator()
    shots = 1024

    test_cases = [
        (
            "2×2 correct",
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
            None,
        ),  # Will compute correct C
        (
            "2×2 wrong (1 error)",
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
            np.array([[19, 22], [43, 51]]),
        ),
        ("3×3 identity", np.eye(3), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), None),
        (
            "3×3 wrong",
            np.eye(3),
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]]),
        ),  # Error at (2,2)
    ]

    all_match = True

    print(f"\n{'Test Case':<25} {'Classical':<12} {'Quantum':<12} {'Match':<8}")
    print("-" * 60)

    for name, A, B, C in test_cases:
        if C is None:
            C = A @ B  # Compute correct product

        verifier = MatrixProductVerification(A, B, C)

        # Classical result
        classical_result = verifier.verify_classically()

        # Quantum result
        circuit = verifier.circuit
        job = simulator.run(circuit, shots=shots)
        counts = job.result().get_counts()
        quantum_result = verifier._verifier.interpret_results(counts)

        # Check match
        match = classical_result.is_equal == quantum_result.is_equal
        all_match = all_match and match

        classical_str = "EQUAL" if classical_result.is_equal else "NOT EQUAL"
        quantum_str = "EQUAL" if quantum_result.is_equal else "NOT EQUAL"
        match_str = "✓" if match else "✗"

        print(f"{name:<25} {classical_str:<12} {quantum_str:<12} {match_str:<8}")

    print("-" * 60)
    print(f"All tests match: {'✓ YES' if all_match else '✗ NO'}")

    return all_match


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Run demonstration
    demonstrate_freivalds_algorithm()

    # Run all classical algorithm tests
    print("\n")
    results = run_all_tests(verbose=True)

    # Run Phase 1 class structure tests
    class_tests_passed = test_base_class_structure()

    # Run non-square matrix tests
    non_square_passed = test_non_square_matrices()

    # Run Phase 2 quantum circuit tests
    circuit_tests_passed = test_quantum_freivalds_circuit()

    # Run Phase 2 simulation tests
    simulation_tests_passed = test_quantum_freivalds_simulation()

    # Run quantum vs classical comparison
    comparison_passed = test_quantum_vs_classical_comparison()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Classical Algorithm Tests:      {results['passed']}/{results['total']} passed")
    print(f"Phase 1 Class Structure:        {'PASSED' if class_tests_passed else 'FAILED'}")
    print(f"Non-Square Matrix Tests:        {'PASSED' if non_square_passed else 'FAILED'}")
    print(f"Phase 2 Circuit Tests:          {'PASSED' if circuit_tests_passed else 'FAILED'}")
    print(f"Phase 2 Simulation Tests:       {'PASSED' if simulation_tests_passed else 'FAILED'}")
    print(f"Quantum vs Classical:           {'PASSED' if comparison_passed else 'FAILED'}")
    print("=" * 60)

    # Exit with appropriate code
    import sys

    all_passed = (
        results["failed"] == 0
        and class_tests_passed
        and non_square_passed
        and circuit_tests_passed
        and simulation_tests_passed
        and comparison_passed
    )
    sys.exit(0 if all_passed else 1)
