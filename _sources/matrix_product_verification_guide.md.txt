# Matrix Product Verification Guide

This guide explains how the matrix product verification algorithm works and how
to run it with optional circuit diagrams.

## Overview

The verification checks whether `A x B = C` without fully computing the product.
The default quantum implementation is **Quantum Freivalds**, which:

1. Builds the error matrix `D = A x B - C`.
2. Finds binary vectors `r` such that `D x r != 0` (these detect errors).
3. Marks the error-detecting states with an oracle.
4. Uses Grover-style amplification to increase the probability of measuring
   error-detecting states.
5. Measures and decides if `A x B = C`.

## Quick Start

```python
import numpy as np
from qward.algorithms import MatrixProductVerification

A = np.array([[1, 2], [3, 4]], dtype=float)
B = np.array([[5, 6], [7, 8]], dtype=float)
C = A @ B

verifier = MatrixProductVerification(A, B, C)
result = verifier.verify(shots=1024)

print(result.is_equal, result.confidence)
```

## Print Diagrams While Running

Pass `print_diagrams=True` to print the verification circuit. For the Quantum
Freivalds implementation, the oracle and diffuser diagrams are also printed.

```python
result = verifier.verify(
    shots=1024,
    print_diagrams=True,
    diagram_output="text",
)
```

Supported `diagram_output` values are `text`, `mpl`, and `latex`. Use `text`
for terminal output.

## Interpreting Results

The returned `VerificationResult` contains:

- `is_equal`: Final verification decision.
- `confidence`: Confidence estimate from measurement statistics.
- `measurement_counts`: Raw counts from the backend.
- `details`: Diagnostic information such as error-detection probability.

If `is_equal` is `False`, the circuit measured error-detecting states with a
high probability.

## Notes on Bit Ordering

Qiskit measurement strings are **MSB-first**. The verification implementation
uses this convention for error-detecting states and result interpretation.

## Troubleshooting

- If you see very low confidence, increase `shots`.
- For large matrices, expect longer build times due to state enumeration.
- The Buhrman-Spalek method is a placeholder and is not yet implemented.
