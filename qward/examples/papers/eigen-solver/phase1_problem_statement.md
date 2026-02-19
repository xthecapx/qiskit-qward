# Phase 1: Problem Statement -- Quantum Eigensolver for Small Hermitian Matrices

**Author**: Quantum Computing Researcher
**Date**: 2026-02-19
**Status**: Complete
**Phase**: 1 -- Ideation

---

## 1. Problem Definition

### 1.1 Objective

Design and implement a Variational Quantum Eigensolver (VQE) for computing eigenvalues of small Hermitian matrices (2x2, 3x3, 4x4) with rigorous classical validation against NumPy/SciPy eigensolvers.

### 1.2 Formal Statement

Given an n x n Hermitian matrix M (where M = M^dagger), find the eigenvalues {lambda_k} satisfying:

```
M |v_k> = lambda_k |v_k>
```

The VQE approach reformulates this as a minimization problem:

```
lambda_min = min_{|psi>} <psi| H |psi>
```

where H is the Hamiltonian encoding of M, decomposed into the Pauli basis:

```
H = sum_i c_i P_i
```

with P_i being tensor products of Pauli operators {I, X, Y, Z} and c_i being real coefficients.

### 1.3 Scope

- **In scope**: 2x2, 3x3, and 4x4 Hermitian matrices; ideal and noisy simulation; minimum eigenvalue via VQE; all eigenvalues via deflation
- **Out of scope**: Matrices larger than 4x4; real QPU execution (Phase 5 may include this); non-Hermitian matrices; time-dependent problems

---

## 2. VQE Feasibility Assessment

### 2.1 VQE for General Hermitian Matrices -- Confirmed Feasible

VQE is appropriate for general Hermitian matrices because:

1. **Hermiticity guarantees real eigenvalues**, which is necessary for the variational principle to hold. The expectation value E(theta) = <psi(theta)|H|psi(theta)> is real-valued and bounded below by lambda_min.

2. **Any Hermitian matrix can be decomposed into the Pauli basis.** For an n x n matrix acting on q = ceil(log_2(n)) qubits, the decomposition is:

   ```
   H = (1/2^q) sum_{P in {I,X,Y,Z}^q} Tr(M * P) * P
   ```

   This decomposition is exact and preserves all eigenvalues.

3. **The variational principle guarantees**: For any trial state |psi(theta)>,

   ```
   <psi(theta)|H|psi(theta)> >= lambda_min
   ```

   with equality if and only if |psi(theta)> is the ground state.

4. **Small matrix sizes (1--2 qubits) are ideal for VQE**: The Hilbert space is small enough that a hardware-efficient ansatz can be made fully expressive, and barren plateaus are not a concern.

### 2.2 Qubit Requirements

| Matrix Size | Qubits Required | Hilbert Space Dim | Padding Needed |
|:-----------:|:---------------:|:-----------------:|:--------------:|
| 2x2         | 1               | 2                 | No             |
| 3x3         | 2               | 4                 | Yes (1 state)  |
| 4x4         | 2               | 4                 | No             |

**3x3 Embedding Note**: A 3x3 matrix requires embedding into a 4x4 matrix (2-qubit system) by padding the (4,4) entry with a penalty value larger than the maximum eigenvalue. This ensures:
- The ground state of the embedded system corresponds to the ground state of the original system
- The first 3 eigenvalues of the embedded system match the original system exactly
- The 4th eigenvalue is a known artifact that is discarded

### 2.3 Circuit Depth Estimates

| System | Ansatz Reps | Parameters | RY Gates | CX Gates | Est. Depth |
|:------:|:-----------:|:----------:|:--------:|:--------:|:----------:|
| 1-qubit | 1          | 2          | 2        | 0        | 2          |
| 1-qubit | 2          | 3          | 3        | 0        | 3          |
| 2-qubit | 2          | 6          | 6        | 2        | 8          |
| 2-qubit | 3          | 8          | 8        | 3        | 11         |

**Recommended starting configurations:**
- 2x2 matrices (1 qubit): RY ansatz with 2 repetitions (3 parameters)
- 3x3/4x4 matrices (2 qubits): RealAmplitudes/EfficientSU2 with 2--3 repetitions (6--8 parameters)

### 2.4 Pauli Decomposition Complexity

| Matrix | Pauli Terms | Notes |
|:------:|:-----------:|:------|
| M1 (Pauli Z) | 1 (Z) | Trivial -- already a Pauli operator |
| M2 (Pauli X) | 1 (X) | Trivial -- already a Pauli operator |
| M3 (General 2x2) | 4 (I, X, Y, Z) | Maximum for 1-qubit system |
| M4 (3x3 embedded) | 8 | Moderate, due to penalty padding |
| M5 (Heisenberg XXX) | 3 (XX, YY, ZZ) | Sparse, physically motivated |

The number of Pauli terms directly impacts the number of measurement circuits needed. For these small systems, all decompositions are manageable (at most 16 terms for a 2-qubit system).

---

## 3. Test Matrices with Classical Eigenvalues

All eigenvalues have been independently verified using `numpy.linalg.eigvalsh`.

### 3.1 M1: Pauli Z (2x2, diagonal, trivial)

```
M1 = [[1, 0], [0, -1]]
```

- **Eigenvalues**: {-1, 1}
- **Spectral range**: 2.0
- **Pauli decomposition**: H = Z
- **Notes**: Trivial case. Ground state is |1> with energy -1. No optimization needed -- serves as a sanity check.

### 3.2 M2: Pauli X (2x2, off-diagonal)

```
M2 = [[0, 1], [1, 0]]
```

- **Eigenvalues**: {-1, 1}
- **Spectral range**: 2.0
- **Pauli decomposition**: H = X
- **Notes**: Ground state is |-> = (|0> - |1>)/sqrt(2). Requires the ansatz to create a superposition state.

### 3.3 M3: General Hermitian (2x2, complex off-diagonal)

```
M3 = [[2, 1-1j], [1+1j, 3]]
```

- **Eigenvalues**: {1, 4}
  *(Note: The original plan stated {~1.27, ~3.73}, which was incorrect. The characteristic polynomial is lambda^2 - 5*lambda + 4 = 0, giving exact roots lambda = 1 and lambda = 4.)*
- **Spectral range**: 3.0
- **Pauli decomposition**: H = 2.5*I + 1.0*X + 1.0*Y - 0.5*Z
- **Condition number**: 4.0
- **Notes**: Tests the ansatz's ability to represent states with complex amplitudes. The ground state eigenvector has complex entries, requiring RZ + RY gates or a full U3 parameterization (not just RY alone).

### 3.4 M4: Symmetric Real (3x3, embedded in 4x4)

```
M4 = [[2, 1, 0], [1, 3, 1], [0, 1, 2]]
```

- **Eigenvalues**: {1, 2, 4}
  *(Note: The original plan stated {~0.59, ~2.00, ~4.41}, which was incorrect. The characteristic polynomial factors as (2-lambda)(lambda-1)(lambda-4) = 0.)*
- **Spectral range**: 3.0
- **Embedding**: Pad to 4x4 with M[3,3] = penalty >> max(eigenvalues). Recommended penalty: 100.
- **Embedded eigenvalues**: {1, 2, 4, 100} (last is artifact)
- **Pauli decomposition (embedded)**: 8 terms
- **Notes**: Tests the 3x3 embedding strategy. VQE must find the correct ground state while the optimizer avoids the penalized subspace.

### 3.5 M5: Heisenberg XXX Model (4x4, physically motivated)

```
M5 = XX + YY + ZZ  (Heisenberg XXX Hamiltonian, J=1)

     [[ 1,  0,  0,  0],
      [ 0, -1,  2,  0],
      [ 0,  2, -1,  0],
      [ 0,  0,  0,  1]]
```

- **Eigenvalues**: {-3, 1, 1, 1}
- **Spectral range**: 4.0
- **Degeneracy**: Eigenvalue 1 has multiplicity 3 (triplet state)
- **Pauli decomposition**: H = XX + YY + ZZ (3 terms, already in Pauli form)
- **Condition number**: 3.0
- **Notes**: Physically relevant 2-qubit model. The ground state is the singlet |psi-> = (|01> - |10>)/sqrt(2) with energy -3. Tests the ansatz's ability to create entangled states. The 3-fold degeneracy at eigenvalue 1 tests deflation robustness.

---

## 4. Success Criteria

### 4.1 Primary Criteria

| Criterion | Threshold | Metric |
|:----------|:---------:|:-------|
| **Ideal VQE accuracy** | <= 1% relative error | \|lambda_VQE - lambda_exact\| / spectral_range < 0.01 |
| **Noisy VQE accuracy** | <= 5% relative error | \|lambda_VQE - lambda_exact\| / spectral_range < 0.05 |
| **Convergence** | <= 200 iterations | Optimizer iterations to convergence |
| **All eigenvalues** | <= 5% per eigenvalue (ideal) | Via deflation method |

### 4.2 Error Metric Definition

We define the **normalized error** relative to the spectral range to provide a scale-invariant measure:

```
epsilon = |lambda_VQE - lambda_exact| / (lambda_max - lambda_min)
```

This ensures the error threshold is meaningful regardless of the absolute scale of the matrix.

### 4.3 Per-Matrix Absolute Thresholds

| Matrix | Spectral Range | 1% Threshold (ideal) | 5% Threshold (noisy) |
|:------:|:--------------:|:--------------------:|:--------------------:|
| M1     | 2.0            | 0.020                | 0.100                |
| M2     | 2.0            | 0.020                | 0.100                |
| M3     | 3.0            | 0.030                | 0.150                |
| M4     | 3.0            | 0.030                | 0.150                |
| M5     | 4.0            | 0.040                | 0.200                |

### 4.4 Statistical Requirements

- Ideal simulation: Statevector (exact) for primary validation, then shot-based (4096 shots) for realistic comparison
- Noisy simulation: 8192 shots minimum, 10 independent trials per configuration
- Report: mean +/- standard deviation across trials
- Confidence: Results must be within threshold for >= 8/10 trials

---

## 5. Risk Assessment

### 5.1 Barren Plateaus -- LOW RISK

For 1--2 qubit systems, barren plateaus are not a practical concern:

- **1 qubit**: Gradient variance lower bound ~ 0.50 (negligible vanishing)
- **2 qubits**: Gradient variance lower bound ~ 0.25 (negligible vanishing)

The parameter spaces are small enough (2--8 parameters) that even gradient-free optimizers like COBYLA can efficiently explore the landscape. Barren plateaus become problematic only for systems with many qubits (typically > 10).

### 5.2 Local Minima -- LOW RISK

For the target matrix sizes:
- 1-qubit systems have smooth, well-behaved cost landscapes with no local minima for typical ansatze
- 2-qubit systems with 2--3 ansatz layers may have local minima, but random restarts (3--5 initial points) effectively mitigate this

### 5.3 Ansatz Expressibility -- MODERATE RISK

- **Real-valued ansatze (RealAmplitudes) cannot represent complex eigenstates.** Matrix M3 has complex eigenvectors, so a RY-only ansatz will fail. Mitigation: use EfficientSU2 (RY+RZ) for matrices with complex off-diagonal elements.
- **2-qubit ansatze must create entanglement.** M5's ground state is maximally entangled (Bell state). At least 1 CX gate is required. Mitigation: ensure minimum 1 entanglement layer.

### 5.4 3x3 Embedding -- MODERATE RISK

- The penalty padding value must be chosen carefully. Too small: the penalized state may interact with the physical spectrum. Too large: numerical precision issues in Pauli coefficients.
- **Recommended penalty**: 10x the spectral range of the original matrix (penalty = 30 for M4)
- **Mitigation for deflation**: When finding all eigenvalues, explicitly discard the last eigenvalue and only return the first n=3 eigenvalues.

### 5.5 Optimizer Convergence -- LOW RISK

- COBYLA: Gradient-free, robust for small parameter spaces. Expected convergence in 50--100 iterations for 1-qubit, 100--200 for 2-qubit.
- SPSA: Stochastic gradient approximation, better for noisy evaluations. Recommended for noisy simulations.
- **Risk**: Slow convergence if initial parameters are far from optimal. Mitigation: use multiple random starts.

### 5.6 Noise Impact -- MODERATE RISK

- Depolarizing and readout errors will shift the estimated eigenvalue toward the mean of the spectrum (regression to the mean effect).
- Gate errors accumulate with circuit depth. For 2-qubit circuits at depth 8--11, expect moderate degradation.
- **Mitigation**: Keep circuits shallow (2--3 ansatz repetitions). Consider error mitigation techniques (ZNE, PEC) if 5% threshold is not met.

### 5.7 Degenerate Eigenvalues -- LOW RISK

- M5 has a 3-fold degenerate eigenvalue (lambda=1). VQE will find one eigenstate from the degenerate subspace, not necessarily a specific one.
- For deflation: degenerate eigenvalues require careful penalty construction to avoid numerical instability in the deflated Hamiltonian.
- **Mitigation**: Use a small regularization in the deflation penalty term.

---

## 6. Algorithm Selection Rationale

### 6.1 Why VQE (not QPE, QAOA, or classical)?

| Algorithm | Qubits | Depth | Noise Tolerance | Suitability |
|:----------|:------:|:-----:|:---------------:|:-----------:|
| **VQE** | n | O(poly(n)) | High (shallow circuits) | Best for NISQ |
| QPE | n + m ancilla | O(2^m) | Low (deep circuits) | Fault-tolerant era |
| QAOA | n | O(p * n) | Moderate | Combinatorial optimization |
| Classical | N/A | N/A | N/A | Polynomial for dense matrices |

**VQE is selected because:**
1. It uses the shallowest circuits, maximizing noise tolerance on NISQ devices
2. It requires no ancilla qubits, minimizing qubit overhead
3. The variational approach naturally adapts to hardware constraints
4. For small matrices, the classical-quantum hybrid loop is fast
5. It provides a natural framework for finding excited states via deflation

### 6.2 Classical Comparison

For n x n matrices, classical eigensolvers (e.g., `numpy.linalg.eigh`) run in O(n^3) time and are clearly superior for small matrices. **The purpose of this project is not to demonstrate quantum advantage**, but rather to:

1. Validate the VQE pipeline for Hamiltonian eigenvalue problems
2. Establish accuracy benchmarks against known classical results
3. Characterize noise impact on variational algorithms
4. Build infrastructure for scaling to larger, classically intractable problems

---

## 7. Preliminary VQE Protocol

### 7.1 Workflow

```
Input: Hermitian matrix M (n x n)
  |
  v
[1] Validate M = M^dagger
  |
  v
[2] Embed if needed (n != 2^q, add penalty padding)
  |
  v
[3] Pauli decomposition: H = sum_i c_i P_i
  |
  v
[4] Select ansatz based on matrix properties
    - Real eigenvalues + real eigenvectors: RealAmplitudes
    - Complex eigenvectors: EfficientSU2
  |
  v
[5] Initialize parameters (random or heuristic)
  |
  v
[6] VQE optimization loop:
    - Prepare |psi(theta)> via ansatz
    - Measure E(theta) = <psi(theta)|H|psi(theta)>
    - Update theta via classical optimizer
    - Repeat until |E_{k+1} - E_k| < epsilon
  |
  v
[7] Return minimum eigenvalue and optimal parameters
  |
  v
[8] (Optional) Deflation: H' = H + alpha |psi_0><psi_0|, repeat for excited states
```

### 7.2 Key Design Decisions (for Phase 2)

1. **Pauli decomposition**: Trace-based formula, exact for all Hermitian matrices
2. **Ansatz**: Hardware-efficient (RealAmplitudes / EfficientSU2), parameterized depth
3. **Optimizer**: COBYLA (ideal), SPSA (noisy)
4. **Convergence**: |E_{k+1} - E_k| < 10^{-6} for ideal, < 10^{-4} for noisy
5. **Deflation**: Penalty-based H' = H + alpha |psi_0><psi_0| with alpha ~ 10 * spectral_range

---

## 8. Handoff Checklist

- [x] Problem statement defined with mathematical formulation
- [x] VQE feasibility confirmed for general Hermitian matrices
- [x] Test matrices defined with verified classical eigenvalues
- [x] Eigenvalue corrections documented (M3: {1,4}, M4: {1,2,4})
- [x] Success thresholds agreed upon (1% ideal, 5% noisy, normalized by spectral range)
- [x] Qubit and circuit depth estimates provided
- [x] Risk assessment complete with mitigations
- [x] Algorithm selection justified
- [x] Preliminary VQE protocol outlined

**Next**: Phase 2 (Theoretical Design) -- Full VQE formulation, Hamiltonian encoding algorithm, ansatz circuit specification, optimizer selection, and correctness proofs.
