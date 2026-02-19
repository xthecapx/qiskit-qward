# Phase 2: Theoretical Design -- VQE Eigensolver

**Author**: Quantum Computing Researcher
**Date**: 2026-02-19
**Status**: Complete
**Phase**: 2 -- Theoretical Design

---

## 1. Overview

This document provides the complete theoretical specification for a VQE-based eigensolver for small Hermitian matrices (2x2, 3x3, 4x4). It covers:

1. Pauli decomposition algorithm
2. Non-power-of-two embedding strategy
3. Ansatz circuit design
4. Measurement and estimation strategy
5. Classical optimization specification
6. Eigenvalue deflation for full spectrum
7. Correctness proofs
8. Complexity analysis

---

## 2. Pauli Decomposition

### 2.1 Mathematical Foundation

Any 2^q x 2^q Hermitian matrix M can be uniquely decomposed in the q-qubit Pauli basis:

```
H = sum_{P in {I,X,Y,Z}^{otimes q}} c_P * P                               (1)
```

where the coefficients are computed via:

```
c_P = Tr(M * P) / 2^q                                                       (2)
```

**Proof of Eq. (2)**:

The set {I, X, Y, Z}^{otimes q} forms an orthogonal basis for the space of 2^q x 2^q matrices under the Hilbert-Schmidt inner product:

```
<A, B>_HS = Tr(A^dagger * B)
```

For Pauli strings P_i, P_j:

```
Tr(P_i * P_j) = 2^q * delta_{ij}                                           (3)
```

(Each Pauli string is traceless except the identity, and the product of two distinct Pauli strings is another Pauli string, which is traceless.)

Substituting Eq. (1) into Tr(M * P_j):

```
Tr(M * P_j) = sum_i c_i * Tr(P_i * P_j) = sum_i c_i * 2^q * delta_{ij} = 2^q * c_j
```

Solving for c_j gives Eq. (2). QED.

**Properties**:
- If M is Hermitian, all c_P are real (since Pauli operators are Hermitian and the trace of a product of Hermitian operators is real when the product is Hermitian)
- The decomposition is exact (no approximation)
- The number of terms is at most 4^q (but many may be zero for structured matrices)

### 2.2 Algorithm

```python
def pauli_decompose(M: np.ndarray) -> Dict[str, float]:
    """
    Decompose Hermitian matrix M into Pauli string basis.

    Input: n x n Hermitian matrix M (n must be power of 2,
           or will be embedded first)
    Output: Dictionary mapping Pauli string labels to real coefficients

    Algorithm:
    1. Determine q = ceil(log2(n))
    2. If n < 2^q, embed M into 2^q x 2^q matrix (see Section 3)
    3. Generate all q-qubit Pauli strings P
    4. For each P: c_P = Tr(M * P) / 2^q
    5. Filter out terms with |c_P| < epsilon (numerical zero)
    6. Return {P: c_P} for non-zero terms
    """
```

### 2.3 Decompositions for Test Matrices

**M1 (Pauli Z, 1-qubit)**:
```
H_1 = 1.0 * Z
```
1 Pauli term.

**M2 (Pauli X, 1-qubit)**:
```
H_2 = 1.0 * X
```
1 Pauli term.

**M3 (General Hermitian, 1-qubit)**:
```
H_3 = 2.5 * I + 1.0 * X + 1.0 * Y - 0.5 * Z
```
4 Pauli terms. Note the Y component arises from the complex off-diagonal element (1-i).

Verification:
```
2.5*I + 1.0*X + 1.0*Y - 0.5*Z
= 2.5*[[1,0],[0,1]] + 1.0*[[0,1],[1,0]] + 1.0*[[0,-i],[i,0]] + (-0.5)*[[1,0],[0,-1]]
= [[2.5+(-0.5), 1-i], [1+i, 2.5+0.5]]
= [[2, 1-i], [1+i, 3]]  = M3  ✓
```

**M4 (3x3 embedded in 4x4 with penalty=10, 2-qubit)**: See Section 3.

**M5 (Heisenberg XXX, 2-qubit)**:
```
H_5 = 1.0 * XX + 1.0 * YY + 1.0 * ZZ
```
3 Pauli terms. Already in Pauli form by construction.

---

## 3. Non-Power-of-Two Embedding

### 3.1 Problem

A 3x3 matrix cannot be directly encoded on qubits because 3 is not a power of 2. We need q = ceil(log_2(3)) = 2 qubits, giving a 4-dimensional Hilbert space. The extra basis state |11> must be handled.

### 3.2 Penalty Embedding Method

Embed the n x n matrix M into a 2^q x 2^q matrix M_emb:

```
M_emb = [[M, 0], [0, p * I_{2^q - n}]]                                     (4)
```

where p is a penalty value chosen such that p > lambda_max(M).

For the 3x3 case:

```
M_emb = [[M_3x3,  0  ],
          [  0  , p   ]]     (4x4 matrix)
```

### 3.3 Penalty Selection

**Constraint**: The penalty value p must satisfy:

1. **Ground state preservation**: p > lambda_max(M), so the penalty state is never the ground state
2. **Deflation safety**: p > lambda_max(M) + alpha, where alpha is the deflation penalty strength
3. **Numerical stability**: p should not be so large that Pauli coefficients cause excessive shot noise

**Optimal penalty formula**:

```
p = lambda_max(M) + 2 * (lambda_max(M) - lambda_min(M))                     (5)
```

This ensures a safety margin of 2x the spectral range above the largest eigenvalue.

**For M4**: lambda_max = 4, lambda_min = 1, spectral_range = 3

```
p = 4 + 2 * 3 = 10
```

**M4 embedded decomposition (penalty = 10)**:

```
H_4 = 4.25*II + 0.5*IX - 2.25*IZ + 0.5*XX + 0.5*YY - 1.75*ZI + 0.5*ZX + 1.75*ZZ
```

8 Pauli terms. Eigenvalues of H_4: {1, 2, 4, 10}.

### 3.4 Correctness Proof for Embedding

**Theorem**: Let M be an n x n Hermitian matrix with eigenvalues lambda_1 <= ... <= lambda_n and corresponding eigenvectors |v_1>, ..., |v_n>. Let M_emb be the penalty-embedded 2^q x 2^q matrix (Eq. 4) with p > lambda_n. Then:

1. The eigenvalues of M_emb are {lambda_1, ..., lambda_n, p, ..., p}
2. The eigenvectors of M_emb corresponding to lambda_k are |v_k, 0> (zero-padded)
3. The ground state of M_emb is |v_1, 0> with eigenvalue lambda_1

**Proof**: M_emb is block-diagonal, so its eigenvalues are the union of the eigenvalues of each block. Block 1 (M) has eigenvalues {lambda_k}. Block 2 (p * I) has eigenvalue p with multiplicity 2^q - n. Since p > lambda_n >= lambda_k for all k, the minimum eigenvalue of M_emb is lambda_1 = lambda_min(M). The eigenvectors are zero-padded versions of the original eigenvectors. QED.

---

## 4. Ansatz Circuit Design

### 4.1 Design Principles

The ansatz |psi(theta)> = U(theta)|0>^{otimes q} must be:

1. **Expressive**: Able to represent the target eigenstate
2. **Shallow**: Minimize circuit depth for noise resilience
3. **Trainable**: Avoid barren plateaus (guaranteed for 1-2 qubits)

### 4.2 Single-Qubit Ansatz (2x2 matrices)

For matrices with **real eigenvectors** (M1, M2):

```
|psi(theta)> = RY(theta)|0>
```

Circuit:
```
q: ─ RY(θ) ─
```

Parameters: 1
Depth: 1
This parameterizes the XZ great circle of the Bloch sphere, which suffices for any eigenstate of a Hamiltonian of the form a*I + b*X + c*Z.

For matrices with **complex eigenvectors** (M3):

```
|psi(theta_1, theta_2)> = RZ(theta_2) RY(theta_1) |0>
```

Circuit:
```
q: ─ RY(θ₁) ─ RZ(θ₂) ─
```

Parameters: 2
Depth: 2
This parameterizes the full Bloch sphere (equivalent to U3 up to global phase), reaching any single-qubit state.

**General single-qubit recommendation**: Use the RY+RZ parameterization for all 2x2 matrices. The extra parameter adds negligible overhead and ensures universality.

### 4.3 Two-Qubit Ansatz (3x3 and 4x4 matrices)

We use a hardware-efficient ansatz with the following layer structure:

**EfficientSU2-style ansatz**:

```
Layer l:
  q0: ─ RY(θ_{l,0,0}) ─ RZ(θ_{l,0,1}) ─ ● ─
  q1: ─ RY(θ_{l,1,0}) ─ RZ(θ_{l,1,1}) ─ X ─

Repeated L times, with a final rotation layer:
  q0: ─ RY(θ_{L,0,0}) ─ RZ(θ_{L,0,1}) ─
  q1: ─ RY(θ_{L,1,0}) ─ RZ(θ_{L,1,1}) ─
```

For L repetitions:
- Parameters: 4 * (L + 1)
- CX gates: L
- Depth: 3*L + 2 (approx.)

**Recommended configurations**:

| Matrix | Ansatz | Reps (L) | Parameters | CX Gates | Notes |
|:------:|:------:|:--------:|:----------:|:--------:|:------|
| M4 (3x3) | EfficientSU2 | 2 | 12 | 2 | Real matrix, but embedded system may need complex amplitudes |
| M5 (Heisenberg) | EfficientSU2 | 2 | 12 | 2 | Ground state is Bell state, needs entanglement |

### 4.4 Ansatz Expressibility Analysis

**Theorem**: The EfficientSU2 ansatz with L >= 1 repetition on 2 qubits can represent any state in the 4-dimensional Hilbert space.

**Proof sketch**: A single layer of RY, RZ on each qubit provides SU(2) x SU(2) coverage (4 parameters for the product group). One CX gate provides an entangling operation. By the universality theorem for quantum computation, the set {RY, RZ, CX} is universal for SU(4). With L = 1 repetition (8 parameters), the ansatz covers a dense subset of SU(4). With L = 2 (12 parameters), it achieves full expressibility for the 4-dimensional Hilbert space (which has 6 real degrees of freedom for a general state up to global phase).

In practice, L = 2 provides over-parameterization (12 parameters for 6 degrees of freedom), which helps the optimizer avoid local minima.

### 4.5 Ansatz Selection Logic

```python
def select_ansatz(matrix: np.ndarray, num_qubits: int) -> QuantumCircuit:
    """
    Select appropriate ansatz based on matrix properties.

    Logic:
    1. If num_qubits == 1:
       - Use RY + RZ parameterization (2 parameters, universal for 1 qubit)
    2. If num_qubits == 2:
       - Use EfficientSU2 with reps=2 (12 parameters, universal for 2 qubits)
    """
```

---

## 5. Cost Function and Measurement Strategy

### 5.1 Energy Estimation

The VQE cost function is:

```
E(theta) = <psi(theta)| H |psi(theta)>
         = sum_i c_i * <psi(theta)| P_i |psi(theta)>                        (6)
```

Each Pauli expectation value <P_i> is estimated independently by:
1. Applying appropriate basis rotation gates
2. Measuring in the computational basis
3. Computing <P_i> from measurement statistics

### 5.2 Measurement Circuits

For each Pauli string P = P_1 (x) P_2 (x) ... (x) P_q, apply pre-measurement rotations:

| Pauli | Pre-rotation | Measurement maps to |
|:-----:|:------------:|:-------------------:|
| I     | None         | Always +1           |
| Z     | None         | \|0> -> +1, \|1> -> -1 |
| X     | H            | \|+> -> +1, \|-> -> -1 |
| Y     | S^dag then H | \|+i> -> +1, \|-i> -> -1 |

**Grouping**: Pauli strings that share the same tensor product basis (qubit-wise commuting group) can be measured simultaneously. For our test matrices:

- M1: 1 measurement circuit (Z)
- M2: 1 measurement circuit (X)
- M3: 3 measurement circuits (X, Y, Z share no common basis for simultaneous measurement)
- M4: Multiple circuits needed (up to 8 terms, some can be grouped)
- M5: 3 circuits (XX, YY, ZZ require different basis rotations)

### 5.3 Shot Budget Analysis

The standard error of the energy estimate from N_shots per Pauli term is bounded by:

```
SE(E) <= sqrt(sum_i c_i^2 / N_shots)                                        (7)
```

(This is an upper bound assuming worst-case variance Var(<P_i>) = 1.)

| Matrix | sum(c_i^2) excl. I | Shots for SE < 0.02 | Shots for SE < 0.05 |
|:------:|:------------------:|:--------------------:|:--------------------:|
| M1     | 1.00               | 2,500                | 400                  |
| M2     | 1.00               | 2,500                | 400                  |
| M3     | 2.25               | 5,625                | 900                  |
| M4 (p=10) | 12.19           | 30,469               | 4,875                |
| M5     | 3.00               | 7,500                | 1,200                |

**Recommended shot counts**:
- Ideal (statevector) simulation: shots=None (exact expectation values)
- Shot-based simulation: 4,096 shots per Pauli term
- Noisy simulation: 8,192 shots per Pauli term (extra shots compensate noise-induced variance)

### 5.4 Identity Term Handling

The identity term c_I * I contributes a constant c_I to the energy. It does not need to be measured (its expectation value is always 1). The implementation should add c_I analytically:

```
E(theta) = c_I + sum_{P != I} c_P * <psi(theta)|P|psi(theta)>               (8)
```

---

## 6. Classical Optimizer Specification

### 6.1 Optimizer Selection

| Optimizer | Type | Gradient | Best For | Max Iterations |
|:---------:|:----:|:--------:|:--------:|:--------------:|
| **COBYLA** | Gradient-free | None | Ideal simulation, small parameter space | 200 |
| **SPSA** | Stochastic gradient | Approximated | Noisy simulation | 300 |
| **L-BFGS-B** | Quasi-Newton | Exact or finite-diff | Statevector with exact gradients | 100 |

**Primary recommendation**: COBYLA for ideal simulation, SPSA for noisy simulation.

### 6.2 Convergence Criteria

```
Convergence when: |E_{k+1} - E_k| < tol    for consecutive iterations

Tolerances:
  - Ideal (statevector): tol = 1e-6
  - Shot-based (ideal): tol = 1e-4
  - Noisy: tol = 1e-3
```

### 6.3 Initial Parameter Strategy

Parameters are initialized as:

```
theta_init = uniform_random(0, pi) for each parameter                        (9)
```

For robustness, run multiple random restarts:
- 1-qubit systems: 3 restarts (select best)
- 2-qubit systems: 5 restarts (select best)

### 6.4 COBYLA Configuration

```python
from qiskit_algorithms.optimizers import COBYLA

optimizer = COBYLA(
    maxiter=200,         # Maximum iterations
    tol=1e-6,            # Convergence tolerance
    rhobeg=0.5           # Initial trust region radius (radians)
)
```

### 6.5 SPSA Configuration

```python
from qiskit_algorithms.optimizers import SPSA

optimizer = SPSA(
    maxiter=300,              # More iterations needed for noisy
    learning_rate=0.05,       # Step size (auto-calibrated if None)
    perturbation=0.1,         # Finite difference step
    last_avg=10               # Average last 10 iterates for stability
)
```

---

## 7. Eigenvalue Deflation (Finding All Eigenvalues)

### 7.1 Method

To find the k-th smallest eigenvalue after having found the first k-1 eigenstates |psi_0>, ..., |psi_{k-2}>:

```
H_k = H + sum_{j=0}^{k-2} alpha_j |psi_j><psi_j|                          (10)
```

where alpha_j is a positive penalty that shifts the energy of |psi_j> above the remaining spectrum.

### 7.2 Penalty Strength

```
alpha_j = 2 * (lambda_max - lambda_j)                                      (11)
```

This pushes the j-th eigenstate energy from lambda_j to:

```
lambda_j + alpha_j = lambda_j + 2*(lambda_max - lambda_j) = 2*lambda_max - lambda_j
```

which is guaranteed to be above lambda_max, ensuring it is no longer the ground state of H_k.

### 7.3 Deflation Protocol

```
Input: Hamiltonian H, number of eigenvalues to find K
Output: eigenvalues lambda_0, ..., lambda_{K-1}

1. Set H_0 = H
2. For k = 0, 1, ..., K-1:
   a. Run VQE on H_k to find |psi_k> and lambda_k
   b. Compute alpha_k = 2 * (lambda_max_est - lambda_k)
      where lambda_max_est is estimated from the Hamiltonian norm
   c. H_{k+1} = H_k + alpha_k |psi_k><psi_k|
3. Return {lambda_0, ..., lambda_{K-1}}
```

### 7.4 Implementing the Deflation Penalty in Pauli Form

The projector |psi_k><psi_k| must be expressed as a sum of Pauli operators for VQE evaluation.

For a statevector |psi_k> obtained from VQE, the density matrix rho_k = |psi_k><psi_k| can be decomposed:

```
|psi_k><psi_k| = (1/2^q) sum_P Tr(rho_k * P) * P                          (12)
```

This adds up to 4^q new Pauli terms to the Hamiltonian at each deflation step. For 2 qubits, this is at most 16 additional terms per deflation.

### 7.5 Deflation for Embedded 3x3 Matrices

When finding all eigenvalues of an embedded 3x3 matrix:
- Find K = 3 eigenvalues (the original matrix dimension)
- The 4th eigenvalue (penalty artifact) is discarded
- The penalty embedding value must exceed the deflation penalties (Eq. 5 ensures this)

### 7.6 Handling Degeneracies

For degenerate eigenvalues (e.g., M5 has triple degeneracy at lambda=1):
- VQE finds one arbitrary state from the degenerate subspace
- Deflation with this state may produce a slightly different energy than the exact degenerate eigenvalue due to numerical errors in the overlap
- **Mitigation**: Accept eigenvalues within the convergence tolerance. If two consecutive deflated eigenvalues differ by less than 2*tol, treat them as degenerate.

---

## 8. Correctness Proofs

### 8.1 Variational Principle

**Theorem (Variational Principle)**: For Hermitian operator H with ground state energy E_0:

```
<psi|H|psi> >= E_0    for all normalized |psi>                             (13)
```

with equality iff |psi> is a ground state of H.

**Proof**: Let {|k>, E_k} be the eigenbasis of H. Expand |psi> = sum_k a_k |k>. Then:

```
<psi|H|psi> = sum_k |a_k|^2 E_k >= E_0 sum_k |a_k|^2 = E_0
```

since |a_k|^2 >= 0 and E_k >= E_0 for all k. QED.

### 8.2 VQE Convergence

**Theorem**: If the ansatz U(theta) can represent the ground state |psi_0> (i.e., there exists theta* such that U(theta*)|0> = |psi_0>), and the classical optimizer converges to the global minimum, then VQE returns E_0 exactly.

**Proof**: By the variational principle, E(theta) >= E_0 for all theta. By hypothesis, E(theta*) = E_0. Since the optimizer converges to the global minimum of E(theta), it converges to theta* (or another parameter giving E_0). QED.

**Corollary**: For our 1-2 qubit systems, the EfficientSU2 ansatz with L >= 1 is universal, so the ansatz can represent any ground state. The global minimum of E(theta) equals E_0.

### 8.3 Pauli Decomposition Preserves Eigenvalues

**Theorem**: The Pauli decomposition H = sum_i c_i P_i has the same eigenvalues as the original matrix M.

**Proof**: The Pauli decomposition is an exact matrix identity: H = M (they are the same operator expressed in different bases). Therefore they have identical eigenvalues. QED.

### 8.4 Deflation Correctness

**Theorem**: Let |psi_0> be the ground state of H with eigenvalue E_0, and let H' = H + alpha |psi_0><psi_0| with alpha > E_{max} - E_0. Then the ground state of H' is the first excited state of H, with eigenvalue E_1.

**Proof**: For eigenstates |k> of H with k > 0: <k|psi_0> = 0 (orthogonality), so <k|H'|k> = <k|H|k> + alpha*|<k|psi_0>|^2 = E_k. For the ground state: <0|H'|0> = E_0 + alpha > E_0 + (E_{max} - E_0) = E_{max} >= E_k for all k. Therefore the ground state of H' is the state minimizing {E_k : k > 0} = E_1. QED.

---

## 9. Complexity Analysis

### 9.1 Resource Summary

| Resource | 1-qubit (2x2) | 2-qubit (3x3, 4x4) |
|:---------|:-------------:|:-------------------:|
| Qubits | 1 | 2 |
| Ansatz parameters | 2 | 12 (L=2) |
| CX gates per evaluation | 0 | 2 |
| Max Pauli terms | 4 | 16 |
| Measurement circuits | <= 3 | <= 9 |
| Shots per optimization step | 4,096 | 4,096 |
| Optimizer iterations | ~50-100 | ~100-200 |
| Total circuit executions (1 eigenvalue) | ~150-400 | ~900-3,600 |
| Total circuit executions (all eigenvalues) | ~300-800 | ~2,700-14,400 |

### 9.2 Comparison with Classical

| Metric | VQE (our implementation) | Classical (NumPy) |
|:-------|:------------------------:|:-----------------:|
| Time complexity | O(K * I * T * S) | O(n^3) |
| Space | O(2^q) quantum + O(params) classical | O(n^2) |
| Accuracy | <= 1% (ideal) | Machine epsilon |

Where K = number of eigenvalues, I = optimizer iterations, T = Pauli terms, S = shots.

For n <= 4, the classical solver is vastly more efficient. The purpose of this project is validation and pipeline development, not quantum advantage.

### 9.3 Gate Complexity per VQE Evaluation

**1-qubit circuits**:
- Ansatz: 2 rotation gates (RY, RZ)
- Measurement basis rotation: 0-2 gates per term
- Total per Pauli term: 2-4 gates
- Circuit depth: 2-4

**2-qubit circuits**:
- Ansatz: 12 rotation gates + 2 CX gates (L=2)
- Measurement basis rotation: 0-4 gates per term
- Total per Pauli term: 14-18 gates
- Circuit depth: 10-14

### 9.4 Scaling Discussion

While this project targets n <= 4, the asymptotic scaling is relevant for future work:

- **Qubit count**: q = ceil(log_2(n)) -- logarithmic in matrix size
- **Pauli terms**: Up to 4^q = n^2 -- quadratic in matrix size
- **Ansatz depth**: O(q * L) -- linear in qubits for fixed repetitions
- **Optimizer iterations**: Problem-dependent, typically O(poly(params))
- **Total cost**: O(n^2 * q * L * I * S) -- polynomial in n

This polynomial scaling is what makes VQE potentially advantageous for large systems where classical diagonalization (O(n^3)) becomes prohibitive, especially when the matrix has structure that reduces the number of Pauli terms.

---

## 10. Expected Behaviors and Edge Cases

### 10.1 Expected Behaviors (for Test Engineer)

| Behavior | Expected Outcome | Test Condition |
|:---------|:-----------------|:---------------|
| Pauli decomposition of identity | Only I term with coefficient 1 | c_I = 1, all others = 0 |
| Pauli decomposition reconstruction | M_reconstructed == M_original | np.allclose check |
| VQE on diagonal matrix | Converges in few iterations | M1: < 20 iterations |
| VQE on off-diagonal matrix | Creates superposition state | M2: final state ~ \|-> |
| VQE on complex matrix | Correct energy despite complex coefficients | M3: E = 1.0 |
| VQE on embedded 3x3 | Ignores penalty subspace | M4: E = 1.0, not 10.0 |
| VQE on entangled ground state | Creates Bell-like state | M5: E = -3.0 |
| Deflation finds excited states | Each eigenvalue within tolerance | Sequential accuracy |
| Degenerate eigenvalues | All found with correct multiplicity | M5: three values near 1.0 |
| Noisy simulation | Eigenvalue shifted toward spectral mean | Error < 5% of spectral range |

### 10.2 Edge Cases to Test

1. **Zero matrix**: H = 0. All eigenvalues are 0. VQE should converge to 0 regardless of initial parameters.

2. **Identity matrix**: H = I. All eigenvalues are 1. VQE should converge to 1 immediately. Deflation should return all 1s.

3. **Already-diagonal matrix**: The ansatz should converge quickly since the ground state is a computational basis state.

4. **Negative-definite matrix**: All eigenvalues negative. VQE should still find the most negative one.

5. **Large spectral range**: Matrix with eigenvalues {0, 1000}. Tests numerical stability of the optimizer.

6. **Near-degenerate eigenvalues**: Eigenvalues {1.0, 1.001}. Tests deflation's ability to resolve close eigenvalues.

7. **Hermiticity validation**: Non-Hermitian input should raise an error.

8. **Wrong dimensions**: 5x5 matrix (not power of 2) should trigger embedding. 7x7 matrix should use 3 qubits.

---

## 11. Summary for Downstream Agents

### For Test Engineer (Phase 3)

**Test the following interfaces and behaviors**:

1. `pauli_decompose(M)` -- Takes np.ndarray, returns dict of Pauli string coefficients
   - Verify reconstruction matches original matrix
   - Verify Hermiticity preservation
   - Test all 5 matrices plus edge cases

2. `QuantumEigensolver(M).solve()` -- Returns minimum eigenvalue
   - Test against classical eigenvalues with 1% tolerance (ideal)
   - Test with noise presets with 5% tolerance
   - Verify convergence within 200 iterations

3. `QuantumEigensolver(M).solve_all()` -- Returns all eigenvalues
   - Test deflation correctness for all test matrices
   - Test degenerate eigenvalue handling (M5)
   - Verify 3x3 embedding discards penalty eigenvalue

4. Input validation: Hermiticity check, dimension handling

### For Python Architect (Phase 4)

**Implement with these mathematical specifications**:

1. Pauli decomposition via Eq. (2): `c_P = Tr(M * P) / 2^q`
2. Embedding via Eq. (4) with penalty from Eq. (5)
3. EfficientSU2 ansatz with L=2 for 2-qubit systems
4. COBYLA optimizer (ideal), SPSA (noisy)
5. Deflation via Eq. (10) with penalty from Eq. (11)
6. Identity term handled analytically (Eq. 8)

### For Data Scientist (Phase 5)

**Expected baselines for comparison**:

| Matrix | Min Eigenvalue | All Eigenvalues | Spectral Range |
|:------:|:--------------:|:---------------:|:--------------:|
| M1 | -1 | {-1, 1} | 2 |
| M2 | -1 | {-1, 1} | 2 |
| M3 | 1 | {1, 4} | 3 |
| M4 | 1 | {1, 2, 4} | 3 |
| M5 | -3 | {-3, 1, 1, 1} | 4 |

---

## 12. Handoff Checklist

- [x] Hamiltonian encoding complete with formulas (Section 2)
- [x] Non-power-of-two embedding specified with penalty selection (Section 3)
- [x] Ansatz specified with circuit structure and parameter counts (Section 4)
- [x] Measurement strategy with shot budget analysis (Section 5)
- [x] Classical optimizer configuration (Section 6)
- [x] Eigenvalue deflation protocol (Section 7)
- [x] Correctness proofs for variational principle, decomposition, deflation (Section 8)
- [x] Complexity analysis with resource tables (Section 9)
- [x] Expected behaviors and edge cases documented (Section 10)
- [x] Downstream agent summaries (Section 11)

**Next**: Phase 3 (Test Design) -- Use Section 10 and 11 to create the TDD test suite.
