# Phase 2: Mathematical Formalization of Quantum Data Encoding Methods

## 1. Theoretical Framework

### 1.1 Quantum Feature Maps -- General Definition

A **quantum feature map** is a mapping from a classical data space to a quantum Hilbert space:

$$\phi: \mathcal{X} \subseteq \mathbb{R}^d \to \mathcal{H} = (\mathbb{C}^2)^{\otimes m}$$

realized by a parameterized unitary circuit:

$$|\phi(\mathbf{x})\rangle = U(\mathbf{x})|0\rangle^{\otimes m}$$

where $\mathbf{x} = (x_1, x_2, \ldots, x_d) \in \mathcal{X}$ is a classical feature vector, $m$ is the number of qubits, and $U(\mathbf{x}) \in SU(2^m)$ is a unitary operation parameterized by the data.

**Key properties of a feature map:**

1. **Injectivity**: Whether $\phi(\mathbf{x}) = \phi(\mathbf{x}')$ implies $\mathbf{x} = \mathbf{x}'$ (up to global phase). Determines information preservation.

2. **Continuity**: Whether small changes in $\mathbf{x}$ produce small changes in $|\phi(\mathbf{x})\rangle$. Important for generalization.

3. **Kernel function**: The feature map induces an implicit kernel:
$$K(\mathbf{x}, \mathbf{x}') = |\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle|^2$$

4. **Expressibility**: How uniformly the feature map covers the Hilbert space as $\mathbf{x}$ varies over $\mathcal{X}$.

### 1.2 Hilbert Space Dimension vs. Feature Dimension

For $m$ qubits, the Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes m}$ has dimension $2^m$. A pure state $|\psi\rangle \in \mathcal{H}$ is specified (up to global phase) by $2^{m+1} - 2$ real parameters. Different encodings use different submanifolds of this Hilbert space:

| Encoding | Qubits $m$ | Parameters used | Hilbert space dim. | State manifold dim. |
|----------|-----------|-----------------|---------------------|---------------------|
| Basis | $d$ (binary) | $d$ (discrete) | $2^d$ | Finite set ($2^d$ points) |
| Amplitude | $\lceil\log_2 d\rceil$ | $d - 1$ (continuous) | $d$ (padded to $2^m$) | $d - 1$ (unit sphere in $\mathbb{R}^d$) |
| Angle (product) | $d$ | $d$ (continuous) | $2^d$ | $d$-torus $T^d$ |
| IQP | $d$ | $d + \binom{d}{2}$ | $2^d$ | Submanifold of $2^d$-dim space |
| Re-uploading ($L$ layers) | $d$ | $L \cdot d$ (data) | $2^d$ | Up to full $SU(2^d)$ as $L \to \infty$ |

---

## 2. Basis Encoding

### 2.1 Definition

**Input constraint**: $\mathbf{x} = (b_1, b_2, \ldots, b_d) \in \{0, 1\}^d$ (binary data).

**Encoding map**:
$$|\phi(\mathbf{x})\rangle = |b_1 b_2 \cdots b_d\rangle = |b_1\rangle \otimes |b_2\rangle \otimes \cdots \otimes |b_d\rangle$$

**Circuit**: Apply $X$ gate to qubit $i$ if and only if $b_i = 1$:
$$U(\mathbf{x}) = \bigotimes_{i=1}^{d} X^{b_i}$$

### 2.2 Circuit Diagram (quantikz)

```
     ┌───────────┐
|0>──┤ X^{b_1}   ├── |b_1>
     └───────────┘
     ┌───────────┐
|0>──┤ X^{b_2}   ├── |b_2>
     └───────────┘
         ...
     ┌───────────┐
|0>──┤ X^{b_d}   ├── |b_d>
     └───────────┘
```

### 2.3 Properties

**Qubits**: $m = d$ (one qubit per binary feature).

**Circuit depth**: $O(1)$ -- all $X$ gates can be applied in parallel.

**Gate count**: At most $d$ single-qubit gates; zero two-qubit gates.

**Injectivity**: Perfectly injective -- distinct binary strings map to orthogonal computational basis states:
$$\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle = \delta_{\mathbf{x},\mathbf{x}'}$$

**Kernel**:
$$K(\mathbf{x}, \mathbf{x}') = |\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle|^2 = \delta_{\mathbf{x},\mathbf{x}'}$$

This is the **trivial kernel** (identity kernel), which provides zero generalization -- every data point is maximally dissimilar from every other point. This makes basis encoding alone useless for classification; it must be combined with a variational ansatz that creates meaningful state overlap.

**Encoding capacity**: Encodes exactly $2^d$ distinct data points in $d$ qubits.

**Data requirement**: Input must be binary. Continuous data requires a binarization preprocessing step (e.g., thresholding, one-hot encoding, binary discretization), which introduces information loss.

### 2.4 Binarization Preprocessing

For continuous features $x_j \in \mathbb{R}$, binarization options include:

1. **Threshold binarization**: $b_j = \mathbb{1}[x_j > \tau_j]$ for threshold $\tau_j$ (e.g., median). Reduces each feature to 1 bit.

2. **Multi-bit encoding**: Represent $x_j$ with $k$ bits using fixed-point arithmetic: $x_j \approx \sum_{l=0}^{k-1} b_{j,l} 2^{-l}$. Requires $k \cdot d$ qubits total.

3. **One-hot encoding**: For categorical features with $c$ categories, use $c$ binary qubits per feature.

### 2.5 Complexity Summary

| Resource | Scaling |
|----------|---------|
| Qubits | $d$ |
| Depth | $O(1)$ |
| Single-qubit gates | $\leq d$ |
| Two-qubit gates | $0$ |
| NISQ feasibility | Excellent |
| Data requirement | Binary or binarized input |

---

## 3. Amplitude Encoding

### 3.1 Definition

**Input constraint**: $\mathbf{x} = (x_1, x_2, \ldots, x_N) \in \mathbb{R}^N$ with $N = 2^m$ (pad with zeros if necessary).

**Preprocessing**: L2-normalize the input: $\tilde{\mathbf{x}} = \mathbf{x} / \|\mathbf{x}\|_2$.

**Encoding map**:
$$|\phi(\mathbf{x})\rangle = \sum_{i=0}^{2^m - 1} \tilde{x}_i |i\rangle$$

where $|i\rangle$ denotes the $i$-th computational basis state in $m$-qubit binary representation.

### 3.2 State Preparation Circuit

The general state preparation problem -- preparing an arbitrary $m$-qubit state -- requires constructing a unitary $U$ such that $U|0\rangle^{\otimes m} = |\phi(\mathbf{x})\rangle$.

**Mottonen decomposition** (Mottonen et al., 2004): Decomposes $U$ into a sequence of uniformly controlled rotations:

$$U = \prod_{k=1}^{m} UCR_y^{(k)} \cdot UCR_z^{(k)}$$

where $UCR^{(k)}$ denotes uniformly controlled rotations acting on qubit $k$, controlled by qubits $1, \ldots, k-1$.

**Circuit structure** (for 2 qubits encoding 4 features):

```
          ┌──────────┐     ┌───┐     ┌──────────┐
|0>──q0───┤ Ry(α_0)  ├──●──┤   ├──●──┤ Rz(γ_0)  ├──
          └──────────┘  │  │   │  │  └──────────┘
          ┌──────────┐  │  │   │  │  ┌──────────┐
|0>──q1───┤ Ry(α_1)  ├──⊕──┤Ry ├──⊕──┤ Rz(γ_1)  ├──
          └──────────┘     └───┘     └──────────┘
```

The rotation angles $\{\alpha_k, \gamma_k\}$ are computed from the target amplitudes $\{\tilde{x}_i\}$ via recursive decomposition.

### 3.3 Gate Count Analysis

For $m$ qubits (encoding $N = 2^m$ features):

| Resource | Exact count | Asymptotic |
|----------|-------------|------------|
| CNOT gates | $2^{m+1} - 2m - 2$ | $O(2^m) = O(N)$ |
| Single-qubit rotations | $2^{m+1} - 2$ | $O(N)$ |
| Circuit depth | $O(2^m)$ | $O(N)$ |

**Specific counts for relevant dimensions:**

| Features $N$ | Qubits $m$ | CNOT gates | Total gates | Depth |
|-------------|-----------|------------|-------------|-------|
| 4 | 2 | 2 | 6 | ~4 |
| 8 | 3 | 10 | 14 | ~10 |
| 16 | 4 | 26 | 30 | ~26 |
| 32 | 5 | 58 | 62 | ~58 |
| 64 | 6 | 122 | 126 | ~122 |
| 256 | 8 | 498 | 510 | ~498 |

### 3.4 Properties

**Qubits**: $m = \lceil\log_2 N\rceil$ -- **logarithmic** in feature dimension (major advantage).

**Injectivity**: Injective up to global phase and normalization. Two vectors $\mathbf{x}$ and $\mathbf{x}'$ map to the same state if and only if $\mathbf{x} = c\mathbf{x}'$ for some $c > 0$.

**Kernel**: The quantum kernel for amplitude encoding is:
$$K(\mathbf{x}, \mathbf{x}') = |\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle|^2 = \left(\frac{\mathbf{x} \cdot \mathbf{x}'}{\|\mathbf{x}\| \|\mathbf{x}'\|}\right)^2 = \cos^2(\theta_{\mathbf{x},\mathbf{x}'})$$

where $\theta_{\mathbf{x},\mathbf{x}'}$ is the angle between vectors $\mathbf{x}$ and $\mathbf{x}'$. This is the **squared cosine similarity** kernel -- a well-known classical kernel.

**Implication**: Amplitude encoding combined with measurement in the computational basis does not provide any quantum advantage in kernel computation, as $K(\mathbf{x}, \mathbf{x}')$ can be computed classically in $O(d)$ time.

**Normalization sensitivity**: Since amplitude encoding requires $\|\tilde{\mathbf{x}}\|_2 = 1$, the encoding is sensitive to the L2 norm of the original data. Two data points that differ only in magnitude map to the same quantum state. This means:
- Magnitude information is lost
- Only directional information is preserved
- Preprocessing must account for this if magnitude carries class-relevant information

### 3.5 Approximate Amplitude Encoding

Given the prohibitive depth of exact state preparation for large $N$, approximate methods are of practical interest:

**Variational state preparation**: Use a parameterized circuit $V(\theta)$ and optimize:
$$\min_\theta \| V(\theta)|0\rangle^{\otimes m} - |\phi(\mathbf{x})\rangle \|^2$$

This trades exact preparation for bounded-depth circuits at the cost of preparation fidelity and classical optimization overhead.

**Truncated Mottonen**: Apply only the first $k$ levels of the Mottonen decomposition, accepting an approximation error that decreases exponentially with $k$.

### 3.6 Complexity Summary

| Resource | Scaling |
|----------|---------|
| Qubits | $\lceil\log_2 d\rceil$ |
| Depth | $O(d)$ |
| Single-qubit gates | $O(d)$ |
| Two-qubit gates | $O(d)$ |
| NISQ feasibility | $d \leq 16$ (Heron); $d \leq 16$ (IonQ) |
| Data requirement | Real-valued, L2-normalized |

---

## 4. Angle Encoding (Rotation Encoding)

### 4.1 Definition

**Input constraint**: $\mathbf{x} = (x_1, x_2, \ldots, x_d)$ with $x_i \in [0, 2\pi)$ (after normalization).

**Encoding map** (for rotation axis $\alpha \in \{x, y, z\}$):
$$|\phi(\mathbf{x})\rangle = \bigotimes_{i=1}^{d} R_\alpha(x_i)|0\rangle$$

where $R_\alpha(\theta)$ is the single-qubit rotation about axis $\alpha$:

$$R_x(\theta) = e^{-i\theta X/2} = \begin{pmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$

$$R_y(\theta) = e^{-i\theta Y/2} = \begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$

$$R_z(\theta) = e^{-i\theta Z/2} = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

### 4.2 Explicit State for $R_y$ Encoding

For the standard $R_y$ variant (most common in QML literature):

$$|\phi(\mathbf{x})\rangle = \bigotimes_{i=1}^{d} \left(\cos\frac{x_i}{2}|0\rangle + \sin\frac{x_i}{2}|1\rangle\right)$$

In the computational basis:
$$|\phi(\mathbf{x})\rangle = \sum_{\mathbf{b} \in \{0,1\}^d} \prod_{i=1}^{d} \left(\cos\frac{x_i}{2}\right)^{1-b_i} \left(\sin\frac{x_i}{2}\right)^{b_i} |\mathbf{b}\rangle$$

### 4.3 Circuit Diagram

```
     ┌──────────┐
|0>──┤ R_y(x_1) ├──
     └──────────┘
     ┌──────────┐
|0>──┤ R_y(x_2) ├──
     └──────────┘
         ...
     ┌──────────┐
|0>──┤ R_y(x_d) ├──
     └──────────┘
```

### 4.4 Properties

**Qubits**: $m = d$ (one qubit per feature) -- **linear** scaling.

**Circuit depth**: $O(1)$ -- all rotations are applied in parallel.

**Gate count**: Exactly $d$ single-qubit gates; zero two-qubit gates.

**Product state structure**: The encoded state is a **product state** (unentangled). This means:
- Each qubit encodes one feature independently
- No inter-feature interactions are captured by the encoding itself
- Entanglement must be introduced by the subsequent variational ansatz
- The reduced density matrix of qubit $i$ is: $\rho_i = R_y(x_i)|0\rangle\langle 0|R_y^\dagger(x_i)$

**Periodicity**: Since rotations are $2\pi$-periodic, features must be mapped to $[0, 2\pi)$. More precisely, $R_y(x)$ has period $4\pi$ in the statevector but period $2\pi$ in measurement statistics (since global phase is unobservable). The effective encoding range is $[0, \pi]$ for distinguishing states by measurement, because $R_y(x)|0\rangle$ and $R_y(2\pi - x)|0\rangle$ give the same measurement statistics up to a relabeling of $|0\rangle$ and $|1\rangle$.

**Injectivity**: For $R_y$ encoding with $x_i \in (0, \pi)$, the encoding is injective (different feature vectors map to distinguishable states). At the boundary values $x_i \in \{0, \pi\}$, the states are computational basis states that "saturate" the encoding.

### 4.5 Kernel Derivation

**Theorem (Angle Encoding Kernel)**. The quantum kernel induced by $R_y$ angle encoding is:

$$K(\mathbf{x}, \mathbf{x}') = |\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle|^2 = \prod_{i=1}^{d} \cos^2\left(\frac{x_i - x_i'}{2}\right)$$

**Proof**:

The inner product between two encoded states is:
$$\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle = \prod_{i=1}^{d} \langle 0|R_y^\dagger(x_i) R_y(x_i')|0\rangle$$

For each qubit:
$$\langle 0|R_y^\dagger(x_i) R_y(x_i')|0\rangle = \langle 0|R_y(x_i' - x_i)|0\rangle = \cos\frac{x_i' - x_i}{2}$$

Therefore:
$$\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle = \prod_{i=1}^{d} \cos\frac{x_i' - x_i}{2}$$

And the kernel is:
$$K(\mathbf{x}, \mathbf{x}') = \left|\prod_{i=1}^{d} \cos\frac{x_i' - x_i}{2}\right|^2 = \prod_{i=1}^{d} \cos^2\frac{x_i - x_i'}{2} \qquad \square$$

**Classical equivalence**: This kernel is equivalent to the product of squared cosine functions on the feature differences. It can be computed classically in $O(d)$ time, so angle encoding alone does not provide a computational advantage in kernel evaluation.

**Kernel bandwidth**: The effective bandwidth of this kernel is fixed -- the kernel value decays as features diverge, with characteristic scale $\sim 1$ radian. The normalization preprocessing (mapping features to $[0, 2\pi]$ or $[0, \pi]$) effectively controls the bandwidth, analogous to the $\gamma$ parameter in an RBF kernel (Shaydulin & Wild, 2022).

### 4.6 Variants: $R_x$ and $R_z$ Encoding

**$R_x$ kernel**: $K_{R_x}(\mathbf{x}, \mathbf{x}') = \prod_{i=1}^{d} \cos^2\frac{x_i - x_i'}{2}$ (same functional form as $R_y$).

**$R_z$ kernel**:
$$\langle 0|R_z^\dagger(x_i) R_z(x_i')|0\rangle = \langle 0|R_z(x_i' - x_i)|0\rangle = e^{-i(x_i' - x_i)/2}$$

So:
$$K_{R_z}(\mathbf{x}, \mathbf{x}') = \left|\prod_{i=1}^{d} e^{-i(x_i' - x_i)/2}\right|^2 = 1$$

The $R_z$ kernel is **trivially constant** (always equals 1) because $R_z$ only introduces a phase on $|0\rangle$, which is unobservable. $R_z$ encoding requires Hadamard gates before/after to be useful, which is essentially the IQP encoding structure.

**Recommendation**: Use $R_y$ encoding as the default angle encoding variant. $R_x$ gives equivalent kernel properties. $R_z$ alone is not useful for classification.

### 4.7 Complexity Summary

| Resource | Scaling |
|----------|---------|
| Qubits | $d$ |
| Depth | $O(1)$ |
| Single-qubit gates | $d$ |
| Two-qubit gates | $0$ |
| NISQ feasibility | Excellent (up to $d \sim 20$) |
| Data requirement | Real-valued, normalized to $[0, 2\pi)$ |

---

## 5. IQP Encoding (Instantaneous Quantum Polynomial)

### 5.1 Definition

**Input**: $\mathbf{x} = (x_1, x_2, \ldots, x_d) \in \mathbb{R}^d$.

**Encoding map**:
$$|\phi(\mathbf{x})\rangle = U_{\text{IQP}}(\mathbf{x})|0\rangle^{\otimes d}$$

where:
$$U_{\text{IQP}}(\mathbf{x}) = H^{\otimes d} \cdot D(\mathbf{x}) \cdot H^{\otimes d}$$

and the diagonal unitary is:
$$D(\mathbf{x}) = \exp\left(i \sum_{i=1}^{d} x_i Z_i + i \sum_{1 \leq i < j \leq d} x_i x_j Z_i Z_j\right)$$

Since $Z_i$ and $Z_j$ commute for all $i, j$, the diagonal unitary factorizes:
$$D(\mathbf{x}) = \left(\prod_{i=1}^{d} R_z(2x_i)\right) \cdot \left(\prod_{1 \leq i < j \leq d} R_{zz}(2x_i x_j)\right)$$

where $R_{zz}(\theta) = \exp(-i\theta Z_i Z_j / 2)$.

### 5.2 Circuit Diagram (4 qubits)

```
     ┌───┐ ┌────────┐ ┌──────────────┐ ┌───┐
|0>──┤ H ├─┤Rz(2x1) ├─┤ RZZ(2x1*x2) ├─┤   ├─── ...  ──┤ H ├──
     └───┘ └────────┘ └──────────────┘ │   │            └───┘
     ┌───┐ ┌────────┐                   │   │
|0>──┤ H ├─┤Rz(2x2) ├─────────────────┤RZZ├─── ...  ──┤ H ├──
     └───┘ └────────┘                   │   │            └───┘
     ┌───┐ ┌────────┐                   └───┘
|0>──┤ H ├─┤Rz(2x3) ├────────────────── ...  ──┤ H ├──
     └───┘ └────────┘                            └───┘
     ┌───┐ ┌────────┐
|0>──┤ H ├─┤Rz(2x4) ├────────────────── ...  ──┤ H ├──
     └───┘ └────────┘                            └───┘
```

(All pairwise $R_{ZZ}$ interactions applied between the two Hadamard layers.)

### 5.3 Explicit State Derivation

Starting from $|0\rangle^{\otimes d}$:

**Step 1**: Apply $H^{\otimes d}$:
$$H^{\otimes d}|0\rangle^{\otimes d} = |+\rangle^{\otimes d} = \frac{1}{\sqrt{2^d}} \sum_{\mathbf{b} \in \{0,1\}^d} |\mathbf{b}\rangle$$

**Step 2**: Apply $D(\mathbf{x})$. Since $Z_i|\mathbf{b}\rangle = (-1)^{b_i}|\mathbf{b}\rangle$:
$$D(\mathbf{x})|\mathbf{b}\rangle = \exp\left(i \sum_i x_i (-1)^{b_i} + i \sum_{i<j} x_i x_j (-1)^{b_i + b_j}\right)|\mathbf{b}\rangle$$

Define $s_i = (-1)^{b_i} \in \{-1, +1\}$, so:
$$D(\mathbf{x})|\mathbf{b}\rangle = \exp\left(i \sum_i x_i s_i + i \sum_{i<j} x_i x_j s_i s_j\right)|\mathbf{b}\rangle = e^{i f(\mathbf{x}, \mathbf{s})}|\mathbf{b}\rangle$$

where $f(\mathbf{x}, \mathbf{s}) = \sum_i x_i s_i + \sum_{i<j} x_i x_j s_i s_j$.

**Step 3**: Apply $H^{\otimes d}$ again. The final state is:
$$|\phi(\mathbf{x})\rangle = \frac{1}{2^d} \sum_{\mathbf{b}} e^{i f(\mathbf{x}, \mathbf{s}(\mathbf{b}))} \sum_{\mathbf{c}} (-1)^{\mathbf{b} \cdot \mathbf{c}} |\mathbf{c}\rangle$$

This is a non-trivial superposition with interference between all $2^d$ computational basis states.

### 5.4 Properties

**Qubits**: $m = d$.

**Circuit depth**: The $R_{ZZ}$ gates between non-adjacent qubits require SWAP routing on limited-connectivity hardware:
- All-to-all connectivity: $O(d)$ depth (sequential pairwise interactions)
- Linear connectivity: $O(d^2)$ depth (SWAP overhead)
- Heavy-hex connectivity: $O(d^2/\text{degree})$ depth

**Gate count**:
- Hadamard gates: $2d$ (two layers)
- $R_z$ gates: $d$ (single-qubit diagonal)
- $R_{ZZ}$ gates: $\binom{d}{2} = d(d-1)/2$ (pairwise interactions)
- Total two-qubit gates: $d(d-1)/2$ (each $R_{ZZ}$ decomposes into 2 CNOT + 1 $R_z$)
- Total CNOT count: $d(d-1)$ (plus SWAP overhead for limited connectivity)

**Entanglement**: The IQP encoding creates genuine multi-qubit entanglement through the $R_{ZZ}$ interactions. The entanglement structure depends on the data values $\{x_i x_j\}$.

**Non-linearity**: The quadratic terms $x_i x_j$ in the diagonal create non-linear feature interactions, making IQP encoding fundamentally different from angle encoding (which is linear in features).

### 5.5 Kernel Derivation

**Theorem (IQP Kernel)**. The quantum kernel for IQP encoding is:

$$K(\mathbf{x}, \mathbf{x}') = \frac{1}{2^{2d}} \left|\sum_{\mathbf{s} \in \{-1,+1\}^d} e^{i\left(f(\mathbf{x}, \mathbf{s}) - f(\mathbf{x}', \mathbf{s})\right)}\right|^2$$

**Proof**:

$$\langle\phi(\mathbf{x})|\phi(\mathbf{x}')\rangle = \frac{1}{2^d} \sum_{\mathbf{b}} e^{i\left(f(\mathbf{x}', \mathbf{s}(\mathbf{b})) - f(\mathbf{x}, \mathbf{s}(\mathbf{b}))\right)}$$

Therefore:
$$K(\mathbf{x}, \mathbf{x}') = \left|\frac{1}{2^d} \sum_{\mathbf{s}} \exp\left(i\sum_k (x_k' - x_k) s_k + i\sum_{k<l}(x_k' x_l' - x_k x_l) s_k s_l\right)\right|^2 \qquad \square$$

**Key property**: Unlike the angle encoding kernel, the IQP kernel contains **cross-terms** $x_k x_l$ that make it non-factorizable over features. This means the IQP kernel captures feature interactions that the angle encoding kernel cannot.

**Classical hardness**: Computing the IQP kernel classically requires evaluating a sum over $2^d$ terms. While there is no known polynomial-time algorithm for this in general, approximate methods or special structure may allow efficient classical computation in some cases. The connection to IQP circuit simulation (Shepherd & Bremner, 2009) suggests that exact computation is #P-hard under reasonable complexity assumptions (Coyle et al., 2023).

### 5.6 Interaction Depth Variants

For practical NISQ implementation, we consider truncated IQP encodings:

**Nearest-neighbor IQP** (interaction depth 1): Only include $R_{ZZ}$ for adjacent qubits:
$$D_{\text{NN}}(\mathbf{x}) = \left(\prod_i R_z(2x_i)\right) \cdot \left(\prod_{i=1}^{d-1} R_{zz}(2x_i x_{i+1})\right)$$

- Two-qubit gates: $d - 1$ (linear in $d$)
- No SWAP overhead on linear/heavy-hex topology
- Captures only nearest-neighbor feature interactions

**$k$-local IQP**: Include $R_{ZZ}$ interactions only for feature pairs within distance $k$:
$$D_k(\mathbf{x}) = \left(\prod_i R_z(2x_i)\right) \cdot \left(\prod_{|i-j| \leq k} R_{zz}(2x_i x_j)\right)$$

- Two-qubit gates: $O(kd)$
- Interpolates between angle encoding ($k = 0$) and full IQP ($k = d$)

### 5.7 Complexity Summary

| Resource | Full IQP | NN-IQP | $k$-local IQP |
|----------|----------|--------|----------------|
| Qubits | $d$ | $d$ | $d$ |
| Depth (all-to-all) | $O(d)$ | $O(1)$ | $O(k)$ |
| Depth (heavy-hex) | $O(d^2)$ | $O(1)$ | $O(k^2)$ |
| Two-qubit gates | $d(d-1)/2$ | $d-1$ | $O(kd)$ |
| CNOT count | $d(d-1)$ | $2(d-1)$ | $O(kd)$ |
| NISQ feasibility | $d \leq 6$ (Heron) | $d \leq 20$ | Varies |

---

## 6. Data Re-uploading Encoding

### 6.1 Definition

**Input**: $\mathbf{x} = (x_1, x_2, \ldots, x_d) \in \mathbb{R}^d$ and trainable parameters $\boldsymbol{\theta} = (\theta_1, \ldots, \theta_p)$.

**Encoding map**:
$$|\phi(\mathbf{x}, \boldsymbol{\theta})\rangle = U(\mathbf{x}, \boldsymbol{\theta})|0\rangle^{\otimes m}$$

where:
$$U(\mathbf{x}, \boldsymbol{\theta}) = \prod_{l=1}^{L} \left[W(\boldsymbol{\theta}_l) \cdot S(\mathbf{x})\right]$$

Here:
- $S(\mathbf{x})$ is a **data encoding layer** (typically angle encoding): $S(\mathbf{x}) = \bigotimes_{i=1}^{d} R_y(x_i)$
- $W(\boldsymbol{\theta}_l)$ is a **trainable variational layer** with parameters $\boldsymbol{\theta}_l$
- $L$ is the number of re-uploading layers

### 6.2 Circuit Structure

The re-uploading circuit interleaves data encoding with trainable layers:

```
     ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
|0>──┤ Ry(x_1)  ├─┤ Ry(θ_1)  ├─┤   CX     ├─┤ Ry(x_1)  ├─┤ Ry(θ_5)  ├─┤   CX     ├── ...
     └──────────┘ └──────────┘ └────┬─────┘ └──────────┘ └──────────┘ └────┬─────┘
     ┌──────────┐ ┌──────────┐      │       ┌──────────┐ ┌──────────┐      │
|0>──┤ Ry(x_2)  ├─┤ Ry(θ_2)  ├──────⊕──────┤ Ry(x_2)  ├─┤ Ry(θ_6)  ├──────⊕──────── ...
     └──────────┘ └──────────┘              └──────────┘ └──────────┘
     |<--- Layer 1: S(x) + W(θ_1) --->|    |<--- Layer 2: S(x) + W(θ_2) --->|
```

### 6.3 Trainable Layer Structure

For this study, the trainable layer $W(\boldsymbol{\theta}_l)$ is defined as RealAmplitudes with reps=1 within each re-uploading block:

$$W(\boldsymbol{\theta}_l) = \text{Entangling}(\text{CX}) \cdot \bigotimes_{i=1}^{d} R_y(\theta_{l,i})$$

where the entangling layer applies CX gates in a linear chain: $\text{CX}_{1,2} \cdot \text{CX}_{2,3} \cdots \text{CX}_{d-1,d}$.

**Parameters per layer**: $d$ rotation angles + entangling gates (no parameters).
**Total trainable parameters**: $p = L \cdot d$.

### 6.4 Universal Approximation Property

**Theorem (Perez-Salinas et al., 2020)**. For a single qubit ($d = 1$) with data re-uploading, the model:
$$f(\mathbf{x}, \boldsymbol{\theta}) = \langle 0|U^\dagger(\mathbf{x}, \boldsymbol{\theta}) \, M \, U(\mathbf{x}, \boldsymbol{\theta})|0\rangle$$
is a **universal function approximator** as $L \to \infty$, where $M$ is an observable.

**Proof sketch**: Each re-uploading layer applies a rotation on the Bloch sphere. The composition of $L$ rotations with different axes and angles can approximate any continuous function from $\mathbb{R}$ to $[-1, 1]$. The proof uses the Stone-Weierstrass theorem: the set of trigonometric polynomials generated by $L$ layers is dense in the space of continuous functions.

**Multi-qubit extension**: For $d > 1$ qubits with entangling gates between layers, the re-uploading circuit can generate the full Lie algebra $\mathfrak{su}(2^d)$ (assuming the encoding + trainable layers generate sufficient Lie algebra elements). By the Solovay-Kitaev theorem, this implies universal approximation over $SU(2^d)$ as $L \to \infty$.

**Practical limitation**: The number of layers $L$ required for a given approximation accuracy $\epsilon$ may grow polynomially or exponentially depending on the target function's complexity. For our experiments with NISQ constraints, we limit $L \in \{1, 2, 3\}$.

### 6.5 Fourier Spectrum Analysis

The re-uploading model output can be expressed as a **truncated Fourier series** (Schuld, Sweke, & Meyer, 2021):

$$f(\mathbf{x}, \boldsymbol{\theta}) = \sum_{\boldsymbol{\omega} \in \Omega_L} c_{\boldsymbol{\omega}}(\boldsymbol{\theta}) \, e^{i \boldsymbol{\omega} \cdot \mathbf{x}}$$

where $\Omega_L$ is the set of accessible frequencies, which grows with $L$:
- $L = 1$: frequencies $\omega_i \in \{-1, 0, +1\}$ for each feature (linear model)
- $L = 2$: frequencies $\omega_i \in \{-2, -1, 0, +1, +2\}$ (quadratic model)
- $L$ layers: frequencies $\omega_i \in \{-L, \ldots, 0, \ldots, +L\}$

**Implication**: The number of re-uploading layers directly controls the model's frequency spectrum and therefore its capacity to represent high-frequency functions. This is the theoretical justification for data re-uploading's superiority over single-layer angle encoding for complex classification boundaries.

### 6.6 Effective Kernel

Since re-uploading circuits have trainable parameters, the induced kernel is parameter-dependent:
$$K(\mathbf{x}, \mathbf{x}'; \boldsymbol{\theta}) = |\langle\phi(\mathbf{x}, \boldsymbol{\theta})|\phi(\mathbf{x}', \boldsymbol{\theta})\rangle|^2$$

After training, the kernel adapts to the data distribution, potentially achieving better class separation than fixed-kernel encodings. This adaptive kernel perspective (Jerbi et al., 2023) explains why re-uploading can outperform kernel-based methods with fixed feature maps.

### 6.7 Complexity Summary

| Resource | $L$ layers |
|----------|-----------|
| Qubits | $d$ |
| Depth | $O(Ld)$ |
| Single-qubit gates | $2Ld$ ($Ld$ data + $Ld$ trainable) |
| Two-qubit gates | $L(d-1)$ (one CX chain per layer) |
| Trainable parameters | $Ld$ |
| Frequency spectrum | $\omega_i \in \{-L, \ldots, +L\}$ |
| NISQ feasibility | $L \leq 2$ for $d = 8$; $L \leq 3$ for $d = 4$ |

---

## 7. Comparative Analysis

### 7.1 Resource Comparison (for $d = 4$ and $d = 8$ features)

#### $d = 4$ features (Iris, reduced datasets)

| Encoding | Qubits | Depth | 1Q gates | 2Q gates | Total gates |
|----------|--------|-------|----------|----------|-------------|
| Basis | 4 | 1 | 4 | 0 | 4 |
| Amplitude | 2 | ~4 | 6 | 2 | 8 |
| Angle ($R_y$) | 4 | 1 | 4 | 0 | 4 |
| IQP (full) | 4 | ~10 | 12 | 12 | 24 |
| IQP (NN) | 4 | ~4 | 10 | 6 | 16 |
| Re-upload ($L=1$) | 4 | ~4 | 8 | 3 | 11 |
| Re-upload ($L=2$) | 4 | ~8 | 16 | 6 | 22 |
| Re-upload ($L=3$) | 4 | ~12 | 24 | 9 | 33 |

#### $d = 8$ features (PCA-reduced datasets)

| Encoding | Qubits | Depth | 1Q gates | 2Q gates | Total gates |
|----------|--------|-------|----------|----------|-------------|
| Basis | 8 | 1 | 8 | 0 | 8 |
| Amplitude | 3 | ~10 | 14 | 10 | 24 |
| Angle ($R_y$) | 8 | 1 | 8 | 0 | 8 |
| IQP (full) | 8 | ~36 | 24 | 56 | 80 |
| IQP (NN) | 8 | ~4 | 22 | 14 | 36 |
| Re-upload ($L=1$) | 8 | ~4 | 16 | 7 | 23 |
| Re-upload ($L=2$) | 8 | ~8 | 32 | 14 | 46 |

### 7.2 Theoretical Expressibility Hierarchy

Based on the state space coverage analysis:

$$\text{Basis} \prec \text{Angle} \prec \text{Amplitude} \prec \text{IQP (NN)} \prec \text{IQP (full)} \preceq \text{Re-upload}(L \to \infty)$$

where $\prec$ denotes "strictly less expressive" in terms of the set of reachable states.

**Justification**:
1. **Basis $\prec$ Angle**: Basis encoding reaches only $2^d$ discrete points; angle encoding reaches a continuous $d$-dimensional manifold.
2. **Angle $\prec$ Amplitude**: Angle encoding produces product states only; amplitude encoding can (in principle) reach any normalized state in $\mathbb{C}^{2^m}$.
3. **Angle $\prec$ IQP**: Angle encoding produces product states; IQP creates entangled states with non-linear feature interactions.
4. **IQP (NN) $\prec$ IQP (full)**: Nearest-neighbor IQP captures a subset of the pairwise interactions.
5. **IQP $\preceq$ Re-upload**: Re-uploading with sufficient layers can approximate the full $SU(2^d)$, while IQP is restricted to the IQP subclass. The comparison is not strict for finite $L$.

**Important caveat**: Higher expressibility does not guarantee better classification performance. Holmes et al. (2022) showed that highly expressive circuits suffer barren plateaus, and Thanasilp et al. (2024) showed that generic feature maps lead to exponential concentration of the kernel. The optimal encoding balances expressibility with trainability.

### 7.3 Kernel Properties Comparison

| Encoding | Kernel type | Feature interactions | Classical compute | Adaptive |
|----------|-------------|---------------------|-------------------|----------|
| Basis | Trivial ($\delta$) | None | $O(d)$ | No |
| Amplitude | Squared cosine similarity | All (linear) | $O(d)$ | No |
| Angle ($R_y$) | Product cosine | None (factorizable) | $O(d)$ | No |
| IQP | Non-factorizable | Pairwise quadratic | Likely #P-hard | No |
| Re-upload | Parameter-dependent | All (via training) | Quantum only | Yes |

### 7.4 Encoding Equivalences and Distinctions

**Proposition 1 (Angle $\subset$ Re-uploading)**. Angle encoding is a special case of data re-uploading with $L = 1$ and $W(\boldsymbol{\theta}_1) = I$ (identity trainable layer).

**Proposition 2 (IQP $\neq$ Angle)**. For $d \geq 2$, IQP encoding and angle encoding produce states from fundamentally different families:
- Angle encoding: product states (entanglement = 0 for all inputs)
- IQP encoding: entangled states (entanglement > 0 for generic inputs)

**Proof**: Consider $d = 2$ with $\mathbf{x} = (x_1, x_2)$ where $x_1 x_2 \neq 0$. The IQP state after the $R_{ZZ}$ interaction is entangled (the Schmidt decomposition has rank 2), while the angle encoded state $R_y(x_1)|0\rangle \otimes R_y(x_2)|0\rangle$ is by definition a product state. $\square$

**Proposition 3 (Amplitude normalization collapse)**. Two data points $\mathbf{x}$ and $c\mathbf{x}$ (for $c > 0$) map to the same amplitude-encoded state. No other encoding in our study has this property.

**Proposition 4 (Re-uploading frequency advantage)**. For $L \geq 2$ re-uploading layers, the model can represent functions with higher Fourier frequencies than single-layer angle encoding. Specifically, a single layer of angle encoding supports frequencies $\omega_i \in \{-1, 0, +1\}$, while $L$ layers support $\omega_i \in \{-L, \ldots, +L\}$.

---

## 8. Encoding-Data Compatibility Theory

### 8.1 When Basis Encoding is Appropriate

Basis encoding is theoretically optimal when:
1. Data is naturally binary or categorical (e.g., one-hot encoded features)
2. The classification problem depends on discrete feature combinations
3. No distance metric between feature values is needed (categorical reasoning)

**Incompatible data**: Continuous features, ordinal data where proximity matters, data with more than $\sim 20$ binary features (qubit limit).

### 8.2 When Amplitude Encoding is Appropriate

Amplitude encoding is theoretically optimal when:
1. Feature dimensionality $d$ is large relative to available qubits ($d \gg m$)
2. Only directional information matters (not magnitude)
3. Feature vectors have similar L2 norms across classes
4. The classification boundary depends on the angle between feature vectors

**Incompatible data**: Data where feature magnitude carries class information, very high-dimensional data ($d > 16$ on NISQ), data with many zero components (sparse data leads to low-fidelity states dominated by padding zeros).

### 8.3 When Angle Encoding is Appropriate

Angle encoding is theoretically optimal when:
1. Feature dimensionality is moderate ($d \leq 20$)
2. Features are approximately Gaussian or uniformly distributed (good use of the $[0, \pi]$ range)
3. Feature independence is high (the product kernel aligns with the data structure)
4. NISQ constraints limit circuit depth
5. The classification boundary is approximately linear in the feature space

**Incompatible data**: Very high-dimensional data ($d > 20$), data requiring non-linear feature interactions for classification, data with highly correlated features (product kernel is suboptimal for correlated features).

### 8.4 When IQP Encoding is Appropriate

IQP encoding is theoretically optimal when:
1. Non-linear feature interactions are important for classification
2. Feature correlations are significant and class-relevant
3. The data has moderate dimensionality ($d \leq 6$ on NISQ hardware)
4. Higher expressibility is needed than angle encoding provides

**Incompatible data**: High-dimensional data ($d > 8$ on any NISQ platform), data where features are independent (the quadratic terms $x_i x_j$ add noise rather than signal), data requiring deeper circuits than the hardware can support.

### 8.5 When Data Re-uploading is Appropriate

Data re-uploading is theoretically optimal when:
1. The classification boundary is highly non-linear
2. The data distribution has complex structure that benefits from adaptive kernels
3. Moderate circuit depth is acceptable ($L \leq 3$ for NISQ)
4. The number of features is small ($d \leq 8$ for $L = 2$)
5. Sufficient training data is available to optimize the $Ld$ additional parameters

**Incompatible data**: Very high-dimensional data combined with many layers (parameter explosion), extremely small datasets (overfitting risk due to large parameter count), cases where training convergence is an issue (the optimization landscape becomes harder with more layers).

### 8.6 Data-Encoding Compatibility Conditions (Formal)

**Definition (Compatibility Score)**. For a dataset with statistical profile $P = (d_{\text{eff}}, \bar{r}, \gamma, F_{\text{sep}}, S)$ where:
- $d_{\text{eff}}$ = effective dimensionality (PCA 95% variance)
- $\bar{r}$ = average absolute correlation
- $\gamma$ = average skewness magnitude
- $F_{\text{sep}}$ = Fisher separability
- $S$ = sparsity index

We define the encoding-data compatibility as a heuristic score:

**Angle compatibility**: $C_{\text{angle}}(P) \propto (1 - \bar{r}) \cdot \mathbb{1}[d_{\text{eff}} \leq 20]$

**IQP compatibility**: $C_{\text{IQP}}(P) \propto \bar{r} \cdot (1 - F_{\text{sep}}) \cdot \mathbb{1}[d_{\text{eff}} \leq 6]$

**Re-upload compatibility**: $C_{\text{reup}}(P) \propto (1 - F_{\text{sep}}) \cdot \mathbb{1}[d_{\text{eff}} \leq 8]$

**Amplitude compatibility**: $C_{\text{amp}}(P) \propto \mathbb{1}[d_{\text{eff}} > 8] \cdot (1 - S)$

These heuristic scores are to be validated experimentally in Phase 5. They encode the theoretical intuitions:
- Angle encoding prefers uncorrelated features (product kernel alignment)
- IQP encoding benefits from correlated features and low separability (needs non-linear interactions)
- Re-uploading benefits from low separability (needs adaptive boundaries)
- Amplitude encoding is the default for high-dimensional data where other encodings exceed qubit budgets

---

## 9. Encoding Under Noise

### 9.1 Noise Model

Under depolarizing noise with error rate $p$ per gate, the encoded density matrix is:

$$\rho_{\text{noisy}} = (1 - p_{\text{eff}})|\phi(\mathbf{x})\rangle\langle\phi(\mathbf{x})| + p_{\text{eff}} \frac{I}{2^m}$$

where $p_{\text{eff}}$ is the effective depolarization probability depending on the total gate count.

For a circuit with $n_{1Q}$ single-qubit gates and $n_{2Q}$ two-qubit gates:

$$p_{\text{eff}} \approx 1 - (1 - \epsilon_1)^{n_{1Q}} (1 - \epsilon_2)^{n_{2Q}}$$

### 9.2 Noise Impact by Encoding

**Encoding fidelity** $F = \langle\phi(\mathbf{x})|\rho_{\text{noisy}}|\phi(\mathbf{x})\rangle = (1 - p_{\text{eff}}) + p_{\text{eff}}/2^m$:

| Encoding ($d=4$) | $n_{2Q}$ | $F$ (Heron, $\epsilon_2 = 0.5\%$) | $F$ (Rigetti, $\epsilon_2 = 1.5\%$) |
|-------------------|---------|-------------------------------------|--------------------------------------|
| Basis | 0 | ~1.000 | ~1.000 |
| Angle | 0 | ~1.000 | ~1.000 |
| Amplitude | 2 | ~0.990 | ~0.970 |
| IQP (full) | 12 | ~0.942 | ~0.835 |
| Re-upload ($L=2$) | 6 | ~0.970 | ~0.913 |

| Encoding ($d=8$) | $n_{2Q}$ | $F$ (Heron) | $F$ (Rigetti) |
|-------------------|---------|-------------|---------------|
| Basis | 0 | ~1.000 | ~1.000 |
| Angle | 0 | ~1.000 | ~1.000 |
| Amplitude | 10 | ~0.951 | ~0.860 |
| IQP (full) | 56 | ~0.755 | ~0.429 |
| Re-upload ($L=2$) | 14 | ~0.932 | ~0.810 |

### 9.3 Noise-Induced Kernel Concentration

Under noise, the quantum kernel concentrates toward a constant (Thanasilp et al., 2024):

$$K_{\text{noisy}}(\mathbf{x}, \mathbf{x}') \approx (1 - p_{\text{eff}})^2 K(\mathbf{x}, \mathbf{x}') + p_{\text{eff}}(2 - p_{\text{eff}})/2^m$$

For large $p_{\text{eff}}$, the kernel approaches $1/2^m$ for all pairs, rendering it useless for classification. This effect is most severe for deep encodings (IQP, multi-layer re-uploading) and large qubit counts.

---

## 10. Summary and Handoff Notes

### 10.1 Key Theoretical Results

1. **Angle encoding** produces product states with a factorizable kernel $K = \prod_i \cos^2((x_i - x_i')/2)$. It is the most NISQ-friendly encoding but cannot capture feature interactions.

2. **IQP encoding** produces entangled states with a non-factorizable kernel that includes quadratic feature interactions $x_i x_j$. The kernel is likely classically hard to compute exactly (#P-hard), but the circuit depth scales quadratically with features.

3. **Data re-uploading** achieves universal approximation as $L \to \infty$, with each layer adding frequency components $\omega \in \{-L, \ldots, +L\}$ to the Fourier spectrum. It has adaptive kernels that train with the data.

4. **Amplitude encoding** provides logarithmic qubit efficiency but exponential circuit depth. Its kernel (squared cosine similarity) is classical and loses magnitude information.

5. **Basis encoding** maps binary data to orthogonal states with a trivial kernel. It is the shallowest encoding but requires binary input.

### 10.2 Encoding Selection Matrix

| Data property | Best encoding | Rationale |
|---------------|---------------|-----------|
| Binary/categorical | Basis | Natural binary representation |
| Low-dim ($d \leq 4$), Gaussian | Angle | Simple, sufficient, noise-resilient |
| Low-dim, non-linear boundaries | IQP or Re-upload ($L \geq 2$) | Need feature interactions |
| Medium-dim ($4 < d \leq 8$), correlated | IQP (NN) or Re-upload ($L=1$) | Balance depth and interactions |
| High-dim ($d > 8$) | PCA + Angle or PCA + Amplitude | Reduce first, encode second |
| Extreme dim ($d > 20$) | PCA to $d' \leq 8$ + Re-upload | Aggressive reduction needed |

### 10.3 Edge Cases and Limitations

1. **$R_z$ angle encoding is trivial**: Produces constant kernel $K = 1$. Do not use without Hadamard pre/post layers (which turns it into IQP).

2. **Amplitude encoding loses magnitude**: Two vectors $\mathbf{x}$ and $c\mathbf{x}$ ($c > 0$) map to the same state. If class labels depend on magnitude, amplitude encoding will fail.

3. **IQP encoding at $d = 8$ is NISQ-infeasible on superconducting hardware**: 56 CX gates on heavy-hex topology (with SWAP overhead, approximately 120 CX gates) produce fidelity below 0.76.

4. **Re-uploading overfitting risk**: With $L = 3$ layers and $d = 8$ features, there are 24 trainable parameters. For small datasets ($n < 100$), this may cause overfitting. The generalization gap metric in the evaluation framework addresses this.

5. **Kernel concentration at large $d$**: For any encoding, as $d$ grows, the kernel generically concentrates toward a constant, reducing classification power. This is independent of noise and is a fundamental limitation of quantum kernel methods (Thanasilp et al., 2024).

### 10.4 For the Test Engineer

The following behaviors should be tested for each encoding:
- **Correctness**: Verify encoded states match theoretical predictions for known inputs
- **Qubit count**: Verify `required_qubits(d)` returns correct values per the table in Section 7.1
- **Gate count**: Verify circuit gate counts match Section 7.1 tables
- **Kernel value**: For angle encoding, verify $K(\mathbf{x}, \mathbf{x}') = \prod_i \cos^2((x_i - x_i')/2)$ numerically
- **Entanglement**: IQP and re-uploading circuits should produce entangled states for generic inputs; angle and basis should produce product states
- **Normalization**: Amplitude encoding should reject un-normalized inputs or auto-normalize
- **Periodicity**: Angle encoding should handle $x_i$ outside $[0, 2\pi)$ gracefully

### 10.5 For the Python Architect

Key implementation considerations:
- $R_{ZZ}$ gate decomposition: $R_{ZZ}(\theta) = \text{CX}_{i,j} \cdot R_z(\theta)_j \cdot \text{CX}_{i,j}$
- Mottonen state preparation: Use Qiskit's `Initialize` or `StatePreparation` for amplitude encoding
- Re-uploading: Implement as alternating `ParameterVector`-based encoding and trainable layers
- IQP variants (full, NN, $k$-local) should be configurable via `interaction_depth` parameter

---

## References

1. Havlicek, V. et al. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature*, 567, 209-212.
2. Schuld, M. & Killoran, N. (2019). Quantum machine learning in feature Hilbert spaces. *PRL*, 122, 040504.
3. Sim, S., Johnson, P.D. & Aspuru-Guzik, A. (2019). Expressibility and entangling capability of parameterized quantum circuits. *Adv. Quantum Technol.*, 2(12), 1900070.
4. Perez-Salinas, A. et al. (2020). Data re-uploading for a universal quantum classifier. *Quantum*, 4, 226.
5. Mottonen, M. et al. (2004). Transformation of quantum states using uniformly controlled rotations. *Quantum Info. Comput.*, 5(6), 467-473.
6. Holmes, Z. et al. (2022). Connecting ansatz expressibility to gradient magnitudes and barren plateaus. *PRX Quantum*, 3, 010313.
7. Thanasilp, S. et al. (2024). Exponential concentration in quantum kernel methods. *Nature Communications*, 15, 1.
8. Shaydulin, R. & Wild, S.M. (2022). Importance of kernel bandwidth in quantum machine learning. *PRX Quantum*, 3, 040328.
9. Jerbi, S. et al. (2023). Quantum machine learning beyond kernel methods. *Nature Communications*, 14, 3751.
10. Schuld, M., Sweke, R. & Meyer, J.J. (2021). Effect of data encoding on the expressive power of variational quantum machine learning models. *Phys. Rev. A*, 103, 032430.
11. Shepherd, D. & Bremner, M.J. (2009). Temporally unstructured quantum computation. *Proc. R. Soc. A*, 465(2105), 1413-1439.
12. Coyle, B. et al. (2023). The Born supremacy: quantum advantage and training of an IQP circuit. *npj Quantum Information*, 6, 60.
