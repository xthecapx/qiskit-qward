# Phase 1: NISQ Constraints Assessment for Encoding Circuits

## 1. Overview

This document assesses the practical constraints that Noisy Intermediate-Scale Quantum (NISQ) hardware imposes on quantum data encoding circuits. These constraints directly influence which encoding methods are feasible, what circuit depths are achievable, and how noise degrades encoding fidelity. All assessments are based on current state-of-the-art hardware (2024-2025 specifications).

---

## 2. Current Hardware Landscape (2024-2025)

### 2.1 Superconducting Qubit Platforms

#### IBM Heron (ibm_torino, ibm_fez)

| Parameter | Value | Source |
|-----------|-------|--------|
| Qubit count | 133 (Heron R1/R2) | IBM Quantum roadmap |
| Qubit connectivity | Heavy-hex lattice | Fixed topology |
| Median T1 | ~300 us | IBM calibration data |
| Median T2 | ~150 us | IBM calibration data |
| Single-qubit gate error | ~2 x 10^-4 | Randomized benchmarking |
| Two-qubit gate (ECR) error | ~5 x 10^-3 | Randomized benchmarking |
| Single-qubit gate time | ~30 ns | IBM specifications |
| Two-qubit gate time | ~70 ns (ECR) | IBM specifications |
| Readout error | ~1 x 10^-2 | Assignment fidelity |
| Basis gates | {ECR, RZ, SX, X} | Native gate set |

#### IBM Eagle (ibm_brisbane, etc.)

| Parameter | Value |
|-----------|-------|
| Qubit count | 127 |
| Two-qubit gate error | ~1 x 10^-2 (CX) |
| Gate time (CX) | ~300 ns |

#### Rigetti Ankaa-3

| Parameter | Value |
|-----------|-------|
| Qubit count | 84 |
| Connectivity | Square lattice (tunable couplers) |
| T1 | ~20-40 us |
| T2 | ~10-25 us |
| Single-qubit gate error | ~5 x 10^-4 |
| Two-qubit gate (iSWAP/CZ) error | ~1.5 x 10^-2 |
| Gate time | ~40 ns (1Q), ~200 ns (2Q) |

### 2.2 Trapped Ion Platforms

#### IonQ Forte (via AWS Braket / Azure Quantum)

| Parameter | Value |
|-----------|-------|
| Qubit count | 36 (algorithmic qubits) |
| Connectivity | All-to-all |
| T1 | >10 s |
| T2 | ~1 s |
| Single-qubit gate fidelity | 99.97% |
| Two-qubit gate fidelity | 99.5% |
| Gate time | ~10 us (1Q), ~200 us (2Q) |

#### Quantinuum H2

| Parameter | Value |
|-----------|-------|
| Qubit count | 56 |
| Connectivity | All-to-all (QCCD architecture) |
| Two-qubit gate fidelity | 99.8% |
| T2 | >1 s |

### 2.3 Summary Comparison for Encoding

| Platform | Qubits | 2Q Fidelity | Connectivity | Max Practical Depth |
|----------|--------|-------------|--------------|---------------------|
| IBM Heron | 133 | 99.5% | Heavy-hex | ~60-100 2Q layers |
| IBM Eagle | 127 | 99.0% | Heavy-hex | ~20-30 2Q layers |
| Rigetti Ankaa | 84 | 98.5% | Square | ~15-20 2Q layers |
| IonQ Forte | 36 | 99.5% | All-to-all | ~50-80 2Q gates (serial) |
| Quantinuum H2 | 56 | 99.8% | All-to-all | ~100+ 2Q gates |

---

## 3. Encoding-Specific Constraint Analysis

### 3.1 Basis Encoding

**Circuit structure**: $X$ gates on qubits corresponding to 1-bits.

| Constraint | Assessment |
|------------|------------|
| Qubit count | $n_{qubits} = n_{binary\_features}$ |
| Circuit depth | $O(1)$ -- single layer of $X$ gates |
| Two-qubit gates | 0 |
| Noise impact | Minimal (depth-1 circuits have negligible decoherence) |
| NISQ feasibility | **Excellent** -- shallowest possible encoding |
| Limitation | Only encodes binary data; requires binarization preprocessing |

**NISQ verdict**: Fully feasible on all current hardware. The limiting factor is qubit count for the number of binary features, not noise.

### 3.2 Amplitude Encoding

**Circuit structure**: Arbitrary state preparation $|\psi\rangle = \sum_i \alpha_i |i\rangle$ requiring $O(2^n)$ CNOT gates in the general case.

| Constraint | Assessment |
|------------|------------|
| Qubit count | $n_{qubits} = \lceil\log_2(n_{features})\rceil$ -- logarithmic (excellent) |
| Circuit depth | $O(2^n)$ for general state; $O(n^2)$ with Mottonen decomposition |
| Two-qubit gates | $O(2^n)$ general; $O(n^2)$ Mottonen |
| Noise impact | **Severe** for high-dimensional data: 16 features requires 4 qubits but O(16) CNOTs |

**Practical depth analysis**:

| Features ($d$) | Qubits | CX gates (Mottonen) | Estimated error | Feasible? |
|----------------|--------|---------------------|-----------------|-----------|
| 4 | 2 | ~3 | ~1.5% | Yes (all platforms) |
| 8 | 3 | ~12 | ~6% | Yes (Heron, IonQ) |
| 16 | 4 | ~48 | ~24% | Marginal (Heron only) |
| 32 | 5 | ~192 | ~96% | **No** (infeasible on all NISQ) |
| 64 | 6 | ~768 | >100% | **No** |

**Error budget**: Assuming 0.5% per CX gate (Heron), the encoding error for $d$ features is approximately $1 - (1 - 0.005)^{O(d)} \approx 0.005 \cdot d$ for small $d$.

**NISQ verdict**: Feasible only for $d \leq 16$ features on best hardware (IBM Heron). Approximate amplitude encoding via variational state preparation may extend this but introduces training overhead.

### 3.3 Angle Encoding

**Circuit structure**: $R_y(x_i)$ on qubit $i$ for each feature $x_i$.

| Constraint | Assessment |
|------------|------------|
| Qubit count | $n_{qubits} = n_{features}$ -- linear (moderate) |
| Circuit depth | $O(1)$ -- single layer of parallel rotations |
| Two-qubit gates | 0 (product state encoding) |
| Noise impact | Minimal (single-qubit gates only; error ~0.02% per gate) |

**Practical qubit analysis**:

| Features ($d$) | Qubits needed | Feasible? | Notes |
|----------------|---------------|-----------|-------|
| 4 (Iris) | 4 | Yes (all platforms) | Trivial |
| 8 (after PCA) | 8 | Yes (all platforms) | Standard |
| 13 (Wine, Heart) | 13 | Yes (all platforms) | Comfortable |
| 16 (MNIST reduced) | 16 | Yes (all platforms) | Feasible |
| 30 (Cancer) | 30 | Yes (superconducting) | Needs PCA on IonQ |
| 561 (HAR) | 561 | **No** | Must reduce to <= 20 |

**NISQ verdict**: Feasible for $d \leq 20$ features without reduction. Requires PCA for higher dimensions. Excellent noise resilience due to zero two-qubit gate overhead.

### 3.4 IQP Encoding

**Circuit structure**: $H^{\otimes n} \cdot \prod_{i<j} R_{ZZ}(x_i x_j) \cdot \prod_i R_Z(x_i) \cdot H^{\otimes n}$

| Constraint | Assessment |
|------------|------------|
| Qubit count | $n_{qubits} = n_{features}$ -- linear |
| Circuit depth | $O(n^2/\text{connectivity})$ with all pairwise interactions |
| Two-qubit gates | $\binom{n}{2} = n(n-1)/2$ per layer |
| Noise impact | **Significant** -- quadratic in feature count |

**Practical depth analysis with connectivity overhead**:

On heavy-hex topology (IBM), non-adjacent qubits require SWAP gates. Each SWAP = 3 CX gates.

| Features ($d$) | RZZ gates | CX equiv (all-to-all) | CX equiv (heavy-hex) | Estimated error (Heron) | Feasible? |
|----------------|-----------|----------------------|---------------------|------------------------|-----------|
| 4 | 6 | 12 | ~18 | ~9% | Yes |
| 6 | 15 | 30 | ~60 | ~30% | Marginal |
| 8 | 28 | 56 | ~120 | ~60% | **No** (noiseless sim only) |
| 13 | 78 | 156 | ~400 | >100% | **No** |

On all-to-all topology (IonQ, Quantinuum), no SWAP overhead:

| Features ($d$) | RZZ gates | Estimated error (IonQ) | Feasible? |
|----------------|-----------|----------------------|-----------|
| 4 | 6 | ~3% | Yes |
| 8 | 28 | ~14% | Yes |
| 13 | 78 | ~39% | Marginal |

**NISQ verdict**: Feasible for $d \leq 6$ on superconducting hardware (heavy-hex); $d \leq 10$ on trapped-ion (all-to-all connectivity). Represents the most noise-sensitive encoding in our study.

### 3.5 Data Re-uploading

**Circuit structure**: $\prod_{l=1}^{L} [W(\theta_l) \cdot S(x)]$ where $S(x)$ is angle encoding and $W(\theta_l)$ is a trainable entangling layer.

| Constraint | Assessment |
|------------|------------|
| Qubit count | $n_{qubits} = n_{features}$ (or fewer with multiplexing) |
| Circuit depth | $O(L \times n)$ where $L$ = number of re-uploading layers |
| Two-qubit gates | $O(L \times n)$ from entangling layers |
| Noise impact | **Moderate to significant** -- scales linearly with $L$ |

**Practical analysis for $d = 4$ features (Iris)**:

| Layers ($L$) | Total depth | CX gates (heavy-hex) | Error (Heron) | Error (IonQ) | Feasible? |
|--------------|-------------|---------------------|---------------|--------------|-----------|
| 1 | ~6 | ~8 | ~4% | ~4% | Yes |
| 2 | ~12 | ~16 | ~8% | ~8% | Yes |
| 3 | ~18 | ~24 | ~12% | ~12% | Marginal |
| 5 | ~30 | ~40 | ~20% | ~20% | Marginal |
| 10 | ~60 | ~80 | ~40% | ~40% | No |

**Practical analysis for $d = 8$ features (after PCA)**:

| Layers ($L$) | CX gates (heavy-hex) | Error (Heron) | Feasible? |
|--------------|---------------------|---------------|-----------|
| 1 | ~16 | ~8% | Yes |
| 2 | ~32 | ~16% | Marginal |
| 3 | ~48 | ~24% | No |

**NISQ verdict**: Feasible with 1-2 layers for $d \leq 8$. The universal approximation property theoretically requires many layers, but NISQ constraints limit practical depth. Trade-off between expressibility and noise resilience.

---

## 4. Encoding Feasibility Summary

### 4.1 Overall Feasibility Matrix

| Encoding | Max Features (Heron) | Max Features (IonQ) | Max Features (Sim) | Primary Bottleneck |
|----------|---------------------|--------------------|--------------------|-------------------|
| Basis | ~100 | ~36 | ~25 (memory) | Qubit count (binary only) |
| Amplitude | ~16 | ~16 | ~20 | State preparation depth |
| Angle | ~20 | ~20 | ~25 (memory) | Qubit count (linear) |
| IQP | ~6 | ~10 | ~14 | Quadratic 2Q gates |
| Re-uploading (L=2) | ~8 | ~10 | ~16 | Linear depth x layers |

### 4.2 Recommended Qubit Budget for Experiments

Based on the feasibility analysis, we recommend the following standard qubit counts for experiments:

| Encoding | Target Qubits | Reason |
|----------|---------------|--------|
| Angle | 4, 8, 12, 16 | Covers Iris (native) through PCA-reduced datasets |
| IQP | 4, 6, 8 | Limited by quadratic scaling |
| Re-uploading | 4, 8 (L=1,2,3) | Balance layers vs qubits |
| Amplitude | 2, 3, 4 (= 4, 8, 16 features) | Limited by state preparation |
| Basis | 4, 8 (binarized) | After binarization preprocessing |

---

## 5. Noise Impact on Encoding Fidelity

### 5.1 Encoding Fidelity Model

The fidelity of an encoded state under depolarizing noise is approximately:

$$F_{enc} \approx \prod_{g \in \text{gates}} (1 - p_g)$$

where $p_g$ is the error rate for gate $g$.

For a circuit with $n_{1Q}$ single-qubit gates (error rate $\epsilon_1$) and $n_{2Q}$ two-qubit gates (error rate $\epsilon_2$):

$$F_{enc} \approx (1 - \epsilon_1)^{n_{1Q}} \cdot (1 - \epsilon_2)^{n_{2Q}}$$

For IBM Heron ($\epsilon_1 \approx 2 \times 10^{-4}$, $\epsilon_2 \approx 5 \times 10^{-3}$), two-qubit gates dominate the error budget by approximately 25x per gate.

### 5.2 Encoding Error Budget Table

Target: encoding fidelity $F_{enc} \geq 0.90$ (10% error budget for encoding alone, leaving room for ansatz and measurement errors).

| Encoding | Max CX gates for F >= 0.90 (Heron) | Max CX for F >= 0.90 (IonQ) |
|----------|--------------------------------------|------------------------------|
| Basis | N/A (no CX gates) | N/A |
| Angle | N/A (no CX gates) | N/A |
| IQP | ~21 CX -> d <= 4 | ~21 CX -> d <= 5 |
| Re-uploading | ~21 CX -> L=1 with d<=8 | ~21 CX -> L=2 with d<=4 |
| Amplitude | ~21 CX -> d <= ~14 | ~21 CX -> d <= ~14 |

### 5.3 Noise Mitigation Strategies

| Strategy | Description | Applicability |
|----------|-------------|---------------|
| Zero-noise extrapolation (ZNE) | Extrapolate results to zero noise | All encodings; reduces effective error by ~2-3x |
| Probabilistic error cancellation (PEC) | Quasi-probability decomposition | High overhead; useful for IQP/Re-uploading |
| Twirled readout error mitigation (TREX) | Correct measurement errors | All encodings; low overhead |
| Dynamical decoupling | Insert identity gates to suppress decoherence | Useful for idle qubits in encoding circuits |
| Circuit cutting | Decompose large circuits | IQP with many qubits |

### 5.4 QWARD Noise Presets for Experiments

The project will use QWARD's built-in noise presets:

```python
from qward.algorithms import get_preset_noise_config

# Available presets matching current hardware
presets = [
    "IBM-HERON-R1",   # ibm_torino-class: ~0.5% CX error
    "IBM-HERON-R2",   # ibm_fez-class: ~0.4% CX error
    "IBM-HERON-R3",   # Next-gen Heron: projected ~0.3% CX error
    "RIGETTI-ANKAA3",  # Ankaa-3: ~1.5% CZ error
]
```

---

## 6. Connectivity Constraints and SWAP Overhead

### 6.1 Heavy-Hex Topology (IBM)

The heavy-hex lattice has degree 2-3 connectivity. For IQP encoding, which requires all-to-all interactions, the SWAP overhead is significant:

- **Average SWAP distance** for $n$ qubits on heavy-hex: $\approx \sqrt{n}$
- **Each SWAP** = 3 CX gates
- **Overhead factor** for IQP: $\approx 3\sqrt{n}$ additional CX gates per RZZ interaction

**Mitigation**: Use Qiskit transpiler with optimization level 3 to minimize SWAP insertions. Consider routing-aware encoding circuit design.

### 6.2 Square Lattice (Rigetti)

- Degree-4 connectivity
- Lower SWAP overhead than heavy-hex for 2D-local interactions
- Still significant for IQP all-to-all interactions

### 6.3 All-to-All (Trapped Ion)

- No SWAP overhead
- Ideal for IQP encoding
- Limited by sequential gate execution (no native parallelism)

---

## 7. Practical Recommendations for This Study

### 7.1 Simulation-First Approach

Given NISQ constraints, our primary experiments will use noiseless simulation:

1. **Noiseless AerSimulator**: All 160 experiment configurations
2. **QWARD noise models**: Selected configurations (best-performing noiseless) with Heron and Ankaa-3 noise
3. **Real hardware** (if available): 2-3 configurations on IBM Quantum for validation

### 7.2 Feature Dimension Targets

| PCA Target | Qubit Count | Compatible Encodings |
|------------|-------------|---------------------|
| 4 components | 4 qubits | All encodings (including IQP, Re-uploading L=3) |
| 8 components | 8 qubits | Angle, Re-uploading (L=1-2), Amplitude (3 qubits) |
| 12 components | 12 qubits | Angle, Re-uploading (L=1), Amplitude (4 qubits) |
| 16 components | 16 qubits | Angle only (feasible); others marginal |

### 7.3 Recommended Experimental Tiers

**Tier 1 (Primary -- all datasets, noiseless)**:
- Angle encoding with 4, 8 qubits
- IQP encoding with 4, 6 qubits
- Re-uploading with 4, 8 qubits (L=1, 2)
- Amplitude encoding with 2, 3 qubits (4, 8 features)

**Tier 2 (Noise study -- best noiseless configurations)**:
- Top-3 encoding-dataset combinations per tier
- IBM Heron R2 noise model
- Rigetti Ankaa-3 noise model
- With and without ZNE error mitigation

**Tier 3 (Extended -- time permitting)**:
- Higher qubit counts (12, 16) for angle encoding
- Deeper re-uploading (L=3, 4)
- Real hardware execution on IBM Quantum

---

## 8. Constraint Impact on Hypotheses

| Hypothesis | NISQ Constraint Impact |
|------------|----------------------|
| H1 (Statistical structure) | Low impact -- statistical characterization is classical; encoding comparison is feasible in simulation |
| H2 (Classical preprocessing) | Low impact -- preprocessing is classical; reduced dimensions help NISQ feasibility |
| H3 (Encoding advantages) | **High impact** -- IQP and Re-uploading are most constrained by noise; their theoretical advantages may not survive NISQ noise |
| H4 (Real-world data) | **Moderate impact** -- real-world datasets tend to be higher-dimensional, requiring more aggressive PCA which may reduce encoding differences |

### Key Risk: IQP Encoding Viability

The quadratic CX gate scaling of IQP encoding is the primary NISQ risk in our study. With 8 features, IQP requires ~56 CX gates (all-to-all) or ~120+ CX gates (heavy-hex), exceeding the fidelity budget. This means:

1. **IQP experiments may be limited to 4-6 features** on noisy simulations
2. **IQP advantages seen in noiseless simulation may vanish under noise**
3. **This is itself an important finding** -- theoretical encoding advantages vs. NISQ practicality

---

## 9. Summary

| Finding | Implication for Study |
|---------|----------------------|
| Angle encoding is most NISQ-friendly | Serve as primary baseline; test up to 16 qubits |
| IQP is most noise-sensitive | Limit to 4-6 qubits in noise studies; document degradation |
| Re-uploading depth is limited | Test L=1,2 on hardware noise; L=3+ noiseless only |
| Amplitude encoding depth is feature-dependent | Practical for d <= 16 features only |
| Connectivity matters for IQP | Report results for both heavy-hex and all-to-all |
| PCA reduction is necessary for most real-world datasets | Study PCA variance retention vs. encoding performance |
| Noiseless simulation is the primary experimental mode | Noise study is secondary, focusing on top configurations |
