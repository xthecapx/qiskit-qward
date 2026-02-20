# Phase 2 Report: Theoretical Design

## Summary

Phase 2 established the rigorous mathematical foundations for all five quantum data encoding methods under study, analyzed their expressibility and kernel properties, formalized classical preprocessing theory, and designed the controlled experimental framework. The work was completed by the quantum-computing-researcher agent. Four deliverables were produced: encoding theory, expressibility analysis, preprocessing theory, and experimental design. Key theoretical results include kernel derivations for all encodings, a proof that IQP encoding is strictly more expressive than angle encoding for d >= 2, Fourier spectrum analysis of data re-uploading, and a formal analysis of how preprocessing interacts with encoding kernel properties.

## Token Usage

| Metric | Count |
|--------|-------|
| Input tokens | ~120,000 |
| Output tokens | ~40,000 |
| Total tokens | ~160,000 |
| Estimated cost | ~$4.80 (Opus: $1.80 input + $3.00 output) |
| Agents spawned | 0 |
| Agent sessions | quantum-computing-researcher (sole agent) |

**Note**: Token counts are estimates. The researcher agent read all Phase 1 deliverables (~25K tokens input) plus the full plan.md (~10K tokens) before producing the four Phase 2 documents. All work was conducted from the agent's theoretical quantum computing knowledge.

## Key Findings

### Encoding Theory (phase2_encoding_theory.md)
- All five encoding methods formalized with explicit circuit definitions, Dirac notation, and gate decompositions
- Kernel derivations for each encoding:
  - Angle ($R_y$): $K = \prod_i \cos^2((x_i - x_i')/2)$ -- factorizable, classically computable
  - IQP: Non-factorizable kernel involving $2^d$ terms, likely #P-hard to compute classically
  - Amplitude: Squared cosine similarity $K = \cos^2(\theta)$ -- classical
  - Basis: Trivial delta kernel
  - Re-uploading: Adaptive (parameter-dependent) kernel
- $R_z$ angle encoding produces a trivially constant kernel ($K = 1$ for all pairs) and should not be used alone
- Amplitude encoding loses magnitude information due to L2 normalization
- Resource tables computed for d=4 and d=8 features

### Expressibility Analysis (phase2_expressibility_analysis.md)
- Expressibility hierarchy established: Basis << Angle << IQP << Re-upload($L \to \infty$)
- **Proof**: IQP encoding is strictly more expressive than angle encoding for $d \geq 2$ (Section 8)
  - Angle encoding produces only product states (entanglement = 0)
  - IQP produces entangled states for generic inputs
  - The proof uses the entanglement distinction
- Kernel concentration analysis: all encodings face concentration at large qubit counts
  - IQP concentrates faster ($O(1/4^d)$) despite higher expressibility
  - This limits IQP's practical advantage on many qubits
- Barren plateau susceptibility ranked: Angle+RA (lowest risk) < Re-upload < IQP+RA (highest risk)
- Kernel-target alignment proposed as the primary encoding selection criterion
- Fourier spectrum of re-uploading: $L$ layers support frequencies $\omega \in \{-L, ..., +L\}$

### Preprocessing Theory (phase2_preprocessing_theory.md)
- **Critical finding**: PCA preprocessing may neutralize IQP encoding's advantage over angle encoding
  - PCA decorrelates features, removing the correlations that IQP's quadratic terms exploit
  - Testable prediction: PCA + angle $\approx$ PCA + IQP in accuracy
- MinMax normalization is sensitive to outliers, compressing most data into narrow encoding range
- Z-score + sigmoid provides more robust normalization for heavy-tailed distributions
- L2 normalization (for amplitude encoding) destroys magnitude information
- Preprocessing controls the effective kernel bandwidth, analogous to RBF gamma parameter
- Data re-uploading is predicted to be the most preprocessing-robust encoding

### Experimental Design (phase2_experimental_design.md)
- 160 nominal configurations (5 encodings x 4 preprocessings x 8 datasets)
- ~125 valid configurations after exclusion rules
- Tier 1/2/3 execution strategy for efficient resource use
- Statistical analysis plan: Friedman tests with Nemenyi post-hoc, Bonferroni correction
- Meta-model: Random Forest to predict best encoding from data profile (LOO-CV)
- Estimated total computation: ~10-18 hours on AerSimulator

## Methodology

1. **Read Phase 1 deliverables**: Reviewed all four Phase 1 documents to understand the research context, dataset profiles, evaluation metrics, and NISQ constraints.

2. **Encoding formalization**: For each encoding method, derived:
   - Explicit state vector in Dirac notation
   - Circuit diagram with gate sequence
   - Kernel function via inner product calculation
   - Resource counts (qubits, depth, gate count)
   - Data compatibility conditions

3. **Expressibility analysis**: Applied the Sim et al. (2019) framework:
   - Computed analytical fidelity distributions for each encoding
   - Estimated KL divergence from Haar-random distribution
   - Analyzed entanglement capability via Meyer-Wallach measure
   - Proved expressibility hierarchy via entanglement argument

4. **Preprocessing theory**: Analyzed how classical transformations modify:
   - The effective kernel function
   - The kernel bandwidth
   - The information content relevant to classification
   - The barren plateau susceptibility

5. **Experimental design**: Specified:
   - Complete variable taxonomy (independent, controlled, dependent)
   - Exclusion rules for invalid configurations
   - Per-configuration execution protocol
   - Statistical analysis plan with specific tests

## Results

### Phase 2 Deliverables Produced

| Deliverable | File | Content |
|-------------|------|---------|
| Encoding Theory | `phase2_encoding_theory.md` | 5 encoding methods formalized; kernel derivations; resource analysis; compatibility conditions |
| Expressibility Analysis | `phase2_expressibility_analysis.md` | Expressibility bounds; proof of IQP > Angle; kernel theory; capacity bounds; barren plateau analysis |
| Preprocessing Theory | `phase2_preprocessing_theory.md` | 4 normalization schemes; PCA interaction analysis; preprocessing pipeline; failure modes |
| Experimental Design | `phase2_experimental_design.md` | 160 configurations; execution protocol; statistical analysis plan; circuit definitions |
| Phase Report | `reports/phase2_report.md` | This document |

### Theoretical Predictions to Test

| Prediction | Source | How to test |
|-----------|--------|-------------|
| IQP more expressive than Angle | Expressibility analysis, Section 8 | Compute KL divergence; verify IQP < Angle |
| PCA neutralizes IQP advantage | Preprocessing theory, Section 4.2 | Compare PCA+IQP vs PCA+Angle accuracy |
| Re-uploading most preprocessing-robust | Preprocessing theory, Section 5.4 | Smallest accuracy variance across preprocessings |
| Z-score beats MinMax on heavy-tailed data | Preprocessing theory, Section 3.1-3.2 | Compare on Credit Fraud, NSL-KDD |
| Kernel alignment predicts accuracy | Expressibility analysis, Section 6 | Correlate $A(K, Y)$ with test accuracy |
| IQP highest barren plateau risk | Expressibility analysis, Section 7 | Compare gradient variance across encodings |

## Figures & Tables

No figures were generated in Phase 2 (this is a theoretical/design phase). Key tables produced:

- **Table 1**: Resource comparison for d=4 and d=8 (encoding_theory.md, Section 7.1)
- **Table 2**: Expressibility summary (expressibility_analysis.md, Section 2.6)
- **Table 3**: Kernel properties comparison (encoding_theory.md, Section 7.3)
- **Table 4**: Noise impact by encoding (encoding_theory.md, Section 9.2)
- **Table 5**: Optimal preprocessing by encoding (preprocessing_theory.md, Section 5.4)
- **Table 6**: Valid configuration count (experimental_design.md, Section 3.3)
- **Table 7**: PCA variance retention by dataset (preprocessing_theory.md, Section 4.3)

## LaTeX-Ready Content

### Key Equations

**Quantum Feature Map**:
$$|\phi(\mathbf{x})\rangle = U(\mathbf{x})|0\rangle^{\otimes m}$$

**Angle Encoding**:
$$|\phi(\mathbf{x})\rangle = \bigotimes_{i=1}^{d} R_y(x_i)|0\rangle = \bigotimes_{i=1}^{d}\left(\cos\frac{x_i}{2}|0\rangle + \sin\frac{x_i}{2}|1\rangle\right)$$

**Angle Encoding Kernel** (Theorem):
$$K_{\text{angle}}(\mathbf{x}, \mathbf{x}') = \prod_{i=1}^{d} \cos^2\left(\frac{x_i - x_i'}{2}\right)$$

**IQP Encoding**:
$$U_{\text{IQP}}(\mathbf{x}) = H^{\otimes d} \cdot \exp\left(i\sum_{i<j} x_i x_j Z_i Z_j + i\sum_i x_i Z_i\right) \cdot H^{\otimes d}$$

**IQP Kernel**:
$$K_{\text{IQP}}(\mathbf{x}, \mathbf{x}') = \left|\frac{1}{2^d}\sum_{\mathbf{s} \in \{-1,+1\}^d} e^{i(f(\mathbf{x},\mathbf{s}) - f(\mathbf{x}',\mathbf{s}))}\right|^2$$

**Data Re-uploading**:
$$U(\mathbf{x}, \boldsymbol{\theta}) = \prod_{l=1}^{L} W(\boldsymbol{\theta}_l) S(\mathbf{x}), \quad S(\mathbf{x}) = \bigotimes_i R_y(x_i)$$

**Fourier Spectrum**:
$$f(\mathbf{x}, \boldsymbol{\theta}) = \sum_{\boldsymbol{\omega} \in \Omega_L} c_{\boldsymbol{\omega}}(\boldsymbol{\theta}) e^{i\boldsymbol{\omega}\cdot\mathbf{x}}, \quad |\omega_i| \leq L$$

**Amplitude Encoding Kernel**:
$$K_{\text{amp}}(\mathbf{x}, \mathbf{x}') = \left(\frac{\mathbf{x} \cdot \mathbf{x}'}{\|\mathbf{x}\|\|\mathbf{x}'\|}\right)^2 = \cos^2(\theta_{\mathbf{x},\mathbf{x}'})$$

**Expressibility** (Sim et al., 2019):
$$\text{Expr}(U) = D_{KL}\left(\hat{P}_U(F) \| P_{\text{Haar}}(F)\right), \quad P_{\text{Haar}}(F) = (2^m - 1)(1-F)^{2^m - 2}$$

**Kernel-Target Alignment**:
$$A(K, Y) = \frac{\langle K, Y\rangle_F}{\|K\|_F \|Y\|_F}$$

**Encoding Fidelity Under Noise**:
$$F_{\text{enc}} \approx (1 - \epsilon_1)^{n_{1Q}} \cdot (1 - \epsilon_2)^{n_{2Q}}$$

**Noisy Kernel Concentration**:
$$K_{\text{noisy}}(\mathbf{x}, \mathbf{x}') \approx (1 - p_{\text{eff}})^2 K(\mathbf{x}, \mathbf{x}') + \frac{p_{\text{eff}}(2 - p_{\text{eff}})}{2^m}$$

## References

See individual Phase 2 deliverables for full reference lists. Key references for Phase 2:

1. Sim, S., Johnson, P.D. & Aspuru-Guzik, A. (2019). Expressibility and entangling capability. *Adv. Quantum Technol.*, 2(12), 1900070.
2. Perez-Salinas, A. et al. (2020). Data re-uploading for a universal quantum classifier. *Quantum*, 4, 226.
3. Holmes, Z. et al. (2022). Connecting ansatz expressibility to gradient magnitudes. *PRX Quantum*, 3, 010313.
4. Thanasilp, S. et al. (2024). Exponential concentration in quantum kernel methods. *Nature Communications*, 15, 1.
5. Shaydulin, R. & Wild, S.M. (2022). Importance of kernel bandwidth. *PRX Quantum*, 3, 040328.
6. Larocca, M. et al. (2023). Group-theoretic framework for parametrized quantum circuits. *Nature Reviews Physics*, 5, 729-737.
7. Schuld, M., Sweke, R. & Meyer, J.J. (2021). Effect of data encoding on expressive power. *Phys. Rev. A*, 103, 032430.
8. Bowles, J. et al. (2024). Effect of data encoding on expressive power. arXiv:2309.11225.
9. Jerbi, S. et al. (2023). Quantum machine learning beyond kernel methods. *Nature Communications*, 14, 3751.
10. Heredge, J. et al. (2024). The role of classical preprocessing in QML pipelines. *Quantum Machine Intelligence*.
11. Mottonen, M. et al. (2004). Transformation of quantum states using uniformly controlled rotations. *QIC*, 5(6), 467-473.
12. Havlicek, V. et al. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature*, 567, 209-212.

## Handoff Notes

### What Phase 3 (Test Engineer) Needs to Know

1. **Encoding behaviors to test** (from encoding_theory.md, Section 10.4):
   - Verify encoded states match theoretical predictions for known inputs
   - Verify qubit/gate counts match the resource tables
   - Verify angle encoding kernel $K = \prod_i \cos^2((x_i - x_i')/2)$
   - Verify IQP produces entangled states; angle produces product states
   - Verify amplitude encoding auto-normalizes or rejects unnormalized input

2. **Preprocessing behaviors to test** (from preprocessing_theory.md, Section 8.2):
   - MinMax maps to $[0, \pi]$
   - Z-score + sigmoid maps to $(0, \pi)$
   - PCA + MinMax produces decorrelated, bounded features
   - L2 normalization produces unit-norm vectors

3. **Experimental protocol to test** (from experimental_design.md, Section 11.1):
   - Configuration validation (exclusion rules)
   - Reproducibility (same seed = same results)
   - Metric computation correctness
   - Edge case handling

### What Phase 4 (Python Architect) Needs to Know

1. **Encoding implementation details** (from encoding_theory.md, Section 10.5):
   - $R_{ZZ}$ decomposition: CX + Rz + CX
   - Amplitude encoding via Qiskit's `StatePreparation`
   - Re-uploading as alternating ParameterVector layers
   - IQP variants (full, NN, k-local) via `interaction_depth` parameter

2. **Key data structures**:
   - `ExperimentConfig` and `ExperimentResult` dataclasses (experimental_design.md, Section 8)
   - Results CSV format (experimental_design.md, Section 8.2)
   - Pipeline architecture (preprocessing_theory.md, Section 6)

### Open Questions for Future Phases

1. Should we include a 6th encoding method (e.g., Hamiltonian encoding)?
   - **Recommendation**: No, 5 encodings provide sufficient coverage for the study scope.

2. Should hybrid encoding be used for mixed-type datasets?
   - **Recommendation**: Not in primary experiments (adds confounding variable). Document as future work.

3. What is the minimum L for universal approximation at d=4?
   - **Answer**: Theoretically $L = O(4^d/d) \approx 64$, far beyond NISQ limits. Practical L=2-3 provides quadratic/cubic frequency spectrum, which is sufficient for most classification tasks.

4. Can kernel bandwidth be analytically optimized?
   - **Partial answer**: The bandwidth depends on normalization range. Optimal range satisfies $a^* \approx \pi/(\sqrt{d} \cdot \text{median inter-class distance})$. Experimental validation needed.

---

*Phase 2 completed by: quantum-computing-researcher*
*Date: 2026-02-19*
*Status: Complete -- ready for Phase 3 handoff*
