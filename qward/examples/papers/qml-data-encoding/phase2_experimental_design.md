# Phase 2: Controlled Experimental Design

## 1. Experimental Objective

Quantify the effect of quantum data encoding methods on QML classification performance, controlling for model architecture, optimizer, and hardware, to test hypotheses H1-H4.

---

## 2. Experimental Variables

### 2.1 Independent Variables (Systematically Varied)

| Variable | Levels | Values |
|----------|--------|--------|
| Encoding method | 5 | Basis, Amplitude, Angle ($R_y$), IQP (full), Data Re-uploading ($L=2$) |
| Classical preprocessing | 4 | None, MinMax $[0, \pi]$, Z-score + sigmoid, PCA + MinMax |
| Dataset | 8 | Iris, Wine, Cancer, MNIST, Credit Fraud, NSL-KDD, HAR, Heart Disease |

**Total configurations**: $5 \times 4 \times 8 = 160$

### 2.2 Controlled Variables (Fixed)

| Variable | Value | Justification |
|----------|-------|---------------|
| QML model | VQC (Variational Quantum Classifier) | Standard model; isolates encoding effects |
| Ansatz | RealAmplitudes, reps=2 | Provides moderate expressibility without dominating encoding effects |
| Optimizer | COBYLA, maxiter=200 | Gradient-free; stable for noisy objectives |
| Shots | 1024 | Standard NISQ simulation; balances accuracy and speed |
| Train/test split | 80/20, stratified | Standard practice |
| Cross-validation | 5-fold stratified (seeds 42-46) | Robust performance estimation |
| Random seed | 42 (primary), with 4 additional seeds | Reproducibility and variance estimation |
| Simulator | AerSimulator (statevector for kernels) | Noiseless baseline |
| PCA target (when used) | $k = \min(d, 8)$ | Fits NISQ qubit budgets |
| Feature count after preprocessing | 4 or 8 | Primary analysis at $d=4$; secondary at $d=8$ |

### 2.3 Dependent Variables (Measured)

**Primary metrics:**

| Metric | Definition | Use |
|--------|-----------|-----|
| Test accuracy | $\frac{1}{n_{\text{test}}} \sum_i \mathbb{1}[\hat{y}_i = y_i]$ | Primary performance measure |
| Macro F1 | $\frac{1}{C}\sum_{c=1}^{C} F_{1,c}$ | Imbalanced datasets |
| AUC-ROC | Area under ROC curve | Binary classification |

**Secondary metrics:**

| Metric | Definition | Use |
|--------|-----------|-----|
| Convergence iterations | Iterations to 90% of final accuracy | Training efficiency |
| Final training loss | $L(\boldsymbol{\theta}^*)$ | Optimization quality |
| Generalization gap | $\text{Acc}_{\text{train}} - \text{Acc}_{\text{test}}$ | Overfitting detection |
| Circuit depth | $\text{QuantumCircuit.depth()}$ (transpiled) | Resource cost |
| Gate count (total) | $\text{QuantumCircuit.size()}$ (transpiled) | Resource cost |
| CX count | Two-qubit gate count (transpiled) | Noise proxy |
| Expressibility | $D_{KL}(\hat{P}_U \| P_{\text{Haar}})$ | Encoding characterization |
| Entanglement (MW) | $Q = \frac{2}{m}\sum_k (1 - \text{tr}(\rho_k^2))$ | Encoding characterization |
| Kernel-target alignment | $A(K, Y)$ | Encoding-data compatibility |

---

## 3. Exclusion Rules

Certain encoding-dataset-preprocessing combinations are fundamentally incompatible and should be excluded from the experiment matrix.

### 3.1 Encoding Exclusions

| Exclusion | Reason | Affected configurations |
|-----------|--------|------------------------|
| Basis encoding + continuous data (without binarization) | Basis requires binary input | Basis with None, MinMax, Z-score preprocessing |
| Basis encoding + multi-class datasets | Binary encoding incompatible with >2 classes unless one-vs-rest | Basis + Iris (3 classes), Wine (3), HAR (6) |
| Amplitude encoding + raw data (None preprocessing) | Requires normalized input | Amplitude with None |
| Any encoding + $d > 20$ without PCA | Exceeds NISQ qubit budget | Angle/IQP/Re-upload with Cancer(30), MNIST(784), HAR(561) without PCA |

### 3.2 NISQ Feasibility Exclusions

| Exclusion | Reason | Affected configurations |
|-----------|--------|------------------------|
| IQP (full) with $d > 8$ | CX count exceeds fidelity budget ($>56$ CX gates) | IQP + all datasets at $d > 8$ |
| Re-uploading ($L=2$) with $d > 8$ | Depth exceeds practical limits | Re-upload + all datasets at $d > 8$ |

### 3.3 Valid Configuration Count

After exclusions, the estimated valid configuration count:

| Encoding | Valid preprocessings | Valid datasets | Subtotal |
|----------|---------------------|----------------|----------|
| Basis | 1 (binarization only) | 5 (binary-compatible) | ~5 |
| Amplitude | 3 (need normalization) | 8 | ~24 |
| Angle ($R_y$) | 4 | 8 | 32 |
| IQP (full) | 4 | 8 (at $d \leq 8$) | 32 |
| Re-uploading ($L=2$) | 4 | 8 (at $d \leq 8$) | 32 |

**Total valid configurations**: ~125 (out of 160 nominal)

This exceeds the Phase 1 success criterion of $\geq 120$ valid experiments.

---

## 4. Experimental Protocol

### 4.1 Per-Configuration Protocol

For each valid (encoding, preprocessing, dataset) triple:

```
INPUT: Dataset (X, y), Encoding E, Preprocessing T, Seed s

1. PREPROCESS
   X_proc = T.fit_transform(X)  [fit on train set only per fold]

2. BUILD CIRCUIT
   encoding_circuit = E.encode(X_proc[0])  [template]
   ansatz = RealAmplitudes(n_qubits, reps=2)
   full_circuit = encoding_circuit + ansatz

3. RECORD CIRCUIT METRICS (once per configuration)
   depth = transpile(full_circuit).depth()
   gate_count = transpile(full_circuit).size()
   cx_count = count CX gates
   expressibility = compute_expr(E, n_qubits, n_pairs=5000)
   entanglement = compute_MW(E, n_qubits, n_samples=1000)

4. 5-FOLD CROSS-VALIDATION (seeds 42-46)
   FOR fold k = 1..5:
     X_train, X_test, y_train, y_test = stratified_split(X_proc, y, fold=k)

     a. Fit preprocessing on X_train
     b. Transform X_train, X_test
     c. Initialize VQC(encoding=E, ansatz=RealAmplitudes(reps=2))
     d. Train: COBYLA(maxiter=200) on X_train, y_train
     e. Record: loss_history, train_accuracy
     f. Evaluate: test_accuracy, F1, AUC on X_test, y_test
     g. Record generalization gap

5. COMPUTE KERNEL METRICS (once per configuration)
   K = compute_kernel(X_proc, E)
   alignment = kernel_target_alignment(K, y)
   eigenvalues = eigvalsh(K)

6. AGGREGATE RESULTS
   accuracy_mean, accuracy_std = mean/std over 5 folds
   convergence = mean iterations to 90% accuracy
   Store all metrics in results DataFrame
```

### 4.2 Seed Management

| Component | Seed(s) | Purpose |
|-----------|---------|---------|
| Train/test split | Fold index (deterministic from data) | Reproducible splits |
| Parameter initialization | 42 + fold_index | Different initializations per fold |
| Optimizer (COBYLA) | N/A (deterministic given initialization) | COBYLA is deterministic |
| Shot sampling | 42 | Reproducible shot noise |
| PCA | N/A (deterministic) | Deterministic transformation |

### 4.3 Execution Order

To manage computational resources, experiments are organized in tiers:

**Tier 1: Core experiments (Priority)** -- Run first
- All encodings (5) x All preprocessings (4) x Iris + Wine + Heart Disease (small datasets)
- Estimated: 60 configurations x 5 folds = 300 circuit training runs

**Tier 2: Extended datasets** -- Run second
- All encodings x All preprocessings x Cancer + MNIST + Credit Fraud + HAR + NSL-KDD
- Estimated: 65 configurations x 5 folds = 325 circuit training runs

**Tier 3: Noise study** -- Run last (on top performers from Tier 1+2)
- Top-3 encoding-preprocessing combinations per dataset
- Noise models: IBM Heron R2, Rigetti Ankaa-3
- Estimated: ~50 configurations x 5 folds x 2 noise models = 500 runs

---

## 5. Statistical Analysis Plan

### 5.1 Primary Analysis: Encoding Effect

**Test**: Friedman test (non-parametric repeated measures ANOVA)
- Groups: 5 encoding methods (or valid subset per dataset)
- Blocks: datasets x preprocessing combinations (repeated measures)
- Response: test accuracy (or F1 for imbalanced datasets)
- Significance: $\alpha = 0.05$ with Bonferroni correction for $C = 8$ datasets: $\alpha_{\text{adj}} = 0.00625$

**Post-hoc**: Nemenyi test for pairwise encoding comparisons
- Reports critical difference (CD) diagrams
- Identifies statistically distinct groups of encodings

**Effect size**: Kendall's W concordance coefficient
$$W = \frac{12 \sum_{j=1}^k (R_j - \bar{R})^2}{b^2 k(k^2-1)}$$
where $R_j$ is the sum of ranks for encoding $j$, $b$ is the number of blocks, and $k$ is the number of encodings.

### 5.2 Secondary Analysis: Preprocessing Effect

**Test**: Friedman test across 4 preprocessing methods
- Fixed: encoding method
- Blocks: datasets
- Response: test accuracy

**Post-hoc**: Nemenyi test for pairwise preprocessing comparisons

### 5.3 Interaction Analysis: Encoding x Preprocessing

**Test**: Permutation-based interaction test (Aligned Rank Transform ANOVA)
- Factor 1: Encoding (5 levels)
- Factor 2: Preprocessing (4 levels)
- Blocks: datasets
- Tests: main effects and interaction

**Visualization**: Interaction plots showing mean accuracy by encoding for each preprocessing level

### 5.4 Meta-Model: Data -> Encoding Prediction

**Objective**: Predict the best encoding method from dataset statistical profile.

**Input features** (per dataset, from Phase 1 characterization):
- $d_{\text{eff}}$ (effective dimensionality)
- $\bar{r}$ (average absolute correlation)
- $\bar{\gamma}$ (average skewness magnitude)
- $\bar{\kappa}$ (average excess kurtosis magnitude)
- $F_{\text{sep}}$ (Fisher separability)
- $S$ (sparsity index)
- $n$ (sample size)
- $C$ (number of classes)
- Normality score (fraction of features passing Shapiro-Wilk)

**Target**: Best encoding method (categorical: angle, IQP, re-uploading, amplitude, basis)

**Method**: Random Forest classifier (leave-one-dataset-out cross-validation)
- With only 8 datasets, use LOO cross-validation
- Report: accuracy, confusion matrix, feature importances
- Success criterion: $R^2 > 0.6$ or accuracy $> 60\%$ (above random = 20%)

**Backup method**: If 8 datasets are insufficient for reliable prediction, use correlation analysis between data characteristics and encoding performance rankings.

### 5.5 Classical Baseline Comparison

For each valid configuration, run the classical baseline:
- SVM with RBF kernel (same preprocessing, same train/test splits)
- Random Forest (same splits)
- Logistic Regression (same splits)

**Comparison metric**: $\Delta_{\text{QC}} = \text{Acc}_{\text{quantum}} - \text{Acc}_{\text{classical,best}}$

Report the distribution of $\Delta_{\text{QC}}$ across configurations, with paired statistical tests.

---

## 6. Circuit Design for Each Encoding

### 6.1 Full VQC Circuit Architecture

The complete circuit for each experiment is:

$$|out\rangle = A(\boldsymbol{\theta}) \cdot E(\mathbf{x}) |0\rangle^{\otimes m}$$

where $E(\mathbf{x})$ is the encoding layer and $A(\boldsymbol{\theta}) = \text{RealAmplitudes}(m, \text{reps}=2)$.

**RealAmplitudes(reps=2) structure** (for $m$ qubits):
```
      ┌──────────┐     ┌──────────┐     ┌──────────┐
q0 ───┤ Ry(θ_0)  ├──●──┤ Ry(θ_m)  ├──●──┤ Ry(θ_2m) ├──
      └──────────┘  │  └──────────┘  │  └──────────┘
      ┌──────────┐  │  ┌──────────┐  │  ┌──────────┐
q1 ───┤ Ry(θ_1)  ├──⊕──┤ Ry(θ_m+1)├──⊕──┤ Ry(θ2m+1)├──
      └──────────┘     └──────────┘     └──────────┘
      ...
```

**Parameter count for RealAmplitudes(m, reps=2)**: $3m$ parameters ($m$ per rotation layer, 3 layers for reps=2).

### 6.2 Encoding-Specific Circuit Definitions

**Experiment Circuit for Angle Encoding ($d = 4$):**
```
Encoding:                    Ansatz (RealAmplitudes reps=2):
|0> -- Ry(x_1) ------------ Ry(θ_0) --●-- Ry(θ_4) --●-- Ry(θ_8) ---[M]
|0> -- Ry(x_2) ------------ Ry(θ_1) --⊕-- Ry(θ_5) --⊕-- Ry(θ_9) ---[M]
|0> -- Ry(x_3) ------------ Ry(θ_2) --●-- Ry(θ_6) --●-- Ry(θ_10)---[M]
|0> -- Ry(x_4) ------------ Ry(θ_3) --⊕-- Ry(θ_7) --⊕-- Ry(θ_11)---[M]

Total: 4 encoding gates + 12 ansatz params + 4 CX = 20 gates
Depth: ~8
```

**Experiment Circuit for IQP Encoding ($d = 4$):**
```
|0> -- H -- Rz(x_1) -- RZZ(x1*x2) -- RZZ(x1*x3) -- RZZ(x1*x4) -- H -- [Ansatz] -- [M]
|0> -- H -- Rz(x_2) -------|----------- RZZ(x2*x3) -- RZZ(x2*x4) -- H -- [Ansatz] -- [M]
|0> -- H -- Rz(x_3) -----------------------|----------- RZZ(x3*x4) -- H -- [Ansatz] -- [M]
|0> -- H -- Rz(x_4) -------------------------------------------|---- H -- [Ansatz] -- [M]

Total: 8H + 4Rz + 6RZZ (=12CX) + ansatz(12 params + 4CX) = ~40 gates
Depth: ~18
```

**Experiment Circuit for Re-uploading ($L=2$, $d = 4$):**
```
Layer 1:                              Layer 2:
|0> -- Ry(x_1) -- Ry(θ_0) --●--  -- Ry(x_1) -- Ry(θ_4) --●-- [Ansatz] -- [M]
|0> -- Ry(x_2) -- Ry(θ_1) --⊕--  -- Ry(x_2) -- Ry(θ_5) --⊕-- [Ansatz] -- [M]
|0> -- Ry(x_3) -- Ry(θ_2) --●--  -- Ry(x_3) -- Ry(θ_6) --●-- [Ansatz] -- [M]
|0> -- Ry(x_4) -- Ry(θ_3) --⊕--  -- Ry(x_4) -- Ry(θ_7) --⊕-- [Ansatz] -- [M]

Total: 16 Ry encoding + 8 Ry trainable + 4 CX (re-upload) + ansatz = ~40+ gates
Depth: ~16
```

**Experiment Circuit for Amplitude Encoding ($d = 4$, $m = 2$ qubits):**
```
|0> -- [Mottonen state prep for 4 amplitudes] -- [Ansatz(2 qubits)] -- [M]
|0> --                                         --                    -- [M]

Total: ~8 gates (encoding) + ansatz(6 params + 2 CX) = ~16 gates
Depth: ~10
```

### 6.3 Measurement Strategy

**Binary classification** (Cancer, MNIST, Credit Fraud, Heart Disease): Measure qubit 0 in computational basis. $P(y=1) = P(\text{qubit 0} = |1\rangle)$.

**Multi-class classification** (Iris: 3 classes, Wine: 3 classes, HAR: 6 classes, NSL-KDD: 5 classes):
- **Parity method**: Partition measurement bitstrings into $C$ classes based on parity or bit assignment
- For $C = 3$: classes mapped to $\{00, 01, 10\}$ (exclude 11) or use probabilities of first 2 qubits
- For $C = 6$: need $\lceil\log_2 6\rceil = 3$ measurement qubits

**Shot allocation**: 1024 shots per evaluation. For 200 optimizer iterations with $n_{\text{train}} = 120$ (Iris, 80% of 150), each evaluation requires $\sim 120 \times 1024$ shots. To reduce cost, evaluate loss on a mini-batch of 20 samples per iteration.

---

## 7. Computational Resource Estimation

### 7.1 Per-Configuration Estimate

| Component | Cost (simulator) |
|-----------|-----------------|
| Circuit evaluations per optimizer iteration | ~20 (mini-batch) |
| Optimizer iterations | 200 |
| Shots per evaluation | 1024 |
| Total shots per fold | ~20 x 200 x 1024 = 4.1M |
| Folds | 5 |
| Total shots per configuration | ~20.5M |

### 7.2 Total Experiment Budget

| Tier | Configurations | Total shots | Estimated time (AerSimulator) |
|------|---------------|-------------|-------------------------------|
| Tier 1 (small datasets) | ~60 | ~1.2B | ~2-4 hours |
| Tier 2 (large datasets) | ~65 | ~1.3B | ~3-6 hours |
| Tier 3 (noise study) | ~50 | ~1.0B | ~4-8 hours |
| **Total** | **~175** | **~3.5B** | **~10-18 hours** |

**Note**: Statevector simulation is faster than shot-based simulation for small qubit counts. For $m \leq 8$, use statevector simulation with post-hoc shot sampling.

---

## 8. Reproducibility Requirements

### 8.1 Configuration Serialization

Every experiment must be fully described by a configuration dictionary:

```python
config = {
    "dataset": "iris",
    "encoding": "angle_ry",
    "preprocessing": "minmax_pi",
    "n_qubits": 4,
    "ansatz": "RealAmplitudes",
    "ansatz_reps": 2,
    "optimizer": "COBYLA",
    "maxiter": 200,
    "shots": 1024,
    "n_folds": 5,
    "seeds": [42, 43, 44, 45, 46],
    "pca_components": null,
    "subsample_size": null
}
```

### 8.2 Result Storage Format

Results stored as a CSV file (`results/encoding_comparison.csv`) with columns:

```
dataset, encoding, preprocessing, fold, seed,
train_accuracy, test_accuracy, f1_macro, auc_roc,
convergence_iters, final_loss, generalization_gap,
circuit_depth, gate_count, cx_count,
expressibility, entanglement_mw, kernel_alignment,
wall_time_seconds
```

### 8.3 Code Versioning

- Pin all package versions (use `uv.lock`)
- Record git commit hash in results metadata
- Store preprocessing pipeline objects (pickle) for exact reproduction

---

## 9. Success Criteria (from Phase 1)

| Criterion | Threshold | How measured |
|-----------|-----------|-------------|
| Encoding differences | Friedman $p < 0.05$ | At least 3 datasets show significant encoding effect |
| Practical significance | $\geq 5\%$ accuracy gap | Between best and worst encoding per dataset |
| Data-encoding pattern | Correlation $> 0.5$ | Between data profile metrics and encoding performance |
| Coverage | $\geq 120$ valid experiments | Count of successfully completed configurations |
| Classical comparison | Computed for all | $\Delta_{\text{QC}}$ for every configuration |
| Noise analysis | $\geq 2$ noise models | IBM Heron R2 + Rigetti Ankaa-3 |

---

## 10. Validation of Experimental Design

### 10.1 Controls Adequacy

**Why RealAmplitudes(reps=2)?** This ansatz provides:
- 2 entangling layers (sufficient for moderate expressibility)
- $3m$ trainable parameters (manageable for COBYLA)
- Linear entangling topology (no SWAP overhead on any hardware)
- Symmetric treatment across qubits (no architectural bias toward specific features)

**Alternative considered**: EfficientSU2 has 3 rotation axes per qubit per layer ($6m$ parameters for reps=2), which may over-parameterize for small datasets. RealAmplitudes is more conservative and widely used in QML benchmarks.

### 10.2 Confound Analysis

| Potential confound | Mitigation |
|-------------------|-----------|
| Optimizer dependence | Fixed COBYLA(maxiter=200) for all |
| Initialization sensitivity | Average over 5 seeds |
| Data split variance | 5-fold stratified CV |
| Shot noise | Fixed 1024 shots; average over runs |
| Feature count mismatch | PCA to common $k$ (4 or 8) when needed |
| Qubit count varies by encoding | Report accuracy per qubit as secondary metric |
| Encoding adds trainable params (re-uploading) | Report total parameter count; analyze accuracy vs params |

### 10.3 Assumptions

1. **Noiseless simulation approximates ideal performance**: Valid for comparing encoding effectiveness. Noise study (Tier 3) provides practical corrections.
2. **COBYLA(maxiter=200) is sufficient for convergence**: May not converge for all configurations. Record convergence status; exclude non-converged runs from primary analysis.
3. **5-fold CV provides reliable estimates**: Standard for moderate-sized datasets. For Iris ($n=150$), each fold has only 30 test samples -- report confidence intervals.
4. **PCA to $k=8$ is acceptable information loss**: Validated by variance retention analysis (Section 4.3 of phase2_preprocessing_theory.md). Datasets with $R_8 < 0.7$ may show degraded performance attributable to PCA, not encoding.

---

## 11. Summary and Handoff

### 11.1 For the Test Engineer

Design tests for:
1. **Configuration validation**: Verify exclusion rules are correctly applied
2. **Circuit construction**: Verify correct circuit structure for each encoding+ansatz combination
3. **Preprocessing pipeline**: Verify correct transformation for each preprocessing level
4. **Metric computation**: Verify accuracy, F1, AUC, kernel alignment calculations
5. **Reproducibility**: Verify same seed produces same results
6. **Edge cases**: Empty batches, single-class folds (shouldn't occur with stratification), non-convergent runs

### 11.2 For the Python Architect

Implement:
1. `ExperimentConfig` dataclass matching Section 8.1
2. `ExperimentResult` dataclass matching Section 8.2
3. `EncodingExperimentRunner.run_all()` following the protocol in Section 4
4. Exclusion rule validation before running
5. Progress logging and checkpoint saving
6. Results aggregation and CSV export

### 11.3 For the Data Scientist

Execute:
1. Tier 1 experiments first (small datasets, all encodings)
2. Preliminary analysis to verify experimental setup
3. Tier 2 experiments (extended datasets)
4. Full statistical analysis following Section 5
5. Tier 3 noise study on top performers
6. Generate all visualizations from Phase 1 specification

---

## References

1. Friedman, M. (1937). The use of ranks to avoid the assumption of normality. *JASA*, 32(200), 675-701.
2. Nemenyi, P. (1963). *Distribution-free multiple comparisons*. PhD thesis, Princeton University.
3. Cerezo, M. et al. (2022). Challenges and opportunities in QML. *Nature Computational Science*, 2, 567-576.
4. Sim, S., Johnson, P.D. & Aspuru-Guzik, A. (2019). Expressibility and entangling capability. *Adv. Quantum Technol.*, 2(12), 1900070.
