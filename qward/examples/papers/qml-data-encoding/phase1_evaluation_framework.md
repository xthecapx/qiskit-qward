# Phase 1: Evaluation Framework and Metrics

## 1. Overview

This document defines the evaluation framework for systematically comparing quantum data encoding methods. The framework is designed to isolate encoding effects by controlling all other experimental variables (model architecture, optimizer, shots, etc.) and to ensure statistically rigorous comparisons.

---

## 2. Experimental Design

### 2.1 Independent Variables (Systematically Varied)

| Variable | Levels | Description |
|----------|--------|-------------|
| Encoding method | 5 | Basis, Amplitude, Angle, IQP, Data Re-uploading |
| Classical preprocessing | 4 | None, MinMax (to [0, 2pi]), Z-score, PCA + MinMax |
| Dataset | 8 | 4 benchmark + 4 real-world (see dataset selection) |

**Total experiment configurations**: 5 encodings x 4 preprocessings x 8 datasets = 160 configurations (not all combinations are valid; see Section 2.4 for exclusion rules).

### 2.2 Controlled Variables (Fixed Across All Experiments)

| Variable | Fixed Value | Justification |
|----------|-------------|---------------|
| QML Model | Variational Quantum Classifier (VQC) | Most common QML classifier; well-studied |
| Ansatz | RealAmplitudes, reps=2 | Standard shallow ansatz; sufficient expressibility |
| Optimizer | COBYLA, maxiter=200 | Gradient-free; robust for noisy landscapes |
| Measurement shots | 1024 | Standard for NISQ simulation |
| Train/test split | 80/20, stratified | Standard ML practice |
| Cross-validation | 5-fold stratified | Robust performance estimation |
| Random seed | 42 (with 5 repeats at seeds 42-46) | Reproducibility |
| Simulator | AerSimulator (statevector for kernels) | Noiseless baseline |
| PCA components (when used) | min(n_features, 8) | Fits on 8 qubits; reasonable qubit count |

### 2.3 Dependent Variables (Measured)

Organized into three axes as defined in the project plan.

### 2.4 Exclusion Rules

Certain encoding-dataset combinations are excluded due to fundamental incompatibility:

| Exclusion | Reason |
|-----------|--------|
| Basis encoding + continuous data | Basis encoding requires binary input |
| Amplitude encoding + d > 2^16 features | State preparation circuit too deep |
| Any encoding where required qubits > 20 | Beyond practical NISQ simulation |
| Basis encoding + non-binary datasets | Unless binarization is applied as preprocessing |

---

## 3. Evaluation Metrics

### Axis 1: Data Characteristics (Computed Once Per Dataset)

These metrics characterize each dataset's statistical properties to enable data-encoding compatibility analysis.

#### 3.1.1 Distribution Shape Metrics

**Skewness** (per feature):
$$\gamma_1 = \frac{m_3}{m_2^{3/2}} = \frac{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^3}{\left(\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2\right)^{3/2}}$$

**Kurtosis** (per feature, excess):
$$\gamma_2 = \frac{m_4}{m_2^2} - 3 = \frac{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^4}{\left(\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2\right)^2} - 3$$

**Normality test**: Shapiro-Wilk statistic $W$ (for $n < 5000$) or D'Agostino-Pearson $K^2$ (for $n \geq 5000$).

**Aggregated distribution score**:
$$D_{score} = \frac{1}{d}\sum_{j=1}^{d}\left(|\gamma_{1,j}| + |\gamma_{2,j}|\right)$$

where $d$ is the number of features. Higher values indicate more non-Gaussian distributions.

#### 3.1.2 Correlation Structure

**Pearson correlation matrix**: $R_{ij} = \text{corr}(X_i, X_j)$

**Average absolute correlation** (off-diagonal):
$$\bar{r} = \frac{2}{d(d-1)}\sum_{i<j}|R_{ij}|$$

**Mutual information matrix**: $I_{ij} = I(X_i; X_j)$ computed via KDE estimation.

**Average mutual information**:
$$\bar{I} = \frac{2}{d(d-1)}\sum_{i<j}I_{ij}$$

#### 3.1.3 Intrinsic Dimensionality

**PCA-based**: Number of components $k$ such that $\sum_{i=1}^{k}\lambda_i / \sum_{i=1}^{d}\lambda_i \geq 0.95$, where $\lambda_i$ are sorted eigenvalues.

**MLE estimator** (Levina & Bickel, 2004):
$$\hat{m}_k(x) = \left(\frac{1}{k-1}\sum_{j=1}^{k-1}\log\frac{T_k(x)}{T_j(x)}\right)^{-1}$$

where $T_j(x)$ is the distance from $x$ to its $j$-th nearest neighbor. The intrinsic dimensionality is:
$$\hat{d}_{MLE} = \frac{1}{n}\sum_{i=1}^{n}\hat{m}_k(x_i)$$

#### 3.1.4 Class Separability

**Fisher's discriminant ratio** (for each feature $j$ and class pair $(c_1, c_2)$):
$$F_j(c_1, c_2) = \frac{(\mu_{j,c_1} - \mu_{j,c_2})^2}{\sigma^2_{j,c_1} + \sigma^2_{j,c_2}}$$

**Overall separability** (maximum over features, average over class pairs):
$$\text{Sep} = \frac{2}{C(C-1)}\sum_{c_1 < c_2}\max_j F_j(c_1, c_2)$$

**Silhouette score**: Standard silhouette coefficient $s \in [-1, 1]$.

#### 3.1.5 Sparsity

**Sparsity index**: Fraction of values below threshold $\epsilon$:
$$S = \frac{|\{x_{ij} : |x_{ij}| < \epsilon\}|}{n \times d}$$

Default $\epsilon = 10^{-6}$ for continuous data.

---

### Axis 2: Encoding Properties (Computed Per Encoding Method)

These metrics characterize the quantum encoding circuit independent of the data.

#### 3.2.1 Qubit Count

$$Q(E, d) = \text{required\_qubits}(E, d)$$

where $E$ is the encoding method and $d$ is the feature dimension (after preprocessing).

| Encoding | Formula |
|----------|---------|
| Basis | $Q = d$ (binary features) |
| Amplitude | $Q = \lceil\log_2(d)\rceil$ |
| Angle | $Q = d$ |
| IQP | $Q = d$ |
| Re-uploading | $Q = d$ (or fewer with multiplexed encoding) |

#### 3.2.2 Circuit Depth

Measured as the longest path in the circuit DAG. Computed using `QuantumCircuit.depth()` after transpilation to the target basis gate set {CX, RZ, SX, X}.

#### 3.2.3 Gate Count

- **Total gates**: `circuit.size()`
- **Two-qubit gates (CX count)**: `circuit.count_ops().get('cx', 0)` (after transpilation)
- **Gate density**: total gates / (depth x qubits)

#### 3.2.4 Expressibility

Using the Sim et al. (2019) definition:

$$\text{Expr}(U) = D_{KL}\left(\hat{P}_U(F) \| P_{\text{Haar}}(F)\right)$$

where:
- $F = |\langle\psi|\phi\rangle|^2$ is the fidelity between pairs of random encoded states
- $\hat{P}_U(F)$ is estimated from $N_{pairs}$ random data point pairs
- $P_{\text{Haar}}(F) = (2^n - 1)(1 - F)^{2^n - 2}$ is the Haar-random fidelity distribution

**Computation procedure**:
1. Sample $N_{pairs} = 5000$ pairs of random feature vectors $x, x'$
2. Compute $|\phi(x)\rangle = U(x)|0\rangle$ and $|\phi(x')\rangle = U(x')|0\rangle$
3. Compute fidelities $F = |\langle\phi(x)|\phi(x')\rangle|^2$
4. Estimate $\hat{P}_U(F)$ via histogram with 75 bins
5. Compute KL divergence against $P_{\text{Haar}}$

Lower expressibility value = closer to Haar-random = more expressive encoding.

#### 3.2.5 Entanglement Capability

For multi-qubit encodings, measured using the Meyer-Wallach entanglement measure:

$$Q(|\psi\rangle) = \frac{2}{n}\sum_{k=1}^{n}\left(1 - \text{tr}(\rho_k^2)\right)$$

where $\rho_k$ is the reduced density matrix of qubit $k$. Averaged over random encoded states.

#### 3.2.6 Trainability Proxy

Gradient variance of the encoding+ansatz circuit at random parameter initialization:

$$\text{Var}[\partial_i C] = \text{Var}_{\theta}\left[\frac{\partial}{\partial\theta_i}\langle\psi(\theta)|H|\psi(\theta)\rangle\right]$$

Estimated numerically via parameter shift rule over 100 random initializations. Higher variance = easier to train.

---

### Axis 3: QML Performance (Measured Per Experiment)

#### 3.3.1 Classification Accuracy

Standard test set accuracy:
$$\text{Acc} = \frac{1}{n_{test}}\sum_{i=1}^{n_{test}}\mathbb{1}[\hat{y}_i = y_i]$$

Reported as mean +/- std over 5-fold cross-validation.

#### 3.3.2 F1 Score (for Imbalanced Datasets)

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Macro-averaged for multi-class; standard for binary.

#### 3.3.3 Area Under ROC Curve (AUC)

$$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(t))\, dt$$

Particularly important for imbalanced datasets (Credit Fraud).

#### 3.3.4 Training Convergence

**Convergence speed**: Number of optimizer iterations to reach 90% of final accuracy.

**Loss curve**: Full training loss trajectory $L(\theta_t)$ for $t = 1, \ldots, T$.

**Final loss**: $L(\theta^*)$ at optimization termination.

#### 3.3.5 Resource Efficiency

**Accuracy per qubit**:
$$\eta_Q = \frac{\text{Acc}}{Q}$$

**Accuracy per circuit depth**:
$$\eta_D = \frac{\text{Acc}}{D}$$

These metrics identify encodings that achieve good performance with minimal quantum resources.

#### 3.3.6 Generalization Gap

$$\Delta_{\text{gen}} = \text{Acc}_{\text{train}} - \text{Acc}_{\text{test}}$$

Large gaps indicate overfitting; relevant for expressive encodings like Re-uploading.

#### 3.3.7 Noise Robustness (Secondary Experiments)

Performance degradation under noise:
$$\Delta_{\text{noise}} = \text{Acc}_{\text{noiseless}} - \text{Acc}_{\text{noisy}}$$

Tested with IBM Heron R2 and Rigetti Ankaa-3 noise models from QWARD presets.

---

## 4. Statistical Analysis Plan

### 4.1 Primary Comparisons

**Research Question 1** (H1/H3): Does encoding method significantly affect accuracy?

- **Test**: Friedman test (non-parametric repeated measures ANOVA) across encodings for each dataset.
- **Post-hoc**: Nemenyi test for pairwise encoding comparisons.
- **Effect size**: Kendall's W for concordance.
- **Significance level**: $\alpha = 0.05$ with Bonferroni correction for multiple datasets.

**Research Question 2** (H2): Does classical preprocessing significantly affect accuracy?

- **Test**: Friedman test across preprocessing methods for each encoding-dataset combination.
- **Post-hoc**: Nemenyi test for pairwise comparisons.

**Research Question 3** (H1): Do data characteristics predict optimal encoding?

- **Analysis**: Multiple regression or random forest predicting best encoding from data profile features.
- **Metric**: $R^2$ or classification accuracy of the meta-model.
- **Validation**: Leave-one-dataset-out cross-validation.

### 4.2 Secondary Analyses

**Encoding-Preprocessing Interaction**:
- Two-way Friedman test or permutation-based interaction test.
- Visualize via interaction plots.

**Expressibility-Accuracy Correlation**:
- Spearman rank correlation between expressibility and accuracy.
- Scatter plot with encoding method as color.

**Noise Robustness Ranking**:
- Friedman test across encodings under each noise model.
- Paired comparison of noise degradation.

### 4.3 Effect Size Standards

| Metric | Small | Medium | Large |
|--------|-------|--------|-------|
| Cohen's d | 0.2 | 0.5 | 0.8 |
| Kendall's W | 0.1 | 0.3 | 0.5 |
| Accuracy difference | 2% | 5% | 10% |

---

## 5. Classical Baselines

For every encoding experiment, we establish classical baselines using the same preprocessing:

| Classical Model | Purpose |
|-----------------|---------|
| SVM (RBF kernel) | Primary baseline (kernel method comparison) |
| Random Forest | Ensemble baseline |
| Logistic Regression | Linear baseline |
| Classical SVM (quantum kernel approximation) | Kernel matching baseline |

**Baseline requirement**: Classical baselines must be run on the same train/test splits with the same preprocessing pipeline. This enables direct comparison of quantum vs. classical performance.

---

## 6. Reproducibility Standards

### 6.1 Code Requirements

- All experiments executable via a single `EncodingExperimentRunner` configuration
- Random seeds fixed for all stochastic components
- Results saved to CSV with full configuration metadata
- Preprocessing pipelines serialized for exact reproduction

### 6.2 Reporting Requirements

- All results reported as mean +/- std over 5 folds
- Confidence intervals reported for key comparisons
- Effect sizes alongside p-values
- Full result tables in appendix; summary figures in main text

### 6.3 QWARD Integration

All encoding circuits analyzed with QWARD Scanner for:
- Circuit depth (pre and post transpilation)
- Gate count by type
- Complexity score
- Visualization of encoding circuits

---

## 7. Success Criteria

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| Statistical significance | p < 0.05 | Friedman test for encoding differences |
| Practical significance | >= 5% accuracy difference | Between best and worst encoding per dataset |
| Data-encoding pattern | $R^2 > 0.6$ | Meta-model predicting encoding from data profile |
| Coverage | >= 120 valid experiments | Out of 160 total configurations |
| Classical comparison | For all configurations | Quantum vs. classical accuracy difference |
| Noise analysis | >= 3 noise models | Noiseless + 2 QWARD presets |

---

## 8. Summary

This evaluation framework provides:
1. **Controlled experiments**: Fixed model/optimizer/shots to isolate encoding effects
2. **Comprehensive metrics**: 3 axes (data, encoding, performance) with 15+ metrics
3. **Rigorous statistics**: Friedman tests with Bonferroni correction and effect sizes
4. **Classical baselines**: Direct quantum vs. classical comparison
5. **Reproducibility**: Fixed seeds, serialized configurations, QWARD integration
6. **Clear success criteria**: Measurable thresholds for hypothesis validation
