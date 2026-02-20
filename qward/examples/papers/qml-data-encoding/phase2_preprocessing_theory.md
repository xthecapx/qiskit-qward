# Phase 2: Classical Preprocessing Theory for Quantum Encoding

## 1. Motivation

Classical preprocessing transforms data before quantum encoding. The choice of preprocessing directly affects the quantum state space explored, the kernel properties, and ultimately the classification performance. This document formalizes how classical transformations interact with quantum encodings and derives conditions under which preprocessing preserves or destroys information relevant to quantum advantage.

---

## 2. Preprocessing as a Composition of Maps

The full quantum feature map with preprocessing is the composition:

$$\phi_P: \mathbb{R}^d \xrightarrow{T} \mathbb{R}^{d'} \xrightarrow{U} \mathcal{H}$$

where $T: \mathbb{R}^d \to \mathbb{R}^{d'}$ is a classical transformation and $U: \mathbb{R}^{d'} \to \mathcal{H}$ is the quantum encoding.

The composed feature map is:
$$|\phi_P(\mathbf{x})\rangle = U(T(\mathbf{x}))|0\rangle^{\otimes m}$$

The composed kernel is:
$$K_P(\mathbf{x}, \mathbf{x}') = K(T(\mathbf{x}), T(\mathbf{x}'))$$

**Key insight**: Preprocessing $T$ modifies the kernel $K$ by reparameterizing the data space. Different preprocessing choices can dramatically change the effective kernel, even with the same quantum encoding circuit.

---

## 3. Normalization Schemes

### 3.1 MinMax Normalization to $[0, a]$

**Definition**: For each feature $j$:
$$T_{\text{MM}}(\mathbf{x})_j = \frac{x_j - x_j^{\min}}{x_j^{\max} - x_j^{\min}} \cdot a$$

where $a$ is the target range (typically $a = \pi$ or $a = 2\pi$).

**Effect on angle encoding kernel**: The kernel becomes:
$$K_{\text{MM}}(\mathbf{x}, \mathbf{x}') = \prod_{i=1}^d \cos^2\left(\frac{a(x_i - x_i')}{2(x_i^{\max} - x_i^{\min})}\right)$$

**Bandwidth analysis**: The effective kernel bandwidth for feature $j$ is:
$$\sigma_j^{\text{eff}} = \frac{2(x_j^{\max} - x_j^{\min})}{a}$$

For $a = \pi$: $\sigma_j^{\text{eff}} = \frac{2}{\pi}(x_j^{\max} - x_j^{\min})$. Wider features get wider bandwidth.
For $a = 2\pi$: $\sigma_j^{\text{eff}} = \frac{1}{\pi}(x_j^{\max} - x_j^{\min})$. Narrower bandwidth (kernel decays faster).

**Problem with MinMax**: It is sensitive to outliers. A single extreme value stretches the normalization range, compressing the majority of data into a small fraction of $[0, a]$.

**Proposition (MinMax bandwidth distortion)**. If feature $j$ has a heavy-tailed distribution with outlier fraction $f$ and outlier distance ratio $r = x_j^{\max} / x_j^{P_{99}}$ (ratio of max to 99th percentile), then the effective bandwidth for the non-outlier data is compressed by factor $1/r$:
$$\sigma_{j,\text{non-outlier}}^{\text{eff}} = \sigma_j^{\text{eff}} / r$$

This compression squeezes most data points into a small arc of the Bloch sphere, reducing the encoding's ability to distinguish them.

**Recommendation**: For heavy-tailed data (Credit Fraud `Amount` feature: skewness ~16.1), use robust scaling (e.g., IQR-based) or clip outliers before MinMax normalization.

### 3.2 Z-Score Standardization

**Definition**: For each feature $j$:
$$T_z(\mathbf{x})_j = \frac{x_j - \mu_j}{\sigma_j}$$

where $\mu_j$ and $\sigma_j$ are the feature mean and standard deviation.

**Properties**:
- Output range: theoretically $(-\infty, +\infty)$, but in practice $\approx [-3, 3]$ for Gaussian data (99.7%)
- Preserves relative distances (affine transformation)
- Centers data at origin; unit variance per feature
- Does NOT map to $[0, 2\pi]$ -- requires a subsequent mapping step for angle encoding

**Combined Z-score + mapping for angle encoding**: $T(\mathbf{x})_j = \pi \cdot \sigma(z_j)$ where $\sigma(z) = 1/(1+e^{-z})$ is the sigmoid function and $z_j = (x_j - \mu_j)/\sigma_j$. This maps the standardized data to $(0, \pi)$ with most mass in the middle of the range.

**Effect on angle encoding kernel**: With sigmoid mapping:
$$K_z(\mathbf{x}, \mathbf{x}') = \prod_i \cos^2\left(\frac{\pi}{2}(\sigma(z_i) - \sigma(z_i'))\right)$$

The sigmoid acts as a **soft clipper** that compresses extreme values while preserving the middle of the distribution. This provides natural outlier robustness.

### 3.3 L2 Normalization (for Amplitude Encoding)

**Definition**:
$$T_{L2}(\mathbf{x}) = \frac{\mathbf{x}}{\|\mathbf{x}\|_2}$$

**Properties**:
- Maps to unit sphere $S^{d-1}$
- Destroys magnitude information
- Preserves angular information (cosine similarity)

**Information loss analysis**: Define the information content of a dataset as:
$$I(\mathbf{X}) = I_{\text{direction}} + I_{\text{magnitude}}$$

where $I_{\text{direction}} = H(\mathbf{x}/\|\mathbf{x}\|)$ and $I_{\text{magnitude}} = H(\|\mathbf{x}\|)$.

**Theorem (L2 normalization information loss)**. L2 normalization is lossless for classification if and only if:
$$p(y | \mathbf{x}) = p(y | \mathbf{x}/\|\mathbf{x}\|) \quad \forall \mathbf{x}, y$$

i.e., the class label is independent of the feature vector magnitude.

**When L2 normalization destroys information**:
- If $\|\mathbf{x}\|$ differs systematically across classes (e.g., fraud transactions have different magnitude than legitimate ones)
- If class boundaries depend on the absolute scale of features

**When L2 normalization is safe**:
- If data points of the same class share similar directions but may vary in magnitude
- If PCA has already been applied (PCA scores are centered, reducing magnitude-class correlation)

### 3.4 Normalization Comparison for Our Study

| Preprocessing | Output range | Kernel bandwidth | Outlier sensitivity | Information loss |
|---------------|-------------|-----------------|---------------------|-----------------|
| MinMax to $[0, \pi]$ | $[0, \pi]$ | $\propto$ feature range | High | Low (bijective) |
| MinMax to $[0, 2\pi]$ | $[0, 2\pi]$ | $\propto$ feature range / 2 | High | Low (bijective) |
| Z-score + sigmoid | $(0, \pi)$ | Adaptive | Low | Some (sigmoid compression) |
| L2 normalization | Unit sphere | N/A (amplitude encoding) | Low | Magnitude information |
| None (raw data) | Arbitrary | Feature-dependent | N/A | None |

---

## 4. Dimensionality Reduction

### 4.1 PCA (Principal Component Analysis)

**Definition**: The PCA transformation projects data onto the $k$ leading eigenvectors of the covariance matrix:

$$T_{\text{PCA}}(\mathbf{x}) = W_k^T (\mathbf{x} - \boldsymbol{\mu})$$

where $\boldsymbol{\mu}$ is the data mean and $W_k = [\mathbf{w}_1, \ldots, \mathbf{w}_k]$ are the $k$ eigenvectors corresponding to the largest eigenvalues $\lambda_1 \geq \ldots \geq \lambda_k$ of $\Sigma = \frac{1}{n}\sum_i (\mathbf{x}_i - \boldsymbol{\mu})(\mathbf{x}_i - \boldsymbol{\mu})^T$.

**Variance retention**: The fraction of total variance retained is:
$$R_k = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i}$$

The standard threshold is $R_k \geq 0.95$ (95% variance).

**Lossless condition**: PCA reduction is lossless for classification if the class-discriminative information is contained in the top $k$ principal components. Formally:

$$p(y | W_k^T \mathbf{x}) = p(y | \mathbf{x}) \quad \forall \mathbf{x}, y$$

This is satisfied when:
1. The class centroids differ primarily along the top $k$ PCA directions
2. Class-specific variance is concentrated in the top $k$ components
3. The discarded components contain only noise

**When PCA is lossy for classification**: PCA optimizes for variance, not class discrimination. If class-discriminative features have small variance (e.g., a binary indicator feature with small spread), PCA may discard them. This is the classic "Fisher vs PCA" distinction.

### 4.2 PCA Interaction with Quantum Encoding

**PCA + Angle Encoding**: PCA decorrelates features ($\text{Cov}(T_{\text{PCA}}(\mathbf{x}))$ is diagonal). This means:
- The product kernel $K_{\text{angle}}$ is appropriate (uncorrelated features align with factorizable kernel)
- IQP's pairwise interactions $x_i x_j$ capture no correlation structure (since PCA removed correlations)

**Proposition (PCA neutralizes IQP advantage)**. If PCA is applied before encoding, the IQP kernel's quadratic terms $x_i x_j$ model interactions between uncorrelated features. These interactions are less likely to be class-relevant than interactions between the original correlated features.

**Implication**: PCA + angle encoding may be as effective as PCA + IQP encoding, because PCA removes the feature correlations that IQP is designed to capture. This is an experimentally testable prediction.

**PCA + Amplitude Encoding**: PCA reduces $d$ to $k$ components, making amplitude encoding more feasible (depth $O(2^{\lceil\log_2 k\rceil})$ instead of $O(2^{\lceil\log_2 d\rceil})$). However, the PCA scores have a specific distribution (multivariate Gaussian centered at zero), which after L2 normalization concentrates on a narrow region of the unit sphere. This can lead to kernel concentration.

### 4.3 PCA Component Selection for Quantum Experiments

Given NISQ constraints (Section 7.2 of phase1_nisq_constraints.md), the target qubit counts are 4 and 8. The PCA reduction targets are:

| Dataset | Original $d$ | PCA to 4 ($R_4$) | PCA to 8 ($R_8$) | Recommended $k$ |
|---------|-------------|-------------------|-------------------|-----------------|
| Iris | 4 | 97.8% | N/A | 4 (no PCA needed) |
| Wine | 13 | ~65% | ~85% | 8 |
| Cancer | 30 | ~79% | ~90% | 8 |
| MNIST (binary) | 784 | ~40% | ~60% | 8 (with caution) |
| Credit Fraud | 30 | ~50% | ~75% | 8 |
| NSL-KDD | 41 | ~40% | ~65% | 8 |
| HAR | 561 | ~55% | ~75% | 8 |
| Heart Disease | 13 | ~65% | ~85% | 8 |

**Note**: For MNIST and NSL-KDD, even 8 components retain only 60-65% of variance, indicating significant information loss. This is an inherent limitation of the quantum approach for high-dimensional data.

### 4.4 Alternative Dimensionality Reduction

**Feature Selection**: Select the $k$ features with highest Fisher discriminant ratio:
$$F_j = \max_{c_1 \neq c_2} \frac{(\mu_{j,c_1} - \mu_{j,c_2})^2}{\sigma_{j,c_1}^2 + \sigma_{j,c_2}^2}$$

**Advantage over PCA**: Directly optimizes for class discrimination rather than variance. Preserves interpretability.
**Disadvantage**: Ignores feature interactions; may select redundant features.

**For our study**: PCA is the primary reduction method (matches the controlled experimental design). Feature selection is noted as a potential enhancement for Phase 6 review.

---

## 5. Effect of Preprocessing on Kernel Properties

### 5.1 General Framework

Let $T$ be a preprocessing transformation. The preprocessed kernel is:
$$K_T(\mathbf{x}, \mathbf{x}') = K(T(\mathbf{x}), T(\mathbf{x}'))$$

The kernel-target alignment changes to:
$$A(K_T, Y) = \frac{\sum_{ij} K(T(\mathbf{x}_i), T(\mathbf{x}_j)) y_i y_j}{\|K_T\|_F \|Y\|_F}$$

**Theorem (Preprocessing as kernel transformation)**. For angle encoding with $R_y$, the preprocessed kernel is:
$$K_T(\mathbf{x}, \mathbf{x}') = \prod_{i=1}^{d'} \cos^2\left(\frac{T(\mathbf{x})_i - T(\mathbf{x}')_i}{2}\right)$$

This shows that preprocessing acts as a **feature-space warping** that changes the metric on the data space.

### 5.2 Information-Preserving Preprocessing

**Definition**: Preprocessing $T$ is **encoding-compatible** if:
$$A(K_T, Y) \geq A(K_I, Y)$$

where $K_I$ is the kernel with identity (no) preprocessing. In words, preprocessing should improve (or at least not harm) the kernel-target alignment.

**Conditions for encoding-compatible preprocessing**:

1. **For angle encoding**: $T$ should map the data to $[0, \pi]$ such that same-class points are mapped close together and different-class points are mapped far apart in the $\cos^2$ metric.

2. **For IQP encoding**: $T$ should preserve feature correlations that are class-relevant, because IQP's quadratic terms $T(\mathbf{x})_i T(\mathbf{x})_j$ model these correlations.

3. **For amplitude encoding**: $T$ should normalize data so that class-discriminative information is in the direction (not magnitude) of the feature vector.

### 5.3 Preprocessing Failure Modes

**Failure Mode 1: Bandwidth mismatch (MinMax)**

If features have very different scales (e.g., Wine dataset: feature ranges from ~0.5 to ~1700), MinMax normalization maps each feature independently to $[0, a]$. This means features with large ranges get fine-grained encoding while features with small ranges get coarse encoding. If the small-range features are class-discriminative, their resolution is poor.

**Example**: Wine dataset, features "Alcohol" (range 11.0-14.8) and "Proline" (range 278-1680).
- MinMax to $[0, \pi]$: "Alcohol" gets $\pi$ radians per 3.8 units; "Proline" gets $\pi$ radians per 1402 units.
- The kernel sensitivity to "Proline" changes is 370x lower than to "Alcohol" changes.

**Mitigation**: Z-score standardization equalizes feature sensitivity.

**Failure Mode 2: Correlation destruction (PCA + IQP)**

PCA decorrelates features, making IQP's cross-terms $T(\mathbf{x})_i T(\mathbf{x})_j$ model noise rather than structure.

**Example**: Cancer dataset has features with $r > 0.9$ correlations. These correlations are class-relevant (correlated features indicate tumor characteristics). PCA removes these correlations, potentially reducing IQP's advantage.

**Mitigation**: For IQP encoding, consider using standardized (non-PCA) features.

**Failure Mode 3: Magnitude loss (L2 normalization + amplitude)**

L2 normalization discards magnitude information. If class separation depends on feature magnitudes, this information is lost.

**Example**: Credit Fraud dataset "Amount" feature. Fraudulent transactions may have systematically different amounts than legitimate ones. L2 normalization removes this signal.

**Mitigation**: Include magnitude as an additional feature before normalization: $\tilde{\mathbf{x}} = [\mathbf{x}/\|\mathbf{x}\|, \|\mathbf{x}\|] \in \mathbb{R}^{d+1}$.

### 5.4 Optimal Preprocessing by Encoding

| Encoding | Best preprocessing | Rationale | Worst preprocessing | Why |
|----------|-------------------|-----------|--------------------|----|
| Angle | Z-score + sigmoid | Equalizes features, robust to outliers | Raw data | Unbounded range, kernel bandwidth undefined |
| IQP | MinMax (no PCA) | Preserves correlations for cross-terms | PCA | Destroys correlations IQP exploits |
| Re-uploading | Z-score + sigmoid or MinMax | Flexible (trainable layers adapt) | L2 (loses magnitude) | Unless magnitude is irrelevant |
| Amplitude | PCA + L2 | Reduces dim, normalizes to unit sphere | MinMax without L2 | Violates normalization constraint |
| Basis | Threshold binarization | Converts to required binary format | Z-score | Continuous output inappropriate |

---

## 6. Hybrid Classical-Quantum Preprocessing Pipeline

### 6.1 Pipeline Architecture

The full preprocessing pipeline for our experiments:

```
Raw Data X ∈ R^{n x d}
    |
    v
[1. Missing Value Imputation] (if needed)
    |
    v
[2. Categorical Encoding] (if mixed types: one-hot or ordinal)
    |
    v
[3. Dimensionality Reduction] (if d > target qubits)
    |  - PCA to k components (k = 4 or 8)
    |  - or Feature Selection (top-k by Fisher ratio)
    |
    v
X' ∈ R^{n x k}
    |
    v
[4. Normalization] (encoding-specific)
    |  - Angle: MinMax to [0, pi] or Z-score + sigmoid
    |  - IQP: MinMax to [0, pi]
    |  - Amplitude: L2 normalization
    |  - Basis: Threshold binarization
    |  - Re-uploading: MinMax to [0, pi] or Z-score + sigmoid
    |
    v
X'' ∈ R^{n x k} (encoding-ready)
    |
    v
[5. Quantum Encoding U(x'')]
    |
    v
|phi(x'')> ∈ H = (C^2)^{⊗m}
```

### 6.2 Pipeline for Each Preprocessing Level

Our experiment has 4 preprocessing levels. Formally:

**Level 0 -- None**: $T_0(\mathbf{x}) = \mathbf{x}$ (identity; raw data fed to encoding after only basic range clipping to $[0, 2\pi]$).

**Level 1 -- MinMax to $[0, \pi]$**: $T_1(\mathbf{x})_j = \frac{x_j - x_j^{\min}}{x_j^{\max} - x_j^{\min}} \cdot \pi$

**Level 2 -- Z-score standardization**: $T_2(\mathbf{x})_j = \pi \cdot \sigma\left(\frac{x_j - \mu_j}{\sigma_j}\right)$ where $\sigma(z) = 1/(1+e^{-z})$.

**Level 3 -- PCA + MinMax**: $T_3(\mathbf{x}) = \text{MinMax}_{[0,\pi]}(\text{PCA}_k(\mathbf{x}))$ where $k = \min(d, 8)$.

### 6.3 Dataset-Specific Preprocessing Requirements

| Dataset | Step 1 (Missing) | Step 2 (Categorical) | Step 3 (Reduction) | Notes |
|---------|------|---------|------|-------|
| Iris | None | None | None ($d=4$) | Already encoding-ready |
| Wine | None | None | PCA to 8 or use all 13 | Feature scale varies greatly |
| Cancer | None | None | PCA to 8 (from 30) | High correlation; PCA effective |
| MNIST | None | None | PCA to 8 (from 784) | Heavy reduction |
| Credit Fraud | None | None | Select 8 (from 30) | Already PCA-transformed |
| NSL-KDD | None | One-hot (3 features) | PCA to 8 (from ~44) | Mixed types |
| HAR | None | None | PCA to 8 (from 561) | Very high dim |
| Heart Disease | Median imputation (6 missing) | Ordinal encoding | PCA to 8 or use all 13 | Mixed types |

---

## 7. Theoretical Predictions for Preprocessing-Encoding Interactions

### 7.1 Testable Hypotheses

**H2a**: MinMax normalization produces higher accuracy than raw data for angle encoding on datasets with features of different scales (Wine, HAR).

**H2b**: Z-score normalization produces higher accuracy than MinMax for angle encoding on datasets with outliers (Credit Fraud, NSL-KDD).

**H2c**: PCA preprocessing reduces or eliminates the accuracy advantage of IQP encoding over angle encoding, because PCA removes the feature correlations that IQP exploits.

**H2d**: For amplitude encoding, L2 normalization after PCA produces higher accuracy than MinMax normalization, because amplitude encoding requires normalized input.

**H2e**: Data re-uploading is the most robust to preprocessing choice, because its trainable parameters can adapt to different data distributions.

### 7.2 Expected Interaction Patterns

| | None | MinMax | Z-score | PCA + MinMax |
|--|------|--------|---------|--------------|
| **Angle** | Poor (unbounded) | Good | Better (outlier robust) | Good (decorrelated) |
| **IQP** | Poor | Good | Moderate | Moderate (loses correlations) |
| **Re-upload** | Poor | Good | Good | Good |
| **Amplitude** | Invalid | Poor (not normalized) | Poor (not normalized) | Good (with L2) |
| **Basis** | Invalid | Invalid | Invalid | Poor (needs binarization) |

### 7.3 Preprocessing and Barren Plateaus

Preprocessing affects trainability through its effect on the loss landscape. The key mechanism is:

**Narrower data distribution** (aggressive normalization to small range) -> higher kernel values between data points -> flatter loss landscape -> harder optimization.

**Wider data distribution** (normalization to full $[0, 2\pi]$) -> lower kernel values (more diverse states) -> steeper loss landscape -> easier optimization but higher variance.

**Optimal trade-off**: Normalization range should be matched to the data's intrinsic scale. For $R_y$ angle encoding, the optimal range $[0, a^*]$ satisfies:

$$a^* \approx \frac{\pi}{\sqrt{d}} \cdot \frac{1}{\text{median inter-class distance}}$$

This ensures that the kernel can distinguish between classes without being too dispersive.

---

## 8. Summary

### 8.1 Key Findings

1. **Preprocessing is not neutral**: Different normalization schemes create fundamentally different kernel properties, even with the same quantum encoding.

2. **MinMax sensitivity**: MinMax normalization is sensitive to outliers and creates unequal feature resolution for mixed-scale data. Z-score + sigmoid mapping provides more robust results.

3. **PCA-IQP conflict**: PCA preprocessing may undermine IQP encoding's advantage by removing the feature correlations that IQP is designed to capture. This is a key testable prediction.

4. **Amplitude encoding requires specific preprocessing**: L2 normalization is mandatory but loses magnitude information. Include magnitude as an additional feature when magnitude is class-relevant.

5. **Preprocessing controls kernel bandwidth**: The normalization range acts as an implicit bandwidth parameter for the quantum kernel. This provides an indirect but important tool for optimizing encoding effectiveness.

6. **Re-uploading is the most preprocessing-robust**: Its trainable parameters can compensate for suboptimal preprocessing choices.

### 8.2 For the Test Engineer

Test the following preprocessing behaviors:
- MinMax correctly maps to $[0, \pi]$ for all features
- Z-score + sigmoid produces outputs in $(0, \pi)$
- PCA + MinMax produces decorrelated, bounded features
- L2 normalization produces unit-norm vectors
- Raw data (no preprocessing) does not crash encoding circuits (handle edge cases)

### 8.3 For the Python Architect

Implementation requirements:
- Each preprocessing level should be a sklearn-compatible transformer
- Preprocessing should store fitted parameters for test-time transformation
- The pipeline should validate preprocessing-encoding compatibility (e.g., warn if non-normalized data is fed to amplitude encoding)
- Consider a `PreprocessingRecommender` that suggests preprocessing based on data profile and encoding choice

---

## References

1. Heredge, J. et al. (2024). The role of classical preprocessing in quantum machine learning pipelines. *Quantum Machine Intelligence*.
2. Peters, E. et al. (2023). Machine learning of high-dimensional data on a noisy quantum processor. *npj Quantum Information*.
3. Shaydulin, R. & Wild, S.M. (2022). Importance of kernel bandwidth in quantum machine learning. *PRX Quantum*, 3, 040328.
4. Thanasilp, S. et al. (2024). Exponential concentration in quantum kernel methods. *Nature Communications*, 15, 1.
5. Cerezo, M. et al. (2022). Challenges and opportunities in quantum machine learning. *Nature Computational Science*, 2, 567-576.
6. Schuld, M. & Petruccione, F. (2021). *Machine Learning with Quantum Computers*. Springer.
