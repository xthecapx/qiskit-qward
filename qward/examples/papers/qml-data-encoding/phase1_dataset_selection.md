# Phase 1: Target Dataset Selection with Statistical Profiles

## 1. Selection Criteria

Datasets were selected to span the following axes of variation, ensuring our study covers the space of data characteristics that QML practitioners encounter:

| Criterion | Range Covered |
|-----------|---------------|
| Dimensionality | 4 features (Iris) to 561+ features (HAR) |
| Sample size | 150 (Iris) to ~285K (Credit Fraud) |
| Distribution shape | Gaussian, multimodal, heavy-tailed |
| Class balance | Balanced (Iris) to extreme imbalance (Credit Fraud: 0.17%) |
| Feature types | Continuous, mixed, categorical+continuous |
| Correlation structure | Low (Iris) to high inter-feature correlations (Cancer) |
| Domain | Biology, finance, sensor, medical, image |

---

## 2. Benchmark Datasets (Baseline Comparison)

These well-studied datasets serve as baselines where classical and quantum methods have known performance profiles.

### 2.1 Iris

| Property | Value |
|----------|-------|
| **Source** | UCI ML Repository (Fisher, 1936) |
| **Samples** | 150 |
| **Features** | 4 (sepal length, sepal width, petal length, petal width) |
| **Classes** | 3 (Setosa, Versicolor, Virginica) |
| **Balance** | Balanced (50/50/50) |
| **Distribution** | Approximately Gaussian per class |
| **Correlations** | Low to moderate (petal features correlated, r ~ 0.96) |
| **Intrinsic dimensionality** | ~2 (PCA: 2 components explain ~97.8% variance) |
| **Class separability** | High (Setosa linearly separable; Versicolor/Virginica overlap) |
| **Encoding relevance** | 4 features map directly to 4 qubits for angle encoding; 2 qubits for amplitude encoding (2^2 = 4) |
| **Classical baseline** | SVM with RBF: ~97% accuracy |
| **Why selected** | Universal QML benchmark; simple enough to validate encoding implementations |

### 2.2 Wine

| Property | Value |
|----------|-------|
| **Source** | UCI ML Repository |
| **Samples** | 178 |
| **Features** | 13 (chemical analysis measurements) |
| **Classes** | 3 (cultivar types) |
| **Balance** | Slightly imbalanced (59/71/48) |
| **Distribution** | Mixed -- some Gaussian, some skewed (e.g., proline: skewness ~1.0) |
| **Correlations** | Moderate; feature scales vary by 2 orders of magnitude |
| **Intrinsic dimensionality** | ~5-6 (PCA: 6 components explain ~85% variance) |
| **Class separability** | Moderate (Fisher ratio ~1.5) |
| **Encoding relevance** | 13 features require either 13 qubits (angle) or PCA reduction; tests mixed-scale sensitivity |
| **Classical baseline** | SVM with RBF: ~98% accuracy |
| **Why selected** | Tests encoding robustness to mixed feature scales and moderate dimensionality |

### 2.3 Breast Cancer Wisconsin (Diagnostic)

| Property | Value |
|----------|-------|
| **Source** | UCI ML Repository (Wolberg et al., 1995) |
| **Samples** | 569 |
| **Features** | 30 (computed from digitized FNA images) |
| **Classes** | 2 (Malignant, Benign) |
| **Balance** | Slightly imbalanced (212/357, ~37%/63%) |
| **Distribution** | Right-skewed for many features; mean features approximately Gaussian |
| **Correlations** | High inter-feature correlations (many feature pairs with r > 0.9) |
| **Intrinsic dimensionality** | ~6-7 (PCA: 7 components explain ~90% variance) |
| **Class separability** | High (Fisher ratio ~3.0 for best features) |
| **Encoding relevance** | 30 features necessitate dimensionality reduction; tests PCA+encoding pipelines |
| **Classical baseline** | Logistic Regression: ~96%; SVM: ~97% |
| **Why selected** | Tests high-dimensional encoding with correlated features; binary task simplifies QML evaluation |

### 2.4 MNIST (Digit Subset)

| Property | Value |
|----------|-------|
| **Source** | LeCun et al. (1998) |
| **Samples** | 1000 (subset: digits 0 and 1) |
| **Features** | 784 -> 16 (after PCA/autoencoder reduction) |
| **Classes** | 2 (binary: 0 vs 1) or 10 (full) |
| **Balance** | Approximately balanced per digit |
| **Distribution** | Sparse (many zero pixels); non-Gaussian |
| **Correlations** | Spatial correlations (adjacent pixel correlations) |
| **Intrinsic dimensionality** | ~12-15 for binary; ~30-40 for full 10-class |
| **Class separability** | High for binary (0 vs 1); moderate for multi-class |
| **Encoding relevance** | Requires heavy dimensionality reduction; tests encoding after lossy compression |
| **Classical baseline** | CNN: ~99.7%; SVM on PCA: ~98% (binary) |
| **Why selected** | Tests extreme dimensionality reduction pipeline; image data as a contrast to tabular |

---

## 3. Real-World Datasets (Primary Focus)

These datasets present challenges not found in benchmarks and are the primary focus of our study.

### 3.1 Credit Card Fraud Detection

| Property | Value |
|----------|-------|
| **Source** | Kaggle (ULB Machine Learning Group) |
| **Samples** | 284,807 |
| **Features** | 30 (28 PCA-transformed + Time + Amount) |
| **Classes** | 2 (Legitimate: 99.83%, Fraud: 0.17%) |
| **Balance** | Extreme imbalance (492 fraud out of 284,807) |
| **Distribution** | PCA features approximately Gaussian; Amount is heavy-tailed (skewness ~16.1) |
| **Correlations** | Low by construction (PCA features are orthogonal) |
| **Intrinsic dimensionality** | ~8-10 (already PCA-transformed) |
| **Class separability** | Low (fraud cases are rare and overlap with legitimate) |
| **Encoding relevance** | Tests encoding under extreme class imbalance; pre-PCA'd features test whether additional quantum encoding adds value |
| **Classical baseline** | XGBoost: ~0.85 F1; Isolation Forest: ~0.30 F1 |
| **Why selected** | Extreme imbalance tests whether encoding can create better class boundaries; already PCA-preprocessed (tests H2) |
| **Subsampling strategy** | Use all 492 fraud + 1000 random legitimate (1492 samples) for quantum experiments |

### 3.2 Network Intrusion Detection (NSL-KDD)

| Property | Value |
|----------|-------|
| **Source** | Canadian Institute for Cybersecurity |
| **Samples** | ~125,973 (train) + ~22,544 (test) |
| **Features** | 41 (mixed: 38 continuous + 3 categorical) |
| **Classes** | 5 (Normal, DoS, Probe, R2L, U2R) or binary |
| **Balance** | Imbalanced (Normal: ~53%, DoS: ~36%, others rare) |
| **Distribution** | Highly non-Gaussian; many features with heavy tails and zero inflation |
| **Correlations** | Moderate; categorical features require encoding |
| **Intrinsic dimensionality** | ~10-12 |
| **Class separability** | Moderate for DoS/Normal; low for R2L/U2R |
| **Encoding relevance** | Mixed feature types require hybrid encoding strategies; tests categorical data handling |
| **Classical baseline** | Random Forest: ~0.82 accuracy (multi-class) |
| **Why selected** | Mixed categorical+continuous features test encoding flexibility; cybersecurity domain |
| **Preprocessing notes** | One-hot encode categorical features (protocol_type, service, flag) before quantum encoding |

### 3.3 Human Activity Recognition (HAR)

| Property | Value |
|----------|-------|
| **Source** | UCI ML Repository (Anguita et al., 2013) |
| **Samples** | 10,299 (7,352 train + 2,947 test) |
| **Features** | 561 (time and frequency domain features from accelerometer/gyroscope) |
| **Classes** | 6 (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying) |
| **Balance** | Approximately balanced (1226-1777 per class) |
| **Distribution** | Mixed; many features approximately Gaussian; some bounded [-1, 1] |
| **Correlations** | High; many redundant features from overlapping time windows |
| **Intrinsic dimensionality** | ~15-20 (PCA: 20 components explain ~90% variance) |
| **Class separability** | Moderate (sitting/standing overlap; walking variants overlap) |
| **Encoding relevance** | Extremely high-dimensional; requires aggressive reduction (561 -> 8-16 qubits); tests whether encoding can preserve discriminative information after heavy compression |
| **Classical baseline** | SVM: ~96%; Random Forest: ~93% |
| **Why selected** | Highest dimensionality in our study; tests encoding after extreme PCA reduction |

### 3.4 Heart Disease (Cleveland)

| Property | Value |
|----------|-------|
| **Source** | UCI ML Repository (Janosi et al., 1988) |
| **Samples** | 303 (after removing missing values: ~297) |
| **Features** | 13 (age, sex, chest pain type, blood pressure, cholesterol, etc.) |
| **Classes** | 2 (presence/absence of heart disease) or 5 (severity) |
| **Balance** | Slightly imbalanced (~54% / ~46% for binary) |
| **Distribution** | Mixed types: 5 continuous, 3 ordinal, 3 binary, 2 nominal |
| **Correlations** | Moderate; complex non-linear relationships |
| **Intrinsic dimensionality** | ~5-7 |
| **Class separability** | Low to moderate (Fisher ratio ~0.8) |
| **Encoding relevance** | Mixed feature types in low dimensions; 13 features fit directly on 13 qubits for angle encoding; tests encoding of mixed-type data |
| **Classical baseline** | Logistic Regression: ~84%; SVM: ~83% |
| **Why selected** | Small sample, mixed types, low separability -- represents many real-world medical datasets; missing data handling is necessary |

---

## 4. Dataset-Encoding Compatibility Matrix (Hypothesis)

Based on the literature review, we hypothesize the following initial compatibility mapping (to be validated experimentally):

| Dataset | Angle | IQP | Re-uploading | Amplitude | Basis |
|---------|-------|-----|--------------|-----------|-------|
| Iris | Good | Overkill | Overkill | Good | N/A |
| Wine | Good (after scaling) | Good | Good | Moderate | N/A |
| Cancer | Moderate (needs PCA) | Good (after PCA) | Good | Moderate | N/A |
| MNIST | Poor (too many features) | Moderate (after PCA) | Good | Good (if state prep efficient) | N/A |
| Credit Fraud | Moderate | Good (non-linear boundaries needed) | Best (adaptive boundaries) | Moderate | N/A |
| NSL-KDD | Poor (mixed types) | Moderate | Good | Poor | Partial (categorical) |
| HAR | Poor (too many features) | Good (after PCA to ~16) | Good | Good | N/A |
| Heart Disease | Good | Good | Good | Good | Partial (binary features) |

**Legend**: Good = expected strong performance; Moderate = may work with tuning; Poor = fundamental mismatch; N/A = not applicable (non-binary data for basis encoding); Overkill = unnecessarily complex for simple data.

---

## 5. Data Acquisition Plan

| Dataset | Access Method | License |
|---------|--------------|---------|
| Iris | `sklearn.datasets.load_iris()` | Public domain |
| Wine | `sklearn.datasets.load_wine()` | Public domain |
| Breast Cancer | `sklearn.datasets.load_breast_cancer()` | Public domain |
| MNIST | `sklearn.datasets.fetch_openml('mnist_784')` | CC BY 4.0 |
| Credit Card Fraud | Kaggle download (requires API key) | Open Database License |
| NSL-KDD | Direct download from CICIDS | Research use |
| HAR | UCI ML Repository download | Research use |
| Heart Disease | `ucimlrepo.fetch_ucirepo(id=45)` or sklearn | CC BY 4.0 |

### Subsampling Strategy for Quantum Experiments

Given NISQ hardware constraints, we define standard subsample sizes for quantum experiments:

| Dataset | Full Size | Quantum Subsample | Rationale |
|---------|-----------|-------------------|-----------|
| Iris | 150 | 150 (full) | Small enough for full quantum processing |
| Wine | 178 | 178 (full) | Small enough for full quantum processing |
| Cancer | 569 | 300 (stratified) | Reduce circuit evaluations |
| MNIST | 70,000 | 200 (100 per class) | Extreme reduction for binary task |
| Credit Fraud | 284,807 | 1,492 (all fraud + 1K legit) | Preserve class representation |
| NSL-KDD | 148,517 | 2,000 (stratified) | Representative subsample |
| HAR | 10,299 | 1,200 (200 per class) | Balanced across 6 classes |
| Heart Disease | 303 | 297 (full, after cleaning) | Small enough for full processing |

---

## 6. Statistical Characterization Plan

For each dataset, we will compute the following profile:

### Per-Feature Statistics
- Mean, standard deviation, min, max
- Skewness, kurtosis
- Normality test (Shapiro-Wilk for n < 5000; D'Agostino-Pearson otherwise)

### Dataset-Level Statistics
- Pearson correlation matrix + mutual information matrix
- Intrinsic dimensionality (PCA explained variance curve; MLE estimator)
- Fisher's discriminant ratio (per class pair)
- Silhouette score (class cluster quality)
- Sparsity index (percentage of near-zero values)

### Encoding-Relevant Metrics
- Feature range (determines normalization needs for angle encoding)
- L2 norm distribution (determines amplitude encoding conditioning)
- Feature independence score (determines IQP interaction term relevance)
- Effective dimension after PCA at 95% variance threshold

---

## 7. Summary

We have selected **4 benchmark** and **4 real-world** datasets spanning:
- **Dimensionality**: 4 to 561 features
- **Class balance**: Balanced to 0.17% minority
- **Feature types**: Continuous, mixed, categorical
- **Distribution shapes**: Gaussian, skewed, heavy-tailed, multimodal
- **Domains**: Biology, finance, cybersecurity, sensor, medical, image

This selection ensures comprehensive coverage of the data characteristics that influence encoding effectiveness, allowing us to rigorously test hypotheses H1-H4.
