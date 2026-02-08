---
name: data-science
description: Data science skill for scientific visualization, statistical analysis, and machine learning with Python. Use when creating plots (matplotlib, seaborn), performing statistical tests (scipy, statsmodels, pingouin), building ML models (scikit-learn), analyzing QWARD metric DataFrames, generating publication-ready figures, conducting hypothesis testing, regression analysis, time series forecasting, or producing APA-formatted statistical reports. Covers pandas, numpy, matplotlib, seaborn, statsmodels, scipy, and scikit-learn workflows.
---

# Data Science

## Overview

Comprehensive data science skill for the QWARD quantum computing project. Covers visualization, statistical analysis, and machine learning workflows for analyzing quantum circuit metrics and producing publication-quality results.

## Quick Start

### Plot QWARD Metrics

```python
import matplotlib.pyplot as plt
import pandas as pd
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics

scanner = Scanner(circuit=circuit)
scanner.add_strategy(QiskitMetrics(circuit))
scanner.add_strategy(ComplexityMetrics(circuit))
results = scanner.calculate_metrics()

# Plot complexity metrics
df = results['complexity']
fig, ax = plt.subplots(figsize=(10, 6))
df.plot(kind='bar', ax=ax)
ax.set_title('Circuit Complexity Metrics')
plt.savefig('qward/examples/img/complexity.png', dpi=300, bbox_inches='tight')
```

### Statistical Comparison

```python
from scipy import stats
import numpy as np

# Compare metrics across circuits
t_stat, p_value = stats.ttest_ind(metrics_circuit_a, metrics_circuit_b)
cohens_d = (np.mean(metrics_circuit_a) - np.mean(metrics_circuit_b)) / np.sqrt(
    (np.std(metrics_circuit_a)**2 + np.std(metrics_circuit_b)**2) / 2
)
print(f"t = {t_stat:.2f}, p = {p_value:.3f}, d = {cohens_d:.2f}")
```

## Core Capabilities

### 1. Visualization (matplotlib + seaborn)

For creating plots, charts, and publication-ready figures.

**Reference:** See `references/visualization.md` for complete guidance on:
- Object-oriented matplotlib API (recommended over pyplot)
- All plot types: line, scatter, bar, histogram, heatmap, violin, radar
- Multi-panel figures with GridSpec and subplot_mosaic
- Seaborn statistical plots with automatic CIs
- Colorblind-safe palettes (Okabe-Ito, viridis, cividis)
- Publication export (PNG 300dpi, PDF/SVG vector, TIFF)
- Journal-specific styling (Nature, Science, Cell dimensions)
- 3D plots, contour plots, annotations

**Quick patterns:**
```python
# Always use OO interface
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
ax.plot(x, y, linewidth=2, label='data')
ax.set_xlabel('X Label (units)')
ax.set_ylabel('Y Label (units)')
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('qward/examples/img/plot.png', dpi=300, bbox_inches='tight')
```

### 2. Statistical Analysis

For hypothesis testing, assumption checking, and reporting.

**Reference:** See `references/statistics.md` for complete guidance on:
- Test selection guide (parametric vs non-parametric)
- Assumption checking (normality, homogeneity, linearity)
- t-tests, ANOVA, chi-square, Mann-Whitney, Kruskal-Wallis
- Effect sizes (Cohen's d, eta-squared, Cramer's V) with CIs
- Power analysis and sample size calculations
- Multiple comparison corrections (Bonferroni, FDR)
- Bayesian alternatives (Bayes Factors, credible intervals)
- APA-style reporting templates

### 3. Statistical Modeling (statsmodels)

For regression, GLMs, time series, and econometric analysis.

**Reference:** See `references/modeling.md` for complete guidance on:
- Linear regression (OLS, WLS, GLS) with diagnostics
- Generalized linear models (logistic, Poisson, Gamma)
- Time series (ARIMA, SARIMAX, VAR, exponential smoothing)
- Formula API (R-style): `smf.ols('y ~ x1 + C(group)', data=df)`
- Robust standard errors (HC, HAC, cluster-robust)
- Model comparison (AIC, BIC, likelihood ratio tests)
- Residual diagnostics and influence analysis

### 4. Machine Learning (scikit-learn)

For classification, regression, clustering, and dimensionality reduction.

**Reference:** See `references/machine-learning.md` for complete guidance on:
- Pipelines and ColumnTransformer for production workflows
- Preprocessing (StandardScaler, OneHotEncoder, imputation)
- Supervised learning (RandomForest, GradientBoosting, SVM, KNN)
- Unsupervised learning (KMeans, DBSCAN, PCA, t-SNE, UMAP)
- Cross-validation and hyperparameter tuning (GridSearchCV)
- Metrics (classification_report, ROC AUC, silhouette score)
- Feature importance and model interpretability

## QWARD-Specific Workflows

### Analyzing Scanner Output

```python
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics
import pandas as pd

scanner = Scanner(circuit=circuit)
scanner.add_strategy(QiskitMetrics(circuit))
scanner.add_strategy(ComplexityMetrics(circuit))
results = scanner.calculate_metrics()

# Each value is a DataFrame
for metric_name, df in results.items():
    print(f"\n{metric_name}:")
    print(df.describe())
```

### Comparing Circuits

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Collect metrics across circuits
all_metrics = []
for name, circuit in circuits.items():
    scanner = Scanner(circuit=circuit)
    scanner.add_strategy(ComplexityMetrics(circuit))
    result = scanner.calculate_metrics()
    df = result['complexity']
    df['circuit'] = name
    all_metrics.append(df)

combined = pd.concat(all_metrics, ignore_index=True)

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=combined, x='circuit', y='gate_density', ax=ax)
ax.set_ylabel('Gate Density')
sns.despine()
plt.savefig('qward/examples/img/comparison.png', dpi=300, bbox_inches='tight')
```

### Correlation Analysis

```python
import seaborn as sns
import numpy as np

# Correlation between complexity metrics
corr = df_metrics.corr(method='spearman')
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, square=True, ax=ax)
plt.savefig('qward/examples/img/correlation.png', dpi=300, bbox_inches='tight')
```

## Best Practices

### Visualization
- Use OO interface (`fig, ax = plt.subplots()`) for production code
- Use `constrained_layout=True` to prevent overlapping
- Use colorblind-friendly palettes (viridis, cividis, Okabe-Ito)
- Save images to `qward/examples/img/` per project convention
- 300 DPI for publications, vector (PDF/SVG) when possible

### Statistics
- Always check assumptions before interpreting tests
- Report effect sizes with confidence intervals, not just p-values
- Use non-parametric tests when normality is violated
- Correct for multiple comparisons when testing many hypotheses
- Use Bayesian methods when you need evidence for the null

### Machine Learning
- Always use Pipelines to prevent data leakage
- Use stratified splits for classification
- Set `random_state` for reproducibility
- Scale features for SVM, KNN, neural networks (not for trees)
- Report cross-validated metrics, not just train/test
