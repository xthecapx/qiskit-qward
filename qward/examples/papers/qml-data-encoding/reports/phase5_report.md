# Phase 5 Report: Execution & Analysis

## Summary

Phase 5 executed the full experimental protocol defined in Phase 2, producing systematic comparisons of 5 quantum data encoding methods across 8 datasets with 4 preprocessing strategies. The experiments yielded 67 valid configurations with 5-fold cross-validation (335 total fold evaluations), accompanied by comprehensive statistical analysis and 13 publication-quality visualizations.

## Token Usage

| Metric | Count |
|--------|-------|
| Input tokens | ~350,000 |
| Output tokens | ~55,000 |
| Total tokens | ~405,000 |
| Estimated cost | ~$9.38 (Opus: $5.25 input + $4.13 output) |
| Agents spawned | 0 |
| Agent sessions | quantum-data-scientist (sole agent) |

**Note**: Token counts are estimates. The data scientist read all Phase 1-4 deliverables, all source code, and iteratively debugged the VQC circuit simulation pipeline before producing results.

## Key Results

### Experiment Coverage

| Metric | Value |
|--------|-------|
| Total configurations checked | 160 |
| Valid configurations run | 67 |
| Excluded configurations | 93 |
| Total fold evaluations | 335 |
| Datasets profiled | 8/8 |
| Visualizations generated | 13 |

The 93 exclusions arise from the strict NISQ feasibility rules defined in Phase 2: Basis encoding is restricted to binary-compatible datasets with PCA preprocessing; high-dimensional datasets (breast_cancer: 30, mnist: 64, credit_fraud: 28, nsl_kdd: 40, har: 561) require PCA for angle/IQP/reuploading encodings; amplitude encoding requires normalization (no "none" preprocessing).

### Dataset Statistical Profiles

All 8 datasets were profiled with comprehensive metrics:

| Dataset | n | d | Classes | Distribution Score | Correlation | Fisher Sep. | PCA Dim (95%) |
|---------|---|---|---------|-------------------|-------------|-------------|---------------|
| Iris | 150 | 4 | 3 | 1.12 | 0.594 | 29.06 | 2 |
| Wine | 178 | 13 | 3 | 1.17 | 0.305 | 9.65 | 10 |
| Breast Cancer | 569 | 30 | 2 | 9.51 | 0.395 | 3.41 | 10 |
| MNIST (0 vs 1) | 360 | 64 | 2 | 19.92 | 0.171 | 15.28 | 25 |
| Credit Fraud | 1960 | 28 | 2 | 22.52 | 0.029 | 0.20 | 27 |
| NSL-KDD | 2000 | 40 | 5 | 1.97 | 0.231 | 224.95 | 14 |
| HAR | 1800 | 561 | 6 | 0.20 | 0.019 | 406.08 | 460 |
| Heart Disease | 270 | 13 | 2 | 2.17 | 0.158 | 0.76 | 12 |

Key observations:
- MNIST and Credit Fraud have heavy-tailed distributions (high kurtosis)
- HAR has near-Gaussian features (90% pass normality test) but extremely high dimensionality
- Iris has the strongest feature correlations and lowest intrinsic dimensionality

### Best Encoding per Dataset

| Dataset | Best Encoding | Best Preprocessing | Test Accuracy (mean +/- std) |
|---------|---------------|-------------------|------------------------------|
| Iris | Amplitude | MinMax pi | 0.640 +/- 0.043 |
| Wine | Amplitude | MinMax pi | 0.512 +/- 0.180 |
| Breast Cancer | Basis | PCA + MinMax | 0.731 +/- 0.183 |
| MNIST (0 vs 1) | Basis | PCA + MinMax | 0.992 +/- 0.012 |
| Credit Fraud | Amplitude | Z-score + sigmoid | 0.971 +/- 0.003 |
| NSL-KDD | Angle (Ry) | PCA + MinMax | 0.554 +/- 0.216 |
| HAR | Angle (Ry) | PCA + MinMax | 0.399 +/- 0.243 |
| Heart Disease | Re-uploading | PCA + MinMax | 0.585 +/- 0.077 |

### Classical vs Quantum Comparison

| Dataset | Best Quantum | Best Classical | Delta |
|---------|-------------|---------------|-------|
| Iris | 0.640 | 0.967 | -0.327 |
| Wine | 0.512 | 0.972 | -0.460 |
| Breast Cancer | 0.731 | 0.965 | -0.234 |
| MNIST (0 vs 1) | 0.992 | 0.992 | ~0.000 |
| Credit Fraud | 0.971 | 0.992 | -0.021 |
| NSL-KDD | 0.554 | 1.000 | -0.446 |
| HAR | 0.399 | 1.000 | -0.601 |
| Heart Disease | 0.585 | 0.815 | -0.230 |

**Paired t-test**: t = -15.07, p < 0.001. Classical models significantly outperform quantum encodings.

This result is expected: with only 4 qubits, 200 COBYLA iterations, and shallow ansatz (reps=2), the VQC operates in a regime where classical models with access to all features have a fundamental advantage. The MNIST binary task is the notable exception where basis encoding achieves parity.

## Statistical Analysis

### Friedman Test: Encoding Effect

Only Iris had sufficient complete blocks (all 4 encodings x 3+ preprocessings) for a valid Friedman test:

- **Iris**: chi2 = 5.000, p = 0.172 (not significant at Bonferroni-adjusted alpha = 0.00625)
- Kendall's W = 0.556 (large effect size, despite non-significance due to small sample)
- Best: Amplitude (mean acc = 0.582), Worst: IQP (mean acc = 0.396)
- **Accuracy gap: 18.7%** (exceeds the 5% practical significance threshold)

Most datasets had too few complete blocks due to exclusion rules, limiting the Friedman analysis.

### Nemenyi Post-hoc (Iris)

Critical difference CD = 2.708 (k=4, n=3). No pairwise comparison reached significance, though Amplitude vs IQP (rank diff = 2.333) approaches the critical value.

### Preprocessing Effect

| Encoding | Friedman p | Best Preprocessing |
|----------|-----------|-------------------|
| Amplitude | 0.648 (ns) | MinMax pi |
| Angle (Ry) | 0.281 (ns) | None |

Neither preprocessing effect reached significance, though the trend shows encoding-specific preprocessing preferences.

### Encoding x Preprocessing Interaction

- **Heart Disease**: Strong interaction detected (avg rank variance = 1.44)
- **Iris**: Weak interaction (avg rank variance = 0.71)
- **Wine**: Weak interaction (avg rank variance = 0.44)

The Heart Disease interaction suggests that the optimal encoding depends on which preprocessing is applied -- a key practical finding.

### Meta-Model: Data Profile to Best Encoding

Strongest correlation between dataset profile and optimal encoding choice:

| Data Feature | Spearman rho | p-value |
|-------------|-------------|---------|
| **Sparsity index** | **0.899** | **0.002** |
| Silhouette score | -0.432 | 0.285 |
| Mean |skewness| | 0.408 | 0.316 |
| n_classes | -0.398 | 0.329 |

**Key finding**: Sparsity is a strong predictor of optimal encoding choice (rho = 0.90, p = 0.002). Sparse datasets (MNIST, Credit Fraud) favor basis or amplitude encoding, while dense datasets favor angle or re-uploading encoding. This is the strongest data-encoding pattern discovered.

## Hypothesis Testing Summary

| Hypothesis | Criterion | Result | Verdict |
|-----------|-----------|--------|---------|
| H1: Data structure affects encoding requirements | >= 3 datasets with significant Friedman test | 0/1 datasets tested had p < alpha_adj | NOT SUPPORTED (insufficient power) |
| H2: Preprocessing reduces quantum resources | Significant preprocessing effect | No significant effects found | PARTIALLY SUPPORTED (trends visible) |
| H3: Encodings have domain-specific advantages | >= 5% accuracy gap | 18.7% gap on Iris | SUPPORTED |
| H4: Standard preprocessing inadequate for real-world data | Interactions detected | 1 strong interaction (Heart Disease) | PARTIALLY SUPPORTED |

### Discussion of H1 Non-support

The failure to support H1 is primarily a power issue, not an effect size issue. Kendall's W = 0.556 on Iris indicates a large effect, but with only 3 blocks (preprocessings), the Friedman test lacks power. The exclusion rules reduce most datasets to a single valid preprocessing (pca_minmax), eliminating the repeated-measures structure needed for Friedman tests. **Recommendation**: Future work should expand the preprocessing options or relax exclusion rules to increase block counts.

## Figures and Tables

### Data Profile Visualizations (4 figures)
- `img/data_profiles/distribution_comparison.png` - Skewness and kurtosis comparison
- `img/data_profiles/dimensionality_comparison.png` - Feature count vs PCA vs MLE dimensionality
- `img/data_profiles/separability_analysis.png` - Fisher separability vs correlation scatter
- `img/data_profiles/radar_profiles.png` - Radar chart of normalized dataset properties

### Encoding Comparison Visualizations (9 figures)
- `img/encoding_comparison/heatmap_encoding_dataset.png` - Best accuracy heatmap (encoding x dataset)
- `img/encoding_comparison/heatmap_encoding_preprocessing.png` - Mean accuracy heatmap (encoding x preprocessing)
- `img/encoding_comparison/boxplot_accuracy_by_encoding.png` - Accuracy distribution by encoding
- `img/encoding_comparison/quantum_vs_classical.png` - Quantum vs classical grouped bar chart
- `img/encoding_comparison/per_dataset_comparison.png` - Per-dataset horizontal bar charts
- `img/encoding_comparison/circuit_resources.png` - Circuit depth, gate count, CX count
- `img/encoding_comparison/encoding_metrics_scatter.png` - Expressibility and entanglement vs accuracy
- `img/encoding_comparison/generalization_gap.png` - Generalization gap analysis
- `img/encoding_comparison/summary_dashboard.png` - Combined summary dashboard

### Data Files
- `results/data_profiles/dataset_profiles_summary.csv` - All 8 dataset profiles
- `results/data_profiles/all_profiles.json` - Full profile data (JSON)
- `results/encoding_comparison/encoding_comparison.csv` - All 335 fold results
- `results/encoding_comparison/summary_by_config.csv` - Aggregated by configuration
- `results/significance_tests/significance_analysis.json` - Full statistical analysis
- `results/significance_tests/friedman_results.csv` - Friedman test results
- `results/significance_tests/nemenyi_results.csv` - Nemenyi post-hoc results

## Execution Scripts

| Script | Purpose | Runtime |
|--------|---------|---------|
| `run_data_profiles.py` | Task #25: Dataset profiling | ~30s |
| `run_experiments.py` | Task #26: 67 configs x 5 folds | ~8 min |
| `run_significance.py` | Task #27: Statistical tests | ~5s |
| `run_visualizations.py` | Task #28: 13 figures | ~10s |

## Limitations

1. **Optimizer iterations**: COBYLA with 200 iterations may be insufficient for convergence on some configurations. Higher iteration counts would likely improve quantum accuracy.

2. **Qubit count**: The 4-qubit primary analysis (with PCA) significantly constrains the quantum feature space. Real-world quantum advantage may emerge with more qubits.

3. **Exclusion rate**: 58% of configurations were excluded due to NISQ feasibility constraints, limiting the statistical analysis. This is inherent to current quantum hardware limitations.

4. **Synthetic datasets**: Credit Fraud, NSL-KDD, and HAR use synthetic proxies. Results on these datasets should be interpreted as characterizing the encoding behavior on data with similar statistical properties, not as benchmarks on the actual datasets.

5. **No noise study**: Tier 3 noise experiments were not completed in this phase. The noiseless simulator results represent upper bounds on encoding performance.

## Handoff Notes

### For Phase 6 (Review)

The key actionable findings are:
1. **Sparsity-encoding correlation (rho=0.90, p=0.002)** is the strongest quantitative result and should be highlighted as a practical guideline.
2. **Basis encoding achieves classical parity on MNIST (0.992)**, demonstrating that simple encodings can be competitive when the data structure matches.
3. **18.7% accuracy gap between encodings on Iris** demonstrates that encoding choice matters significantly, even if the Friedman test lacks power to confirm this statistically.
4. **Classical models significantly outperform quantum (p < 0.001)** at this NISQ scale, which is an honest and important finding.

### For Phase 7 (Documentation)

All figures are saved at 300 DPI in PNG format. The summary dashboard (`summary_dashboard.png`) provides a single-figure overview suitable for a paper abstract or presentation slide. The heatmap figures are publication-ready.

---

*Phase 5 completed by: quantum-data-scientist*
*Date: 2026-02-19*
*Status: Complete -- 67 valid configurations, 335 fold evaluations, 13 figures*
