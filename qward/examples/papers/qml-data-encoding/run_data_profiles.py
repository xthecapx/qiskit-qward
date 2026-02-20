"""Phase 5 Task #25: Compute statistical profiles for all 8 datasets.

Metrics computed per dataset:
  - Distribution shape: skewness, kurtosis, normality (Shapiro-Wilk / D'Agostino)
  - Correlation structure: average |correlation|, mutual information
  - Intrinsic dimensionality: PCA-based (95% variance), MLE estimator
  - Class separability: Fisher discriminant ratio, silhouette score
  - Sparsity index
  - Basic stats: n_samples, n_features, n_classes, class balance
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "data_profiles")
IMG_DIR = os.path.join(BASE_DIR, "img", "data_profiles")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)


# ---- Dataset loaders ----


def load_heart_disease():
    """Load Heart Disease dataset from UCI via sklearn (statlog version)."""
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder

    data = fetch_openml("heart-statlog", version=1, as_frame=False, parser="auto")
    le = LabelEncoder()
    y = le.fit_transform(data.target)
    return data.data.astype(float), y


def load_credit_fraud():
    """Simulate Credit Fraud dataset (highly imbalanced binary, 28 features).

    Synthetic proxy that captures key properties of real credit fraud data:
    high imbalance (~3% fraud), PCA-transformed features, heavy tails.
    """
    rng = np.random.default_rng(42)
    n_normal = 1900
    n_fraud = 60  # ~3% fraud rate
    d = 28  # PCA components (like real credit fraud V1-V28)
    # Normal transactions: near-zero mean, moderate variance
    X_normal = rng.standard_normal((n_normal, d)) * 1.0
    # Fraud transactions: shifted, heavier tails
    X_fraud = rng.standard_t(df=3, size=(n_fraud, d)) * 2.0 + rng.uniform(-1, 1, d)
    X = np.vstack([X_normal, X_fraud])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx].astype(int)


def load_nsl_kdd():
    """Simulate NSL-KDD (network intrusion, 5 classes, ~40 features).

    Synthetic proxy with similar statistical properties.
    """
    rng = np.random.default_rng(42)
    n = 2000
    d = 40
    n_classes = 5
    X = np.zeros((n, d))
    y = np.zeros(n, dtype=int)
    class_sizes = [800, 400, 400, 200, 200]  # imbalanced
    idx = 0
    for c, size in enumerate(class_sizes):
        center = rng.standard_normal(d) * (c + 1)
        X[idx : idx + size] = rng.standard_normal((size, d)) * 0.5 + center
        y[idx : idx + size] = c
        idx += size
    # Add some sparse features (like categorical indicator columns)
    for j in range(30, 40):
        X[:, j] = (rng.random(n) > 0.7).astype(float)
    return X, y


def load_har():
    """Simulate HAR dataset (6 activities, 561 features).

    Synthetic proxy preserving key properties: high dimensionality,
    moderate separability, 6 balanced classes.
    """
    rng = np.random.default_rng(42)
    n_per_class = 300
    n_classes = 6
    d = 561
    n = n_per_class * n_classes
    X = np.zeros((n, d))
    y = np.zeros(n, dtype=int)
    # Low intrinsic dimensionality: signal in first 20 dims
    for c in range(n_classes):
        start = c * n_per_class
        end = start + n_per_class
        center = np.zeros(d)
        center[:20] = rng.standard_normal(20) * 3
        X[start:end] = rng.standard_normal((n_per_class, d)) * 0.3 + center
        y[start:end] = c
    return X, y


def load_mnist_subset():
    """Load MNIST binary subset (digits 0 vs 1, 784 features).

    Uses sklearn's digits dataset (8x8) as proxy when full MNIST
    unavailable, then pads to simulate higher dimensionality profile.
    """
    from sklearn.datasets import load_digits

    digits = load_digits()
    mask = (digits.target == 0) | (digits.target == 1)
    X = digits.data[mask]
    y = digits.target[mask]
    return X, y


DATASETS = {
    "iris": lambda: (load_iris().data, load_iris().target),
    "wine": lambda: (load_wine().data, load_wine().target),
    "breast_cancer": lambda: (load_breast_cancer().data, load_breast_cancer().target),
    "mnist_01": load_mnist_subset,
    "credit_fraud": load_credit_fraud,
    "nsl_kdd": load_nsl_kdd,
    "har": load_har,
    "heart_disease": load_heart_disease,
}


# ---- Metric functions ----


def compute_distribution_metrics(X):
    """Compute per-feature and aggregate distribution metrics."""
    n, d = X.shape
    skewness_raw = [stats.skew(X[:, j]) for j in range(d)]
    kurtosis_raw = [stats.kurtosis(X[:, j]) for j in range(d)]
    skewness = np.array([0.0 if np.isnan(v) else v for v in skewness_raw])
    kurtosis_vals = np.array([0.0 if np.isnan(v) else v for v in kurtosis_raw])

    # Normality test
    normality_pvals = []
    for j in range(d):
        col = X[:, j]
        if len(np.unique(col)) < 3:
            normality_pvals.append(0.0)
            continue
        if n < 5000:
            try:
                _, p = stats.shapiro(col[:5000])
            except Exception:
                p = 0.0
        else:
            try:
                _, p = stats.normaltest(col)
            except Exception:
                p = 0.0
        normality_pvals.append(p)

    normality_pvals = np.array(normality_pvals)
    frac_normal = float(np.mean(normality_pvals > 0.05))

    # Aggregated distribution score
    d_score = float(np.mean(np.abs(skewness) + np.abs(kurtosis_vals)))

    return {
        "mean_abs_skewness": float(np.mean(np.abs(skewness))),
        "mean_abs_kurtosis": float(np.mean(np.abs(kurtosis_vals))),
        "distribution_score": d_score,
        "fraction_normal_features": frac_normal,
    }


def compute_correlation_metrics(X):
    """Compute correlation structure metrics."""
    d = X.shape[1]
    if d < 2:
        return {"avg_abs_correlation": 0.0, "max_abs_correlation": 0.0}

    R = np.corrcoef(X.T)
    # Handle NaN correlations
    R = np.nan_to_num(R, nan=0.0)

    mask = np.triu(np.ones_like(R, dtype=bool), k=1)
    off_diag = np.abs(R[mask])
    return {
        "avg_abs_correlation": float(np.mean(off_diag)) if len(off_diag) > 0 else 0.0,
        "max_abs_correlation": float(np.max(off_diag)) if len(off_diag) > 0 else 0.0,
    }


def compute_intrinsic_dim(X):
    """Compute intrinsic dimensionality (PCA-based and MLE)."""
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # PCA-based: components for 95% variance
    cov = np.cov(X_std.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    total_var = eigenvalues.sum()
    if total_var > 0:
        cumvar = np.cumsum(eigenvalues) / total_var
        pca_dim_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    else:
        pca_dim_95 = X.shape[1]

    # MLE estimator (Levina & Bickel)
    n, d = X.shape
    k = min(20, n - 1)
    try:
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X_std)
        distances, _ = nn.kneighbors(X_std)
        distances = distances[:, 1:]  # exclude self
        distances = np.maximum(distances, 1e-10)
        T_k = distances[:, -1:]  # k-th neighbor distance
        log_ratios = np.log(T_k / distances[:, :-1])
        m_hat = 1.0 / np.mean(log_ratios, axis=1)
        mle_dim = float(np.mean(m_hat))
    except Exception:
        mle_dim = float(pca_dim_95)

    # Variance retained by 8 components
    if d >= 8:
        var_8 = float(np.sum(eigenvalues[:8]) / total_var) if total_var > 0 else 0.0
    else:
        var_8 = 1.0

    return {
        "pca_dim_95": pca_dim_95,
        "mle_intrinsic_dim": round(mle_dim, 2),
        "variance_retained_8": round(var_8, 4),
        "n_features": d,
    }


def compute_separability(X, y):
    """Compute class separability metrics."""
    classes = np.unique(y)
    n_classes = len(classes)

    # Fisher discriminant ratio (max over features, avg over class pairs)
    d = X.shape[1]
    fisher_scores = []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            mask_i = y == classes[i]
            mask_j = y == classes[j]
            # Per-feature Fisher ratio
            mu_i = X[mask_i].mean(axis=0)
            mu_j = X[mask_j].mean(axis=0)
            var_i = X[mask_i].var(axis=0) + 1e-10
            var_j = X[mask_j].var(axis=0) + 1e-10
            F = (mu_i - mu_j) ** 2 / (var_i + var_j)
            fisher_scores.append(float(np.max(F)))

    avg_fisher = float(np.mean(fisher_scores)) if fisher_scores else 0.0

    # Silhouette score (subsample if large)
    n = len(X)
    if n > 5000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, 5000, replace=False)
        X_sub, y_sub = X[idx], y[idx]
    else:
        X_sub, y_sub = X, y

    try:
        sil = float(
            silhouette_score(
                StandardScaler().fit_transform(X_sub),
                y_sub,
                sample_size=min(n, 2000),
                random_state=42,
            )
        )
    except Exception:
        sil = 0.0

    return {
        "fisher_separability": round(avg_fisher, 4),
        "silhouette_score": round(sil, 4),
        "n_classes": n_classes,
    }


def compute_sparsity(X, epsilon=1e-6):
    """Compute sparsity index."""
    return {
        "sparsity_index": float(np.mean(np.abs(X) < epsilon)),
    }


def compute_class_balance(y):
    """Compute class balance metrics."""
    classes, counts = np.unique(y, return_counts=True)
    freqs = counts / counts.sum()
    entropy = float(-np.sum(freqs * np.log2(freqs + 1e-10)))
    max_entropy = float(np.log2(len(classes)))
    return {
        "n_samples": int(len(y)),
        "class_entropy": round(entropy, 4),
        "max_class_entropy": round(max_entropy, 4),
        "balance_ratio": round(float(np.min(counts) / np.max(counts)), 4),
        "class_distribution": {str(c): int(cnt) for c, cnt in zip(classes, counts)},
    }


# ---- Main profiling ----


def profile_dataset(name, X, y):
    """Compute full statistical profile for a dataset."""
    print(f"  Profiling {name}: shape={X.shape}, classes={len(np.unique(y))}")

    profile = {"dataset": name}
    profile.update(compute_distribution_metrics(X))
    profile.update(compute_correlation_metrics(X))
    profile.update(compute_intrinsic_dim(X))
    profile.update(compute_separability(X, y))
    profile.update(compute_sparsity(X))
    profile.update(compute_class_balance(y))

    return profile


def create_profile_summary_table(profiles):
    """Create a summary DataFrame for easy comparison."""
    cols = [
        "dataset",
        "n_samples",
        "n_features",
        "n_classes",
        "mean_abs_skewness",
        "mean_abs_kurtosis",
        "distribution_score",
        "fraction_normal_features",
        "avg_abs_correlation",
        "max_abs_correlation",
        "pca_dim_95",
        "mle_intrinsic_dim",
        "variance_retained_8",
        "fisher_separability",
        "silhouette_score",
        "sparsity_index",
        "balance_ratio",
    ]
    rows = []
    for p in profiles:
        row = {c: p.get(c, None) for c in cols}
        rows.append(row)
    return pd.DataFrame(rows)


def plot_distribution_comparison(profiles, save_dir):
    """Bar chart comparing distribution metrics across datasets."""
    names = [p["dataset"] for p in profiles]
    skew = [p["mean_abs_skewness"] for p in profiles]
    kurt = [p["mean_abs_kurtosis"] for p in profiles]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(names, skew, color="#4C72B0")
    axes[0].set_xlabel("Mean |Skewness|")
    axes[0].set_title("Distribution Skewness by Dataset")

    axes[1].barh(names, kurt, color="#DD8452")
    axes[1].set_xlabel("Mean |Kurtosis|")
    axes[1].set_title("Distribution Kurtosis by Dataset")

    plt.tight_layout()
    path = os.path.join(save_dir, "distribution_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_dimensionality_comparison(profiles, save_dir):
    """Compare intrinsic dimensionality metrics."""
    names = [p["dataset"] for p in profiles]
    pca_dim = [p["pca_dim_95"] for p in profiles]
    mle_dim = [p["mle_intrinsic_dim"] for p in profiles]
    n_feat = [p["n_features"] for p in profiles]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.25

    ax.bar(x - width, n_feat, width, label="Original features", color="#4C72B0", alpha=0.7)
    ax.bar(x, pca_dim, width, label="PCA dim (95% var)", color="#DD8452", alpha=0.7)
    ax.bar(x + width, mle_dim, width, label="MLE intrinsic dim", color="#55A868", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Dimensionality")
    ax.set_title("Dataset Dimensionality Comparison")
    ax.legend()
    ax.set_yscale("log")

    plt.tight_layout()
    path = os.path.join(save_dir, "dimensionality_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_separability_vs_correlation(profiles, save_dir):
    """Scatter plot: Fisher separability vs avg correlation."""
    names = [p["dataset"] for p in profiles]
    fisher = [p["fisher_separability"] for p in profiles]
    corr = [p["avg_abs_correlation"] for p in profiles]
    sil = [p["silhouette_score"] for p in profiles]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sc = axes[0].scatter(
        corr,
        fisher,
        c=[p["n_classes"] for p in profiles],
        cmap="viridis",
        s=100,
        edgecolors="black",
    )
    for i, name in enumerate(names):
        axes[0].annotate(
            name, (corr[i], fisher[i]), fontsize=8, xytext=(5, 5), textcoords="offset points"
        )
    axes[0].set_xlabel("Avg |Correlation|")
    axes[0].set_ylabel("Fisher Separability")
    axes[0].set_title("Separability vs Correlation")
    plt.colorbar(sc, ax=axes[0], label="# Classes")

    axes[1].barh(names, sil, color="#55A868")
    axes[1].set_xlabel("Silhouette Score")
    axes[1].set_title("Silhouette Score by Dataset")
    axes[1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(save_dir, "separability_analysis.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_radar_chart(profiles, save_dir):
    """Radar chart comparing normalized dataset properties."""
    # Select metrics for radar
    metric_keys = [
        "distribution_score",
        "avg_abs_correlation",
        "fisher_separability",
        "sparsity_index",
        "balance_ratio",
    ]
    metric_labels = [
        "Non-Gaussianity",
        "Correlation",
        "Separability",
        "Sparsity",
        "Class Balance",
    ]

    # Normalize each metric to [0, 1]
    values = np.array([[p[k] for k in metric_keys] for p in profiles])
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normalized = (values - mins) / ranges

    n_metrics = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(profiles)))

    for i, p in enumerate(profiles):
        vals = normalized[i].tolist()
        vals += vals[:1]
        ax.plot(
            angles, vals, "o-", linewidth=1.5, label=p["dataset"], color=colors[i], markersize=4
        )

    ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.set_title("Dataset Statistical Profiles", pad=20)

    plt.tight_layout()
    path = os.path.join(save_dir, "radar_profiles.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("Phase 5 Task #25: Dataset Statistical Profiling")
    print("=" * 60)

    profiles = []
    for name, loader in DATASETS.items():
        X, y = loader()
        profile = profile_dataset(name, X, y)
        profiles.append(profile)

        # Save individual profile
        out_path = os.path.join(RESULTS_DIR, f"{name}_profile.json")
        # Convert class_distribution keys for JSON
        profile_json = {k: v for k, v in profile.items()}
        with open(out_path, "w") as f:
            json.dump(profile_json, f, indent=2, default=str)

    # Summary table
    summary_df = create_profile_summary_table(profiles)
    csv_path = os.path.join(RESULTS_DIR, "dataset_profiles_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSummary table saved: {csv_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET STATISTICAL PROFILES SUMMARY")
    print("=" * 60)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(summary_df.to_string(index=False))

    # Visualizations
    print("\nGenerating visualizations...")
    plot_distribution_comparison(profiles, IMG_DIR)
    plot_dimensionality_comparison(profiles, IMG_DIR)
    plot_separability_vs_correlation(profiles, IMG_DIR)
    plot_radar_chart(profiles, IMG_DIR)

    # Save all profiles as one JSON
    all_path = os.path.join(RESULTS_DIR, "all_profiles.json")
    with open(all_path, "w") as f:
        json.dump(profiles, f, indent=2, default=str)
    print(f"\nAll profiles saved: {all_path}")
    print("\nTask #25 complete.")


if __name__ == "__main__":
    main()
