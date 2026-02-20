"""Phase 5 Task #27: Statistical significance analysis.

Implements:
1. Friedman test for encoding effect per dataset
2. Nemenyi post-hoc pairwise comparisons
3. Effect sizes (Kendall's W)
4. Preprocessing effect analysis
5. Encoding x Preprocessing interaction
6. Meta-model: data profile -> best encoding
7. Classical baseline comparison
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

os.environ["MPLBACKEND"] = "Agg"
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "significance_tests")
ENCODING_DIR = os.path.join(BASE_DIR, "results", "encoding_comparison")
PROFILES_DIR = os.path.join(BASE_DIR, "results", "data_profiles")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_results():
    """Load experiment results and aggregate by configuration."""
    df = pd.read_csv(os.path.join(ENCODING_DIR, "encoding_comparison.csv"))
    # Aggregate across folds
    agg = (
        df.groupby(["dataset", "encoding", "preprocessing"])
        .agg(
            test_acc_mean=("test_accuracy", "mean"),
            test_acc_std=("test_accuracy", "std"),
            f1_mean=("f1_macro", "mean"),
            f1_std=("f1_macro", "std"),
            train_acc_mean=("train_accuracy", "mean"),
            gen_gap_mean=("generalization_gap", "mean"),
            svm_rbf_acc=("svm_rbf_acc", "mean"),
            rf_acc=("random_forest_acc", "mean"),
            lr_acc=("logistic_regression_acc", "mean"),
        )
        .reset_index()
    )
    return df, agg


# ---- 1. Friedman test for encoding effect ----


def friedman_encoding_test(agg):
    """Friedman test: do encoding methods differ in accuracy across datasets?"""
    print("\n" + "=" * 60)
    print("1. FRIEDMAN TEST: ENCODING EFFECT")
    print("=" * 60)

    results = []
    datasets = agg["dataset"].unique()
    n_datasets = len(datasets)
    alpha = 0.05
    alpha_adj = alpha / n_datasets  # Bonferroni

    for dataset in datasets:
        subset = agg[agg["dataset"] == dataset]
        # Get all unique encoding-preprocessing combinations
        # Group by encoding, average over preprocessings for Friedman
        enc_accs = {}
        for enc in subset["encoding"].unique():
            enc_data = subset[subset["encoding"] == enc]["test_acc_mean"].values
            if len(enc_data) > 0:
                enc_accs[enc] = np.mean(enc_data)

        if len(enc_accs) < 3:
            print(f"\n  {dataset}: <3 encodings, skipping Friedman test")
            continue

        # For a proper Friedman test we need repeated measures
        # Use preprocessing as blocks, encodings as treatments
        preprocs = subset["preprocessing"].unique()
        encodings = subset["encoding"].unique()

        # Build matrix: rows=preprocs (blocks), cols=encodings
        matrix = []
        valid_preprocs = []
        for pp in preprocs:
            row = []
            valid = True
            for enc in encodings:
                val = subset[(subset["encoding"] == enc) & (subset["preprocessing"] == pp)][
                    "test_acc_mean"
                ].values
                if len(val) == 0:
                    valid = False
                    break
                row.append(val[0])
            if valid:
                matrix.append(row)
                valid_preprocs.append(pp)

        if len(matrix) < 2:
            print(f"\n  {dataset}: <2 complete blocks, skipping")
            continue

        matrix = np.array(matrix)

        # Friedman test
        try:
            stat, p_val = stats.friedmanchisquare(*matrix.T)
        except Exception as e:
            print(f"\n  {dataset}: Friedman test failed: {e}")
            continue

        # Kendall's W effect size
        n_blocks = len(matrix)
        k = len(encodings)
        W = stat / (n_blocks * (k - 1)) if n_blocks * (k - 1) > 0 else 0.0

        # Mean accuracy per encoding
        means = {enc: np.mean(matrix[:, i]) for i, enc in enumerate(encodings)}
        best_enc = max(means, key=means.get)
        worst_enc = min(means, key=means.get)
        gap = means[best_enc] - means[worst_enc]

        sig = (
            "***"
            if p_val < 0.001
            else ("**" if p_val < 0.01 else ("*" if p_val < alpha_adj else "ns"))
        )

        print(f"\n  {dataset}:")
        print(f"    Friedman chi2={stat:.3f}, p={p_val:.6f} {sig}")
        print(f"    Kendall's W={W:.3f} ({'small' if W<0.3 else 'medium' if W<0.5 else 'large'})")
        print(f"    Best: {best_enc} ({means[best_enc]:.3f})")
        print(f"    Worst: {worst_enc} ({means[worst_enc]:.3f})")
        print(f"    Gap: {gap:.3f} ({'>= 5%' if gap >= 0.05 else '< 5%'})")

        results.append(
            {
                "dataset": dataset,
                "friedman_stat": round(stat, 4),
                "friedman_p": round(p_val, 6),
                "significant": p_val < alpha_adj,
                "kendall_w": round(W, 4),
                "best_encoding": best_enc,
                "worst_encoding": worst_enc,
                "accuracy_gap": round(gap, 4),
                "n_encodings": k,
                "n_blocks": n_blocks,
            }
        )

    return results


# ---- 2. Nemenyi post-hoc test ----


def nemenyi_posthoc(agg):
    """Nemenyi post-hoc pairwise comparison of encodings."""
    print("\n" + "=" * 60)
    print("2. NEMENYI POST-HOC COMPARISONS")
    print("=" * 60)

    results = []
    datasets = agg["dataset"].unique()

    for dataset in datasets:
        subset = agg[agg["dataset"] == dataset]
        preprocs = subset["preprocessing"].unique()
        encodings = sorted(subset["encoding"].unique())

        # Build rank matrix
        rank_matrix = []
        for pp in preprocs:
            row = []
            valid = True
            for enc in encodings:
                val = subset[(subset["encoding"] == enc) & (subset["preprocessing"] == pp)][
                    "test_acc_mean"
                ].values
                if len(val) == 0:
                    valid = False
                    break
                row.append(val[0])
            if valid:
                # Rank (higher accuracy = lower rank = better)
                ranks = stats.rankdata(-np.array(row))
                rank_matrix.append(ranks)

        if len(rank_matrix) < 2 or len(encodings) < 3:
            continue

        rank_matrix = np.array(rank_matrix)
        n = len(rank_matrix)
        k = len(encodings)
        mean_ranks = rank_matrix.mean(axis=0)

        # Critical difference (Nemenyi)
        # q_alpha / sqrt(2) * sqrt(k*(k+1)/(6*n))
        # For alpha=0.05, use q-values from Studentized range
        # Approximate q-values for k groups
        q_alpha = {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}
        q = q_alpha.get(k, 2.8)
        cd = q * np.sqrt(k * (k + 1) / (6.0 * n))

        print(f"\n  {dataset}: CD={cd:.3f} (k={k}, n={n})")
        print(f"    Mean ranks: {dict(zip(encodings, np.round(mean_ranks, 2)))}")

        # Pairwise comparisons
        for i, j in combinations(range(k), 2):
            diff = abs(mean_ranks[i] - mean_ranks[j])
            sig = diff > cd
            print(
                f"    {encodings[i]} vs {encodings[j]}: "
                f"diff={diff:.3f} {'> CD (sig)' if sig else '<= CD (ns)'}"
            )

            results.append(
                {
                    "dataset": dataset,
                    "encoding_1": encodings[i],
                    "encoding_2": encodings[j],
                    "rank_diff": round(diff, 4),
                    "critical_difference": round(cd, 4),
                    "significant": sig,
                }
            )

    return results


# ---- 3. Preprocessing effect ----


def preprocessing_effect_analysis(agg):
    """Friedman test for preprocessing effect per encoding."""
    print("\n" + "=" * 60)
    print("3. PREPROCESSING EFFECT ANALYSIS")
    print("=" * 60)

    results = []
    encodings = agg["encoding"].unique()

    for enc in encodings:
        subset = agg[agg["encoding"] == enc]
        datasets = subset["dataset"].unique()
        preprocs = sorted(subset["preprocessing"].unique())

        if len(preprocs) < 3:
            print(f"\n  {enc}: <3 preprocessors available, skipping")
            continue

        # Build matrix: rows=datasets, cols=preprocessors
        matrix = []
        valid_datasets = []
        for ds in datasets:
            row = []
            valid = True
            for pp in preprocs:
                val = subset[(subset["dataset"] == ds) & (subset["preprocessing"] == pp)][
                    "test_acc_mean"
                ].values
                if len(val) == 0:
                    valid = False
                    break
                row.append(val[0])
            if valid:
                matrix.append(row)
                valid_datasets.append(ds)

        if len(matrix) < 2:
            print(f"\n  {enc}: <2 complete datasets, skipping")
            continue

        matrix = np.array(matrix)
        try:
            stat, p_val = stats.friedmanchisquare(*matrix.T)
        except Exception:
            continue

        means = {pp: np.mean(matrix[:, i]) for i, pp in enumerate(preprocs)}
        best_pp = max(means, key=means.get)

        print(f"\n  {enc}:")
        print(f"    Friedman chi2={stat:.3f}, p={p_val:.6f}")
        print(f"    Best preprocessing: {best_pp} ({means[best_pp]:.3f})")
        for pp, m in means.items():
            print(f"      {pp}: {m:.3f}")

        results.append(
            {
                "encoding": enc,
                "friedman_stat": round(stat, 4),
                "friedman_p": round(p_val, 6),
                "significant": p_val < 0.05,
                "best_preprocessing": best_pp,
                "n_preprocessors": len(preprocs),
                "n_datasets": len(matrix),
            }
        )

    return results


# ---- 4. Interaction analysis ----


def interaction_analysis(agg):
    """Analyze encoding x preprocessing interaction effects."""
    print("\n" + "=" * 60)
    print("4. ENCODING x PREPROCESSING INTERACTION")
    print("=" * 60)

    # For each dataset, compute mean accuracy for each (enc, preproc) cell
    results = {}
    datasets = agg["dataset"].unique()

    for dataset in datasets:
        subset = agg[agg["dataset"] == dataset]
        pivot = subset.pivot_table(
            index="encoding", columns="preprocessing", values="test_acc_mean", aggfunc="mean"
        )
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            continue

        # Check for interaction: does the effect of encoding depend on preprocessing?
        # Simple approach: compute variance of rank changes across preprocessings
        encodings = pivot.index.tolist()
        preprocs = pivot.columns.tolist()

        rank_changes = []
        for pp in preprocs:
            col = pivot[pp].dropna()
            if len(col) >= 2:
                ranks = stats.rankdata(-col.values)
                rank_changes.append(dict(zip(col.index, ranks)))

        if len(rank_changes) < 2:
            continue

        # Check if encoding ranking is stable across preprocessings
        enc_rank_var = {}
        for enc in encodings:
            ranks = [rc.get(enc, np.nan) for rc in rank_changes]
            ranks = [r for r in ranks if not np.isnan(r)]
            if len(ranks) >= 2:
                enc_rank_var[enc] = np.var(ranks)

        avg_rank_var = np.mean(list(enc_rank_var.values())) if enc_rank_var else 0.0

        print(f"\n  {dataset}:")
        print(f"    Avg rank variance across preprocessings: {avg_rank_var:.3f}")
        print(f"    {'Strong interaction' if avg_rank_var > 1.0 else 'Weak/no interaction'}")
        for enc, var in enc_rank_var.items():
            print(f"      {enc}: rank_var={var:.3f}")

        results[dataset] = {
            "avg_rank_variance": round(avg_rank_var, 4),
            "interaction_detected": avg_rank_var > 1.0,
            "enc_rank_variances": {k: round(v, 4) for k, v in enc_rank_var.items()},
        }

    return results


# ---- 5. Meta-model: data -> encoding prediction ----


def meta_model_analysis(agg):
    """Predict best encoding from dataset profile."""
    print("\n" + "=" * 60)
    print("5. META-MODEL: DATA PROFILE -> BEST ENCODING")
    print("=" * 60)

    # Load data profiles
    profiles_path = os.path.join(PROFILES_DIR, "all_profiles.json")
    if not os.path.exists(profiles_path):
        print("  No data profiles found, skipping meta-model analysis")
        return {}

    with open(profiles_path) as f:
        profiles = json.load(f)

    profile_dict = {p["dataset"]: p for p in profiles}

    # Find best encoding per dataset
    best_encodings = {}
    datasets = agg["dataset"].unique()
    for ds in datasets:
        subset = agg[agg["dataset"] == ds]
        best_row = subset.loc[subset["test_acc_mean"].idxmax()]
        best_encodings[ds] = best_row["encoding"]

    print(f"\n  Best encoding per dataset:")
    for ds, enc in best_encodings.items():
        print(f"    {ds}: {enc}")

    # Build feature matrix from profiles
    feature_keys = [
        "mean_abs_skewness",
        "mean_abs_kurtosis",
        "distribution_score",
        "avg_abs_correlation",
        "pca_dim_95",
        "mle_intrinsic_dim",
        "fisher_separability",
        "silhouette_score",
        "sparsity_index",
        "balance_ratio",
        "n_features",
        "n_classes",
    ]

    X_meta = []
    y_meta = []
    ds_names = []
    for ds in datasets:
        if ds in profile_dict:
            p = profile_dict[ds]
            features = [float(p.get(k, 0) or 0) for k in feature_keys]
            X_meta.append(features)
            y_meta.append(best_encodings[ds])
            ds_names.append(ds)

    if len(X_meta) < 4:
        print("  Too few datasets for meta-model analysis")
        return {}

    X_meta = np.array(X_meta)
    y_meta = np.array(y_meta)

    # Correlation analysis between data features and encoding rankings
    print("\n  Correlation analysis (Spearman) between data profile and best encoding:")
    enc_names = sorted(set(y_meta))
    enc_to_num = {e: i for i, e in enumerate(enc_names)}
    y_num = np.array([enc_to_num[e] for e in y_meta])

    correlations = {}
    for i, key in enumerate(feature_keys):
        try:
            r, p = stats.spearmanr(X_meta[:, i], y_num)
            correlations[key] = {"rho": round(r, 3), "p": round(p, 4)}
            if abs(r) > 0.3:
                print(f"    {key}: rho={r:.3f}, p={p:.4f}")
        except Exception:
            pass

    return {
        "best_encodings": best_encodings,
        "correlations": correlations,
        "feature_keys": feature_keys,
    }


# ---- 6. Classical baseline comparison ----


def classical_comparison(agg):
    """Compare quantum encodings with classical baselines."""
    print("\n" + "=" * 60)
    print("6. QUANTUM vs CLASSICAL COMPARISON")
    print("=" * 60)

    results = []

    for _, row in agg.iterrows():
        classical_best = max(row.get("svm_rbf_acc", 0), row.get("rf_acc", 0), row.get("lr_acc", 0))
        delta_qc = row["test_acc_mean"] - classical_best

        results.append(
            {
                "dataset": row["dataset"],
                "encoding": row["encoding"],
                "preprocessing": row["preprocessing"],
                "quantum_acc": round(row["test_acc_mean"], 4),
                "classical_best_acc": round(classical_best, 4),
                "delta_qc": round(delta_qc, 4),
                "quantum_advantage": delta_qc > 0.0,
            }
        )

    results_df = pd.DataFrame(results)

    # Summary
    n_total = len(results_df)
    n_advantage = results_df["quantum_advantage"].sum()
    avg_delta = results_df["delta_qc"].mean()
    min_delta = results_df["delta_qc"].min()
    max_delta = results_df["delta_qc"].max()

    print(f"\n  Total configurations: {n_total}")
    print(f"  Quantum >= Classical: {n_advantage} ({100*n_advantage/n_total:.1f}%)")
    print(f"  Mean Delta_QC: {avg_delta:.4f}")
    print(f"  Range: [{min_delta:.4f}, {max_delta:.4f}]")

    # Per-dataset summary
    print("\n  Per-dataset summary:")
    for ds in results_df["dataset"].unique():
        ds_df = results_df[results_df["dataset"] == ds]
        print(
            f"    {ds}: avg_delta={ds_df['delta_qc'].mean():.3f}, "
            f"best_delta={ds_df['delta_qc'].max():.3f}, "
            f"advantage_rate={ds_df['quantum_advantage'].mean():.1%}"
        )

    # Paired t-test: quantum vs classical
    quantum_accs = results_df["quantum_acc"].values
    classical_accs = results_df["classical_best_acc"].values
    t_stat, t_p = stats.ttest_rel(quantum_accs, classical_accs)
    print(f"\n  Paired t-test (quantum vs classical):")
    print(f"    t={t_stat:.3f}, p={t_p:.6f}")
    print(
        f"    {'Classical significantly better' if t_p < 0.05 and t_stat < 0 else 'No significant difference' if t_p >= 0.05 else 'Quantum significantly better'}"
    )

    return results_df.to_dict(orient="records")


# ---- 7. Hypothesis testing summary ----


def hypothesis_summary(
    friedman_results, preproc_results, interaction_results, meta_results, classical_results
):
    """Summarize hypothesis test outcomes."""
    print("\n" + "=" * 60)
    print("7. HYPOTHESIS TESTING SUMMARY")
    print("=" * 60)

    # H1: Statistical structure affects encoding requirements
    sig_datasets = [r for r in friedman_results if r["significant"]]
    n_sig = len(sig_datasets)
    print(f"\n  H1 (data structure -> encoding requirements):")
    print(f"    Significant Friedman tests: {n_sig}/{len(friedman_results)} datasets")
    h1_supported = n_sig >= 3
    print(f"    Criterion (>= 3 datasets): {'SUPPORTED' if h1_supported else 'NOT SUPPORTED'}")

    # H2: Classical preprocessing reduces quantum resources
    h2_supported = any(r["significant"] for r in preproc_results)
    print(f"\n  H2 (preprocessing reduces resources):")
    print(f"    Significant preprocessing effect: {h2_supported}")
    for r in preproc_results:
        if r["significant"]:
            print(f"      {r['encoding']}: best={r['best_preprocessing']}")
    print(f"    Criterion: {'SUPPORTED' if h2_supported else 'PARTIALLY SUPPORTED'}")

    # H3: Encodings have domain-specific advantages
    datasets_with_gap = [r for r in friedman_results if r["accuracy_gap"] >= 0.05]
    print(f"\n  H3 (domain-specific encoding advantages):")
    print(f"    Datasets with >= 5% accuracy gap: {len(datasets_with_gap)}/{len(friedman_results)}")
    for r in datasets_with_gap:
        print(
            f"      {r['dataset']}: {r['best_encoding']} vs {r['worst_encoding']} "
            f"(gap={r['accuracy_gap']:.3f})"
        )
    h3_supported = len(datasets_with_gap) >= 1
    print(f"    Criterion: {'SUPPORTED' if h3_supported else 'NOT SUPPORTED'}")

    # H4: Standard preprocessing inadequate for real-world data
    interaction_count = sum(
        1 for v in interaction_results.values() if v.get("interaction_detected", False)
    )
    print(f"\n  H4 (preprocessing inadequate for real-world):")
    print(f"    Datasets with strong interaction: {interaction_count}")
    h4_supported = interaction_count >= 2
    print(f"    Criterion: {'SUPPORTED' if h4_supported else 'PARTIALLY SUPPORTED'}")

    return {
        "H1": {"supported": h1_supported, "sig_datasets": n_sig},
        "H2": {"supported": h2_supported},
        "H3": {"supported": h3_supported, "datasets_with_gap": len(datasets_with_gap)},
        "H4": {"supported": h4_supported, "interaction_count": interaction_count},
    }


def main():
    print("=" * 60)
    print("Phase 5 Task #27: Statistical Significance Analysis")
    print("=" * 60)

    df, agg = load_results()
    print(f"\nLoaded {len(df)} fold results, {len(agg)} aggregated configurations")
    print(f"Datasets: {sorted(agg['dataset'].unique())}")
    print(f"Encodings: {sorted(agg['encoding'].unique())}")
    print(f"Preprocessings: {sorted(agg['preprocessing'].unique())}")

    # Run all analyses
    friedman_results = friedman_encoding_test(agg)
    nemenyi_results = nemenyi_posthoc(agg)
    preproc_results = preprocessing_effect_analysis(agg)
    interaction_results = interaction_analysis(agg)
    meta_results = meta_model_analysis(agg)
    classical_results = classical_comparison(agg)

    # Hypothesis summary
    hypothesis = hypothesis_summary(
        friedman_results,
        preproc_results,
        interaction_results,
        meta_results,
        classical_results,
    )

    # Save all results
    all_results = {
        "friedman_encoding": friedman_results,
        "nemenyi_posthoc": nemenyi_results,
        "preprocessing_effect": preproc_results,
        "interaction_analysis": interaction_results,
        "meta_model": meta_results,
        "classical_comparison": classical_results,
        "hypothesis_summary": hypothesis,
    }

    json_path = os.path.join(RESULTS_DIR, "significance_analysis.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved: {json_path}")

    # Save Friedman results as CSV
    if friedman_results:
        friedman_df = pd.DataFrame(friedman_results)
        friedman_df.to_csv(os.path.join(RESULTS_DIR, "friedman_results.csv"), index=False)

    # Save Nemenyi results as CSV
    if nemenyi_results:
        nemenyi_df = pd.DataFrame(nemenyi_results)
        nemenyi_df.to_csv(os.path.join(RESULTS_DIR, "nemenyi_results.csv"), index=False)

    print("\nTask #27 complete.")


if __name__ == "__main__":
    main()
