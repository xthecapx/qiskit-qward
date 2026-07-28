"""
Statistical comparison of the DSR evaluation profile components, and of
IBM vs. Rigetti (AWS) under the recomputed profile.

Produces two analyses (both printed to stdout and saved as CSVs):

  1. **Profile-internal comparison** (analogous to the old DSR-vs-HF/TVD
     table): success_rate vs. chance_corrected_success, success_rate vs.
     coarse_tvd_similarity, success_rate vs. coarse_hellinger_fidelity, per
     (algorithm, provider, qubit-count) group.
     -> ``statistical_comparison_profile_internal.csv``

  2. **Provider comparison under the profile** (IBM vs. Rigetti/AWS):
     Mann-Whitney U test + Cliff's delta on chance_corrected_success per
     (algorithm, qubit-count) group. This is the key input for deciding
     whether the "Rigetti signals failure first" narrative survives the
     move from full HF/TVD to the chance-corrected profile.
     -> ``statistical_comparison_profile_provider.csv``

Usage:
  PYTHONPATH=. uv run python qward/examples/papers/statistical_comparison_profile.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from statistical_comparison import bootstrap_median_ci, cliffs_delta, wilcoxon_test

warnings.filterwarnings("ignore", category=RuntimeWarning)

CSV_PATH = Path(__file__).parent / "DSR_result.csv"
OUT_INTERNAL_CSV = Path(__file__).parent / "statistical_comparison_profile_internal.csv"
OUT_PROVIDER_CSV = Path(__file__).parent / "statistical_comparison_profile_provider.csv"

PROFILE_METRICS = [
    "success_rate",
    "chance_corrected_success",
    "coarse_tvd_similarity",
    "coarse_hellinger_fidelity",
]
PROFILE_LABELS = {
    "success_rate": "Success",
    "chance_corrected_success": "Chance-Corrected",
    "coarse_tvd_similarity": "Coarse TVD-Sim",
    "coarse_hellinger_fidelity": "Coarse HF",
}

N_COMPARISONS = 3  # success vs each of the other three components


def _backend_group(name: str) -> str:
    n = str(name).lower()
    if "ankaa" in n or "forte" in n or "rigetti" in n:
        return "Rigetti"
    if "ibm" in n:
        return "IBM"
    return "other"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=PROFILE_METRICS)
    df["provider"] = df["backend_name"].apply(_backend_group)
    return df


# ---------------------------------------------------------------------------
# (1) Profile-internal comparison
# ---------------------------------------------------------------------------


def analyse_internal_group(grp_df: pd.DataFrame) -> dict:
    n = len(grp_df)
    row = {"n": n}

    for m in PROFILE_METRICS:
        label = PROFILE_LABELS[m]
        med, lo, hi = bootstrap_median_ci(grp_df[m].values)
        row[f"{label}_median"] = med
        row[f"{label}_ci_lo"] = lo
        row[f"{label}_ci_hi"] = hi

    success = grp_df["success_rate"].values
    for other, label in [
        ("chance_corrected_success", "Chance-Corrected"),
        ("coarse_tvd_similarity", "Coarse TVD-Sim"),
        ("coarse_hellinger_fidelity", "Coarse HF"),
    ]:
        other_vals = grp_df[other].values
        w, p_raw = wilcoxon_test(success, other_vals)
        p_corr = min(p_raw * N_COMPARISONS, 1.0) if not np.isnan(p_raw) else np.nan
        delta = cliffs_delta(success, other_vals)
        row[f"W_vs_{label}"] = w
        row[f"p_raw_vs_{label}"] = p_raw
        row[f"p_corr_vs_{label}"] = p_corr
        row[f"delta_vs_{label}"] = delta
    return row


def run_internal_analysis(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for algo in ("GROVER", "QFT"):
        sub_algo = df[df["algorithm"] == algo]
        for prov in ("IBM", "Rigetti"):
            sub = sub_algo[sub_algo["provider"] == prov]
            if prov == "IBM" and "optimization_level" in sub.columns:
                opt = pd.to_numeric(sub["optimization_level"], errors="coerce")
                sub = sub[opt.isna() | (opt == 3)]
            if sub.empty:
                continue
            for nq in sorted(sub["num_qubits"].dropna().unique()):
                grp = sub[sub["num_qubits"] == nq]
                if len(grp) < 2:
                    continue
                row = analyse_internal_group(grp)
                row["algorithm"] = algo
                row["provider"] = prov
                row["num_qubits"] = int(nq)
                results.append(row)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# (2) Provider comparison (IBM vs Rigetti) under chance-corrected success
# ---------------------------------------------------------------------------


def mannwhitney_test(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan
    try:
        res = stats.mannwhitneyu(x, y, alternative="two-sided")
        return res.statistic, res.pvalue
    except ValueError:
        return np.nan, np.nan


def run_provider_analysis(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for algo in ("GROVER", "QFT"):
        sub_algo = df[df["algorithm"] == algo]
        common_qubits = sorted(
            set(sub_algo.loc[sub_algo["provider"] == "IBM", "num_qubits"].dropna().unique())
            & set(sub_algo.loc[sub_algo["provider"] == "Rigetti", "num_qubits"].dropna().unique())
        )
        for nq in common_qubits:
            ibm = sub_algo[(sub_algo["provider"] == "IBM") & (sub_algo["num_qubits"] == nq)]
            if "optimization_level" in ibm.columns:
                opt = pd.to_numeric(ibm["optimization_level"], errors="coerce")
                ibm = ibm[opt.isna() | (opt == 3)]
            rig = sub_algo[(sub_algo["provider"] == "Rigetti") & (sub_algo["num_qubits"] == nq)]
            if len(ibm) < 3 or len(rig) < 3:
                continue

            row = {
                "algorithm": algo,
                "num_qubits": int(nq),
                "n_ibm": len(ibm),
                "n_rigetti": len(rig),
            }
            for m in PROFILE_METRICS:
                label = PROFILE_LABELS[m]
                med_ibm, _, _ = bootstrap_median_ci(ibm[m].values)
                med_rig, _, _ = bootstrap_median_ci(rig[m].values)
                u, p = mannwhitney_test(ibm[m].values, rig[m].values)
                delta = cliffs_delta(ibm[m].values, rig[m].values)
                row[f"{label}_median_ibm"] = med_ibm
                row[f"{label}_median_rigetti"] = med_rig
                row[f"{label}_U"] = u
                row[f"{label}_p"] = p
                row[f"{label}_delta_ibm_vs_rigetti"] = delta
            results.append(row)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows with full profile data\n")

    internal = run_internal_analysis(df)
    internal.to_csv(OUT_INTERNAL_CSV, index=False)
    print(f"Profile-internal comparison saved to {OUT_INTERNAL_CSV} ({len(internal)} groups)\n")

    for _, r in internal.iterrows():
        print(f"--- {r['algorithm']} {r['provider']} {int(r['num_qubits'])}q (n={int(r['n'])}) ---")
        for label in PROFILE_LABELS.values():
            med = r[f"{label}_median"]
            print(f"  {label}: median={med:.3f}")
        for comp in ["Chance-Corrected", "Coarse TVD-Sim", "Coarse HF"]:
            p, d = r[f"p_corr_vs_{comp}"], r[f"delta_vs_{comp}"]
            p_str = f"p={p:.4f}" if not np.isnan(p) else "p=n/a"
            d_str = f"delta={d:+.2f}" if not np.isnan(d) else "delta=n/a"
            print(f"  Success vs {comp}: {p_str}, {d_str}")
        print()

    provider = run_provider_analysis(df)
    provider.to_csv(OUT_PROVIDER_CSV, index=False)
    print(
        f"\nProvider (IBM vs Rigetti) comparison saved to {OUT_PROVIDER_CSV} ({len(provider)} groups)\n"
    )

    for _, r in provider.iterrows():
        print(
            f"--- {r['algorithm']} {int(r['num_qubits'])}q "
            f"(IBM n={int(r['n_ibm'])}, Rigetti n={int(r['n_rigetti'])}) ---"
        )
        for label in PROFILE_LABELS.values():
            med_ibm = r[f"{label}_median_ibm"]
            med_rig = r[f"{label}_median_rigetti"]
            p = r[f"{label}_p"]
            d = r[f"{label}_delta_ibm_vs_rigetti"]
            p_str = f"p={p:.4f}" if not np.isnan(p) else "p=n/a"
            print(
                f"  {label}: IBM median={med_ibm:.3f}, Rigetti median={med_rig:.3f}, "
                f"{p_str}, delta(IBM vs Rigetti)={d:+.2f}"
            )
        print()


if __name__ == "__main__":
    main()
