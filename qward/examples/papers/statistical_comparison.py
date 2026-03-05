"""
Statistical comparison of DSR, Hellinger fidelity, and TVD fidelity.

Produces:
  - LaTeX table code (printed to stdout) for Section VII.D of the paper
  - Summary CSV at qward/examples/papers/statistical_comparison_results.csv

Tests performed per (algorithm, provider, qubit-count) group:
  1. Wilcoxon signed-rank (paired): DSR vs HF, DSR vs TVD
     - Bonferroni-corrected for 2 comparisons per group
  2. Cliff's delta (non-parametric effect size)
  3. Bootstrap 95 % CI for the median (10 000 resamples; exact values if n < 5)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

CSV_PATH = Path(__file__).parent / "DSR_result.csv"
OUT_CSV = Path(__file__).parent / "statistical_comparison_results.csv"

METRICS = ["dsr_michelson", "hellinger_fidelity", "tvd_fidelity"]
METRIC_LABELS = {"dsr_michelson": "DSR", "hellinger_fidelity": "HF", "tvd_fidelity": "TVD-F"}

N_BOOT = 10_000
SEED = 42
ALPHA = 0.05
N_COMPARISONS = 2  # DSR vs HF, DSR vs TVD => Bonferroni factor


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def cliffs_delta(x, y):
    """Cliff's delta between two arrays."""
    x, y = np.asarray(x), np.asarray(y)
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return np.nan
    more = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    delta = (more - less) / (n_x * n_y)
    return delta


def bootstrap_median_ci(arr, n_boot=N_BOOT, alpha=ALPHA, seed=SEED):
    """Bootstrap 95 % CI for the median. Returns (median, lo, hi)."""
    arr = np.asarray(arr)
    n = len(arr)
    med = np.median(arr)
    if n < 5:
        return med, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot_medians = np.array(
        [np.median(rng.choice(arr, size=n, replace=True)) for _ in range(n_boot)]
    )
    lo = np.percentile(boot_medians, 100 * alpha / 2)
    hi = np.percentile(boot_medians, 100 * (1 - alpha / 2))
    return med, lo, hi


def wilcoxon_test(x, y):
    """Wilcoxon signed-rank test. Returns (W, p_raw)."""
    x, y = np.asarray(x), np.asarray(y)
    diff = x - y
    diff = diff[diff != 0]
    if len(diff) < 6:
        return np.nan, np.nan
    try:
        res = stats.wilcoxon(diff, alternative="two-sided")
        return res.statistic, res.pvalue
    except ValueError:
        return np.nan, np.nan


# ---------------------------------------------------------------------------
# Data loading and filtering
# ---------------------------------------------------------------------------


def load_data():
    df = pd.read_csv(CSV_PATH)
    # Keep only QPU results with the three metrics available
    df = df.dropna(subset=METRICS)
    return df


def filter_group(df, algorithm, provider):
    """Filter by algorithm + provider, applying opt-level rules."""
    sub = df[df["algorithm"] == algorithm].copy()
    if provider == "IBM":
        sub = sub[sub["noise_model"] == "IBM-QPU"]
        sub = sub[sub["optimization_level"].isin([2, 3])]
    elif provider == "AWS":
        sub = sub[sub["noise_model"] == "AWS-QPU"]
    return sub


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyse_group(grp_df):
    """Run all tests for a single (algorithm, provider, n_qubits) group."""
    n = len(grp_df)
    row = {"n": n}

    for m in METRICS:
        label = METRIC_LABELS[m]
        vals = grp_df[m].values
        med, lo, hi = bootstrap_median_ci(vals)
        row[f"{label}_median"] = med
        row[f"{label}_ci_lo"] = lo
        row[f"{label}_ci_hi"] = hi

    # Paired tests: DSR vs HF, DSR vs TVD
    dsr = grp_df["dsr_michelson"].values
    for other, label in [("hellinger_fidelity", "HF"), ("tvd_fidelity", "TVD-F")]:
        other_vals = grp_df[other].values
        w, p_raw = wilcoxon_test(dsr, other_vals)
        p_corr = min(p_raw * N_COMPARISONS, 1.0) if not np.isnan(p_raw) else np.nan
        delta = cliffs_delta(dsr, other_vals)
        row[f"W_vs_{label}"] = w
        row[f"p_raw_vs_{label}"] = p_raw
        row[f"p_corr_vs_{label}"] = p_corr
        row[f"delta_vs_{label}"] = delta
    return row


def run_analysis():
    df = load_data()
    results = []

    combos = [
        ("GROVER", "IBM"),
        ("GROVER", "AWS"),
        ("QFT", "IBM"),
        ("QFT", "AWS"),
    ]

    for algo, prov in combos:
        sub = filter_group(df, algo, prov)
        if sub.empty:
            continue
        for nq in sorted(sub["num_qubits"].unique()):
            grp = sub[sub["num_qubits"] == nq]
            row = analyse_group(grp)
            row["algorithm"] = algo
            row["provider"] = prov
            row["num_qubits"] = int(nq)
            results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# LaTeX formatting
# ---------------------------------------------------------------------------


def fmt_med_ci(med, lo, hi):
    """Format median [CI] for LaTeX."""
    if np.isnan(lo):
        return f"${med:.2f}$"
    return f"${med:.2f}$\\,[${lo:.2f}$,\\,${hi:.2f}$]"


def fmt_p(p):
    if np.isnan(p):
        return "---"
    if p < 0.0005:
        return "$<$0.001"
    return f"${p:.3f}$"


def fmt_delta(delta):
    if np.isnan(delta):
        return "---"
    return f"${delta:+.2f}$"


def fmt_w(w):
    if np.isnan(w):
        return "---"
    return f"${w:.0f}$"


def generate_latex_table(res_df, algo, caption_algo):
    """Generate one LaTeX table for a given algorithm."""
    sub = res_df[res_df["algorithm"] == algo]
    if sub.empty:
        return ""

    providers = sub["provider"].unique()
    lines = []

    lines.append("\\begin{table*}[ht!]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Statistical comparison of DSR, Hellinger fidelity (HF), "
        "and TVD fidelity (TVD-F) for " + caption_algo + ". Medians with bootstrap 95\\,\\% CIs; "
        "Wilcoxon signed-rank $W$ and Bonferroni-corrected $p$; "
        "Cliff's $\\delta$.}"
    )
    lines.append("\\label{tab:" + algo.lower() + "_stat_comparison}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{cl r ccc cc cc}")
    lines.append("\\hline")
    lines.append(
        "& & & \\multicolumn{3}{c}{\\textbf{Median [95\\,\\% CI]}} "
        "& \\multicolumn{2}{c}{\\textbf{DSR vs.\\ HF}} "
        "& \\multicolumn{2}{c}{\\textbf{DSR vs.\\ TVD-F}} \\\\"
    )
    lines.append(
        "\\textbf{Prov.} & \\textbf{$n_q$} & \\textbf{$n$} "
        "& \\textbf{DSR} & \\textbf{HF} & \\textbf{TVD-F} "
        "& $p_{\\text{corr}}$ & $\\delta$ "
        "& $p_{\\text{corr}}$ & $\\delta$ \\\\"
    )
    lines.append("\\hline")

    for prov in ["IBM", "AWS"]:
        prov_rows = sub[sub["provider"] == prov].sort_values("num_qubits")
        if prov_rows.empty:
            continue
        first = True
        for _, r in prov_rows.iterrows():
            prov_label = prov if first else ""
            first = False
            nq = int(r["num_qubits"])
            n = int(r["n"])
            dsr_str = fmt_med_ci(r["DSR_median"], r["DSR_ci_lo"], r["DSR_ci_hi"])
            hf_str = fmt_med_ci(r["HF_median"], r["HF_ci_lo"], r["HF_ci_hi"])
            tvd_str = fmt_med_ci(r["TVD-F_median"], r["TVD-F_ci_lo"], r["TVD-F_ci_hi"])
            p_hf = fmt_p(r["p_corr_vs_HF"])
            d_hf = fmt_delta(r["delta_vs_HF"])
            p_tvd = fmt_p(r["p_corr_vs_TVD-F"])
            d_tvd = fmt_delta(r["delta_vs_TVD-F"])
            lines.append(
                f"{prov_label} & {nq} & {n} "
                f"& {dsr_str} & {hf_str} & {tvd_str} "
                f"& {p_hf} & {d_hf} "
                f"& {p_tvd} & {d_tvd} \\\\"
            )
        lines.append("\\hline")

    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    res = run_analysis()
    res.to_csv(OUT_CSV, index=False)
    print(f"Results saved to {OUT_CSV}\n")

    # Print summary to console
    for _, r in res.iterrows():
        algo, prov, nq, n = r["algorithm"], r["provider"], int(r["num_qubits"]), int(r["n"])
        print(f"--- {algo} {prov} {nq}q (n={n}) ---")
        for m in ["DSR", "HF", "TVD-F"]:
            med = r[f"{m}_median"]
            lo, hi = r.get(f"{m}_ci_lo", np.nan), r.get(f"{m}_ci_hi", np.nan)
            ci = f" [{lo:.2f}, {hi:.2f}]" if not np.isnan(lo) else ""
            print(f"  {m}: median={med:.2f}{ci}")
        for comp in ["HF", "TVD-F"]:
            p = r[f"p_corr_vs_{comp}"]
            d = r[f"delta_vs_{comp}"]
            p_str = f"p={p:.4f}" if not np.isnan(p) else "p=n/a"
            d_str = f"δ={d:+.2f}" if not np.isnan(d) else "δ=n/a"
            print(f"  DSR vs {comp}: {p_str}, {d_str}")
        print()

    # Print LaTeX tables
    print("=" * 80)
    print("LATEX TABLES")
    print("=" * 80)
    print()
    print(generate_latex_table(res, "GROVER", "the Grover algorithm"))
    print()
    print(generate_latex_table(res, "QFT", "the Quantum Fourier Transform"))


if __name__ == "__main__":
    main()
