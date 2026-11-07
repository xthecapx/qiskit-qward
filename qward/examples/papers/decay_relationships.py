"""Descriptive comparisons of success decay on jointly observed circuit resource levels."""

import json
from itertools import combinations

import numpy as np
import pandas as pd

from qward.examples.papers.decay_groups import compare_groups

EXPORTED_TARGETS = {
    "f.fidelity.hellinger_fidelity": "hellinger_fidelity",
    "f.fidelity.tvd_fidelity": "tvd_similarity",
    "f.fidelity.dsr": "dsr",
}
TARGETS = {
    "success": ("Success rate", "Fraction of accepted shots"),
    "normalized": ("Adjusted success", "Fraction of the chance-to-ideal success gap recovered"),
    "hellinger_fidelity": (
        "Hellinger fidelity",
        "Squared overlap of measured and ideal outcome distributions",
    ),
    "tvd_similarity": (
        "TVD similarity",
        "One minus total variation distance from the ideal distribution",
    ),
    "dsr": (
        "Differential success rate",
        "Clipped Michelson contrast of mean expected probability against the strongest unexpected outcome",
    ),
}
WIDTH = "a.num_qubits"
TWO_QUBIT = "a.ComplexityMetrics.gate_based_metrics.two_qubit_count"
# One resource chosen per question for the review figures. All requested metrics remain in the CSVs.
REVIEW = [
    (
        "QFT",
        "PHASE-ESTIMATION",
        TWO_QUBIT,
        "Do QFT and phase estimation share a trend as logical two-qubit gate count grows?",
    ),
    (
        "VTP",
        "RANDOM-VOLUMETRIC",
        WIDTH,
        "Does payload preparation and inverse validation around teleportation track a random mirror circuit at equal total width?",
    ),
    (
        "GROVER",
        "BERNSTEIN-VAZIRANI",
        WIDTH,
        "Do the two oracle workloads respond similarly to increasing width?",
    ),
    ("VTP", "GHZ", WIDTH, "Does vTP track GHZ state preparation as width increases?"),
    ("QFT", "SHOR", WIDTH, "Does QFT track Shor order finding, which uses an inverse QFT?"),
    (
        "PHASE-ESTIMATION",
        "SHOR",
        WIDTH,
        "Does phase estimation track Shor order finding at shared widths?",
    ),
]


def profile_rho(pair):
    """Spearman correlation is undefined when either profile is constant."""
    ranks = pair.rank()
    return ranks.iloc[:, 0].corr(ranks.iloc[:, 1]) if ranks.nunique().min() > 1 else np.nan


def paired_decay(levels):
    """Describe two observed success profiles indexed by the same numeric resource levels."""
    values = levels.index.to_numpy(dtype=float)
    rates = levels.to_numpy(dtype=float)
    slopes = np.polyfit(values - values.mean(), rates, 1)[0]
    changes = np.diff(rates, axis=0) / np.diff(values)[:, None]
    signs = np.sign(changes)
    # Rank correlation on the observations avoids the built-in correlation of two fitted lines.
    return {
        "levels": len(values),
        "metric_min": values[0],
        "metric_max": values[-1],
        "profile_spearman": profile_rho(levels),
        "step_rate_spearman": profile_rho(pd.DataFrame(changes)) if len(changes) >= 3 else np.nan,
        "slope_1": slopes[0],
        "slope_2": slopes[1],
        "slope_difference_2_minus_1": slopes[1] - slopes[0],
        "adjacent_intervals": len(changes),
        "both_decline": int(np.all(signs < 0, axis=1).sum()),
        "both_rise": int(np.all(signs > 0, axis=1).sum()),
        "opposite_direction": int((signs.prod(axis=1) < 0).sum()),
        "either_flat": int((signs.prod(axis=1) == 0).sum()),
    }


def analyze_decay(data, metrics, families, output, plots=True, group_tolerance=5.0):
    """Compare equally weighted configuration means at exact shared levels, within periods."""
    output.mkdir(parents=True, exist_ok=True)
    pairs = [(family, *pair) for family, pair in families.items()]
    pairs += [("composition", algorithm, "SHOR") for algorithm in families["fourier"]]
    workloads = [algorithm for pair in families.values() for algorithm in pair] + ["SHOR"]
    pairs += [
        ("mirror_protocol" if algorithm == "VTP" else "control", algorithm, "RANDOM-VOLUMETRIC")
        for algorithm in workloads
    ]
    annotations = {frozenset((a, b)): (role, a, b) for role, a, b in pairs}
    pairs = [
        annotations.get(frozenset((a, b)), ("other", a, b))
        for a, b in combinations(sorted(data.algorithm.unique()), 2)
    ]
    frame = data.copy()
    gap = frame["ideal_success"] - frame["chance"]
    frame["normalized"] = (frame["success"] - frame["chance"]) / gap.where(gap > 1e-8)
    scores, matched, coverage, controls = [], [], [], []
    for metric in metrics:
        for period, jobs in frame.groupby("period"):
            finite = jobs[np.isfinite(jobs[metric])]
            for outcome in (target for target in TARGETS if target in frame):
                # Canonical summation order also makes floating-point means invariant
                # to the order of jobs in an export.
                valid = finite.dropna(subset=[outcome]).sort_values(
                    ["algorithm", metric, "config", outcome], kind="stable"
                )
                configs = valid.groupby(["algorithm", metric, "config"])[outcome].mean()
                by_level = configs.groupby(level=["algorithm", metric])
                means = by_level.mean().unstack("algorithm")
                counts = by_level.count().unstack("algorithm")
                for role, first, second in pairs:
                    key = dict(
                        role=role,
                        algorithm_1=first,
                        algorithm_2=second,
                        metric=metric,
                        period=period,
                        outcome=outcome,
                    )
                    pair = means.reindex(columns=[first, second]).dropna().sort_index()
                    coverage.append({**key, "shared_levels": len(pair), "eligible": len(pair) >= 3})
                    if len(pair) < 3:
                        continue
                    fit = paired_decay(pair)
                    scores.append({**key, **fit})
                    if role in {*families, "composition"}:
                        rv = "RANDOM-VOLUMETRIC"
                        triple = means.reindex(columns=[first, second, rv]).dropna()
                        if len(triple) >= 3:
                            controls.append(
                                {
                                    **key,
                                    "levels": len(triple),
                                    "shared_values": ",".join(str(v) for v in triple.index),
                                    "family_rho": profile_rho(triple[[first, second]]),
                                    "first_vs_rv_rho": profile_rho(triple[[first, rv]]),
                                    "second_vs_rv_rho": profile_rho(triple[[second, rv]]),
                                }
                            )
                    center = pair.index.to_numpy(dtype=float).mean()
                    means_at_center = pair.mean()
                    for value, row in pair.iterrows():
                        matched.append(
                            {
                                **key,
                                "metric_value": value,
                                "mean_1": row[first],
                                "mean_2": row[second],
                                "fit_1": means_at_center[first] + fit["slope_1"] * (value - center),
                                "fit_2": means_at_center[second]
                                + fit["slope_2"] * (value - center),
                                "configs_1": int(counts.at[value, first]),
                                "configs_2": int(counts.at[value, second]),
                            }
                        )
    keys = ["role", "algorithm_1", "algorithm_2", "metric", "period", "outcome"]
    results = pd.DataFrame(
        scores,
        columns=[
            *keys,
            "levels",
            "metric_min",
            "metric_max",
            "profile_spearman",
            "step_rate_spearman",
            "slope_1",
            "slope_2",
            "slope_difference_2_minus_1",
            "adjacent_intervals",
            "both_decline",
            "both_rise",
            "opposite_direction",
            "either_flat",
        ],
    )
    results.to_csv(output / "decay_comparisons.csv", index=False)
    observations = pd.DataFrame(
        matched,
        columns=[
            *keys,
            "metric_value",
            "mean_1",
            "mean_2",
            "configs_1",
            "configs_2",
            "fit_1",
            "fit_2",
        ],
    )
    observations.to_csv(output / "matched_levels.csv", index=False)
    coverage = pd.DataFrame(coverage, columns=[*keys, "shared_levels", "eligible"])
    coverage.to_csv(output / "overlap.csv", index=False)
    compare_groups(
        results,
        coverage,
        sorted(data.algorithm.unique()),
        output,
        plots,
        group_tolerance,
        data.backend.iloc[0],
    )
    pd.DataFrame(
        controls,
        columns=[
            *keys,
            "levels",
            "shared_values",
            "family_rho",
            "first_vs_rv_rho",
            "second_vs_rv_rho",
        ],
    ).to_csv(output / "matched_controls.csv", index=False)
    figure = plot_widths(observations, output, families) if plots and matched else None
    write_trend_review(frame, results, observations, output, plots)
    write_decay_report(output, data, results, figure)
    return results


def plot_widths(observations, output, families):
    """Show empirical means for each eligible period and exact shared width."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    width = observations[
        observations.metric.eq("a.num_qubits") & observations.outcome.eq("success")
    ]
    pairs = [*families.values(), ("PHASE-ESTIMATION", "SHOR")]
    if width.empty:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=True, layout="constrained")
    labels = {
        "PHASE-ESTIMATION": "Phase estimation",
        "BERNSTEIN-VAZIRANI": "Bernstein–Vazirani",
        "GROVER": "Grover",
        "VTP": "vTP",
        "SHOR": "Shor",
    }
    for ax, (first, second) in zip(axes.flat, pairs):
        rows = width[width.algorithm_1.eq(first) & width.algorithm_2.eq(second)]
        for index, (_, group) in enumerate(rows.groupby("period")):
            for column, algorithm, color in (
                ("mean_1", first, "#2463a0"),
                ("mean_2", second, "#a14818"),
            ):
                ax.plot(
                    group.metric_value,
                    100 * group[column],
                    ".-",
                    color=color,
                    alpha=0.75,
                    label=labels.get(algorithm, algorithm) if index == 0 else None,
                )
        if rows.empty:
            ax.text(0.5, 0.5, "Insufficient shared widths", ha="center", transform=ax.transAxes)
        else:
            ax.legend(frameon=False, fontsize=9)
        ax.set_title(f"{labels.get(first, first)} / {labels.get(second, second)}", fontsize=11)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(-2, 102)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Observed success at shared circuit widths")
    fig.supxlabel("Total logical qubits")
    fig.supylabel("Mean success (%)")
    filename = "shared_widths.png"
    fig.savefig(output / filename, dpi=150)
    plt.close(fig)
    return filename


def write_decay_report(output, data, scores, figure):
    """Make the Fez baseline reproducible on another chip without a predictive threshold."""
    lines = [
        "# Do related circuits lose success in the same way?",
        "",
        f"Backend {', '.join(data['backend'].unique())}. The cohort contains {len(data):,} jobs "
        f"and {data['cluster'].nunique()} configurations in {data['period'].nunique()} "
        "collection periods.",
        "",
        "[Linear fits, target definitions, and comparison table](trend_review.md)",
        "",
        "[All algorithm pairs and exploratory decay groups](groups/report.md)",
        "",
        "Each comparison uses exact resource values observed for both algorithms in the same "
        "campaign and window. At each resource value we average repetitions within each "
        "configuration, then give each configuration equal weight. At least three shared values "
        "are required. Each value has equal weight in the descriptive straight-line slope.",
        "",
        "Spearman rho measures whether the observed success means rank the shared resource "
        "values similarly. The slopes show the direction and average size of change per unit. "
        "Step-rate Spearman rho ranks the observed changes per resource unit between adjacent "
        "levels and compares those ranks when at least three intervals exist. "
        "The adjacent-interval counts show how often both workloads actually decline or rise "
        "over the same step. A high rho can accompany very different slopes. Correlating two "
        "fitted nonconstant straight lines on the same grid would always give +1 or −1.",
        "",
        "Error is one minus success, so the raw error slopes have the opposite signs and the "
        "same magnitudes. Normalized comparisons use (success − chance) / (ideal success − "
        "chance) wherever the denominator exceeds 1e-8.",
        "",
        "## Width comparisons",
        "",
        "Ranges below span eligible collection periods and describe variation across periods. "
        "Repeated configurations recur across periods. Width is total logical qubits and may "
        "represent different computational resources across workloads.",
        "",
    ]
    if figure:
        lines += [
            f"![Observed success at shared widths]({figure})",
            "",
            "Each line connects observed means within one collection period.",
            "",
        ]
    if not scores.empty:
        width = scores[scores.metric.eq("a.num_qubits") & scores.outcome.eq("success")]
        if not width.empty:
            lines += [
                "| Pair | Periods | Shared widths per period | Spearman ρ | "
                "Success change A (pp/qubit) | Success change B (pp/qubit) |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
            for (first, second), pair in width.groupby(["algorithm_1", "algorithm_2"], sort=False):

                def span(values, digits):
                    values = values.dropna()
                    if values.empty:
                        return "undefined"
                    bounds = [f"{v:.{digits}f}" for v in (values.min(), values.max())]
                    return bounds[0] if bounds[0] == bounds[1] else " to ".join(bounds)

                lines.append(
                    f"| {first} / {second} | {len(pair)} | {span(pair.levels, 0)} | "
                    f"{span(pair.profile_spearman, 2)} | {span(100 * pair.slope_1, 2)} | "
                    f"{span(100 * pair.slope_2, 2)} |"
                )
    if not scores.empty:
        gates = scores[
            scores.role.eq("fourier")
            & scores.outcome.eq("success")
            & scores.metric.eq("a.ComplexityMetrics.gate_based_metrics.two_qubit_count")
        ]
        if not gates.empty:
            lines += [
                "",
                "## QFT and phase estimation by logical two-qubit gate count",
                "",
                "| Period | Shared levels | Profile ρ | QFT loss (pp/gate) | "
                "PE loss (pp/gate) | Correlation of local rates |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
            for row in gates.itertuples():
                local = (
                    f"{row.step_rate_spearman:.2f}"
                    if np.isfinite(row.step_rate_spearman)
                    else "undefined"
                )
                lines.append(
                    f"| {row.period} | {row.levels} | {row.profile_spearman:.2f} | "
                    f"{-100 * row.slope_1:.2f} | {-100 * row.slope_2:.2f} | {local} |"
                )
    lines += [
        "",
        "## Interpretation",
        "",
        "These are descriptive comparisons. With three shared sizes, even perfect rank "
        "agreement uses only three points. Changing configuration mix across widths, "
        "different success criteria, and shared hardware noise can affect trends. "
        "Random-volumetric profiles provide a generic comparator. Interpreting differences "
        "between pairs requires checking the common levels used by each pair. "
        "[Matched controls](matched_controls.csv) compare both family members with the mirror "
        "circuit at the same levels in the same period.",
        "",
        "The comparison does not assign a predictive R² threshold or infer slope equivalence. "
        "The existing family interaction models provide configuration-clustered uncertainty "
        "for slope differences across their fitted ranges, which can extend beyond the "
        "exact shared levels used here.",
        "",
        "See [all metric comparisons](decay_comparisons.csv), [observed level means]"
        "(matched_levels.csv), and [coverage including unsupported comparisons](overlap.csv).",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_linear_trend(chart, output):
    """Draw one separate fit per workload and eligible collection period."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    periods = chart["periods"]
    cols = min(3, len(periods))
    rows = (len(periods) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows,
        cols,
        squeeze=False,
        figsize=(3.55 * cols + 0.6, 2.8 * rows + 1.1),
        sharey=True,
        layout="constrained",
    )
    colors = ("#216291", "#b44c25")
    for ax, period in zip(axes.flat, periods):
        points = pd.DataFrame(period["points"])
        for i, color in enumerate(colors, 1):
            ax.scatter(points.x, 100 * points[f"mean_{i}"], color=color, s=28)
            ax.plot(points.x, 100 * points[f"fit_{i}"], color=color, linewidth=1.8)
        ax.set_title(period["label"], fontsize=10)
        ax.grid(axis="y", alpha=0.2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        if len(points) <= 8:
            ax.set_xticks(points.x)
        ax.set_xlabel(chart["x_label"], fontsize=9)
    for ax in list(axes.flat)[len(periods) :]:
        ax.set_visible(False)
    names = [display_name(chart[f"algorithm_{i}"]) for i in (1, 2)]
    fig.legend(
        [Line2D([], [], marker="o", color=c) for c in colors],
        names,
        loc="outside lower center",
        ncols=2,
        frameon=False,
    )
    fig.suptitle(f"{names[0]} and {names[1]}  ·  {TARGETS[chart['outcome']][0]}")
    if chart["outcome"] != "normalized":
        bottom, top = axes.flat[0].get_ylim()
        axes.flat[0].set_ylim(min(0, bottom), max(100, top))
    fig.supylabel(
        "Success rate (%)" if chart["outcome"] == "success" else "Target score × 100", fontsize=10
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)


def display_name(algorithm):
    return {
        "PHASE-ESTIMATION": "Phase estimation",
        "RANDOM-VOLUMETRIC": "Random volumetric",
        "VTP": "vTP",
        "BERNSTEIN-VAZIRANI": "BV",
        "GROVER": "Grover",
        "SHOR": "Shor",
    }.get(algorithm, algorithm)


def value_range(series):
    values = series.dropna()
    if values.empty:
        return "undefined"
    low, high = (f"{n:.2f}" for n in (values.min(), values.max()))
    return low if low == high else f"{low} to {high}"


def write_trend_review(data, scores, observations, output, plots):
    """Export the table, figure series and plotted coordinates from the same regressions."""
    period_names = {p: f"P{i}" for i, p in enumerate(sorted(data.period.unique()), 1)}
    charts, review = [], []
    lines = [
        "# Comparing linear decay trends",
        "",
        "Dots show the observed mean per shared resource level, with equal weight for each "
        "configuration. Lines are separate least-squares fits per workload and collection period. "
        "Each observed level has equal weight. The lines stop at the observed endpoints.",
        "",
        "Profile ρ ranks the observed level means. Local ρ ranks the changes per unit between "
        "successive levels (at least three intervals). Slopes are target points on a ×100 scale "
        "per resource unit. For success these are percentage points. Ranges span eligible periods. "
        "Compare outcomes over the same configurations, levels and periods. Missing outcomes can "
        "change the support in future exports. All comparisons are exploratory.",
        "",
        "| Target | Measurement | Available jobs |",
        "| --- | --- | ---: |",
    ]
    for outcome, (name, meaning) in TARGETS.items():
        available = int(data[outcome].count()) if outcome in data else 0
        lines.append(f"| {name} | {meaning} | {available} / {len(data)} |")
    lines += [
        "",
        "For an ideal distribution concentrated on a single accepted bitstring, "
        "Hellinger fidelity and TVD similarity both equal success. Agreement across these "
        "scores can therefore reflect the same observation.",
        "",
        "The Fez vTP configurations use specified one-qubit payload gates with inverse "
        "validation and two teleportation variants. RV uses random two-qubit layers and "
        "their inverses. Equal total width gives vTP three protocol qubits in addition to "
        "its payload. This comparison measures the two recorded workloads, including "
        "their different gates and success criteria.",
        "",
        *[f"- {label} = {period}" for period, label in period_names.items()],
        "",
    ]
    figures = output / "linear_trends"
    if plots:
        figures.mkdir(exist_ok=True)
    for first, second, metric, question in REVIEW:
        baseline = {}
        x_label = "Logical two-qubit gates" if metric == TWO_QUBIT else "Total logical qubits"
        lines += [
            f"## {display_name(first)} and {display_name(second)}",
            "",
            question,
            "",
            f"Resource measured: {x_label.lower()}.",
            "",
            "| Target | Periods × levels¹ | Slope A | Slope B | Profile ρ | Local ρ | Conclusion | Figure |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for outcome, (target_name, _) in TARGETS.items():
            key = dict(algorithm_1=first, algorithm_2=second, metric=metric, outcome=outcome)
            fitted = scores
            points = observations
            for column, value in key.items():
                fitted = fitted[fitted[column].eq(value)]
                points = points[points[column].eq(value)]
            chart = {**key, "x_label": x_label, "question": question, "periods": []}
            rho_deltas = []
            for s in fitted.itertuples():
                pt = points[points.period.eq(s.period)].sort_values("metric_value").copy()
                pt = pt.rename(columns={"metric_value": "x"})
                if outcome == "success":
                    baseline[s.period] = (pt.x.tolist(), s.profile_spearman)
                else:
                    levels, rho = baseline.get(s.period, ([], np.nan))
                    jobs = data[
                        data.period.eq(s.period)
                        & data.algorithm.isin((first, second))
                        & data[metric].isin(pt.x)
                    ]
                    if (
                        pt.x.tolist() == levels
                        and jobs[outcome].notna().all()
                        and np.isfinite([rho, s.profile_spearman]).all()
                    ):
                        rho_deltas.append(s.profile_spearman - rho)
                chart["periods"].append(
                    {
                        "period": s.period,
                        "label": period_names[s.period],
                        "slope_1": s.slope_1,
                        "slope_2": s.slope_2,
                        "points": json.loads(
                            pt[
                                [
                                    "x",
                                    "mean_1",
                                    "mean_2",
                                    "configs_1",
                                    "configs_2",
                                    "fit_1",
                                    "fit_2",
                                ]
                            ].to_json(orient="records", double_precision=14)
                        ),
                    }
                )
            filename = f"{first.lower()}_{second.lower()}_{outcome}.png"
            link = "—"
            if chart["periods"] and plots:
                plot_linear_trend(chart, figures / filename)
                link = f"[plot](linear_trends/{filename})"
            charts.append(chart)
            periods = len(fitted)
            descending = (fitted.slope_1.lt(-1e-12) & fitted.slope_2.lt(-1e-12)).sum()
            conclusion = (
                f"Both decline in {descending}/{periods} periods"
                if periods
                else "Insufficient shared levels or target coverage"
            )
            if rho_deltas:
                stronger = sum(d > 1e-10 for d in rho_deltas)
                tied = sum(abs(d) <= 1e-10 for d in rho_deltas)
                conclusion += f". Profile ρ vs success: {stronger} higher, {tied} tied, {len(rho_deltas)-stronger-tied} lower"
            row = {
                **key,
                "question": question,
                "periods": periods,
                "levels_per_period": ", ".join(str(int(n)) for n in sorted(fitted.levels.unique())),
                "slope_A_x100": value_range(100 * fitted.slope_1),
                "slope_B_x100": value_range(100 * fitted.slope_2),
                "profile_rho": value_range(fitted.profile_spearman),
                "local_rho": value_range(fitted.step_rate_spearman),
                "periods_matched_to_success": len(rho_deltas),
                "conclusion": conclusion,
            }
            review.append(row)
            lines.append(
                f"| {target_name} | {periods} × {row['levels_per_period'] or '—'} | "
                f"{row['slope_A_x100']} | {row['slope_B_x100']} | {row['profile_rho']} | "
                f"{row['local_rho']} | {conclusion} | {link} |"
            )
        lines.append("")
    lines += [
        "¹ Number of eligible periods and distinct shared-level counts per period.",
        "",
        "Machine-readable outputs: [review table](trend_review.csv) and "
        "[observations and fitted lines](linear_trends.json).",
        "",
    ]
    pd.DataFrame(review).to_csv(output / "trend_review.csv", index=False)
    (output / "trend_review.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "linear_trends.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": data.backend.iloc[0],
                "targets": TARGETS,
                "scale": "Original target units for observed and fitted values and slopes; figures multiply by 100.",
                "charts": charts,
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
