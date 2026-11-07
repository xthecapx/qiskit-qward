"""Exploratory grouping by changes in fitted outcomes over observed shared resource ranges."""

from itertools import combinations
import json

import numpy as np
import pandas as pd

LABELS = {
    "PHASE-ESTIMATION": "PE",
    "RANDOM-VOLUMETRIC": "RV",
    "BERNSTEIN-VAZIRANI": "BV",
    "GROVER": "Grover",
    "VTP": "vTP",
    "SHOR": "Shor",
}
MIN_PERIODS = 2


def group_by_decay(algorithms, distances, tolerance):
    """Complete linkage with missing pairwise comparisons left unsupported."""
    groups = [(algorithm,) for algorithm in sorted(algorithms)]
    while True:
        candidates = []
        for left, right in combinations(groups, 2):
            diameter = max(
                distances.get(tuple(sorted((a, b))), np.inf) for a in left for b in right
            )
            if diameter <= tolerance:
                candidates.append((diameter, left, right))
        if not candidates:
            return sorted(groups)
        _, left, right = min(candidates)
        groups.remove(left)
        groups.remove(right)
        groups.append(tuple(sorted(left + right)))
        groups.sort()


def compare_groups(scores, coverage, algorithms, output, plots=True, tolerance=5.0, backend=None):
    """Group independently for each resource/outcome, then compare with expected relationships."""
    folder = output / "groups"
    folder.mkdir(exist_ok=True)
    panel = ["metric", "outcome"]
    pair = [*panel, "algorithm_1", "algorithm_2"]
    catalog = coverage[[*pair, "role"]].drop_duplicates()
    scores = scores.copy()
    scores["gap_points"] = (
        100 * scores.slope_difference_2_minus_1.abs() * (scores.metric_max - scores.metric_min)
    )
    stats = scores.groupby(pair).agg(
        periods=("period", "size"),
        gap_points=("gap_points", "median"),
        min_gap=("gap_points", "min"),
        max_gap=("gap_points", "max"),
        min_shared_levels=("levels", "min"),
        median_profile_rho=("profile_spearman", "median"),
    )
    catalog = catalog.merge(stats, on=pair, how="left")
    catalog["periods"] = catalog.periods.fillna(0).astype(int)
    catalog["supported"] = catalog.periods.ge(MIN_PERIODS)
    catalog["within_tolerance"] = (
        catalog.gap_points.le(tolerance).where(catalog.supported).astype("boolean")
    )
    catalog["grouped_together"] = pd.Series(pd.NA, index=catalog.index, dtype="boolean")
    memberships = []
    for (metric, outcome), subset in catalog.groupby(panel):
        for minimum in (1, MIN_PERIODS):
            supported = subset[subset.periods.ge(minimum)]
            distances = {
                tuple(sorted((r.algorithm_1, r.algorithm_2))): r.gap_points
                for r in supported.itertuples()
            }
            assessed = {name for edge in distances for name in edge}
            for cut in (tolerance / 2, tolerance, 2 * tolerance):
                groups = group_by_decay(assessed, distances, cut)
                membership = {a: group for group in groups for a in group}
                for algorithm in algorithms:
                    group = membership.get(algorithm, ())
                    memberships.append(
                        dict(
                            metric=metric,
                            outcome=outcome,
                            tolerance_points=cut,
                            minimum_periods=minimum,
                            algorithm=algorithm,
                            group=" / ".join(group),
                            status=(
                                "insufficient overlap"
                                if not group
                                else ("singleton" if len(group) == 1 else "grouped")
                            ),
                        )
                    )
                if cut == tolerance and minimum == MIN_PERIODS:
                    for idx, r in supported.iterrows():
                        catalog.at[idx, "grouped_together"] = (
                            membership[r.algorithm_1] == membership[r.algorithm_2]
                        )
    members = pd.DataFrame(
        memberships,
        columns=[
            "metric",
            "outcome",
            "tolerance_points",
            "minimum_periods",
            "algorithm",
            "group",
            "status",
        ],
    )
    pair_summary = (
        catalog.groupby(["algorithm_1", "algorithm_2", "role", "outcome"])
        .agg(
            resources_attempted=("supported", "size"),
            resources_supported=("supported", "sum"),
            resources_within_tolerance=("within_tolerance", "sum"),
            resources_grouped=("grouped_together", "sum"),
        )
        .reset_index()
    )
    catalog.to_csv(folder / "pair_comparisons.csv", index=False)
    members.to_csv(folder / "memberships.csv", index=False)
    pair_summary.to_csv(folder / "pair_recurrence.csv", index=False)
    payload = {
        "schema_version": 1,
        "backend": backend,
        "algorithms": list(algorithms),
        "tolerance_points": tolerance,
        "minimum_periods": MIN_PERIODS,
        "distance": "median of 100 × absolute slope difference × shared resource span, across eligible periods",
        "comparisons": json.loads(catalog.to_json(orient="records")),
        "memberships": json.loads(members.to_json(orient="records")),
    }
    (folder / "decay_groups.json").write_text(
        json.dumps(payload, allow_nan=False), encoding="utf-8"
    )
    figure = plot_gaps(catalog, algorithms, folder) if plots and not catalog.empty else None
    write_group_report(folder, catalog, members, algorithms, tolerance, figure)
    return catalog, members


def short_group(group):
    return " + ".join(LABELS.get(name, name) for name in group.split(" / "))


def write_group_report(folder, catalog, members, algorithms, tolerance, figure):
    central = members[
        members.tolerance_points.eq(tolerance) & members.minimum_periods.eq(MIN_PERIODS)
    ]
    lines = [
        "# Which algorithms have similar decay?",
        "",
        f"Compared all {len(algorithms)*(len(algorithms)-1)//2} pairs of observed algorithms. "
        "Each resource metric and outcome has its own grouping. Family labels enter only after clustering.",
        "",
        "In each campaign/window, fit each algorithm’s line on at least three identical observed "
        "resource levels. The absolute difference in fitted changes from the first shared level to "
        "the last is the absolute slope difference multiplied by that resource span. Multiply by 100 "
        "to express the gap in target points (percentage points for success). This measures slope "
        "agreement independently of the starting success rate. Different pairs can have different "
        "spans, and short spans can hide slope differences.",
        "",
        f"Use the median gap over at least {MIN_PERIODS} eligible periods. Start with one group "
        "per supported algorithm and merge the groups whose largest cross-group gap is smallest. "
        f"Stop when no merge keeps every pair within {tolerance:g} points. A missing distance "
        "prevents a merge. Ties resolve alphabetically. Pairwise support can come from different "
        "periods or levels within a group.",
        "",
        f"The exploratory tolerance of {tolerance:g} points is a display choice. Memberships are "
        f"also exported at {tolerance/2:g} and {2*tolerance:g} points. The provisional grouping admits pairs "
        "observed in only one period (minimum_periods=1 in memberships.csv). Repeated periods reuse configurations. "
        "The groups describe these data and have no inferential equivalence interpretation.",
        "",
    ]
    if figure:
        lines += [f"![Pairwise differences in fitted success loss]({figure})", ""]
    lines += [
        "| Resource | Outcome | Supported pairs | Groups with ≥2 members | Singletons | Insufficient overlap |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for (metric, outcome), sub in central.groupby(["metric", "outcome"], sort=False):
        n = catalog[catalog.metric.eq(metric) & catalog.outcome.eq(outcome)].supported.sum()
        groups = [short_group(g) for g in sub.loc[sub.status.eq("grouped"), "group"].unique()]
        other = [
            ", ".join(LABELS.get(a, a) for a in sub.loc[sub.status.eq(status), "algorithm"]) or "—"
            for status in ("singleton", "insufficient overlap")
        ]
        lines.append(
            f"| {metric} | {outcome} | {n} | {' · '.join(groups) or '—'} | {other[0]} | {other[1]} |"
        )
    lines += [
        "",
        "[Pairwise gaps and support](pair_comparisons.csv) · [Membership at each tolerance](memberships.csv) · "
        "[Pair recurrence across resources, with family annotations](pair_recurrence.csv) · "
        "[Dashboard data](decay_groups.json). The per-period slopes, exact levels, and observed "
        "profiles are in the parent directory’s decay_comparisons.csv and matched_levels.csv.",
        "",
    ]
    (folder / "report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_gaps(catalog, algorithms, folder):
    """Display pairwise distances for success on the thesis's width and gate-count axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = {
        "a.num_qubits": "Total logical qubits",
        "a.ComplexityMetrics.gate_based_metrics.two_qubit_count": "Logical two-qubit gates",
    }
    metrics = {key: name for key, name in metrics.items() if catalog.metric.eq(key).any()}
    if not metrics or not catalog.outcome.eq("success").any():
        return None
    arrays = []
    for metric in metrics:
        matrix = pd.DataFrame(np.nan, index=algorithms, columns=algorithms)
        sub = catalog[catalog.metric.eq(metric) & catalog.outcome.eq("success") & catalog.supported]
        for r in sub.itertuples():
            matrix.at[r.algorithm_1, r.algorithm_2] = matrix.at[r.algorithm_2, r.algorithm_1] = (
                r.gap_points
            )
        arrays.append(matrix.to_numpy())
    finite = [x[np.isfinite(x)] for x in arrays if np.isfinite(x).any()]
    if not finite:
        return None
    high = max(1.0, max(x.max() for x in finite))
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(5.0 * len(metrics) + 1, 5.0), squeeze=False, layout="constrained"
    )
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#eeeeee")
    names = [LABELS.get(a, a) for a in algorithms]
    for ax, name, matrix in zip(axes.flat, metrics.values(), arrays):
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=high)
        ax.set_title(name)
        ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
        ax.set_yticks(range(len(names)), names)
        for i, j in np.ndindex(matrix.shape):
            val = matrix[i, j]
            label = f"{val:.1f}" if np.isfinite(val) else ("·" if i == j else "—")
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if val > high * 0.6 else "#222222",
            )
    fig.colorbar(im, ax=list(axes.flat), shrink=0.7, label="Gap in fitted success loss (pp)")
    fig.suptitle("How far apart are the fitted losses on shared resource levels?")
    fig.supxlabel(
        "Smaller values mean closer decay. A dash indicates insufficient overlap.", fontsize=10
    )
    fig.savefig(folder / "success_gaps.png", dpi=150)
    plt.close(fig)
    return "success_gaps.png"
