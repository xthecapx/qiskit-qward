"""Optional Fez clustering check. Run as a module with scikit-learn (tested with 1.7.2)."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from qward.examples.papers.decay_groups import group_by_decay
from qward.examples.papers.scalability_regression import PAPERS, load_data

WIDTH = "a.num_qubits"
WIDTHS = (4, 5, 6, 7, 8)
COHORT = sorted(
    ["QFT", "PHASE-ESTIMATION", "BERNSTEIN-VAZIRANI", "GHZ", "VTP", "RANDOM-VOLUMETRIC"]
)
TOLERANCE = 5.0


def width_changes():
    """Signed fitted change in success (percentage points), using identical levels in every fit."""
    data, _ = load_data(PAPERS / "dataset_out", [WIDTH], backend="ibm_fez")
    jobs = data[data.algorithm.isin(COHORT) & data[WIDTH].isin(WIDTHS)]
    configs = (
        jobs.sort_values(["period", "algorithm", WIDTH, "config", "success"])
        .groupby(["period", "algorithm", WIDTH, "config"])
        .success.mean()
    )
    profiles = configs.groupby(level=["period", "algorithm", WIDTH]).mean().unstack(WIDTH)
    profiles = profiles.reindex(columns=WIDTHS).dropna()
    changes = {}
    for period, group in profiles.groupby(level="period"):
        group = group.droplevel("period")
        if set(group.index) == set(COHORT):
            slopes = np.polyfit(WIDTHS, group.loc[COHORT].to_numpy().T, 1)[0]
            changes[period] = 100 * (WIDTHS[-1] - WIDTHS[0]) * slopes
    return pd.DataFrame(changes, index=COHORT)


def partitions(train):
    """Keep period coordinates in pp. Use the baseline median gap for the neighborhood methods."""
    x = train.to_numpy()
    distances = np.median(np.abs(x[:, None, :] - x[None, :, :]), axis=2)
    gaps = {(a, b): distances[i, j] for (i, a), (j, b) in combinations(enumerate(train.index), 2)}
    groups = group_by_decay(train.index, gaps, TOLERANCE)
    labels = {name: i for i, group in enumerate(groups) for name in group}
    yield "Complete linkage (5 pp)", np.array([labels[name] for name in train.index])
    for m in (2, 3):
        model = DBSCAN(eps=TOLERANCE, min_samples=m, metric="precomputed")
        yield f"DBSCAN (5 pp, min={m})", model.fit_predict(distances)
    for k in (2, 3):
        # Repeatable starts and canonical row order. K is fixed before validation.
        yield f"K-means (K={k})", KMeans(n_clusters=k, random_state=0, n_init=20).fit_predict(x)
        mixture = GaussianMixture(
            n_components=k, covariance_type="tied", reg_covar=1.0, random_state=0, n_init=10
        )
        yield f"Gaussian mixture (K={k})", mixture.fit_predict(x)


def compare(changes):
    """Score co-membership by whether the held-out fitted changes also fall within 5 pp."""
    if changes.shape[1] < 3:
        raise ValueError("At least three commonly observed periods are required")
    rows = []
    for period in changes:
        train = changes.drop(columns=period)
        for method, labels in partitions(train):
            for i, j in combinations(range(len(changes)), 2):
                gap = abs(changes[period].iloc[i] - changes[period].iloc[j])
                rows.append(
                    {
                        "period": period,
                        "method": method,
                        "a": changes.index[i],
                        "b": changes.index[j],
                        "grouped": labels[i] >= 0 and labels[i] == labels[j],
                        "heldout_gap_pp": gap,
                        "within_5pp": gap <= TOLERANCE,
                    }
                )
    pairs = pd.DataFrame(rows)
    pairs["confirmed"] = pairs.grouped & pairs.within_5pp
    summary = pairs.groupby("method", sort=False).agg(
        grouped=("grouped", "sum"),
        confirmed=("confirmed", "sum"),
        eligible_close=("within_5pp", "sum"),
    )
    summary["precision"] = summary.confirmed / summary.grouped.replace(0, np.nan)
    summary["recall"] = summary.confirmed / summary.eligible_close.replace(0, np.nan)
    return pairs, summary


def main():
    folder = Path("report/scalability_regression/decay/method_comparison")
    folder.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        changes = width_changes()
        pairs, summary = compare(changes)
    changes.rename_axis("algorithm").to_csv(folder / "width_changes_pp.csv")
    pairs.to_csv(folder / "heldout_pairs.csv", index=False)
    summary.to_csv(folder / "comparison.csv")
    print(summary.round(3).to_string())
    # The existing all-pairs gaps can also be used directly as DBSCAN neighborhoods.
    catalog = pd.read_csv(folder.parent / "groups" / "pair_comparisons.csv")
    panels = []
    for (metric, outcome), pairs in catalog.groupby(["metric", "outcome"]):
        supported = pairs[pairs.supported]
        names = sorted(set(supported.algorithm_1) | set(supported.algorithm_2))
        if not names:
            continue
        # Distances above eps are equivalent to absent edges for DBSCAN.
        distances = pd.DataFrame(TOLERANCE + 1, index=names, columns=names)
        for name in names:
            distances.at[name, name] = 0.0
        for row in supported.itertuples():
            distances.at[row.algorithm_1, row.algorithm_2] = row.gap_points
            distances.at[row.algorithm_2, row.algorithm_1] = row.gap_points
        gaps = {
            tuple(sorted((r.algorithm_1, r.algorithm_2))): r.gap_points
            for r in supported.itertuples()
        }
        baseline = group_by_decay(names, gaps, TOLERANCE)
        members = {pair for group in baseline for pair in combinations(group, 2)}
        for minimum in (2, 3):
            labels = DBSCAN(eps=TOLERANCE, min_samples=minimum, metric="precomputed").fit_predict(
                distances.to_numpy()
            )
            found = {
                (a, b)
                for (a, i), (b, j) in combinations(zip(names, labels), 2)
                if i == j and i >= 0
            }
            panels.append(
                dict(
                    metric=metric,
                    outcome=outcome,
                    min_samples=minimum,
                    supported_pairs=len(supported),
                    identical_pairs=members == found,
                    added_pairs=len(found - members),
                    removed_pairs=len(members - found),
                    above_tolerance=sum(gaps[p] > TOLERANCE for p in found if p in gaps),
                    unsupported_grouped_pairs=len(found - gaps.keys()),
                )
            )
    pd.DataFrame(panels).to_csv(folder / "dbscan_vs_complete.csv", index=False)


if __name__ == "__main__":
    main()
