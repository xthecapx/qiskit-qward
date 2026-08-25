"""Reproduce the corpus checks reported in ``draft.tex``.

The script is read only: it loads ``../DSR_result.csv`` and prints a JSON
summary for the transformation, aggregation, reference availability, and
Grover threshold checks used in the article.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr


DATA_PATH = Path(__file__).resolve().parents[1] / "DSR_result.csv"
PROVIDER_BY_EXECUTION_TYPE = {"IBM_QPU": "IBM", "AWS_BRAKET": "Rigetti"}


def _pair_summary(
    frame: pd.DataFrame,
    distance: str,
    similarity: str,
    transform: Callable[[pd.Series], pd.Series],
) -> dict[str, float | int]:
    paired = frame[[distance, similarity]].dropna()
    return {
        "rows": int(len(paired)),
        "maximum_transform_error": float(
            np.abs(paired[similarity] - transform(paired[distance])).max()
        ),
        "spearman_rho": float(spearmanr(paired[distance], paired[similarity]).statistic),
    }


def _provider_inversions(
    frame: pd.DataFrame,
    grouping: list[str],
    *,
    minimum_group_size: int,
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for distance, similarity, name in (
        ("hellinger_distance", "hellinger_fidelity", "hellinger"),
        ("tvd", "tvd_fidelity", "tvd"),
    ):
        paired = frame.dropna(subset=[distance, similarity])
        aggregated = (
            paired.groupby([*grouping, "provider"], dropna=False)
            .agg(
                n=(distance, "size"),
                distance_mean=(distance, "mean"),
                similarity_mean=(similarity, "mean"),
                distance_median=(distance, "median"),
                similarity_median=(similarity, "median"),
            )
            .reset_index()
        )

        comparable = 0
        mean_inversions = 0
        median_inversions = 0
        grouped = [("all", aggregated)] if not grouping else aggregated.groupby(grouping)
        for _, groups in grouped:
            if set(groups["provider"]) != {"IBM", "Rigetti"}:
                continue
            if not (groups["n"] >= minimum_group_size).all():
                continue
            comparable += 1
            ibm = groups.loc[groups["provider"].eq("IBM")].iloc[0]
            rigetti = groups.loc[groups["provider"].eq("Rigetti")].iloc[0]

            mean_distance_order = np.sign(ibm["distance_mean"] - rigetti["distance_mean"])
            mean_similarity_order = np.sign(
                ibm["similarity_mean"] - rigetti["similarity_mean"]
            )
            if mean_distance_order == mean_similarity_order and mean_distance_order != 0:
                mean_inversions += 1

            median_distance_order = np.sign(
                ibm["distance_median"] - rigetti["distance_median"]
            )
            median_similarity_order = np.sign(
                ibm["similarity_median"] - rigetti["similarity_median"]
            )
            if median_distance_order == median_similarity_order and median_distance_order != 0:
                median_inversions += 1

        result[name] = {
            "comparable_groupings": comparable,
            "mean_inversions": mean_inversions,
            "median_inversions": median_inversions,
        }
    return result


def analyze() -> dict[str, object]:
    data = pd.read_csv(DATA_PATH)
    transformation = {
        "hellinger": _pair_summary(
            data,
            "hellinger_distance",
            "hellinger_fidelity",
            lambda values: (1.0 - values**2) ** 2,
        ),
        "tvd": _pair_summary(
            data,
            "tvd",
            "tvd_fidelity",
            lambda values: 1.0 - values,
        ),
    }

    provider_rows = data.loc[data["execution_type"].isin(PROVIDER_BY_EXECUTION_TYPE)].copy()
    provider_rows["provider"] = provider_rows["execution_type"].map(
        PROVIDER_BY_EXECUTION_TYPE
    )
    aggregation = {
        "provider": _provider_inversions(provider_rows, [], minimum_group_size=5),
        "provider_algorithm": _provider_inversions(
            provider_rows, ["algorithm"], minimum_group_size=5
        ),
        "provider_algorithm_qubits": _provider_inversions(
            provider_rows, ["algorithm", "num_qubits"], minimum_group_size=5
        ),
        "grover_matched_configuration_qubits": _provider_inversions(
            provider_rows.loc[provider_rows["algorithm"].eq("GROVER")],
            ["config_id", "num_qubits"],
            minimum_group_size=1,
        ),
    }

    missing_reference = data.loc[
        data["hellinger_fidelity"].isna() & data["success_rate"].notna()
    ]
    missing_by_algorithm_and_qubits = {
        f"{algorithm}:{int(num_qubits)}": int(count)
        for (algorithm, num_qubits), count in missing_reference.groupby(
            ["algorithm", "num_qubits"]
        ).size().items()
    }

    grover = data.loc[data["algorithm"].eq("GROVER")]
    paired_grover = grover.dropna(
        subset=["coarse_hellinger_fidelity", "hellinger_fidelity"]
    )
    decision_subset = paired_grover.loc[paired_grover["num_qubits"].isin([3, 4])]
    disagreements = {}
    for threshold in (0.7, 0.8):
        coarse_pass = decision_subset["coarse_hellinger_fidelity"].ge(threshold)
        full_pass = decision_subset["hellinger_fidelity"].ge(threshold)
        disagreements[str(threshold)] = {
            "total": int(coarse_pass.ne(full_pass).sum()),
            "coarse_fail_full_pass": int((~coarse_pass & full_pass).sum()),
            "coarse_pass_full_fail": int((coarse_pass & ~full_pass).sum()),
        }

    return {
        "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "dataset_rows": int(len(data)),
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "transformation_checks": transformation,
        "provider_rows": {
            str(provider): int(count)
            for provider, count in provider_rows["provider"].value_counts().items()
        },
        "aggregation_checks": aggregation,
        "reference_availability": {
            "success_available_full_hellinger_unavailable": int(len(missing_reference)),
            "by_algorithm_and_qubits": missing_by_algorithm_and_qubits,
        },
        "grover": {
            "rows": int(len(grover)),
            "paired_coarse_and_full_hellinger": int(len(paired_grover)),
            "maximum_absolute_fidelity_difference": float(
                np.abs(
                    paired_grover["coarse_hellinger_fidelity"]
                    - paired_grover["hellinger_fidelity"]
                ).max()
            ),
            "three_or_four_qubit_decision_rows": int(len(decision_subset)),
            "threshold_disagreements": disagreements,
        },
    }


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2, sort_keys=True))
