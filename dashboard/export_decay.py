"""Package a completed --decay-only report for the static experiment explorer."""

import argparse
import csv
import json
import math
from pathlib import Path


def records(path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def number(value):
    return float(value) if value and math.isfinite(float(value)) else None


def package(report):
    provenance = json.loads((report / "provenance.json").read_text())
    groups = json.loads((report / "groups/decay_groups.json").read_text())
    cells = {}
    for r in groups["comparisons"]:
        key = (r["metric"], r["outcome"], r["algorithm_1"], r["algorithm_2"])
        cells[key] = dict(
            a=key[2],
            b=key[3],
            role=r["role"],
            gap=r["gap_points"],
            grouped=r["grouped_together"],
            periods={},
            overlap={},
        )
    for r in records(report / "overlap.csv"):
        key = (r["metric"], r["outcome"], r["algorithm_1"], r["algorithm_2"])
        cells[key]["overlap"][r["period"]] = int(r["shared_levels"])
    for r in records(report / "decay_comparisons.csv"):
        key = (r["metric"], r["outcome"], r["algorithm_1"], r["algorithm_2"])
        cells[key]["periods"][r["period"]] = dict(
            slopeA=number(r["slope_1"]),
            slopeB=number(r["slope_2"]),
            gap=100
            * abs(float(r["slope_difference_2_minus_1"]))
            * (float(r["metric_max"]) - float(r["metric_min"])),
            rho=number(r["profile_spearman"]),
            points=[],
        )
    for r in records(report / "matched_levels.csv"):
        key = (r["metric"], r["outcome"], r["algorithm_1"], r["algorithm_2"])
        cells[key]["periods"][r["period"]]["points"].append(
            [float(r[c]) for c in ("metric_value", "mean_1", "mean_2", "fit_1", "fit_2")]
        )
    panels = {}
    for (metric, outcome, _, _), cell in cells.items():
        panels.setdefault(metric, {}).setdefault(outcome, []).append(cell)
    return dict(
        schemaVersion=1,
        backend=provenance["backend"],
        algorithms=groups["algorithms"],
        jobs=provenance["selected_jobs"],
        tolerance=groups["tolerance_points"],
        minPeriods=groups["minimum_periods"],
        minLevels=provenance["minimum_shared_levels"],
        manifestHash=provenance["manifest_sha256"],
        inputs={
            **{p["file"]: p["sha256"] for p in provenance["inputs"]},
            "references.csv": provenance["references_sha256"],
        },
        panels=panels,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    data = package(args.report)
    path = Path(__file__).resolve().parent / "experiments/decay" / (data["backend"] + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8"
    )
    print(path)
