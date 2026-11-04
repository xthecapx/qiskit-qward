"""Build a self-contained, static experiment dashboard from the campaign tables."""

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path
from statistics import NormalDist

PROJECT = Path(__file__).resolve().parent
TABLES = PROJECT.parent / "qward/examples/papers/dataset_out"
METRICS = {
    "success": "success_rate",
    "dsr": "dsr",
    "tvd": "tvd",
    "hellinger": "hellinger_fidelity",
    "coarse": "coarse_tvd",
    "chance": "chance_corrected_success",
    "baseline": "chance_baseline",
}


def family(row):
    """Group only comparable configurations into a sweep. Keep other cases individual."""
    algorithm, config = row["meta.algorithm"], row["meta.config_id"]
    if algorithm == "GHZ":
        return "GHZ states"
    if algorithm == "BERNSTEIN-VAZIRANI":
        return config.split("-")[-1] + " secret"
    if algorithm == "VTP":
        return f"{row['config.protocol_type']} · {row['config.gates_per_qubit'].split('.')[0]} gates per payload qubit"
    if algorithm == "PHASE-ESTIMATION":
        return row["config.test_case"]
    if algorithm == "RANDOM-VOLUMETRIC":
        return f"{row['config.num_qubits'].split('.')[0]} qubits · seed {row['config.seed'].split('.')[0]}"
    if algorithm == "QFT":
        if re.fullmatch(r"SR\d+", config):
            return "SR · round trip"
        if re.fullmatch(r"(SP|PV)\d+-P\d+", config):
            return f"{config[:2]} · period {row['config.period'].split('.')[0]}"
    if algorithm == "GROVER" and re.fullmatch(r"S\d+-1", config):
        return "Single marked state · S series"
    return config


def success_interval(accepted_shots, shots):
    """Two-sided 99% Wilson interval for a job's accepted-shot probability."""
    z = NormalDist().inv_cdf(0.995)
    proportion = accepted_shots / shots
    denominator = shots + z**2
    center = (accepted_shots + z**2 / 2) / denominator
    radius = z * math.sqrt(shots * proportion * (1 - proportion) + z**2 / 4) / denominator
    return [
        0 if accepted_shots == 0 else center - radius,
        1 if accepted_shots == shots else center + radius,
    ]


def load_tables(directory):
    """Project only public fields. The renderer never needs the original job journals."""
    csv.field_size_limit(64 * 1024 * 1024)
    refs = {}
    with (directory / "references.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["run_id"]
            if key in refs:
                raise ValueError(f"Duplicate reference row for {key}")
            refs[key] = {
                "accepted": json.loads(row["accepted_json"]),
                "ideal": json.loads(row["ideal_json"]),
            }
    jobs = []
    seen = set()
    with (directory / "dataset.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["meta.job_id"]
            ref = row["meta.run_id"]
            counts = json.loads(row["f.counts_json"])
            shots = int(row["meta.shots"])
            if key in seen or sum(counts.values()) != shots or shots <= 0 or ref not in refs:
                raise ValueError(f"Invalid job or reference join for {key}")
            seen.add(key)
            target = refs[ref]
            if len(target["accepted"] or []) != int(row["g.success_set_size"]):
                raise ValueError(f"Accepted-label count mismatch for {key}")
            ideal = target["ideal"]
            if ideal and not math.isclose(sum(ideal.values()), 1, abs_tol=1e-6):
                raise ValueError(f"Ideal probabilities do not sum to one for {key}")
            ranked = sorted(counts.values(), reverse=True) + [0]
            job = {
                "backend": row["meta.backend_name"],
                "campaign": row["meta.campaign_id"],
                "window": row["meta.window"],
                "algorithm": row["meta.algorithm"],
                "family": family(row),
                "config": row["meta.config_id"],
                "description": row["config.description"],
                "width": int(row["a.num_qubits"]),
                "bits": int(row["a.num_clbits"]),
                "depth": int(row["b.depth"]),
                "gates": int(row["b.two_qubit_gates"]),
                "designDepth": float(row["config.depth"]) if row["config.depth"] else None,
                "shots": shots,
                "rep": int(row["meta.run_index"]) + 1,
                "ref": ref,
                "finishedAt": row["g.finished_at"],
                "exact": row["g.reference_exact"] == "True",
                "counts": counts,
                "contrast": (ranked[0] - ranked[1]) / (ranked[0] + ranked[1]),
                "rank": row["f.histogram_assessment.rank_status"],
                "alpha": row["f.histogram_assessment.alpha"],
                "rankP": row["f.histogram_assessment.rank_pvalue"],
            }
            if job["algorithm"] == "QFT" and row.get("config.test_mode") == "period_detection":
                job["countingQubits"] = int(float(row["config.num_qubits"]))
            for name, column in METRICS.items():
                value = row[f"f.fidelity.{column}"]
                job[name] = float(value) if value else None
            if target["accepted"] is not None and job["success"] is not None:
                accepted_shots = sum(counts.get(label, 0) for label in target["accepted"])
                job["successInterval"] = success_interval(accepted_shots, shots)
            jobs.append(job)
    if not jobs:
        raise ValueError("The dashboard needs at least one completed job")
    return {"jobs": jobs, "refs": {job["ref"]: refs[job["ref"]] for job in jobs}}


def build(tables, output):
    snapshot = json.dumps(load_tables(tables), separators=(",", ":"), allow_nan=False)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "styles.css", "app.js"):
        shutil.copyfile(PROJECT / name, output / name)
    shutil.copytree(PROJECT / "vendor", output / "vendor", dirs_exist_ok=True)
    (output / "snapshot.js").write_text(f"const snapshot = {snapshot};\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=TABLES)
    parser.add_argument("--out", type=Path, default=PROJECT / "dist")
    args = parser.parse_args()
    build(args.tables, args.out)
    print(f"Dashboard ready at {(args.out / 'index.html').resolve()}")
