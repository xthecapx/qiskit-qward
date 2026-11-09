"""Validate the exported campaign tables and write the dashboard's public JSON data."""

import argparse
import csv
import hashlib
import json
import math
import re
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
    if algorithm == "SHOR":
        return f"N={int(float(row['config.N']))} · a={int(float(row['config.a']))} · {row['config.strategy']}"
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


def dataset_rows(directory):
    """Read ordered CSV parts, validating their headers and declared row counts."""
    manifest_path = directory / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        parts = manifest["parts"]
        names = [part["file"] for part in parts]
        if (
            manifest.get("schema_version") != "1.0"
            or not names
            or len(set(names)) != len(names)
            or any(not re.fullmatch(r"dataset-\d{3,}\.csv", name) for name in names)
        ):
            raise ValueError("Invalid dataset manifest")
    else:
        manifest, parts = None, [{"file": "dataset.csv"}]
    total, header = 0, None
    for part in parts:
        path = directory / part["file"]
        if "bytes" in part and path.stat().st_size != part["bytes"]:
            raise ValueError(f"Dataset part size mismatch: {path.name}")
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if header is not None and reader.fieldnames != header:
                raise ValueError(f"Dataset part header mismatch: {path.name}")
            header = reader.fieldnames
            count = 0
            for row in reader:
                count += 1
                yield row
        if "rows" in part and count != part["rows"]:
            raise ValueError(f"Dataset part row count mismatch: {path.name}")
        total += count
    if manifest is not None and total != manifest["rows"]:
        raise ValueError("Dataset manifest total row count mismatch")


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
    for row in dataset_rows(directory):
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
            "designDepth": float(row["config.depth"]) if row.get("config.depth") else None,
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
            "executionCalibration": {
                "at": row.get("d.calibrated_at"),
                "source": row.get("d.source"),
                "values": {
                    key: float(row[f"d.{column}.mean"]) if row.get(f"d.{column}.mean") else None
                    for key, column in (
                        ("readout", "readout_error"),
                        ("single", "one_qubit_gate_error"),
                        ("pair", "two_qubit_gate_error"),
                        ("t1", "t1_us"),
                        ("t2", "t2_us"),
                    )
                },
            },
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
    calibrations = {}
    path = directory / "calibrations.csv"
    if path.exists():
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                key = row["run_id"]
                if key in calibrations or key not in refs:
                    raise ValueError(f"Invalid calibration reference: {key}")
                calibrations[key] = {
                    "backend": row["backend"],
                    "at": row["calibrated_at"],
                    "collectedAt": row["collected_at"],
                    "source": row["source"],
                    "qubits": json.loads(row["qubits_json"]),
                    "operations": json.loads(row["operations_json"]),
                    "measurements": json.loads(row["measurements_json"]),
                }
        for job in jobs:
            if job["ref"] not in calibrations:
                raise ValueError(f"Missing calibration reference: {job['ref']}")
            if calibrations[job["ref"]]["backend"] != job["backend"]:
                raise ValueError("Calibration chip does not match job")
    return {
        "jobs": jobs,
        "refs": {job["ref"]: refs[job["ref"]] for job in jobs},
        "calibrations": calibrations,
    }


SCHEMA_VERSION = 1


def write_json(path, payload):
    """Write compact, deterministic JSON. NaN is rejected so the browser never parses invalid data."""
    path.write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )


def build(tables, output):
    """Validate the tables and write the public data files that the Vite app fetches.

    ``index.json`` holds jobs and references. Calibrations are split into
    ``calibrations.json`` and only fetched when the calibration explorer opens.
    """
    data = load_tables(tables)
    output.mkdir(parents=True, exist_ok=True)
    decay_reports = {}
    manifest = Path(tables) / "dataset_manifest.json"
    if manifest.is_file():
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        for path in sorted((PROJECT / "experiments/decay").glob("*.json")):
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            if snapshot["manifestHash"] != digest:
                continue
            if any(
                hashlib.sha256((Path(tables) / name).read_bytes()).hexdigest() != expected
                for name, expected in snapshot["inputs"].items()
            ):
                continue
            filename = f"decay-{path.stem}-{digest[:12]}.json"
            write_json(output / filename, snapshot)
            decay_reports[snapshot["backend"]] = filename
    write_json(
        output / "index.json",
        {
            "schemaVersion": SCHEMA_VERSION,
            "jobs": data["jobs"],
            "refs": data["refs"],
            "decayReports": decay_reports,
        },
    )
    write_json(
        output / "calibrations.json",
        {"schemaVersion": SCHEMA_VERSION, "calibrations": data["calibrations"]},
    )
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=TABLES)
    parser.add_argument(
        "--data-out",
        type=Path,
        default=PROJECT / "public/data",
        help="Directory for the validated JSON consumed by the Vite build (default: public/data)",
    )
    args = parser.parse_args()
    summary = build(args.tables, args.data_out)
    print(
        f"Validated {len(summary['jobs'])} jobs and {len(summary['calibrations'])} calibration "
        f"records. Data written to {args.data_out.resolve()}"
    )
