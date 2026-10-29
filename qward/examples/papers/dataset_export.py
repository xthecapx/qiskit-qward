"""Export v2 QPU records to one training row per job.

    uv run -m qward.examples.papers.dataset_export --backend ibm_fez --out OUT_DIR

Reads ``<algorithm>/data/qpu/v2/*.json`` (records written by
``ibm_campaign``), flattens feature groups A-G into columns, validates the
result, and writes:

    dataset.csv         one row per completed job
    schema.json         column -> group, dtype, non-null count
    export_report.json  skipped records, warnings and counts

Column naming (``<prefix>.<path>``; nested keys joined by dots, lists as JSON):

    meta.*  identity           config.*  config description scalars
    a.*     logical circuit    b.*       executed (ISA) circuit
    c.*     hardware @ submit  d.*       hardware @ execution
    e.*     baselines @ submit e_exec.*  baselines re-predicted at execution
    f.*     targets            g.*       provenance and timing

Validation is strict by default: simulated backends, duplicate jobs, mixed
schema versions, unavailable groups and counts that do not match the shot
number are errors. ``--allow-incomplete`` writes the dataset anyway and keeps
the problems in the report.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

from qward.examples.papers.ibm_campaign import RECORD_SCHEMA_VERSION

PAPERS = Path(__file__).resolve().parent
DATASET_SCHEMA_VERSION = "1.0"
GROUP_PREFIXES = ("meta", "config", "a", "b", "c", "d", "e", "e_exec", "f", "g")
# Group E may be missing only for these documented reasons.
ALLOWED_E_REASONS = {"dynamic_circuit", "no_reference"}
ALLOWED_QUANTUM_SPECIFIC_REASONS = ("size_limit", "dynamic_circuit")


def flatten(value: Any, prefix: str, out: Dict[str, Any]) -> None:
    """Flatten nested dicts into dotted scalar columns; lists become JSON text."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            flatten(item, f"{prefix}.{key}" if prefix else str(key), out)
    elif isinstance(value, (list, tuple)):
        out[prefix] = json.dumps(value, default=str)
    else:
        out[prefix] = value


def _group(prefix: str, value: Mapping[str, Any], out: Dict[str, Any]) -> None:
    if not isinstance(value, Mapping):
        out[f"{prefix}.available"] = False
        out[f"{prefix}.reason"] = "missing_from_record"
        return
    flatten(value, prefix, out)


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _seconds(start: Optional[str], end: Optional[str]) -> Optional[float]:
    a, b = _parse_time(start), _parse_time(end)
    return (b - a).total_seconds() if a and b else None


def discover(root: Path = PAPERS) -> List[Path]:
    """Journal files of every algorithm's v2 directory."""
    paths = []
    for path in sorted(Path(root).glob("*/data/qpu/v2/*.json")):
        try:
            head = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "artifact_directory" in head and "submission_journal" in head:
            paths.append(path)
    return paths


def record_rows(journal_path: Path) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Rows for the completed jobs of one record; (rows, skip_reason)."""
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    artifacts = journal_path.parent / journal["artifact_directory"]
    prepared = json.loads((artifacts / "prepared.json").read_text(encoding="utf-8"))
    results = {r["job_id"]: r for r in journal["individual_results"]}
    if not results:
        return [], f"no completed jobs (status={journal['status']})"
    features = prepared["payload"].get("features", {})
    rows = []
    for entry in journal["submission_journal"]:
        result = results.get(entry.get("job_id"))
        if result is None:
            continue
        row: Dict[str, Any] = {}
        row.update(
            {
                "meta.dataset_schema_version": DATASET_SCHEMA_VERSION,
                "meta.record_schema_version": journal["schema_version"],
                "meta.algorithm": journal["algorithm"],
                "meta.config_id": journal["config_id"],
                "meta.job_id": entry["job_id"],
                "meta.run_index": entry["run_index"],
                "meta.run_id": journal["run_id"],
                "meta.campaign_id": journal["campaign_id"],
                "meta.window": journal["window"],
                "meta.batch_id": journal.get("batch_id"),
                "meta.backend_name": journal["backend_name"],
                "meta.shots": result["shots"],
                "meta.optimization_level": prepared["optimization_level"],
                "meta.record_path": str(journal_path),
            }
        )
        flatten({k: v for k, v in prepared["config"].items() if k != "config_id"}, "config", row)
        logical, isa = dict(features.get("logical", {})), dict(features.get("isa", {}))
        for source, prefix in ((logical, "a"), (isa, "b")):
            metrics = source.pop("qward_metrics", None)
            _group(prefix, source, row)
            if metrics:
                flatten(metrics, prefix, row)
        _group("c", features.get("hardware_at_submission"), row)
        _group("d", entry.get("hardware_at_execution"), row)
        _group("e", features.get("baselines"), row)
        _group("e_exec", entry.get("baselines_at_execution"), row)
        targets = dict(result.get("targets") or {})
        counts = result["counts"]
        row["f.counts_json"] = json.dumps(counts, sort_keys=True)
        _group("f", targets, row)
        success = row.get("f.fidelity.success_rate")
        if success is not None:
            row["f.observed_error"] = 1 - success
        for source, name in (("e", "m1_error"), ("e", "m1_mixture_error"), ("e", "m2_error")):
            predicted = row.get(f"{source}.{name}")
            if predicted is not None and success is not None:
                row[f"f.residual_vs_{name}"] = (1 - success) - predicted
        metrics_t = entry.get("execution_metrics") or {}
        stamps = metrics_t.get("timestamps") or {}
        usage = metrics_t.get("usage") or {}
        row.update(
            {
                "g.seed_transpiler": prepared["seed_transpiler"],
                "g.scheduling_method": prepared["scheduling_method"],
                "g.dynamic_circuit": prepared["dynamic_circuit"],
                "g.reference_exact": prepared["payload"].get("reference_exact"),
                "g.reference_support": len(prepared["reference"] or {}),
                "g.success_set_size": len(prepared["payload"].get("success_set") or []),
                "g.twirling_gates": prepared["runtime_options"]["twirling"]["enable_gates"],
                "g.dynamical_decoupling": prepared["runtime_options"]["dynamical_decoupling"][
                    "enable"
                ],
                "g.prepared_at": prepared["prepared_at"],
                "g.submitted_at": entry.get("submitted_at"),
                "g.collected_at": entry.get("collected_at"),
                "g.running_at": stamps.get("running"),
                "g.finished_at": stamps.get("finished"),
                "g.queue_seconds": _seconds(stamps.get("created"), stamps.get("running")),
                "g.run_seconds": _seconds(stamps.get("running"), stamps.get("finished")),
                "g.quantum_seconds": usage.get("quantum_seconds"),
                "g.calibrated_at_submission": entry.get("calibrated_at_submission"),
                "g.recalibrated_before_submission": entry.get("recalibrated_before_submission"),
            }
        )
        for package, version in prepared["versions"].items():
            row[f"g.version.{package}"] = version
        rows.append(row)
    return rows, None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_row(row: pd.Series, errors: List[str], warnings: List[str]) -> None:
    """Per-job checks; appends to ``errors`` / ``warnings``."""
    tag = f"{row['meta.config_id']} job {row['meta.job_id']}"
    for group in ("a", "b", "c", "d"):
        if row.get(f"{group}.available") is not True:
            errors.append(f"{tag}: group {group} unavailable ({row.get(f'{group}.reason')})")
    if row.get("e.available") is not True and row.get("e.reason") not in ALLOWED_E_REASONS:
        errors.append(f"{tag}: baselines unavailable ({row.get('e.reason')})")
    if row.get("f.available") is not True:
        errors.append(f"{tag}: targets unavailable ({row.get('f.reason')})")
    else:
        for part in ("fidelity", "histogram_assessment"):
            if row.get(f"f.{part}.available") is not True:
                errors.append(f"{tag}: f.{part} unavailable ({row.get(f'f.{part}.reason')})")
    counts = json.loads(row["f.counts_json"])
    if sum(counts.values()) != row["meta.shots"]:
        errors.append(f"{tag}: counts sum {sum(counts.values())} != shots {row['meta.shots']}")
    coverage = row.get("c.calibration_coverage")
    if coverage is not None and not math.isclose(coverage, 1.0):
        errors.append(f"{tag}: calibration coverage {coverage:.3f} < 1")
    reason = row.get("a.quantum_specific.reason")
    if row.get("a.quantum_specific.available") is not True:
        if not str(reason).startswith(ALLOWED_QUANTUM_SPECIFIC_REASONS):
            warnings.append(f"{tag}: QuantumSpecific unavailable ({reason})")
    if row.get("d.changed_since_prepare") is True:
        warnings.append(f"{tag}: calibration changed between prepare and execution")
    if row.get("g.reference_exact") is False:
        warnings.append(f"{tag}: sampled (not exact) reference")


def validate(frame: pd.DataFrame, *, allow_fake: bool = False) -> Dict[str, List[str]]:
    """Return {"errors": [...], "warnings": [...]}; empty errors means a clean dataset."""
    errors: List[str] = []
    warnings: List[str] = []
    if frame.empty:
        return {"errors": ["no rows"], "warnings": []}
    if frame["meta.job_id"].duplicated().any():
        errors.append(
            "duplicate job IDs: "
            + ", ".join(frame.loc[frame["meta.job_id"].duplicated(), "meta.job_id"])
        )
    if frame["meta.record_schema_version"].nunique() > 1:
        errors.append(
            f"mixed record schema versions: {sorted(frame['meta.record_schema_version'].unique())}"
        )
    elif frame["meta.record_schema_version"].iloc[0] != RECORD_SCHEMA_VERSION:
        errors.append(f"unexpected record schema {frame['meta.record_schema_version'].iloc[0]}")
    if not allow_fake:
        fake = frame["meta.backend_name"].str.lower().str.startswith("fake")
        if fake.any():
            errors.append(
                "simulated records in dataset: "
                + ", ".join(sorted(frame.loc[fake, "meta.backend_name"].unique()))
            )
    for _, row in frame.iterrows():
        _validate_row(row, errors, warnings)
    return {"errors": errors, "warnings": warnings}


def column_group(column: str) -> str:
    prefix = column.split(".", 1)[0]
    return prefix if prefix in GROUP_PREFIXES else "other"


def build_schema(frame: pd.DataFrame) -> Dict[str, Any]:
    """Column catalogue written next to the CSV."""
    return {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "rows": int(len(frame)),
        "columns": [
            {
                "name": name,
                "group": column_group(name),
                "dtype": str(frame[name].dtype),
                "non_null": int(frame[name].notna().sum()),
            }
            for name in frame.columns
        ],
    }


def build_frame(paths: Iterable[Path]) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for path in paths:
        record_rows_, reason = record_rows(path)
        if reason:
            skipped.append({"record": str(path), "reason": reason})
        rows.extend(record_rows_)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, skipped
    # A gate absent from an executed circuit has count zero, not "unknown".
    for column in [c for c in frame.columns if c.startswith("b.count_ops.")]:
        frame[column] = frame[column].fillna(0).astype(int)
    ordered = sorted(
        frame.columns,
        key=lambda c: (
            GROUP_PREFIXES.index(column_group(c)) if column_group(c) in GROUP_PREFIXES else 99,
            c,
        ),
    )
    return frame[ordered], skipped


def export(
    out_dir: Path,
    *,
    root: Path = PAPERS,
    backend: Optional[str] = None,
    allow_fake: bool = False,
    allow_incomplete: bool = False,
) -> Dict[str, Any]:
    """Build, validate and write the dataset. Raises on errors unless allowed."""
    paths = discover(root)
    frame, skipped = build_frame(paths)
    if backend and not frame.empty:
        frame = frame[frame["meta.backend_name"] == backend].reset_index(drop=True)
    problems = validate(frame, allow_fake=allow_fake)
    report = {
        "exported_at": datetime.now().isoformat(),
        "records_found": len(paths),
        "rows": int(len(frame)),
        "skipped_records": skipped,
        "errors": problems["errors"],
        "warnings": problems["warnings"],
        "algorithms": sorted(frame["meta.algorithm"].unique()) if not frame.empty else [],
    }
    if problems["errors"] and not allow_incomplete:
        raise ValueError(
            f"{len(problems['errors'])} dataset error(s); first: {problems['errors'][0]} "
            "(use --allow-incomplete to write anyway)"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "dataset.csv", index=False)
    (out_dir / "schema.json").write_text(
        json.dumps(build_schema(frame), indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "export_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=PAPERS)
    parser.add_argument("--backend", help="Keep only this backend (for example ibm_fez)")
    parser.add_argument(
        "--allow-fake", action="store_true", help="Permit simulated backends (tests)"
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    report = export(
        args.out,
        root=args.root,
        backend=args.backend,
        allow_fake=args.allow_fake,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "warnings"}, indent=2))
    print(f"{len(report['warnings'])} warning(s); see export_report.json")


if __name__ == "__main__":
    main()
