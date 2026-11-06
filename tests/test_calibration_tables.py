"""Calibration sidecars preserve hardware identity without duplicating gate use."""

import csv
import json
import runpy
from pathlib import Path

import pytest

from qward.examples.papers.calibration_tables import calibration_record, write_calibrations


@pytest.fixture
def prepared():
    """Repeated gates and an overwritten measurement bit in a small snapshot."""
    gate = {"name": "cz", "qubits": [24, 25], "error": 0.003, "clbits": []}
    return {
        "run_id": "test-run",
        "backend_name": "test-chip",
        "calibration_snapshot": {
            "calibrated_at": "2026-11-04T10:00:00+00:00",
            "collected_at": "2026-11-04T10:05:00+00:00",
            "qubits": {"24": {"t1_s": 0.0002, "idle_s": 0.00001}},
            "instructions": [
                gate,
                dict(gate),
                {"name": "measure", "qubits": [24], "clbits": [0], "error": 0.01},
                {"name": "measure", "qubits": [25], "clbits": [0], "error": None},
                {"name": "delay", "qubits": [26], "clbits": [], "error": None},
            ],
        },
    }


def test_unique_gates_and_measurement_order(prepared):
    record = calibration_record(prepared)
    operations = json.loads(record["operations_json"])
    assert len(operations) == 3
    assert operations[0]["qubits"] == [24, 25]
    assert operations[-1]["error"] is None
    assert json.loads(record["measurements_json"]) == [
        {"qubits": [24], "bits": [0]},
        {"qubits": [25], "bits": [0]},
    ]
    assert "idle_s" not in json.loads(record["qubits_json"])["24"]
    assert record["source"] == "prepare_snapshot"


def test_conflicting_repeated_operation_is_rejected(prepared):
    prepared["calibration_snapshot"]["instructions"][1]["error"] = 0.4
    with pytest.raises(ValueError, match="Conflicting calibration"):
        calibration_record(prepared)


def test_empty_sidecar_has_header(tmp_path):
    path = tmp_path / "calibrations.csv"
    write_calibrations(path, [])
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert "run_id" in reader.fieldnames
        assert list(reader) == []


@pytest.fixture
def dashboard_tables(tmp_path, prepared):
    """One self-contained tabulated job, without raw journal access."""
    row = {
        "meta.job_id": "job",
        "meta.run_id": "test-run",
        "meta.shots": "10",
        "meta.backend_name": "test-chip",
        "meta.campaign_id": "campaign",
        "meta.window": "1",
        "meta.algorithm": "GHZ",
        "meta.config_id": "GHZ2",
        "config.description": "test",
        "a.num_qubits": "2",
        "a.num_clbits": "2",
        "b.depth": "3",
        "b.two_qubit_gates": "1",
        "config.depth": "",
        "meta.run_index": "0",
        "g.finished_at": "2026-11-04T10:30:00Z",
        "g.reference_exact": "True",
        "g.success_set_size": "2",
        "f.counts_json": '{"00": 8, "01": 2}',
        "f.histogram_assessment.rank_status": "test",
        "f.histogram_assessment.alpha": "",
        "f.histogram_assessment.rank_pvalue": "",
        "d.calibrated_at": "2026-11-04T10:20:00Z",
        "d.source": "job.properties()",
        "d.readout_error.mean": "0.02",
    }
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "dashboard/build.py"))
    for column in module["METRICS"].values():
        row[f"f.fidelity.{column}"] = ""
    for name, rows in (
        ("dataset.csv", [row]),
        (
            "references.csv",
            [
                {
                    "run_id": "test-run",
                    "accepted_json": '["00", "11"]',
                    "ideal_json": '{"00": 0.5, "11": 0.5}',
                }
            ],
        ),
    ):
        with (tmp_path / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_calibrations(tmp_path / "calibrations.csv", [calibration_record(prepared)])
    return module["load_tables"], tmp_path


def test_dashboard_distinguishes_preparation_from_execution(dashboard_tables):
    loader, directory = dashboard_tables
    data = loader(directory)
    calibration = data["calibrations"][data["jobs"][0]["ref"]]
    execution = data["jobs"][0]["executionCalibration"]
    assert calibration["source"] == "prepare_snapshot"
    assert execution["source"] == "job.properties()"
    assert calibration["at"] != execution["at"]
    assert execution["values"]["readout"] == 0.02
    assert execution["values"]["t1"] is None
    assert "job_id" not in json.dumps(data)


def test_dashboard_accepts_older_exports_without_calibration(dashboard_tables):
    loader, directory = dashboard_tables
    (directory / "calibrations.csv").unlink()
    assert loader(directory)["calibrations"] == {}


@pytest.mark.parametrize("field,value", [("backend_name", "wrong-chip"), ("run_id", "wrong-run")])
def test_dashboard_rejects_incorrect_calibration_join(dashboard_tables, prepared, field, value):
    loader, directory = dashboard_tables
    prepared[field] = value
    write_calibrations(directory / "calibrations.csv", [calibration_record(prepared)])
    with pytest.raises(ValueError, match="Calibration chip|Invalid calibration reference"):
        loader(directory)


def test_dashboard_writes_versioned_json_for_the_web_app(dashboard_tables, tmp_path):
    _, directory = dashboard_tables
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "dashboard/build.py"))
    out = tmp_path / "data"
    module["build"](directory, out)
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    calibrations = json.loads((out / "calibrations.json").read_text(encoding="utf-8"))
    assert index["schemaVersion"] == calibrations["schemaVersion"] == module["SCHEMA_VERSION"]
    assert [job["ref"] for job in index["jobs"]] == list(index["refs"])
    assert set(calibrations["calibrations"]) == set(index["refs"])
    assert "job_id" not in (out / "index.json").read_text(encoding="utf-8")
