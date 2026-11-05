"""Dataset exporter: flattening, validation and refusal of bad data (offline)."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import qiskit_ibm_runtime
from qiskit_ibm_runtime.fake_provider import FakeFez

from qward.examples.papers import dataset_export as D
from qward.examples.papers.ghz.ghz_ibm import GHZIBMExperiment
from qward.examples.papers.vtp.vtp_ibm import VTPIBMExperiment
from tests.test_ibm_campaign import FakeRuntime


@pytest.fixture(scope="module")
def fez():
    backend = FakeFez()
    backend.refresh = lambda: None
    return backend


@pytest.fixture(scope="module")
def runtime(fez):
    """Exporter tests need completed journals with counts and calibration."""
    fake = FakeRuntime(lambda reg, shots: {"0" * reg.size: shots})
    service = fake.service(fez)
    job_factory = service.job

    def job(job_id):
        result = job_factory(job_id)
        result.properties = fez.properties
        return result

    service.job = job
    return fake.SamplerV2, service


@pytest.fixture(scope="module")
def root(tmp_path_factory, fez, runtime):
    """<root>/<algo>/data/qpu/v2 with GHZ (2 configs x 2 runs) and dynamic vTP."""
    root = tmp_path_factory.mktemp("papers")
    sampler, service = runtime
    mp = pytest.MonkeyPatch()
    mp.setattr(qiskit_ibm_runtime, "SamplerV2", sampler)
    for cls, algorithm, configs in (
        (GHZIBMExperiment, "ghz", ["GHZ3", "GHZ4"]),
        (VTPIBMExperiment, "vtp", ["VTP-S2-SINGLE"]),
    ):
        experiment = cls()
        experiment.include_quantum_specific = lambda config: False
        experiment.get_campaign_output_dir = lambda a=algorithm: root / a / "data" / "qpu" / "v2"
        store = experiment.campaign().store
        for config_id in configs:
            path = store.prepare(
                config_id, fez, shots=128, campaign_id="export-test", planned_executions=10
            )
            store.submit(path, fez, runs=2)
            store.recover(path, service)
    mp.undo()
    return root


def test_one_row_per_job_with_all_groups(root):
    frame, skipped = D.build_frame(D.discover(root))
    assert skipped == [] and len(frame) == 6
    assert {D.column_group(c) for c in frame.columns} == set(D.GROUP_PREFIXES)
    assert frame["meta.job_id"].is_unique
    ghz = frame[frame["meta.algorithm"] == "GHZ"]
    assert ghz["e.available"].all() and ghz["a.available"].all() and ghz["d.available"].all()
    assert ghz["f.observed_error"].tolist() == pytest.approx(
        (1 - ghz["f.fidelity.success_rate"]).tolist()
    )
    expected = (1 - ghz["f.fidelity.success_rate"]) - ghz["e.m1_error"]
    assert ghz["f.residual_vs_m1_error"].tolist() == pytest.approx(expected.tolist())
    assert (ghz["b.two_qubit_gates"] > 0).all() and (ghz["c.num_used_qubits"] > 1).all()
    assert (frame["g.seed_transpiler"] == 11).all() and (
        frame["meta.optimization_level"] == 3
    ).all()


def test_dynamic_circuit_rows_keep_documented_missing_groups(root):
    frame, _ = D.build_frame(D.discover(root))
    vtp = frame[frame["meta.algorithm"] == "VTP"]
    assert (vtp["e.available"] == False).all()  # noqa: E712
    assert (vtp["e.reason"] == "dynamic_circuit").all()
    assert vtp["e.m1_error"].isna().all()
    assert vtp["a.available"].all() and vtp["f.available"].all()


def test_gates_absent_from_executed_circuit_are_zero_not_missing(root):
    frame, _ = D.build_frame(D.discover(root))
    count_columns = [c for c in frame.columns if c.startswith("b.count_ops.")]
    assert count_columns and not frame[count_columns].isna().any().any()


def test_simulated_backends_are_refused_by_default(root, tmp_path):
    with pytest.raises(ValueError, match="simulated records"):
        D.export(tmp_path / "out", root=root)
    report = D.export(tmp_path / "out", root=root, allow_fake=True)
    assert report["errors"] == [] and report["rows"] == 6
    jobs = pd.read_csv(tmp_path / "out" / "dataset.csv")
    references = pd.read_csv(tmp_path / "out" / "references.csv").set_index("run_id")
    assert references.index.is_unique
    assert set(jobs["meta.run_id"]) == set(references.index)
    calibrations = pd.read_csv(tmp_path / "out" / "calibrations.csv").set_index("run_id")
    assert calibrations.index.is_unique
    assert set(calibrations.index) == set(references.index)
    assert set(calibrations["source"]) == {"prepare_snapshot"}
    ghz_id = jobs.loc[jobs["meta.config_id"] == "GHZ3", "meta.run_id"].iloc[0]
    assert set(json.loads(references.loc[ghz_id, "accepted_json"])) == {"000", "111"}
    assert json.loads(references.loc[ghz_id, "ideal_json"]) == pytest.approx(
        {"000": 0.5, "111": 0.5}
    )
    schema = json.loads((tmp_path / "out" / "schema.json").read_text())
    assert schema["rows"] == 6
    assert {c["group"] for c in schema["columns"]} <= set(D.GROUP_PREFIXES)
    assert all("dtype" in c and "non_null" in c for c in schema["columns"])


def test_backend_filter(root, tmp_path):
    report = D.export(
        tmp_path / "none", root=root, backend="ibm_fez", allow_fake=True, allow_incomplete=True
    )
    assert report["rows"] == 0
    kept = D.export(tmp_path / "all", root=root, backend="fake_fez", allow_fake=True)
    assert kept["rows"] == 6


def _mutated(root, tmp_path, mutate):
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    for path in D.discover(copy):
        journal = json.loads(path.read_text())
        if journal["algorithm"] == "GHZ" and journal["config_id"] == "GHZ3":
            mutate(path, journal)
    return copy


def test_counts_not_matching_shots_is_an_error(root, tmp_path):
    def mutate(path, journal):
        journal["individual_results"][0]["counts"]["000"] += 5
        path.write_text(json.dumps(journal))

    copy = _mutated(root, tmp_path, mutate)
    with pytest.raises(ValueError, match="dataset error"):
        D.export(tmp_path / "out", root=copy, allow_fake=True)
    report = D.export(tmp_path / "out", root=copy, allow_fake=True, allow_incomplete=True)
    assert any("counts sum" in e for e in report["errors"])


def test_unavailable_group_is_an_error(root, tmp_path):
    def mutate(path, journal):
        entry = journal["submission_journal"][0]
        entry["hardware_at_execution"] = {
            "available": False,
            "reason": "job.properties() returned None",
        }
        path.write_text(json.dumps(journal))

    copy = _mutated(root, tmp_path, mutate)
    frame, _ = D.build_frame(D.discover(copy))
    errors = D.validate(frame, allow_fake=True)["errors"]
    assert any("group d unavailable" in e for e in errors)


def test_duplicate_jobs_and_mixed_schema_are_errors(root):
    frame, _ = D.build_frame(D.discover(root))
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    assert any("duplicate job IDs" in e for e in D.validate(duplicated, allow_fake=True)["errors"])
    mixed = frame.copy()
    mixed.loc[0, "meta.record_schema_version"] = "1.0"
    assert any("mixed record schema" in e for e in D.validate(mixed, allow_fake=True)["errors"])


def test_records_without_completed_jobs_are_skipped_and_reported(root, tmp_path, fez):
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    experiment = GHZIBMExperiment()
    experiment.include_quantum_specific = lambda config: False
    experiment.get_campaign_output_dir = lambda: copy / "ghz" / "data" / "qpu" / "v2"
    experiment.campaign().store.prepare("GHZ5", fez, shots=64)  # Prepared, never submitted.
    report = D.export(tmp_path / "out", root=copy, allow_fake=True)
    assert report["rows"] == 6
    assert len(report["skipped_records"]) == 1
    assert "no completed jobs" in report["skipped_records"][0]["reason"]


def test_legacy_files_are_never_read(tmp_path):
    records = tmp_path / "ghz/data/qpu/v2"
    records.mkdir(parents=True)
    record = records / "run.json"
    journal = json.dumps({"artifact_directory": "artifacts/run", "submission_journal": []})
    record.write_text(journal, encoding="utf-8")
    legacy = tmp_path / "ghz/data/qpu/raw"
    legacy.mkdir()
    (legacy / "old.json").write_text(journal, encoding="utf-8")
    assert D.discover(tmp_path) == [record]


def test_excluded_algorithms_never_enter_the_dataset(root, tmp_path, fez, monkeypatch, runtime):
    """Coin-toss pilot records stay on disk but are filtered out of the export."""
    import shutil

    from qward.examples.papers.coin_toss.coin_toss_ibm import CoinTossIBMExperiment

    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    sampler, service = runtime
    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", sampler)
    experiment = CoinTossIBMExperiment()
    experiment.include_quantum_specific = lambda config: False
    experiment.get_campaign_output_dir = lambda: copy / "coin_toss" / "data" / "qpu" / "v2"
    store = experiment.campaign().store
    path = store.prepare("CT3", fez, shots=64, planned_executions=4)
    store.submit(path, fez)
    store.recover(path, service)
    assert any("COIN_TOSS" in p.read_text() for p in D.discover(copy))
    report = D.export(tmp_path / "out", root=copy, allow_fake=True)
    assert report["rows"] == 6 and report["excluded_rows"] == 1
    assert "COIN_TOSS" not in set(pd.read_csv(tmp_path / "out" / "dataset.csv")["meta.algorithm"])
