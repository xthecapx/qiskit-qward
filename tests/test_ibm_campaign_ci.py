"""Offline bounds for the cloud campaign entry point."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest


def load_ci():
    spec = importlib.util.spec_from_file_location(
        "ibm_campaign_ci", Path(__file__).resolve().parents[1] / "scripts/ibm_campaign_ci.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ci(tmp_path):
    module = load_ci()
    for directory in module.DIRECTORIES.values():
        target = tmp_path / directory / "plans"
        target.mkdir(parents=True)
        for path in (module.PAPERS / directory / "plans").glob("*-w[23].json"):
            shutil.copyfile(path, target / path.name)
    module.PAPERS = tmp_path
    return module


def test_validate_dates_without_credentials(ci, monkeypatch):
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    ci.execute("2026-11-05", "validate", 3)
    ci.execute("2026-11-06", "validate", 2)
    assert not list(ci.PAPERS.glob("*/plans/*-d*.json"))
    with pytest.raises(ValueError):
        ci.plans("2026-02-30")


@pytest.mark.parametrize(
    "field,value", [("shots", 4096), ("backend", "other_chip"), ("planned_executions", 999)]
)
def test_changed_plan_fails_before_execution(ci, field, value):
    path = ci.PAPERS / "ghz/plans/ghz-w2.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan[field] = value
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected campaign"):
        ci.execute("2026-11-05", "run", 3)


def test_existing_manifest_prevents_fresh_execution(ci):
    path = ci.manifests(ci.plans("2026-11-05", create=True))["ghz"]
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="already has manifests"):
        ci.execute("2026-11-05", "run", 3)


def test_completion_requires_all_unique_jobs(ci):
    paths = ci.plans("2026-11-05", create=True)
    for key, path in ci.manifests(paths).items():
        count = json.loads(paths[key].read_text(encoding="utf-8"))["planned_executions"]
        path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "items": [{"job_ids": [f"{key}-{i}" for i in range(count)]}],
                }
            ),
            encoding="utf-8",
        )
    ci.execute("2026-11-05", "complete", 3)
    path = ci.manifests(paths)["ghz"]
    data = json.loads(path.read_text(encoding="utf-8"))
    data["items"][0]["job_ids"][1] = data["items"][0]["job_ids"][0]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Campaign incomplete"):
        ci.execute("2026-11-05", "complete", 3)


def test_recovery_never_submits_prepared_batches(ci, monkeypatch):
    from types import SimpleNamespace
    from qward.examples.papers import campaign_matrix, ibm_campaign, ibm_experiment_base

    calls = []
    state = ci.manifests(ci.plans("2026-11-05", create=True))
    for key, path in state.items():
        path.write_text(
            json.dumps(
                {
                    "status": "submitted" if key == "ghz" else "prepared",
                    "items": [{"job_ids": ["existing"] if key == "ghz" else []}],
                }
            ),
            encoding="utf-8",
        )
    campaign = SimpleNamespace(
        wait_manifest=lambda *a, **k: calls.append("wait"),
        recover_manifest=lambda *a: calls.append("recover"),
    )
    monkeypatch.setattr(
        campaign_matrix, "load_runner", lambda key: SimpleNamespace(campaign=lambda: campaign)
    )
    monkeypatch.setattr(ibm_campaign, "service_from_env", lambda **k: object())
    monkeypatch.setattr(ibm_experiment_base, "resolve_ibm_credentials", dict)
    monkeypatch.delenv("CAMPAIGN_CHECKPOINT", raising=False)
    with pytest.raises(ValueError, match="Campaign incomplete"):
        ci.execute("2026-11-05", "recover", 3)
    assert calls == ["wait", "recover"]


@pytest.mark.parametrize("value", ["20261105", "2026-1-5", "2026-02-30", "../2026-11-05"])
def test_invalid_dates_are_rejected(ci, value):
    with pytest.raises(ValueError):
        ci.plans(value)


def test_dated_collections_are_distinct_and_preserve_legacy_plans(ci):
    legacy = ci.PAPERS / "ghz/plans/ghz-w2.json"
    original = legacy.read_bytes()
    first = ci.plans("2026-11-05", create=True)
    second = ci.plans("2026-12-05", create=True)
    a = json.loads(first["ghz"].read_text(encoding="utf-8"))
    b = json.loads(second["ghz"].read_text(encoding="utf-8"))
    assert a["campaign_id"] == "fez-dataset-v2-2026-11-05"
    assert b["campaign_id"] == "fez-dataset-v2-2026-12-05"
    assert a["window"] == 20261105
    assert a["items"] == b["items"]
    assert ci.manifests(first)["ghz"] != ci.manifests(second)["ghz"]
    assert legacy.read_bytes() == original
    assert ci.plans("2026-11-05", create=True) == first


def test_dated_plan_cannot_be_silently_replaced(ci):
    path = ci.plans("2026-11-05", create=True)["ghz"]
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["shots"] = 1
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="Existing dated plan differs"):
        ci.plans("2026-11-05", create=True)
