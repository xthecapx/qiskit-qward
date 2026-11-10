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
        if directory == "shor":
            shutil.copyfile(
                module.PAPERS / "shor/plans/shor-growth-profile.json",
                target / "shor-growth-profile.json",
            )
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
                    "backend_name": "ibm_fez",
                    "campaign_id": "fez-dataset-v2-2026-11-05",
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


@pytest.mark.parametrize("backend", ["ibm_fez", "ibm_kingston", "ibm_marrakesh"])
def test_recovery_never_submits_prepared_batches(ci, monkeypatch, backend):
    from types import SimpleNamespace
    from qward.examples.papers import campaign_matrix, ibm_campaign, ibm_experiment_base

    calls = []
    state = ci.manifests(ci.plans("2026-11-05", create=True, backend=backend))
    for key, path in state.items():
        path.write_text(
            json.dumps(
                {
                    "status": "submitted" if key == "ghz" else "prepared",
                    "batch_id": "batch-ghz" if key == "ghz" else None,
                    "backend_name": backend,
                    "campaign_id": ci.identity("2026-11-05", backend)["campaign_id"],
                    "items": [{"job_ids": ["existing"] if key == "ghz" else []}],
                }
            ),
            encoding="utf-8",
        )
    campaign = SimpleNamespace(
        wait_manifest=lambda *a, **k: calls.append("wait"),
        recover_manifest=lambda *a, **k: (calls.append("recover") or {"status": "completed"}),
    )
    monkeypatch.setattr(
        campaign_matrix, "load_runner", lambda key: SimpleNamespace(campaign=lambda: campaign)
    )
    monkeypatch.setattr(ibm_campaign, "service_from_env", lambda **k: object())
    monkeypatch.setattr(ibm_experiment_base, "resolve_ibm_credentials", dict)
    monkeypatch.delenv("CAMPAIGN_CHECKPOINT", raising=False)
    with pytest.raises(ValueError, match="Campaign incomplete"):
        ci.execute("2026-11-05", "recover", 3, backend=backend)
    assert calls == ["recover"]


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


def test_same_date_on_three_chips_has_isolated_state_and_same_workload(ci):
    """Backend selection changes hardware identity, not experiment selection."""
    collections = [ci.identity("2026-11-06", backend) for backend in ci.BACKENDS]
    for field in ("branch", "marker", "plan_suffix", "campaign_id"):
        assert len({collection[field] for collection in collections}) == 3
    assert ci.identity("2026-11-05")["marker"] == ".github/campaign-runs/2026-11-05.json"
    all_paths = set()
    templates = ci.templates()
    for backend in ci.BACKENDS:
        ci.execute("2026-11-06", "validate", 3, backend=backend)
        paths = ci.plans("2026-11-06", create=True, backend=backend)
        assert not all_paths.intersection(paths.values())
        all_paths.update(paths.values())
        total = 0
        for key, path in paths.items():
            plan = json.loads(path.read_text(encoding="utf-8"))
            assert plan["backend"] == backend
            assert plan["items"] == templates[key]["items"]
            assert plan["shots"] == 1024
            assert plan["seed_transpiler"] == templates[key]["seed_transpiler"]
            total += plan["planned_executions"]
        assert total == 344


@pytest.mark.parametrize("backend", ["ibm_unknown", "../ibm_fez", "ibm_fez;echo bad"])
def test_invalid_backend_cannot_create_files(ci, backend):
    """Reject invalid identities before any writes or IBM access."""
    with pytest.raises(ValueError, match="Unsupported backend"):
        ci.plans("2026-11-06", create=True, backend=backend)
    assert not list(ci.PAPERS.glob("*/plans/*-d*.json"))


@pytest.mark.parametrize("mode", ["recover", "complete"])
def test_wrong_chip_manifest_is_rejected_before_recovery(ci, mode):
    """A misplaced journal must never recover another chip's collection."""
    paths = ci.plans("2026-11-06", create=True, backend="ibm_kingston")
    ci.manifests(paths)["ghz"].write_text(
        json.dumps({"backend_name": "ibm_fez", "campaign_id": "fez-dataset-v2-2026-11-06"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="another backend or campaign"):
        ci.execute("2026-11-06", mode, 3, backend="ibm_kingston")


@pytest.fixture
def cloud_runtime(ci, monkeypatch):
    """Model the full profile without transpilation, network, sleeps or QPU work."""
    from types import SimpleNamespace
    from qward.examples.papers import campaign_matrix, campaign_submit, ibm_campaign
    from qward.examples.papers import ibm_experiment_base

    sent, recovered, prepared = [], [], []
    fail_recovery = set()

    class Campaign:
        def prepare_plan(self, path, service):
            prepared.append(path)
            plan = json.loads(path.read_text(encoding="utf-8"))
            items = [
                {
                    "config": i["config"],
                    "runs": i.get("runs", plan["runs"]),
                    "record_path": i["config"] + ".json",
                    "isa_depth": 31,
                    "isa_two_qubit_gates": 7,
                    "job_ids": [],
                }
                for i in plan["items"]
            ]
            manifest = {
                "algorithm": path.name.split("-", 1)[0],
                "status": "prepared",
                "backend_name": plan["backend"],
                "campaign_id": plan["campaign_id"],
                "window": plan["window"],
                "batch_id": None,
                "jobs_planned": plan["planned_executions"],
                "items": items,
                "job_order": ibm_campaign.job_order(items, plan["order_seed"]),
            }
            ibm_campaign.write_json(path.with_name(path.stem + "_manifest.json"), manifest)

        def submit_manifest(self, path, backend, *, batch_max_time):
            assert path not in sent
            assert batch_max_time == 3600
            manifest = json.loads(path.read_text(encoding="utf-8"))
            assert manifest["jobs_planned"] <= 10
            sent.append(path)
            for item in manifest["items"]:
                assert not item["job_ids"]
                item["job_ids"] = [f"{path.name}:{item['config']}:{r}" for r in range(item["runs"])]
            manifest.update(status="submitted", batch_id=path.stem)
            ibm_campaign.write_json(path, manifest)
            return manifest

        def recover_manifest(self, path, service, *, skip_pending):
            assert skip_pending
            recovered.append(path)
            if path in fail_recovery:
                raise RuntimeError("IBM read unavailable")
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["status"] = "completed"
            ibm_campaign.write_json(path, manifest)
            return manifest

    runner = lambda key: SimpleNamespace(campaign=Campaign)
    service = SimpleNamespace(
        backend=lambda name: name,
        job=lambda job_id: SimpleNamespace(in_final_state=lambda: True, status=lambda: "DONE"),
    )
    monkeypatch.setattr(campaign_matrix, "load_runner", runner)
    monkeypatch.setattr(campaign_submit, "load_runner", runner)
    monkeypatch.setattr(ibm_campaign, "service_from_env", lambda **kw: service)
    monkeypatch.setattr(ibm_experiment_base, "resolve_ibm_credentials", dict)
    monkeypatch.setattr(ci.time, "sleep", lambda seconds: None)
    monkeypatch.delenv("CAMPAIGN_CHECKPOINT", raising=False)
    return SimpleNamespace(sent=sent, recovered=recovered, prepared=prepared, failed=fail_recovery)


def test_separate_stages_preserve_all_jobs_and_resume_without_duplicates(ci, cloud_runtime):
    from qward.examples.papers.campaign_parts import manifest_parts

    ci.execute("2026-11-06", "prepare", 3)
    state = ci.manifests(ci.plans("2026-11-06"))
    parts = manifest_parts(state)
    assert len(parts) == 37
    assert len([key for key in parts if key.startswith("pe:")]) == 5
    assert not cloud_runtime.sent and not cloud_runtime.recovered
    ci.execute("2026-11-06", "prepare", 3)
    assert len(cloud_runtime.prepared) == 8  # Resume reuses prepared circuits.
    ci.execute("2026-11-06", "submit", 3)
    assert len(cloud_runtime.sent) == 37
    assert not cloud_runtime.recovered  # No result retrieval in the submission step.
    ids_before = [json.loads(p.read_text(encoding="utf-8"))["items"] for p in parts.values()]
    ci.execute("2026-11-06", "submit", 3)
    assert len(cloud_runtime.sent) == 37
    ci.execute("2026-11-06", "recover", 3)
    ci.execute("2026-11-06", "complete", 3)
    assert len(cloud_runtime.recovered) == 37
    assert ids_before == [
        json.loads(p.read_text(encoding="utf-8"))["items"] for p in parts.values()
    ]
    ci.execute("2026-11-06", "recover", 3)
    assert len(cloud_runtime.recovered) == 37


def test_full_submit_only_keeps_batches_and_recovers_later(ci, cloud_runtime, capsys):
    options = {"backend": "ibm_kingston", "profile": "full", "max_jobs_per_batch": 6}
    day = "2026-11-09"
    ci.execute(day, "prepare", 3, **options)
    ci.execute(day, "submit-only", None, **options)
    assert "391 distinct IBM job IDs saved" in capsys.readouterr().out
    batches = [json.loads(path.read_text()) for path in cloud_runtime.sent]
    groups = [batch["submission_group"] for batch in batches]
    assert groups == sorted(groups) and set(groups) == {1, 2, 3}
    assert all(batch["jobs_planned"] <= 6 and batch["window"] == 20261109 for batch in batches)
    assert not cloud_runtime.recovered
    ci.execute(day, "submit-only", None, **options)
    ci.execute(day, "recover", 3, **options)
    assert len(batches) == len(cloud_runtime.recovered) == len(cloud_runtime.sent)


def test_recovery_retries_reads_continues_other_parts_and_never_replaces_jobs(ci, cloud_runtime):
    ci.execute("2026-11-06", "prepare", 3)
    ci.execute("2026-11-06", "submit", 3)
    failed = cloud_runtime.sent[0]
    cloud_runtime.failed.add(failed)
    with pytest.raises(ValueError, match="Campaign incomplete"):
        ci.execute("2026-11-06", "recover", 3)
    assert cloud_runtime.recovered.count(failed) == 3
    assert len(set(cloud_runtime.recovered)) == 37
    ci.execute("2026-11-06", "submit", 3)
    assert len(cloud_runtime.sent) == 37  # Failed retrieval cannot cause a replacement.
    cloud_runtime.failed.clear()
    ci.execute("2026-11-06", "recover", 3)
    assert cloud_runtime.recovered.count(failed) == 4
    assert len(cloud_runtime.recovered) == 40  # Completed parts were skipped.


def test_submit_requires_every_algorithm_prepared(ci, cloud_runtime):
    with pytest.raises(ValueError, match="Prepare every algorithm"):
        ci.execute("2026-11-06", "submit", 3)
    assert not cloud_runtime.sent


def test_partition_tampering_blocks_submission(ci, cloud_runtime):
    from qward.examples.papers.campaign_parts import manifest_parts

    ci.execute("2026-11-06", "prepare", 3)
    state = ci.manifests(ci.plans("2026-11-06"))
    parts = manifest_parts(state)
    path = parts["pe:02"]
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["items"][0]["runs"] += 1
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="changed configuration"):
        ci.execute("2026-11-06", "submit", 3)
    assert not cloud_runtime.sent


def test_resume_after_checkpoint_failure_only_submits_remaining_parts(
    ci, cloud_runtime, monkeypatch
):
    ci.execute("2026-11-06", "prepare", 3)
    calls = []

    def fail_once():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("Checkpoint unavailable")

    monkeypatch.setattr(ci, "checkpoint", fail_once)
    with pytest.raises(RuntimeError, match="Checkpoint unavailable"):
        ci.execute("2026-11-06", "submit", 3)
    assert len(cloud_runtime.sent) == 1
    first = cloud_runtime.sent[0]
    saved_ids = json.loads(first.read_text(encoding="utf-8"))["items"]
    ci.execute("2026-11-06", "prepare", 3)
    ci.execute("2026-11-06", "submit", 3)
    ci.execute("2026-11-06", "recover", 3)
    assert len(cloud_runtime.sent) == 37
    assert json.loads(first.read_text(encoding="utf-8"))["items"] == saved_ids
    assert len(cloud_runtime.prepared) == 8


@pytest.mark.parametrize(
    "profile,configs,jobs", [("shor-growth", 13, 39), ("full-shor-growth", 184, 381)]
)
def test_shor_growth_profile_isolated_from_legacy_and_completed_offline(
    ci, cloud_runtime, profile, configs, jobs
):
    from qward.examples.papers.campaign_parts import manifest_parts
    from qward.examples.papers.shor.shor_configs import get_config

    legacy = ci.identity("2026-11-05")
    proposed = ci.identity("2026-11-05", profile=profile)
    assert all(legacy[field] != proposed[field] for field in legacy)
    plans = ci.plans("2026-11-05", create=True, profile=profile)
    contents = [json.loads(p.read_text(encoding="utf-8")) for p in plans.values()]
    assert sum(len(p["items"]) for p in contents) == configs
    assert sum(p["planned_executions"] for p in contents) == jobs
    shor = json.loads(plans["shor"].read_text(encoding="utf-8"))
    assert shor["runs"] == 3 and shor["shots"] == 1024
    selected = [get_config(i["config"]) for i in shor["items"]]
    assert min(c.num_qubits for c in selected) == 6
    assert max(c.num_qubits for c in selected) == 10
    assert {c.a for c in selected} == {2, 4, 7}
    assert all(c.for_qpu and c.strategy == "swap_network" for c in selected)
    ci.execute("2026-11-05", "prepare", 3, profile=profile)
    parts = manifest_parts(ci.manifests(plans))
    assert len([key for key in parts if key.startswith("shor:")]) == 5
    ci.execute("2026-11-05", "submit", 3, profile=profile)
    ci.execute("2026-11-05", "recover", 3, profile=profile)
    ci.execute("2026-11-05", "complete", 3, profile=profile)
    assert not ci.manifests(ci.plans("2026-11-05"))["shor"].exists()


def test_shor_growth_ideal_targets_and_dashboard_families():
    from qward.examples.papers.shor.shor_configs import get_config
    from qward.examples.papers.shor.shor_ibm import ShorIBMExperiment
    from qward.examples.papers.dataset_features import ideal_reference
    from dashboard.build import family

    profile = load_ci().templates("shor-growth")["shor"]
    runner = ShorIBMExperiment()
    families = set()
    for item in profile["items"]:
        config = get_config(item["config"])
        reference = ideal_reference(runner.create_circuit(config))
        success = runner.create_success_criteria(config)
        assert sum(p for label, p in reference.items() if success(label)) == pytest.approx(0.5)
        assert config.classical_random_prob < 0.5
        row = {
            "meta.algorithm": "SHOR",
            "meta.config_id": config.config_id,
            "config.N": str(config.N),
            "config.a": str(config.a),
            "config.strategy": config.strategy,
        }
        families.add(family(row))
    assert len(families) == 3


@pytest.mark.parametrize("backend", ["ibm_fez", "ibm_kingston", "ibm_marrakesh"])
@pytest.mark.parametrize(
    "profile,configs,jobs", [("oracle-growth", 34, 68), ("full-oracle-growth", 189, 391)]
)
def test_oracle_growth_window(ci, cloud_runtime, backend, profile, configs, jobs, capsys):
    from qward.examples.papers.grover.grover_configs import get_config
    from qward.examples.papers.bv.bv_configs import get_config as get_bv

    day = "2026-11-07"
    paths = ci.plans(day, create=True, backend=backend, profile=profile)
    plans = {key: json.loads(path.read_text()) for key, path in paths.items()}
    assert sum(len(plan["items"]) for plan in plans.values()) == configs
    assert sum(plan["planned_executions"] for plan in plans.values()) == jobs
    assert all(
        plan["backend"] == backend
        and plan["window"] == 20261107
        and plan["shots"] == 1024
        and plan["optimization_level"] == 3
        and plan["seed_transpiler"] == 11
        for plan in plans.values()
    )
    assert all(
        plan["campaign_id"] == ci.identity(day, backend, profile)["campaign_id"]
        for plan in plans.values()
    )
    if profile == "full-oracle-growth":
        previous = ci.templates("full-shor-growth")
        assert plans.keys() == previous.keys()
        for key, plan in previous.items():
            assert plans[key]["items"][: len(plan["items"])] == plan["items"]
            assert plans[key]["runs"] == plan["runs"]
    else:
        assert set(plans) == {"grover", "bv"}
        assert {get_bv(item["config"]).num_qubits + 1 for item in plans["bv"]["items"]} == {3, 4, 5}
        assert {get_config(item["config"]).num_qubits for item in plans["grover"]["items"]} == {
            3,
            4,
            5,
        }
    five = [
        get_config(item["config"])
        for item in plans["grover"]["items"]
        if get_config(item["config"]).num_qubits == 5
    ]
    assert len(five) == 6 and all(c.num_marked == 1 and c.theoretical_iterations == 4 for c in five)
    assert {c.marked_states[0].count("1") for c in five} == set(range(6))
    assert {c.config_id for c in five if c.marked_states == ["01100"]} == {"S5-1"}
    ci.execute(day, "prepare", 3, backend=backend, profile=profile)
    assert "exact shared levels:" in capsys.readouterr().out
    ci.execute(day, "submit", 3, backend=backend, profile=profile)
    ci.execute(day, "recover", 3, backend=backend, profile=profile)
    ci.execute(day, "complete", 3, backend=backend, profile=profile)
    assert not ci.manifests(ci.plans(day, backend=backend, profile="full-shor-growth"))[
        "grover"
    ].exists()


@pytest.mark.parametrize("backend", ["ibm_fez", "ibm_kingston", "ibm_marrakesh"])
def test_public_full_has_every_extension_and_keeps_expanded_identity(ci, cloud_runtime, backend):
    day = "2026-11-08"
    assert ci.PUBLIC_PROFILES == ("full", "oracle-growth", "shor-growth")
    assert ci.templates("full") == ci.templates("full-oracle-growth")
    assert ci.identity(day, backend, "full") == ci.identity(day, backend, "full-oracle-growth")
    paths = ci.plans(day, create=True, backend=backend, profile="full")
    assert paths == ci.plans(day, backend=backend, profile="full-oracle-growth")
    total = sum(json.loads(path.read_text())["planned_executions"] for path in paths.values())
    assert total == 391 and set(paths) == set(ci.DIRECTORIES)
    assert sum(p["planned_executions"] for p in ci.templates("legacy-full").values()) == 344
    ci.execute(day, "prepare", 3, backend=backend, profile="full")
    ci.execute(day, "submit", 3, backend=backend, profile="full")
    ci.execute(day, "recover", 3, backend=backend, profile="full")
    ci.execute(day, "complete", 3, backend=backend, profile="full-oracle-growth")
    assert (
        ci.identity("2026-11-05", profile="legacy-full")["campaign_id"]
        == "fez-dataset-v2-2026-11-05"
    )
