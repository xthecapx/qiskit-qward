"""Shared IBM lifecycle: offline tests with FakeFez and a mocked Runtime.

No IBM service is contacted and no circuit is executed on hardware.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from qiskit.primitives.containers import BitArray, DataBin
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit_ibm_runtime.options import SamplerOptions

from qward.examples.papers import ibm_campaign
from qward.examples.papers.ghz.ghz_ibm import GHZIBMExperiment
from qward.examples.papers.vtp.vtp_ibm import VTPIBMExperiment


@pytest.fixture(scope="module")
def fez():
    backend = FakeFez()
    backend.refresh = lambda: None
    return backend


def _experiment(cls, tmp_path, monkeypatch, *, quantum_specific=False):
    experiment = cls()
    # QuantumSpecificMetrics are slow; exercised separately in test_dataset_features.
    experiment.include_quantum_specific = lambda config: quantum_specific
    experiment.get_campaign_output_dir = lambda: tmp_path / "v2"
    return experiment


class FakeRuntime:
    """Records submissions; returns deterministic counts on recovery."""

    def __init__(self, counts_for):
        self.submitted = []
        self.batches = []
        self.counts_for = counts_for
        runtime = self

        class Batch:
            def __init__(self, *, backend):
                runtime.batches.append(backend)
                self.session_id = f"batch-{len(runtime.batches)}"

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class Sampler:
            def __init__(self, *, mode, options):
                self.mode = mode
                self.options = SamplerOptions(**options)

            def run(self, pubs, *, shots):
                job_id = f"job-{len(runtime.submitted)}"
                runtime.submitted.append(
                    {
                        "job_id": job_id,
                        "circuit": pubs[0],
                        "shots": shots,
                        "tags": list(self.options.environment.job_tags),
                        "mode": self.mode,
                    }
                )
                return SimpleNamespace(job_id=lambda: job_id)

        self.Batch, self.SamplerV2 = Batch, Sampler

    def service(self, backend):
        runtime = self

        class Job:
            def __init__(self, job_id):
                self.job_id = job_id

            def result(self, timeout=None):
                sent = next(s for s in runtime.submitted if s["job_id"] == self.job_id)
                circuit = sent["circuit"]
                data = {
                    reg.name: BitArray.from_counts(
                        runtime.counts_for(reg, sent["shots"]), num_bits=reg.size
                    )
                    for reg in circuit.cregs
                }
                return [SimpleNamespace(data=DataBin(**data), metadata={"fixture": True})]

            def metrics(self):
                return {"timestamps": {"running": "fixture"}}

        return SimpleNamespace(backend=lambda name: backend, job=Job)


@pytest.fixture
def runtime(monkeypatch):
    import qiskit_ibm_runtime

    fake = FakeRuntime(lambda reg, shots: {"0" * reg.size: shots})
    monkeypatch.setattr(qiskit_ibm_runtime, "Batch", fake.Batch)
    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", fake.SamplerV2)
    return fake


def _plan(tmp_path, **plan):
    path = tmp_path / "plans" / "ghz-w1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "backend": "fake_fez",
                "campaign_id": "test",
                "window": 1,
                "shots": 128,
                "runs": 2,
                "order_seed": 3,
                **plan,
            }
        )
    )
    return path


def test_prepare_freezes_isa_once_with_seed_and_used_qubit_calibration(tmp_path, fez, monkeypatch):
    experiment = _experiment(GHZIBMExperiment, tmp_path, monkeypatch)
    store = experiment.campaign().store
    path = store.prepare("GHZ4", fez, shots=128, optimization_level=3)
    journal, prepared, circuits = store.load(path)
    assert journal["status"] == "prepared" and journal["submission_journal"] == []
    assert prepared["seed_transpiler"] == 11 and prepared["optimization_level"] == 3
    assert prepared["physical_layout"] is not None
    snapshot = prepared["calibration_snapshot"]
    assert snapshot["calibrated_at"] is not None
    assert snapshot["backend_properties"] is None  # Used-qubit data only.
    assert set(snapshot["qubits"]) <= {str(q) for q in range(fez.num_qubits)}
    assert all(q["prob_meas1_prep0"] is not None for q in snapshot["qubits"].values())
    features = prepared["payload"]["features"]
    assert features["logical"]["available"] and features["isa"]["available"]
    assert features["isa"]["physical_layout"] == prepared["physical_layout"]
    assert features["baselines"]["available"]
    assert prepared["reference"] == {"0000": 0.5, "1111": 0.5}
    assert prepared["runtime_options"]["twirling"]["enable_gates"] is False
    assert circuits["isa.qpy"].num_qubits == fez.num_qubits
    (path.parent / journal["artifact_directory"] / "isa.qpy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        store.load(path)


def test_dynamic_circuit_is_prepared_unscheduled(tmp_path, fez, monkeypatch):
    experiment = _experiment(VTPIBMExperiment, tmp_path, monkeypatch)
    store = experiment.campaign().store
    _, prepared, circuits = store.load(store.prepare("VTP-S2-SINGLE", fez, shots=64))
    assert prepared["dynamic_circuit"] is True
    assert prepared["scheduling_method"] is None
    assert any(i.operation.name == "if_else" for i in circuits["isa.qpy"].data)


def test_plan_submits_one_batch_with_seeded_round_robin_order(tmp_path, fez, monkeypatch, runtime):
    experiment = _experiment(GHZIBMExperiment, tmp_path, monkeypatch)
    campaign = experiment.campaign()
    plan = _plan(
        tmp_path, items=[{"config": "GHZ3"}, {"config": "GHZ4"}, {"config": "GHZ5", "runs": 1}]
    )
    manifest = campaign.prepare_plan(plan, runtime.service(fez))
    assert manifest["jobs_planned"] == 5
    order = manifest["job_order"]
    assert [r for _, r in order] == [0, 0, 0, 1, 1]  # Run 1 of all before run 2.
    assert order == ibm_campaign.job_order(manifest["items"], 3)
    with pytest.raises(ValueError, match="never submitted"):
        campaign.recover_manifest(Path(manifest["manifest_path"]), runtime.service(fez))
    submitted = campaign.submit_manifest(Path(manifest["manifest_path"]), fez)
    assert len(runtime.batches) == 1 and submitted["batch_id"] == "batch-1"
    assert len(runtime.submitted) == 5
    assert all(not isinstance(s["mode"], FakeFez) for s in runtime.submitted)
    assert [len(i["job_ids"]) for i in submitted["items"]] == [2, 2, 1]
    with pytest.raises(ValueError, match="fresh prepared manifest"):
        campaign.submit_manifest(Path(manifest["manifest_path"]), fez)
    recovered = campaign.recover_manifest(Path(manifest["manifest_path"]), runtime.service(fez))
    assert recovered["status"] == "completed"
    assert len(runtime.submitted) == 5  # Recovery never resubmits.
    record = json.loads(Path(submitted["items"][0]["record_path"]).read_text())
    assert record["batch_id"] == "batch-1"
    assert [r["run_index"] for r in record["individual_results"]] == [0, 1]
    assert record["individual_results"][0]["shots"] == 128
    assert "success_rate" in record["individual_results"][0]
    again = campaign.recover_manifest(Path(manifest["manifest_path"]), runtime.service(fez))
    assert again["status"] == "completed"  # Idempotent.
    status = campaign.status(Path(manifest["manifest_path"]))
    assert status["jobs_submitted"] == 5 and status["records_by_status"] == {"completed": 3}


def test_plan_validation_leaves_no_records(tmp_path, fez, monkeypatch, runtime):
    experiment = _experiment(GHZIBMExperiment, tmp_path, monkeypatch)
    campaign = experiment.campaign()
    cases = [
        ({"items": [{"config": "NOPE"}]}, "Unknown"),
        ({"items": [{"config": "GHZ3", "runs": 0}]}, "runs"),
        ({"items": [{"config": "GHZ3"}, {"config": "GHZ3"}]}, "more than once"),
        ({"items": [{"config": "GHZ3"}], "planned_executions": 1}, "budget"),
    ]
    for overrides, message in cases:
        with pytest.raises(ValueError, match=message):
            campaign.prepare_plan(_plan(tmp_path, **overrides), runtime.service(fez))
    assert not list((tmp_path / "v2").glob("*.json")) if (tmp_path / "v2").exists() else True


def test_multi_register_counts_are_joined_like_aer(tmp_path, fez, monkeypatch):
    import qiskit_ibm_runtime

    # alice_meas (first register) all ones, test_result (last register) all zeros.
    fake = FakeRuntime(
        lambda reg, shots: {("1" if reg.name == "alice_meas" else "0") * reg.size: shots}
    )
    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", fake.SamplerV2)
    experiment = _experiment(VTPIBMExperiment, tmp_path, monkeypatch)
    store = experiment.campaign().store
    path = store.prepare("VTP-S2-SINGLE", fez, shots=64)
    store.submit(path, fez, runs=1)
    assert isinstance(fake.submitted[0]["mode"], FakeFez)  # Single record: job mode.
    record = store.recover(path, fake.service(fez))
    counts = record["individual_results"][0]["counts"]
    assert counts == {"00 11": 64}  # Last register leftmost, space separated.
    assert record["individual_results"][0]["success_rate"] == 1.0


def test_submission_failure_is_journaled_and_not_retried(tmp_path, fez, monkeypatch):
    import qiskit_ibm_runtime

    class Failing:
        def __init__(self, *, mode, options):
            self.options = SamplerOptions(**options)

        def run(self, pubs, *, shots):
            raise RuntimeError("network down")

    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", Failing)
    experiment = _experiment(GHZIBMExperiment, tmp_path, monkeypatch)
    store = experiment.campaign().store
    path = store.prepare("GHZ3", fez, shots=64)
    with pytest.raises(RuntimeError):
        store.submit(path, fez)
    journal = json.loads(path.read_text())
    assert journal["status"] == "submission_unknown"
    assert journal["submission_journal"][0]["job_id"] is None
    with pytest.raises(ValueError, match="fresh prepared"):
        store.submit(path, fez)


def test_runtime_options_hook_merges_onto_defaults(tmp_path, fez, monkeypatch):
    experiment = _experiment(GHZIBMExperiment, tmp_path, monkeypatch)
    experiment.get_runtime_options = lambda config: {"twirling": {"enable_gates": True}}
    store = experiment.campaign().store
    _, prepared, _ = store.load(store.prepare("GHZ3", fez, shots=64))
    assert prepared["runtime_options"]["twirling"] == {
        "enable_gates": True,
        "enable_measure": False,
    }
    assert prepared["runtime_options"]["dynamical_decoupling"] == {"enable": False}


def test_cli_lifecycle_flags_do_not_touch_legacy_run(tmp_path, fez, monkeypatch, runtime):
    experiment = _experiment(GHZIBMExperiment, tmp_path, monkeypatch)
    monkeypatch.setattr(ibm_campaign, "service_from_env", lambda **kw: runtime.service(fez))
    monkeypatch.setattr(experiment, "run", lambda *a, **k: pytest.fail("legacy run called"))
    result = experiment.run_cli(
        [
            "--prepare-only",
            "--config",
            "GHZ3",
            "--backend",
            "fake_fez",
            "--opt-levels",
            "3",
            "--shots",
            "64",
        ]
    )
    assert result["status"] == "prepared"
    summary = experiment.run_cli(["--submit-record", result["record_path"], "--runs", "1"])
    assert summary["status"] == "submitted"
    recovered = experiment.run_cli(["--recover-record", result["record_path"]])
    assert recovered["status"] == "completed"


def test_extract_counts_matches_aer_register_order():
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSampler

    q = QuantumRegister(4)
    first, last = ClassicalRegister(2, "first"), ClassicalRegister(2, "last")
    circuit = QuantumCircuit(q, first, last)
    circuit.x(0)
    circuit.h(1)
    circuit.x(3)
    circuit.measure(q[0], first[0])
    circuit.measure(q[1], first[1])
    circuit.measure(q[2], last[0])
    circuit.measure(q[3], last[1])
    pub = AerSampler(seed=5).run([circuit], shots=200).result()[0]
    expected = AerSimulator(seed_simulator=5).run(circuit, shots=200).result().get_counts()
    assert sorted(ibm_campaign.extract_counts(pub)) == sorted(expected)
    assert sum(ibm_campaign.extract_counts(pub).values()) == 200
