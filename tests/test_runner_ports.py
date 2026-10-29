"""Every thesis runner through the shared lifecycle with groups A-G.

FakeFez provides the calibration; counts come from a noisy local Aer sampler.
No IBM service is contacted.
"""

import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import qiskit_ibm_runtime
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit_ibm_runtime.options import SamplerOptions

RUNNERS = [
    ("ghz.ghz_ibm", "GHZ4", {}),
    ("bv.bv_ibm", "BV4-ALT", {}),
    ("qft.qft_ibm", "SP4-P2", {}),
    ("phase_estimation.pe_ibm", "PE-T3", {}),
    ("grover.grover_ibm", "M3-2", {}),
    ("vtp.vtp_ibm", "VTP-V2-SINGLE", {}),
    ("vtp.vtp_ibm", "VTP-S2-SINGLE", {"dynamic": True}),
    ("random_volumetric.rv_ibm", "RV2-D2", {}),
    ("coin_toss.coin_toss_ibm", "CT3-B25", {}),
    ("shor.shor_ibm", "SHOR-N15-M3", {"options": True}),
]


@pytest.fixture(scope="module")
def fez():
    backend = FakeFez()
    backend.refresh = lambda: None
    return backend


def _runner(module_name):
    module = importlib.import_module(f"qward.examples.papers.{module_name}")
    cls = [
        c
        for n, c in inspect.getmembers(module, inspect.isclass)
        if n.endswith("IBMExperiment") and c.__module__ == module.__name__
    ][0]
    return cls()


class AerRuntime:
    """SamplerV2 stand-in: noisy Aer counts, job lookup by ID."""

    def __init__(self, backend):
        self.backend, self.results = backend, {}
        runtime = self

        class Sampler:
            def __init__(self, *, mode, options):
                self.options = SamplerOptions(**options)

            def run(self, pubs, *, shots):
                job_id = f"aer-{len(runtime.results)}"
                sampler = AerSampler.from_backend(backend, seed=len(runtime.results))
                runtime.results[job_id] = sampler.run(pubs, shots=shots).result()
                return SimpleNamespace(job_id=lambda: job_id)

        class Job:
            def __init__(self, job_id):
                self.job_id = job_id

            def result(self, timeout=None):
                return runtime.results[self.job_id]

            def metrics(self):
                return {}

            def properties(self):
                return backend.properties()

        self.Sampler = Sampler
        self.service = SimpleNamespace(backend=lambda name: backend, job=Job)


@pytest.mark.parametrize("module_name,config_id,flags", RUNNERS, ids=[r[1] for r in RUNNERS])
def test_runner_produces_every_feature_group(
    module_name, config_id, flags, tmp_path, fez, monkeypatch
):
    runtime = AerRuntime(fez)
    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", runtime.Sampler)
    experiment = _runner(module_name)
    experiment.include_quantum_specific = lambda config: False
    experiment.get_campaign_output_dir = lambda: tmp_path / "v2"
    store = experiment.campaign().store
    path = store.prepare(config_id, fez, shots=128)
    store.submit(path, fez, runs=1)
    record = store.recover(path, runtime.service)
    _, prepared, circuits = store.load(path)

    assert record["status"] == "completed"
    features = prepared["payload"]["features"]
    assert features["logical"]["available"], features["logical"]  # A
    assert features["logical"]["quantum_specific"]["reason"] == "disabled"
    assert features["isa"]["available"], features["isa"]  # B
    assert features["isa"]["physical_layout"] == prepared["physical_layout"]
    hardware = features["hardware_at_submission"]  # C
    assert hardware["calibration_coverage"] == 1.0
    assert (hardware["uncalibrated_control_flow_ops"] > 0) == bool(flags.get("dynamic"))
    if flags.get("dynamic"):
        assert features["baselines"] == {"available": False, "reason": "dynamic_circuit"}
        assert prepared["dynamic_circuit"] and prepared["scheduling_method"] is None
        assert prepared["payload"]["reference_exact"] is False
    else:
        assert features["baselines"]["available"], features["baselines"]  # E
        assert 0 <= features["baselines"]["m1_error"] <= 1
        assert 0 <= features["baselines"]["m2_tvd"] <= 1
    entry = record["submission_journal"][0]
    assert entry["hardware_at_execution"]["available"]  # D
    assert entry["hardware_at_execution"]["changed_since_prepare"] is False
    row = record["individual_results"][0]
    assert row["shots"] == 128 and sum(row["counts"].values()) == 128
    assert row["targets"]["fidelity"]["available"], row["targets"]  # F
    assert row["targets"]["histogram_assessment"]["available"]
    assert row["targets"]["fidelity"]["success_rate"] == pytest.approx(
        row["success_rate"], abs=1e-5
    )
    assert prepared["seed_transpiler"] == 11 and prepared["optimization_level"] == 3  # G
    assert prepared["versions"]["qiskit"] != "unknown"
    assert prepared["reference"] and sum(prepared["reference"].values()) == pytest.approx(1)
    assert prepared["payload"]["success_set"]
    assert circuits["isa.qpy"].num_qubits == fez.num_qubits


def test_shor_options_and_guard_apply_to_the_lifecycle(tmp_path, fez):
    experiment = _runner("shor.shor_ibm")
    experiment.include_quantum_specific = lambda config: False
    experiment.get_campaign_output_dir = lambda: tmp_path / "v2"
    store = experiment.campaign().store
    _, prepared, _ = store.load(store.prepare("SHOR-N15-M3", fez, shots=64))
    assert prepared["runtime_options"]["twirling"]["enable_gates"] is True
    assert prepared["runtime_options"]["dynamical_decoupling"] == {
        "enable": True,
        "sequence_type": "XpXm",
    }
    with pytest.raises(ValueError, match="simulator-only"):
        store.hooks.build("SHOR-N15-A2-SIM")


def test_other_runners_keep_default_runtime_options(tmp_path, fez):
    experiment = _runner("ghz.ghz_ibm")
    experiment.include_quantum_specific = lambda config: False
    experiment.get_campaign_output_dir = lambda: tmp_path / "v2"
    store = experiment.campaign().store
    _, prepared, _ = store.load(store.prepare("GHZ3", fez, shots=64))
    assert prepared["runtime_options"]["twirling"]["enable_gates"] is False
    assert prepared["runtime_options"]["dynamical_decoupling"] == {"enable": False}


@pytest.mark.parametrize(
    "module_name",
    sorted({r[0] for r in RUNNERS} | {"bv.bv_signal_background_ibm"}),
)
def test_campaign_records_are_kept_apart_from_legacy_data(module_name):
    experiment = _runner(module_name)
    legacy, campaign_dir = experiment.get_output_dir(), experiment.get_campaign_output_dir()
    assert campaign_dir.name == "v2" and campaign_dir != legacy
    assert campaign_dir.parent == legacy.parent
