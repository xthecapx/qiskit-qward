"""Feature/target groups A-G for the ibm_fez dataset (offline, FakeFez)."""

import json
from pathlib import Path

import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeFez

from qward.examples.papers import dataset_features as F
from qward.examples.papers.coin_toss.coin_toss_ibm import CoinTossIBMExperiment
from qward.examples.papers.ghz.ghz_ibm import GHZIBMExperiment
from qward.examples.papers.qft.qft_ibm import QFTIBMExperiment
from qward.examples.papers.vtp.vtp_ibm import VTPIBMExperiment
from qward.metrics.circuit_error_prediction import capture_calibration


@pytest.fixture(scope="module")
def fez():
    return FakeFez()


def _isa(circuit, backend, scheduled=True):
    return generate_preset_pass_manager(
        optimization_level=3,
        backend=backend,
        seed_transpiler=11,
        scheduling_method="alap" if scheduled else None,
    ).run(circuit)


def test_compact_isa_keeps_gates_and_depth(fez):
    experiment = QFTIBMExperiment()
    isa = _isa(experiment.create_circuit(experiment.get_config("SR5")), fez)
    compact = F.compact_isa(isa)
    assert compact.num_qubits == 5
    # Delays and barriers spanning idle device wires inflate full-device depth.
    assert compact.depth() == isa.depth(lambda i: i.operation.name not in {"delay", "barrier"})
    unscheduled = _isa(experiment.create_circuit(experiment.get_config("SR5")), fez, False)
    assert compact.depth() == unscheduled.depth()
    assert sum(len(i.qubits) == 2 for i in compact.data) == sum(
        len(i.qubits) == 2 for i in isa.data
    )
    assert "delay" not in compact.count_ops()


def test_isa_metrics_are_computed_on_executed_circuit(fez):
    experiment = GHZIBMExperiment()
    logical = experiment.create_circuit(experiment.get_config("GHZ5"))
    isa = _isa(logical, fez)
    group = F.isa_features(isa)
    assert group["available"], group
    assert group["two_qubit_gates"] == 4
    assert len(group["physical_layout"]) == 5
    assert set(group["qward_metrics"]) >= {"QiskitMetrics", "ComplexityMetrics"}
    a = F.logical_features(logical, include_quantum_specific=False)
    assert a["available"] and a["quantum_specific"] == F.missing("disabled")


def test_quantum_specific_limits_are_explicit():
    big = QuantumCircuit(F.QUANTUM_SPECIFIC_MAX_QUBITS + 1)
    assert F.quantum_specific(big)["reason"].startswith("size_limit")
    experiment = VTPIBMExperiment()
    dynamic = experiment.create_circuit(experiment.get_config("VTP-S2-SINGLE"))
    assert F.quantum_specific(dynamic) == F.missing("dynamic_circuit")
    small = QuantumCircuit(2)
    small.h(0)
    small.cx(0, 1)
    computed = F.quantum_specific(small)
    assert computed["available"] and "magic" in computed and "compute_seconds" in computed


def test_hardware_and_baselines_from_snapshot(fez):
    experiment = GHZIBMExperiment()
    config = experiment.get_config("GHZ4")
    isa = _isa(experiment.create_circuit(config), fez)
    snapshot = capture_calibration(isa, fez, include_backend_properties=False)
    hw = F.hardware_features(snapshot)
    assert hw["calibration_coverage"] == 1.0
    assert hw["num_used_qubits"] == 4
    assert hw["two_qubit_gate_error"]["max"] >= hw["two_qubit_gate_error"]["min"] > 0
    assert hw["readout_p0_given_1"]["mean"] is not None
    assert hw["t1_us"]["min"] > 0 and hw["scheduled"] is True
    reference = experiment.get_reference(config)
    assert reference == pytest.approx({"0000": 0.5, "1111": 0.5})
    baselines = F.baseline_predictions(isa, snapshot, reference)
    assert baselines["available"], baselines
    assert 0 < baselines["m1_error"] < 1
    # Two targets: predicted success-set mass, comparable with 1 - success_rate.
    assert 0 < baselines["m1_mixture_error"] < baselines["m1_error"]
    assert 0 < baselines["m2_error"] < 1 and baselines["m2_tvd"] is not None


def test_single_target_baselines_match_point_predictions(fez):
    from qward.examples.papers.bv.bv_ibm import BVIBMExperiment
    from qward.metrics.circuit_error_prediction import CircuitErrorPrediction

    experiment = BVIBMExperiment()
    config = experiment.get_config("BV3-ALT")
    isa = _isa(experiment.create_circuit(config), fez)
    snapshot = capture_calibration(isa, fez, include_backend_properties=False)
    reference = experiment.get_reference(config)
    group = F.baseline_predictions(isa, snapshot, reference)
    point = CircuitErrorPrediction(isa, snapshot=snapshot, reference=reference).get_metrics()
    assert group["m1_error"] == pytest.approx(1 - point.m1_success)
    assert group["m2_error"] == pytest.approx(1 - point.m2_success)
    assert group["m1_mixture_error"] == pytest.approx(point.mixture_tvd)


def test_dynamic_circuit_baselines_are_missing(fez):
    experiment = VTPIBMExperiment()
    config = experiment.get_config("VTP-S2-SINGLE")
    isa = _isa(experiment.create_circuit(config), fez, scheduled=False)
    snapshot = capture_calibration(isa, fez, include_backend_properties=False)
    assert F.baseline_predictions(isa, snapshot, {"00 00": 1.0}) == F.missing("dynamic_circuit")
    assert F.hardware_features(snapshot)["scheduled"] is False


def test_targets_handle_multi_register_labels():
    experiment = VTPIBMExperiment()
    config = experiment.get_config("VTP-S2-SINGLE")
    circuit = experiment.create_circuit(config)
    reference = experiment.get_reference(config)  # Dynamic: noiseless sampling.
    assert set(reference) == {"00 00", "00 01", "00 10", "00 11"}
    assert all(v == pytest.approx(0.25, abs=0.01) for v in reference.values())
    assert F.reference_is_exact(circuit) is False
    counts = {"00 00": 400, "00 01": 300, "01 10": 100, "00 11": 224}
    group = F.targets(
        counts, circuit, reference, success=experiment.create_success_criteria(config)
    )
    assert group["fidelity"]["success_rate"] == pytest.approx(924 / 1024)
    assert group["fidelity"]["tvd"] == pytest.approx(0.28125, abs=0.01)
    assert group["histogram_assessment"]["tvd"] == pytest.approx(group["fidelity"]["tvd"])


def test_missing_reference_is_explicit():
    circuit = QuantumCircuit(1, 1)
    assert F.targets({"0": 1}, circuit, None) == F.missing("no_reference")
    with pytest.raises(ValueError, match="accept no"):
        F.reference_from_success(circuit, lambda label: False)


def test_execution_hardware_reuses_snapshot_instructions(fez):
    experiment = GHZIBMExperiment()
    config = experiment.get_config("GHZ3")
    isa = _isa(experiment.create_circuit(config), fez)
    snapshot = capture_calibration(isa, fez, include_backend_properties=False)

    class Job:
        @staticmethod
        def properties():
            return fez.properties()

    out = F.execution_hardware(Job(), snapshot, isa, experiment.get_reference(config))
    late = out["hardware_at_execution"]
    assert late["source"] == "job.properties()" and late["changed_since_prepare"] is False
    early = F.hardware_features(snapshot)
    assert late["two_qubit_gate_error"] == early["two_qubit_gate_error"]
    assert out["baselines_at_execution"]["m1_error"] == pytest.approx(
        F.baseline_predictions(isa, snapshot, experiment.get_reference(config))["m1_error"]
    )


def test_full_record_contains_every_group(tmp_path, monkeypatch, fez):
    """Prepare -> submit -> recover a GHZ record and check A-G are present."""
    import qiskit_ibm_runtime
    from tests.test_ibm_campaign import FakeRuntime

    runtime = FakeRuntime(
        lambda reg, shots: {"0" * reg.size: shots // 2, "1" * reg.size: shots // 2}
    )
    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", runtime.SamplerV2)
    experiment = GHZIBMExperiment()
    experiment.include_quantum_specific = lambda config: True
    experiment.get_campaign_output_dir = lambda: tmp_path / "v2"
    store = experiment.campaign().store
    path = store.prepare("GHZ3", fez, shots=256)
    store.submit(path, fez, runs=1)
    service = runtime.service(fez)
    job_cls = service.job

    def job_with_properties(job_id):
        job = job_cls(job_id)
        job.properties = fez.properties
        return job

    service.job = job_with_properties
    record = store.recover(path, service)
    _, prepared, _ = store.load(path)
    features = prepared["payload"]["features"]
    assert {"logical", "isa", "hardware_at_submission", "baselines"} <= set(features)  # A B C E
    assert features["logical"]["quantum_specific"]["available"]
    entry = record["submission_journal"][0]
    assert entry["hardware_at_execution"]["available"]  # D
    assert entry["baselines_at_execution"]["available"]
    row = record["individual_results"][0]
    assert row["targets"]["fidelity"]["success_rate"] == 1.0  # F
    assert row["targets"]["histogram_assessment"]["available"]
    for key in ("run_id", "campaign_id", "window", "backend_name"):  # G
        assert record[key] is not None
    assert prepared["versions"]["qiskit"] != "unknown"
    assert prepared["seed_transpiler"] == 11
    json.dumps(record)  # Serializable as stored.


def test_biased_coin_reference_override_is_respected():
    experiment = CoinTossIBMExperiment()
    config = experiment.get_config("CT3")
    reference = experiment.get_reference(config)
    assert len(reference) == 8 and sum(reference.values()) == pytest.approx(1)
    biased = experiment.get_reference(experiment.get_config("CT3-B25"))
    assert biased["000"] == pytest.approx(0.75**3, abs=1e-3)  # Exact, not the success set.


def test_ideal_reference_differs_from_success_set_where_it_matters():
    from qward.examples.papers.grover.grover_ibm import GroverIBMExperiment
    from qward.examples.papers.phase_estimation.pe_ibm import PEIBMExperiment
    from qward.examples.papers.shor.shor_ibm import ShorIBMExperiment

    grover = GroverIBMExperiment()
    ideal = grover.get_reference(grover.get_config("S3-1"))
    assert ideal["011"] == pytest.approx(0.9453, abs=1e-3) and len(ideal) == 8
    pe = PEIBMExperiment()
    config = pe.get_config("PE-T3")
    assert pe.get_reference(config) == pytest.approx({"001": 1.0})
    winners = F.success_labels(pe.create_circuit(config), pe.create_success_criteria(config))
    assert len(winners) == 3  # Tolerant success set; ideal is a single peak.
    shor = ShorIBMExperiment()
    assert len(shor.get_reference(shor.get_config("SHOR-N15-M3"))) == 4


def test_baselines_use_success_set_and_ideal_mass(fez):
    from qward.examples.papers.grover.grover_ibm import GroverIBMExperiment

    experiment = GroverIBMExperiment()
    config = experiment.get_config("S3-1")
    circuit = experiment.create_circuit(config)
    isa = _isa(circuit, fez)
    snapshot = capture_calibration(isa, fez, include_backend_properties=False)
    reference = experiment.get_reference(config)
    winners = F.success_labels(circuit, experiment.create_success_criteria(config))
    group = F.baseline_predictions(isa, snapshot, reference, success_set=winners)
    assert group["ideal_success_mass"] == pytest.approx(reference["011"])
    assert group["m1_error"] == pytest.approx(1 - group["esp_score"] * reference["011"])
    assert 0 < group["m2_tvd"] < 1


def test_quantum_specific_is_reproducible_and_restores_rng():
    import torch

    from qward.examples.papers.grover.grover_ibm import GroverIBMExperiment

    experiment = GroverIBMExperiment()
    circuit = experiment.create_circuit(experiment.get_config("S3-1"))
    first, second = F.quantum_specific(circuit), F.quantum_specific(circuit)
    for key in ("magic", "coherence", "sensitivity", "entanglement_ratio"):
        assert first[key] == second[key]
    torch.manual_seed(5)
    expected = torch.rand(1).item()
    torch.manual_seed(5)
    F.quantum_specific(circuit)
    assert torch.rand(1).item() == expected  # Caller's RNG state untouched.


def test_quantum_specific_cache_roundtrip_and_key(tmp_path):
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    first = F.quantum_specific(circuit, cache_dir=tmp_path)
    assert first["available"] and first["cached"] is False
    assert len(list(tmp_path.glob("*.json"))) == 1
    again = F.quantum_specific(circuit, cache_dir=tmp_path)
    assert again["cached"] is True
    assert {k: v for k, v in again.items() if k != "cached"} == {
        k: v for k, v in first.items() if k != "cached"
    }
    other = QuantumCircuit(2)
    other.h(0)
    other.cx(1, 0)
    F.quantum_specific(other, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 2  # Different circuit, different entry.
    assert F.quantum_specific(circuit, cache_dir=None)["cached"] is False
    assert len(list(tmp_path.glob("*.json"))) == 2  # cache_dir=None writes nothing.


def test_unavailable_results_are_never_cached(tmp_path):
    big = QuantumCircuit(F.QUANTUM_SPECIFIC_MAX_QUBITS + 1)
    assert F.quantum_specific(big, cache_dir=tmp_path)["available"] is False
    assert list(tmp_path.glob("*.json")) == []
