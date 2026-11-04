"""Offline tests of analytic circuits, inference, and durable experiment tracking.

No experiment, shot simulator or network service is executed. Runtime jobs are
test doubles, and Clifford equality checks use polynomial-size tableaux.
"""

from dataclasses import replace
from pathlib import Path
import json
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Clifford
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.options import SamplerOptions

from qward import Scanner
from qward.algorithms import BernsteinVazirani
from qward.examples.papers.bv import bv_calibration_ibm as campaign
from qward.examples.papers.bv.bv_calibration_configs import (
    CONFIGS,
    analytic_references,
    build_circuit,
)
from qward.examples.papers.bv.bv_calibration_analysis import (
    compare_histogram_error,
    export,
)
from qward.metrics import CircuitErrorPrediction, HistogramAssessment
from qward.metrics.circuit_error_prediction import capture_calibration, record_hash
from qward.metrics.histogram_assessment import binomial_interval, minimum_binomial_shots


@pytest.fixture
def config():
    return next(
        c
        for c in CONFIGS.values()
        if c.num_qubits == 2
        and c.pattern == "ALT"
        and c.family == "bv_padded"
        and c.forward_layers == 0
    )


@pytest.fixture
def backend():
    return GenericBackendV2(3, basis_gates=["x", "sx", "rz", "cx"], seed=5)


@pytest.fixture
def prepared(tmp_path, config, backend, monkeypatch):
    monkeypatch.setattr(campaign, "calculate_qward_metrics", lambda circuit: {"fixture": True})
    experiment = campaign.BVCalibrationIBMExperiment(output_dir=tmp_path)
    path = experiment.prepare(config.config_id, backend, initial_layout=[0, 1, 2])
    return experiment, path


def test_complete_matrix_and_analytic_references():
    assert len(CONFIGS) == 100
    assert sum(c.family == "bv_padded" for c in CONFIGS.values()) == 48
    assert (
        sum(c.num_qubits == 2 and c.layout_label == "low_predicted_error" for c in CONFIGS.values())
        == 25
    )
    for config in CONFIGS.values():
        circuit = build_circuit(config)
        assert circuit.num_qubits == config.num_total_qubits
        assert circuit.num_clbits == config.num_qubits
        nominal, injected = analytic_references(config)
        assert sum(nominal.values()) == pytest.approx(1)
        assert sum(injected.values()) == pytest.approx(1)
        assert all(len(k) == config.num_qubits for k in nominal.keys() | injected.keys())


@pytest.mark.parametrize("layers", [0, 2, 8, 32])
@pytest.mark.parametrize("pattern", ["ONES", "ALT", "SINGLE"])
def test_bv_padding_preserves_library_clifford(config, layers, pattern):
    config = replace(config, forward_layers=layers, pattern=pattern)
    baseline = BernsteinVazirani(config.bv_config.secret_string).circuit
    adapted = build_circuit(config)
    assert Clifford(adapted.remove_final_measurements(inplace=False)) == Clifford(
        baseline.remove_final_measurements(inplace=False)
    )
    if layers == 0:
        assert adapted == baseline
    else:
        assert adapted.count_ops()["cx"] > baseline.count_ops().get("cx", 0)
    assert analytic_references(config)[0] == {config.bv_config.secret_string[::-1]: 1.0}


def test_bv_fault_distinguishes_correct_circuit_from_wrong_task(config):
    fault = replace(config, family="bv_target_deviation", angle=math.pi)
    nominal, injected = analytic_references(fault)
    assert set(nominal).isdisjoint(injected)
    operations = [item.operation.name for item in build_circuit(fault).data]
    assert operations.index("ry") > max(i for i, op in enumerate(operations) if op == "h")
    assert operations.index("ry") < operations.index("measure")


def test_prediction_uses_native_calibration_and_snapshot_integrity(config, backend):
    isa = generate_preset_pass_manager(backend=backend, optimization_level=0).run(
        build_circuit(config)
    )
    snapshot = capture_calibration(isa, backend)
    _, reference = analytic_references(config)
    prediction = CircuitErrorPrediction(isa, snapshot=snapshot, reference=reference).get_metrics()
    expected = math.prod(1 - e["error"] for e in snapshot["instructions"] if e["name"] != "delay")
    assert prediction.esp_score == pytest.approx(expected)
    assert prediction.mixture_tvd == pytest.approx((1 - expected) * 0.75)
    assert sum(prediction.mixture_probabilities.values()) == pytest.approx(1)
    changed_circuit = isa.copy()
    changed_circuit.x(0)
    with pytest.raises(ValueError, match="different ISA circuit"):
        HistogramAssessment(
            changed_circuit, counts={"01": 10}, reference=reference, prediction=prediction
        )
    snapshot["instructions"][0]["error"] = None
    snapshot["snapshot_hash"] = record_hash(
        {k: v for k, v in snapshot.items() if k != "snapshot_hash"}
    )
    missing = CircuitErrorPrediction(isa, snapshot=snapshot, reference=reference).get_metrics()
    assert missing.esp_score is None
    assert missing.calibration_coverage < 1
    snapshot["instructions"][0]["error"] = 0
    with pytest.raises(ValueError, match="changed"):
        CircuitErrorPrediction(isa, snapshot=snapshot, reference=reference).get_metrics()


def test_scanner_and_boundary_intervals(config, backend):
    isa = generate_preset_pass_manager(backend=backend, optimization_level=0).run(
        build_circuit(config)
    )
    snapshot = capture_calibration(isa, backend)
    reference = analytic_references(config)[0]
    target = next(iter(reference))
    prediction = CircuitErrorPrediction(isa, snapshot=snapshot, reference=reference)
    assessment = HistogramAssessment(isa, counts={target: 2048}, reference=reference)
    results = Scanner(isa, strategies=[prediction, assessment]).calculate_metrics()
    assert set(results) == {"CircuitErrorPrediction", "HistogramAssessment"}
    assert binomial_interval(0, 1, 0.05) == pytest.approx((0, 0.975))
    assert binomial_interval(1, 1, 0.05) == pytest.approx((0.025, 1))
    assert assessment.get_metrics().success_rate == 1
    assert assessment.get_metrics().tvd_interval[1] > 0
    assert "task_status" not in assessment.get_metrics().model_dump()
    uncertain = HistogramAssessment(isa, counts={target: 1}, reference=reference).get_metrics()
    assert uncertain.success_interval == pytest.approx((0.025, 1))


def test_wrong_peak_and_uniform_are_not_misclassified(config):
    circuit = build_circuit(config)
    reference = analytic_references(config)[0]
    wrong = HistogramAssessment(circuit, counts={"11": 2048}, reference=reference).get_metrics()
    assert wrong.rank_status == "verified"
    assert wrong.success_rate == 0
    assert wrong.tvd == 1
    uniform = {f"{x:02b}": 0.25 for x in range(4)}
    good_uniform = HistogramAssessment(
        circuit, counts={k: 25000 for k in uniform}, reference=uniform
    ).get_metrics()
    assert good_uniform.tvd == 0
    assert good_uniform.tvd_interval[1] > 0
    assert good_uniform.rank_status == "unresolved"
    tied = HistogramAssessment(
        circuit, counts={"00": 50, "01": 50}, reference=reference
    ).get_metrics()
    assert tied.gap_standard_error == pytest.approx(0.1)


@pytest.mark.parametrize(
    "counts", [{}, {"00": 0}, {"noise_0": 5}, {"00": -1}, {"00": 0.5}, {"00": True}]
)
def test_assessment_rejects_invalid_counts(counts):
    with pytest.raises(ValueError):
        HistogramAssessment(QuantumCircuit(2, 2), counts=counts, reference={"00": 1})


def test_exact_shot_planning():
    assert minimum_binomial_shots(0, 1, power=0.9) == 1
    assert minimum_binomial_shots(0.8, 0.7) is None
    assert minimum_binomial_shots(0.8, 0.9, maximum=1) is None
    count = minimum_binomial_shots(0.5, 0.8, alpha=0.05, power=0.9, maximum=100)
    assert count is not None and 1 < count < 100


def test_preflight_has_no_service_side_effect(config, monkeypatch):
    import qiskit_ibm_runtime

    service = Mock(side_effect=AssertionError("Network forbidden"))
    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", service)
    experiment = campaign.BVCalibrationIBMExperiment()
    report = experiment.preflight(config.config_id)
    assert report["jobs_submitted"] == 0
    service.assert_not_called()
    with pytest.raises(SystemExit):
        experiment.create_argument_parser().parse_args(["--config", config.config_id])


def test_preparation_and_tamper_detection(prepared):
    experiment, path = prepared
    journal, frozen, _ = experiment._load(path)
    assert journal["status"] == "prepared"
    assert journal["submission_journal"] == []
    assert frozen["calibration_snapshot"]["scheduled"]
    assert frozen["prediction"]["features"]["idle_over_t1"] is not None
    artifacts = path.parent / journal["artifact_directory"]
    (artifacts / "isa.qpy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        experiment._load(path)


def test_mock_submission_recovery_and_tracking_export(prepared, backend, monkeypatch, tmp_path):
    import qiskit_ibm_runtime

    experiment, path = prepared
    original = experiment._load(path)[1]
    submitted = []

    class FakeSampler:
        def __init__(self, *, mode, options):
            assert mode is backend  # Job mode: no Batch or Session.
            self.options = SamplerOptions(**options)

        def run(self, pubs, *, shots):
            submitted.append((pubs, shots))
            return SimpleNamespace(job_id=lambda: "mock-job")

    monkeypatch.setattr(
        qiskit_ibm_runtime, "Batch", Mock(side_effect=AssertionError("Batch not used"))
    )
    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", FakeSampler)
    experiment.submit_prepared(path, backend)
    assert len(submitted) == 1
    assert submitted[0][1] == original["shots"]
    from qward.metrics.circuit_error_prediction import circuit_hash

    assert circuit_hash(submitted[0][0][0]) == original["isa_circuit_hash"]
    with pytest.raises(ValueError, match="fresh prepared"):
        experiment.submit_prepared(path, backend)
    service = Mock()
    service.job.return_value.result.side_effect = TimeoutError("fixture timeout")
    failed = experiment.collect(path, service)
    assert failed["status"] == "incomplete"
    assert failed["submission_journal"][0]["job_id"] == "mock-job"
    target = next(iter(original["nominal_reference"]))
    service.job.return_value.result.side_effect = None
    service.job.return_value.result.return_value = [
        SimpleNamespace(
            data=SimpleNamespace(c=SimpleNamespace(get_counts=lambda: {target: 2048})),
            metadata={},
        )
    ]
    service.job.return_value.metrics.return_value = {"timestamps": {"finished": "fixture"}}
    completed = experiment.collect(path, service)
    assert completed["status"] == "completed"
    assert completed["prediction"] == original["prediction"]
    assert completed["individual_results"][0]["counts"] == {target: 2048}
    assert completed["individual_results"][0]["histogram_assessment"]["success_rate"] == 1
    assert "quantum_advantage_demonstrated" not in completed["batch_summary"]
    report = export(path.parent, tmp_path / "analysis")
    assert report["executions"] == report["primary_executions"] == 1
    assert report["shots"] == 2048
    assert len(submitted) == 1  # Recovery never submits another job.
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["prepared_hash"] == record_hash(original)


def test_schedule_is_reproducible_and_matches_budgets():
    experiment = campaign.BVCalibrationIBMExperiment()
    smoke = experiment.campaign_plan(smoke_test=True)
    full = experiment.campaign_plan()
    assert smoke["executions"] == 25 and smoke["shots"] == 51200
    assert full["executions"] == 300 and full["shots"] == 614400
    assert full == experiment.campaign_plan()
    assert (
        full["windows"][0]["config_ids_in_execution_order"]
        != full["windows"][1]["config_ids_in_execution_order"]
    )


def test_padded_treatment_survives_compilation(config, backend, tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "calculate_qward_metrics", lambda circuit: {})
    padded = next(
        c
        for c in CONFIGS.values()
        if c.num_qubits == 2
        and c.pattern == config.pattern
        and c.layout_label == config.layout_label
        and c.forward_layers == 2
    )
    experiment = campaign.BVCalibrationIBMExperiment(output_dir=tmp_path)
    path = experiment.prepare(padded.config_id, backend, initial_layout=[0, 1, 2])
    assert experiment._load(path)[1]["prediction"]["features"][
        "native_two_qubit_gates"
    ] > config.bv_config.secret_string.count("1")


def test_injected_perfect_wrong_answer_can_match_hardware_prediction(config, backend):
    fault = replace(config, family="bv_target_deviation", angle=math.pi)
    circuit = build_circuit(fault)
    isa = generate_preset_pass_manager(backend=backend, optimization_level=0).run(circuit)
    snapshot = capture_calibration(isa, backend)
    for entry in snapshot["instructions"]:
        if entry["name"] != "delay":
            entry["error"] = 0.0
    snapshot["snapshot_hash"] = record_hash(
        {k: v for k, v in snapshot.items() if k != "snapshot_hash"}
    )
    nominal, injected = analytic_references(fault)
    prediction = CircuitErrorPrediction(isa, snapshot=snapshot, reference=injected).get_metrics()
    counts = {next(iter(injected)): 2048}
    assessment = HistogramAssessment(
        isa, counts=counts, reference=nominal, prediction=prediction
    ).get_metrics()
    assert assessment.success_rate == 0
    assert assessment.tvd == 1
    assert assessment.model_status == "consistent"
    assert assessment.prediction_residual == 0


def test_ghz_phase_is_invisible_in_z_and_visible_in_x():
    ghz = next(c for c in CONFIGS.values() if c.family == "ghz_phase" and c.num_qubits == 4)
    plus_z = analytic_references(replace(ghz, angle=0, basis="Z"))[1]
    minus_z = analytic_references(replace(ghz, angle=math.pi, basis="Z"))[1]
    plus_x = analytic_references(replace(ghz, angle=0, basis="X"))[1]
    minus_x = analytic_references(replace(ghz, angle=math.pi, basis="X"))[1]
    assert plus_z == minus_z
    assert set(plus_x).isdisjoint(minus_x)


def test_unused_backend_qubits_do_not_contribute_idle_exposure(config):
    backend = GenericBackendV2(5, basis_gates=["x", "sx", "rz", "cx"], seed=5)
    isa = generate_preset_pass_manager(
        backend=backend,
        optimization_level=0,
        initial_layout=[0, 1, 2],
        scheduling_method="alap",
    ).run(build_circuit(config))
    snapshot = capture_calibration(isa, backend)
    assert set(snapshot["qubits"]) == {"0", "1", "2"}
    assert any(e["name"] == "delay" and e["qubits"] == [3] for e in snapshot["instructions"])
    predicted = CircuitErrorPrediction(
        isa, snapshot=snapshot, reference=analytic_references(config)[1]
    ).get_metrics()
    expected = sum(q["idle_s"] / q["t1_s"] for q in snapshot["qubits"].values())
    assert predicted.features["idle_over_t1"] == pytest.approx(expected)


def test_runtime_execution_metadata_is_preserved(tmp_path):
    from qiskit_ibm_runtime.execution_span import ExecutionSpans

    path = tmp_path / "metadata.json"
    campaign._write(path, {"execution": {"execution_spans": ExecutionSpans([])}})
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["execution"]["execution_spans"]["__type__"] == "ExecutionSpans"


def test_expected_error_is_compared_to_same_reference_as_observed_histogram():
    assessment = {"tvd": 0.2, "tvd_interval": [0.175, 0.226], "alpha": 0.05}
    prediction = {
        "m1_success": 0.9,
        "mixture_tvd": 0.075,
        "reference_probabilities": {"01": 1.0},
        "mixture_probabilities": {"00": 0.025, "01": 0.925, "10": 0.025, "11": 0.025},
    }
    comparison = compare_histogram_error(
        assessment, prediction, {"01": 800, "11": 200}, primary=True
    )
    assert comparison["predicted_histogram_error_m1_success"] == pytest.approx(0.1)
    assert comparison["observed_histogram_error"] == 0.2
    assert comparison["histogram_error_residual_m1_success"] == pytest.approx(0.1)
    assert comparison["histogram_error_residual_m1_mixture"] == pytest.approx(0.125)
    assert comparison["mixture_prediction_histogram_tvd"] == pytest.approx(0.175)
    assert comparison["target_count"] == 800
    assert comparison["m1_success_error_in_observed_interval"] is False
    assert comparison["m1_success_budget_exceeds_upper_budget"] is True
    assert comparison["m1_success_budget_residual_shot_z"] > 10
    assert comparison["target_signal_resolved_under_uniform_null"] is True
    assert comparison["uncertainty_review_action"] == "investigate_prediction"
    # A deliberately perturbed circuit and its nominal task have different
    # references; their errors must not be compared as if they were the same.
    control = compare_histogram_error(assessment, prediction, {"01": 800, "11": 200}, primary=False)
    assert "predicted_histogram_error_m1_success" not in control
    assert "uncertainty_review_action" not in control


def test_error_budget_compares_only_identical_deterministic_references(config, backend):
    isa = generate_preset_pass_manager(backend=backend, optimization_level=0).run(
        build_circuit(config)
    )
    reference = analytic_references(config)[0]
    target = next(iter(reference))
    prediction = CircuitErrorPrediction(
        isa, snapshot=capture_calibration(isa, backend), reference=reference
    ).get_metrics()
    same = HistogramAssessment(
        isa, counts={target: 100}, reference=reference, prediction=prediction
    ).get_metrics()
    assert same.error_budget is not None
    assert same.error_budget.count == 0
    assert same.error_budget.predicted_probability == pytest.approx(1 - prediction.m1_success)
    assert same.to_flat_dict()["error_budget_scope"] == "conditional_on_fixed_probability"
    wrong_reference = {"00" if target != "00" else "11": 1.0}
    different = HistogramAssessment(
        isa, counts={target: 100}, reference=wrong_reference, prediction=prediction
    ).get_metrics()
    assert different.error_budget is None


def _snapshot(instructions):
    return {"instructions": instructions}


def test_wide_sparse_m2_matches_dense_asymmetric_readout():
    from qward.metrics.circuit_error_prediction import depolarizing_prediction, flip_distribution

    width = 13
    labels = [f"{i:013b}" for i in range(2**width)]
    reference = {labels[1]: 0.3, labels[-2]: 0.7}
    # Different rates on different bits also exercise the clbit order.
    flips = {i: (0.01 + i / 100, 0.45 - i / 50) for i in range(width)}
    gates = [{"name": "sx", "qubits": [0], "clbits": [], "error": 0.1}]
    measures = [{"name": "measure", "qubits": [i], "clbits": [i], "error": 0.1} for i in flips]
    snapshot = {
        "instructions": gates + measures,
        "qubits": {
            str(i): {"prob_meas1_prep0": up, "prob_meas0_prep1": down}
            for i, (up, down) in flips.items()
        },
    }
    m2 = depolarizing_prediction(snapshot, reference, width)
    dense = flip_distribution(
        {s: 0.8 * reference.get(s, 0) + 0.2 / len(labels) for s in labels}, flips, width
    )
    assert m2["probabilities"] is None
    assert m2["reference_mass"] == pytest.approx(sum(dense[k] for k in reference))
    assert m2["tvd"] == pytest.approx(sum(abs(dense[k] - reference.get(k, 0)) for k in labels) / 2)


def test_m2_converts_infidelity_to_depolarization_and_caps_at_uniform():
    from qward.metrics.circuit_error_prediction import depolarizing_prediction

    gates = [
        {"name": "sx", "qubits": [0], "clbits": [], "error": 0.01},
        {"name": "cx", "qubits": [0, 1], "clbits": [], "error": 0.03},
    ]
    readout = [
        {"name": "measure", "qubits": [0], "clbits": [0], "error": 0.02},
        {"name": "measure", "qubits": [1], "clbits": [1], "error": 0.05},
    ]
    m2 = depolarizing_prediction(_snapshot(gates + readout), {"01": 1.0}, 2)
    polarization = (1 - 0.01 * 2) * (1 - 0.03 * 4 / 3)
    assert m2["polarization"] == pytest.approx(polarization)
    assert m2["success"] == pytest.approx(polarization * 0.98 * 0.95 + (1 - polarization) / 4)
    assert sum(m2["probabilities"].values()) == pytest.approx(1)
    # Clbit 0 is the rightmost label bit: flipping it alone turns 01 into 00.
    assert m2["probabilities"]["00"] == pytest.approx(
        polarization * 0.02 * 0.95 + (1 - polarization) / 4
    )
    assert m2["tvd"] == pytest.approx(1 - m2["success"])
    # A fully scrambling circuit predicts at most the uniform error 1 - 1/D.
    scrambled = [{"name": "cx", "qubits": [0, 1], "clbits": [], "error": 0.75}] * 3
    worst = depolarizing_prediction(_snapshot(scrambled + readout), {"01": 1.0}, 2)
    assert worst["polarization"] == 0
    assert 1 - worst["success"] == pytest.approx(0.75)


def test_m2_requires_complete_calibration_and_measurement_coverage():
    from qward.metrics.circuit_error_prediction import depolarizing_prediction

    gates = [{"name": "sx", "qubits": [0], "clbits": [], "error": None}]
    readout = [{"name": "measure", "qubits": [0], "clbits": [0], "error": 0.02}]
    assert depolarizing_prediction(_snapshot(gates + readout), {"0": 1.0}, 1)["success"] is None
    assert depolarizing_prediction(_snapshot(readout), {"00": 1.0}, 2)["success"] is None


def test_m2_is_frozen_with_prediction_and_budgeted_by_assessment(config, backend):
    isa = generate_preset_pass_manager(backend=backend, optimization_level=0).run(
        build_circuit(config)
    )
    reference = analytic_references(config)[0]
    target = next(iter(reference))
    prediction = CircuitErrorPrediction(
        isa, snapshot=capture_calibration(isa, backend), reference=reference
    ).get_metrics()
    assert prediction.m2_success is not None
    # At low noise d/(d-1) makes M2 slightly more pessimistic than M1, but it
    # can never predict more error than a fully scrambled output.
    assert 1 - prediction.m1_success < 1 - prediction.m2_success <= 1 - 2.0**-isa.num_clbits
    shots = 1000
    expected = round(prediction.m2_success * shots)
    other = target[:-1] + ("1" if target[-1] == "0" else "0")
    assert other in (prediction.m2_probabilities or {})
    counts = {target: expected, other: shots - expected}
    assessment = HistogramAssessment(
        isa, counts=counts, reference=reference, prediction=prediction
    ).get_metrics()
    assert assessment.m2_error_budget.predicted_probability == pytest.approx(
        1 - prediction.m2_success
    )
    assert assessment.m2_error_budget.range_status == "within_range"
    assert "m2_error_budget_range_status" in assessment.to_flat_dict()
    assert "m2_probabilities" not in prediction.to_flat_dict()


@pytest.mark.parametrize(
    "resolved,status,expected",
    [
        (True, "within_range", "success"),
        (True, "above_range", "degraded_success"),
        (False, "within_range", "expected_failure"),
        (False, "above_range", "anomalous_failure"),
        (True, None, "prediction_unavailable"),
    ],
)
def test_outcome_class_combines_signal_and_prediction(resolved, status, expected):
    from qward.examples.papers.bv.bv_calibration_analysis import classify_outcome

    assert classify_outcome(resolved, status) == expected


def test_old_records_get_m2_from_archived_snapshot():
    from qward.examples.papers.bv.bv_calibration_analysis import with_m2

    snapshot = _snapshot(
        [
            {"name": "cx", "qubits": [0, 1], "clbits": [], "error": 0.03},
            {"name": "measure", "qubits": [0], "clbits": [0], "error": 0.0},
        ]
    )
    old = {"reference_probabilities": {"1": 1.0}, "output_width": 1}
    upgraded = with_m2(old, snapshot)
    assert upgraded["m2_source"] == "recomputed_from_archived_snapshot"
    assert upgraded["m2_success"] == pytest.approx(0.96 + 0.04 / 2)
    assert with_m2({**old, "m2_success": 0.5}, None)["m2_source"] == "frozen_prediction"


def test_ibm_calibration_flow_with_real_fake_backend(tmp_path, monkeypatch):
    """Snapshot of a real IBM device: directional readout, drift re-prediction."""
    from qiskit_ibm_runtime.fake_provider import FakeFez
    from qward.metrics.circuit_error_prediction import snapshot_at_properties

    monkeypatch.setattr(campaign, "calculate_qward_metrics", lambda circuit: {})
    fez = FakeFez()
    experiment = campaign.BVCalibrationIBMExperiment(output_dir=tmp_path)
    config_id = "BVCP4-low_predicted_error-ALT-L008"
    ranked = experiment.rank_layouts(config_id, fez, top=2)
    best = ranked["low_predicted_error"][0]
    assert best["m1_error"] < ranked["contrasting_predicted_error"][0]["m1_error"]
    path = experiment.prepare(config_id, fez, initial_layout=best["layout"])
    _, prepared, circuits = experiment._load(path)
    snapshot = prepared["calibration_snapshot"]
    assert snapshot["calibrated_at"] is not None
    assert all(q["prob_meas1_prep0"] is not None for q in snapshot["qubits"].values())
    assert prepared["prediction"]["calibration_coverage"] == 1
    # Re-reading the same calibration reproduces the frozen prediction exactly.
    replay = snapshot_at_properties(snapshot, fez.properties())
    job = SimpleNamespace(properties=fez.properties)
    diagnostic = experiment._execution_time_prediction(job, prepared, circuits)
    assert diagnostic["execution_calibration_changed"] is False
    frozen = prepared["prediction"]
    assert diagnostic["execution_time_prediction"]["m1_success"] == pytest.approx(
        frozen["m1_success"]
    )
    assert diagnostic["execution_time_prediction"]["m2_success"] == pytest.approx(
        frozen["m2_success"]
    )
    assert replay["source"] == "execution_time_properties"


def test_recalibration_before_submission_is_flagged_not_refused(tmp_path, monkeypatch, capsys):
    from qiskit_ibm_runtime.fake_provider import FakeFez

    monkeypatch.setattr(campaign, "calculate_qward_metrics", lambda circuit: {})
    fez = FakeFez()
    experiment = campaign.BVCalibrationIBMExperiment(output_dir=tmp_path)
    path = experiment.prepare(
        "BVCP2-low_predicted_error-ALT-L000", fez, initial_layout=[86, 87, 88]
    )
    prepared = experiment._load(path)[1]
    same = experiment._calibration_at_submission(prepared, fez)
    assert same["recalibrated_before_submission"] is False
    recalibrated = SimpleNamespace(
        properties=lambda refresh=False: SimpleNamespace(last_update_date="2099-01-01")
    )
    changed = experiment._calibration_at_submission(prepared, recalibrated)
    assert changed == {
        "calibrated_at_submission": "2099-01-01",
        "recalibrated_before_submission": True,
    }
    assert "recalibrated" in capsys.readouterr().err


def test_record_summary_is_compact():
    journal = {
        "config_id": "X",
        "status": "submitted",
        "prediction": {"m1_success": 0.8, "mixture_tvd": 0.15, "m2_success": 0.75},
        "submission_journal": [
            {"run_index": 0, "job_id": "j", "status": "submitted"},
        ],
        "individual_results": [],
    }
    summary = campaign.summarize_record(journal)
    assert summary["frozen_predicted_error_m1"] == pytest.approx(0.2)
    assert summary["jobs"][0]["job_id"] == "j"
    assert summary["jobs"][0]["observed_error"] is None
    assert summary["next_step"].startswith("--recover")


def test_batch_prepare_submit_recover_with_mocked_runtime(tmp_path, monkeypatch):
    import qiskit_ibm_runtime
    from qiskit_ibm_runtime.fake_provider import FakeFez

    monkeypatch.setattr(campaign, "calculate_qward_metrics", lambda circuit: {})
    fez = FakeFez()
    fez.refresh = lambda: None
    service = SimpleNamespace(backend=lambda name: fez)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "backend": "fake_fez",
                "campaign_id": "batch-test",
                "planned_executions": 4,
                "items": [
                    {"config": "BVCP2-low_predicted_error-ALT-L000", "layout": [86, 87, 88]},
                    {
                        "config": "BVCP2-low_predicted_error-ALT-L000",
                        "layout": [86, 87, 88],
                        "runs": 2,
                        "twirl_gates": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    experiment = campaign.BVCalibrationIBMExperiment(output_dir=tmp_path / "raw")
    manifest = experiment.prepare_batch(plan, service)
    assert manifest["jobs_planned"] == 3
    assert [i["arm"] for i in manifest["items"]] == ["baseline", "gate_twirling"]
    batches, sent = [], []

    class FakeBatch:
        session_id = "batch-1"

        def __init__(self, *, backend):
            batches.append(backend)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class FakeSampler:
        def __init__(self, *, mode, options):
            assert isinstance(mode, FakeBatch)
            self.options = SamplerOptions(**options)

        def run(self, pubs, *, shots):
            sent.append(self.options.twirling.enable_gates)
            return SimpleNamespace(job_id=lambda n=len(sent): f"job-{n}")

    monkeypatch.setattr(qiskit_ibm_runtime, "Batch", FakeBatch)
    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", FakeSampler)
    path = Path(manifest["manifest_path"])
    # Recovering before submission must refuse and leave everything submittable.
    with pytest.raises(ValueError, match="never submitted"):
        experiment.recover_manifest(path, service)
    with pytest.raises(ValueError, match="never submitted"):
        experiment.collect(Path(manifest["items"][0]["record_path"]), service)
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "prepared"
    submitted = experiment.submit_batch(path, fez)
    assert len(batches) == 1 and submitted["batch_id"] == "batch-1"
    assert sent == [False, True, True]
    assert [i["job_ids"] for i in submitted["items"]] == [["job-1"], ["job-2", "job-3"]]
    with pytest.raises(ValueError, match="fresh prepared batch"):
        experiment.submit_batch(path, fez)
    over = tmp_path / "over.json"
    over.write_text(
        plan.read_text(encoding="utf-8").replace('"runs": 2', '"runs": 4'), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="budget"):
        experiment.submit_batch(Path(experiment.prepare_batch(over, service)["manifest_path"]), fez)


def test_manifest_recovery_does_not_shadow_base_recover_batch():
    import inspect

    from qward.examples.papers.ibm_experiment_base import IBMExperimentBase

    own = inspect.signature(campaign.BVCalibrationIBMExperiment.recover_batch)
    assert own == inspect.signature(IBMExperimentBase.recover_batch)
    assert hasattr(campaign.BVCalibrationIBMExperiment, "recover_manifest")


def test_prepare_alpha_matches_offline_analysis_claims(prepared):
    from qward.examples.papers.bv.bv_calibration_analysis import CLAIMS_PER_RUN

    experiment, path = prepared
    frozen = experiment._load(path)[1]
    assert frozen["claims_per_run"] == CLAIMS_PER_RUN
    assert frozen["assessment_alpha"] == pytest.approx(
        0.05 / (CLAIMS_PER_RUN * frozen["planned_executions_per_window"])
    )


def _plan(tmp_path, name, **overrides):
    plan = {
        "backend": "fake_fez",
        "campaign_id": "plan-test",
        "planned_executions": 4,
        "items": [{"config": "BVCP2-low_predicted_error-ALT-L000", "layout": [86, 87, 88]}],
        **overrides,
    }
    path = tmp_path / name
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_prepare_batch_validates_before_writing_records(tmp_path, monkeypatch):
    from qiskit_ibm_runtime.fake_provider import FakeFez

    monkeypatch.setattr(campaign, "calculate_qward_metrics", lambda circuit: {})
    fez = FakeFez()
    fez.refresh = lambda: None
    service = SimpleNamespace(backend=lambda name: fez)
    raw = tmp_path / "raw"
    experiment = campaign.BVCalibrationIBMExperiment(output_dir=raw)
    existing = _plan(tmp_path, "existing.json")
    (tmp_path / "existing_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        experiment.prepare_batch(existing, service)
    bad_runs = _plan(
        tmp_path,
        "bad.json",
        items=[{"config": "BVCP2-low_predicted_error-ALT-L000", "layout": [86, 87, 88], "runs": 0}],
    )
    with pytest.raises(ValueError, match="runs"):
        experiment.prepare_batch(bad_runs, service)
    over = _plan(
        tmp_path,
        "over.json",
        items=[{"config": "BVCP2-low_predicted_error-ALT-L000", "layout": [86, 87, 88], "runs": 5}],
    )
    with pytest.raises(ValueError, match="budget"):
        experiment.prepare_batch(over, service)
    unknown = _plan(tmp_path, "unknown.json", items=[{"config": "NOPE", "layout": [1, 2, 3]}])
    with pytest.raises(ValueError, match="Unknown"):
        experiment.prepare_batch(unknown, service)
    assert not raw.exists() or not list(raw.glob("*.json"))


def test_batch_budget_uses_each_campaign_window_limit(tmp_path, monkeypatch):
    from qiskit_ibm_runtime.fake_provider import FakeFez

    monkeypatch.setattr(campaign, "calculate_qward_metrics", lambda circuit: {})
    fez = FakeFez()
    fez.refresh = lambda: None
    service = SimpleNamespace(backend=lambda name: fez)
    experiment = campaign.BVCalibrationIBMExperiment(output_dir=tmp_path / "raw")
    big = experiment.prepare_batch(
        _plan(tmp_path, "big.json", campaign_id="big", planned_executions=10), service
    )
    small = experiment.prepare_batch(
        _plan(tmp_path, "small.json", campaign_id="small", planned_executions=1), service
    )
    manifest = json.loads(Path(big["manifest_path"]).read_text(encoding="utf-8"))
    manifest["items"] += json.loads(Path(small["manifest_path"]).read_text(encoding="utf-8"))[
        "items"
    ]
    manifest["items"][1]["runs"] = 2  # Exceeds "small" only; the first item's limit is 10.
    merged = tmp_path / "merged_manifest.json"
    merged.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Repetitions exceed|budget"):
        experiment.submit_batch(merged, fez)
